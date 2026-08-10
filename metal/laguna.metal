// Laguna-specific primitives. The C graph owns model semantics and scheduling;
// these kernels only cover operations that are not represented by the shared
// DeepSeek/GLM Metal API.

/* A single fixed-size reduction is enough for Laguna's one-row greedy decode.
 * Each lane scans a strided portion of the row, then the threadgroup reduces
 * the lane winners. The sentinel and strict comparison intentionally mirror
 * sample_argmax: NaNs never update, equal values retain the lower index, and
 * values at or below -1e30f leave index zero selected. */
kernel void kernel_laguna_argmax_f32(
        device const float *logits [[buffer(0)]],
        device int32_t     *out_idx [[buffer(1)]],
        constant uint      &n_vocab [[buffer(2)]],
        threadgroup float *best_values [[threadgroup(0)]],
        threadgroup uint  *best_indices [[threadgroup(1)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]]) {
    const uint nth = (uint)ntg_u.x;
    float best_value = -1.0e30f;
    uint best_index = 0u;
    for (uint i = tid; i < n_vocab; i += nth) {
        const float value = logits[i];
        /* Metal libraries default to fast math. Classify the IEEE-754 bits
         * before comparing so every NaN payload/sign is ignored explicitly;
         * the numeric compare alone is not a sufficient contract there. */
        const uint bits = as_type<uint>(value);
        const bool is_nan =
            (bits & 0x7f800000u) == 0x7f800000u &&
            (bits & 0x007fffffu) != 0u;
        if (!is_nan && value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1u; step != 0u; step >>= 1u) {
        if (tid < step) {
            const float other_value = best_values[tid + step];
            const uint other_index = best_indices[tid + step];
            if (other_value > best_values[tid] ||
                (other_value == best_values[tid] &&
                 other_index < best_indices[tid])) {
                best_values[tid] = other_value;
                best_indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) out_idx[0] = (int32_t)best_indices[0];
}

/*
 * Optional Q8_0 Laguna lm-head screen.
 *
 * The screen is deliberately kept separate from the ordinary logits path:
 * it is only useful for the raw top-1 route, and its scratch values must
 * never be mistaken for a complete logits row.  The packed block keeps the
 * original half scale and the high nibble of each signed Q8 value.  For a
 * raw byte q, u = (high_nibble(q) xor 8), and q0 = 16*u - 120.5.  Therefore
 * |q-q0| <= 7.5 for every possible int8 q.
 */
struct ds4_metal_args_laguna_q8_lmhead_pack {
    uint32_t n_blocks;
    uint32_t packed_block_bytes;
};

struct ds4_metal_args_laguna_q8_lmhead_coarse {
    uint32_t n_blocks;
    uint32_t n_rows;
    uint32_t packed_block_bytes;
    uint32_t pad0;
};

struct ds4_metal_laguna_q8_lmhead_screen_stats {
    atomic_uint candidate_rows;
    atomic_uint candidate_row_blocks;
    atomic_uint coarse_nonfinite;
    atomic_uint exact_row_blocks;
    float      winner_value;
    int32_t    winner_index;
    uint32_t   pad1;
    /* Keep the complete v1 prefix ABI stable for parent-source overrides. */
    uint        compact_pair_count;
    atomic_uint exact_dispatch_groups;
};

static inline bool laguna_q8_screen_nonfinite(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

static inline bool laguna_q8_screen_is_nan(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x7f800000u) == 0x7f800000u &&
        (bits & 0x007fffffu) != 0u;
}

static inline bool laguna_q8_screen_x_safe(float value) {
    const uint bits = as_type<uint>(value);
    const uint abs_bits = bits & 0x7fffffffu;
    if ((abs_bits & 0x7f800000u) == 0x7f800000u) return false;
    if (abs_bits == 0u) return true;
    /* 2^-78 <= |x| <= 2^90.  Outside this interval the coarse A path can
     * lose a nonzero product to FTZ/fast-math; admit the whole row instead. */
    return abs_bits >= 0x18800000u && abs_bits <= 0x6c800000u;
}

static inline float laguna_q8_screen_next_up(float value) {
    const uint bits = as_type<uint>(value);
    const uint abs_bits = bits & 0x7fffffffu;
    if (abs_bits == 0u) return as_type<float>(0x00000001u);
    if ((bits & 0x80000000u) != 0u) {
        return as_type<float>(bits - 1u);
    }
    return as_type<float>(bits + 1u);
}

static inline uchar laguna_q8_screen_nibble(uchar raw) {
    return (uchar)(((raw >> 4u) ^ 8u) & 0x0fu);
}

kernel void kernel_laguna_q8_lmhead_pack(
        device const block_q8_0 *src [[buffer(1)]],
        device uchar            *dst [[buffer(2)]],
        constant ds4_metal_args_laguna_q8_lmhead_pack &args [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_blocks) return;

    const device uchar *raw = (device const uchar *)(src + gid);
    device uchar *packed = dst + (uint64_t)gid * args.packed_block_bytes;
    packed[0] = raw[0];
    packed[1] = raw[1];
    for (uint i = 0u; i < 16u; i++) {
        const uchar lo = laguna_q8_screen_nibble(raw[2u + i * 2u]);
        const uchar hi = laguna_q8_screen_nibble(raw[3u + i * 2u]);
        packed[2u + i] = (uchar)(lo | (uchar)(hi << 4u));
    }
}

kernel void kernel_laguna_q8_lmhead_coarse(
        constant ds4_metal_args_laguna_q8_lmhead_coarse &args [[buffer(0)]],
        device const uchar *packed [[buffer(1)]],
        device const float *x [[buffer(2)]],
        device float *coarse [[buffer(3)]],
        device float *delta [[buffer(4)]],
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = 2;
    constexpr short NW = N_SIMDWIDTH;
    const short nsg = FC_mul_mv_nsg;
    const uint r0 = tgpig.x * NR0;
    const uint lane = (uint)sgitg * (uint)NW + (uint)tiisg;
    const uint stride = (uint)nsg * (uint)NW;

    float sum_c[NR0] = { 0.0f, 0.0f };
    float sum_a[NR0] = { 0.0f, 0.0f };
    bool unsafe[NR0] = { false, false };
    for (uint block = lane; block < args.n_blocks; block += stride) {
        const uint x0 = block * 32u;
        const bool have0 = r0 < args.n_rows;
        const bool have1 = r0 + 1u < args.n_rows;
        const device uchar *src0 = have0 ? packed +
            ((uint64_t)r0 * args.n_blocks + block) *
                args.packed_block_bytes : nullptr;
        const device uchar *src1 = have1 ? packed +
            ((uint64_t)(r0 + 1u) * args.n_blocks + block) *
                args.packed_block_bytes : nullptr;
        const float d0 = have0 ? (float)*((device const half *)src0) : 0.0f;
        const float d1 = have1 ? (float)*((device const half *)src1) : 0.0f;
        if (have0 && laguna_q8_screen_nonfinite(d0)) unsafe[0] = true;
        if (have1 && laguna_q8_screen_nonfinite(d1)) unsafe[1] = true;
        float block_abs_x = 0.0f;
        for (uint i = 0u; i < 16u; i++) {
            const float x0v = x[x0 + i * 2u];
            const float x1v = x[x0 + i * 2u + 1u];
            if (!laguna_q8_screen_x_safe(x0v) ||
                !laguna_q8_screen_x_safe(x1v)) {
                if (have0) unsafe[0] = true;
                if (have1) unsafe[1] = true;
            }
            block_abs_x += fabs(x0v) + fabs(x1v);
            if (have0) {
                const uchar packed_q = src0[2u + i];
                const uint u0 = (uint)(packed_q & 0x0fu);
                const uint u1 = (uint)(packed_q >> 4u);
                const float q0 = 16.0f * (float)u0 - 120.5f;
                const float q1 = 16.0f * (float)u1 - 120.5f;
                sum_c[0] += d0 * q0 * x0v;
                sum_c[0] += d0 * q1 * x1v;
            }
            if (have1) {
                const uchar packed_q = src1[2u + i];
                const uint u0 = (uint)(packed_q & 0x0fu);
                const uint u1 = (uint)(packed_q >> 4u);
                const float q0 = 16.0f * (float)u0 - 120.5f;
                const float q1 = 16.0f * (float)u1 - 120.5f;
                sum_c[1] += d1 * q0 * x0v;
                sum_c[1] += d1 * q1 * x1v;
            }
        }
        if (have0) sum_a[0] += fabs(d0) * block_abs_x;
        if (have1) sum_a[1] += fabs(d1) * block_abs_x;
    }

    for (short row = 0; row < NR0; row++) {
        if (unsafe[row]) {
            sum_c[row] = as_type<float>(0x7fc00000u);
            sum_a[row] = as_type<float>(0x7fc00000u);
        }
    }

    threadgroup float *sh_c = (threadgroup float *)shmem;
    threadgroup float *sh_a = sh_c + NR0 * NW;
    for (short row = 0; row < NR0; row++) {
        if (sgitg == 0) {
            sh_c[row * NW + tiisg] = 0.0f;
            sh_a[row * NW + tiisg] = 0.0f;
        }
        sum_c[row] = simd_sum(sum_c[row]);
        sum_a[row] = simd_sum(sum_a[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; row++) {
        if (tiisg == 0) {
            sh_c[row * NW + sgitg] = sum_c[row];
            sh_a[row * NW + sgitg] = sum_a[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; row++) {
        const float c = simd_sum(sh_c[row * NW + tiisg]);
        const float a = simd_sum(sh_a[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0 && r0 + row < args.n_rows) {
            coarse[r0 + row] = c;
            /* The factor eight includes the 0.5*A rounding surplus.  Any
             * nonfinite operand/result is admitted by the candidate pass. */
            delta[r0 + row] = 8.0f * a;
        }
    }
}

/* The stock Q8 helper writes one contiguous NR0=2 pair.  This seed kernel
 * selects the coarse winner dynamically, then calls that helper verbatim so
 * the exact seed follows the production reduction tree. */
kernel void kernel_laguna_q8_lmhead_seed_exact(
        constant ds4_metal_args_mul_mv &args [[buffer(0)]],
        device const char *src0 [[buffer(1)]],
        device const char *src1 [[buffer(2)]],
        device char *dst [[buffer(3)]],
        device const int32_t *seed_idx [[buffer(4)]],
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int row = seed_idx[0];
    if (row < 0 || row >= args.ne01) return;
    kernel_mul_mv_q8_0_f32_impl<2, constant ds4_metal_args_mul_mv &>(
        args, src0, src1, dst, shmem,
        uint3((uint)row / 2u, 0u, 0u), tiisg, sgitg);
}

kernel void kernel_laguna_q8_lmhead_candidates(
        constant uint &n_rows [[buffer(0)]],
        constant uint &n_blocks [[buffer(1)]],
        device const float *coarse [[buffer(2)]],
        device const float *delta [[buffer(3)]],
        device const float *exact_values [[buffer(4)]],
        device const int32_t *seed_idx [[buffer(5)]],
        device uchar *candidate [[buffer(6)]],
        device ds4_metal_laguna_q8_lmhead_screen_stats *stats [[buffer(7)]],
        constant uint &collect_stats [[buffer(8)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n_rows) return;
    const int32_t seed = seed_idx[0];
    float threshold = -1.0e30f;
    bool seed_nonfinite = false;
    if (seed >= 0 && (uint)seed < n_rows) {
        const float seed_value = exact_values[(uint)seed];
        seed_nonfinite = laguna_q8_screen_nonfinite(seed_value);
        if (!seed_nonfinite && seed_value > threshold) {
            threshold = seed_value;
        }
    }
    const float c = coarse[gid];
    const float d = delta[gid];
    const bool nonfinite = laguna_q8_screen_nonfinite(c) ||
        laguna_q8_screen_nonfinite(d) ||
        laguna_q8_screen_nonfinite(threshold);
    const float upper = c + d;
    const float outward_upper = laguna_q8_screen_nonfinite(upper) ?
        upper : laguna_q8_screen_next_up(upper);
    /* A nonfinite exact seed invalidates the threshold for every row.  This
     * is deliberately broader than the per-row coarse check: the seed is
     * the only value available before screening, and +/-Inf are valid stock
     * argmax values (only NaNs are ignored). */
    const bool admit = gid == 0u || seed_nonfinite || (int32_t)gid == seed ||
        nonfinite ||
        laguna_q8_screen_nonfinite(outward_upper) ||
        outward_upper >= threshold;
    candidate[gid] = admit ? (uchar)1 : (uchar)0;
    if (collect_stats && nonfinite) {
        atomic_fetch_add_explicit(&stats->coarse_nonfinite, 1u,
                                  memory_order_relaxed);
    }
    if (collect_stats && admit) {
        atomic_fetch_add_explicit(&stats->candidate_rows, 1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->candidate_row_blocks, n_blocks,
                                  memory_order_relaxed);
    }
}

/* Compact admitted NR0=2 pairs without a CPU readback.  This is intentionally
 * a single deterministic GPU thread rather than an atomic append: the
 * candidate pass already did the expensive per-row proof, and a bounded
 * 50,176-pair scan avoids introducing a trace-independent atomic hot spot.
 * The same pass writes Metal's three-u32 indirect-dispatch argument. */
struct ds4_metal_args_laguna_q8_lmhead_compact {
    uint32_t n_rows;
    uint32_t n_blocks;
    uint32_t pair_capacity;
};

kernel void kernel_laguna_q8_lmhead_compact_pairs(
        constant ds4_metal_args_laguna_q8_lmhead_compact &args [[buffer(0)]],
        device const uchar *candidate [[buffer(1)]],
        device uint *pair_ids [[buffer(2)]],
        device uint *dispatch_args [[buffer(3)]],
        device ds4_metal_laguna_q8_lmhead_screen_stats *stats [[buffer(4)]],
        constant uint &collect_stats [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;

    const uint n_pairs = (args.n_rows + 1u) / 2u;
    uint count = 0u;
    uint exact_rows = 0u;
    for (uint pair = 0u; pair < n_pairs; pair++) {
        const uint r0 = pair * 2u;
        const bool active0 = candidate[r0] != 0u;
        const bool active1 = r0 + 1u < args.n_rows &&
            candidate[r0 + 1u] != 0u;
        if (!active0 && !active1) continue;

        /* n_pairs is the capacity for a row-bounded pair list.  Keep the
         * guard beside the write so a malformed future shape can never make
         * the indirect argument exceed the allocated list. */
        if (count < args.pair_capacity) {
            pair_ids[count] = pair;
            count++;
            /* The exact helper is NR0=2: admitting either row evaluates the
             * whole pair (and the seed pair is evaluated once separately).
             * Keep this physical traffic stat distinct from logical rows. */
            exact_rows += r0 + 1u < args.n_rows ? 2u : 1u;
        }
    }

    dispatch_args[0] = count;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
    if (collect_stats) {
        stats->compact_pair_count = count;
        atomic_store_explicit(&stats->exact_dispatch_groups,
                              0u,
                              memory_order_relaxed);
        atomic_store_explicit(&stats->exact_row_blocks,
                              exact_rows * args.n_blocks,
                              memory_order_relaxed);
    }
}

/* Exact values are only written for admitted rows.  Non-admitted rows may
 * retain arbitrary scratch bytes; the final reducer consults candidate first.
 * Calling the shared helper is intentional: it preserves the stock Q8
 * accumulation and simd/workgroup reduction order. */
kernel void kernel_laguna_q8_lmhead_exact_candidates(
        constant ds4_metal_args_mul_mv &args [[buffer(0)]],
        device const char *src0 [[buffer(1)]],
        device const char *src1 [[buffer(2)]],
        device char *dst [[buffer(3)]],
        device const uchar *candidate [[buffer(4)]],
        device const int32_t *seed_idx [[buffer(5)]],
        device ds4_metal_laguna_q8_lmhead_screen_stats *stats [[buffer(6)]],
        constant uint &collect_stats [[buffer(7)]],
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint r0 = tgpig.x * 2u;
    if (r0 >= (uint)args.ne01) return;
    /* seed_exact already evaluated this complete NR0=2 pair.  Reuse those
     * values instead of issuing duplicate exact traffic; the host initializes
     * exact_row_blocks with this mandatory pair when trace collection is on. */
    const uint seed_pair = ((uint)seed_idx[0] / 2u) * 2u;
    if (r0 == seed_pair) return;
    if (candidate[r0] == 0u &&
        (r0 + 1u >= (uint)args.ne01 || candidate[r0 + 1u] == 0u)) {
        return;
    }
    if (collect_stats && tiisg == 0u && sgitg == 0u) {
        const uint rows = r0 + 1u < (uint)args.ne01 ? 2u : 1u;
        atomic_fetch_add_explicit(&stats->exact_row_blocks,
                                  rows * ((uint)args.ne00 / 32u),
                                  memory_order_relaxed);
    }
    kernel_mul_mv_q8_0_f32_impl<2, constant ds4_metal_args_mul_mv &>(
        args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

/* Exact NR0=2 work for one compacted pair.  The seed pair is represented in
 * the compact list so the indirect group count is the logical compact pair
 * count; its exact value was already produced by seed_exact and is reused. */
kernel void kernel_laguna_q8_lmhead_exact_compacted(
        constant ds4_metal_args_mul_mv &args [[buffer(0)]],
        device const char *src0 [[buffer(1)]],
        device const char *src1 [[buffer(2)]],
        device char *dst [[buffer(3)]],
        device const uint *pair_ids [[buffer(4)]],
        device const int32_t *seed_idx [[buffer(5)]],
        device ds4_metal_laguna_q8_lmhead_screen_stats *stats [[buffer(6)]],
        constant uint &collect_stats [[buffer(7)]],
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint pair = pair_ids[tgpig.x];
    const uint r0 = pair * 2u;
    if (r0 >= (uint)args.ne01) return;

    const uint seed_pair = ((uint)seed_idx[0] / 2u) * 2u;
    if (r0 == seed_pair) {
        if (collect_stats && tiisg == 0u && sgitg == 0u) {
            atomic_fetch_add_explicit(&stats->exact_dispatch_groups,
                                      1u,
                                      memory_order_relaxed);
        }
        return;
    }

    kernel_mul_mv_q8_0_f32_impl<2, constant ds4_metal_args_mul_mv &>(
        args, src0, src1, dst, shmem,
        uint3(pair, 0u, 0u), tiisg, sgitg);
    if (collect_stats && tiisg == 0u && sgitg == 0u) {
        atomic_fetch_add_explicit(&stats->exact_dispatch_groups,
                                  1u,
                                  memory_order_relaxed);
    }
    /* exact_row_blocks is written once by compaction.  This execution counter
     * is trace-only, so the benchmark path still performs no stats atomics. */
}

kernel void kernel_laguna_q8_lmhead_argmax_candidates(
        device const float *values [[buffer(0)]],
        device const uchar *candidate [[buffer(1)]],
        device int32_t *out_idx [[buffer(2)]],
        device float *out_value [[buffer(3)]],
        device ds4_metal_laguna_q8_lmhead_screen_stats *stats [[buffer(4)]],
        constant uint &n_rows [[buffer(5)]],
        constant uint &collect_stats [[buffer(6)]],
        threadgroup float *best_values [[threadgroup(0)]],
        threadgroup uint *best_indices [[threadgroup(1)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]]) {
    const uint nth = (uint)ntg_u.x;
    float best_value = -1.0e30f;
    uint best_index = 0u;
    for (uint i = tid; i < n_rows; i += nth) {
        if (candidate[i] == 0u) continue;
        const float raw_value = values[i];
        const float value = raw_value == 0.0f ? 0.0f : raw_value;
        /* sample_argmax ignores NaN, but accepts +/-Inf as ordinary ordered
         * values.  Do not use the broader nonfinite predicate here. */
        if (laguna_q8_screen_is_nan(value)) continue;
        if (value > best_value ||
            (value == best_value && i < best_index)) {
            best_value = value;
            best_index = i;
        }
    }
    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1u; step != 0u; step >>= 1u) {
        if (tid < step) {
            const float other_value = best_values[tid + step];
            const uint other_index = best_indices[tid + step];
            if (other_value > best_values[tid] ||
                (other_value == best_values[tid] &&
                 other_index < best_indices[tid])) {
                best_values[tid] = other_value;
                best_indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        const uint winner = best_indices[0];
        const float raw_winner = values[winner];
        out_idx[0] = (int32_t)winner;
        /* Return the exact selected row's bits, including a stock -0 or a
         * value below the sentinel when index zero remains selected.  The
         * normalized value above is used only for ordering. */
        out_value[0] = raw_winner;
        if (collect_stats) {
            stats->winner_index = (int32_t)winner;
            stats->winner_value = raw_winner;
        }
    }
}

struct ds4_metal_args_laguna_norm_rope {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float    eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t pad0;
};

static inline void laguna_head_rms_norm_rope_neox(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *row,
        device const float *weight,
        threadgroup float  *scratch [[threadgroup(0)]],
        uint tid,
        uint nth,
        uint token) {
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nth) {
        const float v = row[i];
        ss += v * v;
    }
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1u; step != 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(scratch[0] / (float)args.head_dim + args.eps);
    for (uint i = tid; i < args.head_dim; i += nth) {
        row[i] = row[i] * inv * weight[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    const uint half_rot = args.n_rot >> 1u;
    if (tid >= half_rot) return;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        rope_yarn_corr_dims((int)args.n_rot,
                            (int)args.n_ctx_orig,
                            args.freq_base,
                            args.beta_fast,
                            args.beta_slow,
                            corr_dims);
    }
    const int rel_i0 = (int)(tid * 2u);
    const float inv_ndims = -1.0f / (float)args.n_rot;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
    const float theta = (float)(args.pos0 + token) *
        exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
    const float theta = (float)(args.pos0 + token) *
        pow(args.freq_base, inv_ndims * (float)rel_i0);
#endif
    float cos_theta;
    float sin_theta;
    rope_yarn(theta,
              args.freq_scale,
              corr_dims,
              rel_i0,
              args.ext_factor,
              args.attn_factor,
              &cos_theta,
              &sin_theta);
    const float x0 = row[tid];
    const float x1 = row[tid + half_rot];
    row[tid] = x0 * cos_theta - x1 * sin_theta;
    row[tid + half_rot] = x0 * sin_theta + x1 * cos_theta;
}

// Laguna uses Qwen-style per-head RMSNorm and NeoX rotary pairs. Rotary
// dimensions occupy the prefix of each head; any remaining dimensions are
// normalized but left unrotated.
kernel void kernel_laguna_head_rms_norm_rope_neox(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *x,
        device const float *weight,
        threadgroup float  *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens ||
        args.head_dim == 0u || args.n_rot > args.head_dim ||
        (args.n_rot & 1u) != 0u) {
        return;
    }

    device float *row = x +
        ((uint64_t)token * args.n_head + head) * args.head_dim;
    laguna_head_rms_norm_rope_neox(
        args, row, weight, scratch, tid, ntg_u.x, token);
}

// Decode uses the same norm/RoPE arithmetic for Q and K. Keeping both tensors
// in one grid removes a small Metal dispatch without changing the per-head
// reduction order.
kernel void kernel_laguna_qk_head_rms_norm_rope_neox(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *q,
        device float       *k,
        device const float *q_weight,
        device const float *k_weight,
        constant uint      &n_q_head,
        threadgroup float  *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint combined_head = tgpig.x;
    const uint token = tgpig.y;
    if (combined_head >= args.n_head || token >= args.n_tokens ||
        n_q_head >= args.n_head || args.head_dim == 0u ||
        args.n_rot > args.head_dim || (args.n_rot & 1u) != 0u) {
        return;
    }

    const bool is_q = combined_head < n_q_head;
    const uint tensor_head = is_q ? combined_head : combined_head - n_q_head;
    const uint tensor_n_head = is_q ? n_q_head : args.n_head - n_q_head;
    device float *row = (is_q ? q : k) +
        ((uint64_t)token * tensor_n_head + tensor_head) * args.head_dim;
    device const float *weight = is_q ? q_weight : k_weight;
    laguna_head_rms_norm_rope_neox(
        args, row, weight, scratch, tid, ntg_u.x, token);
}

struct ds4_metal_args_laguna_kv_store {
    uint32_t cache_cap;
    uint32_t cache_row;
    uint32_t n_head_kv;
    uint32_t head_dim;
};

kernel void kernel_laguna_store_kv_f16(
        constant ds4_metal_args_laguna_kv_store &args,
        device const float *k,
        device const float *v,
        device half *key_cache,
        device half *value_cache,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    if (gid >= width || args.cache_row >= args.cache_cap) return;
    const uint64_t dst = (uint64_t)args.cache_row * width + gid;
    key_cache[dst] = (half)k[gid];
    value_cache[dst] = (half)v[gid];
}

struct ds4_metal_args_laguna_prefill_attention {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    float    scale;
    uint32_t pad0;
};

// Before the sliding window wraps, verifier rows occupy distinct cache slots.
// Store the complete speculative block at once; each query still limits its
// key count, so later rows cannot become visible to earlier queries.
kernel void kernel_laguna_store_kv_rows_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *k,
        device const float *v,
        device half *key_cache,
        device half *value_cache,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    const uint values = args.n_tokens * width;
    if (gid >= values) return;
    const uint token = gid / width;
    const uint col = gid - token * width;
    const uint cache_row = (args.pos0 + token) % args.cache_cap;
    const uint64_t dst = (uint64_t)cache_row * width + col;
    key_cache[dst] = (half)k[gid];
    value_cache[dst] = (half)v[gid];
}

// Stage the current chunk as f16 before attention. This preserves the same KV
// precision as decode without overwriting sliding-window rows that early
// queries in the chunk still need.
kernel void kernel_laguna_stage_kv_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *k,
        device const float *v,
        device half *staged_key,
        device half *staged_value,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    const uint values = args.n_tokens * width;
    if (gid >= values) return;
    staged_key[gid] = (half)k[gid];
    staged_value[gid] = (half)v[gid];
}

// One SIMD group owns one query head. Queries in a prefill chunk execute in
// parallel, while each query visits keys in causal order. Keys from the current
// chunk come from the staging buffer; older keys come from the persistent ring.
kernel void kernel_laguna_attention_prefill_gqa_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const half  *staged_key,
        device const half  *staged_value,
        device float       *out,
        ushort lane [[thread_index_in_simdgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        args.cache_cap == 0u) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint query_pos = args.pos0 + token;
    const uint key_count = min(query_pos + 1u, args.cache_cap);
    const uint key_start = query_pos + 1u - key_count;
    device const float *qh = q +
        ((uint64_t)token * args.n_head + head) * args.head_dim;

    float4 acc = float4(0.0f);
    float max_score = -INFINITY;
    float score_sum = 0.0f;
    for (uint key_pos = key_start; key_pos <= query_pos; key_pos++) {
        const bool current = key_pos >= args.pos0;
        const uint source_row = current ?
            key_pos - args.pos0 : key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)source_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            const float key_value = current ?
                (float)staged_key[kv_base + d] :
                (float)key_cache[kv_base + d];
            partial += qh[d] * key_value;
        }
        const float score = simd_sum(partial) * args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ?
            0.0f : exp(max_score - next_max);
        const float value_scale = exp(score - next_max);
        score_sum = score_sum * old_scale + value_scale;
        const uint d0 = lane;
        const float4 value = current ?
            float4((float)staged_value[kv_base + d0],
                   (float)staged_value[kv_base + d0 + 32u],
                   (float)staged_value[kv_base + d0 + 64u],
                   (float)staged_value[kv_base + d0 + 96u]) :
            float4((float)value_cache[kv_base + d0],
                   (float)value_cache[kv_base + d0 + 32u],
                   (float)value_cache[kv_base + d0 + 64u],
                   (float)value_cache[kv_base + d0 + 96u]);
        acc = acc * old_scale + value * value_scale;
        max_score = next_max;
    }

    const float inv_sum = score_sum > 0.0f ? 1.0f / score_sum : 0.0f;
    const float gate_value = gate[(uint64_t)token * args.n_head + head];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    device float *oh = out +
        ((uint64_t)token * args.n_head + head) * args.head_dim;
    oh[lane]       = acc.x * inv_sum * gate_scale;
    oh[lane + 32u] = acc.y * inv_sum * gate_scale;
    oh[lane + 64u] = acc.z * inv_sum * gate_scale;
    oh[lane + 96u] = acc.w * inv_sum * gate_scale;
}

// Three adjacent query heads sharing one KV head execute in one SIMD group.
// The per-head reduction and causal key order are unchanged, while each K/V
// value is fetched only once for the group.
kernel void kernel_laguna_attention_prefill_gqa3_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const half  *staged_key,
        device const half  *staged_value,
        device float       *out,
        ushort lane [[thread_index_in_simdgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head0 = tgpig.x * 3u;
    const uint head1 = head0 + 1u;
    const uint head2 = head0 + 2u;
    const uint token = tgpig.y;
    if (head2 >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        args.cache_cap == 0u) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if (head2 / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint query_pos = args.pos0 + token;
    const uint key_count = min(query_pos + 1u, args.cache_cap);
    const uint key_start = query_pos + 1u - key_count;
    device const float *qh0 = q +
        ((uint64_t)token * args.n_head + head0) * args.head_dim;
    device const float *qh1 = qh0 + args.head_dim;
    device const float *qh2 = qh1 + args.head_dim;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float max2 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (uint key_pos = key_start; key_pos <= query_pos; key_pos++) {
        const bool current = key_pos >= args.pos0;
        const uint source_row = current ?
            key_pos - args.pos0 : key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)source_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            const float key_value = current ?
                (float)staged_key[kv_base + d] :
                (float)key_cache[kv_base + d];
            partial0 += qh0[d] * key_value;
            partial1 += qh1[d] * key_value;
            partial2 += qh2[d] * key_value;
        }
        const float score0 = simd_sum(partial0) * args.scale;
        const float score1 = simd_sum(partial1) * args.scale;
        const float score2 = simd_sum(partial2) * args.scale;
        const float next_max0 = max(max0, score0);
        const float next_max1 = max(max1, score1);
        const float next_max2 = max(max2, score2);
        const float old_scale0 = max0 == -INFINITY ?
            0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ?
            0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ?
            0.0f : exp(max2 - next_max2);
        const float value_scale0 = exp(score0 - next_max0);
        const float value_scale1 = exp(score1 - next_max1);
        const float value_scale2 = exp(score2 - next_max2);
        sum0 = sum0 * old_scale0 + value_scale0;
        sum1 = sum1 * old_scale1 + value_scale1;
        sum2 = sum2 * old_scale2 + value_scale2;
        const uint d0 = lane;
        const float4 value = current ?
            float4((float)staged_value[kv_base + d0],
                   (float)staged_value[kv_base + d0 + 32u],
                   (float)staged_value[kv_base + d0 + 64u],
                   (float)staged_value[kv_base + d0 + 96u]) :
            float4((float)value_cache[kv_base + d0],
                   (float)value_cache[kv_base + d0 + 32u],
                   (float)value_cache[kv_base + d0 + 64u],
                   (float)value_cache[kv_base + d0 + 96u]);
        acc0 = acc0 * old_scale0 + value * value_scale0;
        acc1 = acc1 * old_scale1 + value * value_scale1;
        acc2 = acc2 * old_scale2 + value * value_scale2;
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
    }

    const float inv_sum0 = sum0 > 0.0f ? 1.0f / sum0 : 0.0f;
    const float inv_sum1 = sum1 > 0.0f ? 1.0f / sum1 : 0.0f;
    const float inv_sum2 = sum2 > 0.0f ? 1.0f / sum2 : 0.0f;
    const float gate_value0 = gate[(uint64_t)token * args.n_head + head0];
    const float gate_value1 = gate[(uint64_t)token * args.n_head + head1];
    const float gate_value2 = gate[(uint64_t)token * args.n_head + head2];
    const float gate_scale0 = gate_value0 > 20.0f ?
        gate_value0 : log(1.0f + exp(gate_value0));
    const float gate_scale1 = gate_value1 > 20.0f ?
        gate_value1 : log(1.0f + exp(gate_value1));
    const float gate_scale2 = gate_value2 > 20.0f ?
        gate_value2 : log(1.0f + exp(gate_value2));
    device float *oh0 = out +
        ((uint64_t)token * args.n_head + head0) * args.head_dim;
    device float *oh1 = oh0 + args.head_dim;
    device float *oh2 = oh1 + args.head_dim;
    oh0[lane]       = acc0.x * inv_sum0 * gate_scale0;
    oh0[lane + 32u] = acc0.y * inv_sum0 * gate_scale0;
    oh0[lane + 64u] = acc0.z * inv_sum0 * gate_scale0;
    oh0[lane + 96u] = acc0.w * inv_sum0 * gate_scale0;
    oh1[lane]       = acc1.x * inv_sum1 * gate_scale1;
    oh1[lane + 32u] = acc1.y * inv_sum1 * gate_scale1;
    oh1[lane + 64u] = acc1.z * inv_sum1 * gate_scale1;
    oh1[lane + 96u] = acc1.w * inv_sum1 * gate_scale1;
    oh2[lane]       = acc2.x * inv_sum2 * gate_scale2;
    oh2[lane + 32u] = acc2.y * inv_sum2 * gate_scale2;
    oh2[lane + 64u] = acc2.z * inv_sum2 * gate_scale2;
    oh2[lane + 96u] = acc2.w * inv_sum2 * gate_scale2;
}

// Global Laguna layers have six query heads per KV head. Keeping all six in
// one SIMD group preserves each head's reduction order while halving K/V
// traffic relative to the three-head kernel.
kernel void kernel_laguna_attention_prefill_gqa6_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const half  *staged_key,
        device const half  *staged_value,
        device float       *out,
        ushort lane [[thread_index_in_simdgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head0 = tgpig.x * 6u;
    const uint head5 = head0 + 5u;
    const uint token = tgpig.y;
    if (head5 >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        args.cache_cap == 0u) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if (head5 / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint query_pos = args.pos0 + token;
    const uint key_count = min(query_pos + 1u, args.cache_cap);
    const uint key_start = query_pos + 1u - key_count;
    device const float *qh0 = q +
        ((uint64_t)token * args.n_head + head0) * args.head_dim;
    device const float *qh1 = qh0 + args.head_dim;
    device const float *qh2 = qh1 + args.head_dim;
    device const float *qh3 = qh2 + args.head_dim;
    device const float *qh4 = qh3 + args.head_dim;
    device const float *qh5 = qh4 + args.head_dim;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    float4 acc4 = float4(0.0f);
    float4 acc5 = float4(0.0f);
    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float max2 = -INFINITY;
    float max3 = -INFINITY;
    float max4 = -INFINITY;
    float max5 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    for (uint key_pos = key_start; key_pos <= query_pos; key_pos++) {
        const bool current = key_pos >= args.pos0;
        const uint source_row = current ?
            key_pos - args.pos0 : key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)source_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        float partial3 = 0.0f;
        float partial4 = 0.0f;
        float partial5 = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            const float key_value = current ?
                (float)staged_key[kv_base + d] :
                (float)key_cache[kv_base + d];
            partial0 += qh0[d] * key_value;
            partial1 += qh1[d] * key_value;
            partial2 += qh2[d] * key_value;
            partial3 += qh3[d] * key_value;
            partial4 += qh4[d] * key_value;
            partial5 += qh5[d] * key_value;
        }
        const float score0 = simd_sum(partial0) * args.scale;
        const float score1 = simd_sum(partial1) * args.scale;
        const float score2 = simd_sum(partial2) * args.scale;
        const float score3 = simd_sum(partial3) * args.scale;
        const float score4 = simd_sum(partial4) * args.scale;
        const float score5 = simd_sum(partial5) * args.scale;
        const float next_max0 = max(max0, score0);
        const float next_max1 = max(max1, score1);
        const float next_max2 = max(max2, score2);
        const float next_max3 = max(max3, score3);
        const float next_max4 = max(max4, score4);
        const float next_max5 = max(max5, score5);
        const float old_scale0 = max0 == -INFINITY ?
            0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ?
            0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ?
            0.0f : exp(max2 - next_max2);
        const float old_scale3 = max3 == -INFINITY ?
            0.0f : exp(max3 - next_max3);
        const float old_scale4 = max4 == -INFINITY ?
            0.0f : exp(max4 - next_max4);
        const float old_scale5 = max5 == -INFINITY ?
            0.0f : exp(max5 - next_max5);
        const float value_scale0 = exp(score0 - next_max0);
        const float value_scale1 = exp(score1 - next_max1);
        const float value_scale2 = exp(score2 - next_max2);
        const float value_scale3 = exp(score3 - next_max3);
        const float value_scale4 = exp(score4 - next_max4);
        const float value_scale5 = exp(score5 - next_max5);
        sum0 = sum0 * old_scale0 + value_scale0;
        sum1 = sum1 * old_scale1 + value_scale1;
        sum2 = sum2 * old_scale2 + value_scale2;
        sum3 = sum3 * old_scale3 + value_scale3;
        sum4 = sum4 * old_scale4 + value_scale4;
        sum5 = sum5 * old_scale5 + value_scale5;
        const uint d0 = lane;
        const float4 value = current ?
            float4((float)staged_value[kv_base + d0],
                   (float)staged_value[kv_base + d0 + 32u],
                   (float)staged_value[kv_base + d0 + 64u],
                   (float)staged_value[kv_base + d0 + 96u]) :
            float4((float)value_cache[kv_base + d0],
                   (float)value_cache[kv_base + d0 + 32u],
                   (float)value_cache[kv_base + d0 + 64u],
                   (float)value_cache[kv_base + d0 + 96u]);
        acc0 = acc0 * old_scale0 + value * value_scale0;
        acc1 = acc1 * old_scale1 + value * value_scale1;
        acc2 = acc2 * old_scale2 + value * value_scale2;
        acc3 = acc3 * old_scale3 + value * value_scale3;
        acc4 = acc4 * old_scale4 + value * value_scale4;
        acc5 = acc5 * old_scale5 + value * value_scale5;
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
        max3 = next_max3;
        max4 = next_max4;
        max5 = next_max5;
    }

    const float inv_sum0 = sum0 > 0.0f ? 1.0f / sum0 : 0.0f;
    const float inv_sum1 = sum1 > 0.0f ? 1.0f / sum1 : 0.0f;
    const float inv_sum2 = sum2 > 0.0f ? 1.0f / sum2 : 0.0f;
    const float inv_sum3 = sum3 > 0.0f ? 1.0f / sum3 : 0.0f;
    const float inv_sum4 = sum4 > 0.0f ? 1.0f / sum4 : 0.0f;
    const float inv_sum5 = sum5 > 0.0f ? 1.0f / sum5 : 0.0f;
    const uint64_t gate_base = (uint64_t)token * args.n_head + head0;
    const float gate_value0 = gate[gate_base];
    const float gate_value1 = gate[gate_base + 1u];
    const float gate_value2 = gate[gate_base + 2u];
    const float gate_value3 = gate[gate_base + 3u];
    const float gate_value4 = gate[gate_base + 4u];
    const float gate_value5 = gate[gate_base + 5u];
    const float gate_scale0 = gate_value0 > 20.0f ?
        gate_value0 : log(1.0f + exp(gate_value0));
    const float gate_scale1 = gate_value1 > 20.0f ?
        gate_value1 : log(1.0f + exp(gate_value1));
    const float gate_scale2 = gate_value2 > 20.0f ?
        gate_value2 : log(1.0f + exp(gate_value2));
    const float gate_scale3 = gate_value3 > 20.0f ?
        gate_value3 : log(1.0f + exp(gate_value3));
    const float gate_scale4 = gate_value4 > 20.0f ?
        gate_value4 : log(1.0f + exp(gate_value4));
    const float gate_scale5 = gate_value5 > 20.0f ?
        gate_value5 : log(1.0f + exp(gate_value5));
    device float *oh0 = out + gate_base * args.head_dim;
    device float *oh1 = oh0 + args.head_dim;
    device float *oh2 = oh1 + args.head_dim;
    device float *oh3 = oh2 + args.head_dim;
    device float *oh4 = oh3 + args.head_dim;
    device float *oh5 = oh4 + args.head_dim;
    oh0[lane]       = acc0.x * inv_sum0 * gate_scale0;
    oh0[lane + 32u] = acc0.y * inv_sum0 * gate_scale0;
    oh0[lane + 64u] = acc0.z * inv_sum0 * gate_scale0;
    oh0[lane + 96u] = acc0.w * inv_sum0 * gate_scale0;
    oh1[lane]       = acc1.x * inv_sum1 * gate_scale1;
    oh1[lane + 32u] = acc1.y * inv_sum1 * gate_scale1;
    oh1[lane + 64u] = acc1.z * inv_sum1 * gate_scale1;
    oh1[lane + 96u] = acc1.w * inv_sum1 * gate_scale1;
    oh2[lane]       = acc2.x * inv_sum2 * gate_scale2;
    oh2[lane + 32u] = acc2.y * inv_sum2 * gate_scale2;
    oh2[lane + 64u] = acc2.z * inv_sum2 * gate_scale2;
    oh2[lane + 96u] = acc2.w * inv_sum2 * gate_scale2;
    oh3[lane]       = acc3.x * inv_sum3 * gate_scale3;
    oh3[lane + 32u] = acc3.y * inv_sum3 * gate_scale3;
    oh3[lane + 64u] = acc3.z * inv_sum3 * gate_scale3;
    oh3[lane + 96u] = acc3.w * inv_sum3 * gate_scale3;
    oh4[lane]       = acc4.x * inv_sum4 * gate_scale4;
    oh4[lane + 32u] = acc4.y * inv_sum4 * gate_scale4;
    oh4[lane + 64u] = acc4.z * inv_sum4 * gate_scale4;
    oh4[lane + 96u] = acc4.w * inv_sum4 * gate_scale4;
    oh5[lane]       = acc5.x * inv_sum5 * gate_scale5;
    oh5[lane + 32u] = acc5.y * inv_sum5 * gate_scale5;
    oh5[lane + 64u] = acc5.z * inv_sum5 * gate_scale5;
    oh5[lane + 96u] = acc5.w * inv_sum5 * gate_scale5;
}

kernel void kernel_laguna_commit_kv_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const half *staged_key,
        device const half *staged_value,
        device half *key_cache,
        device half *value_cache,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    const uint values = args.n_tokens * width;
    if (gid >= values) return;
    const uint token = gid / width;
    const uint col = gid - token * width;
    const uint cache_row = (args.pos0 + token) % args.cache_cap;
    const uint64_t dst = (uint64_t)cache_row * width + col;
    key_cache[dst] = staged_key[gid];
    value_cache[dst] = staged_value[gid];
}

struct ds4_metal_args_laguna_attention {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t cache_cap;
    uint32_t key_start;
    uint32_t key_count;
    float    scale;
    uint32_t pad0;
};

struct ds4_metal_args_laguna_gqa3_decode {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t key_count0;
    uint32_t n_tokens;
    uint32_t nsg;
    uint32_t nwg;
    float    scale;
};

// Global layers split keys across 32 workgroups. Evaluate three query heads
// sharing one KV head together so each K/V row is loaded once instead of three
// times. DFlash verifier rows use the second grid dimension while retaining
// the exact one-row arithmetic and established gated reduction.
kernel void kernel_laguna_attention_decode_gqa3_split_f16(
        constant ds4_metal_args_laguna_gqa3_decode &args,
        device const float *q,
        device const half  *key_cache,
        device const half  *value_cache,
        device float       *tmp,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head0 = tgpig.x * 3u;
    const uint head2 = head0 + 2u;
    const uint token = tgpig.y;
    const uint iwg = tgpig.z;
    const uint simd_group = (uint)simd_group_u;
    const uint key_count = args.key_count0 + token;
    if (head2 >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        key_count == 0u ||
        args.nsg == 0u || simd_group >= args.nsg || iwg >= args.nwg) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if (head2 / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint64_t query_base =
        (uint64_t)token * args.n_head * args.head_dim;
    device const float *qh0 = q + query_base +
                              (uint64_t)head0 * args.head_dim;
    device const float *qh1 = qh0 + args.head_dim;
    device const float *qh2 = qh1 + args.head_dim;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float max2 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    const uint first = iwg * args.nsg + simd_group;
    const uint stride = args.nwg * args.nsg;
    for (uint i = first; i < key_count; i += stride) {
        const uint64_t kv_base =
            (uint64_t)i * cache_width +
            (uint64_t)kv_head * args.head_dim;
        const uint d0 = lane;
        const float4 key = float4((float)key_cache[kv_base + d0],
                                  (float)key_cache[kv_base + d0 + 32u],
                                  (float)key_cache[kv_base + d0 + 64u],
                                  (float)key_cache[kv_base + d0 + 96u]);
        const float score0 = simd_sum(dot(float4(qh0[d0],
                                                   qh0[d0 + 32u],
                                                   qh0[d0 + 64u],
                                                   qh0[d0 + 96u]), key)) * args.scale;
        const float score1 = simd_sum(dot(float4(qh1[d0],
                                                   qh1[d0 + 32u],
                                                   qh1[d0 + 64u],
                                                   qh1[d0 + 96u]), key)) * args.scale;
        const float score2 = simd_sum(dot(float4(qh2[d0],
                                                   qh2[d0 + 32u],
                                                   qh2[d0 + 64u],
                                                   qh2[d0 + 96u]), key)) * args.scale;
        const float next_max0 = max(max0, score0);
        const float next_max1 = max(max1, score1);
        const float next_max2 = max(max2, score2);
        const float old_scale0 = max0 == -INFINITY ? 0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ? 0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ? 0.0f : exp(max2 - next_max2);
        const float value_scale0 = exp(score0 - next_max0);
        const float value_scale1 = exp(score1 - next_max1);
        const float value_scale2 = exp(score2 - next_max2);
        sum0 = sum0 * old_scale0 + value_scale0;
        sum1 = sum1 * old_scale1 + value_scale1;
        sum2 = sum2 * old_scale2 + value_scale2;
        const float4 value = float4((float)value_cache[kv_base + d0],
                                    (float)value_cache[kv_base + d0 + 32u],
                                    (float)value_cache[kv_base + d0 + 64u],
                                    (float)value_cache[kv_base + d0 + 96u]);
        acc0 = acc0 * old_scale0 + value * value_scale0;
        acc1 = acc1 * old_scale1 + value * value_scale1;
        acc2 = acc2 * old_scale2 + value * value_scale2;
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + 3u * args.nsg;
    threadgroup float *partial_value = partial_sum + 3u * args.nsg;
    const uint slots[3] = {simd_group,
                           args.nsg + simd_group,
                           2u * args.nsg + simd_group};
    const float maxima[3] = {max0, max1, max2};
    const float sums[3] = {sum0, sum1, sum2};
    const float4 values[3] = {acc0, acc1, acc2};
    for (uint h = 0u; h < 3u; h++) {
        const uint slot = slots[h];
        if (lane == 0u) {
            partial_max[slot] = maxima[h];
            partial_sum[slot] = sums[h];
        }
        const uint base = slot * args.head_dim + lane;
        partial_value[base] = values[h].x;
        partial_value[base + 32u] = values[h].y;
        partial_value[base + 64u] = values[h].z;
        partial_value[base + 96u] = values[h].w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;
    const uint nrows = args.n_tokens * args.n_head;
    device float *stats =
        tmp + (uint64_t)nrows * args.head_dim * args.nwg;
    for (uint h = 0u; h < 3u; h++) {
        const uint slot_base = h * args.nsg;
        float global_max = partial_max[slot_base];
        for (uint sg = 1u; sg < args.nsg; sg++) {
            global_max = max(global_max, partial_max[slot_base + sg]);
        }
        float merged_sum = 0.0f;
        float4 merged = float4(0.0f);
        for (uint sg = 0u; sg < args.nsg; sg++) {
            const uint slot = slot_base + sg;
            const float weight = partial_sum[slot] > 0.0f ?
                exp(partial_max[slot] - global_max) : 0.0f;
            merged_sum += partial_sum[slot] * weight;
            const uint base = slot * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }

        const uint row = token * args.n_head + head0 + h;
        const uint64_t row_base =
            (uint64_t)row * args.head_dim * args.nwg;
        const uint dims[4] = {lane, lane + 32u, lane + 64u, lane + 96u};
        const float outputs[4] = {merged.x, merged.y, merged.z, merged.w};
        for (uint j = 0u; j < 4u; j++) {
            const uint d = dims[j];
            tmp[row_base + (d / 4u) * args.nwg * 4u + iwg * 4u + d % 4u] =
                outputs[j];
        }
        if (lane == 0u) {
            const uint64_t stat = (uint64_t)row * 2u * args.nwg + 2u * iwg;
            stats[stat] = merged_sum;
            stats[stat + 1u] = global_max;
        }
    }
}

// GQA9 sibling of the split decode kernel: one simdgroup evaluates all 9
// query heads of a single KV head together.  The production SWA layer is 72 Q
// heads over 8 KV heads (heads_per_kv = 9), so grouping 9 heads removes the 3x
// redundant K/V traffic the GQA3 kernel still pays on a 9:1 ratio.  The per-
// head arithmetic, key striding (first = iwg*nsg + simd_group,
// stride = nwg*nsg), online-softmax update order, partial-merge layout, and
// tmp/stats interleave are kept identical to the GQA3 kernel: only the head
// group width changes.  Because each head's operation sequence is unchanged,
// the merged output is bit-exact with the GQA3 kernel for the same shape, so
// the reduce/gate kernel is reused unchanged.
kernel void kernel_laguna_attention_decode_gqa9_split_f16(
        constant ds4_metal_args_laguna_gqa3_decode &args,
        device const float *q,
        device const half  *key_cache,
        device const half  *value_cache,
        device float       *tmp,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint GROUP = 9u;
    const uint head0 = tgpig.x * GROUP;
    const uint head8 = head0 + (GROUP - 1u);
    const uint token = tgpig.y;
    const uint iwg = tgpig.z;
    const uint simd_group = (uint)simd_group_u;
    const uint key_count = args.key_count0 + token;
    if (head8 >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        key_count == 0u ||
        args.nsg == 0u || simd_group >= args.nsg || iwg >= args.nwg) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if (head8 / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint64_t query_base =
        (uint64_t)token * args.n_head * args.head_dim;
    device const float *qbase = q + query_base +
                                (uint64_t)head0 * args.head_dim;

    float4 acc[GROUP];
    float maxv[GROUP];
    float sumv[GROUP];
    for (uint h = 0u; h < GROUP; h++) {
        acc[h] = float4(0.0f);
        maxv[h] = -INFINITY;
        sumv[h] = 0.0f;
    }
    const uint first = iwg * args.nsg + simd_group;
    const uint stride = args.nwg * args.nsg;
    for (uint i = first; i < key_count; i += stride) {
        const uint64_t kv_base =
            (uint64_t)i * cache_width +
            (uint64_t)kv_head * args.head_dim;
        const uint d0 = lane;
        const float4 key = float4((float)key_cache[kv_base + d0],
                                  (float)key_cache[kv_base + d0 + 32u],
                                  (float)key_cache[kv_base + d0 + 64u],
                                  (float)key_cache[kv_base + d0 + 96u]);
        float score[GROUP];
        float next_max[GROUP];
        float old_scale[GROUP];
        float value_scale[GROUP];
        for (uint h = 0u; h < GROUP; h++) {
            device const float *qh = qbase + (uint64_t)h * args.head_dim;
            score[h] = simd_sum(dot(float4(qh[d0],
                                             qh[d0 + 32u],
                                             qh[d0 + 64u],
                                             qh[d0 + 96u]), key)) * args.scale;
            next_max[h] = max(maxv[h], score[h]);
            old_scale[h] = maxv[h] == -INFINITY ? 0.0f : exp(maxv[h] - next_max[h]);
            value_scale[h] = exp(score[h] - next_max[h]);
            sumv[h] = sumv[h] * old_scale[h] + value_scale[h];
        }
        const float4 value = float4((float)value_cache[kv_base + d0],
                                    (float)value_cache[kv_base + d0 + 32u],
                                    (float)value_cache[kv_base + d0 + 64u],
                                    (float)value_cache[kv_base + d0 + 96u]);
        for (uint h = 0u; h < GROUP; h++) {
            acc[h] = acc[h] * old_scale[h] + value * value_scale[h];
            maxv[h] = next_max[h];
        }
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + GROUP * args.nsg;
    threadgroup float *partial_value = partial_sum + GROUP * args.nsg;
    for (uint h = 0u; h < GROUP; h++) {
        const uint slot = h * args.nsg + simd_group;
        if (lane == 0u) {
            partial_max[slot] = maxv[h];
            partial_sum[slot] = sumv[h];
        }
        const uint base = slot * args.head_dim + lane;
        partial_value[base] = acc[h].x;
        partial_value[base + 32u] = acc[h].y;
        partial_value[base + 64u] = acc[h].z;
        partial_value[base + 96u] = acc[h].w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;
    const uint nrows = args.n_tokens * args.n_head;
    device float *stats =
        tmp + (uint64_t)nrows * args.head_dim * args.nwg;
    for (uint h = 0u; h < GROUP; h++) {
        const uint slot_base = h * args.nsg;
        float global_max = partial_max[slot_base];
        for (uint sg = 1u; sg < args.nsg; sg++) {
            global_max = max(global_max, partial_max[slot_base + sg]);
        }
        float merged_sum = 0.0f;
        float4 merged = float4(0.0f);
        for (uint sg = 0u; sg < args.nsg; sg++) {
            const uint slot = slot_base + sg;
            const float weight = partial_sum[slot] > 0.0f ?
                exp(partial_max[slot] - global_max) : 0.0f;
            merged_sum += partial_sum[slot] * weight;
            const uint base = slot * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }

        const uint row = token * args.n_head + head0 + h;
        const uint64_t row_base =
            (uint64_t)row * args.head_dim * args.nwg;
        const uint dims[4] = {lane, lane + 32u, lane + 64u, lane + 96u};
        const float outputs[4] = {merged.x, merged.y, merged.z, merged.w};
        for (uint j = 0u; j < 4u; j++) {
            const uint d = dims[j];
            tmp[row_base + (d / 4u) * args.nwg * 4u + iwg * 4u + d % 4u] =
                outputs[j];
        }
        if (lane == 0u) {
            const uint64_t stat = (uint64_t)row * 2u * args.nwg + 2u * iwg;
            stats[stat] = merged_sum;
            stats[stat + 1u] = global_max;
        }
    }
}

// One threadgroup owns one query head. Short histories retain the original
// single-SIMD online reduction. Longer histories are striped over eight SIMD
// groups, then their independently normalized partials are merged in
// threadgroup memory. This keeps decode latency from growing serially with the
// absolute context position while preserving the short-context arithmetic.
kernel void kernel_laguna_attention_decode_gqa_f16(
        constant ds4_metal_args_laguna_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device float       *out,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint head [[threadgroup_position_in_grid]]) {
    constexpr uint split_simd_groups = 8u;
    constexpr uint split_threshold = 256u;
    if (head >= args.n_head || args.n_head_kv == 0u ||
        args.head_dim != 128u || args.key_count == 0u) {
        return;
    }
    const uint simd_group = (uint)simd_group_u;
    const bool split = args.key_count > split_threshold;
    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const uint cache_width = args.n_head_kv * args.head_dim;
    device const float *qh = q + (uint64_t)head * args.head_dim;

    float4 acc = float4(0.0f);
    float max_score = -INFINITY;
    float score_sum = 0.0f;
    const uint key_first = split ? simd_group : 0u;
    const uint key_stride = split ? split_simd_groups : 1u;
    for (uint i = key_first; i < args.key_count; i += key_stride) {
        const uint key_pos = args.key_start + i;
        const uint cache_row = key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)cache_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            partial += qh[d] * (float)key_cache[kv_base + d];
        }
        const float score = simd_sum(partial) * args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ?
            0.0f : exp(max_score - next_max);
        const float value_scale = exp(score - next_max);
        score_sum = score_sum * old_scale + value_scale;
        const uint d0 = lane;
        acc.x = acc.x * old_scale +
            value_scale * (float)value_cache[kv_base + d0];
        acc.y = acc.y * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 32u];
        acc.z = acc.z * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 64u];
        acc.w = acc.w * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 96u];
        max_score = next_max;
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + split_simd_groups;
    threadgroup float *partial_value = partial_sum + split_simd_groups;
    if (lane == 0u) {
        partial_max[simd_group] = max_score;
        partial_sum[simd_group] = score_sum;
    }
    const uint value_base = simd_group * args.head_dim;
    partial_value[value_base + lane] = acc.x;
    partial_value[value_base + lane + 32u] = acc.y;
    partial_value[value_base + lane + 64u] = acc.z;
    partial_value[value_base + lane + 96u] = acc.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;

    float4 merged = acc;
    float merged_sum = score_sum;
    if (split) {
        float global_max = partial_max[0];
        for (uint sg = 1u; sg < split_simd_groups; sg++) {
            global_max = max(global_max, partial_max[sg]);
        }
        merged = float4(0.0f);
        merged_sum = 0.0f;
        for (uint sg = 0u; sg < split_simd_groups; sg++) {
            const float weight = partial_sum[sg] > 0.0f ?
                exp(partial_max[sg] - global_max) : 0.0f;
            merged_sum += partial_sum[sg] * weight;
            const uint base = sg * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }
    }

    const float inv_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;
    const float gate_value = gate[head];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    device float *oh = out + (uint64_t)head * args.head_dim;
    oh[lane]       = merged.x * inv_sum * gate_scale;
    oh[lane + 32u] = merged.y * inv_sum * gate_scale;
    oh[lane + 64u] = merged.z * inv_sum * gate_scale;
    oh[lane + 96u] = merged.w * inv_sum * gate_scale;
}

// Batch independent pre-wrap verifier rows in the grid's Y dimension. This is
// deliberately the same per-head reduction as decode; only command encoding
// and dispatch are shared across rows.
kernel void kernel_laguna_attention_decode_rows_gqa_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device float       *out,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint split_simd_groups = 8u;
    constexpr uint split_threshold = 256u;
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u) {
        return;
    }
    const uint key_count = min(args.pos0 + token + 1u, args.cache_cap);
    if (key_count == 0u) return;
    const uint key_start = args.pos0 + token + 1u - key_count;
    const uint simd_group = (uint)simd_group_u;
    const bool split = key_count > split_threshold;
    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint64_t row = (uint64_t)token * args.n_head + head;
    device const float *qh = q + row * args.head_dim;

    float4 acc = float4(0.0f);
    float max_score = -INFINITY;
    float score_sum = 0.0f;
    const uint key_first = split ? simd_group : 0u;
    const uint key_stride = split ? split_simd_groups : 1u;
    for (uint i = key_first; i < key_count; i += key_stride) {
        const uint key_pos = key_start + i;
        const uint cache_row = key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)cache_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            partial += qh[d] * (float)key_cache[kv_base + d];
        }
        const float score = simd_sum(partial) * args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ?
            0.0f : exp(max_score - next_max);
        const float value_scale = exp(score - next_max);
        score_sum = score_sum * old_scale + value_scale;
        const uint d0 = lane;
        acc.x = acc.x * old_scale +
            value_scale * (float)value_cache[kv_base + d0];
        acc.y = acc.y * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 32u];
        acc.z = acc.z * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 64u];
        acc.w = acc.w * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 96u];
        max_score = next_max;
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + split_simd_groups;
    threadgroup float *partial_value = partial_sum + split_simd_groups;
    if (lane == 0u) {
        partial_max[simd_group] = max_score;
        partial_sum[simd_group] = score_sum;
    }
    const uint value_base = simd_group * args.head_dim;
    partial_value[value_base + lane] = acc.x;
    partial_value[value_base + lane + 32u] = acc.y;
    partial_value[value_base + lane + 64u] = acc.z;
    partial_value[value_base + lane + 96u] = acc.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;

    float4 merged = acc;
    float merged_sum = score_sum;
    if (split) {
        float global_max = partial_max[0];
        for (uint sg = 1u; sg < split_simd_groups; sg++) {
            global_max = max(global_max, partial_max[sg]);
        }
        merged = float4(0.0f);
        merged_sum = 0.0f;
        for (uint sg = 0u; sg < split_simd_groups; sg++) {
            const float weight = partial_sum[sg] > 0.0f ?
                exp(partial_max[sg] - global_max) : 0.0f;
            merged_sum += partial_sum[sg] * weight;
            const uint base = sg * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }
    }

    const float inv_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;
    const float gate_value = gate[row];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    device float *oh = out + row * args.head_dim;
    oh[lane]       = merged.x * inv_sum * gate_scale;
    oh[lane + 32u] = merged.y * inv_sum * gate_scale;
    oh[lane + 64u] = merged.z * inv_sum * gate_scale;
    oh[lane + 96u] = merged.w * inv_sum * gate_scale;
}

// Laguna's split-K decode attention consumes the generic FlashAttention
// partial layout, but every reduced head is immediately multiplied by its
// learned gate. Apply that epilogue before the final store so decode does not
// write and reread the complete head buffer in a second dispatch.
kernel void kernel_laguna_flash_attn_reduce_gate_f32(
        constant ds4_metal_args_flash_attn_ext_vec_reduce &args,
        device const char  *htmp,
        device       char  *dst,
        device const float *gate,
        uint tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t row = tgpig;
    device const float *stats = (device const float *)htmp +
        (uint64_t)args.nrows * DV * NWG;

    float sum = stats[row * (2 * NWG) + 2 * lane];
    float max_value = stats[row * (2 * NWG) + 2 * lane + 1];
    const float global_max = simd_max(max_value);
    const float scale = exp(max_value - global_max);
    sum = simd_sum(sum * scale);
    const float inv_sum = sum == 0.0f ? 0.0f : 1.0f / sum;

    const float gate_value = gate[row];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    const short DV4 = DV / 4;
    device const float4 *partials =
        (device const float4 *)htmp + row * DV4 * NWG;
    device volatile float4 *out =
        (device volatile float4 *)dst + row * DV4;

    for (short i = simd_group; i < DV4; i += NWG) {
        const float4 value = simd_sum(partials[i * NWG + lane] * scale);
        if (lane == 0) {
            // Keep the same F32 materialization boundary as the unfused
            // reducer followed by the gate kernel. The volatile round trip
            // prevents fast-math from reassociating the two multiplies.
            out[i] = value * inv_sum;
            out[i] = out[i] * gate_scale;
        }
    }

#undef NWG
#undef DV
}

struct ds4_metal_args_laguna_q6_matmul {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_tokens;
    uint32_t pad0;
    uint64_t row_bytes;
};

// Dense Q6_K projection used by Laguna's down projections and output head.
// The quantized arithmetic follows DwarfStar's existing Q6_K routed-down
// implementation, but addresses a single dense matrix directly.
kernel void kernel_laguna_q6_K_matmul_f32(
        constant ds4_metal_args_laguna_q6_matmul &args,
        device const char  *weight,
        device const float *x,
        device float       *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint rows_per_simd = 2u;
    constexpr uint simd_groups = 2u;
    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;
    constexpr uint qk_k = 256u;

    const uint row0 = (tgpig.x * simd_groups + simd_group) * rows_per_simd;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const int n_blocks = (int)(args.in_dim / qk_k);
    const short tid = (short)(lane / 2u);
    const short ix = (short)(lane & 1u);
    const short ip = (short)(tid / 8);
    const short il = (short)(tid % 8);
    const short l0 = (short)(4 * il);
    const short is = (short)(8 * ip + l0 / 16);
    const short y_offset = (short)(128 * ip + l0);
    const short q_offset_l = (short)(64 * ip + l0);
    const short q_offset_h = (short)(32 * ip + l0);
    device const float *input = x + (uint64_t)token * args.in_dim;
    float sums[rows_per_simd] = {0.0f, 0.0f};
    float yl[16];

    for (int ib = ix; ib < n_blocks; ib += 2) {
        device const float *y = input + (uint64_t)ib * qk_k + y_offset;
        for (short l = 0; l < 4; l++) {
            yl[4 * l + 0] = y[l + 0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }

        for (uint r = 0u; r < rows_per_simd && row0 + r < args.out_dim; r++) {
            device const block_q6_K *block =
                (device const block_q6_K *)(weight +
                    (uint64_t)(row0 + r) * args.row_bytes) + ib;
            device const uchar *q1 = block->ql + q_offset_l;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = block->qh + q_offset_h;
            device const char *sc = block->scales + is;
            float4 part = float4(0.0f);
            for (short l = 0; l < 4; l++) {
                const uint h = (uint)qh[l];
                part[0] += yl[4 * l + 0] *
                    (float)((int)((q1[l] & 0x0Fu) | ((h & kmask1) << 4u)) - 32);
                part[1] += yl[4 * l + 1] *
                    (float)((int)((q2[l] & 0x0Fu) | ((h & kmask2) << 2u)) - 32);
                part[2] += yl[4 * l + 2] *
                    (float)((int)((q1[l] >> 4u) | (h & kmask3)) - 32);
                part[3] += yl[4 * l + 3] *
                    (float)((int)((q2[l] >> 4u) | ((h & kmask4) >> 2u)) - 32);
            }
            sums[r] += (float)block->d *
                (part[0] * (float)sc[0] + part[1] * (float)sc[2] +
                 part[2] * (float)sc[4] + part[3] * (float)sc[6]);
        }
    }

    for (uint r = 0u; r < rows_per_simd && row0 + r < args.out_dim; r++) {
        const float sum = simd_sum(sums[r]);
        if (lane == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + r] = sum;
        }
    }
}
