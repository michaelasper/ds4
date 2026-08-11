struct lgn2_metal_args_dsv4_router_select_one {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
};

struct lgn2_metal_args_glm_router_select_one {
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t stats_enabled;
};

kernel void kernel_dsv4_router_weights_one(
        device const char *probs,
        device const char *selected,
        device       char *weights,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= 6) return;

    device const float *p = (device const float *)probs;
    device const int   *s = (device const int *)selected;

    float sum = 0.0f;
    for (uint i = 0; i < 6; i++) {
        sum += p[s[i]];
    }
    sum = max(sum, 6.103515625e-5f);

    device float *w = (device float *)weights;
    w[tid] = p[s[tid]] / sum * 1.5f;
}

static inline float lgn2_glm_router_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = exp(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = exp(x);
        return e / (1.0f + e);
    }
}

static inline bool lgn2_glm_router_better(
        threadgroup const float *scores,
        int32_t                  a,
        int32_t                  b) {
    const float sa = scores[(uint)a];
    const float sb = scores[(uint)b];
    return sa > sb || (sa == sb && a < b);
}

static inline bool lgn2_glm_router_better_values(
        float   sa,
        int32_t a,
        float   sb,
        int32_t b) {
    return sa > sb || (sa == sb && a < b);
}

kernel void kernel_glm_router_select_one(
        constant lgn2_metal_args_glm_router_select_one & args,
        device const float *logits,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        device float *probs,
        threadgroup float *scratch [[threadgroup(0)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + 256);
    device const float *token_logits = logits + (uint64_t)token * args.n_expert;
    device int32_t *token_selected = selected + (uint64_t)token * args.n_expert_used;
    device float *token_weights = weights + (uint64_t)token * args.n_expert_used;
    device float *token_probs = probs + (uint64_t)token * args.n_expert;

    const uint n_expert = min(args.n_expert, 256u);
    const bool active = tid < n_expert;
    const float p = active ? lgn2_glm_router_sigmoid(token_logits[tid]) : 0.0f;
    if (active) token_probs[tid] = p;
    sel_scores[tid] = active ? p + bias[tid] : -INFINITY;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid) {
                const int32_t a = idx[tid];
                const int32_t b = idx[other];
                const bool descending = (tid & k) == 0;
                const bool swap = descending
                    ? lgn2_glm_router_better(sel_scores, b, a)
                    : lgn2_glm_router_better(sel_scores, a, b);
                if (swap) {
                    idx[tid] = b;
                    idx[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint k_used = min(args.n_expert_used, n_expert);
    if (tid < k_used) {
        token_selected[tid] = idx[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < k_used) {
        float sum = 0.0f;
        for (uint i = 0; i < k_used; i++) {
            sum += token_probs[(uint)token_selected[i]];
        }
        sum = max(sum, 6.103515625e-5f);
        token_weights[tid] = token_probs[(uint)token_selected[tid]] / sum * args.expert_weight_scale;
    }
}

// Opt-in decode router selector for the production GLM-5.2 shape.  The first
// SIMD-group owns all 256 experts (8 per lane), reduces one local winner at a
// time, and therefore avoids the 256-thread bitonic sort for the usual
// n_expert=256, n_expert_used=10 case.  The complete score/probability row is
// staged before choosing the path.  Any non-finite probability or biased score
// takes the exact stock bitonic body below, without CPU readback or a partial
// selected/weight write.  Keeping the stock body in this kernel also makes the
// opt-in safe for unusual test inputs and future model variants.
// Shared body of kernel_glm_router_select_one_simd, templated on the logits
// row address space so the fused decode router below can stage the logits in
// threadgroup memory instead of a device round trip.
template<typename logits_ptr_t>
static void glm_router_select_one_simd_body(
        const uint n_expert_arg,
        const uint n_expert_used,
        const float expert_weight_scale,
        const uint stats_enabled,
        logits_ptr_t token_logits,
        device const float *bias,
        device int32_t *token_selected,
        device float *token_weights,
        device float *token_probs,
        device atomic_uint *stats,
        threadgroup float *scratch,
        uint tid) {
    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + 256);
    threadgroup uint *valid = (threadgroup uint *)(scratch + 512);

    const uint n_expert = min(n_expert_arg, 256u);
    const bool active = tid < n_expert;

    const float p = active ? lgn2_glm_router_sigmoid(token_logits[tid]) : 0.0f;
    const float score = active ? p + bias[tid] : -INFINITY;
    if (active) token_probs[tid] = p;
    sel_scores[tid] = score;
    idx[tid] = (int32_t)tid;
    valid[tid] = active && (!isfinite(p) || !isfinite(score)) ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_device);

    /* A single scalar reduction is deliberately used here: it is outside the
     * finite fast path's top-k loop and keeps all fallback decisions uniform
     * before any selected or weight output is touched. */
    if (tid == 0) {
        uint any_invalid = 0u;
        for (uint i = 0; i < 256u; i++) any_invalid |= valid[i];
        valid[0] = any_invalid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k_used = min(n_expert_used, n_expert);
    const bool fallback = valid[0] != 0u ||
        n_expert != 256u || n_expert_used != 10u;
    if (stats_enabled != 0u && tid == 0u) {
        atomic_fetch_add_explicit(stats + (fallback ? 1u : 0u),
                                  1u, memory_order_relaxed);
    }
    if (fallback) {
        for (uint k = 2; k <= 256; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                const uint other = tid ^ j;
                if (other > tid) {
                    const int32_t a = idx[tid];
                    const int32_t b = idx[other];
                    const bool descending = (tid & k) == 0;
                    const bool swap = descending
                        ? lgn2_glm_router_better(sel_scores, b, a)
                        : lgn2_glm_router_better(sel_scores, a, b);
                    if (swap) {
                        idx[tid] = b;
                        idx[other] = a;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (tid < k_used) token_selected[tid] = idx[tid];
        threadgroup_barrier(mem_flags::mem_device);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < k_used) {
            float sum = 0.0f;
            for (uint i = 0; i < k_used; i++) {
                sum += token_probs[(uint)token_selected[i]];
            }
            sum = max(sum, 6.103515625e-5f);
            token_weights[tid] =
                token_probs[(uint)token_selected[tid]] / sum *
                expert_weight_scale;
        }
        return;
    }

    /* Exactly one SIMD-group is active here.  All 256 threads participated in
     * the staging/fallback barriers above, so the rest can return without a
     * second synchronization. */
    if (tid >= 32u) return;

    float local_scores[8];
    int32_t local_ids[8];
    for (uint j = 0; j < 8u; j++) {
        const uint expert = tid * 8u + j;
        local_scores[j] = sel_scores[expert];
        local_ids[j] = (int32_t)expert;
    }

    int32_t chosen[10];
    for (uint round = 0; round < 10u; round++) {
        float winner_score = local_scores[0];
        int32_t winner_id = local_ids[0];
        for (uint j = 1; j < 8u; j++) {
            if (lgn2_glm_router_better_values(
                    local_scores[j], local_ids[j], winner_score, winner_id)) {
                winner_score = local_scores[j];
                winner_id = local_ids[j];
            }
        }

        for (ushort step = 16; step > 0; step >>= 1) {
            const float peer_score = simd_shuffle_xor(winner_score, step);
            const int32_t peer_id = simd_shuffle_xor(winner_id, step);
            if (lgn2_glm_router_better_values(
                    peer_score, peer_id, winner_score, winner_id)) {
                winner_score = peer_score;
                winner_id = peer_id;
            }
        }

        chosen[round] = winner_id;
        for (uint j = 0; j < 8u; j++) {
            if (local_ids[j] == winner_id) {
                local_scores[j] = -INFINITY;
                local_ids[j] = -1;
            }
        }
    }

    if (tid < k_used) {
        token_selected[tid] = chosen[tid];
        float sum = 0.0f;
        for (uint i = 0; i < 10u; i++) {
            sum += token_probs[(uint)chosen[i]];
        }
        sum = max(sum, 6.103515625e-5f);
        token_weights[tid] =
            token_probs[(uint)chosen[tid]] / sum * expert_weight_scale;
    }
}

kernel void kernel_glm_router_select_one_simd(
        constant lgn2_metal_args_glm_router_select_one & args,
        device const float *logits,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        device float *probs,
        device atomic_uint *stats,
        threadgroup float *scratch [[threadgroup(0)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    glm_router_select_one_simd_body(
        args.n_expert,
        args.n_expert_used,
        args.expert_weight_scale,
        args.stats_enabled,
        logits + (uint64_t)token * args.n_expert,
        bias,
        selected + (uint64_t)token * args.n_expert_used,
        weights + (uint64_t)token * args.n_expert_used,
        probs + (uint64_t)token * args.n_expert,
        stats, scratch, tid);
}

struct lgn2_metal_args_laguna_router_fused {
    uint32_t in_dim;
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t stats_enabled;
};

// Opt-in fused Laguna decode router for one token: computes the 256 router
// logits in threadgroup memory and runs the SIMD top-k selection in the same
// dispatch, removing the logits round trip.  The matvec stage replicates
// kernel_mul_mv_f32_f32_4's NSG=8 reduction tree row by row (per-lane
// partial, simd_sum, 8-partial cross-group simd_sum over zero-padded lanes),
// so the staged logits are bit-identical to the two-dispatch path and the
// shared selection body sees exactly the same inputs.  The device logits row
// is still written for downstream debug dumps.
kernel void kernel_laguna_router_decode_fused(
        constant lgn2_metal_args_laguna_router_fused & args,
        device const float *weight,
        device const float *x,
        device const float *bias,
        device float *logits,
        device int32_t *selected,
        device float *weights,
        device float *probs,
        device atomic_uint *stats,
        threadgroup float *shmem [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = NF / 4;
    constexpr short NSG = 8;

    threadgroup float *sg_part = shmem;                  // NSG * n_expert
    threadgroup float *tg_logits = sg_part + NSG * 256;  // n_expert
    threadgroup float *select_scratch = tg_logits + 256; // select scratch

    const int n_blocks = (int)args.in_dim / NB;
    const short ix = lane / (NW / NF);
    const short il = lane % (NW / NF);
    const int block0 = simd_group * NF + ix;
    device const float4 *x4 = (device const float4 *)x;

    for (uint r = 0; r < args.n_expert; r++) {
        device const float4 *w4 =
            (device const float4 *)(weight + (uint64_t)r * args.in_dim);
        device const float4 *yb = x4 + (block0 * NB + il * NF) / 4;
        float sumf = 0.0f;
        for (int block = block0; block < n_blocks; block += NSG * NF) {
            device const float4 *wb = w4 + (block * NB + il * NF) / 4;
            float part = 0.0f;
            FOR_UNROLL (short i = 0; i < NF4; i++) {
                part += dot(wb[i], yb[i]);
            }
            sumf += part;
            yb += NSG * NF * NW / 4;
        }
        const float tot = simd_sum(sumf);
        if (lane == 0) sg_part[simd_group * args.n_expert + r] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rows_per_group = args.n_expert / NSG;
    for (uint rr = 0; rr < rows_per_group; rr++) {
        const uint r = simd_group * rows_per_group + rr;
        const float v =
            lane < NSG ? sg_part[(uint)lane * args.n_expert + r] : 0.0f;
        const float tot = simd_sum(v);
        if (lane == 0) {
            tg_logits[r] = tot;
            logits[r] = tot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    glm_router_select_one_simd_body(
        args.n_expert,
        args.n_expert_used,
        args.expert_weight_scale,
        args.stats_enabled,
        (threadgroup const float *)tg_logits,
        bias, selected, weights, probs, stats, select_scratch,
        simd_group * NW + lane);
}

// Batched Flash-router weight finalization after selection is already known.
// Six active lanes deliberately match kernel_sum_rows_f32_f32's reduction
// topology. The denominator and divided weights cross threadgroup storage
// boundaries so division cannot be reassociated with the final scale.
kernel void kernel_dsv4_router_weights_batch(
        constant float &scale,
        device const float *probs,
        device const int32_t *selected,
        device float *weights,
        threadgroup volatile float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    if (tid >= 6) return;

    threadgroup volatile float *sum_scratch = scratch;
    threadgroup volatile float *denom_scratch = scratch + 32;
    threadgroup volatile float *div_scratch = scratch + 33;
    const uint out_index = row * 6u + (uint)tid;
    const int32_t expert = selected[out_index];
    const float p = probs[row * 256u + (uint)expert];

    // Keep this sequence identical to kernel_sum_rows_f32_f32 for width 6.
    if (sgitg == 0) {
        sum_scratch[tiisg] = 0.0f;
    }
    float sumf = 0.0f;
    sumf += p;
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_scratch[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = sum_scratch[tiisg];
    sumf = simd_sum(sumf);

    if (tid == 0) {
        denom_scratch[0] = clamp(sumf, 6.103515625e-5f, INFINITY);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    div_scratch[tid] = p / denom_scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    weights[out_index] = div_scratch[tid] * scale;
}

// Laguna router normalization also uses the generic row-reduction ABI. Keep
// the implementation in this retained source so the legacy standalone
// reduction surface can stay removed without changing the retained Laguna
// reduction or the order used by the host encoder.
#define FC_LAGUNA_SUM_ROWS 1400
#define OP_LAGUNA_SUM_ROWS_NUM_SUM_ROWS 10
#define OP_LAGUNA_SUM_ROWS_NUM_MEAN     11

struct lgn2_metal_args_laguna_sum_rows {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

static inline float laguna_sum_rows_sum(float x) {
    return x;
}

constant short FC_laguna_sum_rows_op [[function_constant(FC_LAGUNA_SUM_ROWS + 0)]];

template <typename T0, typename T>
kernel void kernel_laguna_sum_rows_impl(
        constant lgn2_metal_args_laguna_sum_rows & args,
        device const char * src0,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
#define FC_OP FC_laguna_sum_rows_op
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;
    threadgroup T0 *shmem_t = (threadgroup T0 *)shmem;
    if (sgitg == 0) {
        shmem_t[tiisg] = 0.0f;
    }
    device const T0 *src_row = (device const T0 *)
        (src0 + i1 * args.nb01 + i2 * args.nb02 + i3 * args.nb03);
    device T *dst_row = (device T *)
        (dst + i1 * args.nb1 + i2 * args.nb2 + i3 * args.nb3);
    T0 sumf = T0(0.0f);
    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        sumf += src_row[i0];
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem_t[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = shmem_t[tiisg];
    sumf = simd_sum(sumf);
    if (tpitg.x == 0) {
        if (FC_OP == OP_LAGUNA_SUM_ROWS_NUM_MEAN) {
            dst_row[0] = laguna_sum_rows_sum(sumf) / args.ne00;
        } else {
            dst_row[0] = laguna_sum_rows_sum(sumf);
        }
    }
#undef FC_OP
}

typedef decltype(kernel_laguna_sum_rows_impl<float, float>)
    kernel_laguna_sum_rows_t;

template [[host_name("kernel_sum_rows_f32_f32")]] kernel
    kernel_laguna_sum_rows_t kernel_laguna_sum_rows_impl<float, float>;

// Decode router selection for one token after the existing
// sqrt(softplus(logit)) probability kernel has run. Bias affects only top-k
// selection. Route-weight normalization deliberately stays in the old one-token
// kernel: even tiny denominator-order changes here are amplified by 43 MoE
// layers, so this kernel only replaces the selection work.
kernel void kernel_dsv4_router_finalize_one(
        constant lgn2_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256) return;

    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + 256);
    const float p = probs[tid];
    sel_scores[tid] = args.has_bias ? p + bias[tid] : p;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (args.hash_mode) {
        if (tid == 0) {
            const uint token = args.use_token_buffer ? (uint)tokens[0] : args.token;
            const uint row = min(token, args.hash_rows - 1u);
            device const int32_t *src = hash + row * 6u;
            for (uint i = 0; i < 6; i++) {
                selected[i] = src[i];
            }
        }
    } else {
        for (uint k = 2; k <= 256; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                const uint other = tid ^ j;
                if (other > tid) {
                    if ((tid & k) == 0) {
                        if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    } else {
                        if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (tid < 6) {
            selected[tid] = idx[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// M3 decode specialization for the non-hash one-token router. Scores and ids
// stay in registers. Intra-SIMD bitonic stages use shuffle-xor; the six stages
// that cross 32-lane SIMD groups exchange through alternating threadgroup
// banks. The next bank's publish barrier proves every prior-bank read finished;
// by the time a bank is reused two cross stages later, no reader can remain.
kernel void kernel_dsv4_router_finalize_one_simd(
        constant lgn2_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256 || args.hash_mode) return;

    (void)hash;
    (void)tokens;
    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = probs[tid];
    float score = args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;

            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }

    if (tid < 6) {
        selected[tid] = idx;
    }
}

// M3 decode specialization that extends the register/TG SIMD selection above
// through the existing six-value serial weight normalization. The selected ids
// cross the same device-memory boundary as the standalone weight kernel;
// volatile TG stores pin its left-fold and scaled-reciprocal rounding points.
kernel void kernel_dsv4_router_finalize_weights_one_simd(
        constant lgn2_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        device float *weights,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256 || args.hash_mode) return;

    (void)hash;
    (void)tokens;
    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = probs[tid];
    float score = args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;

            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }

    if (tid < 6) {
        selected[tid] = idx;
    }
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup volatile float *norm_scratch =
        (threadgroup volatile float *)scratch;
    if (tid == 0) {
        device const int32_t *s = selected;
        norm_scratch[0] = 0.0f;
        for (uint i = 0; i < 6; i++) {
            norm_scratch[0] = norm_scratch[0] + probs[s[i]];
        }
        norm_scratch[0] = max(norm_scratch[0], 6.103515625e-5f);
        norm_scratch[1] = 1.5f / norm_scratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6) {
        device const int32_t *s = selected;
        weights[tid] = probs[s[tid]] * norm_scratch[1];
    }
}

// M3 decode specialization that materializes the probability
// transform in device memory before running the exact SIMD selection and
// weight normalization above. The volatile reload after the device barrier
// pins the same float store/load boundary as the standalone transform dispatch.
kernel void kernel_dsv4_router_transform_finalize_weights_one_simd(
        constant lgn2_metal_args_dsv4_router_select_one & args,
        device const float *logits,
        device float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        device float *weights,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256 || args.hash_mode) return;

    if (tid < 64) {
        device const float4 *s = (device const float4 *)logits;
        device float4 *d = (device float4 *)probs;
        const float4 x = s[tid];
        const float4 sp = select(log(1.0f + exp(x)), x, x > 20.0f);
        d[tid] = sqrt(sp);
    }
    threadgroup_barrier(mem_flags::mem_device);
    device volatile const float *reloaded_probs =
        (device volatile const float *)probs;

    (void)hash;
    (void)tokens;
    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = reloaded_probs[tid];
    float score = args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;

            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }

    if (tid < 6) {
        selected[tid] = idx;
    }
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup volatile float *norm_scratch =
        (threadgroup volatile float *)scratch;
    if (tid == 0) {
        device const int32_t *s = selected;
        norm_scratch[0] = 0.0f;
        for (uint i = 0; i < 6; i++) {
            norm_scratch[0] =
                norm_scratch[0] + reloaded_probs[s[i]];
        }
        norm_scratch[0] = max(norm_scratch[0], 6.103515625e-5f);
        norm_scratch[1] = 1.5f / norm_scratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6) {
        device const int32_t *s = selected;
        weights[tid] = reloaded_probs[s[tid]] * norm_scratch[1];
    }
}
