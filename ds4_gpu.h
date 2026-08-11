#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stdint.h>

/* Strict lifecycle configuration parsers shared by production selectors and
 * focused tests.  They intentionally accept only decimal integers with
 * optional surrounding ASCII whitespace; malformed, empty, and overflowing
 * values take the documented fallback. */
static inline uint32_t ds4_gpu_q8_mv_ext_max_tokens_parse(
        const char *value) {
    const uint64_t fallback = 16u;
    const uint64_t min_value = 2u;
    const uint64_t max_value = 128u;
    if (!value) return (uint32_t)fallback;
    while (*value == ' ' || *value == '\t' || *value == '\n' ||
           *value == '\r' || *value == '\f' || *value == '\v') value++;
    if (!*value) return (uint32_t)fallback;

    uint64_t parsed = 0;
    const char *p = value;
    while (*p >= '0' && *p <= '9') {
        const uint64_t digit = (uint64_t)(*p - '0');
        if (parsed > (UINT64_MAX - digit) / 10u) {
            return (uint32_t)fallback;
        }
        parsed = parsed * 10u + digit;
        p++;
    }
    if (p == value) return (uint32_t)fallback;
    while (*p == ' ' || *p == '\t' || *p == '\n' ||
           *p == '\r' || *p == '\f' || *p == '\v') p++;
    if (*p) return (uint32_t)fallback;
    /* Preserve the stock parser's historical behavior for values below two
     * while clamping valid values above the supported ceiling. */
    if (parsed < min_value) return (uint32_t)fallback;
    if (parsed > max_value) parsed = max_value;
    return (uint32_t)parsed;
}

static inline uint32_t ds4_gpu_laguna_moe_min_tokens_parse(
        const char *value) {
    const uint64_t fallback = 96u;
    const uint64_t min_value = 32u;
    const uint64_t max_value = 4096u;
    if (!value) return (uint32_t)fallback;
    while (*value == ' ' || *value == '\t' || *value == '\n' ||
           *value == '\r' || *value == '\f' || *value == '\v') value++;
    if (!*value) return (uint32_t)fallback;

    uint64_t parsed = 0;
    const char *p = value;
    while (*p >= '0' && *p <= '9') {
        const uint64_t digit = (uint64_t)(*p - '0');
        if (parsed > (UINT64_MAX - digit) / 10u) {
            return (uint32_t)fallback;
        }
        parsed = parsed * 10u + digit;
        p++;
    }
    if (p == value) return (uint32_t)fallback;
    while (*p == ' ' || *p == '\t' || *p == '\n' ||
           *p == '\r' || *p == '\f' || *p == '\v') p++;
    if (*p) return (uint32_t)fallback;
    if (parsed < min_value) parsed = min_value;
    if (parsed > max_value) parsed = max_value;
    return (uint32_t)parsed;
}

static inline int ds4_gpu_laguna_direct_kv_prefill_env_mode(
        const char *value) {
    if (!value || value[0] == '\0' ||
        (value[0] == '0' && value[1] == '\0')) return 0;
    if (value[0] == '1' && value[1] == '\0') return 1;
    return -1;
}

static inline int ds4_gpu_laguna_output_head_norm_fuse_env_mode(
        const char *value) {
    if (!value || value[0] == '\0' ||
        (value[0] == '0' && value[1] == '\0')) return 0;
    if (value[0] == '1' && value[1] == '\0') return 1;
    return -1;
}

/* Strict parser shared by the Laguna production selector and focused tests.
 * It is backend-independent so malformed requests can be rejected before a
 * graph is allocated even in a non-Metal build. */
static inline int ds4_gpu_laguna_dense_q8_gate_up_swiglu_env_mode(
        const char *value) {
    if (!value || value[0] == '\0' ||
        (value[0] == '0' && value[1] == '\0')) return 0;
    if (value[0] == '1' && value[1] == '\0') return 1;
    return -1;
}

/* The production one-token dense decode is certified against the NR2
 * topology.  An unset/empty Q8 row selector has that default; an explicit
 * legacy/alternate value is retained as a named conflict so preflight can
 * reject it before graph mutation, and malformed values are rejected rather
 * than normalized. */
static inline int ds4_gpu_laguna_dense_q8_gate_up_swiglu_rows_env_mode(
        const char *value) {
    if (!value || value[0] == '\0') return 2;
    if (value[0] == '2' && value[1] == '\0') return 2;
    if (value[0] == '4' && value[1] == '\0') return 4;
    return -1;
}

/* Pure production-route decisions shared by the graph and focused tests.
 * The one-token fused decode is certified only for the NR2 topology; the
 * ordinary prefill and exact verifier stay on stock matmul paths.  A negative
 * result means an explicit request must fail before graph/KV mutation, zero
 * means the selector is disabled, and one means the decode certificate passed. */
static inline int ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
        int selector_mode,
        int weights_eligible,
        int ranges_valid,
        int rows_mode,
        int mid_pipeline_ready) {
    if (selector_mode < 0 || selector_mode > 1) return -1;
    if (selector_mode == 0) return 0;
    if (!weights_eligible || !ranges_valid || rows_mode != 2 ||
        !mid_pipeline_ready) {
        return -1;
    }
    return 1;
}

enum ds4_gpu_laguna_dense_q8_gate_up_swiglu_route {
    DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_STOCK = 0,
    DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_DECODE_MID = 1,
    DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_ORDINARY_PREFILL_STOCK = 2,
    DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_BATCH_FUSED = 3,
};

/* Select the route after preflight.  Exact verifier rows map to generic stock;
 * only one-token decode uses the fused route. */
static inline int ds4_gpu_laguna_dense_q8_gate_up_swiglu_route(
        int enabled, int decode, int exact_rows) {
    if (!enabled) return DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_STOCK;
    if (exact_rows) return DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_STOCK;
    if (decode) return DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_DECODE_MID;
    return DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_ORDINARY_PREFILL_STOCK;
}

/* Ordinary prefill must preserve stock mul_mv_ext arithmetic through the
 * lifecycle-snapshotted ceiling.  The batched fused route is only eligible
 * strictly above that ceiling and never for exact verifier rows. */
static inline int ds4_gpu_laguna_dense_q8_gate_up_swiglu_prefill_route(
        int enabled,
        int exact_rows,
        uint32_t n_tokens,
        uint32_t mv_ext_max_tokens) {
    if (!enabled || exact_rows || n_tokens <= mv_ext_max_tokens) {
        return DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_ORDINARY_PREFILL_STOCK;
    }
    return DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_BATCH_FUSED;
}

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

#ifndef DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_BATCH_MAX 32u
typedef struct {
    uint64_t raw_kv;
    uint64_t comp_kv;
    uint64_t topk;
    uint32_t pos;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t window;
    uint32_t ratio;
    uint32_t indexed;
} ds4_gpu_attention_decode_row;
#endif

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);

#ifdef __APPLE__
/* Process-lifecycle snapshot for the supported world-1 Q8 decode dispatch
 * selectors.  The snapshot is intentionally immutable for the lifetime of a
 * process; changing the environment after the first GPU lifecycle probe does
 * not change an in-flight or subsequently-created graph.  A fresh process is
 * the reset boundary. */
typedef struct ds4_gpu_q8_decode_config {
    int32_t q8_mv_nsg_override; /* -1 malformed, 0 default, 1..8 override */
    int32_t q8_mv_rows;         /* -1 malformed, otherwise 2 or 4 */
} ds4_gpu_q8_decode_config;

/* Returns 1 for a valid snapshot and -1 for a malformed explicit selector.
 * `out` may be NULL when only validation is required. */
int ds4_gpu_q8_decode_config_snapshot(ds4_gpu_q8_decode_config *out);
#endif

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count);
int ds4_gpu_moe_handoff_pack_tensor(
        ds4_gpu_tensor       *packed,
        const ds4_gpu_tensor *ffn_norm,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_expert);
int ds4_gpu_pack_slot_rows_f32_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *slots,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap);

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_encoder(void);
int ds4_gpu_flush_commands(void);
/* Commit the active batch without waiting; later batches remain queue-ordered. */
int ds4_gpu_submit_commands(void);
int ds4_gpu_wait_submitted_commands(void);
int ds4_gpu_discard_commands(void);
int ds4_gpu_commands_active(void);
#ifdef __APPLE__
int ds4_gpu_parallel_ffn_finish(void);
void ds4_gpu_parallel_ffn_abort(void);
#ifdef DS4_TEST_HOOKS
/* Test-only synthetic arm used to exercise terminal command-boundary
 * cleanup without requiring a model-shaped concurrent FFN dispatch. */
int ds4_gpu_parallel_ffn_test_arm_state(void);
int ds4_gpu_parallel_ffn_test_state_is_clean(void);
/* Exercise real Metal partial-init failure unwinds, successful retry, command
 * drain, workspace tracking, and mmap-backed view/cache cleanup. */
int ds4_gpu_test_lifecycle_cleanup(void);
/* Inject a one-shot synchronize result failure after the command boundary has
 * waited for all submitted work.  This exercises engine-close cleanup after a
 * reported drain failure without leaving GPU work in flight. */
void ds4_gpu_test_inject_synchronize_failure(void);
/* Inject a one-shot submitted-wait failure only after a real pending command
 * buffer has completed; completion evidence is suppressed for that wait. */
void ds4_gpu_test_inject_wait_submitted_failure(void);
/* Diagnostics used by engine-close ownership tests. */
int ds4_gpu_test_tensor_tracking_state(uint64_t *live_handles,
                                       uint64_t *live_bytes);
int ds4_gpu_test_cleanup_state_is_clean(void);
#endif
int ds4_gpu_parallel_ffn_start(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        const ds4_gpu_tensor *x,
        float                 clamp);
#ifdef DS4_TEST_HOOKS
/* Diagnostics/tests only: read the pending Metal command-buffer array.
 * Inference hot paths must not use this getter or add a counter for it. */
uint32_t ds4_gpu_diagnostic_pending_command_buffer_count(void);
#endif
#endif
int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value);
int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label);
int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label);
int ds4_gpu_end_commands(void);
/* Terminal boundary: a zero result reports command-buffer/backend failure,
 * but all submitted work, if any, has still been waited before returning. */
int ds4_gpu_synchronize(void);

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);

int ds4_gpu_pro_q4_expert_table_auto_available(void);
int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                     uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                     uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
                                     uint32_t n_total_expert);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_set_tensor_matmul_suppressed(bool suppressed);
void ds4_gpu_set_glm_model(bool enabled);
#ifdef __APPLE__
int ds4_gpu_device_is_pre_m5_apple_silicon(void);
int ds4_gpu_device_is_m5_apple_silicon(void);
#else
static inline int ds4_gpu_device_is_pre_m5_apple_silicon(void) { return 0; }
static inline int ds4_gpu_device_is_m5_apple_silicon(void) { return 0; }
#endif
void ds4_gpu_print_memory_report(const char *label);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed the retained token embedding paths.
 */

int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd);

int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd);

int ds4_gpu_embed_token_quant_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          weight_type,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd);

int ds4_gpu_embed_tokens_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd);


int ds4_gpu_dspark_markov_argmax_tensor(ds4_gpu_tensor *out_idx,
                                        const ds4_gpu_tensor *logits_row,
                                        const void *model_map,
                                        uint64_t model_size,
                                        uint64_t w1_offset,
                                        uint64_t w2_offset,
                                        uint32_t prev_token,
                                        uint32_t vocab,
                                        uint32_t rank);
int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

int ds4_gpu_indexer_top1_value_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const ds4_gpu_tensor *scores,
        uint32_t              n_comp,
        uint32_t              n_tokens,
        uint32_t              index_offset);

int ds4_gpu_matmul_q8_0_top1_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              index_offset);

int ds4_gpu_set_decode_fast_attention(int enabled);
int ds4_gpu_set_decode_score_vec4(int enabled);

/* GPU argmax over n_vocab F32 logits. Writes the winning index as int32 at
 * out_idx[0]. Tie-break: lower index wins (matches host sample_argmax). */
int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab);

#ifdef __APPLE__
/* Laguna raw-generation prototype: one threadgroup reads an F32 logit row and
 * writes one int32 winner. This is deliberately separate from the generic
 * multi-dispatch top-k path and is not used by session/public-logits or
 * verifier APIs. */
int ds4_gpu_laguna_argmax_available(void);
int ds4_gpu_laguna_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t              n_vocab);

/* Opt-in Laguna decode-only MoE residual fusion.  This Apple-only API is
 * intentionally append-only: dense layer 0 reuses the cross-backend
 * ds4_gpu_add_rms_norm_weight_rows_tensor API above, while this entry point
 * supplies the three-input add3 variant used by MoE layers. */
static inline int ds4_gpu_laguna_decode_residual_norm_env_mode(
        const char *value) {
    if (!value || value[0] == '\0' ||
        (value[0] == '0' && value[1] == '\0')) return 0;
    if (value[0] == '1' && value[1] == '\0') return 1;
    return -1;
}
int ds4_gpu_laguna_decode_residual_norm_available(void);
/* Returns whether the Metal Q8_0 gate/up+SwiGLU kernel family used by the
 * opt-in Laguna leading-dense route can be compiled on this device. */
int ds4_gpu_shared_mid_swiglu_q8_0_available(void);
int ds4_gpu_add3_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n,
        uint32_t              rows,
        float                 eps);

/* Optional Laguna S2.1 Q8_0 lm-head top-1 screen.  The plan owns a packed
 * high-nibble sidecopy and small per-row scratch; it never exposes an
 * approximate logits row.  Creation performs the sidecopy build, so callers
 * can reject missing pipelines before mutating graph/KV state. */
typedef struct ds4_gpu_laguna_q8_lmhead_screen
    ds4_gpu_laguna_q8_lmhead_screen;

ds4_gpu_laguna_q8_lmhead_screen *
ds4_gpu_laguna_q8_lmhead_screen_create(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    weight_offset,
        uint64_t    in_dim,
        uint64_t    out_dim);
void ds4_gpu_laguna_q8_lmhead_screen_destroy(
        ds4_gpu_laguna_q8_lmhead_screen *screen);
int ds4_gpu_laguna_q8_lmhead_screen_tensor(
        ds4_gpu_laguna_q8_lmhead_screen *screen,
        ds4_gpu_tensor       *out_idx,
        ds4_gpu_tensor       *out_value,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        const ds4_gpu_tensor *x);
/* The opt-in v2 arm uses a GPU-compacted NR0=2 pair list and a GPU-written
 * indirect dispatch argument.  It is selected only by the literal
 * DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2=1; malformed values are rejected.
 * DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2_FALLBACK=1 explicitly permits a v1
 * fallback when the indirect path or its pipelines are unavailable. */
int ds4_gpu_laguna_q8_lmhead_screen_v2_available(void);
int ds4_gpu_laguna_q8_lmhead_screen_v2_enabled(
        const ds4_gpu_laguna_q8_lmhead_screen *screen);
int ds4_gpu_laguna_q8_lmhead_screen_stats_v2(
        const ds4_gpu_laguna_q8_lmhead_screen *screen,
        uint32_t *compact_pair_count,
        uint32_t *exact_dispatch_groups);
/* For v2 these are the logical compacted NR0=2 pair count and the number of
 * indirect workgroups observed by the exact kernel.  The seed-pair workgroup
 * records its no-op completion; non-seed workgroups record completion only
 * after the stock exact helper returns.  Both are trace-only GPU stats and
 * must be read after the containing command has completed. */
/* Read only after the command containing screen_tensor has completed.  Stats
 * collection is captured at plan creation using ds4_gpu_env_bool (the
 * DS4_METAL_LAGUNA_Q8_LMHEAD_TRACE or focused-test flag); this function
 * returns 0 without reading GPU memory when collection was disabled, even if
 * the environment is changed later.  While an explicit command batch is
 * active, at most one stats-enabled screen_tensor call is accepted; standalone
 * completed calls remain repeatable.
 * Tests can use this to prove candidate reduction and positive dispatch.
 * candidate_row_blocks is the logical admitted-row traffic; exact_row_blocks
 * is the trace-only pair-expanded traffic actually evaluated by NR0=2,
 * including the mandatory seed pair.  sidecopy_init_ms excludes lazy
 * pipeline compilation and model wrapping; use an external wall timer for
 * total plan creation. */
int ds4_gpu_laguna_q8_lmhead_screen_stats(
        const ds4_gpu_laguna_q8_lmhead_screen *screen,
        uint32_t *candidate_rows,
        uint32_t *candidate_row_blocks,
        uint32_t *coarse_nonfinite,
        uint64_t *dispatch_count,
        uint64_t *packed_bytes,
        double   *sidecopy_init_ms,
        uint32_t *exact_row_blocks,
        int32_t  *winner_index,
        float    *winner_value);
#endif

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections and output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_dflash_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* The original Laguna Q4_K_M recipe keeps selected down projections and the
 * output head in Q6_K. */
int ds4_gpu_matmul_q6_K_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint64_t              n_tok);

/* Optional fused GPU operations.
 *
 * These are acceleration hooks, not required backend primitives.  A backend
 * that does not provide the fused kernel must still define the symbol and
 * return 0.  Callers then use the portable sequence of required primitives.
 * Backends that return nonzero from a fused half-output operation must also
 * implement the matching half-input expansion helpers below.
 */
int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Multi-row decode projections that preserve the one-row reduction order. */
int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);
int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out0_dim,
        uint64_t              out1_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

/* Implemented by Metal; the DFlash verifier is the only caller, and it needs
 * each row bit-identical to single-row decode. */
int ds4_gpu_matmul_f32_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t              n,
        float                 clamp,
        float                 weight);

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t              n);

int ds4_gpu_add3_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        uint32_t              n);

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *prequant,
        uint32_t                expert_split,
        bool                    home_rank);

int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

#ifdef __APPLE__
/* Opt-in batched prefill sibling of the fused dense Q8 gate/up+SwiGLU: one
 * tiled pass over all rows, emitting only the SwiGLU mid. */
int ds4_gpu_laguna_dense_q8_gate_up_swiglu_batch_available(void);
int ds4_gpu_laguna_dense_q8_gate_up_swiglu_batch_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
uint32_t ds4_gpu_laguna_q8_mv_ext_max_tokens(void);

/* Test-only route evidence.  Production callers do not arm these counters. */
#ifdef DS4_TEST_HOOKS
void ds4_gpu_test_laguna_route_counters_reset(void);
int ds4_gpu_test_laguna_route_counters(uint64_t *direct_kv,
                                       uint64_t *wrap_kv,
                                       uint64_t *fused_q8,
                                       uint64_t *stock_q8);
/* Completion-scoped decode route evidence: ordinary, GQA3, GQA9, staged,
 * and the default global grouped floor route.  A count is visible only after
 * its owning command buffer completed successfully. */
int ds4_gpu_test_laguna_decode_route_counters(uint64_t *ordinary,
                                              uint64_t *gqa3,
                                              uint64_t *gqa9,
                                              uint64_t *staged,
                                              uint64_t *global_grouped);
int ds4_gpu_test_laguna_q8_bco_counters(uint64_t *bco_false,
                                        uint64_t *bco_true);
/* Test-only view of the supported Q8 descriptor geometry. */
int ds4_gpu_test_q8_decode_nsg(void);
/* Test-only malformed lifecycle injection for fail-before-mutation coverage. */
void ds4_gpu_test_laguna_set_direct_kv_mode(int mode);
#endif
#endif

int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp);

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Opt-in fused decode path: fold the supported RMS normalization into the
 * F16 matvec (single token), failing closed outside its shape certificate. */
int ds4_gpu_matmul_f16_rms_norm_mv_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        float                 eps);

/* Admission-only check for the fused output head.  It validates the literal
 * selector's requested source/PSO and the exact TEW/shape certificate without
 * opening a command buffer or mutating activation/KV state. */
int ds4_gpu_matmul_f16_rms_norm_mv_preflight(
        uint32_t in_dim,
        uint32_t out_dim);

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_add_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_laguna_head_rms_norm_rope_tensor(
        ds4_gpu_tensor *x,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        n_rot,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow,
        float           eps);

#ifdef __APPLE__
/* DFlash support-only K staging.  Under the strict atlas plan this consumes
 * the independent one-family support atlas and remains separate evidence. */
int ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
        ds4_gpu_tensor *x,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        n_rot,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow,
        float           eps);
#endif

int ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
        ds4_gpu_tensor *q,
        ds4_gpu_tensor *k,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        q_weight_offset,
        uint64_t        k_weight_offset,
        uint32_t        n_tokens,
        uint32_t        n_q_head,
        uint32_t        n_k_head,
        uint32_t        head_dim,
        uint32_t        n_rot,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow,
        float           eps);

#ifdef __APPLE__
/* Strict opt-in state: -1 is a malformed value, 0 is unset/empty/0, and 1 is
 * the literal requested value.  These diagnostics do not encode graph work. */
int ds4_gpu_laguna_qk_head_norm_rope_simd32_env_mode(void);
int ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
        uint32_t n_q_head,
        uint32_t n_k_head,
        uint32_t head_dim,
        uint32_t n_rot);
/* Frozen graph-plan state: -2 is unplanned, -1 malformed/unavailable, 0 is
 * disabled, and 1 is the literal opt-in.  Production routes use these
 * cached values instead of reparsing the environment per layer. */
int ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached(void);
int ds4_gpu_laguna_qk_head_norm_rope_simd32_trace_enabled(void);
/* Returns 1 when the frozen plan was reset, 0 when active/pending work made
 * reset unsafe.  Production never calls this test-only hook. */
#ifdef DS4_TEST_HOOKS
int ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test(void);
#endif

/* Encoded-dispatch evidence for the distinct SIMD32 PSO.  This is not a
 * completion counter: callers must end/wait their command batch before using
 * it as path evidence. */
uint64_t ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count(void);
/* Completion-scoped evidence: increments only when an owning command buffer
 * containing the SIMD32 route completes successfully. */
uint64_t ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count(void);

/* Strict Laguna S 2.1 RoPE angle-atlas experiment.  The atlas contains both
 * the global 64-rotary and SWA 128-rotary families, generated once per token
 * batch on the GPU and consumed by the atlas-specific Q/K PSOs. */
int ds4_gpu_laguna_rope_atlas_env_mode(void);
/* Frozen graph-plan state: -2 is unplanned, -1 malformed, 0 disabled, 1
 * enabled.  Production callers use this instead of reparsing the environment
 * after preflight. */
int ds4_gpu_laguna_rope_atlas_plan_mode_cached(void);
/* Read-only completed-key query.  A target key with only an encoded/pending
 * generation is deliberately not reusable across graph command boundaries. */
int ds4_gpu_laguna_rope_atlas_target_reuse_ready(
    uint32_t n_tokens,
    uint32_t pos0);
int ds4_gpu_laguna_rope_atlas_trace_enabled(void);
int ds4_gpu_laguna_rope_atlas_preflight(
    uint32_t n_q_head,
    uint32_t n_k_head,
    uint32_t head_dim,
    uint32_t n_rot);
/* Test-only plan reset for subprocess-style environment matrix tests. */
/* Returns 1 when the frozen plan/cache was reset, 0 when active/pending work
 * made reset unsafe. */
#ifdef DS4_TEST_HOOKS
int ds4_gpu_laguna_rope_atlas_plan_reset_for_test(void);
#endif
/* Direct atlas generate/consumer wrappers require a successful atlas
 * preflight for the active Laguna geometry; an unplanned (-2) state is
 * intentionally treated as disabled rather than auto-selecting a plan. */
int ds4_gpu_laguna_rope_atlas_generate(uint32_t n_tokens, uint32_t pos0);
uint64_t ds4_gpu_laguna_rope_atlas_encoded_dispatch_count(void);
uint64_t ds4_gpu_laguna_rope_atlas_consumed_dispatch_count(void);
uint64_t ds4_gpu_laguna_rope_atlas_consumed_family_count(uint32_t family);
uint64_t ds4_gpu_laguna_rope_atlas_completed_generated_count(void);
uint64_t ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count(void);
uint64_t ds4_gpu_laguna_rope_atlas_completed_family_count(uint32_t family);
uint64_t ds4_gpu_laguna_rope_support_atlas_encoded_dispatch_count(void);
uint64_t ds4_gpu_laguna_rope_support_atlas_consumed_dispatch_count(void);
uint64_t ds4_gpu_laguna_rope_support_atlas_completed_generated_count(void);
uint64_t ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count(void);
#ifdef DS4_TEST_HOOKS
/* Test-only cache proof: reports whether the current target key was promoted
 * by a successfully completed generator command buffer. */
int ds4_gpu_laguna_rope_atlas_valid_completed_for_test(void);
/* Focused-test poison hook: encodes a GPU fill into the active command batch
 * without invalidating the corresponding cache key, so consumers must prove
 * they load the atlas rather than recomputing angles or using stock RoPE. */
int ds4_gpu_laguna_rope_atlas_test_poison(uint32_t support_atlas);
/* Test-only deferred graph evidence seam.  Snapshot while an owning command
 * batch is active, then complete after ds4_gpu_end_commands(); production
 * builds do not export these helpers. */
int ds4_laguna_test_rope_atlas_deferred_snapshot(
    uint64_t generated_before,
    uint64_t expected_generated,
    uint64_t consumed_before,
    uint64_t family0_before,
    uint64_t family1_before,
    uint32_t n_tokens);
int ds4_laguna_test_rope_atlas_deferred_complete(void);
#endif

/* Pure graph route seam: atlas or SIMD32 plans force paired Q/K even for
 * ordinary prefill, capture, verifier, or draft-token batches. */
int ds4_laguna_graph_prefill_qk_norm_rope_paired_route(
    int simd32_plan_mode,
    int atlas_plan_mode,
    int exact_q8_rows,
    int gpu_draft_tokens,
    int capture,
    int explicit_request);
#endif

int ds4_gpu_laguna_qkvg_f16_tensor(
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        ds4_gpu_tensor       *gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint64_t              k_weight_offset,
        uint64_t              v_weight_offset,
        uint64_t              gate_weight_offset,
        uint32_t              in_dim,
        uint32_t              q_dim,
        uint32_t              kv_dim,
        uint32_t              gate_dim,
        const ds4_gpu_tensor *x);

int ds4_gpu_laguna_attn_output_residual_f16_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual);

int ds4_gpu_laguna_store_attention_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *gate,
        uint32_t              pos,
        uint32_t              cache_cap,
        uint32_t              key_start,
        uint32_t              key_count,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale);

int ds4_gpu_laguna_attention_prefill_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        ds4_gpu_tensor       *staged_key,
        ds4_gpu_tensor       *staged_value,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *gate,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale,
        int                   split_decode_rows);

int ds4_gpu_dflash_capture_rows_tensor(
        ds4_gpu_tensor       *features,
        const ds4_gpu_tensor *src,
        uint32_t              src_row0,
        uint32_t              dst_row0,
        uint32_t              n_rows,
        uint32_t              n_embd,
        uint32_t              n_aux,
        uint32_t              aux_index);

int ds4_gpu_dflash_aux_norm_tensor(
        ds4_gpu_tensor *features,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        n_rows,
        uint32_t        n_embd,
        uint32_t        n_aux,
        float           eps);

int ds4_gpu_dflash_commit_kv_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        uint32_t              pos0,
        uint32_t              n_rows,
        uint32_t              cache_cap,
        uint32_t              n_head_kv,
        uint32_t              head_dim);

int ds4_gpu_dflash_probabilities_tensor(
        ds4_gpu_tensor       *probabilities,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *argmax,
        uint32_t              n_rows,
        uint32_t              n_vocab);

int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale);

int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

/* Opt-in fused decode router: F32 router-logit matvec + SIMD top-k select
 * in one dispatch (single token).  Bit-identical to
 * ds4_gpu_matmul_f32_decode_rows_exact_tensor followed by
 * ds4_gpu_glm_router_select_batch_tensor on the supported shape class;
 * fails closed (returns 0) outside it. */
int ds4_gpu_laguna_router_decode_fused_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        ds4_gpu_tensor       *logits,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        const ds4_gpu_tensor *x,
        uint32_t              in_dim,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        float                 expert_weight_scale);

/* Admission-only certificate for the normal one-token Laguna router.  The
 * check validates shape, source-provided PSO, TEW, and threadgroup capacity
 * without opening a command buffer or touching router/KV tensors. */
int ds4_gpu_laguna_router_decode_fused_preflight(
        uint32_t in_dim,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float    expert_weight_scale);

#ifdef DS4_TEST_HOOKS
/* Completion-scoped diagnostics for focused production-graph tests. */
void ds4_gpu_laguna_router_decode_fused_stats_reset(void);
int ds4_gpu_laguna_router_decode_fused_stats(
        uint64_t *encoded_dispatches,
        uint64_t *completed_dispatches);
#endif

/* Laguna S2.1 decode-router SIMD top-k graph preflight.  Returns 0 when the
 * public opt-in is off, 1 when the exact 256/10/2.5 path is available, and
 * -1 for a malformed/conflicting request or unavailable source/PSO.  Env-off
 * and malformed requests do not initialize Metal.  An explicit literal-1
 * request may initialize Metal and lazily compile the optional PSO, but does
 * not open command batches or mutate graph/KV state. */
int ds4_gpu_laguna_router_simd_topk_preflight(
        uint32_t n_expert,
        uint32_t n_expert_used,
        float    expert_weight_scale);

/* Laguna S2.1 SWA GQA9 decode preflight.  Returns 0 when the explicit
 * GQA9 selector is off (or is suppressed by the higher-precedence staged-SWA
 * route), 1 when the exact 512/512/72/8/128 route has an available PSO with
 * the required thread execution width/threadgroup resources, and -1 for a
 * malformed selector, incompatible shape, or unavailable/old Metal source.
 * A requested route is checked before any command batch, attention/KV store,
 * graph allocation/capture, or cache mutation. */
int ds4_gpu_laguna_swa_gqa9_preflight(
        uint32_t cache_cap,
        uint32_t key_count,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t head_dim);

/* Laguna S2.1 decode-router SIMD top-k diagnostics.  These are diagnostics
 * only: reset before a completed graph/dispatch, then use the *_after_wait
 * form (or otherwise wait for the graph) before reading.  optimized is the
 * finite fast-path row count and fallback is the in-kernel stock-bitonic row
 * count.  encoded_rows/encoded_dispatches count work recorded into command
 * encoders; they are not GPU-completion timestamps. */
void ds4_gpu_laguna_router_simd_topk_stats_reset(void);
int ds4_gpu_laguna_router_simd_topk_stats(
        uint32_t *optimized_rows,
        uint32_t *fallback_rows,
        uint64_t *encoded_rows,
        uint64_t *encoded_dispatches);
int ds4_gpu_laguna_router_simd_topk_stats_after_wait(
        uint32_t *optimized_rows,
        uint32_t *fallback_rows,
        uint64_t *encoded_rows,
        uint64_t *encoded_dispatches);

int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        bool                    force_resident);

typedef struct {
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t up_expert_bytes;
    uint64_t up_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
} ds4_gpu_laguna_moe_desc;

/* Decode-only Laguna path. Routed and shared experts use independent thread
 * groups in two common dispatches, preserving each projection's arithmetic. */
int ds4_gpu_laguna_routed_shared_moe_one_tensor(
        ds4_gpu_tensor                   *routed_out,
        ds4_gpu_tensor                   *routed_mid,
        ds4_gpu_tensor                   *shared_out,
        ds4_gpu_tensor                   *shared_mid,
        const void                       *model_map,
        uint64_t                          model_size,
        const ds4_gpu_laguna_moe_desc    *routed,
        const ds4_gpu_laguna_moe_desc    *shared,
        uint32_t                          expert_in_dim,
        uint32_t                          expert_mid_dim,
        uint32_t                          out_dim,
        const ds4_gpu_tensor             *selected,
        const ds4_gpu_tensor             *weights,
        uint32_t                          n_total_expert,
        uint32_t                          n_expert,
        const ds4_gpu_tensor             *shared_selected,
        const ds4_gpu_tensor             *shared_weight,
        const ds4_gpu_tensor             *x);

int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride,
        bool                    force_resident);

/* DFlash verifier path: batch Q2_K/Q3_K rows while retaining decode math.
 * Implemented by Metal. */
int ds4_gpu_glm_routed_moe_batch_decode_exact_q2_q3_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              up_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              up_expert_bytes,
        uint64_t              up_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              layer_index,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens,
        uint32_t              mid_token_stride);

#if defined(__APPLE__)
/* Metal verifier path: batch rows while preserving decode's Q4 kernels. */
int ds4_gpu_glm_routed_moe_batch_decode_exact_q4_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              up_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              up_expert_bytes,
        uint64_t              up_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              layer_index,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens,
        uint32_t              mid_token_stride);
#endif

/* Decode-island graph-capture descriptor. Metal/CPU builds stub capture out
 * and stay eager. Design ported from the Entrpi/ds4 batched-serving fork's
 * per-layer decode graph capture. The key identifies a captured island:
 * layer, island index, and the activation buffers whose addresses captured
 * kernels bake in. Backend implementations must mirror this layout
 * byte-for-byte; keep the fields in sync. */
typedef struct ds4_decode_graph_key {
    uint32_t il;
    uint32_t island;    /* 0: layer top to pre-rope; 1: attn-out to layer end */
    uint32_t variant;
    uint32_t _pad;
    void    *cur_hc;
    void    *after_attn_hc;
    void    *after_ffn_hc;
    void    *attn_norm;
} ds4_decode_graph_key;

int  ds4_gpu_decode_graphs_supported(void);
/* 1: replayed (island already executed; skip encoding it)
 * 0: capturing (encode the island, then call _end)
 * -1: run eagerly */
int  ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key);
/* 0: capture committed and launched; -1: capture failed (entry retired;
 * the caller must re-encode the island eagerly -- no work was executed). */
int  ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key);
void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key);
void ds4_gpu_decode_graphs_invalidate(void);

#ifdef __cplusplus
}
#endif

#endif
