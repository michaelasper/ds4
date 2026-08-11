#ifndef LGN_GRAPH_H
#define LGN_GRAPH_H

/* Private Laguna target-graph ownership boundary.  This header intentionally
 * exposes the complete resource owner because lgn2_session embeds the graph
 * inline; changing that to a pointer would change the established session
 * layout and lifetime contract.  Decode/prefill scheduling and extension
 * state remain in lgn2_engine.c.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "lgn2_gpu.h"
#include "lgn_model.h"

#ifndef LGN2_NO_GPU

typedef struct {
    uint64_t decode_mid_fused;
    uint64_t ordinary_prefill_stock;
} lgn_dense_q8_gate_up_swiglu_counters;

typedef struct lgn_gpu_graph {
    uint32_t ctx_size;
    uint32_t prefill_cap;
    uint64_t scratch_bytes;
    uint64_t kv_bytes;

    lgn2_gpu_tensor *tokens;
    lgn2_gpu_tensor *cur;
    lgn2_gpu_tensor *next;
    lgn2_gpu_tensor *attn_norm;
    lgn2_gpu_tensor *q;
    lgn2_gpu_tensor *k;
    lgn2_gpu_tensor *v;
    lgn2_gpu_tensor *gate;
    lgn2_gpu_tensor *heads;
    lgn2_gpu_tensor *attn_out;
    lgn2_gpu_tensor *after_attn;
    lgn2_gpu_tensor *ffn_norm;
    lgn2_gpu_tensor *ffn_gate;
    lgn2_gpu_tensor *ffn_up;
    lgn2_gpu_tensor *ffn_mid;
    lgn2_gpu_tensor *ffn_out;
    lgn2_gpu_tensor *shared_out;
    lgn2_gpu_tensor *routed_mid;
    lgn2_gpu_tensor *router_logits;
    lgn2_gpu_tensor *router_probs;
    lgn2_gpu_tensor *router_selected;
    lgn2_gpu_tensor *router_weights;
    lgn2_gpu_tensor *shared_selected;
    lgn2_gpu_tensor *shared_weight;
    lgn2_gpu_tensor *staged_key;
    lgn2_gpu_tensor *staged_value;
    lgn2_gpu_tensor *output_norm;
    lgn2_gpu_tensor *logits;
    lgn2_gpu_tensor *argmax;
    bool gpu_argmax_enabled;
    int32_t gpu_argmax_result;
    bool dense_q8_fusion_decision_valid;
    bool dense_q8_fusion_enabled;
#ifdef __APPLE__
    /* Ordinary-prefill batch selector is frozen on the first graph forward;
     * later environment changes cannot switch arithmetic halfway through a
     * graph. */
    bool dense_q8_batch_decision_valid;
    bool dense_q8_batch_enabled;
    lgn2_gpu_laguna_q8_lmhead_screen *lmhead_screen;
    bool q8_lmhead_screen_dispatched;
#endif
    lgn2_gpu_tensor *spec_output_norm;
    lgn2_gpu_tensor *spec_logits;
    lgn2_gpu_tensor *spec_argmax;
    lgn2_gpu_tensor *spec_key_backup[LGN_MODEL_MAX_LAYER];
    lgn2_gpu_tensor *spec_value_backup[LGN_MODEL_MAX_LAYER];
    lgn2_gpu_tensor *key_cache[LGN_MODEL_MAX_LAYER];
    lgn2_gpu_tensor *value_cache[LGN_MODEL_MAX_LAYER];
    uint32_t cache_cap[LGN_MODEL_MAX_LAYER];
    /* Evidence is attached to this graph/command-buffer owner.  It is
     * promoted to the process report only after the owning work is waited. */
    lgn_dense_q8_gate_up_swiglu_counters dense_q8_pending;
#ifdef __APPLE__
    /* A deferred DFlash verifier keeps the target command buffer open while
     * support features are appended.  Retain encoded and completion-counter
     * snapshots until that command is actually completed for diagnostics. */
    uint64_t qk_simd32_target_encoded_before;
    uint64_t qk_simd32_target_completed_before;
    uint32_t qk_simd32_target_n_tokens;
    bool qk_simd32_target_capture;
    bool qk_simd32_target_verifier;
    bool qk_simd32_target_evidence_pending;
    uint64_t rope_atlas_target_generated_before;
    uint64_t rope_atlas_target_consumed_before;
    uint64_t rope_atlas_target_family0_before;
    uint64_t rope_atlas_target_family1_before;
    uint64_t rope_atlas_target_generation_expected;
    uint32_t rope_atlas_target_n_tokens;
    bool rope_atlas_target_capture;
    bool rope_atlas_target_verifier;
    bool rope_atlas_target_evidence_pending;
#endif
} lgn_gpu_graph;

/* Source-compatible aliases are intentionally private to this transition;
 * the owning type and all new storage entrypoints use the lgn_* names. */
typedef lgn_gpu_graph lgn2_laguna_gpu_graph;
typedef lgn_dense_q8_gate_up_swiglu_counters
    laguna_dense_q8_gate_up_swiglu_counters;

/* Allocate/free only the target graph's base storage: transient scratch,
 * persistent target KV caches, storage counters, and optional argmax output.
 * The caller owns extension fields (speculative scratch, diagnostics, Metal
 * screen state, selector/evidence state), performs family/backend preflight,
 * and zeroes the complete owner after lgn_graph_free returns. */
bool lgn_graph_alloc(lgn_gpu_graph   *g,
                     uint32_t         ctx_size,
                     const lgn2_shape *shape);
void lgn_graph_free(lgn_gpu_graph *g);

#endif /* !LGN2_NO_GPU */

#endif /* LGN_GRAPH_H */
