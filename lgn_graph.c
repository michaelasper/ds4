/*
 * lgn_graph.c - private Laguna target graph ownership.
 *
 * This slice owns only the target graph's resource type and lifecycle.  The
 * decode/prefill scheduling and DFlash graph remain in ds4.c until a later,
 * separately reviewed extraction.
 */

#include "lgn_graph.h"

#ifndef DS4_NO_GPU

#include <stdio.h>

static void lgn_graph_free_tensor(ds4_gpu_tensor **slot) {
    if (!slot) return;
    ds4_gpu_tensor_free(*slot);
    *slot = NULL;
}

void lgn_graph_free(lgn_gpu_graph *g) {
    if (!g) return;
#define LGN_GRAPH_FREE(name) lgn_graph_free_tensor(&g->name)
    LGN_GRAPH_FREE(tokens);
    LGN_GRAPH_FREE(cur);
    LGN_GRAPH_FREE(next);
    LGN_GRAPH_FREE(attn_norm);
    LGN_GRAPH_FREE(q);
    LGN_GRAPH_FREE(k);
    LGN_GRAPH_FREE(v);
    LGN_GRAPH_FREE(gate);
    LGN_GRAPH_FREE(heads);
    LGN_GRAPH_FREE(attn_out);
    LGN_GRAPH_FREE(after_attn);
    LGN_GRAPH_FREE(ffn_norm);
    LGN_GRAPH_FREE(ffn_gate);
    LGN_GRAPH_FREE(ffn_up);
    LGN_GRAPH_FREE(ffn_mid);
    LGN_GRAPH_FREE(ffn_out);
    LGN_GRAPH_FREE(shared_out);
    LGN_GRAPH_FREE(routed_mid);
    LGN_GRAPH_FREE(router_logits);
    LGN_GRAPH_FREE(router_probs);
    LGN_GRAPH_FREE(router_selected);
    LGN_GRAPH_FREE(router_weights);
    LGN_GRAPH_FREE(shared_selected);
    LGN_GRAPH_FREE(shared_weight);
    LGN_GRAPH_FREE(staged_key);
    LGN_GRAPH_FREE(staged_value);
    LGN_GRAPH_FREE(output_norm);
    LGN_GRAPH_FREE(logits);
    LGN_GRAPH_FREE(argmax);
#undef LGN_GRAPH_FREE
    for (uint32_t il = 0; il < LGN_MODEL_MAX_LAYER; il++) {
        lgn_graph_free_tensor(&g->key_cache[il]);
        lgn_graph_free_tensor(&g->value_cache[il]);
        g->cache_cap[il] = 0;
    }
    g->ctx_size = 0;
    g->prefill_cap = 0;
    g->scratch_bytes = 0;
    g->kv_bytes = 0;
}

bool lgn_graph_alloc(lgn_gpu_graph   *g,
                     uint32_t         ctx_size,
                     const ds4_shape *shape) {
    if (!g || !shape || ctx_size == 0 ||
        (uint64_t)ctx_size > shape->context_length ||
        shape->family != DS4_MODEL_FAMILY_LAGUNA ||
        shape->n_layer != LGN_LAYER_COUNT ||
        shape->n_layer > LGN_MODEL_MAX_LAYER) {
        return false;
    }
    g->ctx_size = ctx_size;
    g->prefill_cap = ctx_size < 16384u ? ctx_size : 16384u;

    const uint64_t f32 = sizeof(float);
    const uint64_t rows = g->prefill_cap;
    const uint64_t embd = shape->n_embd;
    const uint64_t q_dim = (uint64_t)shape->n_head * shape->n_head_dim;
    const uint64_t kv_dim = (uint64_t)shape->n_head_kv * shape->n_head_dim;
    const uint64_t ffn_max = shape->n_ff_dense >
        (uint64_t)shape->n_expert_used * shape->n_ff_exp ?
        shape->n_ff_dense : (uint64_t)shape->n_expert_used * shape->n_ff_exp;

#define LGN_GRAPH_ALLOC(name, bytes) do { \
        const uint64_t lgn_graph_bytes_ = (uint64_t)(bytes); \
        g->name = ds4_gpu_tensor_alloc(lgn_graph_bytes_); \
        if (!g->name) goto fail; \
        g->scratch_bytes += lgn_graph_bytes_; \
    } while (0)
    LGN_GRAPH_ALLOC(tokens, rows * sizeof(uint32_t));
    LGN_GRAPH_ALLOC(cur, rows * embd * f32);
    LGN_GRAPH_ALLOC(next, rows * embd * f32);
    LGN_GRAPH_ALLOC(attn_norm, rows * embd * f32);
    LGN_GRAPH_ALLOC(q, rows * q_dim * f32);
    LGN_GRAPH_ALLOC(k, rows * kv_dim * f32);
    LGN_GRAPH_ALLOC(v, rows * kv_dim * f32);
    LGN_GRAPH_ALLOC(gate, rows * shape->n_head * f32);
    LGN_GRAPH_ALLOC(heads, rows * q_dim * f32);
    LGN_GRAPH_ALLOC(attn_out, rows * embd * f32);
    LGN_GRAPH_ALLOC(after_attn, rows * embd * f32);
    LGN_GRAPH_ALLOC(ffn_norm, rows * embd * f32);
    LGN_GRAPH_ALLOC(ffn_gate, rows * ffn_max * f32);
    LGN_GRAPH_ALLOC(ffn_up, rows * ffn_max * f32);
    LGN_GRAPH_ALLOC(ffn_mid, rows * ffn_max * f32);
    LGN_GRAPH_ALLOC(ffn_out, rows * embd * f32);
    LGN_GRAPH_ALLOC(shared_out, rows * embd * f32);
    LGN_GRAPH_ALLOC(routed_mid,
                    rows * shape->n_expert_used * shape->n_ff_exp * f32);
    LGN_GRAPH_ALLOC(router_logits, rows * shape->n_expert * f32);
    LGN_GRAPH_ALLOC(router_probs, rows * shape->n_expert * f32);
    LGN_GRAPH_ALLOC(router_selected,
                    rows * shape->n_expert_used * sizeof(int32_t));
    LGN_GRAPH_ALLOC(router_weights, rows * shape->n_expert_used * f32);
    LGN_GRAPH_ALLOC(shared_selected, sizeof(int32_t));
    LGN_GRAPH_ALLOC(shared_weight, sizeof(float));
    LGN_GRAPH_ALLOC(staged_key, rows * kv_dim * sizeof(uint16_t));
    LGN_GRAPH_ALLOC(staged_value, rows * kv_dim * sizeof(uint16_t));
    LGN_GRAPH_ALLOC(output_norm, embd * f32);
    LGN_GRAPH_ALLOC(logits, shape->n_vocab * f32);
#undef LGN_GRAPH_ALLOC

    const int32_t shared_id = 0;
    const float shared_weight = 1.0f;
    if (!ds4_gpu_tensor_write(g->shared_selected,
                              0,
                              &shared_id,
                              sizeof(shared_id)) ||
        !ds4_gpu_tensor_write(g->shared_weight,
                              0,
                              &shared_weight,
                              sizeof(shared_weight))) {
        goto fail;
    }

    for (uint32_t il = 0; il < shape->n_layer; il++) {
        uint32_t cap = lgn_model_layer_is_swa(il) ? shape->n_swa : ctx_size;
        if (cap > ctx_size) cap = ctx_size;
        if (cap == 0) cap = 1;
        const uint64_t bytes =
            (uint64_t)cap * shape->n_head_kv * shape->n_head_dim *
            sizeof(uint16_t);
        g->key_cache[il] = ds4_gpu_tensor_alloc(bytes);
        g->value_cache[il] = ds4_gpu_tensor_alloc(bytes);
        if (!g->key_cache[il] || !g->value_cache[il]) goto fail;
        g->cache_cap[il] = cap;
        g->kv_bytes += 2u * bytes;
    }

    fprintf(stderr,
            "ds4: Laguna GPU graph: ctx=%u, prefill=%u, KV %.2f GiB, "
            "scratch %.2f MiB\n",
            ctx_size,
            g->prefill_cap,
            (double)g->kv_bytes / 1073741824.0,
            (double)g->scratch_bytes / 1048576.0);
    return true;

fail:
    lgn_graph_free(g);
    return false;
}

#endif /* !DS4_NO_GPU */
