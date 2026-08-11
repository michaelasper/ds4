/*
 * lgn_dflash_graph.c - private DFlash support-graph storage lifecycle.
 *
 * Graph execution and command scheduling remain in ds4.c.  This module owns
 * only the fixed support graph's tensors and byte counters.
 */

#include "lgn_dflash_graph.h"

#ifndef DS4_NO_GPU

#include <stdio.h>
#include <string.h>

static void lgn_dflash_graph_free_tensor(ds4_gpu_tensor **slot) {
    if (!slot) return;
    ds4_gpu_tensor_free(*slot);
    *slot = NULL;
}

void lgn_dflash_graph_free(lgn_dflash_graph *g) {
    if (!g) return;
#define LGN_DFLASH_GRAPH_FREE(name) lgn_dflash_graph_free_tensor(&g->name)
    LGN_DFLASH_GRAPH_FREE(features);
    LGN_DFLASH_GRAPH_FREE(encoder);
    LGN_DFLASH_GRAPH_FREE(encoder_norm);
    LGN_DFLASH_GRAPH_FREE(norm);
    LGN_DFLASH_GRAPH_FREE(tokens);
    LGN_DFLASH_GRAPH_FREE(cur);
    LGN_DFLASH_GRAPH_FREE(next);
    LGN_DFLASH_GRAPH_FREE(q);
    LGN_DFLASH_GRAPH_FREE(k);
    LGN_DFLASH_GRAPH_FREE(v);
    LGN_DFLASH_GRAPH_FREE(gate);
    LGN_DFLASH_GRAPH_FREE(heads);
    LGN_DFLASH_GRAPH_FREE(attn_out);
    LGN_DFLASH_GRAPH_FREE(after_attn);
    LGN_DFLASH_GRAPH_FREE(ffn_norm);
    LGN_DFLASH_GRAPH_FREE(ffn_gate);
    LGN_DFLASH_GRAPH_FREE(ffn_up);
    LGN_DFLASH_GRAPH_FREE(ffn_mid);
    LGN_DFLASH_GRAPH_FREE(ffn_out);
    LGN_DFLASH_GRAPH_FREE(staged_key);
    LGN_DFLASH_GRAPH_FREE(staged_value);
    LGN_DFLASH_GRAPH_FREE(output_norm);
    LGN_DFLASH_GRAPH_FREE(logits);
    LGN_DFLASH_GRAPH_FREE(argmax);
    LGN_DFLASH_GRAPH_FREE(probabilities);
#undef LGN_DFLASH_GRAPH_FREE
    for (uint32_t il = 0; il < LGN_DFLASH_N_LAYER; il++) {
        lgn_dflash_graph_free_tensor(&g->key_cache[il]);
        lgn_dflash_graph_free_tensor(&g->value_cache[il]);
    }
    memset(g, 0, sizeof(*g));
}

bool lgn_dflash_graph_alloc(lgn_dflash_graph *g) {
    if (!g) return false;
    const lgn_dflash_profile *profile = lgn_dflash_profile_get();
    const ds4_shape *shape = lgn_model_shape();
    if (!profile || !shape) return false;

    g->feature_cap = profile->cache_cap;
    g->block_cap = profile->block_size;
    g->cache_cap = profile->cache_cap;

    const uint64_t f32 = sizeof(float);
    const uint64_t embd = profile->n_embd;
    const uint64_t q_dim = (uint64_t)profile->n_head * profile->n_head_dim;
    const uint64_t kv_dim =
        (uint64_t)profile->n_head_kv * profile->n_head_dim;
    const uint64_t ff = profile->n_ff_dense;
    const uint64_t feature_rows = g->feature_cap;
    const uint64_t block_rows = g->block_cap;

#define LGN_DFLASH_GRAPH_ALLOC(name, bytes) do { \
        const uint64_t lgn_dflash_bytes_ = (uint64_t)(bytes); \
        g->name = ds4_gpu_tensor_alloc(lgn_dflash_bytes_); \
        if (!g->name) goto fail; \
        g->scratch_bytes += lgn_dflash_bytes_; \
    } while (0)
    LGN_DFLASH_GRAPH_ALLOC(features,
                           feature_rows * profile->n_aux * embd * f32);
    LGN_DFLASH_GRAPH_ALLOC(encoder, feature_rows * embd * f32);
    LGN_DFLASH_GRAPH_ALLOC(encoder_norm, feature_rows * embd * f32);
    LGN_DFLASH_GRAPH_ALLOC(norm, feature_rows * embd * f32);
    LGN_DFLASH_GRAPH_ALLOC(tokens, block_rows * sizeof(uint32_t));
    LGN_DFLASH_GRAPH_ALLOC(cur, block_rows * embd * f32);
    LGN_DFLASH_GRAPH_ALLOC(next, block_rows * embd * f32);
    LGN_DFLASH_GRAPH_ALLOC(q, block_rows * q_dim * f32);
    LGN_DFLASH_GRAPH_ALLOC(k, feature_rows * kv_dim * f32);
    LGN_DFLASH_GRAPH_ALLOC(v, feature_rows * kv_dim * f32);
    LGN_DFLASH_GRAPH_ALLOC(gate, block_rows * profile->n_head * f32);
    LGN_DFLASH_GRAPH_ALLOC(heads, block_rows * q_dim * f32);
    LGN_DFLASH_GRAPH_ALLOC(attn_out, block_rows * embd * f32);
    LGN_DFLASH_GRAPH_ALLOC(after_attn, block_rows * embd * f32);
    LGN_DFLASH_GRAPH_ALLOC(ffn_norm, block_rows * embd * f32);
    LGN_DFLASH_GRAPH_ALLOC(ffn_gate, block_rows * ff * f32);
    LGN_DFLASH_GRAPH_ALLOC(ffn_up, block_rows * ff * f32);
    LGN_DFLASH_GRAPH_ALLOC(ffn_mid, block_rows * ff * f32);
    LGN_DFLASH_GRAPH_ALLOC(ffn_out, block_rows * embd * f32);
    LGN_DFLASH_GRAPH_ALLOC(staged_key,
                           block_rows * kv_dim * sizeof(uint16_t));
    LGN_DFLASH_GRAPH_ALLOC(staged_value,
                           block_rows * kv_dim * sizeof(uint16_t));
    LGN_DFLASH_GRAPH_ALLOC(output_norm, block_rows * embd * f32);
    LGN_DFLASH_GRAPH_ALLOC(logits, block_rows * shape->n_vocab * f32);
    LGN_DFLASH_GRAPH_ALLOC(argmax, block_rows * sizeof(int32_t));
    LGN_DFLASH_GRAPH_ALLOC(probabilities, block_rows * f32);
#undef LGN_DFLASH_GRAPH_ALLOC

    const uint64_t cache_bytes =
        (uint64_t)g->cache_cap * kv_dim * sizeof(uint16_t);
    /* The owner has exactly six cache pairs; keep this bound tied to the
     * storage layout even though the immutable profile currently agrees. */
    for (uint32_t il = 0; il < LGN_DFLASH_N_LAYER; il++) {
        g->key_cache[il] = ds4_gpu_tensor_alloc(cache_bytes);
        g->value_cache[il] = ds4_gpu_tensor_alloc(cache_bytes);
        if (!g->key_cache[il] || !g->value_cache[il]) goto fail;
        g->kv_bytes += 2u * cache_bytes;
    }

    fprintf(stderr,
            "ds4: DFlash graph: block=%u, history=%u, KV %.2f MiB, "
            "scratch %.2f MiB\n",
            g->block_cap,
            g->cache_cap,
            (double)g->kv_bytes / 1048576.0,
            (double)g->scratch_bytes / 1048576.0);
    return true;

fail:
    lgn_dflash_graph_free(g);
    return false;
}

#endif /* !DS4_NO_GPU */
