#ifndef LGN_DFLASH_GRAPH_H
#define LGN_DFLASH_GRAPH_H

/* Private DFlash graph storage boundary.  The support graph is a flat
 * resource owner: it has no scheduler, command-buffer, model-map, or
 * speculative evidence state, so its complete storage owner can be reset on
 * free without changing the session layout. */

#include <stdint.h>

#include "ds4_gpu.h"
#include "lgn_dflash.h"

#ifndef DS4_NO_GPU

typedef struct lgn_dflash_graph {
    uint32_t feature_cap;
    uint32_t block_cap;
    uint32_t cache_cap;
    uint64_t scratch_bytes;
    uint64_t kv_bytes;

    ds4_gpu_tensor *features;
    ds4_gpu_tensor *encoder;
    ds4_gpu_tensor *encoder_norm;
    ds4_gpu_tensor *norm;
    ds4_gpu_tensor *tokens;
    ds4_gpu_tensor *cur;
    ds4_gpu_tensor *next;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *k;
    ds4_gpu_tensor *v;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *ffn_gate;
    ds4_gpu_tensor *ffn_up;
    ds4_gpu_tensor *ffn_mid;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *staged_key;
    ds4_gpu_tensor *staged_value;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *argmax;
    ds4_gpu_tensor *probabilities;
    ds4_gpu_tensor *key_cache[LGN_DFLASH_N_LAYER];
    ds4_gpu_tensor *value_cache[LGN_DFLASH_N_LAYER];
} lgn_dflash_graph;

/* Keep the existing private engine spelling source-compatible while the
 * owning type and lifecycle entrypoints use the lgn_* namespace. */
typedef lgn_dflash_graph ds4_dflash_gpu_graph;

/* The owner must be fresh/zeroed before allocation.  Allocate/free storage
 * only; Laguna-family and Metal admission remain in the ds4.c wrapper, and
 * this module never opens, submits, or waits a batch. */
bool lgn_dflash_graph_alloc(lgn_dflash_graph *g);
void lgn_dflash_graph_free(lgn_dflash_graph *g);

#endif /* !DS4_NO_GPU */

#endif /* LGN_DFLASH_GRAPH_H */
