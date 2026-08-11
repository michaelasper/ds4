#ifndef LGN_DFLASH_GRAPH_H
#define LGN_DFLASH_GRAPH_H

/* Private DFlash graph storage boundary.  The support graph is a flat
 * resource owner: it has no scheduler, command-buffer, model-map, or
 * speculative evidence state, so its complete storage owner can be reset on
 * free without changing the session layout. */

#include <stdint.h>

#include "lgn2_gpu.h"
#include "lgn_dflash.h"

#ifndef LGN2_NO_GPU

typedef struct lgn_dflash_graph {
    uint32_t feature_cap;
    uint32_t block_cap;
    uint32_t cache_cap;
    uint64_t scratch_bytes;
    uint64_t kv_bytes;

    lgn2_gpu_tensor *features;
    lgn2_gpu_tensor *encoder;
    lgn2_gpu_tensor *encoder_norm;
    lgn2_gpu_tensor *norm;
    lgn2_gpu_tensor *tokens;
    lgn2_gpu_tensor *cur;
    lgn2_gpu_tensor *next;
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
    lgn2_gpu_tensor *staged_key;
    lgn2_gpu_tensor *staged_value;
    lgn2_gpu_tensor *output_norm;
    lgn2_gpu_tensor *logits;
    lgn2_gpu_tensor *argmax;
    lgn2_gpu_tensor *probabilities;
    lgn2_gpu_tensor *key_cache[LGN_DFLASH_N_LAYER];
    lgn2_gpu_tensor *value_cache[LGN_DFLASH_N_LAYER];
} lgn_dflash_graph;

/* Keep the existing private engine spelling source-compatible while the
 * owning type and lifecycle entrypoints use the lgn_* namespace. */
typedef lgn_dflash_graph lgn2_dflash_gpu_graph;

/* The owner must be fresh/zeroed before allocation.  Allocate/free storage
 * only; Laguna-family and Metal admission remain in the lgn2.c wrapper, and
 * this module never opens, submits, or waits a batch. */
bool lgn_dflash_graph_alloc(lgn_dflash_graph *g);
void lgn_dflash_graph_free(lgn_dflash_graph *g);

#endif /* !LGN2_NO_GPU */

#endif /* LGN_DFLASH_GRAPH_H */
