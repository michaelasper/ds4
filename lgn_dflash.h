#ifndef LGN_DFLASH_H
#define LGN_DFLASH_H

/* Private Laguna DFlash support-model boundary.  The engine owns opening and
 * closing ds4_model mappings plus graph/draft orchestration; this module owns
 * the fixed support profile, tensor table, and BF16 shadow-map policy. */

#include <stdbool.h>
#include <stdint.h>

#include "lgn_model.h"

enum {
    LGN_DFLASH_N_LAYER     = 6u,
    LGN_DFLASH_N_AUX       = 6u,
    LGN_DFLASH_BLOCK_SIZE  = 16u,
    LGN_DFLASH_CACHE_CAP   = 512u,
};

/* Compatibility aliases keep existing ds4 graph code source-compatible while
 * moving the constants themselves out of ds4.c. */
#define DS4_DFLASH_N_LAYER    LGN_DFLASH_N_LAYER
#define DS4_DFLASH_N_AUX      LGN_DFLASH_N_AUX
#define DS4_DFLASH_BLOCK_SIZE LGN_DFLASH_BLOCK_SIZE
#define DS4_DFLASH_CACHE_CAP  LGN_DFLASH_CACHE_CAP

typedef struct {
    uint32_t n_layer;
    uint32_t n_aux;
    uint32_t block_size;
    uint32_t cache_cap;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_value_dim;
    uint32_t n_rot;
    uint32_t n_ff_dense;
    uint64_t context_length;
    uint32_t mask_token_id;
    float rope_freq_base;
    float rms_eps;
    uint32_t target_layers[LGN_DFLASH_N_AUX];
} lgn_dflash_profile;

const lgn_dflash_profile *lgn_dflash_profile_get(void);
bool lgn_dflash_profile_matches_laguna(void);

typedef struct {
    ds4_tensor *attn_norm;
    ds4_tensor *attn_q;
    ds4_tensor *attn_k;
    ds4_tensor *attn_v;
    ds4_tensor *attn_gate;
    ds4_tensor *attn_q_norm;
    ds4_tensor *attn_k_norm;
    ds4_tensor *attn_output;
    ds4_tensor *ffn_norm;
    ds4_tensor *ffn_gate;
    ds4_tensor *ffn_up;
    ds4_tensor *ffn_down;
} lgn_dflash_layer_weights;

typedef struct {
    ds4_tensor *aux_norm;
    ds4_tensor *fc;
    ds4_tensor *encoder_output_norm;
    ds4_tensor *output_norm;
    lgn_dflash_layer_weights layer[LGN_DFLASH_N_LAYER];
    uint32_t target_layers[LGN_DFLASH_N_AUX];
    uint32_t block_size;
    uint32_t mask_token_id;
} lgn_dflash_weights;

/* Existing engine field names are private compatibility aliases, not public
 * API.  Keeping them avoids changing the DFlash graph's ABI-shaped layout. */
typedef lgn_dflash_layer_weights ds4_dflash_layer_weights;
typedef lgn_dflash_weights ds4_dflash_weights;

bool lgn_dflash_binding_eligible(const lgn_dflash_weights *weights);
void lgn_dflash_weights_bind(lgn_dflash_weights *weights,
                             const ds4_model *model);
void lgn_dflash_weights_validate_layout(const lgn_dflash_weights *weights);

/* A BF16 support model is exposed to graph code through an anonymous F16
 * shadow mapping.  Quantized support models continue to use their file map. */
void *lgn_dflash_prepare_f16_map(const ds4_model *model);
void lgn_dflash_release_f16_map(void *map, uint64_t map_size);
const void *lgn_dflash_weight_map(const ds4_model *model,
                                  const void *f16_map);
uint64_t lgn_dflash_weight_map_size(const ds4_model *model,
                                    const void *f16_map,
                                    uint64_t f16_map_size);
int lgn_dflash_weight_map_fd(const ds4_model *model,
                             const void *f16_map);

#endif /* LGN_DFLASH_H */
