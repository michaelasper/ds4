#ifndef LGN_MODEL_H
#define LGN_MODEL_H

/* Private Laguna model boundary.
 *
 * This header is intentionally not included by lgn.h.  It is the temporary
 * facade shared by the GGUF loader in ds4.c and the Laguna S2.1 model module;
 * keeping these declarations private prevents tensor/GGUF details from
 * becoming part of the public Laguna API.  The ds4_* names remain as
 * compatibility aliases for the existing engine structures while the
 * model-specific policy lives in lgn_model.c.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "lgn.h"

enum {
    /* Retain the old ds4 weight-table capacity in this private facade while
     * Laguna executes only its fixed 48-layer profile. */
    LGN_MODEL_MAX_LAYER = 79u,
    LGN_MODEL_MAX_DIMS  = 8u,
};

/* GGUF scalar and tensor codes are shared by the private model modules.  Keep
 * these in the facade rather than making each model-specific binder carry a
 * second copy of the wire-format numbers. */
enum {
    LGN_GGUF_VALUE_UINT8   = 0,
    LGN_GGUF_VALUE_INT8    = 1,
    LGN_GGUF_VALUE_UINT16  = 2,
    LGN_GGUF_VALUE_INT16   = 3,
    LGN_GGUF_VALUE_UINT32  = 4,
    LGN_GGUF_VALUE_INT32   = 5,
    LGN_GGUF_VALUE_FLOAT32 = 6,
    LGN_GGUF_VALUE_BOOL    = 7,
    LGN_GGUF_VALUE_STRING  = 8,
    LGN_GGUF_VALUE_ARRAY   = 9,
    LGN_GGUF_VALUE_UINT64  = 10,
    LGN_GGUF_VALUE_INT64   = 11,
    LGN_GGUF_VALUE_FLOAT64 = 12,

    LGN_TENSOR_F32     = 0,
    LGN_TENSOR_F16     = 1,
    LGN_TENSOR_Q4_0    = 2,
    LGN_TENSOR_Q4_1    = 3,
    LGN_TENSOR_Q5_0    = 6,
    LGN_TENSOR_Q5_1    = 7,
    LGN_TENSOR_Q8_0    = 8,
    LGN_TENSOR_Q8_1    = 9,
    LGN_TENSOR_Q2_K    = 10,
    LGN_TENSOR_Q3_K    = 11,
    LGN_TENSOR_Q4_K    = 12,
    LGN_TENSOR_Q5_K    = 13,
    LGN_TENSOR_Q6_K    = 14,
    LGN_TENSOR_Q8_K    = 15,
    LGN_TENSOR_IQ2_XXS = 16,
    LGN_TENSOR_IQ2_XS  = 17,
    LGN_TENSOR_IQ3_XXS = 18,
    LGN_TENSOR_IQ1_S   = 19,
    LGN_TENSOR_IQ4_NL  = 20,
    LGN_TENSOR_IQ3_S   = 21,
    LGN_TENSOR_IQ2_S   = 22,
    LGN_TENSOR_IQ4_XS  = 23,
    LGN_TENSOR_I8      = 24,
    LGN_TENSOR_I16     = 25,
    LGN_TENSOR_I32     = 26,
    LGN_TENSOR_I64     = 27,
    LGN_TENSOR_F64     = 28,
    LGN_TENSOR_IQ1_M   = 29,
    LGN_TENSOR_BF16    = 30,
    LGN_TENSOR_MXFP4   = 39,
};

typedef enum {
    /* Keep the historical numeric space stable for private callers that
     * inspect rejected model identities.  Admission accepts Laguna only. */
    DS4_MODEL_FAMILY_DEEPSEEK4 = 0,
    DS4_MODEL_FAMILY_LAGUNA    = 2,
} ds4_model_family;

typedef enum {
    /* The legacy values remain reserved so Laguna's KVC identity stays the
     * explicit value 3 rather than being renumbered during the fork. */
    DS4_VARIANT_FLASH = 0,
    DS4_VARIANT_PRO   = 1,
    DS4_VARIANT_LAGUNA_S21 = 3,
} ds4_variant;

typedef struct {
    const char *name;
    ds4_model_family family;
    ds4_variant variant;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_value_dim;
    uint32_t n_rot;
    uint32_t n_out_group;
    uint32_t n_lora_q;
    uint32_t n_lora_o;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_shared;
    uint32_t n_ff_exp;
    uint32_t n_ff_shared;
    uint32_t n_ff_dense;
    uint32_t n_hash_layer;
    uint32_t n_swa;
    uint32_t n_indexer_head;
    uint32_t n_indexer_head_dim;
    uint32_t n_indexer_top_k;
    uint32_t n_hc;
    uint32_t n_hc_sinkhorn_iter;
    uint32_t n_nextn_predict;
    uint32_t n_leading_dense;
    uint32_t n_kv_lora;
    uint32_t n_key_mla;
    uint32_t n_value_mla;
    uint32_t n_rot_swa;
    float rms_eps;
    float hc_eps;
    float expert_weight_scale;
    float swiglu_clamp_exp;
    float rope_freq_base;
    float rope_scale_factor;
    float rope_yarn_beta_fast;
    float rope_yarn_beta_slow;
    float rope_yarn_attn_factor;
    float rope_freq_base_swa;
    float compress_rope_freq_base;
    uint64_t context_length;
    uint64_t rope_orig_ctx;
} ds4_shape;

/* Stable, validated fields used by the user-facing model summary.  Laguna's
 * attention head count is layer-varying in GGUF, so n_head is the profile's
 * maximum/SWA head count while the validator checks every layer's array. */
typedef struct {
    uint32_t n_layer;
    uint64_t context_length;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_swa;
    uint32_t n_expert;
    uint32_t n_expert_used;
} lgn_model_summary_fields;

typedef struct {
    const char *ptr;
    uint64_t len;
} ds4_str;

typedef struct {
    ds4_str key;
    uint32_t type;
    uint64_t value_pos;
} ds4_kv;

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} lgn_model_array;

typedef struct ds4_tensor {
    ds4_str name;
    uint32_t ndim;
    uint64_t dim[LGN_MODEL_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} ds4_tensor;

typedef struct ds4_model {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;

    ds4_kv *kv;
    ds4_tensor *tensors;
} ds4_model;

typedef struct ds4_layer_weights {
    ds4_tensor *hc_attn_fn;
    ds4_tensor *hc_attn_scale;
    ds4_tensor *hc_attn_base;
    ds4_tensor *attn_norm;
    ds4_tensor *attn_q;
    ds4_tensor *attn_k;
    ds4_tensor *attn_v;
    ds4_tensor *attn_gate;
    ds4_tensor *attn_q_norm;
    ds4_tensor *attn_k_norm;
    ds4_tensor *attn_q_a;
    ds4_tensor *attn_q_a_norm;
    ds4_tensor *attn_q_b;
    ds4_tensor *attn_kv;
    ds4_tensor *attn_kv_a_mqa;
    ds4_tensor *attn_kv_a_norm;
    ds4_tensor *attn_k_b;
    ds4_tensor *attn_v_b;
    ds4_tensor *attn_sinks;
    ds4_tensor *attn_output;
    ds4_tensor *attn_output_a;
    ds4_tensor *attn_output_b;
    ds4_tensor *attn_compressor_ape;
    ds4_tensor *attn_compressor_kv;
    ds4_tensor *attn_compressor_gate;
    ds4_tensor *attn_compressor_norm;
    ds4_tensor *indexer_attn_q_b;
    ds4_tensor *indexer_attn_k;
    ds4_tensor *indexer_k_norm;
    ds4_tensor *indexer_k_norm_b;
    ds4_tensor *indexer_proj;
    ds4_tensor *indexer_compressor_ape;
    ds4_tensor *indexer_compressor_kv;
    ds4_tensor *indexer_compressor_gate;
    ds4_tensor *indexer_compressor_norm;
    ds4_tensor *hc_ffn_fn;
    ds4_tensor *hc_ffn_scale;
    ds4_tensor *hc_ffn_base;
    ds4_tensor *ffn_norm;
    ds4_tensor *ffn_gate_tid2eid;
    ds4_tensor *ffn_gate;
    ds4_tensor *ffn_up;
    ds4_tensor *ffn_down;
    ds4_tensor *ffn_gate_inp;
    ds4_tensor *ffn_exp_probs_b;
    ds4_tensor *ffn_gate_exps;
    ds4_tensor *ffn_up_exps;
    ds4_tensor *ffn_down_exps;
    ds4_tensor *ffn_gate_shexp;
    ds4_tensor *ffn_up_shexp;
    ds4_tensor *ffn_down_shexp;
    ds4_tensor *nextn_eh_proj;
    ds4_tensor *nextn_enorm;
    ds4_tensor *nextn_hnorm;
    ds4_tensor *nextn_shared_head_norm;
} ds4_layer_weights;

typedef struct ds4_weights {
    ds4_tensor *token_embd;
    ds4_tensor *output_hc_base;
    ds4_tensor *output_hc_fn;
    ds4_tensor *output_hc_scale;
    ds4_tensor *output_norm;
    ds4_tensor *output;
    ds4_layer_weights layer[LGN_MODEL_MAX_LAYER];
} ds4_weights;

/* The engine's model shape is mutable only while validating a GGUF.  Laguna
 * owns the immutable S2.1 profile and returns it through this private accessor
 * so ds4.c can retain its existing shape macros during the transition. */
const ds4_shape *lgn_model_shape(void);
/* The caller must have admitted and validated the Laguna target before
 * requesting this profile summary. */
void lgn_model_get_validated_summary(lgn_model_summary_fields *out);
uint32_t lgn_model_layer_head_count(uint32_t il);
bool lgn_model_layer_is_swa(uint32_t il);

/* Read-only GGUF metadata/tensor accessors for private model binders.  The
 * returned strings and tensors borrow the model mapping and remain valid until
 * the owning ds4_model is closed. */
bool lgn_model_get_string(const ds4_model *m,
                          const char *key,
                          ds4_str *out);
bool lgn_model_get_u32(const ds4_model *m,
                       const char *key,
                       uint32_t *out);
bool lgn_model_get_token_id(const ds4_model *m,
                            const char *key,
                            int *out);
bool lgn_model_get_u64_compat(const ds4_model *m,
                              const char *key,
                              uint64_t *out);
bool lgn_model_get_f32_compat(const ds4_model *m,
                              const char *key,
                              float *out);
bool lgn_model_get_bool(const ds4_model *m,
                        const char *key,
                        bool *out);
bool lgn_model_get_array(const ds4_model *m,
                         const char *key,
                         lgn_model_array *out);
bool lgn_model_get_u32_array(const ds4_model *m,
                             const char *key,
                             uint32_t *out,
                             uint32_t cap,
                             uint32_t *n_out);

ds4_tensor *lgn_model_find_tensor(const ds4_model *m, const char *name);
ds4_tensor *lgn_model_required_tensor(const ds4_model *m, const char *name);
ds4_tensor *lgn_model_required_tensorf(const ds4_model *m,
                                       const char *format,
                                       uint32_t layer);
const char *lgn_model_tensor_type_name(uint32_t type);
bool lgn_model_tensor_type_is_dense_quant(uint32_t type);
void lgn_model_validate_tensor_layout(const ds4_tensor *tensor,
                                      uint32_t type,
                                      uint32_t ndim,
                                      uint64_t d0,
                                      uint64_t d1,
                                      uint64_t d2);

bool lgn_model_is_laguna(const ds4_model *m, ds4_str *arch_out);
void lgn_model_require_laguna_architecture(const ds4_model *m);
void lgn_model_validate_config(const ds4_model *m);

bool lgn_weights_have_output_head(const ds4_weights *w);
bool lgn_weights_have_partial_output_head(const ds4_weights *w);
bool lgn_weights_laguna_layer_has_required(const ds4_layer_weights *l,
                                           uint32_t il);

void lgn_weights_validate_layout(const ds4_weights *w,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 bool require_token_embd,
                                 bool require_output);

void lgn_weights_bind(ds4_weights *w,
                      const ds4_model *m);

#endif /* LGN_MODEL_H */
