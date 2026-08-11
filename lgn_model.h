#ifndef LGN_MODEL_H
#define LGN_MODEL_H

/* Private Laguna model boundary.
 *
 * This header is intentionally not included by lgn.h.  It is the temporary
 * facade shared by the GGUF loader in lgn2_engine.c and the Laguna S2.1 model module;
 * keeping these declarations private prevents tensor/GGUF details from
 * becoming part of the public Laguna API.  The lgn2_* names remain as
 * compatibility aliases for the existing engine structures while the
 * model-specific policy lives in lgn_model.c.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "lgn.h"

enum {
    /* The private weight/cache tables match Laguna S2.1's immutable profile. */
    LGN_MODEL_MAX_LAYER = 48u,
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
    LGN2_MODEL_FAMILY_DEEPSEEK4 = 0,
    LGN2_MODEL_FAMILY_LAGUNA    = 2,
} lgn2_model_family;

typedef enum {
    /* The legacy values remain reserved so Laguna's KVC identity stays the
     * explicit value 3 rather than being renumbered during the fork. */
    LGN2_VARIANT_FLASH = 0,
    LGN2_VARIANT_PRO   = 1,
    LGN2_VARIANT_LAGUNA_S21 = 3,
} lgn2_variant;

typedef struct {
    const char *name;
    lgn2_model_family family;
    lgn2_variant variant;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_value_dim;
    uint32_t n_rot;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_shared;
    uint32_t n_ff_exp;
    uint32_t n_ff_shared;
    uint32_t n_ff_dense;
    uint32_t n_swa;
    uint32_t n_leading_dense;
    uint32_t n_rot_swa;
    float rms_eps;
    float expert_weight_scale;
    float rope_freq_base;
    float rope_scale_factor;
    float rope_yarn_beta_fast;
    float rope_yarn_beta_slow;
    float rope_yarn_attn_factor;
    float rope_freq_base_swa;
    uint64_t context_length;
    uint64_t rope_orig_ctx;
} lgn2_shape;

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
} lgn2_str;

typedef struct {
    lgn2_str key;
    uint32_t type;
    uint64_t value_pos;
} lgn2_kv;

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} lgn_model_array;

typedef struct lgn2_tensor {
    lgn2_str name;
    uint32_t ndim;
    uint64_t dim[LGN_MODEL_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} lgn2_tensor;

typedef struct lgn2_model {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;

    lgn2_kv *kv;
    lgn2_tensor *tensors;
} lgn2_model;

typedef struct lgn2_layer_weights {
    lgn2_tensor *attn_norm;
    lgn2_tensor *attn_q;
    lgn2_tensor *attn_k;
    lgn2_tensor *attn_v;
    lgn2_tensor *attn_gate;
    lgn2_tensor *attn_q_norm;
    lgn2_tensor *attn_k_norm;
    lgn2_tensor *attn_output;
    lgn2_tensor *ffn_norm;
    lgn2_tensor *ffn_gate;
    lgn2_tensor *ffn_up;
    lgn2_tensor *ffn_down;
    lgn2_tensor *ffn_gate_inp;
    lgn2_tensor *ffn_exp_probs_b;
    lgn2_tensor *ffn_gate_exps;
    lgn2_tensor *ffn_up_exps;
    lgn2_tensor *ffn_down_exps;
    lgn2_tensor *ffn_gate_shexp;
    lgn2_tensor *ffn_up_shexp;
    lgn2_tensor *ffn_down_shexp;
} lgn2_layer_weights;

typedef struct lgn2_weights {
    lgn2_tensor *token_embd;
    lgn2_tensor *output_norm;
    lgn2_tensor *output;
    lgn2_layer_weights layer[LGN_MODEL_MAX_LAYER];
} lgn2_weights;

/* The engine's model shape is mutable only while validating a GGUF.  Laguna
 * owns the immutable S2.1 profile and returns it through this private accessor
 * so lgn2_engine.c can retain its existing shape macros during the transition. */
const lgn2_shape *lgn_model_shape(void);
/* The caller must have admitted and validated the Laguna target before
 * requesting this profile summary. */
void lgn_model_get_validated_summary(lgn_model_summary_fields *out);
uint32_t lgn_model_layer_head_count(uint32_t il);
bool lgn_model_layer_is_swa(uint32_t il);

/* Read-only GGUF metadata/tensor accessors for private model binders.  The
 * returned strings and tensors borrow the model mapping and remain valid until
 * the owning lgn2_model is closed. */
bool lgn_model_get_string(const lgn2_model *m,
                          const char *key,
                          lgn2_str *out);
bool lgn_model_get_u32(const lgn2_model *m,
                       const char *key,
                       uint32_t *out);
bool lgn_model_get_token_id(const lgn2_model *m,
                            const char *key,
                            int *out);
bool lgn_model_get_u64_compat(const lgn2_model *m,
                              const char *key,
                              uint64_t *out);
bool lgn_model_get_f32_compat(const lgn2_model *m,
                              const char *key,
                              float *out);
bool lgn_model_get_bool(const lgn2_model *m,
                        const char *key,
                        bool *out);
bool lgn_model_get_array(const lgn2_model *m,
                         const char *key,
                         lgn_model_array *out);
bool lgn_model_get_u32_array(const lgn2_model *m,
                             const char *key,
                             uint32_t *out,
                             uint32_t cap,
                             uint32_t *n_out);

lgn2_tensor *lgn_model_find_tensor(const lgn2_model *m, const char *name);
lgn2_tensor *lgn_model_required_tensor(const lgn2_model *m, const char *name);
lgn2_tensor *lgn_model_required_tensorf(const lgn2_model *m,
                                       const char *format,
                                       uint32_t layer);
const char *lgn_model_tensor_type_name(uint32_t type);
bool lgn_model_tensor_type_is_dense_quant(uint32_t type);
void lgn_model_validate_tensor_layout(const lgn2_tensor *tensor,
                                      uint32_t type,
                                      uint32_t ndim,
                                      uint64_t d0,
                                      uint64_t d1,
                                      uint64_t d2);

bool lgn_model_is_laguna(const lgn2_model *m, lgn2_str *arch_out);
void lgn_model_require_laguna_architecture(const lgn2_model *m);
void lgn_model_validate_config(const lgn2_model *m);

bool lgn_weights_have_output_head(const lgn2_weights *w);
bool lgn_weights_have_partial_output_head(const lgn2_weights *w);
bool lgn_weights_laguna_layer_has_required(const lgn2_layer_weights *l,
                                           uint32_t il);

void lgn_weights_validate_layout(const lgn2_weights *w,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 bool require_token_embd,
                                 bool require_output);

void lgn_weights_bind(lgn2_weights *w,
                      const lgn2_model *m);

#endif /* LGN_MODEL_H */
