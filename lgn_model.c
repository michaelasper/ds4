/*
 * lgn_model.c - private Laguna S 2.1 model admission and weight binding.
 *
 * The engine still owns GGUF mapping and execution.  This module owns the
 * immutable Laguna profile and the model-specific metadata, topology, tensor
 * layout, output-head, and layer-binding rules.  Its header is private so the
 * GGUF/GPU facade does not become part of lgn.h.
 */

#include "lgn_model.h"

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const ds4_shape LGN_SHAPE_LAGUNA_S21 = {
    .name = "Laguna S 2.1",
    .family = DS4_MODEL_FAMILY_LAGUNA,
    .variant = DS4_VARIANT_LAGUNA_S21,
    .n_layer = 48,
    .n_embd = 3072,
    .n_vocab = 100352,
    .n_head = 72,
    .n_head_kv = 8,
    .n_head_dim = 128,
    .n_value_dim = 128,
    .n_rot = 64,
    .n_expert = 256,
    .n_expert_used = 10,
    .n_expert_shared = 1,
    .n_ff_exp = 1024,
    .n_ff_shared = 1024,
    .n_ff_dense = 12288,
    .n_swa = 512,
    .n_leading_dense = 1,
    .n_rot_swa = 128,
    .rms_eps = 1.0e-6f,
    .expert_weight_scale = 2.5f,
    .rope_freq_base = 500000.0f,
    .rope_scale_factor = 32.0f,
    .rope_yarn_beta_fast = 32.0f,
    .rope_yarn_beta_slow = 1.0f,
    .rope_yarn_attn_factor = 1.0f,
    .rope_freq_base_swa = 10000.0f,
    .context_length = 262144,
    .rope_orig_ctx = 8192,
};

const ds4_shape *lgn_model_shape(void) {
    return &LGN_SHAPE_LAGUNA_S21;
}

uint32_t lgn_model_layer_head_count(uint32_t il) {
    return lgn_layer_head_count(il);
}

bool lgn_model_layer_is_swa(uint32_t il) {
    return lgn_layer_is_swa(il);
}

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
} lgn_cursor;

static void lgn_die(const char *message) {
    fprintf(stderr, "ds4: %s\n", message);
    exit(1);
}

static void lgn_die_missing(const char *kind, const char *name) {
    fprintf(stderr, "ds4: required %s is missing: %s\n", kind, name);
    exit(1);
}

static bool lgn_cursor_has(const lgn_cursor *c, uint64_t n) {
    return c && c->base && n <= c->size && c->pos <= c->size - n;
}

static bool lgn_cursor_read(lgn_cursor *c, void *dst, uint64_t n) {
    if ((n != 0 && !dst) || !lgn_cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool lgn_cursor_u32(lgn_cursor *c, uint32_t *out) {
    return lgn_cursor_read(c, out, sizeof(*out));
}

static bool lgn_cursor_u64(lgn_cursor *c, uint64_t *out) {
    return lgn_cursor_read(c, out, sizeof(*out));
}

static bool lgn_cursor_string(lgn_cursor *c, ds4_str *out) {
    uint64_t len = 0;
    if (!out || !lgn_cursor_u64(c, &len) || !lgn_cursor_has(c, len)) {
        return false;
    }
    out->ptr = (const char *)(c->base + c->pos);
    out->len = len;
    c->pos += len;
    return true;
}

static lgn_cursor lgn_cursor_at(const ds4_model *m, uint64_t pos) {
    lgn_cursor c = {
        .base = m ? m->map : NULL,
        .size = m ? m->size : 0,
        .pos = pos,
    };
    return c;
}

static bool lgn_streq(ds4_str value, const char *literal) {
    const size_t n = strlen(literal);
    return value.ptr && value.len == n && memcmp(value.ptr, literal, n) == 0;
}

static ds4_kv *lgn_model_find_kv(const ds4_model *m, const char *key) {
    if (!m || !key || (m->n_kv != 0 && !m->kv)) return NULL;
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (lgn_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

bool lgn_model_get_string(const ds4_model *m,
                          const char *key,
                          ds4_str *out) {
    if (!out) return false;
    ds4_kv *kv = lgn_model_find_kv(m, key);
    if (!kv || kv->type != LGN_GGUF_VALUE_STRING) return false;
    lgn_cursor c = lgn_cursor_at(m, kv->value_pos);
    return lgn_cursor_string(&c, out);
}

bool lgn_model_get_u32(const ds4_model *m,
                       const char *key,
                       uint32_t *out) {
    if (!out) return false;
    ds4_kv *kv = lgn_model_find_kv(m, key);
    if (!kv || kv->type != LGN_GGUF_VALUE_UINT32) return false;
    lgn_cursor c = lgn_cursor_at(m, kv->value_pos);
    return lgn_cursor_u32(&c, out);
}

bool lgn_model_get_token_id(const ds4_model *m,
                            const char *key,
                            int *out) {
    ds4_kv *kv = lgn_model_find_kv(m, key);
    if (!kv || !out) return false;

    lgn_cursor c = lgn_cursor_at(m, kv->value_pos);
    switch (kv->type) {
    case LGN_GGUF_VALUE_UINT32: {
        uint32_t value = 0;
        if (!lgn_cursor_u32(&c, &value) || value > (uint32_t)INT_MAX) {
            return false;
        }
        *out = (int)value;
        return true;
    }
    case LGN_GGUF_VALUE_INT32: {
        int32_t value = 0;
        if (!lgn_cursor_read(&c, &value, sizeof(value)) || value < 0) {
            return false;
        }
        *out = (int)value;
        return true;
    }
    case LGN_GGUF_VALUE_UINT64: {
        uint64_t value = 0;
        if (!lgn_cursor_u64(&c, &value) || value > (uint64_t)INT_MAX) {
            return false;
        }
        *out = (int)value;
        return true;
    }
    case LGN_GGUF_VALUE_INT64: {
        int64_t value = 0;
        if (!lgn_cursor_read(&c, &value, sizeof(value)) || value < 0 ||
            value > (int64_t)INT_MAX) {
            return false;
        }
        *out = (int)value;
        return true;
    }
    default:
        return false;
    }
}

bool lgn_model_get_u64_compat(const ds4_model *m,
                              const char *key,
                              uint64_t *out) {
    if (!out) return false;
    ds4_kv *kv = lgn_model_find_kv(m, key);
    if (!kv) return false;
    lgn_cursor c = lgn_cursor_at(m, kv->value_pos);
    if (kv->type == LGN_GGUF_VALUE_UINT64) return lgn_cursor_u64(&c, out);
    if (kv->type == LGN_GGUF_VALUE_UINT32) {
        uint32_t value = 0;
        if (!lgn_cursor_u32(&c, &value)) return false;
        *out = value;
        return true;
    }
    return false;
}

bool lgn_model_get_f32_compat(const ds4_model *m,
                              const char *key,
                              float *out) {
    if (!out) return false;
    ds4_kv *kv = lgn_model_find_kv(m, key);
    if (!kv) return false;
    lgn_cursor c = lgn_cursor_at(m, kv->value_pos);
    if (kv->type == LGN_GGUF_VALUE_FLOAT32) {
        return lgn_cursor_read(&c, out, sizeof(*out));
    }
    if (kv->type == LGN_GGUF_VALUE_FLOAT64) {
        double value = 0.0;
        if (!lgn_cursor_read(&c, &value, sizeof(value))) return false;
        *out = (float)value;
        return true;
    }
    if (kv->type == LGN_GGUF_VALUE_UINT32) {
        uint32_t value = 0;
        if (!lgn_cursor_u32(&c, &value)) return false;
        *out = (float)value;
        return true;
    }
    if (kv->type == LGN_GGUF_VALUE_INT32) {
        int32_t value = 0;
        if (!lgn_cursor_read(&c, &value, sizeof(value))) return false;
        *out = (float)value;
        return true;
    }
    return false;
}

bool lgn_model_get_bool(const ds4_model *m,
                        const char *key,
                        bool *out) {
    if (!out) return false;
    ds4_kv *kv = lgn_model_find_kv(m, key);
    if (!kv || kv->type != LGN_GGUF_VALUE_BOOL) return false;
    lgn_cursor c = lgn_cursor_at(m, kv->value_pos);
    uint8_t value = 0;
    if (!lgn_cursor_read(&c, &value, sizeof(value))) return false;
    *out = value != 0;
    return true;
}

bool lgn_model_get_array(const ds4_model *m,
                         const char *key,
                         lgn_model_array *out) {
    ds4_kv *kv = lgn_model_find_kv(m, key);
    if (!kv || kv->type != LGN_GGUF_VALUE_ARRAY || !out) return false;
    lgn_cursor c = lgn_cursor_at(m, kv->value_pos);
    if (!lgn_cursor_u32(&c, &out->type) || !lgn_cursor_u64(&c, &out->len)) {
        return false;
    }
    out->data_pos = c.pos;
    return true;
}

bool lgn_model_get_u32_array(const ds4_model *m,
                             const char *key,
                             uint32_t *out,
                             uint32_t cap,
                             uint32_t *n_out) {
    if (n_out) *n_out = 0;
    if (!out || cap == 0 || !n_out) return false;

    lgn_model_array array = {0};
    if (!lgn_model_get_array(m, key, &array) ||
        (array.type != LGN_GGUF_VALUE_UINT32 &&
         array.type != LGN_GGUF_VALUE_INT32) ||
        array.len > cap) {
        return false;
    }

    lgn_cursor c = lgn_cursor_at(m, array.data_pos);
    for (uint64_t i = 0; i < array.len; i++) {
        if (array.type == LGN_GGUF_VALUE_UINT32) {
            if (!lgn_cursor_u32(&c, &out[i])) return false;
        } else {
            int32_t value = 0;
            if (!lgn_cursor_read(&c, &value, sizeof(value)) || value < 0) {
                return false;
            }
            out[i] = (uint32_t)value;
        }
    }
    *n_out = (uint32_t)array.len;
    return true;
}

ds4_tensor *lgn_model_find_tensor(const ds4_model *m, const char *name) {
    if (!m || !name || (m->n_tensors != 0 && !m->tensors)) return NULL;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (lgn_streq(m->tensors[i].name, name)) return &m->tensors[i];
    }
    return NULL;
}

ds4_tensor *lgn_model_required_tensor(const ds4_model *m, const char *name) {
    if (!name) lgn_die("required tensor name is missing");
    ds4_tensor *tensor = lgn_model_find_tensor(m, name);
    if (!tensor) lgn_die_missing("tensor", name);
    return tensor;
}

ds4_tensor *lgn_model_required_tensorf(const ds4_model *m,
                                       const char *format,
                                       uint32_t layer) {
    if (!format) lgn_die("required tensor format is missing");
    char name[128];
    const int n = snprintf(name, sizeof(name), format, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) {
        lgn_die("tensor name is too long");
    }
    return lgn_model_required_tensor(m, name);
}

static ds4_tensor *lgn_required_tensor(const ds4_model *m, const char *name) {
    ds4_tensor *tensor = lgn_model_find_tensor(m, name);
    if (!tensor) lgn_die_missing("tensor", name);
    return tensor;
}

static ds4_tensor *lgn_required_tensorf(const ds4_model *m,
                                        const char *format,
                                        uint32_t layer) {
    char name[128];
    const int n = snprintf(name, sizeof(name), format, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) lgn_die("tensor name is too long");
    return lgn_required_tensor(m, name);
}

const char *lgn_model_tensor_type_name(uint32_t type) {
    switch (type) {
    case LGN_TENSOR_F32:  return "f32";
    case LGN_TENSOR_F16:  return "f16";
    case LGN_TENSOR_Q4_0: return "q4_0";
    case LGN_TENSOR_Q4_1: return "q4_1";
    case LGN_TENSOR_Q5_0: return "q5_0";
    case LGN_TENSOR_Q5_1: return "q5_1";
    case LGN_TENSOR_Q2_K: return "q2_k";
    case LGN_TENSOR_Q3_K: return "q3_k";
    case LGN_TENSOR_Q4_K: return "q4_k";
    case LGN_TENSOR_Q5_K: return "q5_k";
    case LGN_TENSOR_Q6_K: return "q6_k";
    case LGN_TENSOR_Q8_0: return "q8_0";
    case LGN_TENSOR_Q8_K: return "q8_k";
    case LGN_TENSOR_Q8_1: return "q8_1";
    case LGN_TENSOR_IQ2_XXS: return "iq2_xxs";
    case LGN_TENSOR_IQ2_XS: return "iq2_xs";
    case LGN_TENSOR_IQ3_XXS: return "iq3_xxs";
    case LGN_TENSOR_IQ1_S: return "iq1_s";
    case LGN_TENSOR_IQ4_NL: return "iq4_nl";
    case LGN_TENSOR_IQ3_S: return "iq3_s";
    case LGN_TENSOR_IQ2_S: return "iq2_s";
    case LGN_TENSOR_IQ4_XS: return "iq4_xs";
    case LGN_TENSOR_I8: return "i8";
    case LGN_TENSOR_I16: return "i16";
    case LGN_TENSOR_I32: return "i32";
    case LGN_TENSOR_I64: return "i64";
    case LGN_TENSOR_F64: return "f64";
    case LGN_TENSOR_IQ1_M: return "iq1_m";
    case LGN_TENSOR_BF16: return "bf16";
    case LGN_TENSOR_MXFP4: return "mxfp4";
    default:              return "unknown";
    }
}

static bool lgn_tensor_is_routed_expert_type(uint32_t type) {
    return type == LGN_TENSOR_Q8_0 ||
           type == LGN_TENSOR_IQ2_XXS ||
           type == LGN_TENSOR_Q2_K ||
           type == LGN_TENSOR_Q3_K ||
           type == LGN_TENSOR_Q4_K ||
           type == LGN_TENSOR_Q5_K ||
           type == LGN_TENSOR_Q6_K ||
           type == LGN_TENSOR_MXFP4;
}

bool lgn_model_tensor_type_is_dense_quant(uint32_t type) {
    return type == LGN_TENSOR_Q8_0 ||
           type == LGN_TENSOR_Q4_K ||
           type == LGN_TENSOR_Q4_0;
}

static void lgn_tensor_expect_layout(const ds4_tensor *tensor,
                                     uint32_t type,
                                     uint32_t ndim,
                                     uint64_t d0,
                                     uint64_t d1,
                                     uint64_t d2) {
    if (!tensor) lgn_die("internal error: missing tensor while validating layout");
    if (tensor->type != type) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected %s\n",
                (int)tensor->name.len,
                tensor->name.ptr,
                lgn_model_tensor_type_name(tensor->type),
                lgn_model_tensor_type_name(type));
        exit(1);
    }
    if (tensor->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)tensor->name.len,
                tensor->name.ptr,
                tensor->ndim,
                ndim);
        exit(1);
    }
    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (i < 3u && tensor->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)tensor->name.len,
                tensor->name.ptr,
                i,
                tensor->dim[i],
                i < 3u ? want[i] : 0u);
        exit(1);
    }
}

void lgn_model_validate_tensor_layout(const ds4_tensor *tensor,
                                      uint32_t type,
                                      uint32_t ndim,
                                      uint64_t d0,
                                      uint64_t d1,
                                      uint64_t d2) {
    lgn_tensor_expect_layout(tensor, type, ndim, d0, d1, d2);
}

static void lgn_config_expect_u32(const char *name,
                                  uint32_t got,
                                  uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%u for %s, got %u\n",
            name, expected, LGN_SHAPE_LAGUNA_S21.name, got);
    exit(1);
}

static void lgn_config_expect_u64(const char *name,
                                  uint64_t got,
                                  uint64_t expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%" PRIu64 " for %s, got %" PRIu64 "\n",
            name, expected, LGN_SHAPE_LAGUNA_S21.name, got);
    exit(1);
}

static void lgn_config_expect_f32(const char *name,
                                  float got,
                                  float expected) {
    const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "ds4: expected %s=%.9g for %s, got %.9g\n",
            name,
            (double)expected,
            LGN_SHAPE_LAGUNA_S21.name,
            (double)got);
    exit(1);
}

static void lgn_config_expect_bool(const char *name,
                                   bool got,
                                   bool expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%s for %s, got %s\n",
            name,
            expected ? "true" : "false",
            LGN_SHAPE_LAGUNA_S21.name,
            got ? "true" : "false");
    exit(1);
}

static uint32_t lgn_required_u32(const ds4_model *m, const char *key) {
    uint32_t value = 0;
    if (!lgn_model_get_u32(m, key, &value)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return value;
}

static uint64_t lgn_required_u64(const ds4_model *m, const char *key) {
    uint64_t value = 0;
    if (!lgn_model_get_u64_compat(m, key, &value)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return value;
}

static float lgn_required_f32(const ds4_model *m, const char *key) {
    float value = 0.0f;
    if (!lgn_model_get_f32_compat(m, key, &value)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return value;
}

static bool lgn_required_bool(const ds4_model *m, const char *key) {
    bool value = false;
    if (!lgn_model_get_bool(m, key, &value)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return value;
}

bool lgn_model_is_laguna(const ds4_model *m, ds4_str *arch_out) {
    ds4_str arch = {0};
    if (!m || !lgn_model_get_string(m, "general.architecture", &arch)) {
        return false;
    }
    if (arch_out) *arch_out = arch;
    if (arch.len > SIZE_MAX) return false;
    return lgn_architecture_is_supported(arch.ptr, (size_t)arch.len);
}

void lgn_model_require_laguna_architecture(const ds4_model *m) {
    ds4_str arch = {0};
    if (lgn_model_is_laguna(m, &arch)) return;
    if (!arch.ptr) {
        lgn_die("GGUF general.architecture is required and must be literal laguna");
    }
    const int shown = arch.len > 128u ? 128 : (int)arch.len;
    fprintf(stderr,
            "ds4: unsupported GGUF general.architecture '%.*s%s'; "
            "only literal laguna is supported\n",
            shown,
            arch.ptr,
            arch.len > 128u ? "..." : "");
    exit(1);
}

void lgn_model_get_validated_summary(lgn_model_summary_fields *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));

    /* lgn_model_validate_config() admits only this exact profile before the
     * runtime can print a target summary.  Read the immutable profile rather
     * than stale family-prefixed metadata from the removed model families. */
    const ds4_shape *s = lgn_model_shape();
    out->n_layer = s->n_layer;
    out->context_length = s->context_length;
    out->n_head = s->n_head;
    out->n_head_kv = s->n_head_kv;
    out->n_head_dim = s->n_head_dim;
    out->n_swa = s->n_swa;
    out->n_expert = s->n_expert;
    out->n_expert_used = s->n_expert_used;
}

void lgn_model_validate_config(const ds4_model *m) {
    const ds4_shape *s = &LGN_SHAPE_LAGUNA_S21;
    const uint32_t n_layer = lgn_required_u32(m, "laguna.block_count");
    const uint64_t n_ctx = lgn_required_u64(m, "laguna.context_length");
    const uint32_t n_embd = lgn_required_u32(m, "laguna.embedding_length");
    const uint32_t n_vocab = lgn_required_u32(m, "laguna.vocab_size");
    const uint32_t n_ff_dense = lgn_required_u32(m, "laguna.feed_forward_length");
    const uint32_t n_head_kv = lgn_required_u32(m, "laguna.attention.head_count_kv");
    const uint32_t n_head_dim = lgn_required_u32(m, "laguna.attention.key_length");
    const uint32_t n_value_dim = lgn_required_u32(m, "laguna.attention.value_length");
    const uint32_t n_rot = lgn_required_u32(m, "laguna.rope.dimension_count");
    const uint32_t n_rot_swa = lgn_required_u32(m, "laguna.rope.dimension_count_swa");
    const uint32_t n_swa = lgn_required_u32(m, "laguna.attention.sliding_window");
    const uint32_t n_expert = lgn_required_u32(m, "laguna.expert_count");
    const uint32_t n_expert_used = lgn_required_u32(m, "laguna.expert_used_count");
    const uint32_t n_ff_exp = lgn_required_u32(m, "laguna.expert_feed_forward_length");
    const uint32_t n_ff_shared =
        lgn_required_u32(m, "laguna.expert_shared_feed_forward_length");
    const uint32_t expert_gating_func = lgn_required_u32(m, "laguna.expert_gating_func");
    const uint32_t n_leading_dense =
        lgn_required_u32(m, "laguna.leading_dense_block_count");

    lgn_config_expect_u32("block_count", n_layer, s->n_layer);
    lgn_config_expect_u64("context_length", n_ctx, s->context_length);
    lgn_config_expect_u32("embedding_length", n_embd, s->n_embd);
    lgn_config_expect_u32("vocab_size", n_vocab, s->n_vocab);
    lgn_config_expect_u32("feed_forward_length", n_ff_dense, s->n_ff_dense);
    lgn_config_expect_u32("attention.head_count_kv", n_head_kv, s->n_head_kv);
    lgn_config_expect_u32("attention.key_length", n_head_dim, s->n_head_dim);
    lgn_config_expect_u32("attention.value_length", n_value_dim, s->n_value_dim);
    lgn_config_expect_u32("rope.dimension_count", n_rot, s->n_rot);
    lgn_config_expect_u32("rope.dimension_count_swa", n_rot_swa, s->n_rot_swa);
    lgn_config_expect_u32("attention.sliding_window", n_swa, s->n_swa);
    lgn_config_expect_u32("expert_count", n_expert, s->n_expert);
    lgn_config_expect_u32("expert_used_count", n_expert_used, s->n_expert_used);
    lgn_config_expect_u32("expert_feed_forward_length", n_ff_exp, s->n_ff_exp);
    lgn_config_expect_u32("expert_shared_feed_forward_length", n_ff_shared, s->n_ff_shared);
    lgn_config_expect_u32("expert_gating_func", expert_gating_func, 2);
    lgn_config_expect_u32("leading_dense_block_count", n_leading_dense, s->n_leading_dense);

    lgn_model_array heads = {0};
    if (!lgn_model_get_array(m, "laguna.attention.head_count", &heads) ||
        (heads.type != LGN_GGUF_VALUE_UINT32 && heads.type != LGN_GGUF_VALUE_INT32) ||
        heads.len != s->n_layer) {
        lgn_die("laguna.attention.head_count must be an int32/uint32 array with one entry per layer");
    }
    lgn_cursor hc = lgn_cursor_at(m, heads.data_pos);
    for (uint32_t il = 0; il < s->n_layer; il++) {
        uint32_t got = 0;
        if (heads.type == LGN_GGUF_VALUE_UINT32) {
            if (!lgn_cursor_u32(&hc, &got)) lgn_die("truncated Laguna head-count metadata");
        } else {
            int32_t value = 0;
            if (!lgn_cursor_read(&hc, &value, sizeof(value))) {
                lgn_die("truncated Laguna head-count metadata");
            }
            if (value <= 0) lgn_die("Laguna head-count metadata contains a non-positive value");
            got = (uint32_t)value;
        }
        const uint32_t expected = lgn_layer_head_count(il);
        if (got != expected) {
            fprintf(stderr,
                    "ds4: unexpected Laguna head count at layer %u: got %u, expected %u\n",
                    il, got, expected);
            exit(1);
        }
    }

    ds4_str rope_type = {0};
    if (!lgn_model_get_string(m, "laguna.rope.scaling.type", &rope_type) ||
        !lgn_streq(rope_type, "yarn")) {
        lgn_die("Laguna requires rope.scaling.type=yarn");
    }
    lgn_config_expect_u64("rope.scaling.original_context_length",
                          lgn_required_u64(m, "laguna.rope.scaling.original_context_length"),
                          s->rope_orig_ctx);
    lgn_config_expect_f32("rope.freq_base",
                          lgn_required_f32(m, "laguna.rope.freq_base"),
                          s->rope_freq_base);
    lgn_config_expect_f32("rope.freq_base_swa",
                          lgn_required_f32(m, "laguna.rope.freq_base_swa"),
                          s->rope_freq_base_swa);
    lgn_config_expect_f32("rope.scaling.factor",
                          lgn_required_f32(m, "laguna.rope.scaling.factor"),
                          s->rope_scale_factor);
    lgn_config_expect_f32("rope.scaling.yarn_attn_factor",
                          lgn_required_f32(m, "laguna.rope.scaling.yarn_attn_factor"),
                          s->rope_yarn_attn_factor);
    lgn_config_expect_f32("rope.scaling.yarn_beta_fast",
                          lgn_required_f32(m, "laguna.rope.scaling.yarn_beta_fast"),
                          s->rope_yarn_beta_fast);
    lgn_config_expect_f32("rope.scaling.yarn_beta_slow",
                          lgn_required_f32(m, "laguna.rope.scaling.yarn_beta_slow"),
                          s->rope_yarn_beta_slow);
    lgn_config_expect_f32("attention.layer_norm_rms_epsilon",
                          lgn_required_f32(m, "laguna.attention.layer_norm_rms_epsilon"),
                          s->rms_eps);
    lgn_config_expect_f32("expert_weights_scale",
                          lgn_required_f32(m, "laguna.expert_weights_scale"),
                          s->expert_weight_scale);
    lgn_config_expect_bool("expert_weights_norm",
                           lgn_required_bool(m, "laguna.expert_weights_norm"),
                           true);
}

bool lgn_weights_have_output_head(const ds4_weights *w) {
    return w && w->output_norm && w->output;
}

bool lgn_weights_have_partial_output_head(const ds4_weights *w) {
    return w && (w->output_norm || w->output);
}

bool lgn_weights_laguna_layer_has_required(const ds4_layer_weights *l,
                                           uint32_t il) {
    if (!l ||
        !l->attn_norm ||
        !l->attn_q ||
        !l->attn_k ||
        !l->attn_v ||
        !l->attn_gate ||
        !l->attn_q_norm ||
        !l->attn_k_norm ||
        !l->attn_output ||
        !l->ffn_norm) {
        return false;
    }
    if (il < LGN_SHAPE_LAGUNA_S21.n_leading_dense) {
        return l->ffn_gate && l->ffn_up && l->ffn_down;
    }
    return l->ffn_gate_inp &&
           l->ffn_exp_probs_b &&
           l->ffn_gate_exps &&
           l->ffn_up_exps &&
           l->ffn_down_exps &&
           l->ffn_gate_shexp &&
           l->ffn_up_shexp &&
           l->ffn_down_shexp;
}

void lgn_weights_validate_layout(const ds4_weights *w,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 bool require_token_embd,
                                 bool require_output) {
    const ds4_shape *s = &LGN_SHAPE_LAGUNA_S21;
    if (!w) lgn_die("internal error: missing weights while validating Laguna layout");
    if (layer_start >= s->n_layer) lgn_die("invalid first layer in Laguna weight layout validation");
    if (layer_end == UINT32_MAX) layer_end = s->n_layer - 1u;
    if (layer_end >= s->n_layer || layer_end < layer_start) {
        lgn_die("invalid layer range in Laguna weight layout validation");
    }

    /* Poolside published two coherent recipes under the same Q4_K_M
     * filename. The embedding type identifies full models; attention Q is
     * the equivalent marker for layer-only weight views. */
    const ds4_tensor *layout_marker = w->token_embd;
    if (!layout_marker) layout_marker = w->layer[layer_start].attn_q;
    if (!layout_marker) lgn_die("cannot identify Laguna quantization layout");
    const bool signal_q8 = layout_marker->type == LGN_TENSOR_Q8_0;
    const bool legacy_layout =
        (w->token_embd && layout_marker->type == LGN_TENSOR_Q4_K) ||
        (!w->token_embd && layout_marker->type == LGN_TENSOR_F16);
    if (!signal_q8 && !legacy_layout) {
        fprintf(stderr,
                "ds4: unsupported Laguna quantization layout marker %s; "
                "expected legacy Q4_K/F16 or Q8_0 signal weights\n",
                lgn_model_tensor_type_name(layout_marker->type));
        exit(1);
    }

    if (require_token_embd && !w->token_embd) {
        lgn_die("required token embedding tensor is missing");
    }
    if (w->token_embd) {
        lgn_tensor_expect_layout(w->token_embd,
                                 signal_q8 ? LGN_TENSOR_Q8_0 : LGN_TENSOR_Q4_K,
                                 2, s->n_embd, s->n_vocab, 0);
    }

    const bool have_output = lgn_weights_have_output_head(w);
    if (require_output && !have_output) lgn_die("required output head tensors are missing");
    if (lgn_weights_have_partial_output_head(w) && !have_output) {
        lgn_die("partial output head in GGUF");
    }
    if (have_output) {
        lgn_tensor_expect_layout(w->output_norm, LGN_TENSOR_F32,
                                 1, s->n_embd, 0, 0);
        lgn_tensor_expect_layout(w->output,
                                 signal_q8 ? LGN_TENSOR_Q8_0 : LGN_TENSOR_Q6_K,
                                 2, s->n_embd, s->n_vocab, 0);
    }

    for (uint32_t il = layer_start; il <= layer_end; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        if (!lgn_weights_laguna_layer_has_required(l, il)) {
            fprintf(stderr, "ds4: required Laguna tensors for layer %u are missing\n", il);
            exit(1);
        }
        const uint32_t n_head = lgn_layer_head_count(il);
        const uint64_t q_dim = (uint64_t)n_head * s->n_head_dim;
        const uint64_t kv_dim = (uint64_t)s->n_head_kv * s->n_head_dim;

        lgn_tensor_expect_layout(l->attn_norm, LGN_TENSOR_F32,
                                 1, s->n_embd, 0, 0);
        const uint32_t attn_type = signal_q8 ? LGN_TENSOR_Q8_0 : LGN_TENSOR_F16;
        lgn_tensor_expect_layout(l->attn_q, attn_type, 2, s->n_embd, q_dim, 0);
        lgn_tensor_expect_layout(l->attn_k, attn_type, 2, s->n_embd, kv_dim, 0);
        lgn_tensor_expect_layout(l->attn_v, attn_type, 2, s->n_embd, kv_dim, 0);
        lgn_tensor_expect_layout(l->attn_gate, attn_type, 2, s->n_embd, n_head, 0);
        lgn_tensor_expect_layout(l->attn_q_norm, LGN_TENSOR_F32,
                                 1, s->n_head_dim, 0, 0);
        lgn_tensor_expect_layout(l->attn_k_norm, LGN_TENSOR_F32,
                                 1, s->n_head_dim, 0, 0);
        lgn_tensor_expect_layout(l->attn_output, attn_type,
                                 2, q_dim, s->n_embd, 0);
        lgn_tensor_expect_layout(l->ffn_norm, LGN_TENSOR_F32,
                                 1, s->n_embd, 0, 0);

        if (il < s->n_leading_dense) {
            lgn_tensor_expect_layout(l->ffn_gate,
                                     signal_q8 ? LGN_TENSOR_Q8_0 : LGN_TENSOR_Q4_K,
                                     2, s->n_embd, s->n_ff_dense, 0);
            lgn_tensor_expect_layout(l->ffn_up,
                                     signal_q8 ? LGN_TENSOR_Q8_0 : LGN_TENSOR_Q4_K,
                                     2, s->n_embd, s->n_ff_dense, 0);
            lgn_tensor_expect_layout(l->ffn_down,
                                     signal_q8 ? LGN_TENSOR_Q8_0 : LGN_TENSOR_Q6_K,
                                     2, s->n_ff_dense, s->n_embd, 0);
            continue;
        }

        lgn_tensor_expect_layout(l->ffn_gate_inp, LGN_TENSOR_F32,
                                 2, s->n_embd, s->n_expert, 0);
        lgn_tensor_expect_layout(l->ffn_exp_probs_b, LGN_TENSOR_F32,
                                 1, s->n_expert, 0, 0);
        /* Mixed files may spend more bits on selected layers, but all three
         * routed projections within one layer must use a coherent layout. */
        const uint32_t layer_routed_type = l->ffn_gate_exps->type;
        if (layer_routed_type != LGN_TENSOR_Q4_K &&
            layer_routed_type != LGN_TENSOR_Q3_K &&
            layer_routed_type != LGN_TENSOR_Q2_K) {
            fprintf(stderr,
                    "ds4: Laguna routed experts for layer %u have unsupported type %s\n",
                    il, lgn_model_tensor_type_name(layer_routed_type));
            exit(1);
        }
        if (!lgn_tensor_is_routed_expert_type(layer_routed_type)) {
            lgn_die("unsupported Laguna routed expert type");
        }
        lgn_tensor_expect_layout(l->ffn_gate_exps, layer_routed_type,
                                 3, s->n_embd, s->n_ff_exp, s->n_expert);
        lgn_tensor_expect_layout(l->ffn_up_exps, layer_routed_type,
                                 3, s->n_embd, s->n_ff_exp, s->n_expert);
        const bool down_supported =
            l->ffn_down_exps->type == layer_routed_type ||
            (layer_routed_type == LGN_TENSOR_Q4_K &&
             !signal_q8 && l->ffn_down_exps->type == LGN_TENSOR_Q6_K);
        if (!down_supported) {
            fprintf(stderr,
                    "ds4: Laguna routed down tensor for layer %u has type %s, "
                    "incompatible with %s gate/up experts\n",
                    il,
                    lgn_model_tensor_type_name(l->ffn_down_exps->type),
                    lgn_model_tensor_type_name(layer_routed_type));
            exit(1);
        }
        lgn_tensor_expect_layout(l->ffn_down_exps, l->ffn_down_exps->type,
                                 3, s->n_ff_exp, s->n_embd, s->n_expert);
        lgn_tensor_expect_layout(l->ffn_gate_shexp,
                                 signal_q8 ? LGN_TENSOR_Q8_0 : LGN_TENSOR_Q4_K,
                                 2, s->n_embd, s->n_ff_shared, 0);
        lgn_tensor_expect_layout(l->ffn_up_shexp,
                                 signal_q8 ? LGN_TENSOR_Q8_0 : LGN_TENSOR_Q4_K,
                                 2, s->n_embd, s->n_ff_shared, 0);
        const uint32_t shared_down_type =
            signal_q8 ? LGN_TENSOR_Q8_0 : l->ffn_down_shexp->type;
        if (!signal_q8 &&
            shared_down_type != LGN_TENSOR_Q4_K &&
            shared_down_type != LGN_TENSOR_Q6_K) {
            fprintf(stderr,
                    "ds4: Laguna shared down tensor for layer %u has unsupported type %s\n",
                    il, lgn_model_tensor_type_name(shared_down_type));
            exit(1);
        }
        lgn_tensor_expect_layout(l->ffn_down_shexp, shared_down_type,
                                 2, s->n_ff_shared, s->n_embd, 0);
    }
}

static void lgn_weights_bind_output(ds4_weights *w,
                                    const ds4_model *m,
                                    bool required,
                                    bool optional) {
    if (required) {
        w->output_norm = lgn_required_tensor(m, "output_norm.weight");
        w->output = lgn_required_tensor(m, "output.weight");
    } else if (optional) {
        w->output_norm = lgn_model_find_tensor(m, "output_norm.weight");
        w->output = lgn_model_find_tensor(m, "output.weight");
    }
    if (optional &&
        lgn_weights_have_partial_output_head(w) &&
        !lgn_weights_have_output_head(w)) {
        lgn_die("partial output head in GGUF");
    }
}

static void lgn_weights_bind_layer(ds4_layer_weights *l,
                                   const ds4_model *m,
                                   uint32_t il) {
    l->attn_norm = lgn_required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->attn_q = lgn_required_tensorf(m, "blk.%u.attn_q.weight", il);
    l->attn_k = lgn_required_tensorf(m, "blk.%u.attn_k.weight", il);
    l->attn_v = lgn_required_tensorf(m, "blk.%u.attn_v.weight", il);
    l->attn_gate = lgn_required_tensorf(m, "blk.%u.attn_gate.weight", il);
    l->attn_q_norm = lgn_required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
    l->attn_k_norm = lgn_required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
    l->attn_output = lgn_required_tensorf(m, "blk.%u.attn_output.weight", il);
    l->ffn_norm = lgn_required_tensorf(m, "blk.%u.ffn_norm.weight", il);

    if (il < LGN_SHAPE_LAGUNA_S21.n_leading_dense) {
        l->ffn_gate = lgn_required_tensorf(m, "blk.%u.ffn_gate.weight", il);
        l->ffn_up = lgn_required_tensorf(m, "blk.%u.ffn_up.weight", il);
        l->ffn_down = lgn_required_tensorf(m, "blk.%u.ffn_down.weight", il);
        return;
    }

    l->ffn_gate_inp = lgn_required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
    l->ffn_exp_probs_b = lgn_required_tensorf(m, "blk.%u.exp_probs_b.bias", il);
    l->ffn_gate_exps = lgn_required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
    l->ffn_up_exps = lgn_required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
    l->ffn_down_exps = lgn_required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
    l->ffn_gate_shexp = lgn_required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
    l->ffn_up_shexp = lgn_required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
    l->ffn_down_shexp = lgn_required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);
}

void lgn_weights_bind(ds4_weights *w,
                      const ds4_model *m) {
    const uint32_t executable_layers = LGN_SHAPE_LAGUNA_S21.n_layer;
    if (executable_layers != LGN_MODEL_MAX_LAYER) {
        lgn_die("Laguna weight-table capacity does not match the immutable profile");
    }
    memset(w, 0, sizeof(*w));

    /* Laguna uses a whole-model mmap.  Every tensor in the exact 48-layer
     * profile, including token embeddings and the output head, is bound up
     * front; partial layer loading is not part of the product contract. */
    w->token_embd = lgn_required_tensor(m, "token_embd.weight");
    lgn_weights_bind_output(w, m, true, false);
    for (uint32_t il = 0; il < executable_layers; il++) {
        lgn_weights_bind_layer(&w->layer[il], m, il);
    }
}
