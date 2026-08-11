/*
 * lgn_dflash.c - Laguna DFlash support-model profile and binding.
 *
 * DFlash graph construction and speculative verification stay in ds4.c.  This
 * file deliberately contains only the immutable six-layer support profile,
 * GGUF metadata/layout checks, direct tensor binding, and the BF16 shadow-map
 * conversion needed by the retained Metal graph code.
 */

#include "lgn_dflash.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static const lgn_dflash_profile LGN_DFLASH_PROFILE = {
    .n_layer = LGN_DFLASH_N_LAYER,
    .n_aux = LGN_DFLASH_N_AUX,
    .block_size = LGN_DFLASH_BLOCK_SIZE,
    .cache_cap = LGN_DFLASH_CACHE_CAP,
    .n_embd = 3072u,
    .n_head = 72u,
    .n_head_kv = 8u,
    .n_head_dim = 128u,
    .n_value_dim = 128u,
    .n_rot = 128u,
    .n_ff_dense = 12288u,
    .context_length = UINT64_C(1048576),
    .mask_token_id = 12u,
    .rope_freq_base = 500000.0f,
    .rms_eps = 1.0e-6f,
    .target_layers = { 2u, 11u, 20u, 30u, 39u, 48u },
};

const lgn_dflash_profile *lgn_dflash_profile_get(void) {
    return &LGN_DFLASH_PROFILE;
}

bool lgn_dflash_profile_matches_laguna(void) {
    const lgn_dflash_profile *profile = &LGN_DFLASH_PROFILE;
    const ds4_shape *shape = lgn_model_shape();
    if (!shape) return false;
    return profile->n_embd == shape->n_embd &&
           profile->n_head == shape->n_head &&
           profile->n_head_kv == shape->n_head_kv &&
           profile->n_head_dim == shape->n_head_dim &&
           profile->n_value_dim == shape->n_value_dim &&
           profile->n_rot == shape->n_rot_swa &&
           profile->n_ff_dense == shape->n_ff_dense &&
           profile->rope_freq_base == shape->rope_freq_base &&
           profile->rms_eps == shape->rms_eps;
}

static void lgn_dflash_die(const char *message) {
    fprintf(stderr, "ds4: %s\n", message);
    exit(1);
}

static bool lgn_dflash_streq(ds4_str value, const char *literal) {
    const size_t len = strlen(literal);
    return value.ptr && value.len == len && memcmp(value.ptr, literal, len) == 0;
}

static uint32_t lgn_dflash_required_u32(const ds4_model *model,
                                        const char *key) {
    uint32_t value = 0;
    if (!lgn_model_get_u32(model, key, &value)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return value;
}

static uint64_t lgn_dflash_required_u64(const ds4_model *model,
                                        const char *key) {
    uint64_t value = 0;
    if (!lgn_model_get_u64_compat(model, key, &value)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return value;
}

static float lgn_dflash_required_f32(const ds4_model *model,
                                     const char *key) {
    float value = 0.0f;
    if (!lgn_model_get_f32_compat(model, key, &value)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return value;
}

static void lgn_dflash_expect_u32(const char *name,
                                  uint32_t got,
                                  uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%u for Laguna S 2.1, got %u\n",
            name, expected, got);
    exit(1);
}

static void lgn_dflash_expect_u64(const char *name,
                                  uint64_t got,
                                  uint64_t expected) {
    if (got == expected) return;
    fprintf(stderr,
            "ds4: expected %s=%" PRIu64 " for Laguna S 2.1, got %" PRIu64 "\n",
            name, expected, got);
    exit(1);
}

static void lgn_dflash_expect_f32(const char *name,
                                  float got,
                                  float expected) {
    const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr,
            "ds4: expected %s=%.9g for Laguna S 2.1, got %.9g\n",
            name, (double)expected, (double)got);
    exit(1);
}

static bool lgn_dflash_layer_binding_eligible(
        const lgn_dflash_layer_weights *layer) {
    return layer &&
           layer->attn_norm &&
           layer->attn_q &&
           layer->attn_k &&
           layer->attn_v &&
           layer->attn_gate &&
           layer->attn_q_norm &&
           layer->attn_k_norm &&
           layer->attn_output &&
           layer->ffn_norm &&
           layer->ffn_gate &&
           layer->ffn_up &&
           layer->ffn_down;
}

bool lgn_dflash_binding_eligible(const lgn_dflash_weights *weights) {
    const lgn_dflash_profile *profile = &LGN_DFLASH_PROFILE;
    if (!weights ||
        !weights->aux_norm ||
        !weights->fc ||
        !weights->encoder_output_norm ||
        !weights->output_norm ||
        weights->block_size != profile->block_size ||
        weights->mask_token_id != profile->mask_token_id) {
        return false;
    }
    for (uint32_t il = 0; il < profile->n_layer; il++) {
        if (!lgn_dflash_layer_binding_eligible(&weights->layer[il])) {
            return false;
        }
    }
    for (uint32_t i = 0; i < profile->n_aux; i++) {
        if (weights->target_layers[i] != profile->target_layers[i]) {
            return false;
        }
    }
    return true;
}

void lgn_dflash_weights_validate_layout(
        const lgn_dflash_weights *weights) {
    const lgn_dflash_profile *profile = &LGN_DFLASH_PROFILE;
    if (!lgn_dflash_profile_matches_laguna()) {
        lgn_dflash_die("internal error: DFlash profile does not match Laguna S 2.1");
    }
    if (!weights || !weights->fc) {
        lgn_dflash_die("internal error: missing DFlash weights while validating layout");
    }

    const ds4_shape *shape = lgn_model_shape();
    const uint64_t q_dim = (uint64_t)shape->n_head * shape->n_head_dim;
    const uint64_t kv_dim = (uint64_t)shape->n_head_kv * shape->n_head_dim;
    const uint32_t matrix_type = weights->fc->type;

    if (matrix_type != LGN_TENSOR_BF16 &&
        !lgn_model_tensor_type_is_dense_quant(matrix_type)) {
        fprintf(stderr,
                "ds4: DFlash matrices have unsupported type %s\n",
                lgn_model_tensor_type_name(matrix_type));
        exit(1);
    }

    lgn_model_validate_tensor_layout(weights->aux_norm, LGN_TENSOR_F32, 2,
                                     shape->n_embd, profile->n_aux, 0);
    lgn_model_validate_tensor_layout(weights->fc, matrix_type, 2,
                                     (uint64_t)profile->n_aux * shape->n_embd,
                                     shape->n_embd, 0);
    lgn_model_validate_tensor_layout(weights->encoder_output_norm,
                                     LGN_TENSOR_F32, 1, shape->n_embd, 0, 0);
    lgn_model_validate_tensor_layout(weights->output_norm,
                                     LGN_TENSOR_F32, 1, shape->n_embd, 0, 0);

    for (uint32_t il = 0; il < profile->n_layer; il++) {
        const lgn_dflash_layer_weights *layer = &weights->layer[il];
        lgn_model_validate_tensor_layout(layer->attn_norm, LGN_TENSOR_F32,
                                         1, shape->n_embd, 0, 0);
        lgn_model_validate_tensor_layout(layer->attn_q, matrix_type, 2,
                                         shape->n_embd, q_dim, 0);
        lgn_model_validate_tensor_layout(layer->attn_k, matrix_type, 2,
                                         shape->n_embd, kv_dim, 0);
        lgn_model_validate_tensor_layout(layer->attn_v, matrix_type, 2,
                                         shape->n_embd, kv_dim, 0);
        lgn_model_validate_tensor_layout(layer->attn_gate, matrix_type, 2,
                                         shape->n_embd, shape->n_head, 0);
        lgn_model_validate_tensor_layout(layer->attn_q_norm,
                                         LGN_TENSOR_F32, 1,
                                         shape->n_head_dim, 0, 0);
        lgn_model_validate_tensor_layout(layer->attn_k_norm,
                                         LGN_TENSOR_F32, 1,
                                         shape->n_head_dim, 0, 0);
        lgn_model_validate_tensor_layout(layer->attn_output, matrix_type, 2,
                                         q_dim, shape->n_embd, 0);
        lgn_model_validate_tensor_layout(layer->ffn_norm, LGN_TENSOR_F32,
                                         1, shape->n_embd, 0, 0);
        lgn_model_validate_tensor_layout(layer->ffn_gate, matrix_type, 2,
                                         shape->n_embd, profile->n_ff_dense, 0);
        lgn_model_validate_tensor_layout(layer->ffn_up, matrix_type, 2,
                                         shape->n_embd, profile->n_ff_dense, 0);
        lgn_model_validate_tensor_layout(layer->ffn_down, matrix_type, 2,
                                         profile->n_ff_dense, shape->n_embd, 0);
    }
}

void lgn_dflash_weights_bind(lgn_dflash_weights *weights,
                             const ds4_model *model) {
    const lgn_dflash_profile *profile = &LGN_DFLASH_PROFILE;
    const ds4_shape *shape = lgn_model_shape();
    if (!lgn_dflash_profile_matches_laguna()) {
        lgn_dflash_die("internal error: DFlash profile does not match Laguna S 2.1");
    }
    memset(weights, 0, sizeof(*weights));

    weights->aux_norm = lgn_model_required_tensor(model, "enc.aux_norm.weight");
    weights->fc = lgn_model_required_tensor(model, "fc.weight");
    weights->encoder_output_norm =
        lgn_model_required_tensor(model, "enc.output_norm.weight");
    weights->output_norm = lgn_model_required_tensor(model, "output_norm.weight");
    for (uint32_t il = 0; il < profile->n_layer; il++) {
        lgn_dflash_layer_weights *layer = &weights->layer[il];
        layer->attn_norm =
            lgn_model_required_tensorf(model, "blk.%u.attn_norm.weight", il);
        layer->attn_q =
            lgn_model_required_tensorf(model, "blk.%u.attn_q.weight", il);
        layer->attn_k =
            lgn_model_required_tensorf(model, "blk.%u.attn_k.weight", il);
        layer->attn_v =
            lgn_model_required_tensorf(model, "blk.%u.attn_v.weight", il);
        layer->attn_gate =
            lgn_model_required_tensorf(model, "blk.%u.attn_gate.weight", il);
        layer->attn_q_norm =
            lgn_model_required_tensorf(model, "blk.%u.attn_q_norm.weight", il);
        layer->attn_k_norm =
            lgn_model_required_tensorf(model, "blk.%u.attn_k_norm.weight", il);
        layer->attn_output =
            lgn_model_required_tensorf(model, "blk.%u.attn_output.weight", il);
        layer->ffn_norm =
            lgn_model_required_tensorf(model, "blk.%u.ffn_norm.weight", il);
        layer->ffn_gate =
            lgn_model_required_tensorf(model, "blk.%u.ffn_gate.weight", il);
        layer->ffn_up =
            lgn_model_required_tensorf(model, "blk.%u.ffn_up.weight", il);
        layer->ffn_down =
            lgn_model_required_tensorf(model, "blk.%u.ffn_down.weight", il);
    }

    lgn_dflash_expect_u32("DFlash block_count",
                          lgn_dflash_required_u32(model, "dflash.block_count"),
                          profile->n_layer);
    lgn_dflash_expect_u64("DFlash context_length",
                          lgn_dflash_required_u64(model, "dflash.context_length"),
                          profile->context_length);
    lgn_dflash_expect_u32("DFlash embedding_length",
                          lgn_dflash_required_u32(model, "dflash.embedding_length"),
                          shape->n_embd);
    lgn_dflash_expect_u32("DFlash feed_forward_length",
                          lgn_dflash_required_u32(model, "dflash.feed_forward_length"),
                          shape->n_ff_dense);
    lgn_dflash_expect_u32("DFlash attention.head_count",
                          lgn_dflash_required_u32(model,
                                                  "dflash.attention.head_count"),
                          shape->n_head);
    lgn_dflash_expect_u32("DFlash attention.head_count_kv",
                          lgn_dflash_required_u32(model,
                                                  "dflash.attention.head_count_kv"),
                          shape->n_head_kv);
    lgn_dflash_expect_u32("DFlash attention.key_length",
                          lgn_dflash_required_u32(model,
                                                  "dflash.attention.key_length"),
                          shape->n_head_dim);
    lgn_dflash_expect_u32("DFlash attention.value_length",
                          lgn_dflash_required_u32(model,
                                                  "dflash.attention.value_length"),
                          shape->n_value_dim);
    lgn_dflash_expect_u32("DFlash rope.dimension_count",
                          lgn_dflash_required_u32(model,
                                                  "dflash.rope.dimension_count"),
                          shape->n_rot_swa);
    lgn_dflash_expect_u32("DFlash attention.sliding_window",
                          lgn_dflash_required_u32(
                              model, "dflash.attention.sliding_window"),
                          profile->cache_cap);

    uint32_t swa_pattern[LGN_DFLASH_N_LAYER] = {0};
    uint32_t n_swa_pattern = 0;
    if (!lgn_model_get_u32_array(model,
                                 "dflash.attention.sliding_window_pattern",
                                 swa_pattern,
                                 profile->n_layer,
                                 &n_swa_pattern) ||
        n_swa_pattern != profile->n_layer) {
        lgn_dflash_die("DFlash sliding_window_pattern must contain six entries");
    }
    for (uint32_t il = 0; il < profile->n_layer; il++) {
        lgn_dflash_expect_u32("DFlash sliding-window layer",
                              swa_pattern[il], 1u);
    }
    lgn_dflash_expect_f32("DFlash rope.freq_base",
                          lgn_dflash_required_f32(model, "dflash.rope.freq_base"),
                          profile->rope_freq_base);
    lgn_dflash_expect_f32(
        "DFlash attention.layer_norm_rms_epsilon",
        lgn_dflash_required_f32(model,
                                "dflash.attention.layer_norm_rms_epsilon"),
        profile->rms_eps);

    weights->block_size = lgn_dflash_required_u32(model, "dflash.block_size");
    lgn_dflash_expect_u32("DFlash block_size", weights->block_size,
                          profile->block_size);
    int mask_token = -1;
    if (!lgn_model_get_token_id(model, "tokenizer.ggml.mask_token_id",
                                &mask_token) ||
        mask_token < 0) {
        lgn_dflash_die("DFlash tokenizer.ggml.mask_token_id is missing");
    }
    weights->mask_token_id = (uint32_t)mask_token;
    lgn_dflash_expect_u32("DFlash mask token", weights->mask_token_id,
                          profile->mask_token_id);

    uint32_t n_target = 0;
    if (!lgn_model_get_u32_array(model,
                                 "dflash.target_layers",
                                 weights->target_layers,
                                 profile->n_aux,
                                 &n_target) ||
        n_target != profile->n_aux) {
        lgn_dflash_die("DFlash target_layers must contain six layer indices");
    }
    for (uint32_t i = 0; i < profile->n_aux; i++) {
        if (weights->target_layers[i] != profile->target_layers[i]) {
            fprintf(stderr,
                    "ds4: DFlash target_layers[%u]=%u, expected %u\n",
                    i, weights->target_layers[i], profile->target_layers[i]);
            exit(1);
        }
    }

    ds4_str decoder = {0};
    if (!lgn_model_get_string(model, "dflash.decoder_arch", &decoder) ||
        !lgn_dflash_streq(decoder, "laguna")) {
        lgn_dflash_die("DFlash support model must declare decoder_arch=laguna");
    }
    ds4_str rope_scaling = {0};
    if (!lgn_model_get_string(model, "dflash.rope.scaling.type",
                              &rope_scaling) ||
        !lgn_dflash_streq(rope_scaling, "none")) {
        lgn_dflash_die("DFlash support model must use rope.scaling.type=none");
    }

    lgn_dflash_weights_validate_layout(weights);
}

static inline uint16_t lgn_dflash_f32_to_f16(float value) {
#if defined(__ARM_NEON)
    const float32x4_t fv = vdupq_n_f32(value);
    const float16x4_t hv = vcvt_f16_f32(fv);
    return vget_lane_u16(vreinterpret_u16_f16(hv), 0);
#else
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
#endif
}

static double lgn_dflash_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

typedef struct {
    const uint16_t *src;
    uint16_t *dst;
} lgn_dflash_convert_ctx;

static void lgn_dflash_convert_bf16_rows(void *opaque,
                                         uint64_t begin,
                                         uint64_t end) {
    lgn_dflash_convert_ctx *ctx = opaque;
    uint64_t i = begin;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    for (; i + 8u <= end; i += 8u) {
        const uint16x8_t b = vld1q_u16(ctx->src + i);
        uint32x4_t lo = vmovl_u16(vget_low_u16(b));
        uint32x4_t hi = vmovl_high_u16(b);
        lo = vshlq_n_u32(lo, 16);
        hi = vshlq_n_u32(hi, 16);
        const float16x4_t hlo =
            vcvt_f16_f32(vreinterpretq_f32_u32(lo));
        const float16x4_t hhi =
            vcvt_f16_f32(vreinterpretq_f32_u32(hi));
        vst1q_u16(ctx->dst + i,
                  vcombine_u16(vreinterpret_u16_f16(hlo),
                               vreinterpret_u16_f16(hhi)));
    }
#endif
    for (; i < end; i++) {
        const uint32_t bits = (uint32_t)ctx->src[i] << 16;
        float value = 0.0f;
        memcpy(&value, &bits, sizeof(value));
        ctx->dst[i] = lgn_dflash_f32_to_f16(value);
    }
}

void *lgn_dflash_prepare_f16_map(const ds4_model *model,
                                 lgn_dflash_parallel_for_fn parallel_for,
                                 void *parallel_ctx) {
    if (!model || !model->map || model->size == 0 ||
        model->size > (uint64_t)SIZE_MAX ||
        (model->n_tensors != 0 && !model->tensors)) {
        return NULL;
    }
#if defined(MAP_ANONYMOUS)
    const int anon_flag = MAP_ANONYMOUS;
#else
    const int anon_flag = MAP_ANON;
#endif
    uint8_t *shadow = mmap(NULL,
                           (size_t)model->size,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | anon_flag,
                           -1,
                           0);
    if (shadow == MAP_FAILED) {
        fprintf(stderr,
                "ds4: could not allocate %.2f GiB for DFlash F16 weights: %s\n",
                (double)model->size / 1073741824.0,
                strerror(errno));
        return NULL;
    }

    const double t0 = lgn_dflash_now_sec();
    uint64_t converted = 0;
    uint64_t copied = 0;
    for (uint64_t i = 0; i < model->n_tensors; i++) {
        const ds4_tensor *tensor = &model->tensors[i];
        if (tensor->abs_offset > model->size ||
            tensor->bytes > model->size - tensor->abs_offset) {
            fprintf(stderr,
                    "ds4: DFlash tensor %.*s is outside its GGUF mapping\n",
                    (int)tensor->name.len, tensor->name.ptr);
            munmap(shadow, (size_t)model->size);
            return NULL;
        }
        if (tensor->type == LGN_TENSOR_F32) {
            memcpy(shadow + tensor->abs_offset,
                   model->map + tensor->abs_offset,
                   (size_t)tensor->bytes);
            copied += tensor->bytes;
            continue;
        }
        if (tensor->type != LGN_TENSOR_BF16 ||
            tensor->elements > UINT64_MAX / sizeof(uint16_t) ||
            tensor->bytes != tensor->elements * sizeof(uint16_t)) {
            fprintf(stderr,
                    "ds4: DFlash tensor %.*s has unsupported type %s\n",
                    (int)tensor->name.len, tensor->name.ptr,
                    lgn_model_tensor_type_name(tensor->type));
            munmap(shadow, (size_t)model->size);
            return NULL;
        }
        lgn_dflash_convert_ctx ctx = {
            .src = (const uint16_t *)(model->map + tensor->abs_offset),
            .dst = (uint16_t *)(shadow + tensor->abs_offset),
        };
        if (parallel_for) {
            parallel_for(parallel_ctx,
                         tensor->elements,
                         lgn_dflash_convert_bf16_rows,
                         &ctx,
                         UINT64_C(1) << 18);
        } else {
            lgn_dflash_convert_bf16_rows(&ctx, 0, tensor->elements);
        }
        converted += tensor->bytes;
    }

    fprintf(stderr,
            "ds4: DFlash BF16 support converted to F16 in %.2f s "
            "(%.2f GiB matrices + %.2f MiB F32 metadata weights)\n",
            lgn_dflash_now_sec() - t0,
            (double)converted / 1073741824.0,
            (double)copied / 1048576.0);
    return shadow;
}

void lgn_dflash_release_f16_map(void *map, uint64_t map_size) {
    if (map && map_size != 0 && map_size <= (uint64_t)SIZE_MAX) {
        munmap(map, (size_t)map_size);
    }
}

const void *lgn_dflash_weight_map(const ds4_model *model,
                                  const void *f16_map) {
    return f16_map ? f16_map : (model ? model->map : NULL);
}

uint64_t lgn_dflash_weight_map_size(const ds4_model *model,
                                    const void *f16_map,
                                    uint64_t f16_map_size) {
    return f16_map ? f16_map_size : (model ? model->size : 0);
}

int lgn_dflash_weight_map_fd(const ds4_model *model,
                             const void *f16_map) {
    return f16_map ? -1 : (model ? model->fd : -1);
}
