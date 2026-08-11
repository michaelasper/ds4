#include "../lgn.h"
#include "../lgn_dflash.h"
#include "../lgn_model.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(condition, message) do {                                      \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL: %s\n", message);                            \
        failures++;                                                          \
    }                                                                         \
} while (0)

static void test_architecture(void) {
    CHECK(lgn_architecture_is_supported("laguna", 6),
          "literal Laguna architecture is accepted");
    CHECK(!lgn_architecture_is_supported(NULL, 0),
          "missing architecture is rejected");
    CHECK(!lgn_architecture_is_supported("", 0),
          "empty architecture is rejected");
    CHECK(!lgn_architecture_is_supported("Laguna", 6),
          "architecture matching is case-sensitive");
    CHECK(!lgn_architecture_is_supported("laguna ", 7),
          "architecture whitespace is rejected");
    CHECK(!lgn_architecture_is_supported("deepseek4", 9),
          "DeepSeek architecture is rejected");
    CHECK(!lgn_architecture_is_supported("glm-dsa", 7),
          "GLM architecture is rejected");
}

static void test_ladder_parser(void) {
    const uint64_t all_expected =
        (UINT64_C(1) << 0) |
        (UINT64_C(1) << 1) |
        (UINT64_C(1) << 7) |
        (UINT64_C(1) << 15) |
        (UINT64_C(1) << 23) |
        (UINT64_C(1) << 31) |
        (UINT64_C(1) << 39) |
        (UINT64_C(1) << 47);
    uint64_t mask = UINT64_MAX;

    CHECK(lgn_decode_ladder_parse(NULL, 48, &mask), "NULL ladder parses");
    CHECK(mask == 0, "NULL ladder clears mask");
    mask = UINT64_MAX;
    CHECK(lgn_decode_ladder_parse("", 48, &mask), "empty ladder parses");
    CHECK(mask == 0, "empty ladder clears mask");
    CHECK(lgn_decode_ladder_parse("7,15,23,31,39,47", 48, &mask),
          "valid sparse ladder parses");
    CHECK(mask == (all_expected & ~UINT64_C(0x3)),
          "valid sparse ladder mask");
    CHECK(lgn_decode_ladder_parse("0,1,7,15,23,31,39,47", 48, &mask),
          "valid full ladder parses");
    CHECK(mask == all_expected, "valid full ladder mask");
    CHECK(lgn_decode_ladder_parse("1,7,15,23,31,39", 48, &mask),
          "valid partial ladder parses");
    CHECK(mask == (all_expected & ~UINT64_C(0x800000000001)),
          "valid partial ladder mask");
    CHECK(lgn_decode_ladder_parse("47", 48, &mask), "last layer parses");
    CHECK(mask == (UINT64_C(1) << 47), "last layer mask");
    CHECK(lgn_decode_ladder_parse("0,63", 64, &mask),
          "64-layer boundary parses");
    CHECK(mask == ((UINT64_C(1) << 0) | (UINT64_C(1) << 63)),
          "64-layer boundary mask");

    mask = UINT64_MAX;
    CHECK(lgn_decode_ladder_parse(NULL, 65, &mask),
          "NULL ladder ignores layer-count limit");
    CHECK(mask == 0, "NULL ladder clears mask above limit");
    mask = UINT64_MAX;
    CHECK(lgn_decode_ladder_parse("", 79, &mask),
          "empty ladder ignores layer-count limit");
    CHECK(mask == 0, "empty ladder clears mask above limit");

    static const char *invalid[] = {
        "7,7", "15,7", "48", "-1", "1, 7", "1,,7", "1,7,",
        "1,x", "1,7\n", "18446744073709551616",
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        mask = UINT64_MAX;
        CHECK(!lgn_decode_ladder_parse(invalid[i], 48, &mask),
              "invalid ladder is rejected");
        CHECK(mask == 0, "invalid ladder clears mask");
    }
    mask = UINT64_MAX;
    CHECK(!lgn_decode_ladder_parse("0", 0, &mask),
          "zero-layer nonempty ladder is rejected");
    CHECK(mask == 0, "zero-layer rejection clears mask");
    mask = UINT64_MAX;
    CHECK(!lgn_decode_ladder_parse("0", 65, &mask),
          "nonempty ladder above mask limit is rejected");
    CHECK(mask == 0, "above-limit rejection clears mask");
    CHECK(!lgn_decode_ladder_parse("0", 48, NULL),
          "NULL output is rejected");
}

static void test_ladder_formatter(void) {
    const uint64_t mask =
        (UINT64_C(1) << 0) |
        (UINT64_C(1) << 1) |
        (UINT64_C(1) << 7) |
        (UINT64_C(1) << 15) |
        (UINT64_C(1) << 23) |
        (UINT64_C(1) << 31) |
        (UINT64_C(1) << 39) |
        (UINT64_C(1) << 47);
    char out[128];
    CHECK(lgn_decode_ladder_format(mask, 48, out, sizeof(out)),
          "valid ladder formats");
    CHECK(strcmp(out, "0,1,7,15,23,31,39,47") == 0,
          "ladder formatter is canonical");
    CHECK(lgn_decode_ladder_format(0, 48, out, sizeof(out)),
          "empty ladder formats");
    CHECK(strcmp(out, "") == 0, "empty ladder formatter output");
    char short_out[4];
    CHECK(!lgn_decode_ladder_format(mask, 48, short_out, sizeof(short_out)),
          "short ladder output is rejected");
    CHECK(!lgn_decode_ladder_format(mask, 65, out, sizeof(out)),
          "formatter rejects more than 64 layers");
    CHECK(!lgn_decode_ladder_format(mask, 48, NULL, sizeof(out)),
          "formatter rejects NULL output");
    CHECK(!lgn_decode_ladder_format(mask, 48, out, 0),
          "formatter rejects zero output size");
}

static void test_s21_topology(void) {
    CHECK(LGN_LAYER_COUNT == 48u, "S2.1 has 48 layers");
    CHECK(LGN_GLOBAL_HEAD_COUNT == 48u, "S2.1 global layers have 48 heads");
    CHECK(LGN_SWA_HEAD_COUNT == 72u, "S2.1 SWA layers have 72 heads");
    for (uint32_t il = 0; il < LGN_LAYER_COUNT; il++) {
        const bool global = (il % 4u) == 0;
        CHECK(lgn_layer_head_count(il) == (global ? 48u : 72u),
              "S2.1 layer head count");
        CHECK(lgn_layer_is_swa(il) == !global, "S2.1 SWA topology");
    }
    CHECK(lgn_layer_head_count(LGN_LAYER_COUNT) == 0,
          "out-of-range topology returns zero");
    CHECK(!lgn_layer_is_swa(LGN_LAYER_COUNT),
          "out-of-range topology is not SWA");
}

static void test_s21_model_profile(void) {
    const ds4_shape *shape = lgn_model_shape();
    CHECK(shape != NULL, "Laguna model profile is available");
    CHECK(shape->family == DS4_MODEL_FAMILY_LAGUNA,
          "Laguna profile selects the Laguna family");
    CHECK(shape->variant == DS4_VARIANT_LAGUNA_S21,
          "Laguna profile selects S2.1");
    CHECK(shape->n_layer == LGN_LAYER_COUNT && shape->n_embd == 3072u,
          "Laguna profile dimensions");
    CHECK(shape->n_vocab == 100352u && shape->n_expert == 256u,
          "Laguna profile vocabulary and expert count");
    CHECK(shape->n_head_kv == 8u && shape->n_head_dim == 128u,
          "Laguna profile attention dimensions");
    CHECK(shape->n_ff_dense == 12288u && shape->n_ff_exp == 1024u,
          "Laguna profile feed-forward dimensions");
    CHECK(shape->context_length == UINT64_C(262144) &&
              shape->rope_orig_ctx == UINT64_C(8192),
          "Laguna profile context limits");
    CHECK(lgn_model_layer_head_count(0) == LGN_GLOBAL_HEAD_COUNT &&
              lgn_model_layer_head_count(1) == LGN_SWA_HEAD_COUNT,
          "private Laguna topology adapter");
    CHECK(!lgn_model_layer_is_swa(0) && lgn_model_layer_is_swa(1),
          "private Laguna SWA topology adapter");

    ds4_tensor output_norm = { .type = 0 };
    ds4_tensor output = { .type = 1 };
    ds4_weights weights;
    memset(&weights, 0, sizeof(weights));
    CHECK(!lgn_weights_have_output_head(&weights),
          "empty weights do not have an output head");
    weights.output_norm = &output_norm;
    CHECK(lgn_weights_have_partial_output_head(&weights) &&
              !lgn_weights_have_output_head(&weights),
          "one output tensor is a partial output head");
    weights.output = &output;
    CHECK(lgn_weights_have_output_head(&weights),
          "paired output tensors form an output head");
}

static bool synthetic_tensor_add(ds4_tensor *tensors,
                                 size_t *n_tensors,
                                 size_t capacity,
                                 const char *name) {
    if (!tensors || !n_tensors || *n_tensors >= capacity || !name) {
        return false;
    }
    const size_t length = strlen(name);
    char *copy = malloc(length + 1u);
    if (!copy) return false;
    memcpy(copy, name, length + 1u);
    tensors[*n_tensors].name.ptr = copy;
    tensors[*n_tensors].name.len = length;
    (*n_tensors)++;
    return true;
}

static void test_whole_model_weight_bind(void) {
    static const char *const common_suffixes[] = {
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_gate.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "attn_output.weight",
        "ffn_norm.weight",
    };
    static const char *const dense_suffixes[] = {
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    };
    static const char *const routed_suffixes[] = {
        "ffn_gate_inp.weight", "exp_probs_b.bias",
        "ffn_gate_exps.weight", "ffn_up_exps.weight",
        "ffn_down_exps.weight", "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    };
    const uint32_t n_layer = lgn_model_shape()->n_layer;
    const size_t expected = 3u +
        sizeof(common_suffixes) / sizeof(common_suffixes[0]) +
        sizeof(dense_suffixes) / sizeof(dense_suffixes[0]) +
        (size_t)(n_layer - 1u) *
            (sizeof(common_suffixes) / sizeof(common_suffixes[0]) +
             sizeof(routed_suffixes) / sizeof(routed_suffixes[0]));
    ds4_tensor *tensors = calloc(expected, sizeof(*tensors));
    CHECK(tensors != NULL, "whole-model bind fixture allocates tensor table");
    if (!tensors) return;

    size_t n_tensors = 0;
    bool ok = synthetic_tensor_add(tensors, &n_tensors, expected,
                                   "token_embd.weight") &&
              synthetic_tensor_add(tensors, &n_tensors, expected,
                                   "output_norm.weight") &&
              synthetic_tensor_add(tensors, &n_tensors, expected,
                                   "output.weight");
    char name[96];
    for (uint32_t il = 0; ok && il < n_layer; il++) {
        for (size_t i = 0;
             ok && i < sizeof(common_suffixes) / sizeof(common_suffixes[0]);
             i++) {
            snprintf(name, sizeof(name), "blk.%u.%s", il, common_suffixes[i]);
            ok = synthetic_tensor_add(tensors, &n_tensors, expected, name);
        }
        const char *const *suffixes = il == 0u ? dense_suffixes : routed_suffixes;
        const size_t suffix_count = il == 0u ?
            sizeof(dense_suffixes) / sizeof(dense_suffixes[0]) :
            sizeof(routed_suffixes) / sizeof(routed_suffixes[0]);
        for (size_t i = 0; ok && i < suffix_count; i++) {
            snprintf(name, sizeof(name), "blk.%u.%s", il, suffixes[i]);
            ok = synthetic_tensor_add(tensors, &n_tensors, expected, name);
        }
    }
    CHECK(ok && n_tensors == expected,
          "whole-model bind fixture covers every executable layer tensor");

    ds4_model model = {
        .n_tensors = n_tensors,
        .tensors = tensors,
    };
    ds4_weights weights;
    memset(&weights, 0, sizeof(weights));
    if (ok) lgn_weights_bind(&weights, &model);
    CHECK(ok && weights.token_embd && weights.output_norm && weights.output,
          "whole-model bind includes embeddings and output head");
    bool all_layers_bound = ok;
    for (uint32_t il = 0; all_layers_bound && il < n_layer; il++) {
        const ds4_layer_weights *layer = &weights.layer[il];
        all_layers_bound = layer->attn_norm && layer->attn_q &&
            layer->attn_k && layer->attn_v && layer->attn_gate &&
            layer->attn_q_norm && layer->attn_k_norm && layer->attn_output &&
            layer->ffn_norm;
        if (il == 0u) {
            all_layers_bound = all_layers_bound && layer->ffn_gate &&
                layer->ffn_up && layer->ffn_down;
        } else {
            all_layers_bound = all_layers_bound && layer->ffn_gate_inp &&
                layer->ffn_exp_probs_b && layer->ffn_gate_exps &&
                layer->ffn_up_exps && layer->ffn_down_exps &&
                layer->ffn_gate_shexp && layer->ffn_up_shexp &&
                layer->ffn_down_shexp;
        }
    }
    CHECK(all_layers_bound,
          "whole-model bind populates all 48 layer records");

    for (size_t i = 0; i < n_tensors; i++) {
        free((void *)tensors[i].name.ptr);
    }
    free(tensors);
}

static void test_model_admission(void) {
    static const char key[] = "general.architecture";
    uint8_t value[32] = {0};
    uint64_t length = 6;
    memcpy(value, &length, sizeof(length));
    memcpy(value + sizeof(length), "laguna", 6);

    ds4_kv kv = {
        .key = { key, sizeof(key) - 1u },
        .type = 8u, /* GGUF_VALUE_STRING */
        .value_pos = 0,
    };
    ds4_model model = {
        .map = value,
        .size = sizeof(value),
        .n_kv = 1,
        .kv = &kv,
    };
    ds4_str arch = {0};
    CHECK(lgn_model_is_laguna(&model, &arch),
          "Laguna model admission accepts literal architecture");
    CHECK(arch.len == 6u && memcmp(arch.ptr, "laguna", 6) == 0,
          "Laguna model admission returns architecture span");

    memcpy(value + sizeof(length), "glm-dsa", 7);
    length = 7;
    memcpy(value, &length, sizeof(length));
    CHECK(!lgn_model_is_laguna(&model, NULL),
          "Laguna model admission rejects non-Laguna architecture");
    CHECK(!lgn_model_is_laguna(NULL, NULL),
          "Laguna model admission rejects a missing model");
}

static void test_dflash_profile_and_binding(void) {
    const lgn_dflash_profile *profile = lgn_dflash_profile_get();
    CHECK(profile != NULL, "DFlash profile is available");
    CHECK(lgn_dflash_profile_matches_laguna(),
          "DFlash decoder profile matches Laguna dimensions");
    CHECK(profile->n_layer == 6u && profile->n_aux == 6u,
          "DFlash profile layer and auxiliary counts");
    CHECK(profile->block_size == 16u && profile->cache_cap == 512u,
          "DFlash profile block and cache limits");
    CHECK(profile->n_embd == 3072u && profile->n_head == 72u &&
              profile->n_head_kv == 8u && profile->n_head_dim == 128u,
          "DFlash profile attention dimensions");
    CHECK(profile->context_length == UINT64_C(1048576) &&
              profile->mask_token_id == 12u,
          "DFlash profile context and mask token");
    static const uint32_t expected_targets[] = { 2u, 11u, 20u, 30u, 39u, 48u };
    CHECK(memcmp(profile->target_layers,
                 expected_targets,
                 sizeof(expected_targets)) == 0,
          "DFlash profile target layer list");

    ds4_tensor tensor = {0};
    lgn_dflash_weights weights;
    memset(&weights, 0, sizeof(weights));
    weights.aux_norm = &tensor;
    weights.fc = &tensor;
    weights.encoder_output_norm = &tensor;
    weights.output_norm = &tensor;
    weights.block_size = profile->block_size;
    weights.mask_token_id = profile->mask_token_id;
    memcpy(weights.target_layers,
           profile->target_layers,
           sizeof(weights.target_layers));
    for (uint32_t il = 0; il < profile->n_layer; il++) {
        lgn_dflash_layer_weights *layer = &weights.layer[il];
        layer->attn_norm = &tensor;
        layer->attn_q = &tensor;
        layer->attn_k = &tensor;
        layer->attn_v = &tensor;
        layer->attn_gate = &tensor;
        layer->attn_q_norm = &tensor;
        layer->attn_k_norm = &tensor;
        layer->attn_output = &tensor;
        layer->ffn_norm = &tensor;
        layer->ffn_gate = &tensor;
        layer->ffn_up = &tensor;
        layer->ffn_down = &tensor;
    }
    CHECK(lgn_dflash_binding_eligible(&weights),
          "complete DFlash binding is eligible");
    weights.layer[3].ffn_up = NULL;
    CHECK(!lgn_dflash_binding_eligible(&weights),
          "incomplete DFlash layer binding is rejected");
}

enum {
    TEST_DFLASH_KV_CAP = 20u,
    TEST_DFLASH_TENSOR_CAP = 4u + LGN_DFLASH_N_LAYER * 12u,
    TEST_DFLASH_STORAGE = 8192u,
};

typedef struct {
    uint8_t map[TEST_DFLASH_STORAGE];
    size_t cursor;
    ds4_kv kv[TEST_DFLASH_KV_CAP];
    uint32_t n_kv;
    ds4_tensor tensors[TEST_DFLASH_TENSOR_CAP];
    char tensor_names[TEST_DFLASH_TENSOR_CAP][96];
    uint32_t n_tensors;
    ds4_model model;
} test_dflash_bind_fixture;

static uint64_t test_dflash_reserve(test_dflash_bind_fixture *fixture,
                                    size_t bytes) {
    if (!fixture || bytes > sizeof(fixture->map) - fixture->cursor) {
        return UINT64_MAX;
    }
    const uint64_t pos = (uint64_t)fixture->cursor;
    fixture->cursor += bytes;
    return pos;
}

static void test_dflash_put(test_dflash_bind_fixture *fixture,
                            const void *data,
                            size_t bytes,
                            uint64_t *pos_out) {
    const uint64_t pos = test_dflash_reserve(fixture, bytes);
    CHECK(pos != UINT64_MAX, "DFlash fixture storage has room");
    if (pos == UINT64_MAX) {
        *pos_out = 0;
        return;
    }
    memcpy(fixture->map + pos, data, bytes);
    *pos_out = pos;
}

static void test_dflash_add_kv(test_dflash_bind_fixture *fixture,
                               const char *key,
                               uint32_t type,
                               uint64_t value_pos) {
    CHECK(fixture->n_kv < TEST_DFLASH_KV_CAP,
          "DFlash fixture metadata table has room");
    if (fixture->n_kv >= TEST_DFLASH_KV_CAP) return;
    fixture->kv[fixture->n_kv++] = (ds4_kv){
        .key = { key, strlen(key) },
        .type = type,
        .value_pos = value_pos,
    };
}

static void test_dflash_add_u32(test_dflash_bind_fixture *fixture,
                                const char *key,
                                uint32_t value) {
    uint64_t pos = 0;
    test_dflash_put(fixture, &value, sizeof(value), &pos);
    test_dflash_add_kv(fixture, key, LGN_GGUF_VALUE_UINT32, pos);
}

static void test_dflash_add_u64(test_dflash_bind_fixture *fixture,
                                const char *key,
                                uint64_t value) {
    uint64_t pos = 0;
    test_dflash_put(fixture, &value, sizeof(value), &pos);
    test_dflash_add_kv(fixture, key, LGN_GGUF_VALUE_UINT64, pos);
}

static void test_dflash_add_f32(test_dflash_bind_fixture *fixture,
                                const char *key,
                                float value) {
    uint64_t pos = 0;
    test_dflash_put(fixture, &value, sizeof(value), &pos);
    test_dflash_add_kv(fixture, key, LGN_GGUF_VALUE_FLOAT32, pos);
}

static void test_dflash_add_string(test_dflash_bind_fixture *fixture,
                                   const char *key,
                                   const char *value) {
    const uint64_t length = strlen(value);
    const size_t bytes = sizeof(length) + (size_t)length;
    const uint64_t pos = test_dflash_reserve(fixture, bytes);
    CHECK(pos != UINT64_MAX, "DFlash fixture string has room");
    if (pos == UINT64_MAX) return;
    memcpy(fixture->map + pos, &length, sizeof(length));
    memcpy(fixture->map + pos + sizeof(length), value, (size_t)length);
    test_dflash_add_kv(fixture, key, LGN_GGUF_VALUE_STRING, pos);
}

static void test_dflash_add_array(test_dflash_bind_fixture *fixture,
                                  const char *key,
                                  uint32_t element_type,
                                  const uint32_t *values,
                                  size_t count) {
    const uint64_t length = (uint64_t)count;
    const size_t bytes = sizeof(element_type) + sizeof(length) +
                         count * sizeof(values[0]);
    const uint64_t pos = test_dflash_reserve(fixture, bytes);
    CHECK(pos != UINT64_MAX, "DFlash fixture array has room");
    if (pos == UINT64_MAX) return;
    memcpy(fixture->map + pos, &element_type, sizeof(element_type));
    memcpy(fixture->map + pos + sizeof(element_type), &length, sizeof(length));
    memcpy(fixture->map + pos + sizeof(element_type) + sizeof(length),
           values,
           count * sizeof(values[0]));
    test_dflash_add_kv(fixture, key, LGN_GGUF_VALUE_ARRAY, pos);
}

static ds4_tensor *test_dflash_add_tensor(test_dflash_bind_fixture *fixture,
                                           const char *name,
                                           uint32_t type,
                                           uint32_t ndim,
                                           uint64_t d0,
                                           uint64_t d1) {
    CHECK(fixture->n_tensors < TEST_DFLASH_TENSOR_CAP,
          "DFlash fixture tensor table has room");
    if (fixture->n_tensors >= TEST_DFLASH_TENSOR_CAP) return NULL;
    const uint32_t index = fixture->n_tensors++;
    const int n = snprintf(fixture->tensor_names[index],
                           sizeof(fixture->tensor_names[index]),
                           "%s", name);
    CHECK(n >= 0 && (size_t)n < sizeof(fixture->tensor_names[index]),
          "DFlash fixture tensor name has room");
    ds4_tensor *tensor = &fixture->tensors[index];
    tensor->name = (ds4_str){ fixture->tensor_names[index], (size_t)n };
    tensor->ndim = ndim;
    tensor->dim[0] = d0;
    tensor->dim[1] = d1;
    tensor->type = type;
    return tensor;
}

static void test_dflash_bind_fixture_init(test_dflash_bind_fixture *fixture) {
    memset(fixture, 0, sizeof(*fixture));
    fixture->cursor = 64u;
    const lgn_dflash_profile *profile = lgn_dflash_profile_get();
    static const uint32_t all_one[] = { 1u, 1u, 1u, 1u, 1u, 1u };

    test_dflash_add_u32(fixture, "dflash.block_count", profile->n_layer);
    test_dflash_add_u64(fixture, "dflash.context_length",
                        profile->context_length);
    test_dflash_add_u32(fixture, "dflash.embedding_length", profile->n_embd);
    test_dflash_add_u32(fixture, "dflash.feed_forward_length",
                        profile->n_ff_dense);
    test_dflash_add_u32(fixture, "dflash.attention.head_count",
                        profile->n_head);
    test_dflash_add_u32(fixture, "dflash.attention.head_count_kv",
                        profile->n_head_kv);
    test_dflash_add_u32(fixture, "dflash.attention.key_length",
                        profile->n_head_dim);
    test_dflash_add_u32(fixture, "dflash.attention.value_length",
                        profile->n_value_dim);
    test_dflash_add_u32(fixture, "dflash.rope.dimension_count", profile->n_rot);
    test_dflash_add_u32(fixture, "dflash.attention.sliding_window",
                        profile->cache_cap);
    test_dflash_add_array(fixture,
                          "dflash.attention.sliding_window_pattern",
                          LGN_GGUF_VALUE_UINT32,
                          all_one,
                          sizeof(all_one) / sizeof(all_one[0]));
    test_dflash_add_f32(fixture, "dflash.rope.freq_base",
                        profile->rope_freq_base);
    test_dflash_add_f32(fixture,
                        "dflash.attention.layer_norm_rms_epsilon",
                        profile->rms_eps);
    test_dflash_add_u32(fixture, "dflash.block_size", profile->block_size);
    test_dflash_add_u32(fixture, "tokenizer.ggml.mask_token_id",
                        profile->mask_token_id);
    test_dflash_add_array(fixture,
                          "dflash.target_layers",
                          LGN_GGUF_VALUE_UINT32,
                          profile->target_layers,
                          profile->n_aux);
    test_dflash_add_string(fixture, "dflash.decoder_arch", "laguna");
    test_dflash_add_string(fixture, "dflash.rope.scaling.type", "none");

    static const uint32_t malformed[] = { 0u, 1u, 2u, 3u, 4u, 5u, 6u };
    test_dflash_add_array(fixture,
                          "test.bad.array",
                          LGN_GGUF_VALUE_UINT32,
                          malformed,
                          sizeof(malformed) / sizeof(malformed[0]));
    const uint64_t truncated_pos = test_dflash_reserve(
        fixture, sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t));
    CHECK(truncated_pos != UINT64_MAX, "DFlash truncated array has room");
    if (truncated_pos != UINT64_MAX) {
        const uint32_t type = LGN_GGUF_VALUE_UINT32;
        const uint64_t length = 2u;
        memcpy(fixture->map + truncated_pos, &type, sizeof(type));
        memcpy(fixture->map + truncated_pos + sizeof(type),
               &length,
               sizeof(length));
        memcpy(fixture->map + truncated_pos + sizeof(type) + sizeof(length),
               &all_one[0],
               sizeof(all_one[0]));
        test_dflash_add_kv(fixture,
                           "test.truncated.array",
                           LGN_GGUF_VALUE_ARRAY,
                           truncated_pos);
    }

    const uint64_t q_dim = (uint64_t)profile->n_head * profile->n_head_dim;
    const uint64_t kv_dim = (uint64_t)profile->n_head_kv * profile->n_head_dim;
    test_dflash_add_tensor(fixture, "enc.aux_norm.weight", LGN_TENSOR_F32,
                           2u, profile->n_embd, profile->n_aux);
    test_dflash_add_tensor(fixture, "fc.weight", LGN_TENSOR_BF16,
                           2u, (uint64_t)profile->n_aux * profile->n_embd,
                           profile->n_embd);
    test_dflash_add_tensor(fixture, "enc.output_norm.weight", LGN_TENSOR_F32,
                           1u, profile->n_embd, 0);
    test_dflash_add_tensor(fixture, "output_norm.weight", LGN_TENSOR_F32,
                           1u, profile->n_embd, 0);
    for (uint32_t il = 0; il < profile->n_layer; il++) {
        char name[96];
#define ADD_DFLASH_TENSOR(format, type, ndim, d0, d1) do {                  \
            snprintf(name, sizeof(name), format, il);                     \
            test_dflash_add_tensor(fixture, name, type, ndim, d0, d1);     \
        } while (0)
        ADD_DFLASH_TENSOR("blk.%u.attn_norm.weight", LGN_TENSOR_F32,
                          1u, profile->n_embd, 0);
        ADD_DFLASH_TENSOR("blk.%u.attn_q.weight", LGN_TENSOR_BF16,
                          2u, profile->n_embd, q_dim);
        ADD_DFLASH_TENSOR("blk.%u.attn_k.weight", LGN_TENSOR_BF16,
                          2u, profile->n_embd, kv_dim);
        ADD_DFLASH_TENSOR("blk.%u.attn_v.weight", LGN_TENSOR_BF16,
                          2u, profile->n_embd, kv_dim);
        ADD_DFLASH_TENSOR("blk.%u.attn_gate.weight", LGN_TENSOR_BF16,
                          2u, profile->n_embd, profile->n_head);
        ADD_DFLASH_TENSOR("blk.%u.attn_q_norm.weight", LGN_TENSOR_F32,
                          1u, profile->n_head_dim, 0);
        ADD_DFLASH_TENSOR("blk.%u.attn_k_norm.weight", LGN_TENSOR_F32,
                          1u, profile->n_head_dim, 0);
        ADD_DFLASH_TENSOR("blk.%u.attn_output.weight", LGN_TENSOR_BF16,
                          2u, q_dim, profile->n_embd);
        ADD_DFLASH_TENSOR("blk.%u.ffn_norm.weight", LGN_TENSOR_F32,
                          1u, profile->n_embd, 0);
        ADD_DFLASH_TENSOR("blk.%u.ffn_gate.weight", LGN_TENSOR_BF16,
                          2u, profile->n_embd, profile->n_ff_dense);
        ADD_DFLASH_TENSOR("blk.%u.ffn_up.weight", LGN_TENSOR_BF16,
                          2u, profile->n_embd, profile->n_ff_dense);
        ADD_DFLASH_TENSOR("blk.%u.ffn_down.weight", LGN_TENSOR_BF16,
                          2u, profile->n_ff_dense, profile->n_embd);
#undef ADD_DFLASH_TENSOR
    }
    fixture->model = (ds4_model){
        .map = fixture->map,
        .size = sizeof(fixture->map),
        .n_kv = fixture->n_kv,
        .n_tensors = fixture->n_tensors,
        .kv = fixture->kv,
        .tensors = fixture->tensors,
    };
}

static void test_dflash_binding_fixture(void) {
    test_dflash_bind_fixture fixture;
    test_dflash_bind_fixture_init(&fixture);
    CHECK(fixture.n_kv == TEST_DFLASH_KV_CAP,
          "synthetic DFlash fixture contains all metadata records");
    CHECK(fixture.n_tensors == TEST_DFLASH_TENSOR_CAP,
          "synthetic DFlash fixture contains all tensor names");

    lgn_dflash_weights weights;
    memset(&weights, 0, sizeof(weights));
    lgn_dflash_weights_bind(&weights, &fixture.model);
    CHECK(lgn_dflash_binding_eligible(&weights),
          "synthetic DFlash GGUF binds every required tensor and metadata");
    CHECK(weights.fc == &fixture.tensors[1] &&
              weights.layer[LGN_DFLASH_N_LAYER - 1u].ffn_down != NULL,
          "synthetic DFlash binding retains expected tensor pointers");

    uint32_t values[LGN_DFLASH_N_AUX] = {0};
    uint32_t count = 0;
    CHECK(!lgn_model_get_u32_array(&fixture.model,
                                   "test.bad.array",
                                   values,
                                   LGN_DFLASH_N_AUX,
                                   &count) && count == 0,
          "DFlash metadata array rejects oversized input");
    ds4_model truncated = fixture.model;
    truncated.size = fixture.cursor;
    CHECK(!lgn_model_get_u32_array(&truncated,
                                   "test.truncated.array",
                                   values,
                                   LGN_DFLASH_N_AUX,
                                   &count) && count == 0,
          "DFlash metadata array rejects truncated input");
    CHECK(!lgn_model_get_string(&fixture.model,
                                "dflash.decoder_arch",
                                NULL),
          "model string getter rejects NULL output");
    CHECK(!lgn_model_get_u32(&fixture.model,
                             "dflash.block_count",
                             NULL),
          "model u32 getter rejects NULL output");
    CHECK(!lgn_model_get_token_id(&fixture.model,
                                  "tokenizer.ggml.mask_token_id",
                                  NULL),
          "model token getter rejects NULL output");
    CHECK(!lgn_model_get_u64_compat(&fixture.model,
                                    "dflash.context_length",
                                    NULL),
          "model u64 getter rejects NULL output");
    CHECK(!lgn_model_get_f32_compat(&fixture.model,
                                    "dflash.rope.freq_base",
                                    NULL),
          "model f32 getter rejects NULL output");
    CHECK(!lgn_model_get_bool(&fixture.model, "dflash.decoder_arch", NULL),
          "model bool getter rejects NULL output");
    CHECK(!lgn_model_get_array(&fixture.model,
                               "dflash.target_layers",
                               NULL),
          "model array getter rejects NULL output");
    CHECK(!lgn_model_get_u32_array(&fixture.model,
                                   "dflash.target_layers",
                                   NULL,
                                   LGN_DFLASH_N_AUX,
                                   &count),
          "model u32 array getter rejects NULL storage");

    ds4_model missing_kv = fixture.model;
    missing_kv.kv = NULL;
    CHECK(!lgn_model_get_u32(&missing_kv,
                             "dflash.block_count",
                             &count),
          "model getter rejects a nonzero KV count with NULL table");
    ds4_model missing_storage = fixture.model;
    missing_storage.map = NULL;
    CHECK(!lgn_model_get_u32(&missing_storage,
                             "dflash.block_count",
                             &count),
          "model getter rejects a missing backing storage mapping");
    ds4_model missing_tensors = fixture.model;
    missing_tensors.tensors = NULL;
    CHECK(lgn_model_find_tensor(&missing_tensors, "fc.weight") == NULL,
          "tensor finder rejects a nonzero tensor count with NULL table");
}

typedef struct {
    uint32_t calls;
    uint64_t n_rows;
    uint64_t min_parallel_rows;
    uint64_t begin[4];
    uint64_t end[4];
    uint32_t ranges;
} test_dflash_parallel_capture;

static void test_dflash_parallel_for(void *parallel_ctx,
                                     uint64_t n_rows,
                                     lgn_dflash_range_fn fn,
                                     void *ctx,
                                     uint64_t min_parallel_rows) {
    test_dflash_parallel_capture *capture = parallel_ctx;
    capture->calls++;
    capture->n_rows = n_rows;
    capture->min_parallel_rows = min_parallel_rows;
    const uint64_t split = n_rows / 2u;
    capture->begin[capture->ranges] = 0;
    capture->end[capture->ranges++] = split;
    fn(ctx, 0, split);
    capture->begin[capture->ranges] = split;
    capture->end[capture->ranges++] = n_rows;
    fn(ctx, split, n_rows);
}

static void test_dflash_shadow_map(void) {
    static const uint16_t bf16_input[] = {
        0x0000u, 0x3f80u, 0xbf80u, 0x3f81u,
        0x7f80u, 0x0001u, 0x8000u, 0x3c00u,
    };
    static const uint16_t expected_f16[] = {
        0x0000u, 0x3c00u, 0xbc00u, 0x3c08u,
        0x7c00u, 0x0000u, 0x8000u, 0x2000u,
    };
    uint8_t source[128] = {0};
    ds4_tensor tensors[2] = {0};
    char names[2][16] = { "f32", "bf16" };
    uint32_t f32_bits[] = { UINT32_C(0x11223344), UINT32_C(0x55667788) };
    memcpy(source + 16u, f32_bits, sizeof(f32_bits));
    memcpy(source + 64u, bf16_input, sizeof(bf16_input));
    tensors[0] = (ds4_tensor){
        .name = { names[0], 3u },
        .type = LGN_TENSOR_F32,
        .abs_offset = 16u,
        .elements = 2u,
        .bytes = sizeof(f32_bits),
    };
    tensors[1] = (ds4_tensor){
        .name = { names[1], 4u },
        .type = LGN_TENSOR_BF16,
        .abs_offset = 64u,
        .elements = sizeof(bf16_input) / sizeof(bf16_input[0]),
        .bytes = sizeof(bf16_input),
    };
    ds4_model model = {
        .map = source,
        .size = sizeof(source),
        .n_tensors = 2u,
        .tensors = tensors,
    };

    void *serial_map = lgn_dflash_prepare_f16_map(&model, NULL, NULL);
    CHECK(serial_map != NULL, "DFlash BF16 serial shadow map is created");
    if (serial_map) {
        CHECK(memcmp((const uint8_t *)serial_map + 16u,
                     f32_bits,
                     sizeof(f32_bits)) == 0,
              "DFlash F32 metadata bytes are copied unchanged");
        CHECK(memcmp((const uint8_t *)serial_map + 64u,
                     expected_f16,
                     sizeof(expected_f16)) == 0,
              "DFlash serial BF16 conversion matches F16 bit patterns");
        lgn_dflash_release_f16_map(serial_map, model.size);
    }

    test_dflash_parallel_capture capture = {0};
    void *parallel_map = lgn_dflash_prepare_f16_map(
        &model, test_dflash_parallel_for, &capture);
    CHECK(parallel_map != NULL, "DFlash injected shadow map is created");
    CHECK(capture.calls == 1u &&
              capture.n_rows == sizeof(bf16_input) / sizeof(bf16_input[0]) &&
              capture.min_parallel_rows == (UINT64_C(1) << 18) &&
              capture.ranges == 2u && capture.begin[0] == 0u &&
              capture.end[0] == 4u && capture.begin[1] == 4u &&
              capture.end[1] == 8u,
          "DFlash BF16 conversion invokes the injected parallel adapter");
    if (parallel_map) {
        CHECK(memcmp((const uint8_t *)parallel_map + 64u,
                     expected_f16,
                     sizeof(expected_f16)) == 0,
              "DFlash injected BF16 conversion matches F16 bit patterns");
        lgn_dflash_release_f16_map(parallel_map, model.size);
    }

    tensors[1].abs_offset = 120u;
    CHECK(lgn_dflash_prepare_f16_map(&model, NULL, NULL) == NULL,
          "DFlash shadow map rejects tensor ranges outside the mapping");
    tensors[1].abs_offset = 64u;
    tensors[1].type = LGN_TENSOR_Q4_0;
    CHECK(lgn_dflash_prepare_f16_map(&model, NULL, NULL) == NULL,
          "DFlash shadow map rejects unsupported tensor types");
    tensors[1].type = LGN_TENSOR_BF16;
    tensors[1].bytes--;
    CHECK(lgn_dflash_prepare_f16_map(&model, NULL, NULL) == NULL,
          "DFlash shadow map rejects malformed BF16 byte counts");
    tensors[1].bytes = sizeof(bf16_input);
    tensors[1].elements = UINT64_MAX;
    tensors[1].bytes = 0;
    CHECK(lgn_dflash_prepare_f16_map(&model, NULL, NULL) == NULL,
          "DFlash shadow map rejects overflowing BF16 element counts");
    tensors[1].elements = sizeof(bf16_input) / sizeof(bf16_input[0]);
    tensors[1].bytes = sizeof(bf16_input);
    ds4_model missing_tensors = model;
    missing_tensors.tensors = NULL;
    CHECK(lgn_dflash_prepare_f16_map(&missing_tensors, NULL, NULL) == NULL,
          "DFlash shadow map rejects a nonzero tensor count with NULL table");
    ds4_model missing_map = model;
    missing_map.map = NULL;
    CHECK(lgn_dflash_prepare_f16_map(&missing_map, NULL, NULL) == NULL,
          "DFlash shadow map rejects a missing model mapping");
}

typedef struct {
    size_t count;
    size_t length[64];
    unsigned char bytes[64][128];
    size_t fail_after;
} span_capture;

typedef struct {
    const unsigned char *bytes;
    size_t length;
} expected_span;

static bool capture_span(const char *piece, size_t piece_len, void *userdata) {
    span_capture *capture = userdata;
    if (capture->count >= capture->fail_after) return false;
    if (capture->count >= sizeof(capture->length) / sizeof(capture->length[0]) ||
        piece_len > sizeof(capture->bytes[0])) {
        return false;
    }
    capture->length[capture->count] = piece_len;
    memcpy(capture->bytes[capture->count], piece, piece_len);
    capture->count++;
    return true;
}

static void check_pretokenized(const char          *name,
                               const char          *text,
                               const expected_span *expected,
                               size_t               expected_count) {
    span_capture capture;
    memset(&capture, 0, sizeof(capture));
    capture.fail_after = (size_t)-1;

    CHECK(lgn_bpe_pretokenize(text, capture_span, &capture), name);
    if (capture.count != expected_count) {
        fprintf(stderr,
                "FAIL: %s (got %zu spans, expected %zu)\n",
                name,
                capture.count,
                expected_count);
        failures++;
        return;
    }
    for (size_t i = 0; i < expected_count; i++) {
        if (capture.length[i] != expected[i].length ||
            memcmp(capture.bytes[i], expected[i].bytes, expected[i].length) != 0) {
            fprintf(stderr, "FAIL: %s (span %zu differs)\n", name, i);
            failures++;
        }
    }
}

#define EXPECT_LITERAL(value) \
    { (const unsigned char *)(value), sizeof(value) - 1u }

static void test_laguna_pretokenizer(void) {
    static const expected_span contractions[] = {
        EXPECT_LITERAL("can"),
        EXPECT_LITERAL("'t"),
        EXPECT_LITERAL("we"),
        EXPECT_LITERAL("'LL"),
        EXPECT_LITERAL("'re"),
        EXPECT_LITERAL("'M"),
        EXPECT_LITERAL("'d"),
    };
    check_pretokenized("ASCII words, contractions, and case folding",
                       "can'twe'LL're'M'd",
                       contractions,
                       sizeof(contractions) / sizeof(contractions[0]));

    static const expected_span words_numbers[] = {
        EXPECT_LITERAL("abc"),
        EXPECT_LITERAL("1"),
        EXPECT_LITERAL("2"),
        EXPECT_LITERAL("3"),
        EXPECT_LITERAL("bar"),
    };
    check_pretokenized("max_digits=1 keeps every ASCII digit separate",
                       "abc123bar",
                       words_numbers,
                       sizeof(words_numbers) / sizeof(words_numbers[0]));

    static const expected_span punctuation[] = {
        EXPECT_LITERAL("hello"),
        EXPECT_LITERAL(","),
        EXPECT_LITERAL(" world"),
        EXPECT_LITERAL("!!"),
    };
    check_pretokenized("punctuation runs and leading-space words",
                       "hello, world!!",
                       punctuation,
                       sizeof(punctuation) / sizeof(punctuation[0]));

    static const expected_span punctuation_before_word[] = {
        EXPECT_LITERAL("!word"),
        EXPECT_LITERAL("?"),
    };
    check_pretokenized("punctuation immediately before a letter joins that word",
                       "!word?",
                       punctuation_before_word,
                       sizeof(punctuation_before_word) /
                           sizeof(punctuation_before_word[0]));

    static const expected_span spaces_before_word[] = {
        EXPECT_LITERAL(" "),
        EXPECT_LITERAL(" abc"),
        EXPECT_LITERAL("  "),
    };
    check_pretokenized("multiple spaces split before a following word",
                       "  abc  ",
                       spaces_before_word,
                       sizeof(spaces_before_word) / sizeof(spaces_before_word[0]));

    static const expected_span spaces_before_punctuation[] = {
        EXPECT_LITERAL(" "),
        EXPECT_LITERAL(" !!"),
        EXPECT_LITERAL("x"),
    };
    check_pretokenized("spaces before punctuation stay with the punctuation run",
                       "  !!x",
                       spaces_before_punctuation,
                       sizeof(spaces_before_punctuation) /
                           sizeof(spaces_before_punctuation[0]));

    static const expected_span newline_runs[] = {
        EXPECT_LITERAL("foo"),
        EXPECT_LITERAL("\n\n"),
        EXPECT_LITERAL("bar"),
    };
    check_pretokenized("LF runs are one top-level span",
                       "foo\n\nbar",
                       newline_runs,
                       sizeof(newline_runs) / sizeof(newline_runs[0]));

    static const expected_span crlf[] = {
        EXPECT_LITERAL("foo"),
        EXPECT_LITERAL("\r"),
        EXPECT_LITERAL("\n"),
        EXPECT_LITERAL("bar"),
    };
    check_pretokenized("CR stays before the top-level LF span",
                       "foo\r\nbar",
                       crlf,
                       sizeof(crlf) / sizeof(crlf[0]));

    static const expected_span punctuation_crlf[] = {
        EXPECT_LITERAL("!\r"),
        EXPECT_LITERAL("\n"),
    };
    check_pretokenized("punctuation absorbs trailing CR but not LF",
                       "!\r\n",
                       punctuation_crlf,
                       sizeof(punctuation_crlf) / sizeof(punctuation_crlf[0]));

    static const expected_span whitespace_crlf[] = {
        EXPECT_LITERAL(" \r"),
        EXPECT_LITERAL("\n"),
    };
    check_pretokenized("whitespace keeps CR with its preceding spaces",
                       " \r\n",
                       whitespace_crlf,
                       sizeof(whitespace_crlf) / sizeof(whitespace_crlf[0]));

    static const expected_span unicode[] = {
        EXPECT_LITERAL("\xC3\xA9"),
        EXPECT_LITERAL("\xC2\xA9!"),
        EXPECT_LITERAL("\xC2\xA0x"),
    };
    check_pretokenized("Unicode letters, punctuation, and whitespace classify correctly",
                       "\xC3\xA9\xC2\xA9!\xC2\xA0x",
                       unicode,
                       sizeof(unicode) / sizeof(unicode[0]));

    static const char unicode_numbers_text[] =
        "\xD9\xA1\xD9\xA2\xEF\xBC\x93\x41";
    static const expected_span unicode_numbers[] = {
        EXPECT_LITERAL("\xD9\xA1"),
        EXPECT_LITERAL("\xD9\xA2"),
        EXPECT_LITERAL("\xEF\xBC\x93"),
        EXPECT_LITERAL("A"),
    };
    check_pretokenized("Unicode decimal digits also obey max_digits=1",
                       unicode_numbers_text,
                       unicode_numbers,
                       sizeof(unicode_numbers) / sizeof(unicode_numbers[0]));

    static const char unicode_space_text[] = "\xE2\x80\x83x";
    static const expected_span unicode_space[] = {
        EXPECT_LITERAL("\xE2\x80\x83x"),
    };
    check_pretokenized("Unicode whitespace before a letter joins that word",
                       unicode_space_text,
                       unicode_space,
                       sizeof(unicode_space) / sizeof(unicode_space[0]));

    static const char malformed_join_text[] = "\xC2\x41";
    static const expected_span malformed_join[] = {
        EXPECT_LITERAL("\xC2\x41"),
    };
    check_pretokenized("malformed UTF-8 keeps the original bytes and boundary",
                       malformed_join_text,
                       malformed_join,
                       sizeof(malformed_join) / sizeof(malformed_join[0]));

    static const char malformed_continuations_text[] = "\x80\x81!";
    static const expected_span malformed_continuations[] = {
        EXPECT_LITERAL("\x80\x81"),
        EXPECT_LITERAL("!"),
    };
    check_pretokenized("malformed continuation bytes remain lossless",
                       malformed_continuations_text,
                       malformed_continuations,
                       sizeof(malformed_continuations) /
                           sizeof(malformed_continuations[0]));

    static const char malformed_tail_text[] = "\xE2\x82";
    static const expected_span malformed_tail[] = {
        EXPECT_LITERAL("\xE2\x82"),
    };
    check_pretokenized("truncated UTF-8 at end remains one byte-preserving span",
                       malformed_tail_text,
                       malformed_tail,
                       sizeof(malformed_tail) / sizeof(malformed_tail[0]));

    span_capture failure;
    memset(&failure, 0, sizeof(failure));
    failure.fail_after = 1;
    CHECK(!lgn_bpe_pretokenize("first second", capture_span, &failure),
          "callback failure propagates");
    CHECK(failure.count == 1 &&
              failure.length[0] == sizeof("first") - 1u &&
              memcmp(failure.bytes[0], "first", sizeof("first") - 1u) == 0,
          "callback failure stops before the next span");

    span_capture empty;
    memset(&empty, 0, sizeof(empty));
    empty.fail_after = (size_t)-1;
    CHECK(lgn_bpe_pretokenize("", capture_span, &empty),
          "empty text pre-tokenizes successfully");
    CHECK(empty.count == 0, "empty text has no spans");
    CHECK(!lgn_bpe_pretokenize(NULL, capture_span, &empty),
          "NULL text is rejected");
    CHECK(!lgn_bpe_pretokenize("text", NULL, &empty),
          "NULL callback is rejected");
}

int main(void) {
    test_architecture();
    test_ladder_parser();
    test_ladder_formatter();
    test_s21_topology();
    test_s21_model_profile();
    test_whole_model_weight_bind();
    test_model_admission();
    test_dflash_profile_and_binding();
    test_dflash_binding_fixture();
    test_dflash_shadow_map();
    test_laguna_pretokenizer();
    if (failures != 0) {
        fprintf(stderr, "test-lgn: %d failure(s)\n", failures);
        return 1;
    }
    puts("test-lgn: ok");
    return 0;
}
