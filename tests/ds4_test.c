#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#include "../lgn.h"
#include "../ds4_gpu.h"

static void test_laguna_architecture_gate(void) {
    static const struct {
        const char *architecture;
        int accepted;
    } cases[] = {
        {"laguna", 1},
        {"deepseek4", 0},
        {"glm-dsa", 0},
        {"unknown", 0},
        {"", 0},
        {"Laguna", 0},
        {"laguna ", 0},
        {NULL, 0},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const char *architecture = cases[i].architecture;
        const size_t len = architecture ? strlen(architecture) : 0;
        TEST_ASSERT((int)lgn_architecture_is_supported(architecture, len) ==
                    cases[i].accepted);
    }
}

/* These selectors are backend-independent, so keep their strict parser and
 * route-boundary checks runnable in the default CPU/no-GPU test binary too. */
static void test_laguna_selector_parser(void) {
    TEST_ASSERT(ds4_gpu_q8_mv_ext_max_tokens_parse(NULL) == 16u);
    TEST_ASSERT(ds4_gpu_q8_mv_ext_max_tokens_parse("") == 16u);
    TEST_ASSERT(ds4_gpu_q8_mv_ext_max_tokens_parse("  2 \t") == 2u);
    TEST_ASSERT(ds4_gpu_q8_mv_ext_max_tokens_parse("16") == 16u);
    TEST_ASSERT(ds4_gpu_q8_mv_ext_max_tokens_parse("128") == 128u);
    TEST_ASSERT(ds4_gpu_q8_mv_ext_max_tokens_parse("129") == 128u);
    TEST_ASSERT(ds4_gpu_q8_mv_ext_max_tokens_parse("1") == 16u);
    TEST_ASSERT(ds4_gpu_q8_mv_ext_max_tokens_parse("-1") == 16u);
    TEST_ASSERT(ds4_gpu_q8_mv_ext_max_tokens_parse("12x") == 16u);
    TEST_ASSERT(ds4_gpu_q8_mv_ext_max_tokens_parse(
                    "18446744073709551616") == 16u);

    TEST_ASSERT(ds4_gpu_laguna_moe_min_tokens_parse(NULL) == 96u);
    TEST_ASSERT(ds4_gpu_laguna_moe_min_tokens_parse("") == 96u);
    TEST_ASSERT(ds4_gpu_laguna_moe_min_tokens_parse(" 32 ") == 32u);
    TEST_ASSERT(ds4_gpu_laguna_moe_min_tokens_parse("16") == 32u);
    TEST_ASSERT(ds4_gpu_laguna_moe_min_tokens_parse("96") == 96u);
    TEST_ASSERT(ds4_gpu_laguna_moe_min_tokens_parse("4096") == 4096u);
    TEST_ASSERT(ds4_gpu_laguna_moe_min_tokens_parse("4097") == 4096u);
    TEST_ASSERT(ds4_gpu_laguna_moe_min_tokens_parse("-1") == 96u);
    TEST_ASSERT(ds4_gpu_laguna_moe_min_tokens_parse("12x") == 96u);
    TEST_ASSERT(ds4_gpu_laguna_moe_min_tokens_parse(
                    "18446744073709551616") == 96u);

    TEST_ASSERT(ds4_gpu_laguna_direct_kv_prefill_env_mode(NULL) == 0);
    TEST_ASSERT(ds4_gpu_laguna_direct_kv_prefill_env_mode("") == 0);
    TEST_ASSERT(ds4_gpu_laguna_direct_kv_prefill_env_mode("0") == 0);
    TEST_ASSERT(ds4_gpu_laguna_direct_kv_prefill_env_mode("1") == 1);
    TEST_ASSERT(ds4_gpu_laguna_direct_kv_prefill_env_mode("01") < 0);
    TEST_ASSERT(ds4_gpu_laguna_direct_kv_prefill_env_mode("true") < 0);

    TEST_ASSERT(ds4_gpu_laguna_output_head_norm_fuse_env_mode(NULL) == 0);
    TEST_ASSERT(ds4_gpu_laguna_output_head_norm_fuse_env_mode("") == 0);
    TEST_ASSERT(ds4_gpu_laguna_output_head_norm_fuse_env_mode("0") == 0);
    TEST_ASSERT(ds4_gpu_laguna_output_head_norm_fuse_env_mode("1") == 1);
    TEST_ASSERT(ds4_gpu_laguna_output_head_norm_fuse_env_mode("01") < 0);
    TEST_ASSERT(ds4_gpu_laguna_output_head_norm_fuse_env_mode("true") < 0);

    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_prefill_route(
                    1, 0, 1, 16) ==
                DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_ORDINARY_PREFILL_STOCK);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_prefill_route(
                    1, 0, 2, 16) ==
                DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_ORDINARY_PREFILL_STOCK);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_prefill_route(
                    1, 0, 16, 16) ==
                DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_ORDINARY_PREFILL_STOCK);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_prefill_route(
                    1, 0, 17, 16) ==
                DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_BATCH_FUSED);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_prefill_route(
                    1, 0, 128, 128) ==
                DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_ORDINARY_PREFILL_STOCK);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_prefill_route(
                    1, 0, 129, 128) ==
                DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_BATCH_FUSED);
}

#ifndef DS4_NO_GPU
#include <math.h>

bool ds4_test_dspark_cache_window_crop(void);

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static bool test_env_bool(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

static uint32_t test_env_u32(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long n = strtoul(v, &end, 10);
    if (end == v) return 0;
    return n > UINT32_MAX ? UINT32_MAX : (uint32_t)n;
}

static uint64_t test_env_gib(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long long n = strtoull(v, &end, 10);
    if (end == v || n == 0) return 0;
    const uint64_t one_gib = 1024ull * 1024ull * 1024ull;
    if (n > UINT64_MAX / one_gib) return UINT64_MAX;
    return (uint64_t)n * one_gib;
}

static char *test_save_env(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    size_t len = strlen(value);
    char *copy = malloc(len + 1);
    TEST_ASSERT(copy != NULL);
    if (!copy) return NULL;
    memcpy(copy, value, len + 1);
    return copy;
}

static void test_restore_env(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}

#ifdef __APPLE__
/* SWA route selectors are captured at ds4_gpu_init.  Focused A/B legs must
 * therefore restart the Metal lifecycle after changing the environment. */
static bool test_restart_metal_lifecycle(void) {
    ds4_gpu_cleanup();
    return ds4_gpu_init() != 0;
}
#endif
typedef enum {
    TEST_LAGUNA_SOURCE_BUILTIN = 0,
    TEST_LAGUNA_SOURCE_CURRENT,
    TEST_LAGUNA_SOURCE_OLD,
    TEST_LAGUNA_SOURCE_MISSING,
    TEST_LAGUNA_SOURCE_INVALID,
} test_laguna_source_mode;

/* Focused Metal source matrices run in fresh processes.  The explicit mode
 * variable makes the expected result part of the invocation rather than
 * inferring success from the mere presence of a nonempty override. */
static test_laguna_source_mode test_laguna_source_mode_from_env(void) {
    const char *source = getenv("DS4_METAL_LAGUNA_SOURCE");
    if (!source || !source[0]) return TEST_LAGUNA_SOURCE_BUILTIN;
    const char *expected = getenv("DS4_TEST_LAGUNA_SOURCE_MODE");
    if (!expected || !expected[0]) return TEST_LAGUNA_SOURCE_INVALID;
    if (!strcmp(expected, "current")) return TEST_LAGUNA_SOURCE_CURRENT;
    if (!strcmp(expected, "old")) return TEST_LAGUNA_SOURCE_OLD;
    if (!strcmp(expected, "missing")) return TEST_LAGUNA_SOURCE_MISSING;
    return TEST_LAGUNA_SOURCE_INVALID;
}

typedef struct {
    char *cold_decode;
    char *batch_selected_addr;
} test_streaming_prefill_env;

static test_streaming_prefill_env test_force_canonical_streaming_prefill(void) {
    test_streaming_prefill_env saved = {
        .cold_decode =
            test_save_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL"),
        .batch_selected_addr =
            test_save_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR"),
    };
    if (test_env_bool("DS4_TEST_SSD_STREAMING")) {
        setenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL", "1", 1);
        setenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR", "1", 1);
    }
    return saved;
}

static void test_restore_canonical_streaming_prefill(
        test_streaming_prefill_env saved) {
    test_restore_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL",
                     saved.cold_decode);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR",
                     saved.batch_selected_addr);
}

static ds4_backend test_model_backend(void) {
    const char *backend = getenv("DS4_TEST_BACKEND");
    if (backend && !strcmp(backend, "cpu")) return DS4_BACKEND_CPU;
#ifdef __APPLE__
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static ds4_engine *test_open_engine(bool quality) {
    ds4_engine *engine = NULL;
    /* DS4_TEST_MTP loads the MTP head on the fast engine so the speculative
     * verify regression can reuse it; draft=4 hits the multi-row verify path. */
    const char *mtp = getenv("DS4_TEST_MTP");
    ds4_engine_options opt = {
        .model_path = test_model_path(),
        .backend = test_model_backend(),
        .quality = quality,
        .ssd_streaming = test_env_bool("DS4_TEST_SSD_STREAMING"),
        .ssd_streaming_cold = test_env_bool("DS4_TEST_SSD_STREAMING_COLD"),
        .ssd_streaming_cache_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS"),
        .ssd_streaming_cache_bytes =
            test_env_gib("DS4_TEST_SSD_STREAMING_CACHE_GB"),
        .ssd_streaming_preload_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_PRELOAD_EXPERTS"),
        .mtp_path = (mtp && mtp[0] && !quality) ? mtp : NULL,
        .mtp_draft_tokens = (mtp && mtp[0] && !quality) ? 4 : 0,
    };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0);
    return engine;
}

static ds4_engine *test_get_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    *slot = test_open_engine(quality);
    return *slot;
}

static void test_close_engines(void) {
    ds4_engine_close(test_engine_fast);
    ds4_engine_close(test_engine_quality);
    test_engine_fast = NULL;
    test_engine_quality = NULL;
}

static void test_close_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    ds4_engine_close(*slot);
    *slot = NULL;
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static uint32_t test_float_ordered_u32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

typedef struct {
    size_t mismatch_count;
    uint32_t max_ulp;
    float max_abs;
} test_float_compare_stats;

static test_float_compare_stats test_compare_float_bits(
        const float *reference,
        const float *actual,
        size_t count) {
    test_float_compare_stats stats = {0};
    for (size_t i = 0; i < count; i++) {
        if (memcmp(&reference[i], &actual[i], sizeof(float)) != 0) {
            stats.mismatch_count++;
        }

        const uint32_t ref_ordered = test_float_ordered_u32(reference[i]);
        const uint32_t actual_ordered = test_float_ordered_u32(actual[i]);
        const uint32_t ulp = ref_ordered > actual_ordered
            ? ref_ordered - actual_ordered
            : actual_ordered - ref_ordered;
        if (ulp > stats.max_ulp) stats.max_ulp = ulp;

        const float abs_error = fabsf(reference[i] - actual[i]);
        if (abs_error > stats.max_abs) stats.max_abs = abs_error;
    }
    return stats;
}

#if defined(__APPLE__)
static const uint32_t test_copy_f32_patterns[] = {
    0x00000000u, 0x80000000u, /* signed zero */
    0x7f800000u, 0xff800000u, /* infinities */
    0x7fc00000u, 0x7fc12345u, 0xffc54321u, 0x7fa00001u, /* NaNs */
    0x00000001u, 0x007fffffu, 0x80000001u, 0x807fffffu, /* F32 subnormals */
    0x32ffffffu, 0x33000000u, 0x33000001u,
    0x337fffffu, 0x33800000u, 0x33800001u, /* minimum F16 subnormal boundary */
    0x387fbfffu, 0x387fc000u, 0x387fffffu,
    0x38800000u, 0x38800001u, /* maximum subnormal/minimum normal boundary */
    0x3f800fffu, 0x3f801000u, 0x3f801001u,
    0x3f802fffu, 0x3f803000u, 0x3f803001u, /* round-to-nearest ties */
    0x477fdfffu, 0x477fe000u, 0x477fefffu,
    0x477ff000u, 0x477ff001u, 0x47800000u, /* maximum/overflow boundary */
    0xc77fe000u, 0xbf801000u, 0x3eaaaaabu, 0xbeaaaaabu,
};

static void test_fill_copy_f32_patterns(void *dst, uint32_t n, uint32_t salt) {
    uint8_t *bytes = dst;
    const uint32_t n_patterns =
        (uint32_t)(sizeof(test_copy_f32_patterns) / sizeof(test_copy_f32_patterns[0]));
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t bits = test_copy_f32_patterns[(i + salt) % n_patterns];
        memcpy(bytes + (uint64_t)i * sizeof(bits), &bits, sizeof(bits));
    }
}
#endif

static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float test_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void test_fill_q8_0_weights(uint8_t *weights,
                                   uint32_t in_dim,
                                   uint32_t out_dim,
                                   uint32_t seed) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 17u + k * 23u + (o ^ k) * 3u +
                                     seed * 29u + ((o + seed) ^ k) * 5u) % 67u) - 33;
                vals[i] = (float)v / 96.0f;
                float av = fabsf(vals[i]);
                if (av > amax) amax = av;
            }
            const uint16_t scale_bits = test_float_to_f16(amax / 127.0f);
            const float scale = test_f16_to_f32(scale_bits);
            memcpy(row + b * 34u, &scale_bits, sizeof(scale_bits));
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                int q = scale != 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[i] = (int8_t)q;
            }
        }
    }
}

static void test_metal_f16_matvec_fast_nr0_4(void) {
    /*
     * This is the short regression for the long-context repetition failure.
     * Decode uses one-token F16 matvecs for several DS4 projections; the fast
     * nr0=4 variant must be numerically equivalent to the plain kernel.
     */
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 512;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            ref += w * x_host[i];
        }
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_f16_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((o * 11u + i * 13u + (o ^ i) * 5u) % 61u) - 30;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16((float)v / 96.0f);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 7u + i * 17u + (t ^ i) * 3u) % 73u) - 36;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                          in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float ref = 0.0f;
            for (uint32_t i = 0; i < in_dim; i++) {
                ref += test_f16_to_f32(weights[(uint64_t)o * in_dim + i]) *
                       x_host[(uint64_t)t * in_dim + i];
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_q8_0_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint8_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights, in_dim, out_dim, 0);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 19u + i * 7u + (t ^ i)) % 71u) - 35;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(out, weights_raw, weight_alloc, 0,
                                           in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const uint8_t *row = weights + (uint64_t)o * row_bytes;
            float ref = 0.0f;
            for (uint32_t b = 0; b < in_dim / 32u; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = test_f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32; i++) {
                    ref += scale * (float)qs[i] *
                           x_host[(uint64_t)t * in_dim + b * 32u + i];
                }
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    fprintf(stderr,
            "ds4-test: Q8_0 prefill matmul max_abs=%g rms=%g\n",
            max_abs, rms);
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_pack_slot_rows_f32(void) {
    const uint32_t n_rows = 3;
    const uint32_t width = 5;
    const uint32_t n_slots = 4;
    const uint32_t slot_cap = 6;
    const uint64_t slot_count = (uint64_t)n_slots * slot_cap * width;
    const uint64_t out_count = (uint64_t)n_rows * n_slots * width;
    const uint64_t slot_bytes = slot_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);

    ds4_gpu_tensor *slots = ds4_gpu_tensor_alloc(slot_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(slots != NULL);
    TEST_ASSERT(out != NULL);
    if (!slots || !out) {
        ds4_gpu_tensor_free(slots);
        ds4_gpu_tensor_free(out);
        return;
    }

    float *slot_host = malloc((size_t)slot_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(slot_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!slot_host || !out_host) {
        free(slot_host);
        free(out_host);
        ds4_gpu_tensor_free(slots);
        ds4_gpu_tensor_free(out);
        return;
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        for (uint32_t row = 0; row < slot_cap; row++) {
            for (uint32_t col = 0; col < width; col++) {
                slot_host[((uint64_t)slot * slot_cap + row) * width + col] =
                    (float)(slot * 1000u + row * 100u + col);
            }
        }
    }
    for (uint64_t i = 0; i < out_count; i++) out_host[i] = -1.0f;

    TEST_ASSERT(ds4_gpu_tensor_write(slots, 0, slot_host, slot_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_pack_slot_rows_f32_tensor(out,
                                                  slots,
                                                  n_rows,
                                                  width,
                                                  n_slots,
                                                  slot_cap) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            for (uint32_t col = 0; col < width; col++) {
                const float ref =
                    slot_host[((uint64_t)slot * slot_cap + row) * width + col];
                const float got =
                    out_host[((uint64_t)row * n_slots + slot) * width + col];
                TEST_ASSERT(got == ref);
            }
        }
    }

    free(slot_host);
    free(out_host);
    ds4_gpu_tensor_free(slots);
    ds4_gpu_tensor_free(out);
}

static void test_metal_store_raw_kv_batch_wrap(void) {
    const uint32_t raw_cap = 5;
    const uint32_t head_dim = 3;
    const uint32_t n_tokens = 4;
    const uint32_t pos0 = 3;
    const uint64_t kv_count = (uint64_t)n_tokens * head_dim;
    const uint64_t raw_count = (uint64_t)raw_cap * head_dim;
    const uint64_t kv_bytes = kv_count * sizeof(float);
    const uint64_t raw_bytes = raw_count * sizeof(float);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(kv_bytes);
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(raw != NULL);
    if (!kv || !raw) {
        ds4_gpu_tensor_free(kv);
        ds4_gpu_tensor_free(raw);
        return;
    }

    float kv_host[12];
    float raw_host[15];
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t d = 0; d < head_dim; d++) {
            kv_host[(uint64_t)t * head_dim + d] = (float)(100u * t + d);
        }
    }
    for (uint64_t i = 0; i < raw_count; i++) raw_host[i] = -1.0f;

    TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
    TEST_ASSERT(ds4_gpu_store_raw_kv_batch_tensor(raw,
                                                  kv,
                                                  raw_cap,
                                                  pos0,
                                                  n_tokens,
                                                  head_dim) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(raw, 0, raw_host, raw_bytes) != 0);

    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t row = (pos0 + t) % raw_cap;
        for (uint32_t d = 0; d < head_dim; d++) {
            const float ref = kv_host[(uint64_t)t * head_dim + d];
            const float got = raw_host[(uint64_t)row * head_dim + d];
            TEST_ASSERT(got == ref);
        }
    }
    for (uint32_t d = 0; d < head_dim; d++) {
        TEST_ASSERT(raw_host[(uint64_t)2u * head_dim + d] == -1.0f);
    }

    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(raw);
}

static void test_dspark_cache_window_crop(void) {
    TEST_ASSERT(ds4_test_dspark_cache_window_crop());
}

static void test_laguna_decode_ladder_parser(void) {
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

    TEST_ASSERT(lgn_decode_ladder_parse(NULL, 48, &mask));
    TEST_ASSERT(mask == 0);
    mask = UINT64_MAX;
    TEST_ASSERT(lgn_decode_ladder_parse("", 48, &mask));
    TEST_ASSERT(mask == 0);
    TEST_ASSERT(lgn_decode_ladder_parse(
                    "7,15,23,31,39,47", 48, &mask));
    TEST_ASSERT(mask == (all_expected & ~UINT64_C(0x3)));
    TEST_ASSERT(lgn_decode_ladder_parse(
                    "0,1,7,15,23,31,39,47", 48, &mask));
    TEST_ASSERT(mask == all_expected);
    TEST_ASSERT(lgn_decode_ladder_parse(
                    "1,7,15,23,31,39", 48, &mask));
    TEST_ASSERT(mask == (all_expected & ~UINT64_C(0x800000000001)));
    TEST_ASSERT(lgn_decode_ladder_parse("47", 48, &mask));
    TEST_ASSERT(mask == (UINT64_C(1) << 47));
    TEST_ASSERT(lgn_decode_ladder_parse("0,63", 64, &mask));
    TEST_ASSERT(mask == ((UINT64_C(1) << 0) | (UINT64_C(1) << 63)));

    mask = UINT64_MAX;
    TEST_ASSERT(lgn_decode_ladder_parse(NULL, 65, &mask));
    TEST_ASSERT(mask == 0);
    mask = UINT64_MAX;
    TEST_ASSERT(lgn_decode_ladder_parse("", 79, &mask));
    TEST_ASSERT(mask == 0);

    static const char *invalid[] = {
        "7,7", "15,7", "48", "-1", "1, 7", "1,,7", "1,7,",
        "1,x", "1,7\n", "18446744073709551616",
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        mask = UINT64_MAX;
        TEST_ASSERT(!lgn_decode_ladder_parse(invalid[i], 48, &mask));
        TEST_ASSERT(mask == 0);
    }
    mask = UINT64_MAX;
    TEST_ASSERT(!lgn_decode_ladder_parse("0", 0, &mask));
    TEST_ASSERT(mask == 0);
    mask = UINT64_MAX;
    TEST_ASSERT(!lgn_decode_ladder_parse("0", 65, &mask));
    TEST_ASSERT(mask == 0);
}

static void test_metal_q8_0_decode_pair_exact_case(
        uint32_t out0_dim,
        uint32_t out1_dim,
        uint32_t seed0,
        uint32_t seed1) {
    /* Exercise the Q-A/KV contract with unequal, odd output extents and
     * independently page-aligned model ranges. The paired kernel must be
     * bit-identical to two standalone decode matvec dispatches. */
    const uint32_t in_dim = 4096;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight0_bytes = (uint64_t)out0_dim * row_bytes;
    const uint64_t weight1_bytes = (uint64_t)out1_dim * row_bytes;
    const uint64_t weight1_offset = test_round_up_u64(weight0_bytes, page);
    const uint64_t weight_alloc =
        test_round_up_u64(weight1_offset + weight1_bytes, page);
    /* Keep the stock pair A/B on the production NR2 selector even when the
     * surrounding test process inherited a conflicting rows setting. */
    char *saved_rows = test_save_env("DS4_METAL_Q8_MV_ROWS");
    TEST_ASSERT(setenv("DS4_METAL_Q8_MV_ROWS", "2", 1) == 0);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)page, (size_t)weight_alloc) == 0);
    if (!weights_raw) {
        test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
        return;
    }
    memset(weights_raw, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights((uint8_t *)weights_raw, in_dim, out0_dim, seed0);
    test_fill_q8_0_weights((uint8_t *)weights_raw + weight1_offset,
                           in_dim, out1_dim, seed1);

    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out0_bytes = (uint64_t)out0_dim * sizeof(float);
    const uint64_t out1_bytes = (uint64_t)out1_dim * sizeof(float);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref0 = ds4_gpu_tensor_alloc(out0_bytes);
    ds4_gpu_tensor *ref1 = ds4_gpu_tensor_alloc(out1_bytes);
    ds4_gpu_tensor *pair0 = ds4_gpu_tensor_alloc(out0_bytes);
    ds4_gpu_tensor *pair1 = ds4_gpu_tensor_alloc(out1_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref0 != NULL);
    TEST_ASSERT(ref1 != NULL);
    TEST_ASSERT(pair0 != NULL);
    TEST_ASSERT(pair1 != NULL);
    if (!x || !ref0 || !ref1 || !pair0 || !pair1) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(ref0);
        ds4_gpu_tensor_free(ref1);
        ds4_gpu_tensor_free(pair0);
        ds4_gpu_tensor_free(pair1);
        free(weights_raw);
        test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *ref0_host = malloc((size_t)out0_bytes);
    float *ref1_host = malloc((size_t)out1_bytes);
    float *pair0_host = malloc((size_t)out0_bytes);
    float *pair1_host = malloc((size_t)out1_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref0_host != NULL);
    TEST_ASSERT(ref1_host != NULL);
    TEST_ASSERT(pair0_host != NULL);
    TEST_ASSERT(pair1_host != NULL);
    if (!x_host || !ref0_host || !ref1_host || !pair0_host || !pair1_host) {
        free(x_host);
        free(ref0_host);
        free(ref1_host);
        free(pair0_host);
        free(pair1_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(ref0);
        ds4_gpu_tensor_free(ref1);
        ds4_gpu_tensor_free(pair0);
        ds4_gpu_tensor_free(pair1);
        free(weights_raw);
        test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        const int v = (int)((i * 29u + (i ^ (i >> 3u)) * 7u) % 127u) - 63;
        x_host[i] = (float)v / 72.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(ref0, weights_raw, weight_alloc, 0,
                                           in_dim, out0_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(ref1, weights_raw, weight_alloc,
                                           weight1_offset,
                                           in_dim, out1_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_pair_tensor(pair0, pair1,
                                                weights_raw, weight_alloc,
                                                0, weight1_offset,
                                                in_dim, out0_dim, out1_dim,
                                                x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref0, 0, ref0_host, out0_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref1, 0, ref1_host, out1_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(pair0, 0, pair0_host, out0_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(pair1, 0, pair1_host, out1_bytes) != 0);

    uint32_t mismatch0 = 0;
    uint32_t mismatch1 = 0;
    float max_abs0 = 0.0f;
    float max_abs1 = 0.0f;
    for (uint32_t i = 0; i < out0_dim; i++) {
        if (memcmp(&ref0_host[i], &pair0_host[i], sizeof(float)) != 0) mismatch0++;
        const float err = fabsf(ref0_host[i] - pair0_host[i]);
        if (err > max_abs0) max_abs0 = err;
    }
    for (uint32_t i = 0; i < out1_dim; i++) {
        if (memcmp(&ref1_host[i], &pair1_host[i], sizeof(float)) != 0) mismatch1++;
        const float err = fabsf(ref1_host[i] - pair1_host[i]);
        if (err > max_abs1) max_abs1 = err;
    }
    if (mismatch0 != 0 || mismatch1 != 0) {
        fprintf(stderr,
                "ds4-test: paired Q8_0 exactness mismatches=%u/%u max_abs=%g, %u/%u max_abs=%g\n",
                mismatch0, out0_dim, max_abs0,
                mismatch1, out1_dim, max_abs1);
    }
    TEST_ASSERT(mismatch0 == 0);
    TEST_ASSERT(mismatch1 == 0);

    free(x_host);
    free(ref0_host);
    free(ref1_host);
    free(pair0_host);
    free(pair1_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(ref0);
    ds4_gpu_tensor_free(ref1);
    ds4_gpu_tensor_free(pair0);
    ds4_gpu_tensor_free(pair1);
    free(weights_raw);
    test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
}

static void test_metal_q8_0_decode_pair_exact(void) {
    /* Cover both possible one-bank tail directions. Distinct seeds ensure a
     * mistaken A-for-B weight binding cannot compare equal by construction. */
    test_metal_q8_0_decode_pair_exact_case(77, 19, 11, 97);
    test_metal_q8_0_decode_pair_exact_case(19, 77, 23, 131);
}

#if defined(__APPLE__)
static void test_metal_q8_0_output_nr4_exact_case(
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t seed) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc =
        test_round_up_u64(weight_bytes, page);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)page,
                               (size_t)weight_alloc) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *candidate = ds4_gpu_tensor_alloc(out_bytes);
    float *x_host = malloc((size_t)x_bytes);
    float *reference_host = malloc((size_t)out_bytes);
    float *candidate_host = malloc((size_t)out_bytes);
    TEST_ASSERT(weights_raw != NULL);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(reference != NULL);
    TEST_ASSERT(candidate != NULL);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(reference_host != NULL);
    TEST_ASSERT(candidate_host != NULL);

    const char *force_env = "DS4_METAL_ENABLE_OUTPUT_Q8_NR4";
    const char *disable_env = "DS4_METAL_DISABLE_M3_OUTPUT_Q8_NR4";
    char *saved_force = test_save_env(force_env);
    char *saved_disable = test_save_env(disable_env);
    test_float_compare_stats stats = {0};

    const bool allocated = weights_raw && x && reference && candidate &&
        x_host && reference_host && candidate_host;
    if (allocated) {
        memset(weights_raw, 0, (size_t)weight_alloc);
        test_fill_q8_0_weights(
            (uint8_t *)weights_raw, in_dim, out_dim, seed);
        for (uint32_t i = 0; i < in_dim; i++) {
            const int value =
                (int)((i * 29u + (i ^ (i >> 3u)) * 7u +
                       seed * 17u) % 127u) - 63;
            x_host[i] = (float)value / 72.0f;
        }
        for (uint32_t i = 0; i < out_dim; i++) {
            const uint32_t poison = 0x7fc00001u + (i & 0x3ffu);
            memcpy(reference_host + i, &poison, sizeof(poison));
            memcpy(candidate_host + i, &poison, sizeof(poison));
        }
        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        reference, 0, reference_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        candidate, 0, candidate_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(
                        weights_raw, weight_alloc) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(unsetenv(force_env) == 0);
        TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                        reference, weights_raw, weight_alloc, 0,
                        in_dim, out_dim, x, 1) != 0);

        TEST_ASSERT(setenv(force_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(disable_env) == 0);
        TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                        candidate, weights_raw, weight_alloc, 0,
                        in_dim, out_dim, x, 1) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        reference, 0, reference_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        candidate, 0, candidate_host, out_bytes) != 0);
        stats = test_compare_float_bits(
            reference_host, candidate_host, out_dim);
    }

    test_restore_env(force_env, saved_force);
    test_restore_env(disable_env, saved_disable);
    fprintf(stderr,
            "ds4-test: output Q8 NR4 exact in=%u out=%u nsg=%u "
            "mismatch=%zu/%u max_ulp=%u max_abs=%g\n",
            in_dim, out_dim, out_dim > 65536u ? 8u : 4u,
            stats.mismatch_count, out_dim, stats.max_ulp, stats.max_abs);
    TEST_ASSERT(stats.mismatch_count == 0);

    free(candidate_host);
    free(reference_host);
    free(x_host);
    ds4_gpu_tensor_free(candidate);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(x);
    free(weights_raw);
}

static void test_metal_q8_0_output_nr4_exact(void) {
    test_metal_q8_0_output_nr4_exact_case(4096, 68, 83);
    test_metal_q8_0_output_nr4_exact_case(128, 65540, 89);
}

static void test_metal_laguna_dense_q8_gate_up_swiglu(void) {
    /* The production arm is intentionally opt-in: two 3072x12288 Q8_0
     * matrices are about 77 MiB and this test compares every output bit. */
    const char *focused = getenv("DS4_TEST_LAGUNA_DENSE_Q8_FUSED");
    if (!focused) {
        fprintf(stderr,
                "ds4-test: Laguna dense Q8 gate/up+SwiGLU skipped "
                "(set DS4_TEST_LAGUNA_DENSE_Q8_FUSED=1)\n");
        return;
    }
    TEST_ASSERT(strcmp(focused, "1") == 0);
    if (strcmp(focused, "1") != 0) return;

    /* Keep the focused test's strict-selector contract executable without a
     * model: production code and tests share this parser, so every malformed
     * value has a direct proof before the model-path cases below. */
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_env_mode(NULL) == 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_env_mode("") == 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_env_mode("0") == 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_env_mode("1") == 1);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_env_mode("01") < 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_env_mode("true") < 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_rows_env_mode(NULL) == 2);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_rows_env_mode("") == 2);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_rows_env_mode("2") == 2);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_rows_env_mode("4") == 4);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_rows_env_mode("3") < 0);

    /* The pure production decision is also the fail-before-mutation hook:
     * selector-on plus an alternate rows setting is an explicit conflict,
     * while off ignores all backend certificates.  Keep route selection in the
     * same shared helper used by the decode entrypoint. */
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
                    -1, 1, 1, 2, 1) < 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
                    2, 1, 1, 2, 1) < 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
                    0, 0, 0, 4, 0) == 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
                    1, 1, 1, 2, 1) == 1);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
                    1, 1, 1, 4, 1) < 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
                    1, 0, 1, 2, 1) < 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
                    1, 1, 0, 2, 1) < 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
                    1, 1, 1, 2, 0) < 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_route(
                    0, 1, 0) == DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_STOCK);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_route(
                    1, 1, 0) == DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_DECODE_MID);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_route(
                    1, 1, 1) == DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_STOCK);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_route(
                    1, 0, 0) ==
                DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_ORDINARY_PREFILL_STOCK);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_route(
                    1, 0, 1) == DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_STOCK);

    /* Exercise the actual environment conflict separately from the model
     * path. A strict request paired with rows=4 must be rejected before any
     * graph/KV mutation; the later numeric reference always pins NR2. */
    char *saved_selector =
        test_save_env("DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU");
    char *saved_conflict_rows = test_save_env("DS4_METAL_Q8_MV_ROWS");
    TEST_ASSERT(setenv("DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU", "1", 1) ==
                0);
    TEST_ASSERT(setenv("DS4_METAL_Q8_MV_ROWS", "4", 1) == 0);
    const int conflict_mode = ds4_gpu_laguna_dense_q8_gate_up_swiglu_env_mode(
        getenv("DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU"));
    const int conflict_rows = ds4_gpu_laguna_dense_q8_gate_up_swiglu_rows_env_mode(
        getenv("DS4_METAL_Q8_MV_ROWS"));
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
                    conflict_mode, 1, 1, conflict_rows, 1) < 0);
    test_restore_env("DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU",
                     saved_selector);
    test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_conflict_rows);

    /* Pin the production stock reference and fused candidates to the only
     * dense Q8 topology currently exposed by the Metal library: NR2. */
    char *saved_rows = test_save_env("DS4_METAL_Q8_MV_ROWS");
    TEST_ASSERT(setenv("DS4_METAL_Q8_MV_ROWS", "2", 1) == 0);
    const bool mid_rows2 = ds4_gpu_shared_mid_swiglu_q8_0_available() != 0;
    if (!mid_rows2) {
        /* A pre-feature DENSE source is valid with the selector off.  This
         * focused selector-on invocation must instead fail closed before any
         * graph/KV mutation, so an absent optional PSO is a passing negative
         * result rather than an unconditional default-suite failure. */
        TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
                        1, 1, 1, 2, 0) < 0);
        fprintf(stderr,
                "ds4-test: Laguna dense Q8 fused selector fail-closed; "
                "optional source PSO unavailable\n");
        test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
        return;
    }

    const uint32_t in_dim = 3072u;
    const uint32_t out_dim = 12288u;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight1_offset = test_round_up_u64(weight_bytes, page);
    const uint64_t weight_alloc =
        test_round_up_u64(weight1_offset + weight_bytes, page);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)page,
                               (size_t)weight_alloc) == 0);
    if (!weights_raw) {
        test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
        return;
    }
    memset(weights_raw, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights((uint8_t *)weights_raw, in_dim, out_dim, 173u);
    test_fill_q8_0_weights((uint8_t *)weights_raw + weight1_offset,
                           in_dim, out_dim, 239u);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_gate = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_up = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_mid = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_gate = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_up = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_mid = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *mid_only = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref_gate != NULL);
    TEST_ASSERT(ref_up != NULL);
    TEST_ASSERT(ref_mid != NULL);
    TEST_ASSERT(fused_gate != NULL);
    TEST_ASSERT(fused_up != NULL);
    TEST_ASSERT(fused_mid != NULL);
    TEST_ASSERT(mid_only != NULL);
    if (!x || !ref_gate || !ref_up || !ref_mid || !fused_gate ||
        !fused_up || !fused_mid || !mid_only) {
        ds4_gpu_tensor_free(mid_only);
        ds4_gpu_tensor_free(fused_mid);
        ds4_gpu_tensor_free(fused_up);
        ds4_gpu_tensor_free(fused_gate);
        ds4_gpu_tensor_free(ref_mid);
        ds4_gpu_tensor_free(ref_up);
        ds4_gpu_tensor_free(ref_gate);
        ds4_gpu_tensor_free(x);
        free(weights_raw);
        test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *ref_gate_host = malloc((size_t)out_bytes);
    float *ref_up_host = malloc((size_t)out_bytes);
    float *ref_mid_host = malloc((size_t)out_bytes);
    float *fused_gate_host = malloc((size_t)out_bytes);
    float *fused_up_host = malloc((size_t)out_bytes);
    float *fused_mid_host = malloc((size_t)out_bytes);
    float *mid_only_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref_gate_host != NULL);
    TEST_ASSERT(ref_up_host != NULL);
    TEST_ASSERT(ref_mid_host != NULL);
    TEST_ASSERT(fused_gate_host != NULL);
    TEST_ASSERT(fused_up_host != NULL);
    TEST_ASSERT(fused_mid_host != NULL);
    TEST_ASSERT(mid_only_host != NULL);
    if (!x_host || !ref_gate_host || !ref_up_host || !ref_mid_host ||
        !fused_gate_host || !fused_up_host || !fused_mid_host ||
        !mid_only_host) {
        free(mid_only_host);
        free(fused_mid_host);
        free(fused_up_host);
        free(fused_gate_host);
        free(ref_mid_host);
        free(ref_up_host);
        free(ref_gate_host);
        free(x_host);
        ds4_gpu_tensor_free(mid_only);
        ds4_gpu_tensor_free(fused_mid);
        ds4_gpu_tensor_free(fused_up);
        ds4_gpu_tensor_free(fused_gate);
        ds4_gpu_tensor_free(ref_mid);
        ds4_gpu_tensor_free(ref_up);
        ds4_gpu_tensor_free(ref_gate);
        ds4_gpu_tensor_free(x);
        free(weights_raw);
        test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
        return;
    }

    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    /* The generic dense fused kernels are compiled with NR2 and load output
     * rows in pairs.  Every host wrapper must reject an odd shape before it
     * can create a command buffer. */
    const uint32_t odd_out_dim = out_dim - 1u;
    TEST_ASSERT(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                    fused_gate, fused_up, fused_mid,
                    weights_raw, weight_alloc, 0, weight1_offset,
                    in_dim, odd_out_dim, x, 0.0f) == 0);
    TEST_ASSERT(ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                    mid_only, weights_raw, weight_alloc, 0,
                    weight1_offset, in_dim, odd_out_dim, x, 0.0f) == 0);
    TEST_ASSERT(ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
                    fused_gate, fused_up, fused_mid,
                    weights_raw, weight_alloc, 0, weight1_offset,
                    in_dim, odd_out_dim, x, 0.0f) == 0);
    TEST_ASSERT(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
                    fused_gate, fused_up, fused_mid,
                    weights_raw, weight_alloc, 0, weight1_offset,
                    in_dim, odd_out_dim, x, 1, 0.0f) == 0);
    TEST_ASSERT(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                    fused_gate, fused_up, fused_mid,
                    weights_raw, weight_alloc, 0, weight1_offset,
                    in_dim, odd_out_dim, x, 1, 0.0f) == 0);
    /* Zero dimensions must fail before range arithmetic or command encoding. */
    TEST_ASSERT(ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                    mid_only, weights_raw, weight_alloc, 0,
                    weight1_offset, 0, out_dim, x, 0.0f) == 0);
    TEST_ASSERT(ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                    mid_only, weights_raw, weight_alloc, 0,
                    weight1_offset, in_dim, 0, x, 0.0f) == 0);
    for (uint32_t ci = 0; ci < 4u; ci++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            uint32_t bits;
            if (ci == 0u) {
                const int value =
                    (int)((i * 37u + (i ^ (i >> 2u)) * 11u) % 127u) - 63;
                x_host[i] = (float)value / 72.0f;
            } else if (ci == 1u) {
                static const uint32_t patterns[] = {
                    0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
                    0x007fffffu, 0x807fffffu, 0x3f800000u, 0xbf800000u,
                };
                bits = patterns[i % (sizeof(patterns) / sizeof(patterns[0]))];
                memcpy(&x_host[i], &bits, sizeof(bits));
            } else if (ci == 2u) {
                /* Alternating signed values force cancellation in a number
                 * of rows while retaining ordinary finite magnitudes. */
                const float magnitude =
                    (float)(1u + ((i * 13u) % 31u)) / 19.0f;
                x_host[i] = (i & 1u) ? -magnitude : magnitude;
            } else {
                const int value =
                    (int)((i * 97u + (i >> 3u) * 5u) % 251u) - 125;
                x_host[i] = (float)value / 211.0f;
            }
        }
        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                        ref_gate, weights_raw, weight_alloc, 0,
                        in_dim, out_dim, x, 1) != 0);
        TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                        ref_up, weights_raw, weight_alloc, weight1_offset,
                        in_dim, out_dim, x, 1) != 0);
        TEST_ASSERT(ds4_gpu_swiglu_tensor(
                        ref_mid, ref_gate, ref_up, out_dim, 0.0f, 1.0f) != 0);
        /* Primary proof: the stock path and both fused candidates use the
         * same actual NR2 dispatch selected by the production verifier. */
        TEST_ASSERT(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                        fused_gate, fused_up, fused_mid,
                        weights_raw, weight_alloc, 0, weight1_offset,
                        in_dim, out_dim, x, 0.0f) != 0);
        TEST_ASSERT(ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                        mid_only, weights_raw, weight_alloc, 0,
                        weight1_offset, in_dim, out_dim, x, 0.0f) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(ref_gate, 0, ref_gate_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(ref_up, 0, ref_up_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(ref_mid, 0, ref_mid_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(fused_gate, 0, fused_gate_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(fused_up, 0, fused_up_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(fused_mid, 0, fused_mid_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(mid_only, 0, mid_only_host, out_bytes) != 0);
        const test_float_compare_stats gate_stats =
            test_compare_float_bits(ref_gate_host, fused_gate_host, out_dim);
        const test_float_compare_stats up_stats =
            test_compare_float_bits(ref_up_host, fused_up_host, out_dim);
        const test_float_compare_stats mid_stats =
            test_compare_float_bits(ref_mid_host, fused_mid_host, out_dim);
        const test_float_compare_stats mid_only_stats =
            test_compare_float_bits(ref_mid_host, mid_only_host, out_dim);
        fprintf(stderr,
                "ds4-test: Laguna dense fused case=%u gate=%zu up=%zu "
                "mid=%zu mid_only=%zu\n",
                ci,
                gate_stats.mismatch_count,
                up_stats.mismatch_count,
                mid_stats.mismatch_count,
                mid_only_stats.mismatch_count);
        TEST_ASSERT(gate_stats.mismatch_count == 0u);
        TEST_ASSERT(up_stats.mismatch_count == 0u);
        TEST_ASSERT(mid_stats.mismatch_count == 0u);
        TEST_ASSERT(mid_only_stats.mismatch_count == 0u);

    }

    /* A production-sized call with a truncated mapped range must fail instead
     * of switching to the stock path after the fused selector was requested. */
    TEST_ASSERT(ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                    mid_only, weights_raw, weight1_offset,
                    0, weight1_offset, in_dim, out_dim, x, 0.0f) == 0);

    free(mid_only_host);
    free(fused_mid_host);
    free(fused_up_host);
    free(fused_gate_host);
    free(ref_mid_host);
    free(ref_up_host);
    free(ref_gate_host);
    free(x_host);
    ds4_gpu_tensor_free(mid_only);
    ds4_gpu_tensor_free(fused_mid);
    ds4_gpu_tensor_free(fused_up);
    ds4_gpu_tensor_free(fused_gate);
    ds4_gpu_tensor_free(ref_mid);
    ds4_gpu_tensor_free(ref_up);
    ds4_gpu_tensor_free(ref_gate);
    ds4_gpu_tensor_free(x);
    free(weights_raw);
    test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
}

/* Q8 dispatch selectors are a process-lifecycle snapshot, not per-world
 * getenv probes.  Mutating either selector after the first Metal probe must
 * leave the descriptor configuration unchanged; cleanup is intentionally not
 * a reset boundary. */
static void test_metal_q8_decode_lifecycle_snapshot(void) {
    ds4_gpu_q8_decode_config first;
    ds4_gpu_q8_decode_config mutated;
    TEST_ASSERT(ds4_gpu_q8_decode_config_snapshot(&first) > 0);
    const int expected_world1 = first.q8_mv_nsg_override > 0 ?
        first.q8_mv_nsg_override : 4;
    const int expected_world2 = first.q8_mv_nsg_override > 0 ?
        first.q8_mv_nsg_override : 2;
    TEST_ASSERT(ds4_gpu_test_q8_decode_nsg_for_world(1) == expected_world1);
    TEST_ASSERT(ds4_gpu_test_q8_decode_nsg_for_world(2) == expected_world2);

    char *saved_nsg = test_save_env("DS4_METAL_Q8_MV_NSG");
    char *saved_rows = test_save_env("DS4_METAL_Q8_MV_ROWS");
    TEST_ASSERT(setenv("DS4_METAL_Q8_MV_NSG", "8", 1) == 0);
    TEST_ASSERT(setenv("DS4_METAL_Q8_MV_ROWS", "4", 1) == 0);
    TEST_ASSERT(ds4_gpu_q8_decode_config_snapshot(&mutated) > 0);
    TEST_ASSERT(mutated.q8_mv_nsg_override == first.q8_mv_nsg_override);
    TEST_ASSERT(mutated.q8_mv_rows == first.q8_mv_rows);
    TEST_ASSERT(ds4_gpu_test_q8_decode_nsg_for_world(1) == expected_world1);
    TEST_ASSERT(ds4_gpu_test_q8_decode_nsg_for_world(2) == expected_world2);

    TEST_ASSERT(unsetenv("DS4_METAL_Q8_MV_NSG") == 0);
    TEST_ASSERT(unsetenv("DS4_METAL_Q8_MV_ROWS") == 0);
    TEST_ASSERT(ds4_gpu_q8_decode_config_snapshot(&mutated) > 0);
    TEST_ASSERT(mutated.q8_mv_nsg_override == first.q8_mv_nsg_override);
    TEST_ASSERT(mutated.q8_mv_rows == first.q8_mv_rows);
    TEST_ASSERT(ds4_gpu_test_q8_decode_nsg_for_world(1) == expected_world1);
    TEST_ASSERT(ds4_gpu_test_q8_decode_nsg_for_world(2) == expected_world2);

    /* Cleanup releases Metal objects but deliberately does not reopen the
     * process snapshot.  Re-init must keep the same world-1/world-2
     * descriptors even after the environment was mutated and then unset. */
    ds4_gpu_cleanup();
    TEST_ASSERT(ds4_gpu_init() != 0);
    TEST_ASSERT(ds4_gpu_q8_decode_config_snapshot(&mutated) > 0);
    TEST_ASSERT(mutated.q8_mv_nsg_override == first.q8_mv_nsg_override);
    TEST_ASSERT(mutated.q8_mv_rows == first.q8_mv_rows);
    TEST_ASSERT(ds4_gpu_test_q8_decode_nsg_for_world(1) == expected_world1);
    TEST_ASSERT(ds4_gpu_test_q8_decode_nsg_for_world(2) == expected_world2);
    test_restore_env("DS4_METAL_Q8_MV_NSG", saved_nsg);
    test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
}

/* Run alone in a fresh process so the Q8 process snapshot is still unset.
 * Both malformed selectors must fail before raw-graph memset/allocation and
 * leave the caller-owned command state untouched. */
static void test_metal_graph_malformed_q8_admission(void) {
    char *saved_nsg = test_save_env("DS4_METAL_Q8_MV_NSG");
    char *saved_rows = test_save_env("DS4_METAL_Q8_MV_ROWS");
    TEST_ASSERT(setenv("DS4_METAL_Q8_MV_NSG", "not-a-number", 1) == 0);
    TEST_ASSERT(setenv("DS4_METAL_Q8_MV_ROWS", "not-a-number", 1) == 0);
    TEST_ASSERT(!ds4_gpu_commands_active());
    TEST_ASSERT(ds4_test_raw_graph_preflight_failure_state(1) != 0);
    TEST_ASSERT(!ds4_gpu_commands_active());
    fprintf(stderr,
            "ds4-test: malformed Q8 snapshot rejected before raw graph "
            "mutation\n");
    test_restore_env("DS4_METAL_Q8_MV_NSG", saved_nsg);
    test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
}

/* Run alone with an optional old DENSE source override. If the current source
 * has the output PSO, a deliberately invalid weight shape still exercises the
 * same preflight boundary. With the old source, the valid-shape branch proves
 * the unavailable PSO fails before graph/KV/command state changes. */
static void test_metal_graph_output_preflight_failure(void) {
    char *saved_output = test_save_env(
        "DS4_METAL_LAGUNA_OUTPUT_HEAD_NORM_FUSE");
    TEST_ASSERT(setenv("DS4_METAL_LAGUNA_OUTPUT_HEAD_NORM_FUSE", "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_init() != 0);
    const int output_pso = ds4_gpu_matmul_f16_rms_norm_mv_preflight(
        16384u, 4u);
    TEST_ASSERT(!ds4_gpu_commands_active());
    TEST_ASSERT(ds4_test_raw_graph_preflight_failure_state(
                    output_pso == 0 ? 1 : 0) != 0);
    TEST_ASSERT(!ds4_gpu_commands_active());
    fprintf(stderr,
            "ds4-test: output-head selector preflight failed cleanly "
            "(optional_pso=%s)\n",
            output_pso ? "available" : "unavailable");
    ds4_gpu_cleanup();
    test_restore_env("DS4_METAL_LAGUNA_OUTPUT_HEAD_NORM_FUSE", saved_output);
}

/* The focused selector starts in a fresh process, so use it to prove that a
 * fast-compiled Metal library cannot be certified merely by setting
 * DS4_METAL_MATH_SAFE after initialization.  The ordinary Metal suite does
 * not pay this cleanup/recompile cost. */
static void test_metal_laguna_q8_lmhead_screen_compile_gate(void) {
    if (!test_env_bool("DS4_TEST_LAGUNA_Q8_LMHEAD_SCREEN") ||
        !test_env_bool("DS4_METAL_MATH_SAFE")) {
        return;
    }
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t valid_bytes =
        100352ull * (3072ull / 32ull) * 34ull;
    const uint64_t valid_alloc = test_round_up_u64(valid_bytes, page);
    void *dummy = NULL;
    TEST_ASSERT(posix_memalign(&dummy, (size_t)page,
                               (size_t)valid_alloc) == 0);
    if (!dummy) return;
    char *saved_math = test_save_env("DS4_METAL_MATH_SAFE");
    char *saved_mpp = test_save_env("DS4_METAL_Q8_DECODE_MPP");
    char *saved_rows = test_save_env("DS4_METAL_Q8_MV_ROWS");
    char *saved_nr4 = test_save_env("DS4_METAL_ENABLE_OUTPUT_Q8_NR4");

    TEST_ASSERT(unsetenv("DS4_METAL_MATH_SAFE") == 0);
    TEST_ASSERT(unsetenv("DS4_METAL_Q8_DECODE_MPP") == 0);
    TEST_ASSERT(unsetenv("DS4_METAL_Q8_MV_ROWS") == 0);
    TEST_ASSERT(unsetenv("DS4_METAL_ENABLE_OUTPUT_Q8_NR4") == 0);
    TEST_ASSERT(ds4_gpu_init() != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(dummy, valid_alloc) != 0);
    TEST_ASSERT(setenv("DS4_METAL_MATH_SAFE", "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_create(
                    dummy, valid_alloc, 0, 3072u, 100352u) == NULL);

    /* No tensor handles or screen plan exist in this helper.  Recompile the
     * library safely for the actual focused test and restore the valid map. */
    ds4_gpu_cleanup();
    test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
    test_restore_env("DS4_METAL_Q8_DECODE_MPP", saved_mpp);
    test_restore_env("DS4_METAL_ENABLE_OUTPUT_Q8_NR4", saved_nr4);
    test_restore_env("DS4_METAL_MATH_SAFE", saved_math);
    TEST_ASSERT(ds4_gpu_init() != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(dummy, valid_alloc) != 0);
    free(dummy);
}

static void test_metal_laguna_q8_lmhead_screen_gates(void) {
    if (!test_env_bool("DS4_TEST_LAGUNA_Q8_LMHEAD_SCREEN") ||
        !test_env_bool("DS4_METAL_MATH_SAFE")) {
        return;
    }
    /* Gate validation is deliberately cheap: create() must reject before it
     * touches the production-sized sidecopy for every incompatible mode. */
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t valid_bytes =
        100352ull * (3072ull / 32ull) * 34ull;
    const uint64_t valid_alloc = test_round_up_u64(valid_bytes, page);
    void *dummy = NULL;
    TEST_ASSERT(posix_memalign(&dummy, (size_t)page,
                               (size_t)valid_alloc) == 0);
    if (!dummy) return;
    char *saved_math = test_save_env("DS4_METAL_MATH_SAFE");
    char *saved_mpp = test_save_env("DS4_METAL_Q8_DECODE_MPP");
    char *saved_rows = test_save_env("DS4_METAL_Q8_MV_ROWS");
    char *saved_nr4 = test_save_env("DS4_METAL_ENABLE_OUTPUT_Q8_NR4");
    char *saved_dense = test_save_env(
        "DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU");
    char *saved_v2 = test_save_env(
        "DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2");
    char *saved_v2_fallback = test_save_env(
        "DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2_FALLBACK");
    char *saved_v2_disable_indirect = test_save_env(
        "DS4_METAL_LAGUNA_Q8_LMHEAD_V2_DISABLE_INDIRECT");

    /* A focused invocation may be the first Metal caller. Compile the
     * library under the caller's safe setting before exercising post-init
     * environment changes. */
    if (saved_math && strcmp(saved_math, "0") != 0) {
        TEST_ASSERT(ds4_gpu_init() != 0);
    }

    /* Register the production-sized range before selector checks.  Without
     * this, a deleted eligibility gate could still be masked by the later
     * model-range wrapper returning NULL. */
    TEST_ASSERT(ds4_gpu_init() != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(dummy, valid_alloc) != 0);

    TEST_ASSERT(unsetenv("DS4_METAL_MATH_SAFE") == 0);
    TEST_ASSERT(unsetenv("DS4_METAL_Q8_DECODE_MPP") == 0);
    TEST_ASSERT(unsetenv("DS4_METAL_Q8_MV_ROWS") == 0);
    TEST_ASSERT(unsetenv("DS4_METAL_ENABLE_OUTPUT_Q8_NR4") == 0);
    TEST_ASSERT(unsetenv("DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2") == 0);
    TEST_ASSERT(unsetenv(
                    "DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2_FALLBACK") == 0);
    TEST_ASSERT(unsetenv(
                    "DS4_METAL_LAGUNA_Q8_LMHEAD_V2_DISABLE_INDIRECT") == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_create(
                    dummy, valid_alloc, 0, 3072u, 100352u) == NULL);

    /* Setting the environment after a fast-math library was initialized must
     * not falsely certify the screen.  The implementation also checks the
     * current flag, so removing it after a safe compile suppresses the path. */
    TEST_ASSERT(setenv("DS4_METAL_MATH_SAFE", "1", 1) == 0);
    TEST_ASSERT(setenv("DS4_METAL_Q8_DECODE_MPP", "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_create(
                    dummy, valid_alloc, 0, 3072u, 100352u) == NULL);
    TEST_ASSERT(unsetenv("DS4_METAL_Q8_DECODE_MPP") == 0);
    TEST_ASSERT(setenv("DS4_METAL_Q8_MV_ROWS", "4", 1) == 0);
    /* Q8 rows are frozen at the first GPU lifecycle probe.  A post-init
     * mutation therefore cannot turn the certified NR2 screen off. */
    ds4_gpu_q8_decode_config q8_snapshot;
    TEST_ASSERT(ds4_gpu_q8_decode_config_snapshot(&q8_snapshot) > 0);
    if (q8_snapshot.q8_mv_rows == 2) {
        ds4_gpu_laguna_q8_lmhead_screen *snapshot_screen =
            ds4_gpu_laguna_q8_lmhead_screen_create(
                dummy, valid_alloc, 0, 3072u, 100352u);
        TEST_ASSERT(snapshot_screen != NULL);
        ds4_gpu_laguna_q8_lmhead_screen_destroy(snapshot_screen);
    } else {
        TEST_ASSERT(q8_snapshot.q8_mv_rows == 4);
        TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_create(
                        dummy, valid_alloc, 0, 3072u, 100352u) == NULL);
    }
    TEST_ASSERT(unsetenv("DS4_METAL_Q8_MV_ROWS") == 0);
    /* The dense selector and the screen share the same certified NR2
     * dispatch.  Exact literal 2 is compatible when both are requested. */
    TEST_ASSERT(setenv("DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU", "1", 1) == 0);
    TEST_ASSERT(setenv("DS4_METAL_Q8_MV_ROWS", "2", 1) == 0);
    ds4_gpu_laguna_q8_lmhead_screen *combined_screen =
        ds4_gpu_laguna_q8_lmhead_screen_create(
            dummy, valid_alloc, 0, 3072u, 100352u);
    TEST_ASSERT(combined_screen != NULL);
    ds4_gpu_laguna_q8_lmhead_screen_destroy(combined_screen);
    TEST_ASSERT(unsetenv("DS4_METAL_Q8_MV_ROWS") == 0);
    TEST_ASSERT(setenv("DS4_METAL_ENABLE_OUTPUT_Q8_NR4", "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_create(
                    dummy, valid_alloc, 0, 3072u, 100352u) == NULL);
    TEST_ASSERT(unsetenv("DS4_METAL_ENABLE_OUTPUT_Q8_NR4") == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_create(
                    dummy, valid_alloc, 0, 3072u, 100351u) == NULL);

    /* v2's selector is strict, and unavailable indirect dispatch has explicit
     * require/fallback behavior rather than silently changing benchmark arms. */
    TEST_ASSERT(setenv("DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2", "bad", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_create(
                    dummy, valid_alloc, 0, 3072u, 100352u) == NULL);
    TEST_ASSERT(setenv("DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2", "1", 1) == 0);
    TEST_ASSERT(setenv("DS4_METAL_LAGUNA_Q8_LMHEAD_V2_DISABLE_INDIRECT",
                       "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_create(
                    dummy, valid_alloc, 0, 3072u, 100352u) == NULL);
    TEST_ASSERT(setenv("DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2_FALLBACK",
                       "1", 1) == 0);
    ds4_gpu_laguna_q8_lmhead_screen *fallback_screen =
        ds4_gpu_laguna_q8_lmhead_screen_create(
            dummy, valid_alloc, 0, 3072u, 100352u);
    TEST_ASSERT(fallback_screen != NULL);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_v2_enabled(
                    fallback_screen) == 0);
    ds4_gpu_laguna_q8_lmhead_screen_destroy(fallback_screen);

    test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
    test_restore_env("DS4_METAL_Q8_DECODE_MPP", saved_mpp);
    test_restore_env("DS4_METAL_ENABLE_OUTPUT_Q8_NR4", saved_nr4);
    test_restore_env("DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU",
                     saved_dense);
    test_restore_env("DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2",
                     saved_v2);
    test_restore_env("DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2_FALLBACK",
                     saved_v2_fallback);
    test_restore_env("DS4_METAL_LAGUNA_Q8_LMHEAD_V2_DISABLE_INDIRECT",
                     saved_v2_disable_indirect);
    test_restore_env("DS4_METAL_MATH_SAFE", saved_math);
    free(dummy);
}

static void test_metal_laguna_q8_lmhead_screen(void) {
    /* The production sidecopy is intentionally large. Keep this opt-in so
     * ordinary kernel CI remains bounded; focused M5 runs set both guards. */
    if (!test_env_bool("DS4_TEST_LAGUNA_Q8_LMHEAD_SCREEN")) {
        fprintf(stderr,
                "ds4-test: Laguna Q8 lm-head screen skipped "
                "(set DS4_TEST_LAGUNA_Q8_LMHEAD_SCREEN=1)\n");
        return;
    }
    if (!test_env_bool("DS4_METAL_MATH_SAFE")) {
        fprintf(stderr,
                "ds4-test: Laguna Q8 lm-head screen skipped "
                "(requires DS4_METAL_MATH_SAFE=1 before Metal init)\n");
        return;
    }

    const uint32_t in_dim = 3072u;
    const uint32_t out_dim = 100352u;
    const uint32_t n_blocks = in_dim / 32u;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes = (uint64_t)n_blocks * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, page);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)page,
                               (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights((uint8_t *)weights_raw, in_dim, out_dim, 211u);

    /* Make a duplicated, cross-NR0 maximum and exercise both signed limits. */
    uint8_t *row0 = (uint8_t *)weights_raw;
    uint8_t *row100 = row0 + (uint64_t)100u * row_bytes;
    memcpy(row100, row0, (size_t)row_bytes);
    uint16_t scale_one = test_float_to_f16(1.0f);
    uint16_t scale_large = test_float_to_f16(65504.0f);
    memcpy(row0, &scale_one, sizeof(scale_one));
    memcpy(row100, &scale_one, sizeof(scale_one));
    int8_t *q0 = (int8_t *)(row0 + 2u);
    int8_t *q100 = (int8_t *)(row100 + 2u);
    q0[0] = 127;
    q0[1] = -128;
    q100[0] = 127;
    q100[1] = -128;
    /* Make the first component a deterministic dominant pair while leaving
     * the remaining block data nontrivial.  This also gives the extreme-input
     * cases a known sentinel/tie shape. */
    for (uint32_t row = 0; row < out_dim; row++) {
        uint8_t *raw = (uint8_t *)weights_raw + (uint64_t)row * row_bytes;
        memcpy(raw, &scale_large, sizeof(scale_large));
        ((int8_t *)(raw + 2u))[0] = 2;
    }
    memcpy(row100, row0, (size_t)row_bytes);
    q0[0] = 127;
    q0[1] = -128;
    q100[0] = 127;
    q100[1] = -128;
    /* A nonfinite scale must force this row through exact evaluation. */
    uint16_t scale_nan = 0x7e00u;
    uint8_t *row200 = (uint8_t *)weights_raw + (uint64_t)200u * row_bytes;
    memcpy(row200, &scale_nan, sizeof(scale_nan));
    /* Reserve a later block for a nonzero +Inf winner.  Every row starts with
     * q=0 at block 8/lane 0; row 300 gets a finite large scale and q=127 so
     * the +FLT_MAX probe cannot falsely pass if the reducer skips +Inf while
     * defaulting to row 0. */
    for (uint32_t row = 0; row < out_dim; row++) {
        uint8_t *block8 = (uint8_t *)weights_raw +
            (uint64_t)row * row_bytes + 8u * 34u;
        ((int8_t *)(block8 + 2u))[0] = 0;
    }
    uint8_t *row300_block8 = (uint8_t *)weights_raw +
        (uint64_t)300u * row_bytes + 8u * 34u;
    memcpy(row300_block8, &scale_large, sizeof(scale_large));
    ((int8_t *)(row300_block8 + 2u))[0] = 127;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *reference_idx = ds4_gpu_tensor_alloc(sizeof(int32_t));
    ds4_gpu_tensor *screen_idx = ds4_gpu_tensor_alloc(sizeof(int32_t));
    ds4_gpu_tensor *screen_value = ds4_gpu_tensor_alloc(sizeof(float));
    ds4_gpu_tensor *poison_logits = ds4_gpu_tensor_alloc(logits_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(reference != NULL);
    TEST_ASSERT(reference_idx != NULL);
    TEST_ASSERT(screen_idx != NULL);
    TEST_ASSERT(screen_value != NULL);
    TEST_ASSERT(poison_logits != NULL);
    if (!x || !reference || !reference_idx || !screen_idx ||
        !screen_value || !poison_logits) goto cleanup;

    char *saved_mpp = test_save_env("DS4_METAL_Q8_DECODE_MPP");
    char *saved_rows = test_save_env("DS4_METAL_Q8_MV_ROWS");
    char *saved_nr4 = test_save_env("DS4_METAL_ENABLE_OUTPUT_Q8_NR4");
    TEST_ASSERT(unsetenv("DS4_METAL_Q8_DECODE_MPP") == 0);
    TEST_ASSERT(unsetenv("DS4_METAL_Q8_MV_ROWS") == 0);
    TEST_ASSERT(unsetenv("DS4_METAL_ENABLE_OUTPUT_Q8_NR4") == 0);
    TEST_ASSERT(ds4_gpu_init() != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);

    float *x_host = calloc(in_dim, sizeof(float));
    float *reference_host = malloc((size_t)logits_bytes);
    float *poison_host = malloc((size_t)logits_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(reference_host != NULL);
    TEST_ASSERT(poison_host != NULL);
    if (!x_host || !reference_host || !poison_host) {
        free(poison_host);
        free(reference_host);
        free(x_host);
        test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
        test_restore_env("DS4_METAL_Q8_DECODE_MPP", saved_mpp);
        test_restore_env("DS4_METAL_ENABLE_OUTPUT_Q8_NR4", saved_nr4);
        goto cleanup;
    }

    x_host[0] = 1.0f;
    for (uint32_t i = 0; i < out_dim; i++) {
        const uint32_t bits = 0x7fc00001u + (i & 0x3ffu);
        memcpy(poison_host + i, &bits, sizeof(bits));
    }
    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(reference, 0, poison_host,
                                     logits_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(poison_logits, 0, poison_host,
                                     logits_bytes) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                    reference, weights_raw, weight_alloc, 0,
                    in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_laguna_argmax_tensor(
                    reference_idx, reference, out_dim) != 0);

    const bool requested_v2 = getenv(
        "DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2") &&
        strcmp(getenv("DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2"), "1") == 0;
    const bool allow_v2_fallback = getenv(
        "DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2_FALLBACK") &&
        strcmp(getenv("DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2_FALLBACK"),
               "1") == 0;
    ds4_gpu_laguna_q8_lmhead_screen *screen =
        ds4_gpu_laguna_q8_lmhead_screen_create(
            weights_raw, weight_alloc, 0, in_dim, out_dim);
    TEST_ASSERT(screen != NULL);
    if (!screen) {
        free(poison_host);
        free(reference_host);
        free(x_host);
        test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
        test_restore_env("DS4_METAL_Q8_DECODE_MPP", saved_mpp);
        test_restore_env("DS4_METAL_ENABLE_OUTPUT_Q8_NR4", saved_nr4);
        goto cleanup;
    }
    const bool using_v2 =
        ds4_gpu_laguna_q8_lmhead_screen_v2_enabled(screen) != 0;
    if (requested_v2 && !allow_v2_fallback) {
        TEST_ASSERT(using_v2);
    }

    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    screen, screen_idx, screen_value,
                    weights_raw, weight_alloc, 0, x) != 0);
    int32_t ref_idx = -1;
    int32_t got_idx = -1;
    float got_value = 0.0f;
    TEST_ASSERT(ds4_gpu_tensor_read(reference_idx, 0,
                                    &ref_idx, sizeof(ref_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_idx, 0,
                                    &got_idx, sizeof(got_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_value, 0,
                                    &got_value, sizeof(got_value)) != 0);
    TEST_ASSERT(got_idx == ref_idx);
    TEST_ASSERT(ref_idx >= 0 && ref_idx < (int32_t)out_dim);
    TEST_ASSERT(ref_idx == 0);
    /* Read the winning stock value separately so the assertion is bitwise. */
    TEST_ASSERT(ds4_gpu_tensor_read(reference,
                                    (uint64_t)ref_idx * sizeof(float),
                                    reference_host + ref_idx,
                                    sizeof(float)) != 0);
    TEST_ASSERT(memcmp(&got_value, reference_host + ref_idx,
                       sizeof(got_value)) == 0);
    uint32_t candidate_rows = 0;
    uint32_t candidate_row_blocks = 0;
    uint32_t coarse_nonfinite = 0;
    uint64_t dispatch_count = 0;
    uint64_t packed_bytes = 0;
    double sidecopy_init_ms = 0.0;
    uint32_t exact_row_blocks = 0;
    int32_t stat_idx = -1;
    float stat_value = 0.0f;
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_stats(
                    screen, &candidate_rows, &candidate_row_blocks,
                    &coarse_nonfinite, &dispatch_count,
                    &packed_bytes, &sidecopy_init_ms, &exact_row_blocks,
                    &stat_idx, &stat_value) != 0);
    fprintf(stderr,
            "ds4-test: Laguna Q8 lm-head screen candidates=%u/%u "
            "row_blocks=%u exact_row_blocks=%u nonfinite=%u "
            "screen_calls=%llu packed_bytes=%llu sidecopy_init_ms=%.3f\n",
            candidate_rows, out_dim, candidate_row_blocks,
            exact_row_blocks, coarse_nonfinite,
            (unsigned long long)dispatch_count,
            (unsigned long long)packed_bytes, sidecopy_init_ms);
    TEST_ASSERT(dispatch_count > 0u);
    TEST_ASSERT(packed_bytes == 173408256ull);
    TEST_ASSERT(sidecopy_init_ms >= 0.0);
    TEST_ASSERT(candidate_rows == 3u);
    TEST_ASSERT(candidate_row_blocks == 288u);
    TEST_ASSERT(exact_row_blocks == 576u);
    TEST_ASSERT(coarse_nonfinite == 1u);
    if (using_v2) {
        uint32_t compact_pair_count = 0;
        uint32_t exact_dispatch_groups = 0;
        TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_stats_v2(
                        screen, &compact_pair_count,
                        &exact_dispatch_groups) != 0);
        TEST_ASSERT(compact_pair_count == 3u);
        TEST_ASSERT(exact_dispatch_groups == 3u);
    } else {
        TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_stats_v2(
                        screen, NULL, NULL) == 0);
    }
    TEST_ASSERT(stat_idx == got_idx);
    TEST_ASSERT(memcmp(&stat_value, &got_value, sizeof(float)) == 0);

    /* The screen must not materialize or overwrite a caller's logits scratch. */
    TEST_ASSERT(ds4_gpu_tensor_read(poison_logits, 0, poison_host,
                                    logits_bytes) != 0);
    for (uint32_t i = 0; i < out_dim; i++) {
        uint32_t bits = 0;
        memcpy(&bits, poison_host + i, sizeof(bits));
        TEST_ASSERT(bits == 0x7fc00001u + (i & 0x3ffu));
    }

    /* Repeat once inside an explicit command batch to cover the graph-style
     * encoder path as well as the standalone API call above. */
    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    screen, screen_idx, screen_value,
                    weights_raw, weight_alloc, 0, x) != 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    screen, screen_idx, screen_value,
                    weights_raw, weight_alloc, 0, x) == 0);
    TEST_ASSERT(ds4_gpu_end_commands() != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference_idx, 0,
                                    &ref_idx, sizeof(ref_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_idx, 0,
                                    &got_idx, sizeof(got_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_value, 0,
                                    &got_value, sizeof(got_value)) != 0);
    TEST_ASSERT(got_idx == ref_idx);
    TEST_ASSERT(ds4_gpu_tensor_read(reference,
                                    (uint64_t)ref_idx * sizeof(float),
                                    reference_host + ref_idx,
                                    sizeof(float)) != 0);
    TEST_ASSERT(memcmp(&got_value, reference_host + ref_idx,
                       sizeof(got_value)) == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_stats(
                    screen, &candidate_rows, &candidate_row_blocks,
                    &coarse_nonfinite, &dispatch_count,
                    &packed_bytes, &sidecopy_init_ms, &exact_row_blocks,
                    &stat_idx, &stat_value) != 0);
    TEST_ASSERT(dispatch_count == 2u);
    TEST_ASSERT(candidate_rows == 3u);
    TEST_ASSERT(candidate_row_blocks == 288u);
    TEST_ASSERT(exact_row_blocks == 576u);
    TEST_ASSERT(coarse_nonfinite == 1u);
    if (using_v2) {
        uint32_t compact_pair_count = 0;
        uint32_t exact_dispatch_groups = 0;
        TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_stats_v2(
                        screen, &compact_pair_count,
                        &exact_dispatch_groups) != 0);
        TEST_ASSERT(compact_pair_count == 3u);
        TEST_ASSERT(exact_dispatch_groups == 3u);
    }
    TEST_ASSERT(stat_idx == got_idx);
    TEST_ASSERT(memcmp(&stat_value, &got_value, sizeof(float)) == 0);

    /* Dense finite probes exercise all 96 packed blocks and block-boundary
     * lanes, not just the first quantized byte.  Keep the activations inside
     * the certificate's safe interval and verify both winner bits and a
     * genuinely pruned candidate set for two deterministic seeds. */
    for (uint32_t case_id = 0; case_id < 2u; case_id++) {
        uint32_t state = 0x243f6a88u ^ (case_id * 0x9e3779b9u);
        for (uint32_t i = 0; i < in_dim; i++) {
            state = state * 1664525u + 1013904223u;
            const float unit = (float)(state >> 8) * (1.0f / 16777216.0f);
            const float magnitude = 0.125f + unit * 0.875f;
            x_host[i] = (state & 0x80000000u) ? -magnitude : magnitude;
        }
        x_host[0] = 1.0f;
        x_host[31] = -0.75f;
        x_host[32] = 0.625f;
        x_host[63] = -0.5f;
        x_host[64] = 0.375f;
        x_host[95] = -0.25f;
        x_host[96] = 0.1875f;
        x_host[in_dim - 1u] = -0.15625f;
        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                        reference, weights_raw, weight_alloc, 0,
                        in_dim, out_dim, x, 1) != 0);
        TEST_ASSERT(ds4_gpu_laguna_argmax_tensor(
                        reference_idx, reference, out_dim) != 0);
        TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_tensor(
                        screen, screen_idx, screen_value,
                        weights_raw, weight_alloc, 0, x) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(reference_idx, 0,
                                        &ref_idx, sizeof(ref_idx)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(screen_idx, 0,
                                        &got_idx, sizeof(got_idx)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(screen_value, 0,
                                        &got_value, sizeof(got_value)) != 0);
        TEST_ASSERT(got_idx == ref_idx);
        TEST_ASSERT(ds4_gpu_tensor_read(reference,
                                        (uint64_t)ref_idx * sizeof(float),
                                        reference_host + ref_idx,
                                        sizeof(float)) != 0);
        TEST_ASSERT(memcmp(&got_value, reference_host + ref_idx,
                           sizeof(got_value)) == 0);
        TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_stats(
                        screen, &candidate_rows, &candidate_row_blocks,
                        &coarse_nonfinite, &dispatch_count,
                        &packed_bytes, &sidecopy_init_ms, &exact_row_blocks,
                        &stat_idx, &stat_value) != 0);
        TEST_ASSERT(candidate_rows > 0u && candidate_rows < out_dim);
        TEST_ASSERT(candidate_row_blocks == candidate_rows * n_blocks);
        TEST_ASSERT(exact_row_blocks >= candidate_row_blocks);
        TEST_ASSERT(stat_idx == got_idx);
        TEST_ASSERT(memcmp(&stat_value, &got_value, sizeof(float)) == 0);
    }

    /* Zero input admits every row but must preserve the stock lower-index tie. */
    memset(x_host, 0, (size_t)x_bytes);
    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                    reference, weights_raw, weight_alloc, 0,
                    in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_laguna_argmax_tensor(
                    reference_idx, reference, out_dim) != 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    screen, screen_idx, screen_value,
                    weights_raw, weight_alloc, 0, x) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference_idx, 0,
                                    &ref_idx, sizeof(ref_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_idx, 0,
                                    &got_idx, sizeof(got_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_value, 0,
                                    &got_value, sizeof(got_value)) != 0);
    TEST_ASSERT(got_idx == ref_idx);
    TEST_ASSERT(got_idx == 0);
    TEST_ASSERT(memcmp(&got_value, &(float){0.0f}, sizeof(float)) == 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference,
                                    (uint64_t)ref_idx * sizeof(float),
                                    reference_host + ref_idx,
                                    sizeof(float)) != 0);
    TEST_ASSERT(memcmp(&got_value, reference_host + ref_idx,
                       sizeof(got_value)) == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_stats(
                    screen, &candidate_rows, &candidate_row_blocks,
                    &coarse_nonfinite, &dispatch_count,
                    &packed_bytes, &sidecopy_init_ms, &exact_row_blocks,
                    &stat_idx, &stat_value) != 0);
    TEST_ASSERT(candidate_rows == out_dim);
    TEST_ASSERT(candidate_row_blocks == out_dim * n_blocks);
    TEST_ASSERT(exact_row_blocks == out_dim * n_blocks);
    TEST_ASSERT(coarse_nonfinite > 0u);
    TEST_ASSERT(stat_idx == got_idx);
    TEST_ASSERT(memcmp(&stat_value, &got_value, sizeof(float)) == 0);

    /* Explicit -0 input checks normalized ordering and bitwise parity with
     * the stock Q8 output; the screen does not promise a raw -0 payload. */
    memset(x_host, 0, (size_t)x_bytes);
    {
        const uint32_t negative_zero_bits = 0x80000000u;
        memcpy(&x_host[0], &negative_zero_bits, sizeof(negative_zero_bits));
    }
    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                    reference, weights_raw, weight_alloc, 0,
                    in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_laguna_argmax_tensor(
                    reference_idx, reference, out_dim) != 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    screen, screen_idx, screen_value,
                    weights_raw, weight_alloc, 0, x) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference_idx, 0,
                                    &ref_idx, sizeof(ref_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_idx, 0,
                                    &got_idx, sizeof(got_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_value, 0,
                                    &got_value, sizeof(got_value)) != 0);
    TEST_ASSERT(got_idx == ref_idx);
    TEST_ASSERT(got_idx == 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference,
                                    (uint64_t)ref_idx * sizeof(float),
                                    reference_host + ref_idx,
                                    sizeof(float)) != 0);
    TEST_ASSERT(memcmp(&got_value, reference_host + ref_idx,
                       sizeof(got_value)) == 0);
    TEST_ASSERT(got_value == 0.0f);

    /* Unsafe subnormal input must force every row through exact evaluation,
     * not rely on the coarse A bound. */
    memset(x_host, 0, (size_t)x_bytes);
    x_host[0] = ldexpf(1.0f, -100);
    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                    reference, weights_raw, weight_alloc, 0,
                    in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_laguna_argmax_tensor(
                    reference_idx, reference, out_dim) != 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    screen, screen_idx, screen_value,
                    weights_raw, weight_alloc, 0, x) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference_idx, 0,
                                    &ref_idx, sizeof(ref_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_idx, 0,
                                    &got_idx, sizeof(got_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_value, 0,
                                    &got_value, sizeof(got_value)) != 0);
    TEST_ASSERT(got_idx == ref_idx);
    TEST_ASSERT(ds4_gpu_tensor_read(reference,
                                    (uint64_t)ref_idx * sizeof(float),
                                    reference_host + ref_idx,
                                    sizeof(float)) != 0);
    TEST_ASSERT(memcmp(&got_value, reference_host + ref_idx,
                       sizeof(got_value)) == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_stats(
                    screen, &candidate_rows, &candidate_row_blocks,
                    &coarse_nonfinite, &dispatch_count,
                    &packed_bytes, &sidecopy_init_ms, &exact_row_blocks,
                    &stat_idx, &stat_value) != 0);
    TEST_ASSERT(candidate_rows == out_dim);
    TEST_ASSERT(candidate_row_blocks == out_dim * n_blocks);
    TEST_ASSERT(exact_row_blocks == out_dim * n_blocks);
    TEST_ASSERT(coarse_nonfinite > 0u);

    /* Nonfinite exact values are valid stock ordering values: +Inf wins,
     * while NaNs are skipped. */
    memset(x_host, 0, (size_t)x_bytes);
    {
        const uint32_t positive_max_bits = 0x7f7fffffu;
        memcpy(&x_host[256], &positive_max_bits, sizeof(positive_max_bits));
    }
    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                    reference, weights_raw, weight_alloc, 0,
                    in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_laguna_argmax_tensor(
                    reference_idx, reference, out_dim) != 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    screen, screen_idx, screen_value,
                    weights_raw, weight_alloc, 0, x) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference_idx, 0,
                                    &ref_idx, sizeof(ref_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_idx, 0,
                                    &got_idx, sizeof(got_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_value, 0,
                                    &got_value, sizeof(got_value)) != 0);
    TEST_ASSERT(got_idx == ref_idx);
    TEST_ASSERT(got_idx == 300);
    TEST_ASSERT(ds4_gpu_tensor_read(reference,
                                    (uint64_t)ref_idx * sizeof(float),
                                    reference_host + ref_idx,
                                    sizeof(float)) != 0);
    TEST_ASSERT(memcmp(&got_value, reference_host + ref_idx,
                       sizeof(got_value)) == 0);
    {
        uint32_t got_bits = 0;
        memcpy(&got_bits, &got_value, sizeof(got_bits));
        TEST_ASSERT(got_bits == 0x7f800000u);
    }

    memset(x_host, 0, (size_t)x_bytes);
    {
        const uint32_t negative_max_bits = 0xff7fffffu;
        memcpy(&x_host[0], &negative_max_bits, sizeof(negative_max_bits));
    }
    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                    reference, weights_raw, weight_alloc, 0,
                    in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_laguna_argmax_tensor(
                    reference_idx, reference, out_dim) != 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    screen, screen_idx, screen_value,
                    weights_raw, weight_alloc, 0, x) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference_idx, 0,
                                    &ref_idx, sizeof(ref_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_idx, 0,
                                    &got_idx, sizeof(got_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_value, 0,
                                    &got_value, sizeof(got_value)) != 0);
    TEST_ASSERT(got_idx == ref_idx);
    TEST_ASSERT(got_idx == 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference,
                                    (uint64_t)ref_idx * sizeof(float),
                                    reference_host + ref_idx,
                                    sizeof(float)) != 0);
    TEST_ASSERT(memcmp(&got_value, reference_host + ref_idx,
                       sizeof(got_value)) == 0);
    {
        uint32_t got_bits = 0;
        memcpy(&got_bits, &got_value, sizeof(got_bits));
        TEST_ASSERT(got_bits == 0xff800000u);
    }

    /* Safe-boundary input keeps every exact value below the -1e30 sentinel;
     * stock therefore keeps index zero even though its winning value is a
     * finite large negative.  Row zero must still be admitted for its bits. */
    memset(x_host, 0, (size_t)x_bytes);
    x_host[0] = -0x1p90f;
    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                    reference, weights_raw, weight_alloc, 0,
                    in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_laguna_argmax_tensor(
                    reference_idx, reference, out_dim) != 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    screen, screen_idx, screen_value,
                    weights_raw, weight_alloc, 0, x) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference_idx, 0,
                                    &ref_idx, sizeof(ref_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_idx, 0,
                                    &got_idx, sizeof(got_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_value, 0,
                                    &got_value, sizeof(got_value)) != 0);
    TEST_ASSERT(got_idx == ref_idx);
    TEST_ASSERT(got_idx == 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference,
                                    (uint64_t)ref_idx * sizeof(float),
                                    reference_host + ref_idx,
                                    sizeof(float)) != 0);
    TEST_ASSERT(memcmp(&got_value, reference_host + ref_idx,
                       sizeof(got_value)) == 0);
    {
        TEST_ASSERT(isfinite(got_value));
        TEST_ASSERT(got_value < -1.0e30f);
    }

    /* NaN input is admitted everywhere and follows stock NaN skipping. */
    memset(x_host, 0, (size_t)x_bytes);
    {
        const uint32_t nan_bits = 0x7fc00000u;
        memcpy(&x_host[0], &nan_bits, sizeof(nan_bits));
    }
    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                    reference, weights_raw, weight_alloc, 0,
                    in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_laguna_argmax_tensor(
                    reference_idx, reference, out_dim) != 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    screen, screen_idx, screen_value,
                    weights_raw, weight_alloc, 0, x) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(reference_idx, 0,
                                    &ref_idx, sizeof(ref_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_idx, 0,
                                    &got_idx, sizeof(got_idx)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(screen_value, 0,
                                    &got_value, sizeof(got_value)) != 0);
    TEST_ASSERT(got_idx == ref_idx);
    TEST_ASSERT(ds4_gpu_tensor_read(reference,
                                    (uint64_t)ref_idx * sizeof(float),
                                    reference_host + ref_idx,
                                    sizeof(float)) != 0);
    TEST_ASSERT(memcmp(&got_value, reference_host + ref_idx,
                       sizeof(got_value)) == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_lmhead_screen_stats(
                    screen, &candidate_rows, &candidate_row_blocks,
                    &coarse_nonfinite, &dispatch_count,
                    &packed_bytes, &sidecopy_init_ms, &exact_row_blocks,
                    &stat_idx, &stat_value) != 0);
    TEST_ASSERT(candidate_rows == out_dim);
    TEST_ASSERT(candidate_row_blocks == out_dim * n_blocks);
    TEST_ASSERT(exact_row_blocks == out_dim * n_blocks);
    TEST_ASSERT(coarse_nonfinite > 0u);
    TEST_ASSERT(stat_idx == got_idx);
    TEST_ASSERT(memcmp(&stat_value, &got_value, sizeof(float)) == 0);
    ds4_gpu_laguna_q8_lmhead_screen_destroy(screen);
    free(poison_host);
    free(reference_host);
    free(x_host);
    test_restore_env("DS4_METAL_Q8_MV_ROWS", saved_rows);
    test_restore_env("DS4_METAL_Q8_DECODE_MPP", saved_mpp);
    test_restore_env("DS4_METAL_ENABLE_OUTPUT_Q8_NR4", saved_nr4);

cleanup:
    ds4_gpu_tensor_free(poison_logits);
    ds4_gpu_tensor_free(screen_value);
    ds4_gpu_tensor_free(screen_idx);
    ds4_gpu_tensor_free(reference_idx);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(x);
    free(weights_raw);
}

static void test_metal_laguna_q8_lmhead_screen_focused(void) {
    if (!test_env_bool("DS4_TEST_LAGUNA_Q8_LMHEAD_SCREEN") ||
        !test_env_bool("DS4_METAL_MATH_SAFE")) {
        TEST_ASSERT(false);
        return;
    }
    test_metal_laguna_q8_lmhead_screen_compile_gate();
    test_metal_laguna_q8_lmhead_screen_gates();
    test_metal_laguna_q8_lmhead_screen();
}

static void test_metal_f16_compressor_pair_state_store_exact_case(
        uint32_t width,
        uint32_t ratio,
        uint32_t pos,
        uint32_t ape_type,
        uint32_t seed,
        bool test_decode_pack) {
    const uint32_t in_dim = 4096u;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t head_dim = width / coff;
    const uint32_t state_rows = coff * ratio;
    const bool emit = ((pos + 1u) % ratio) == 0u;
    TEST_ASSERT(!test_decode_pack ||
                (ratio == 4u && emit &&
                 (head_dim == 128u || head_dim == 512u)));
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_bytes =
        (uint64_t)width * in_dim * sizeof(uint16_t);
    const uint64_t score_weight_offset =
        test_round_up_u64(weight_bytes, page);
    const uint64_t ape_offset = test_round_up_u64(
        score_weight_offset + weight_bytes, page);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * ape_elem_bytes;
    const uint64_t norm_offset =
        test_round_up_u64(ape_offset + ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)head_dim * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page,
                               (size_t)model_bytes) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_kv = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_score = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_kv = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_score = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *fused_comp = ds4_gpu_tensor_alloc(comp_bytes);

    float *x_host = malloc((size_t)x_bytes);
    float *ref_kv_host = malloc((size_t)out_bytes);
    float *ref_score_host = malloc((size_t)out_bytes);
    float *fused_kv_host = malloc((size_t)out_bytes);
    float *fused_score_host = malloc((size_t)out_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);

    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref_kv != NULL);
    TEST_ASSERT(ref_score != NULL);
    TEST_ASSERT(fused_kv != NULL);
    TEST_ASSERT(fused_score != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref_kv_host != NULL);
    TEST_ASSERT(ref_score_host != NULL);
    TEST_ASSERT(fused_kv_host != NULL);
    TEST_ASSERT(fused_score_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);

    const bool allocated = model_raw && x && ref_kv && ref_score && fused_kv &&
        fused_score && ref_state_kv && ref_state_score && fused_state_kv &&
        fused_state_score && ref_comp && fused_comp && x_host && ref_kv_host &&
        ref_score_host && fused_kv_host && fused_score_host &&
        ref_state_kv_host && ref_state_score_host && fused_state_kv_host &&
        fused_state_score_host && ref_comp_host && fused_comp_host;

    const char *pair_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ";
    const char *store_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE";
    const char *decode_pack_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION";
    const char *exact_reduction_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_EXACT_REDUCTION_FUSION";
    const char *exact_reduction_poison_env =
        "DS4_METAL_TEST_POISON_COMPRESSOR_EXACT_REDUCTION_SCRATCH";
    char *saved_pair_disable = test_save_env(pair_disable_env);
    char *saved_store_disable = test_save_env(store_disable_env);
    char *saved_decode_pack_disable =
        test_save_env(decode_pack_disable_env);
    char *saved_exact_reduction_disable =
        test_save_env(exact_reduction_disable_env);
    char *saved_exact_reduction_poison =
        test_save_env(exact_reduction_poison_env);

    test_float_compare_stats kv_stats = {0};
    test_float_compare_stats score_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};
    test_float_compare_stats comp_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        uint16_t *kv_weights = model_raw;
        uint16_t *score_weights =
            (uint16_t *)((uint8_t *)model_raw + score_weight_offset);
        for (uint32_t o = 0; o < width; o++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const int kv_value =
                    (int)((o * 17u + i * 23u + (o ^ i) * 3u +
                           seed * 29u) % 67u) - 33;
                const int score_value =
                    (int)((o * 31u + i * 11u + (o ^ (i >> 2u)) * 5u +
                           seed * 19u) % 71u) - 35;
                const uint64_t wi = (uint64_t)o * in_dim + i;
                kv_weights[wi] = test_float_to_f16(
                    (float)kv_value / 96.0f);
                score_weights[wi] = test_float_to_f16(
                    (float)score_value / 104.0f);
            }
        }

        if (ape_type == 1u) {
            uint16_t *ape =
                (uint16_t *)((uint8_t *)model_raw + ape_offset);
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 13u + (i ^ (i >> 3u)) * 7u +
                           seed * 17u) % 61u) - 30;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = (float *)((uint8_t *)model_raw + ape_offset);
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 13u + (i ^ (i >> 3u)) * 7u +
                           seed * 17u) % 61u) - 30;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f +
                (float)((i * 7u + seed * 3u) % 23u) / 64.0f;
        }

        for (uint32_t i = 0; i < in_dim; i++) {
            const int value =
                (int)((i * 29u + (i ^ (i >> 4u)) * 9u +
                       seed * 11u) % 127u) - 63;
            x_host[i] = (float)value / 88.0f;
        }
        for (uint32_t i = 0; i < width; i++) {
            const uint32_t poison = 0x7fc00001u + (i & 0x3ffu);
            memcpy(ref_kv_host + i, &poison, sizeof(poison));
            memcpy(ref_score_host + i, &poison, sizeof(poison));
            memcpy(fused_kv_host + i, &poison, sizeof(poison));
            memcpy(fused_score_host + i, &poison, sizeof(poison));
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const int kv_value =
                (int)((i * 5u + seed * 13u) % 97u) - 48;
            const int score_value =
                (int)((i * 7u + seed * 5u) % 101u) - 50;
            ref_state_kv_host[i] = (float)kv_value / 64.0f;
            fused_state_kv_host[i] = ref_state_kv_host[i];
            ref_state_score_host[i] = (float)score_value / 72.0f;
            fused_state_score_host[i] = ref_state_score_host[i];
        }
        if (test_decode_pack) {
            static const uint32_t edge_bits[8] = {
                0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
                0x3f800000u, 0xbf800000u, 0x42a00000u, 0xc2a00000u,
            };
            for (uint32_t col = 0; col < 4u; col++) {
                for (uint32_t row = 0; row < 8u; row++) {
                    const uint64_t state_col =
                        (row >= 4u ? head_dim : 0u) + col;
                    const uint64_t state_index =
                        (uint64_t)row * width + state_col;
                    const uint32_t score_bits =
                        edge_bits[(row + col) & 7u];
                    const uint32_t kv_bits =
                        edge_bits[(7u - row + col) & 7u];
                    memcpy(ref_state_score_host + state_index,
                           &score_bits, sizeof(score_bits));
                    memcpy(fused_state_score_host + state_index,
                           &score_bits, sizeof(score_bits));
                    memcpy(ref_state_kv_host + state_index,
                           &kv_bits, sizeof(kv_bits));
                    memcpy(fused_state_kv_host + state_index,
                           &kv_bits, sizeof(kv_bits));
                }
            }
        }
        for (uint32_t i = 0; i < head_dim; i++) {
            const uint32_t poison = 0x7fc01001u + (i & 0x3ffu);
            memcpy(ref_comp_host + i, &poison, sizeof(poison));
            memcpy(fused_comp_host + i, &poison, sizeof(poison));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_kv, 0, ref_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_score, 0, ref_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_kv, 0, fused_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_score, 0, fused_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(unsetenv(pair_disable_env) == 0);
        TEST_ASSERT(unsetenv(store_disable_env) == 0);
        TEST_ASSERT(setenv(exact_reduction_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(exact_reduction_poison_env) == 0);
        if (test_decode_pack) {
            TEST_ASSERT(setenv(decode_pack_disable_env, "1", 1) == 0);
        }

        TEST_ASSERT(ds4_gpu_matmul_f16_pair_tensor(
                        ref_kv, ref_score, model_raw, model_bytes,
                        0, score_weight_offset, in_dim, width, x, 1) != 0);
        TEST_ASSERT(ds4_gpu_compressor_update_tensor(
                        ref_kv, ref_score, ref_state_kv, ref_state_score,
                        ref_comp, model_raw, model_bytes, ape_offset, ape_type,
                        norm_offset, 0, head_dim, ratio, pos, 0, 0, 0,
                        10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                        1.0e-6f, false, test_decode_pack, false) != 0);

        TEST_ASSERT(ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                        fused_kv, fused_score,
                        fused_state_kv, fused_state_score,
                        model_raw, model_bytes, 0, score_weight_offset,
                        ape_offset, ape_type, in_dim, width, x,
                        ratio, pos) == 1);
        if (test_decode_pack) {
            TEST_ASSERT(unsetenv(decode_pack_disable_env) == 0);
            TEST_ASSERT(unsetenv(exact_reduction_disable_env) == 0);
            TEST_ASSERT(setenv(exact_reduction_poison_env, "1", 1) == 0);
        }
        TEST_ASSERT(ds4_gpu_compressor_update_tensor(
                        fused_kv, fused_score,
                        fused_state_kv, fused_state_score,
                        fused_comp, model_raw, model_bytes,
                        ape_offset, ape_type, norm_offset, 0,
                        head_dim, ratio, pos, 0, 0, 0,
                        10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                        1.0e-6f, true, test_decode_pack, false) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_kv, 0, ref_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_score, 0, ref_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_kv, 0, fused_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_score, 0, fused_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);

        kv_stats = test_compare_float_bits(
            ref_kv_host, fused_kv_host, width);
        score_stats = test_compare_float_bits(
            ref_score_host, fused_score_host, width);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host,
            (size_t)state_count);
        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, head_dim);
    }

    test_restore_env(pair_disable_env, saved_pair_disable);
    test_restore_env(store_disable_env, saved_store_disable);
    test_restore_env(decode_pack_disable_env, saved_decode_pack_disable);
    test_restore_env(exact_reduction_disable_env,
                     saved_exact_reduction_disable);
    test_restore_env(
        exact_reduction_poison_env, saved_exact_reduction_poison);

    fprintf(stderr,
            "ds4-test: compressor pair state-store exact width=%u ratio=%u "
            "pos=%u emit=%u ape=%s decode_pack=%u exact_reduce=%u "
            "proj=%zu/%zu state=%zu/%zu "
            "comp=%zu max_ulp=%u/%u/%u/%u/%u\n",
            width, ratio, pos, emit ? 1u : 0u,
            ape_type == 1u ? "f16" : "f32",
            test_decode_pack ? 1u : 0u,
            test_decode_pack ? 1u : 0u,
            kv_stats.mismatch_count, score_stats.mismatch_count,
            state_kv_stats.mismatch_count,
            state_score_stats.mismatch_count,
            comp_stats.mismatch_count,
            kv_stats.max_ulp, score_stats.max_ulp,
            state_kv_stats.max_ulp, state_score_stats.max_ulp,
            comp_stats.max_ulp);
    TEST_ASSERT(kv_stats.mismatch_count == 0);
    TEST_ASSERT(score_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);
    TEST_ASSERT(comp_stats.mismatch_count == 0);

    free(fused_comp_host);
    free(ref_comp_host);
    free(fused_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_score_host);
    free(ref_state_kv_host);
    free(fused_score_host);
    free(fused_kv_host);
    free(ref_score_host);
    free(ref_kv_host);
    free(x_host);
    ds4_gpu_tensor_free(fused_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(fused_state_score);
    ds4_gpu_tensor_free(fused_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(fused_score);
    ds4_gpu_tensor_free(fused_kv);
    ds4_gpu_tensor_free(ref_score);
    ds4_gpu_tensor_free(ref_kv);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_f16_compressor_pair_state_store_exact(void) {
    test_metal_f16_compressor_pair_state_store_exact_case(
        256, 4, 8, 0, 17, false);
    test_metal_f16_compressor_pair_state_store_exact_case(
        256, 4, 11, 1, 23, true);
    test_metal_f16_compressor_pair_state_store_exact_case(
        1024, 4, 11, 1, 29, true);
    test_metal_f16_compressor_pair_state_store_exact_case(
        512, 128, 255, 1, 43, false);
}

static void test_metal_compressor_ape_add_exact_case(
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ape_type,
        uint32_t seed,
        bool test_pack_fusion) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)width * ratio * ape_elem_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *fused_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_score = ds4_gpu_tensor_alloc(state_bytes);
    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);

    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    const uint64_t poison_count = input_count > state_count ?
        input_count : state_count;
    float *poison_host = test_pack_fusion ?
        malloc((size_t)(poison_count * sizeof(float))) : NULL;
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(!test_pack_fusion || poison_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page,
                               (size_t)model_bytes) == 0);
    const bool allocated = kv && sc && ref_comp && fused_comp && ref_state_kv &&
        ref_state_score && fused_state_kv && fused_state_score && kv_host &&
        sc_host && ref_comp_host && fused_comp_host && ref_state_kv_host &&
        ref_state_score_host && fused_state_kv_host && fused_state_score_host &&
        (!test_pack_fusion || poison_host) && model_raw;
    const char *disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    char *saved_disable = test_save_env(disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)width * ratio; i++) {
                const int value = (int)((i * 17u + (i ^ (i >> 4u)) * 5u +
                                         seed * 13u) % 127u) - 63;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)width * ratio; i++) {
                const int value = (int)((i * 17u + (i ^ (i >> 4u)) * 5u +
                                         seed * 13u) % 127u) - 63;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f + (float)((i * 7u + seed) % 19u) / 64.0f;
        }
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value = (int)((i * 29u + (i ^ (i >> 5u)) * 3u +
                                        seed * 23u) % 181u) - 90;
            const int sc_value = (int)((i * 31u + (i ^ (i >> 3u)) * 11u +
                                        seed * 17u) % 173u) - 86;
            kv_host[i] = (float)kv_value / 112.0f;
            sc_host[i] = (float)sc_value / 96.0f;
        }
        kv_host[0] = -0.0f;
        sc_host[0] = -0.0f;

        // Exercise exact-add edge values in the first active APE row. Using
        // bit patterns avoids host fast-math rewriting signed zeros or
        // subnormals before the legacy and fused Metal paths see them.
        const uint32_t cutoff = (n_tokens / ratio) * ratio;
        const uint32_t edge_token = cutoff < n_tokens ? cutoff : cutoff - ratio;
        const uint64_t edge_ape =
            (uint64_t)((pos0 + edge_token) % ratio) * width;
        float *edge_score = sc_host + (uint64_t)edge_token * width;
        static const uint16_t edge_f16[] = {
            0x0000u, 0x8000u, 0x0001u, 0x8001u,
            0x3c00u, 0xbc00u, 0x7bffu, 0xfbffu,
        };
        static const uint32_t edge_f32[] = {
            0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
            0x3f800000u, 0xbf800000u, 0x7f7fffffu, 0xff7fffffu,
        };
        static const uint32_t edge_score_f16[] = {
            0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u,
            0xbf800000u, 0x3f800000u, 0xc77fe000u, 0x477fe000u,
        };
        static const uint32_t edge_score_f32[] = {
            0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u,
            0xbf800000u, 0x3f800000u, 0xff7fffffu, 0x7f7fffffu,
        };
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            memcpy(ape + edge_ape, edge_f16, sizeof(edge_f16));
            for (uint32_t i = 0; i < 8u; i++) {
                memcpy(edge_score + i, edge_score_f16 + i, sizeof(uint32_t));
            }
        } else {
            uint32_t *ape = model_raw;
            memcpy(ape + edge_ape, edge_f32, sizeof(edge_f32));
            for (uint32_t i = 0; i < 8u; i++) {
                memcpy(edge_score + i, edge_score_f32 + i, sizeof(uint32_t));
            }
        }
        for (uint64_t i = 0; i < state_count; i++) {
            ref_state_kv_host[i] = 1234.0f;
            fused_state_kv_host[i] = 1234.0f;
            ref_state_score_host[i] = -1234.0f;
            fused_state_score_host[i] = -1234.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, fused_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, ref_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, fused_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        if (test_pack_fusion) {
            TEST_ASSERT(unsetenv(disable_env) == 0);
        } else {
            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        }
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_tensor(
            ref_comp, ref_state_kv, ref_state_score, kv, sc,
            model_raw, model_bytes, 0, ape_type, norm_offset, 0,
            head_dim, ratio, pos0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        if (test_pack_fusion) {
            // Overwrite every persistent pack cell with qNaN payloads through
            // the legacy replay path. This makes an omitted candidate write
            // observable even for plane-zero padding, which normal legacy
            // packing would otherwise leave at the correct 0/-inf values.
            for (uint64_t i = 0; i < poison_count; i++) {
                const uint32_t bits =
                    0x7fc00001u + (uint32_t)(i & 0x3ffu);
                memcpy(poison_host + i, &bits, sizeof(bits));
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                            kv, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            sc, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_kv, 0, poison_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_score, 0, poison_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                fused_comp, fused_state_kv, fused_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, 0, n_comp * ratio, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            kv, 0, kv_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            sc, 0, sc_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_kv, 0, fused_state_kv_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_score, 0, fused_state_score_host,
                            state_bytes) != 0);
            TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        } else {
            TEST_ASSERT(unsetenv(disable_env) == 0);
        }
        TEST_ASSERT(ds4_gpu_compressor_prefill_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, ape_type, norm_offset, 0,
            head_dim, ratio, pos0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        if (test_pack_fusion) {
            TEST_ASSERT(ds4_gpu_tensor_read(
                            kv, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(memcmp(kv_host, poison_host,
                               (size_t)input_bytes) == 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            sc, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(memcmp(sc_host, poison_host,
                               (size_t)input_bytes) == 0);
        }
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host, (size_t)state_count);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    fprintf(stderr,
            "ds4-test: compressor %s exact head=%u ratio=%u pos=%u "
            "tokens=%u ape=%s comp=%zu/%llu state_kv=%zu/%llu "
            "state_score=%zu/%llu max_ulp=%u/%u/%u\n",
            test_pack_fusion ? "ratio4 pack" : "APE add",
            head_dim, ratio, pos0, n_tokens, ape_type == 1u ? "f16" : "f32",
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count, (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(poison_host);
    free(fused_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_score_host);
    free(ref_state_kv_host);
    free(fused_comp_host);
    free(ref_comp_host);
    free(sc_host);
    free(kv_host);
    ds4_gpu_tensor_free(fused_state_score);
    ds4_gpu_tensor_free(fused_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(fused_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
}

static void test_metal_compressor_ape_add_exact(void) {
    test_metal_compressor_ape_add_exact_case(128, 4, 3, 16, 1, 7, false);
    test_metal_compressor_ape_add_exact_case(512, 4, 0, 16, 1, 13, false);
    test_metal_compressor_ape_add_exact_case(512, 4, 1, 14, 0, 19, false);
    test_metal_compressor_ape_add_exact_case(512, 128, 127, 257, 1, 31, false);
}

static void test_metal_compressor_ratio4_pack_exact(void) {
    test_metal_compressor_ape_add_exact_case(128, 4, 0, 4, 1, 37, true);
    test_metal_compressor_ape_add_exact_case(512, 4, 0, 8, 1, 41, true);
    test_metal_compressor_ape_add_exact_case(512, 4, 1, 14, 0, 43, true);
}

static void test_metal_compressor_ratio4_replay_pack_exact_case(
        uint32_t head_dim,
        uint32_t n_tokens,
        uint32_t seed) {
    const uint32_t width = 2u * head_dim;
    const uint32_t n_comp = n_tokens / 4u;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)8u * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)4u * width * sizeof(uint16_t);
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *fused_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_score = ds4_gpu_tensor_alloc(state_bytes);
    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *state_kv_host = malloc((size_t)state_bytes);
    float *state_score_host = malloc((size_t)state_bytes);
    float *source_after_host = malloc((size_t)input_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);

    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(state_kv_host != NULL);
    TEST_ASSERT(state_score_host != NULL);
    TEST_ASSERT(source_after_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *ape_disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    char *saved_ape_disable = test_save_env(ape_disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    const bool allocated = kv && sc && ref_comp && fused_comp &&
        ref_state_kv && ref_state_score && fused_state_kv &&
        fused_state_score && kv_host && sc_host && state_kv_host &&
        state_score_host && source_after_host && ref_comp_host && fused_comp_host &&
        ref_state_kv_host && fused_state_kv_host && ref_state_score_host &&
        fused_state_score_host && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        uint16_t *ape = model_raw;
        for (uint64_t i = 0; i < (uint64_t)4u * width; i++) {
            const int value =
                (int)((i * 19u + (i ^ (i >> 3u)) * 7u + seed * 11u) %
                      113u) - 56;
            ape[i] = test_float_to_f16((float)value / 72.0f);
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.875f + (float)((i * 5u + seed) % 17u) / 64.0f;
        }
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value =
                (int)((i * 31u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                      193u) - 96;
            const int sc_value =
                (int)((i * 37u + (i ^ (i >> 5u)) * 9u + seed * 17u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 88.0f;
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const int kv_value =
                (int)((i * 23u + (i ^ (i >> 2u)) * 3u + seed * 29u) %
                      167u) - 83;
            const int sc_value =
                (int)((i * 41u + (i ^ (i >> 6u)) * 11u + seed * 7u) %
                      157u) - 78;
            state_kv_host[i] = (float)kv_value / 80.0f;
            state_score_host[i] = (float)sc_value / 92.0f;
        }
        const uint32_t negative_zero = 0x80000000u;
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        memcpy(state_kv_host, &negative_zero, sizeof(negative_zero));

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(unsetenv(ape_disable_env) == 0);
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);

        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            ref_comp, ref_state_kv, ref_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        // Poison every persistent pack cell through the legacy full-fill path
        // so a missing candidate write cannot inherit the reference value.
        for (uint64_t i = 0; i < input_count; i++) {
            const uint32_t bits = 0x7fc00001u + (uint32_t)(i & 0x3ffu);
            memcpy(kv_host + i, &bits, sizeof(bits));
            memcpy(sc_host + i, &bits, sizeof(bits));
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const uint32_t bits = 0x7fc00401u + (uint32_t)(i & 0x3ffu);
            memcpy(ref_state_kv_host + i, &bits, sizeof(bits));
            memcpy(ref_state_score_host + i, &bits, sizeof(bits));
        }
        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        // Restore finite sources and the candidate's original replay state.
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value =
                (int)((i * 31u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                      193u) - 96;
            const int sc_value =
                (int)((i * 37u + (i ^ (i >> 5u)) * 9u + seed * 17u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 88.0f;
        }
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host,
            (size_t)state_count);
    }

    test_restore_env(ape_disable_env, saved_ape_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    fprintf(stderr,
            "ds4-test: compressor ratio4 replay pack exact head=%u "
            "tokens=%u comp=%zu/%llu state_kv=%zu/%llu "
            "state_score=%zu/%llu max_ulp=%u/%u/%u\n",
            head_dim, n_tokens,
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count, (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(fused_state_score_host);
    free(ref_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_kv_host);
    free(fused_comp_host);
    free(ref_comp_host);
    free(source_after_host);
    free(state_score_host);
    free(state_kv_host);
    free(sc_host);
    free(kv_host);
    ds4_gpu_tensor_free(fused_state_score);
    ds4_gpu_tensor_free(fused_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(fused_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
}

static void test_metal_compressor_ratio4_replay_pack_exact(void) {
    test_metal_compressor_ratio4_replay_pack_exact_case(128, 4, 47);
    test_metal_compressor_ratio4_replay_pack_exact_case(512, 8, 53);
}

static void test_metal_compressor_ratio4_direct_pool_exact_case(
        uint32_t head_dim,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ape_type,
        bool replay,
        uint32_t seed) {
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * ape_elem_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    TEST_ASSERT(n_comp != 0);
    TEST_ASSERT(!replay || ((pos0 & 3u) == 0u && (n_tokens & 3u) == 0u));

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *direct_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *direct_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *direct_state_score = ds4_gpu_tensor_alloc(state_bytes);
    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *source_after_host = malloc((size_t)input_bytes);
    float *state_kv_host = malloc((size_t)state_bytes);
    float *state_score_host = malloc((size_t)state_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *direct_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *direct_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *direct_state_score_host = malloc((size_t)state_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);

    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(direct_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(direct_state_kv != NULL);
    TEST_ASSERT(direct_state_score != NULL);
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(source_after_host != NULL);
    TEST_ASSERT(state_kv_host != NULL);
    TEST_ASSERT(state_score_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(direct_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(direct_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(direct_state_score_host != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *ape_disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    const char *direct_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_DIRECT_POOL";
    char *saved_ape_disable = test_save_env(ape_disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    char *saved_direct_disable = test_save_env(direct_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    const bool allocated = kv && sc && ref_comp && direct_comp &&
        ref_state_kv && ref_state_score && direct_state_kv &&
        direct_state_score && kv_host && sc_host && source_after_host &&
        state_kv_host && state_score_host && ref_comp_host &&
        direct_comp_host && ref_state_kv_host && direct_state_kv_host &&
        ref_state_score_host && direct_state_score_host && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 17u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                          127u) - 63;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 17u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                          127u) - 63;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f + (float)((i * 7u + seed) % 19u) / 64.0f;
        }

        for (uint64_t i = 0; i < input_count; i++) {
            const uint32_t token = (uint32_t)(i / width);
            const uint32_t col = (uint32_t)(i - (uint64_t)token * width);
            const int kv_value =
                (int)(((uint64_t)token * 37u + (uint64_t)col * 29u +
                       (col >= head_dim ? 71u : 3u) + seed * 23u) %
                      193u) - 96;
            const int sc_value =
                (int)(((uint64_t)token * 41u + (uint64_t)col * 31u +
                       (col >= head_dim ? 17u : 83u) + seed * 11u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 32.0f;
        }
        const uint32_t negative_zero = 0x80000000u;
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        memcpy(sc_host + width + head_dim, &negative_zero,
               sizeof(negative_zero));

        for (uint32_t row = 0; row < state_rows; row++) {
            for (uint32_t col = 0; col < width; col++) {
                const uint64_t i = (uint64_t)row * width + col;
                if (replay && row < ratio && col < head_dim) {
                    const int kv_value =
                        (int)(((uint64_t)row * 43u + (uint64_t)col * 19u +
                               seed * 29u) % 167u) - 83;
                    const int sc_value =
                        (int)(((uint64_t)row * 47u + (uint64_t)col * 23u +
                               seed * 7u) % 157u) - 78;
                    state_kv_host[i] = (float)kv_value / 80.0f;
                    state_score_host[i] = (float)sc_value / 28.0f;
                } else {
                    const uint32_t kv_bits =
                        0x7fc00001u + (uint32_t)(i & 0x3ffu);
                    const uint32_t score_bits =
                        0x7fc00401u + (uint32_t)(i & 0x3ffu);
                    memcpy(state_kv_host + i, &kv_bits, sizeof(kv_bits));
                    memcpy(state_score_host + i, &score_bits,
                           sizeof(score_bits));
                }
            }
        }

        for (uint64_t i = 0; i < comp_count; i++) {
            const uint32_t bits = 0x7fc00801u + (uint32_t)(i & 0x3ffu);
            memcpy(ref_comp_host + i, &bits, sizeof(bits));
            memcpy(direct_comp_host + i, &bits, sizeof(bits));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_comp, 0, direct_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(unsetenv(ape_disable_env) == 0);
        TEST_ASSERT(setenv(direct_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        int ref_ok;
        if (replay) {
            ref_ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                ref_comp, ref_state_kv, ref_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        } else {
            ref_ok = ds4_gpu_compressor_prefill_tensor(
                ref_comp, ref_state_kv, ref_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, ratio, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        }
        TEST_ASSERT(ref_ok != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);

        TEST_ASSERT(unsetenv(direct_disable_env) == 0);
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);
        int direct_ok;
        if (replay) {
            direct_ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                direct_comp, direct_state_kv, direct_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        } else {
            direct_ok = ds4_gpu_compressor_prefill_tensor(
                direct_comp, direct_state_kv, direct_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, ratio, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        }
        TEST_ASSERT(direct_ok != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_comp, 0, direct_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_state_kv, 0, direct_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_state_score, 0, direct_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, direct_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, direct_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, direct_state_score_host,
            (size_t)state_count);
    }

    test_restore_env(ape_disable_env, saved_ape_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    test_restore_env(direct_disable_env, saved_direct_disable);
    fprintf(stderr,
            "ds4-test: compressor ratio4 direct pool exact mode=%s "
            "head=%u pos=%u tokens=%u comp_rows=%u ape=%s "
            "comp=%zu/%llu state_kv=%zu/%llu state_score=%zu/%llu "
            "max_ulp=%u/%u/%u\n",
            replay ? "replay" : "prefill",
            head_dim, pos0, n_tokens, n_comp,
            ape_type == 1u ? "f16" : "f32",
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count,
            (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(direct_state_score_host);
    free(ref_state_score_host);
    free(direct_state_kv_host);
    free(ref_state_kv_host);
    free(direct_comp_host);
    free(ref_comp_host);
    free(state_score_host);
    free(state_kv_host);
    free(source_after_host);
    free(sc_host);
    free(kv_host);
    ds4_gpu_tensor_free(direct_state_score);
    ds4_gpu_tensor_free(direct_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(direct_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
}

static void test_metal_compressor_ratio4_direct_pool_exact(void) {
    /* n_comp == 1 deliberately stays on the exact GGML reduction path. */
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 0, 4, 1, false, 59);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 1, 16, 0, false, 61);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 3, 14, 1, false, 67);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 2048, 8, 1, true, 71);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 12, 12, 0, true, 73);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 8, 4, 1, true, 79);
}

static void test_metal_inplace_rope_pair_exact(void) {
    /* Generic rope-tail primitive coverage only.  These deliberately include
     * non-Laguna shapes; the production Laguna Q/K matrix is below and uses
     * the exact 48/8 global and 72/8 SWA geometry. */
    typedef struct {
        uint32_t head_dim;
        uint32_t n_rot;
        uint32_t n_head;
        uint32_t n_tok;
        uint32_t pos0;
        bool inverse;
        float ext_factor;
    } rope_case;
    static const rope_case cases[] = {
        { 512, 64,  64,  1, UINT32_MAX, false, 1.0f },
        { 512, 64,   1,  1,  2047, false, 1.0f },
        { 512, 64,  64,  1, 65533,  true, 1.0f },
        { 128, 64,  64,  1,    37, false, 0.0f },
        { 512, 64,   4, 32,     0, false, 0.0f },
        { 512, 64,   7, 33,  2047,  true, 1.0f },
        { 128, 64,  64, 34, 65533, false, 1.0f },
        { 128, 64,  64, 35,    37,  true, 0.0f },
        { 128, 64,   4, 35, UINT32_MAX - 16u, true, 1.0f },
    };
    const char *disable_env = "DS4_METAL_DISABLE_INPLACE_ROPE_PAIR";
    const char *shared_disable_env =
        "DS4_METAL_DISABLE_SHARED_ROPE_COEFF";
    const char *affine_disable_env =
        "DS4_METAL_DISABLE_AFFINE_ROPE_PAIR";
    char *saved_disable = test_save_env(disable_env);
    char *saved_shared_disable = test_save_env(shared_disable_env);
    char *saved_affine_disable = test_save_env(affine_disable_env);
    size_t total_pair_mismatch = 0;
    size_t total_shared_mismatch = 0;
    size_t total_affine_mismatch = 0;
    size_t total_pair_prefix_mismatch = 0;
    size_t total_pair_tail_mismatch = 0;
    size_t total_shared_prefix_mismatch = 0;
    size_t total_shared_tail_mismatch = 0;
    size_t total_affine_prefix_mismatch = 0;
    size_t total_affine_tail_mismatch = 0;
    size_t total_elements = 0;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const rope_case *c = &cases[ci];
        const size_t elements =
            (size_t)c->n_tok * c->n_head * c->head_dim;
        const uint64_t bytes = (uint64_t)elements * sizeof(float);
        const uint32_t n_nope = c->head_dim - c->n_rot;
        const float freq_base = c->ext_factor != 0.0f ? 160000.0f : 10000.0f;
        const float freq_scale = c->ext_factor != 0.0f ? 1.0f / 16.0f : 1.0f;
        const uint32_t n_ctx_orig = c->ext_factor != 0.0f ? 65536u : 0u;
        float attn_factor = 1.0f;
        if (c->ext_factor != 0.0f) {
            attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }

        ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *pair_candidate = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *shared_candidate = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *affine_candidate = ds4_gpu_tensor_alloc(bytes);
        float *input = malloc((size_t)bytes);
        float *reference_host = malloc((size_t)bytes);
        float *pair_host = malloc((size_t)bytes);
        float *shared_host = malloc((size_t)bytes);
        float *affine_host = malloc((size_t)bytes);
        TEST_ASSERT(reference != NULL);
        TEST_ASSERT(pair_candidate != NULL);
        TEST_ASSERT(shared_candidate != NULL);
        TEST_ASSERT(affine_candidate != NULL);
        TEST_ASSERT(input != NULL);
        TEST_ASSERT(reference_host != NULL);
        TEST_ASSERT(pair_host != NULL);
        TEST_ASSERT(shared_host != NULL);
        TEST_ASSERT(affine_host != NULL);

        const bool allocated = reference && pair_candidate &&
            shared_candidate && affine_candidate && input && reference_host &&
            pair_host && shared_host && affine_host;
        if (allocated) {
            for (size_t i = 0; i < elements; i++) {
                const uint32_t key =
                    (uint32_t)(i * 37u + (i ^ (i >> 5u)) * 11u + ci * 101u);
                const int value = (int)(key % 4093u) - 2046;
                input[i] = (float)value / 1024.0f;
                if ((i + ci * 17u) % 257u == 0u) {
                    const uint32_t negative_zero = 0x80000000u;
                    memcpy(&input[i], &negative_zero, sizeof(negative_zero));
                }
            }

            TEST_ASSERT(ds4_gpu_tensor_write(reference, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                pair_candidate, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                shared_candidate, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                affine_candidate, 0, input, bytes) != 0);
            ds4_gpu_set_quality(false);

            TEST_ASSERT(setenv(affine_disable_env, "1", 1) == 0);
            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(setenv(shared_disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                reference,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(unsetenv(disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                pair_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(unsetenv(shared_disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                shared_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(setenv(shared_disable_env, "1", 1) == 0);
            TEST_ASSERT(unsetenv(affine_disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                affine_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                reference, 0, reference_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                pair_candidate, 0, pair_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                shared_candidate, 0, shared_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                affine_candidate, 0, affine_host, bytes) != 0);

            const test_float_compare_stats pair_stats =
                test_compare_float_bits(
                    reference_host, pair_host, elements);
            const test_float_compare_stats shared_stats =
                test_compare_float_bits(
                    pair_host, shared_host, elements);
            const test_float_compare_stats affine_stats =
                test_compare_float_bits(
                    pair_host, affine_host, elements);
            size_t pair_prefix_mismatch = 0;
            size_t pair_tail_mismatch = 0;
            size_t shared_prefix_mismatch = 0;
            size_t shared_tail_mismatch = 0;
            size_t affine_prefix_mismatch = 0;
            size_t affine_tail_mismatch = 0;
            for (uint32_t t = 0; t < c->n_tok; t++) {
                for (uint32_t h = 0; h < c->n_head; h++) {
                    const size_t row =
                        ((size_t)t * c->n_head + h) * c->head_dim;
                    for (uint32_t d = 0; d < n_nope; d++) {
                        if (memcmp(&input[row + d],
                                   &pair_host[row + d],
                                   sizeof(float)) != 0) {
                            pair_prefix_mismatch++;
                        }
                        if (memcmp(&input[row + d],
                                   &shared_host[row + d],
                                   sizeof(float)) != 0) {
                            shared_prefix_mismatch++;
                        }
                        if (memcmp(&input[row + d],
                                   &affine_host[row + d],
                                   sizeof(float)) != 0) {
                            affine_prefix_mismatch++;
                        }
                    }
                    for (uint32_t d = n_nope; d < c->head_dim; d++) {
                        if (memcmp(&reference_host[row + d],
                                   &pair_host[row + d],
                                   sizeof(float)) != 0) {
                            pair_tail_mismatch++;
                        }
                        if (memcmp(&pair_host[row + d],
                                   &shared_host[row + d],
                                   sizeof(float)) != 0) {
                            shared_tail_mismatch++;
                        }
                        if (memcmp(&pair_host[row + d],
                                   &affine_host[row + d],
                                   sizeof(float)) != 0) {
                            affine_tail_mismatch++;
                        }
                    }
                }
            }

            fprintf(stderr,
                    "ds4-test: in-place RoPE exactness case=%zu "
                    "shape=%ux%ux%u pos=%u inverse=%d ext=%g "
                    "pair=%zu/%zu shared=%zu/%zu affine=%zu/%zu "
                    "pair_prefix=%zu pair_tail=%zu "
                    "shared_prefix=%zu shared_tail=%zu "
                    "affine_prefix=%zu affine_tail=%zu "
                    "pair_max_ulp=%u shared_max_ulp=%u affine_max_ulp=%u "
                    "pair_max_abs=%g shared_max_abs=%g affine_max_abs=%g\n",
                    ci,
                    c->n_tok,
                    c->n_head,
                    c->head_dim,
                    c->pos0,
                    c->inverse ? 1 : 0,
                    c->ext_factor,
                    pair_stats.mismatch_count,
                    elements,
                    shared_stats.mismatch_count,
                    elements,
                    affine_stats.mismatch_count,
                    elements,
                    pair_prefix_mismatch,
                    pair_tail_mismatch,
                    shared_prefix_mismatch,
                    shared_tail_mismatch,
                    affine_prefix_mismatch,
                    affine_tail_mismatch,
                    pair_stats.max_ulp,
                    shared_stats.max_ulp,
                    affine_stats.max_ulp,
                    pair_stats.max_abs,
                    shared_stats.max_abs,
                    affine_stats.max_abs);
            TEST_ASSERT(pair_stats.mismatch_count == 0);
            TEST_ASSERT(shared_stats.mismatch_count == 0);
            TEST_ASSERT(affine_stats.mismatch_count == 0);
            TEST_ASSERT(pair_prefix_mismatch == 0);
            TEST_ASSERT(pair_tail_mismatch == 0);
            TEST_ASSERT(shared_prefix_mismatch == 0);
            TEST_ASSERT(shared_tail_mismatch == 0);
            TEST_ASSERT(affine_prefix_mismatch == 0);
            TEST_ASSERT(affine_tail_mismatch == 0);
            total_pair_mismatch += pair_stats.mismatch_count;
            total_shared_mismatch += shared_stats.mismatch_count;
            total_affine_mismatch += affine_stats.mismatch_count;
            total_pair_prefix_mismatch += pair_prefix_mismatch;
            total_pair_tail_mismatch += pair_tail_mismatch;
            total_shared_prefix_mismatch += shared_prefix_mismatch;
            total_shared_tail_mismatch += shared_tail_mismatch;
            total_affine_prefix_mismatch += affine_prefix_mismatch;
            total_affine_tail_mismatch += affine_tail_mismatch;
            total_elements += elements;
        }

        free(affine_host);
        free(shared_host);
        free(pair_host);
        free(reference_host);
        free(input);
        ds4_gpu_tensor_free(affine_candidate);
        ds4_gpu_tensor_free(shared_candidate);
        ds4_gpu_tensor_free(pair_candidate);
        ds4_gpu_tensor_free(reference);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(shared_disable_env, saved_shared_disable);
    test_restore_env(affine_disable_env, saved_affine_disable);
    fprintf(stderr,
            "ds4-test: in-place RoPE total pair=%zu/%zu shared=%zu/%zu "
            "affine=%zu/%zu "
            "pair_prefix=%zu pair_tail=%zu "
            "shared_prefix=%zu shared_tail=%zu "
            "affine_prefix=%zu affine_tail=%zu\n",
            total_pair_mismatch,
            total_elements,
            total_shared_mismatch,
            total_elements,
            total_affine_mismatch,
            total_elements,
            total_pair_prefix_mismatch,
            total_pair_tail_mismatch,
            total_shared_prefix_mismatch,
            total_shared_tail_mismatch,
            total_affine_prefix_mismatch,
            total_affine_tail_mismatch);
    TEST_ASSERT(total_pair_mismatch == 0);
    TEST_ASSERT(total_shared_mismatch == 0);
    TEST_ASSERT(total_affine_mismatch == 0);
    TEST_ASSERT(total_pair_prefix_mismatch == 0);
    TEST_ASSERT(total_pair_tail_mismatch == 0);
    TEST_ASSERT(total_shared_prefix_mismatch == 0);
    TEST_ASSERT(total_shared_tail_mismatch == 0);
    TEST_ASSERT(total_affine_prefix_mismatch == 0);
    TEST_ASSERT(total_affine_tail_mismatch == 0);
}

static void test_metal_contiguous_f32_f16_roundtrip_exact(void) {
    typedef struct {
        uint32_t n;
        uint32_t src_offset;
        uint32_t dst_offset;
    } copy_case;
    static const copy_case cases[] = {
        { 1,  0,  0 },
        { 3,  4,  2 },
        { 4, 16,  8 },
        { 5, 12,  6 },
        { 17, 20, 10 },
        { 65,  4,  2 },
    };
    const char *env_name = "DS4_METAL_DISABLE_CONTIG_F32_F16_COPY";
    char *saved_env = test_save_env(env_name);
    size_t half_mismatch = 0;
    size_t half_guard_mismatch = 0;
    size_t roundtrip_mismatch = 0;
    size_t roundtrip_guard_mismatch = 0;
    size_t half_total = 0;
    size_t roundtrip_total = 0;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const uint32_t n = cases[ci].n;
        const uint64_t src_bytes = cases[ci].src_offset +
                                   (uint64_t)n * sizeof(float) + 16u;
        const uint64_t half_bytes = cases[ci].dst_offset +
                                    (uint64_t)n * sizeof(uint16_t) + 16u;
        const uint32_t raw_cap = 3;
        const uint32_t raw_row = 1;
        const uint64_t raw_bytes =
            (uint64_t)raw_cap * n * sizeof(float);

        ds4_gpu_tensor *src_base = ds4_gpu_tensor_alloc(src_bytes);
        ds4_gpu_tensor *src_view = src_base
            ? ds4_gpu_tensor_view(src_base,
                                  cases[ci].src_offset,
                                  (uint64_t)n * sizeof(float))
            : NULL;
        ds4_gpu_tensor *half_ref = ds4_gpu_tensor_alloc(half_bytes);
        ds4_gpu_tensor *half_vec = ds4_gpu_tensor_alloc(half_bytes);
        ds4_gpu_tensor *raw_ref = ds4_gpu_tensor_alloc(raw_bytes);
        ds4_gpu_tensor *raw_vec = ds4_gpu_tensor_alloc(raw_bytes);
        TEST_ASSERT(src_base != NULL);
        TEST_ASSERT(src_view != NULL);
        TEST_ASSERT(half_ref != NULL);
        TEST_ASSERT(half_vec != NULL);
        TEST_ASSERT(raw_ref != NULL);
        TEST_ASSERT(raw_vec != NULL);

        uint8_t *src_host = malloc((size_t)src_bytes);
        uint16_t *half_init = malloc((size_t)half_bytes);
        uint16_t *half_ref_host = malloc((size_t)half_bytes);
        uint16_t *half_vec_host = malloc((size_t)half_bytes);
        uint32_t *raw_init = malloc((size_t)raw_bytes);
        uint32_t *raw_ref_host = malloc((size_t)raw_bytes);
        uint32_t *raw_vec_host = malloc((size_t)raw_bytes);
        TEST_ASSERT(src_host != NULL);
        TEST_ASSERT(half_init != NULL);
        TEST_ASSERT(half_ref_host != NULL);
        TEST_ASSERT(half_vec_host != NULL);
        TEST_ASSERT(raw_init != NULL);
        TEST_ASSERT(raw_ref_host != NULL);
        TEST_ASSERT(raw_vec_host != NULL);

        const bool allocated = src_base && src_view && half_ref && half_vec &&
            raw_ref && raw_vec && src_host && half_init && half_ref_host &&
            half_vec_host && raw_init && raw_ref_host && raw_vec_host;
        if (allocated) {
            memset(src_host, 0x6d, (size_t)src_bytes);
            test_fill_copy_f32_patterns(src_host + cases[ci].src_offset,
                                        n,
                                        (uint32_t)(ci * 7u));
            const size_t half_words = (size_t)(half_bytes / sizeof(uint16_t));
            for (size_t i = 0; i < half_words; i++) {
                half_init[i] = (uint16_t)(0xa55au ^ (uint16_t)(i * 73u));
            }
            const size_t raw_words = (size_t)raw_cap * n;
            for (size_t i = 0; i < raw_words; i++) {
                raw_init[i] = 0x4a000000u + (uint32_t)i;
            }

            TEST_ASSERT(ds4_gpu_tensor_write(src_base, 0, src_host, src_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(half_ref, 0, half_init, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(half_vec, 0, half_init, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(raw_ref, 0, raw_init, raw_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(raw_vec, 0, raw_init, raw_bytes) != 0);

            TEST_ASSERT(setenv(env_name, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_tensor_copy_f32_to_f16(
                half_ref,
                cases[ci].dst_offset,
                src_base,
                cases[ci].src_offset,
                n) != 0);
            TEST_ASSERT(ds4_gpu_store_raw_kv_tensor(
                raw_ref, src_view, raw_cap, raw_row, n) != 0);

            TEST_ASSERT(setenv(env_name, "0", 1) == 0);
            TEST_ASSERT(ds4_gpu_tensor_copy_f32_to_f16(
                half_vec,
                cases[ci].dst_offset,
                src_base,
                cases[ci].src_offset,
                n) != 0);
            TEST_ASSERT(ds4_gpu_store_raw_kv_tensor(
                raw_vec, src_view, raw_cap, raw_row, n) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                half_ref, 0, half_ref_host, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                half_vec, 0, half_vec_host, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                raw_ref, 0, raw_ref_host, raw_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                raw_vec, 0, raw_vec_host, raw_bytes) != 0);

            const size_t half_first = cases[ci].dst_offset / sizeof(uint16_t);
            const size_t half_last = half_first + n;
            for (size_t i = 0; i < half_words; i++) {
                if (half_ref_host[i] == half_vec_host[i]) continue;
                if (i >= half_first && i < half_last) {
                    half_mismatch++;
                } else {
                    half_guard_mismatch++;
                }
            }

            const size_t raw_first = (size_t)raw_row * n;
            const size_t raw_last = raw_first + n;
            for (size_t i = 0; i < raw_words; i++) {
                if (raw_ref_host[i] == raw_vec_host[i]) continue;
                if (i >= raw_first && i < raw_last) {
                    roundtrip_mismatch++;
                } else {
                    roundtrip_guard_mismatch++;
                }
            }
            half_total += n;
            roundtrip_total += n;
        }

        free(raw_vec_host);
        free(raw_ref_host);
        free(raw_init);
        free(half_vec_host);
        free(half_ref_host);
        free(half_init);
        free(src_host);
        ds4_gpu_tensor_free(raw_vec);
        ds4_gpu_tensor_free(raw_ref);
        ds4_gpu_tensor_free(half_vec);
        ds4_gpu_tensor_free(half_ref);
        ds4_gpu_tensor_free(src_view);
        ds4_gpu_tensor_free(src_base);
    }

    test_restore_env(env_name, saved_env);
    fprintf(stderr,
            "ds4-test: contiguous conversion exactness "
            "f32_f16=%zu/%zu guard=%zu, f16_f32_roundtrip=%zu/%zu guard=%zu\n",
            half_mismatch,
            half_total,
            half_guard_mismatch,
            roundtrip_mismatch,
            roundtrip_total,
            roundtrip_guard_mismatch);
    TEST_ASSERT(half_mismatch == 0);
    TEST_ASSERT(half_guard_mismatch == 0);
    TEST_ASSERT(roundtrip_mismatch == 0);
    TEST_ASSERT(roundtrip_guard_mismatch == 0);
}
#endif

#if defined(__APPLE__)
static void test_metal_gathered_kv_stage_exact(void) {
    const uint32_t head_dim = 512;
    const uint32_t raw_cap = 7;
    const uint32_t n_raw = 5;
    const uint32_t n_comp = 3;
    const uint32_t raw_starts[] = {0, 2, 5, 6};
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t payload_bytes =
        ((uint64_t)n_raw + n_comp) * head_dim * sizeof(uint16_t);
    const uint64_t raw_out_bytes =
        (uint64_t)n_raw * head_dim * sizeof(uint16_t);
    const uint64_t raw_view_offset = 4;
    const uint64_t comp_view_offset = 2;
    const uint64_t dst_view_offset = 6;
    const uint64_t raw_base_bytes = raw_view_offset + raw_bytes + 12;
    const uint64_t comp_base_bytes = comp_view_offset + comp_bytes + 14;
    const uint64_t dst_base_bytes = dst_view_offset + payload_bytes + 10;
    const char *envs[] = {
        "DS4_METAL_REQUIRE_GATHERED_KV_STAGE",
        "DS4_METAL_DISABLE_CONTIG_F32_F16_COPY",
        "DS4_METAL_DISABLE_CONTIG_F16_F16_COPY",
    };
    char *saved[sizeof(envs)/sizeof(envs[0])];
    for (size_t i = 0; i < sizeof(envs)/sizeof(envs[0]); i++) {
        saved[i] = test_save_env(envs[i]);
    }

    ds4_gpu_tensor *raw_base = ds4_gpu_tensor_alloc(raw_base_bytes);
    ds4_gpu_tensor *raw = raw_base
        ? ds4_gpu_tensor_view(raw_base, raw_view_offset, raw_bytes)
        : NULL;
    ds4_gpu_tensor *comp_base = ds4_gpu_tensor_alloc(comp_base_bytes);
    ds4_gpu_tensor *comp = comp_base
        ? ds4_gpu_tensor_view(comp_base, comp_view_offset, comp_bytes)
        : NULL;
    ds4_gpu_tensor *ref_base = ds4_gpu_tensor_alloc(dst_base_bytes);
    ds4_gpu_tensor *ref = ref_base
        ? ds4_gpu_tensor_view(ref_base, dst_view_offset, payload_bytes)
        : NULL;
    ds4_gpu_tensor *fused_base = ds4_gpu_tensor_alloc(dst_base_bytes);
    ds4_gpu_tensor *fused = fused_base
        ? ds4_gpu_tensor_view(fused_base, dst_view_offset, payload_bytes)
        : NULL;
    TEST_ASSERT(raw_base != NULL);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp_base != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(ref_base != NULL);
    TEST_ASSERT(ref != NULL);
    TEST_ASSERT(fused_base != NULL);
    TEST_ASSERT(fused != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    uint8_t *dst_init = malloc((size_t)dst_base_bytes);
    uint8_t *ref_host = malloc((size_t)dst_base_bytes);
    uint8_t *fused_host = malloc((size_t)dst_base_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(dst_init != NULL);
    TEST_ASSERT(ref_host != NULL);
    TEST_ASSERT(fused_host != NULL);

    static const uint16_t half_patterns[] = {
        0x0000u, 0x8000u, 0x0001u, 0x03ffu, 0x0400u,
        0x3555u, 0x3c00u, 0x3c01u, 0x7bffu, 0xfbffu,
        0x7c00u, 0xfc00u, 0x7e00u, 0x7e01u, 0xfe55u,
    };
    const bool allocated = raw_base && raw && comp_base && comp &&
        ref_base && ref && fused_base && fused && raw_host && comp_host &&
        dst_init && ref_host && fused_host;
    size_t raw_mismatch = 0;
    size_t comp_mismatch = 0;
    size_t guard_mismatch = 0;
    if (allocated) {
        for (uint32_t row = 0; row < raw_cap; row++) {
            test_fill_copy_f32_patterns(
                raw_host + (uint64_t)row * head_dim,
                head_dim,
                row * 17u + 3u);
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            comp_host[i] = half_patterns[(i * 7u + (i >> 3u)) %
                (sizeof(half_patterns)/sizeof(half_patterns[0]))];
        }
        for (uint64_t i = 0; i < dst_base_bytes; i++) {
            dst_init[i] = (uint8_t)(0xa5u ^ (uint8_t)(i * 37u));
        }
        TEST_ASSERT(ds4_gpu_tensor_write(
                        raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp, 0, comp_host, comp_bytes) != 0);
        for (size_t ci = 0;
             ci < sizeof(raw_starts)/sizeof(raw_starts[0]);
             ci++) {
            TEST_ASSERT(ds4_gpu_tensor_write(
                            ref_base, 0, dst_init, dst_base_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_base, 0, dst_init, dst_base_bytes) != 0);

            TEST_ASSERT(unsetenv(envs[0]) == 0);
            TEST_ASSERT(unsetenv(envs[1]) == 0);
            TEST_ASSERT(unsetenv(envs[2]) == 0);
            ds4_gpu_set_quality(true);
            TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                            ref, raw, raw_cap, raw_starts[ci], n_raw,
                            comp, 1, n_comp, head_dim) != 0);

            ds4_gpu_set_quality(false);
            TEST_ASSERT(setenv(envs[0], "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                            fused, raw, raw_cap, raw_starts[ci], n_raw,
                            comp, 1, n_comp, head_dim) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                            ref_base, 0, ref_host, dst_base_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            fused_base, 0, fused_host,
                            dst_base_bytes) != 0);
            for (uint64_t i = 0; i < dst_base_bytes; i++) {
                if (i < dst_view_offset ||
                    i >= dst_view_offset + payload_bytes) {
                    if (ref_host[i] != dst_init[i]) guard_mismatch++;
                    if (fused_host[i] != dst_init[i]) guard_mismatch++;
                } else if (ref_host[i] == fused_host[i]) {
                    continue;
                } else if (i < dst_view_offset + raw_out_bytes) {
                    raw_mismatch++;
                } else {
                    comp_mismatch++;
                }
            }
        }

        /* Component-copy diagnostics and quality mode prevent strict
         * selection of the gathered kernel. */
        TEST_ASSERT(setenv(envs[0], "1", 1) == 0);
        TEST_ASSERT(setenv(envs[1], "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        TEST_ASSERT(unsetenv(envs[1]) == 0);
        TEST_ASSERT(setenv(envs[2], "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        TEST_ASSERT(unsetenv(envs[2]) == 0);
        ds4_gpu_set_quality(true);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        ds4_gpu_set_quality(false);
    }

    for (size_t i = 0; i < sizeof(envs)/sizeof(envs[0]); i++) {
        test_restore_env(envs[i], saved[i]);
    }
    fprintf(stderr,
            "ds4-test: gathered KV staging exact cases=%zu "
            "raw_bytes=%zu comp_bytes=%zu guard_bytes=%zu\n",
            sizeof(raw_starts)/sizeof(raw_starts[0]),
            raw_mismatch, comp_mismatch, guard_mismatch);
    TEST_ASSERT(raw_mismatch == 0);
    TEST_ASSERT(comp_mismatch == 0);
    TEST_ASSERT(guard_mismatch == 0);

    free(fused_host);
    free(ref_host);
    free(dst_init);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(fused);
    ds4_gpu_tensor_free(fused_base);
    ds4_gpu_tensor_free(ref);
    ds4_gpu_tensor_free(ref_base);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(comp_base);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(raw_base);
}

static void test_metal_contiguous_compressed_f16_attention_exact(void) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 2;
    const uint32_t raw_cap = 7;
    const uint32_t n_raw = 5;
    const uint32_t raw_start = 5;
    const uint32_t n_comp = 3;
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t q_bytes =
        (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();
    const char *env_name = "DS4_METAL_DISABLE_CONTIG_F16_F16_COPY";
    static const uint16_t half_patterns[] = {
        0x0000u, 0x8000u, 0x0001u, 0x03ffu, 0x0400u,
        0x1001u, 0x3555u, 0x3c00u, 0x3c01u, 0x4000u,
        0xbc00u, 0xc000u, 0x7bffu, 0xfbffu,
    };

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp_base = ds4_gpu_tensor_alloc(comp_bytes + 18u);
    ds4_gpu_tensor *comp = comp_base
        ? ds4_gpu_tensor_view(comp_base, 2u, comp_bytes)
        : NULL;
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *heads_blit = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *heads_compute = ds4_gpu_tensor_alloc(q_bytes);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp_base != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(heads_blit != NULL);
    TEST_ASSERT(heads_compute != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = malloc((size_t)q_bytes);
    float *blit_host = malloc((size_t)q_bytes);
    float *compute_host = malloc((size_t)q_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(blit_host != NULL);
    TEST_ASSERT(compute_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    const bool allocated = raw && comp_base && comp && q && heads_blit &&
        heads_compute && raw_host && comp_host && q_host && blit_host &&
        compute_host && model_raw;
    char *saved_env = test_save_env(env_name);
    test_float_compare_stats stats = {0};
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *sinks = model_raw;
        sinks[0] = -0.375f;
        sinks[1] = 0.1875f;
        for (uint64_t i = 0; i < (uint64_t)raw_cap * head_dim; i++) {
            const int value =
                (int)((i * 19u + (i ^ (i >> 5u)) * 7u) % 193u) - 96;
            raw_host[i] = (float)value / 128.0f;
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            comp_host[i] = half_patterns[(i * 5u + (i >> 4u)) %
                (sizeof(half_patterns) / sizeof(half_patterns[0]))];
        }
        for (uint32_t i = 0; i < n_head * head_dim; i++) {
            const int value = (int)((i * 37u + (i ^ (i >> 3u)) * 11u) % 251u) - 125;
            q_host[i] = (float)value / 96.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(env_name, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            heads_blit,
            model_raw,
            page,
            0,
            q,
            raw,
            n_raw,
            raw_cap,
            raw_start,
            comp,
            1,
            n_comp,
            NULL,
            0,
            n_head,
            head_dim) != 0);

        TEST_ASSERT(setenv(env_name, "0", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            heads_compute,
            model_raw,
            page,
            0,
            q,
            raw,
            n_raw,
            raw_cap,
            raw_start,
            comp,
            1,
            n_comp,
            NULL,
            0,
            n_head,
            head_dim) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
            heads_blit, 0, blit_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
            heads_compute, 0, compute_host, q_bytes) != 0);
        stats = test_compare_float_bits(
            blit_host, compute_host, (size_t)n_head * head_dim);
    }
    test_restore_env(env_name, saved_env);
    fprintf(stderr,
            "ds4-test: contiguous compressed-F16 staging exactness "
            "mismatches=%zu/%u max_ulp=%u max_abs=%g\n",
            stats.mismatch_count,
            n_head * head_dim,
            stats.max_ulp,
            stats.max_abs);
    TEST_ASSERT(stats.mismatch_count == 0);

    free(model_raw);
    free(compute_host);
    free(blit_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(heads_compute);
    ds4_gpu_tensor_free(heads_blit);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(comp_base);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_persistent_zero_attention_mask_exact_case(
        uint32_t raw_cap,
        uint32_t n_raw,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t seed) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 2;
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t q_bytes =
        (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t mask_bytes = (uint64_t)n_comp * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc(mask_bytes);
    ds4_gpu_tensor *legacy = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *persistent = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *masked = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *pad_legacy = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *after_mask = ds4_gpu_tensor_alloc(q_bytes);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(comp_mask != NULL);
    TEST_ASSERT(legacy != NULL);
    TEST_ASSERT(persistent != NULL);
    TEST_ASSERT(masked != NULL);
    TEST_ASSERT(pad_legacy != NULL);
    TEST_ASSERT(after_mask != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = malloc((size_t)q_bytes);
    float *mask_host = malloc((size_t)mask_bytes);
    float *legacy_host = malloc((size_t)q_bytes);
    float *persistent_host = malloc((size_t)q_bytes);
    float *masked_host = malloc((size_t)q_bytes);
    float *pad_legacy_host = malloc((size_t)q_bytes);
    float *after_mask_host = malloc((size_t)q_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(mask_host != NULL);
    TEST_ASSERT(legacy_host != NULL);
    TEST_ASSERT(persistent_host != NULL);
    TEST_ASSERT(masked_host != NULL);
    TEST_ASSERT(pad_legacy_host != NULL);
    TEST_ASSERT(after_mask_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    const bool allocated = raw && comp && q && comp_mask && legacy &&
        persistent && masked && pad_legacy && after_mask && raw_host &&
        comp_host && q_host && mask_host && legacy_host && persistent_host &&
        masked_host && pad_legacy_host && after_mask_host &&
        model_raw;
    const char *disable_env =
        "DS4_METAL_DISABLE_PERSISTENT_ZERO_ATTN_MASK";
    const char *pad_disable_env =
        "DS4_METAL_DISABLE_GATHERED_KV_PAD_FUSION";
    const char *shared_pad_disable_env =
        "DS4_METAL_DISABLE_SHARED_KV_PAD";
    char *saved_disable = test_save_env(disable_env);
    char *saved_pad_disable = test_save_env(pad_disable_env);
    char *saved_shared_pad_disable = test_save_env(shared_pad_disable_env);
    test_float_compare_stats persistent_stats = {0};
    test_float_compare_stats masked_stats = {0};
    test_float_compare_stats pad_stats = {0};
    test_float_compare_stats after_mask_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *sinks = model_raw;
        sinks[0] = -0.3125f;
        sinks[1] = 0.21875f;
        for (uint64_t i = 0; i < (uint64_t)raw_cap * head_dim; i++) {
            const int value = (int)((i * 17u + (i ^ (i >> 5u)) * 11u +
                                     seed * 13u) % 211u) - 105;
            raw_host[i] = (float)value / 128.0f;
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            const int value = (int)((i * 23u + (i ^ (i >> 4u)) * 7u +
                                     seed * 19u) % 193u) - 96;
            comp_host[i] = test_float_to_f16((float)value / 112.0f);
        }
        for (uint32_t i = 0; i < n_head * head_dim; i++) {
            const int value = (int)((i * 31u + (i ^ (i >> 3u)) * 5u +
                                     seed * 29u) % 227u) - 113;
            q_host[i] = (float)value / 104.0f;
        }
        for (uint32_t i = 0; i < n_comp; i++) {
            mask_host[i] = i == 0 ? -8.0f : -(float)(i + 1u) / 8.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp_mask, 0, mask_host, mask_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            legacy, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        unsetenv(disable_env);
        unsetenv(pad_disable_env);
        unsetenv(shared_pad_disable_env);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            persistent, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            masked, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            comp_mask, 1, n_head, head_dim) != 0);

        TEST_ASSERT(setenv(pad_disable_env, "1", 1) == 0);
        TEST_ASSERT(setenv(shared_pad_disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            pad_legacy, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            comp_mask, 1, n_head, head_dim) != 0);
        unsetenv(pad_disable_env);
        unsetenv(shared_pad_disable_env);

        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            after_mask, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        legacy, 0, legacy_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        persistent, 0, persistent_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        masked, 0, masked_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        pad_legacy, 0, pad_legacy_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        after_mask, 0, after_mask_host, q_bytes) != 0);
        persistent_stats = test_compare_float_bits(
            legacy_host, persistent_host, (size_t)n_head * head_dim);
        masked_stats = test_compare_float_bits(
            legacy_host, masked_host, (size_t)n_head * head_dim);
        pad_stats = test_compare_float_bits(
            pad_legacy_host, masked_host, (size_t)n_head * head_dim);
        after_mask_stats = test_compare_float_bits(
            legacy_host, after_mask_host, (size_t)n_head * head_dim);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(pad_disable_env, saved_pad_disable);
    test_restore_env(shared_pad_disable_env, saved_shared_pad_disable);
    fprintf(stderr,
            "ds4-test: persistent zero attention mask exact keys=%u "
            "candidate=%zu/%u max_ulp=%u masked_diff=%zu/%u "
            "pad_fusion=%zu/%u max_ulp=%u "
            "after_mask=%zu/%u max_ulp=%u\n",
            n_raw + n_comp,
            persistent_stats.mismatch_count, n_head * head_dim,
            persistent_stats.max_ulp,
            masked_stats.mismatch_count, n_head * head_dim,
            pad_stats.mismatch_count, n_head * head_dim,
            pad_stats.max_ulp,
            after_mask_stats.mismatch_count, n_head * head_dim,
            after_mask_stats.max_ulp);
    TEST_ASSERT(persistent_stats.mismatch_count == 0);
    TEST_ASSERT(masked_stats.mismatch_count != 0);
    TEST_ASSERT(pad_stats.mismatch_count == 0);
    TEST_ASSERT(after_mask_stats.mismatch_count == 0);

    free(model_raw);
    free(after_mask_host);
    free(pad_legacy_host);
    free(masked_host);
    free(persistent_host);
    free(legacy_host);
    free(mask_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(after_mask);
    ds4_gpu_tensor_free(pad_legacy);
    ds4_gpu_tensor_free(masked);
    ds4_gpu_tensor_free(persistent);
    ds4_gpu_tensor_free(legacy);
    ds4_gpu_tensor_free(comp_mask);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_persistent_zero_attention_mask_exact(void) {
    test_metal_persistent_zero_attention_mask_exact_case(7, 5, 5, 3, 11);
    test_metal_persistent_zero_attention_mask_exact_case(37, 29, 35, 3, 23);
    test_metal_persistent_zero_attention_mask_exact_case(
        1, 1, 0, 8192, 31);
}

typedef enum {
    TEST_METAL_PREFILL_MASK_CACHE_RAW = 1,
    TEST_METAL_PREFILL_MASK_CACHE_RATIO4 = 2,
    TEST_METAL_PREFILL_MASK_CACHE_RATIO128 = 3,
} test_metal_prefill_mask_cache_kind;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_comp;
    uint32_t window;
    uint32_t ratio;
} test_metal_prefill_mask_cache_shape;

static int test_metal_zero_prefix_prefill_mask_cache_call(
        test_metal_prefill_mask_cache_kind  kind,
        ds4_gpu_tensor                    *heads,
        const void                        *model_map,
        uint64_t                           model_size,
        const ds4_gpu_tensor              *q,
        const ds4_gpu_tensor              *raw,
        const ds4_gpu_tensor              *comp,
        const ds4_gpu_tensor              *comp_mask,
        const test_metal_prefill_mask_cache_shape *shape,
        bool                               masked,
        uint32_t                           n_head,
        uint32_t                           head_dim) {
    if (kind == TEST_METAL_PREFILL_MASK_CACHE_RAW) {
        if (masked) return 0;
        return ds4_gpu_attention_prefill_raw_heads_tensor(
            heads, model_map, model_size, 0, q, raw,
            shape->n_tokens, shape->window, n_head, head_dim);
    }

    if (masked) {
        return ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
            heads, model_map, model_size, 0, q, raw, comp, 1, comp_mask,
            shape->n_tokens, shape->n_comp, shape->window, shape->ratio,
            n_head, head_dim);
    }

    return ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        heads, model_map, model_size, 0, q, raw, comp, 1,
        shape->n_tokens, shape->n_comp, shape->window, shape->ratio,
        n_head, head_dim);
}

static bool test_metal_zero_prefix_prefill_mask_cache_run(
        test_metal_prefill_mask_cache_kind  kind,
        ds4_gpu_tensor                    *heads,
        const void                        *model_map,
        uint64_t                           model_size,
        const ds4_gpu_tensor              *q,
        const ds4_gpu_tensor              *raw,
        const ds4_gpu_tensor              *comp,
        const ds4_gpu_tensor              *comp_mask,
        const test_metal_prefill_mask_cache_shape *shape,
        bool                               masked,
        uint32_t                           n_head,
        uint32_t                           head_dim,
        float                             *host) {
    const uint64_t bytes =
        (uint64_t)shape->n_tokens * n_head * head_dim * sizeof(float);
    const int call_ok = test_metal_zero_prefix_prefill_mask_cache_call(
        kind, heads, model_map, model_size, q, raw, comp, comp_mask,
        shape, masked, n_head, head_dim);
    TEST_ASSERT(call_ok != 0);
    if (!call_ok) return false;

    const int read_ok = ds4_gpu_tensor_read(heads, 0, host, bytes);
    TEST_ASSERT(read_ok != 0);
    return read_ok != 0;
}

static void test_metal_zero_prefix_prefill_mask_cache_compare(
        const float *expected,
        const float *actual,
        size_t       count,
        size_t      *total_mismatches,
        uint32_t    *max_ulp) {
    const test_float_compare_stats stats =
        test_compare_float_bits(expected, actual, count);
    *total_mismatches += stats.mismatch_count;
    if (stats.max_ulp > *max_ulp) *max_ulp = stats.max_ulp;
}

static void test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        test_metal_prefill_mask_cache_kind kind,
        uint32_t                           seed) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 1;
    const uint32_t max_tokens = 129;
    const uint32_t max_comp = 32;
    const uint64_t raw_count = (uint64_t)max_tokens * head_dim;
    const uint64_t comp_count = (uint64_t)max_comp * head_dim;
    const uint64_t q_count = (uint64_t)max_tokens * n_head * head_dim;
    const uint64_t mask_count = (uint64_t)max_tokens * max_comp;
    const uint64_t raw_bytes = raw_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(uint16_t);
    const uint64_t q_bytes = q_count * sizeof(float);
    const uint64_t mask_bytes = mask_count * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();

    const test_metal_prefill_mask_cache_shape shape_a = {
        .n_tokens = 128,
        .n_comp = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? 0u :
                  (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 32u : 1u),
        .window = 128,
        .ratio = kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 4u :
                 (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO128 ? 128u : 0u),
    };
    const test_metal_prefill_mask_cache_shape shape_b = {
        .n_tokens = 129,
        .n_comp = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? 0u :
                  (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 31u : 2u),
        .window = 63,
        .ratio = shape_a.ratio,
    };
    const size_t count_a =
        (size_t)shape_a.n_tokens * n_head * head_dim;
    const size_t count_b =
        (size_t)shape_b.n_tokens * n_head * head_dim;

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc(mask_bytes);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_bytes);
    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = calloc((size_t)q_count, sizeof(float));
    float *mask_host = malloc((size_t)mask_bytes);
    float *ref_a = malloc((size_t)q_bytes);
    float *ref_b = malloc((size_t)q_bytes);
    float *actual = malloc((size_t)q_bytes);
    float *masked_actual = malloc((size_t)q_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);

    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(comp_mask != NULL);
    TEST_ASSERT(heads != NULL);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(mask_host != NULL);
    TEST_ASSERT(ref_a != NULL);
    TEST_ASSERT(ref_b != NULL);
    TEST_ASSERT(actual != NULL);
    TEST_ASSERT(masked_actual != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *disable_env =
        "DS4_METAL_DISABLE_ZERO_PREFIX_PREFILL_MASK_CACHE";
    char *saved_disable = test_save_env(disable_env);
    size_t total_mismatches = 0;
    uint32_t max_ulp = 0;
    size_t key_difference = 0;
    size_t masked_difference = 0;

    const bool allocated = raw && comp && q && comp_mask && heads &&
        raw_host && comp_host && q_host && mask_host && ref_a && ref_b &&
        actual && masked_actual && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        ((float *)model_raw)[0] = -1.0f;

        for (uint32_t row = 0; row < max_tokens; row++) {
            for (uint32_t col = 0; col < head_dim; col++) {
                const int value = (int)((row * 37u + col * 17u +
                                         (col ^ (col >> 3u)) * 5u +
                                         seed * 13u) % 257u) - 128;
                raw_host[(uint64_t)row * head_dim + col] =
                    (float)value / 256.0f;
            }
        }
        for (uint32_t row = 0; row < max_comp; row++) {
            for (uint32_t col = 0; col < head_dim; col++) {
                const int value = (int)((row * 29u + col * 11u +
                                         (col ^ (col >> 4u)) * 7u +
                                         seed * 19u) % 193u) - 96;
                comp_host[(uint64_t)row * head_dim + col] =
                    test_float_to_f16(0.375f + (float)value / 384.0f);
            }
        }
        for (uint64_t i = 0; i < mask_count; i++) {
            mask_host[i] = -65504.0f;
        }
        if (shape_a.n_comp != 0) {
            for (uint32_t row = 0; row < shape_a.n_tokens; row++) {
                const uint32_t visible = (row + 1u) / shape_a.ratio;
                for (uint32_t col = 0; col < shape_a.n_comp; col++) {
                    if (col < visible) {
                        mask_host[(uint64_t)row * shape_a.n_comp + col] =
                            (col & 1u) != 0u ? -2.0f : -65504.0f;
                    }
                }
            }
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp_mask, 0, mask_host, mask_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        const bool have_ref_a =
            test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, ref_a);
        const bool have_ref_b =
            test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_b, false, n_head, head_dim, ref_b);

        if (have_ref_a && have_ref_b) {
            const test_float_compare_stats key_stats =
                test_compare_float_bits(ref_a, ref_b, count_a);
            key_difference = key_stats.mismatch_count;
            TEST_ASSERT(key_difference != 0);
        }

        TEST_ASSERT(unsetenv(disable_env) == 0);

        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }
        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }

        if (kind != TEST_METAL_PREFILL_MASK_CACHE_RAW) {
            if (test_metal_zero_prefix_prefill_mask_cache_run(
                    kind, heads, model_raw, page, q, raw, comp, comp_mask,
                    &shape_a, true, n_head, head_dim, masked_actual) &&
                have_ref_a) {
                const test_float_compare_stats masked_stats =
                    test_compare_float_bits(ref_a, masked_actual, count_a);
                masked_difference = masked_stats.mismatch_count;
                TEST_ASSERT(masked_difference != 0);
            }
            if (test_metal_zero_prefix_prefill_mask_cache_run(
                    kind, heads, model_raw, page, q, raw, comp, comp_mask,
                    &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
                test_metal_zero_prefix_prefill_mask_cache_compare(
                    ref_a, actual, count_a,
                    &total_mismatches, &max_ulp);
            }
        }

        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_b, false, n_head, head_dim, actual) && have_ref_b) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_b, actual, count_b, &total_mismatches, &max_ulp);
        }
        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }
    }

    test_restore_env(disable_env, saved_disable);
    const char *kind_name = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? "raw" :
        (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? "ratio4" : "ratio128");
    fprintf(stderr,
            "ds4-test: zero-prefix prefill mask cache %s exact "
            "mismatches=%zu max_ulp=%u key_diff=%zu masked_diff=%zu\n",
            kind_name, total_mismatches, max_ulp,
            key_difference, masked_difference);
    TEST_ASSERT(total_mismatches == 0);
    TEST_ASSERT(max_ulp == 0);

    free(model_raw);
    free(masked_actual);
    free(actual);
    free(ref_b);
    free(ref_a);
    free(mask_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(comp_mask);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_zero_prefix_prefill_mask_cache_exact(void) {
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RAW, 41);
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RATIO4, 43);
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RATIO128, 47);
}

#endif

#ifndef DS4_NO_GPU
static void test_laguna_prefill_attention_numeric_case(
        uint32_t n_head,
        uint32_t n_head_kv) {
    const uint32_t head_dim = 128u;
    const uint32_t n_tokens = 17u;
    const uint32_t cache_cap = 64u;
    const uint32_t cache_width = n_head_kv * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const uint64_t q_values = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_values =
        (uint64_t)n_tokens * n_head_kv * head_dim;
    const uint64_t gate_values = (uint64_t)n_tokens * n_head;
    const uint64_t cache_values = (uint64_t)cache_cap * cache_width;

    ds4_gpu_tensor *heads =
        ds4_gpu_tensor_alloc(q_values * sizeof(float));
    ds4_gpu_tensor *key_cache =
        ds4_gpu_tensor_alloc(cache_values * sizeof(uint16_t));
    ds4_gpu_tensor *value_cache =
        ds4_gpu_tensor_alloc(cache_values * sizeof(uint16_t));
    ds4_gpu_tensor *staged_key =
        ds4_gpu_tensor_alloc(kv_values * sizeof(uint16_t));
    ds4_gpu_tensor *staged_value =
        ds4_gpu_tensor_alloc(kv_values * sizeof(uint16_t));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_values * sizeof(float));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(kv_values * sizeof(float));
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(kv_values * sizeof(float));
    ds4_gpu_tensor *gate =
        ds4_gpu_tensor_alloc(gate_values * sizeof(float));
    float *q_host = malloc((size_t)q_values * sizeof(float));
    float *k_host = malloc((size_t)kv_values * sizeof(float));
    float *v_host = malloc((size_t)kv_values * sizeof(float));
    float *gate_host = malloc((size_t)gate_values * sizeof(float));
    float *actual = malloc((size_t)q_values * sizeof(float));
    TEST_ASSERT(heads && key_cache && value_cache && staged_key &&
                staged_value && q && k && v && gate && q_host && k_host &&
                v_host && gate_host && actual);
    if (!heads || !key_cache || !value_cache || !staged_key ||
        !staged_value || !q || !k || !v || !gate || !q_host || !k_host ||
        !v_host || !gate_host || !actual) {
        goto cleanup;
    }

    for (uint64_t i = 0; i < q_values; i++) {
        const int value =
            (int)((i * 37u + (i >> 3u) * 11u + 5u) % 191u) - 95;
        q_host[i] = (float)value / 384.0f;
    }
    for (uint64_t i = 0; i < kv_values; i++) {
        const int key_value =
            (int)((i * 29u + (i >> 2u) * 17u + 7u) % 181u) - 90;
        const int value_value =
            (int)((i * 31u + (i >> 4u) * 13u + 3u) % 173u) - 86;
        k_host[i] = (float)key_value / 352.0f;
        v_host[i] = (float)value_value / 320.0f;
    }
    for (uint64_t i = 0; i < gate_values; i++) {
        gate_host[i] = ((float)((int)(i % 13u) - 6)) * 0.1875f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(
                    q, 0, q_host, q_values * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    k, 0, k_host, kv_values * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    v, 0, v_host, kv_values * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    gate, 0, gate_host,
                    gate_values * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_laguna_attention_prefill_tensor(
                    heads, key_cache, value_cache, staged_key, staged_value,
                    q, k, v, gate, 0u, n_tokens, cache_cap,
                    n_head, n_head_kv, head_dim, scale, 0) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, actual, q_values * sizeof(float)) != 0);

    {
        double sum_squared = 0.0;
        float max_abs = 0.0f;
        size_t nonfinite = 0u;
        for (uint32_t token = 0; token < n_tokens; token++) {
            for (uint32_t head = 0; head < n_head; head++) {
                const uint32_t kv_head =
                    head / (n_head / n_head_kv);
                double max_score = -DBL_MAX;
                for (uint32_t key = 0; key <= token; key++) {
                    const uint64_t q_base =
                        ((uint64_t)token * n_head + head) * head_dim;
                    const uint64_t kv_base =
                        ((uint64_t)key * n_head_kv + kv_head) * head_dim;
                    double dot = 0.0;
                    for (uint32_t d = 0; d < head_dim; d++) {
                        dot += (double)q_host[q_base + d] *
                            test_f16_to_f32(
                                test_float_to_f16(k_host[kv_base + d]));
                    }
                    const double score = dot * (double)scale;
                    if (score > max_score) max_score = score;
                }
                double denominator = 0.0;
                double numerator[128] = {0};
                for (uint32_t key = 0; key <= token; key++) {
                    const uint64_t q_base =
                        ((uint64_t)token * n_head + head) * head_dim;
                    const uint64_t kv_base =
                        ((uint64_t)key * n_head_kv + kv_head) * head_dim;
                    double dot = 0.0;
                    for (uint32_t d = 0; d < head_dim; d++) {
                        dot += (double)q_host[q_base + d] *
                            test_f16_to_f32(
                                test_float_to_f16(k_host[kv_base + d]));
                    }
                    const double weight =
                        exp(dot * (double)scale - max_score);
                    denominator += weight;
                    for (uint32_t d = 0; d < head_dim; d++) {
                        numerator[d] += weight *
                            test_f16_to_f32(
                                test_float_to_f16(v_host[kv_base + d]));
                    }
                }
                const double gate_scale =
                    log1p(exp((double)gate_host[
                        (uint64_t)token * n_head + head]));
                const uint64_t out_base =
                    ((uint64_t)token * n_head + head) * head_dim;
                for (uint32_t d = 0; d < head_dim; d++) {
                    const double reference =
                        numerator[d] / denominator * gate_scale;
                    const float got = actual[out_base + d];
                    if (!isfinite(got)) {
                        nonfinite++;
                        continue;
                    }
                    const float error = fabsf(got - (float)reference);
                    if (error > max_abs) max_abs = error;
                    sum_squared += (double)error * error;
                }
            }
        }
        const float rms = sqrtf((float)(
            sum_squared / (double)q_values));
        fprintf(stderr,
                "ds4-test: Laguna GQA%u prefill numeric "
                "max_abs=%g rms=%g nonfinite=%zu\n",
                n_head / n_head_kv, max_abs, rms, nonfinite);
        TEST_ASSERT(nonfinite == 0u);
        TEST_ASSERT(max_abs < 5.0e-4f);
        TEST_ASSERT(rms < 1.0e-4f);
    }

    /* Reuse row zero as the next token. This checks that batched prefill
     * committed the staged KV rows in the exact ring layout consumed by
     * ordinary decoding. */
    TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                    heads, key_cache, value_cache, q, k, v, gate,
                    n_tokens, cache_cap, 0u, n_tokens + 1u,
                    n_head, n_head_kv, head_dim, scale) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, actual,
                    (uint64_t)n_head * head_dim * sizeof(float)) != 0);
    {
        double sum_squared = 0.0;
        float max_abs = 0.0f;
        size_t nonfinite = 0u;
        for (uint32_t head = 0; head < n_head; head++) {
            const uint32_t kv_head = head / (n_head / n_head_kv);
            double score[18];
            double max_score = -DBL_MAX;
            for (uint32_t key = 0; key <= n_tokens; key++) {
                const uint32_t source = key == n_tokens ? 0u : key;
                const uint64_t kv_base =
                    ((uint64_t)source * n_head_kv + kv_head) * head_dim;
                const uint64_t q_base = (uint64_t)head * head_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += (double)q_host[q_base + d] *
                        test_f16_to_f32(
                            test_float_to_f16(k_host[kv_base + d]));
                }
                score[key] = dot * (double)scale;
                if (score[key] > max_score) max_score = score[key];
            }
            double denominator = 0.0;
            double numerator[128] = {0};
            for (uint32_t key = 0; key <= n_tokens; key++) {
                const uint32_t source = key == n_tokens ? 0u : key;
                const uint64_t kv_base =
                    ((uint64_t)source * n_head_kv + kv_head) * head_dim;
                const double weight = exp(score[key] - max_score);
                denominator += weight;
                for (uint32_t d = 0; d < head_dim; d++) {
                    numerator[d] += weight *
                        test_f16_to_f32(
                            test_float_to_f16(v_host[kv_base + d]));
                }
            }
            const double gate_scale =
                log1p(exp((double)gate_host[head]));
            const uint64_t out_base = (uint64_t)head * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) {
                const double reference =
                    numerator[d] / denominator * gate_scale;
                const float got = actual[out_base + d];
                if (!isfinite(got)) {
                    nonfinite++;
                    continue;
                }
                const float error = fabsf(got - (float)reference);
                if (error > max_abs) max_abs = error;
                sum_squared += (double)error * error;
            }
        }
        const float rms = sqrtf((float)(
            sum_squared / ((double)n_head * head_dim)));
        fprintf(stderr,
                "ds4-test: Laguna GQA%u prefill/decode transition "
                "max_abs=%g rms=%g nonfinite=%zu\n",
                n_head / n_head_kv, max_abs, rms, nonfinite);
        TEST_ASSERT(nonfinite == 0u);
        TEST_ASSERT(max_abs < 5.0e-4f);
        TEST_ASSERT(rms < 1.0e-4f);
    }

cleanup:
    free(actual);
    free(gate_host);
    free(v_host);
    free(k_host);
    free(q_host);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(staged_value);
    ds4_gpu_tensor_free(staged_key);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
}

static void test_laguna_decode_cpu_reference(
        float          *out,
        const float    *q_host,
        const float    *gate_host,
        const uint16_t *key_cache,
        const uint16_t *value_cache,
        uint32_t        key_start,
        uint32_t        key_count,
        uint32_t        cache_cap,
        uint32_t        n_head,
        uint32_t        n_head_kv,
        uint32_t        head_dim,
        float           scale) {
    const uint32_t cache_width = n_head_kv * head_dim;
    for (uint32_t h = 0; h < n_head; h++) {
        const uint32_t kv_head = h / (n_head / n_head_kv);
        double max_score = -DBL_MAX;
        for (uint32_t i = 0; i < key_count; i++) {
            const uint32_t source = (key_start + i) % cache_cap;
            const uint64_t kv_base =
                (uint64_t)source * cache_width +
                (uint64_t)kv_head * head_dim;
            double dot = 0.0;
            for (uint32_t d = 0; d < head_dim; d++) {
                dot += (double)q_host[(uint64_t)h * head_dim + d] *
                    test_f16_to_f32(key_cache[kv_base + d]);
            }
            const double score = dot * (double)scale;
            if (score > max_score) max_score = score;
        }

        double denominator = 0.0;
        double numerator[128] = {0.0};
        for (uint32_t i = 0; i < key_count; i++) {
            const uint32_t source = (key_start + i) % cache_cap;
            const uint64_t kv_base =
                (uint64_t)source * cache_width +
                (uint64_t)kv_head * head_dim;
            double dot = 0.0;
            for (uint32_t d = 0; d < head_dim; d++) {
                dot += (double)q_host[(uint64_t)h * head_dim + d] *
                    test_f16_to_f32(key_cache[kv_base + d]);
            }
            const double weight = exp(dot * (double)scale - max_score);
            denominator += weight;
            for (uint32_t d = 0; d < head_dim; d++) {
                numerator[d] += weight *
                    test_f16_to_f32(value_cache[kv_base + d]);
            }
        }
        const double gate_scale = log1p(exp((double)gate_host[h]));
        for (uint32_t d = 0; d < head_dim; d++) {
            out[(uint64_t)h * head_dim + d] =
                (float)(numerator[d] / denominator * gate_scale);
        }
    }
}

/* The production Laguna SWA layer is GQA9 (72 query heads over 8 KV heads).
 * Keep this A/B test independent of the model and cover representative
 * full-ring positions, including the first full ring and wrap boundaries.
 * The grouped path is deliberately selected only by the production flag; if
 * its pipeline cannot be built, the Metal entry point returns failure rather
 * than falling back to the ordinary vector path. */
static void test_metal_laguna_swa_gqa3_numeric_ab(void) {
    static const uint32_t positions[] = {
        511u, 512u, 513u, 516u, 767u, 1020u, 1023u, 1024u,
    };
    const char *env_name = "DS4_METAL_LAGUNA_SWA_GQA3";
    const char *staged_env_name = "DS4_METAL_LAGUNA_STAGED_SWA";
    const char *gqa9_env_name = "DS4_METAL_LAGUNA_SWA_GQA9";
    const uint32_t head_dim = 128u;
    const uint32_t n_head = 72u;
    const uint32_t n_head_kv = 8u;
    const uint32_t cache_cap = 512u;
    const uint32_t key_count = 512u;
    const uint32_t cache_width = n_head_kv * head_dim;
    const uint64_t head_values = (uint64_t)n_head * head_dim;
    const uint64_t cache_values = (uint64_t)cache_cap * cache_width;
    const uint64_t cache_bytes = cache_values * sizeof(uint16_t);
    const uint64_t heads_bytes = head_values * sizeof(float);
    const uint64_t current_kv_values = cache_width;
    const uint64_t current_kv_bytes = current_kv_values * sizeof(float);
    const uint64_t gate_bytes = (uint64_t)n_head * sizeof(float);
    const float scale = 1.0f / sqrtf((float)head_dim);

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *key_cache = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *value_cache = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(gate_bytes);
    uint16_t *initial_key = malloc((size_t)cache_bytes);
    uint16_t *initial_value = malloc((size_t)cache_bytes);
    uint16_t *poisoned_key = malloc((size_t)cache_bytes);
    uint16_t *poisoned_value = malloc((size_t)cache_bytes);
    uint16_t *expected_key = malloc((size_t)cache_bytes);
    uint16_t *expected_value = malloc((size_t)cache_bytes);
    uint16_t *off_key = malloc((size_t)cache_bytes);
    uint16_t *off_value = malloc((size_t)cache_bytes);
    uint16_t *on_key = malloc((size_t)cache_bytes);
    uint16_t *on_value = malloc((size_t)cache_bytes);
    float *q_host = malloc((size_t)heads_bytes);
    float *k_host = malloc((size_t)current_kv_bytes);
    float *v_host = malloc((size_t)current_kv_bytes);
    float *gate_host = malloc((size_t)gate_bytes);
    float *heads_seed = malloc((size_t)heads_bytes);
    float *off_heads = malloc((size_t)heads_bytes);
    float *on_heads = malloc((size_t)heads_bytes);
    float *reference = malloc((size_t)heads_bytes);
    float *stale_reference = malloc((size_t)heads_bytes);
    char *saved_env = test_save_env(env_name);
    char *saved_staged_env = test_save_env(staged_env_name);
    char *saved_gqa9_env = test_save_env(gqa9_env_name);

    TEST_ASSERT(heads && key_cache && value_cache && q && k && v && gate &&
                initial_key && initial_value && expected_key &&
                poisoned_key && poisoned_value &&
                expected_value && off_key && off_value && on_key && on_value &&
                q_host && k_host && v_host && gate_host && heads_seed &&
                off_heads && on_heads && reference && stale_reference);
    if (!heads || !key_cache || !value_cache || !q || !k || !v || !gate ||
        !initial_key || !initial_value || !poisoned_key || !poisoned_value ||
        !expected_key || !expected_value || !off_key || !off_value ||
        !on_key || !on_value || !q_host ||
        !k_host || !v_host || !gate_host || !heads_seed || !off_heads ||
        !on_heads || !reference || !stale_reference) {
        goto cleanup;
    }

    /* This standalone A/B is specifically the grouped-vs-ordinary
     * experiment.  Isolate it from an externally exported staged-SWA flag;
     * the staged precedence contract is covered separately below. */
    TEST_ASSERT(unsetenv(staged_env_name) == 0);
    TEST_ASSERT(unsetenv(gqa9_env_name) == 0);

    float max_ab_abs = 0.0f;
    float max_ab_rms = 0.0f;
    uint32_t max_ab_ulp = 0u;
    size_t total_ab_mismatches = 0u;

    for (uint64_t i = 0; i < cache_values; i++) {
        const int key_value =
            (int)((i * 29u + (i >> 4u) * 17u + 13u) % 251u) - 125;
        const int value_value =
            (int)((i * 31u + (i >> 5u) * 11u + 7u) % 239u) - 119;
        initial_key[i] = test_float_to_f16((float)key_value / 192.0f);
        initial_value[i] = test_float_to_f16((float)value_value / 176.0f);
    }
    for (uint64_t i = 0; i < head_values; i++) {
        const int value =
            (int)((i * 37u + (i >> 3u) * 19u + 5u) % 257u) - 128;
        q_host[i] = (float)value / 192.0f;
        heads_seed[i] = 17.0f + (float)(i % 97u) / 32.0f;
    }
    for (uint64_t i = 0; i < current_kv_values; i++) {
        const int key_value =
            (int)((i * 41u + (i >> 4u) * 23u + 3u) % 233u) - 116;
        const int value_value =
            (int)((i * 43u + (i >> 3u) * 13u + 17u) % 227u) - 113;
        k_host[i] = (float)key_value / 168.0f;
        v_host[i] = (float)value_value / 156.0f;
    }
    for (uint32_t h = 0; h < n_head; h++) {
        gate_host[h] = ((float)((int)(h % 17u) - 8)) * 0.21875f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    k, 0, k_host, current_kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    v, 0, v_host, current_kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(gate, 0, gate_host, gate_bytes) != 0);

    for (size_t ci = 0; ci < sizeof(positions) / sizeof(positions[0]); ci++) {
        const uint32_t pos = positions[ci];
        const uint32_t key_start = pos + 1u >= key_count
            ? pos + 1u - key_count
            : 0u;
        const uint32_t cache_row = pos % cache_cap;
        const uint64_t row_base = (uint64_t)cache_row * cache_width;
        memcpy(poisoned_key, initial_key, (size_t)cache_bytes);
        memcpy(poisoned_value, initial_value, (size_t)cache_bytes);
        /* The overwritten row must be materially unlike the current K/V.  A
         * stale attention-before-store implementation should therefore leave
         * a visible output error, rather than being hidden by a similar
         * sentinel row. */
        for (uint32_t i = 0; i < cache_width; i++) {
            poisoned_key[row_base + i] = test_float_to_f16(
                11.0f + (float)(i % 19u) / 16.0f);
            poisoned_value[row_base + i] = test_float_to_f16(
                37.0f + (float)(i % 23u) / 12.0f);
        }
        memcpy(expected_key, poisoned_key, (size_t)cache_bytes);
        memcpy(expected_value, poisoned_value, (size_t)cache_bytes);
        for (uint32_t i = 0; i < cache_width; i++) {
            expected_key[row_base + i] = test_float_to_f16(k_host[i]);
            expected_value[row_base + i] = test_float_to_f16(v_host[i]);
        }

        test_laguna_decode_cpu_reference(
            reference, q_host, gate_host, expected_key, expected_value,
            key_start, key_count, cache_cap, n_head, n_head_kv,
            head_dim, scale);
        test_laguna_decode_cpu_reference(
            stale_reference, q_host, gate_host, poisoned_key, poisoned_value,
            key_start, key_count, cache_cap, n_head, n_head_kv,
            head_dim, scale);

        /* Flag off: this is the production-sized ordinary full-ring path. */
        TEST_ASSERT(ds4_gpu_tensor_write(
                        heads, 0, heads_seed, heads_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        key_cache, 0, poisoned_key, cache_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        value_cache, 0, poisoned_value, cache_bytes) != 0);
        TEST_ASSERT(setenv(env_name, "0", 1) == 0);
        TEST_ASSERT(test_restart_metal_lifecycle());
        ds4_gpu_test_laguna_route_counters_reset();
        TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                        heads, key_cache, value_cache, q, k, v, gate,
                        pos, cache_cap, key_start, key_count,
                        n_head, n_head_kv, head_dim, scale) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        heads, 0, off_heads, heads_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        key_cache, 0, off_key, cache_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        value_cache, 0, off_value, cache_bytes) != 0);
        {
            uint64_t ordinary = 0;
            uint64_t gqa3 = 0;
            uint64_t gqa9 = 0;
            uint64_t staged = 0;
            uint64_t global_grouped = 0;
            TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                            &ordinary, &gqa3, &gqa9, &staged,
                            &global_grouped) != 0);
            TEST_ASSERT(ordinary == 1u && gqa3 == 0u && gqa9 == 0u &&
                        staged == 0u && global_grouped == 0u);
        }

        /* Flag on: the grouped pipeline is mandatory for this leg. */
        TEST_ASSERT(ds4_gpu_tensor_write(
                        heads, 0, heads_seed, heads_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        key_cache, 0, poisoned_key, cache_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        value_cache, 0, poisoned_value, cache_bytes) != 0);
        TEST_ASSERT(setenv(env_name, "1", 1) == 0);
        TEST_ASSERT(test_restart_metal_lifecycle());
        ds4_gpu_test_laguna_route_counters_reset();
        TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                        heads, key_cache, value_cache, q, k, v, gate,
                        pos, cache_cap, key_start, key_count,
                        n_head, n_head_kv, head_dim, scale) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        heads, 0, on_heads, heads_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        key_cache, 0, on_key, cache_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        value_cache, 0, on_value, cache_bytes) != 0);
        {
            uint64_t ordinary = 0;
            uint64_t gqa3 = 0;
            uint64_t gqa9 = 0;
            uint64_t staged = 0;
            uint64_t global_grouped = 0;
            TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                            &ordinary, &gqa3, &gqa9, &staged,
                            &global_grouped) != 0);
            TEST_ASSERT(ordinary == 0u && gqa3 == 1u && gqa9 == 0u &&
                        staged == 0u && global_grouped == 0u);
        }

        const test_float_compare_stats off_cpu =
            test_compare_float_bits(reference, off_heads, head_values);
        const test_float_compare_stats on_cpu =
            test_compare_float_bits(reference, on_heads, head_values);
        const test_float_compare_stats ab =
            test_compare_float_bits(off_heads, on_heads, head_values);
        const test_float_compare_stats stale =
            test_compare_float_bits(reference, stale_reference, head_values);
        double off_rms_sum = 0.0;
        double on_rms_sum = 0.0;
        double ab_rms_sum = 0.0;
        double stale_rms_sum = 0.0;
        size_t nonfinite = 0u;
        for (uint64_t i = 0; i < head_values; i++) {
            const float off_error = fabsf(off_heads[i] - reference[i]);
            const float on_error = fabsf(on_heads[i] - reference[i]);
            const float ab_error = fabsf(off_heads[i] - on_heads[i]);
            const float stale_error =
                fabsf(reference[i] - stale_reference[i]);
            off_rms_sum += (double)off_error * off_error;
            on_rms_sum += (double)on_error * on_error;
            ab_rms_sum += (double)ab_error * ab_error;
            stale_rms_sum += (double)stale_error * stale_error;
            if (!isfinite(off_heads[i]) || !isfinite(on_heads[i])) {
                nonfinite++;
            }
        }
        const float off_rms = sqrtf((float)(off_rms_sum / head_values));
        const float on_rms = sqrtf((float)(on_rms_sum / head_values));
        const float ab_rms = sqrtf((float)(ab_rms_sum / head_values));
        const float stale_rms =
            sqrtf((float)(stale_rms_sum / head_values));
        if (ab.max_abs > max_ab_abs) max_ab_abs = ab.max_abs;
        if (ab_rms > max_ab_rms) max_ab_rms = ab_rms;
        if (ab.max_ulp > max_ab_ulp) max_ab_ulp = ab.max_ulp;
        total_ab_mismatches += ab.mismatch_count;
        fprintf(stderr,
                "ds4-test: Laguna SWA GQA3 A/B pos=%u key_start=%u "
                "cpu_off=%zu/%u/%g/%g cpu_on=%zu/%u/%g/%g "
                "ab=%zu/%u/%g/%g stale=%zu/%u/%g/%g exact=%s "
                "cache=%s/%s nonfinite=%zu\n",
                pos, key_start,
                off_cpu.mismatch_count, off_cpu.max_ulp,
                off_cpu.max_abs, off_rms,
                on_cpu.mismatch_count, on_cpu.max_ulp,
                on_cpu.max_abs, on_rms,
                ab.mismatch_count, ab.max_ulp, ab.max_abs, ab_rms,
                stale.mismatch_count, stale.max_ulp,
                stale.max_abs, stale_rms,
                ab.mismatch_count == 0u ? "yes" : "no",
                memcmp(off_key, expected_key, (size_t)cache_bytes) == 0
                    ? "off-exact" : "off-drift",
                memcmp(on_key, expected_key, (size_t)cache_bytes) == 0
                    ? "on-exact" : "on-drift",
                nonfinite);

        TEST_ASSERT(nonfinite == 0u);
        TEST_ASSERT(off_cpu.max_abs < 5.0e-4f);
        TEST_ASSERT(off_rms < 1.0e-4f);
        TEST_ASSERT(on_cpu.max_abs < 5.0e-4f);
        TEST_ASSERT(on_rms < 1.0e-4f);
        TEST_ASSERT(stale.max_abs > 1.0e-3f);
        TEST_ASSERT(ab.max_abs < 1.0e-6f);
        TEST_ASSERT(ab_rms < 1.0e-7f);
        TEST_ASSERT(memcmp(off_key, expected_key, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(off_value, expected_value, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(on_key, expected_key, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(on_value, expected_value, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(off_key, on_key, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(off_value, on_value, (size_t)cache_bytes) == 0);
    }

    fprintf(stderr,
            "ds4-test: Laguna SWA GQA3 A/B aggregate cases=%zu "
            "mismatches=%zu max_ulp=%u max_abs=%g rms=%g "
            "bound_abs=1e-6 bound_rms=1e-7\n",
            sizeof(positions) / sizeof(positions[0]), total_ab_mismatches,
            max_ab_ulp, max_ab_abs, max_ab_rms);
    TEST_ASSERT(max_ab_abs < 1.0e-6f);
    TEST_ASSERT(max_ab_rms < 1.0e-7f);
    /* A nonzero aggregate delta is the positive path-selection guard: the
     * two legs start from identical state, caches are byte-identical, and
     * grouped/ordinary reductions have measurably different rounding. */
    TEST_ASSERT(total_ab_mismatches > 0u);
    TEST_ASSERT(max_ab_abs > 1.0e-12f);

    /* Combined-flags contract for the single-token entry point: staged SWA
     * takes precedence, so exporting both flags must leave this route
     * bit-exact with the ordinary path.  This call reaches the same encoder
     * whose grouped leg above was proven distinct, and therefore exercises
     * the explicit source-level suppression rather than merely comparing two
     * staged-prefill implementations. */
    const uint32_t combined_pos =
        positions[sizeof(positions) / sizeof(positions[0]) - 1u];
    const uint32_t combined_key_start = combined_pos + 1u - key_count;
    TEST_ASSERT(setenv(staged_env_name, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, heads_seed, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, poisoned_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, poisoned_value, cache_bytes) != 0);
    /* Exercise staged > GQA3 with both selectors enabled in the first leg;
     * the second leg disables GQA3 while retaining staged, so the selected
     * staged route and output must remain identical. */
    TEST_ASSERT(setenv(env_name, "1", 1) == 0);
    TEST_ASSERT(test_restart_metal_lifecycle());
    ds4_gpu_test_laguna_route_counters_reset();
    TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                    heads, key_cache, value_cache, q, k, v, gate,
                    combined_pos, cache_cap, combined_key_start, key_count,
                    n_head, n_head_kv, head_dim, scale) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, off_heads, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, off_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, off_value, cache_bytes) != 0);
    {
        uint64_t ordinary = 0;
        uint64_t gqa3 = 0;
        uint64_t gqa9 = 0;
        uint64_t staged = 0;
        uint64_t global_grouped = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        TEST_ASSERT(ordinary == 0u && gqa3 == 0u && gqa9 == 0u &&
                    staged == 1u && global_grouped == 0u);
    }

    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, heads_seed, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, poisoned_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, poisoned_value, cache_bytes) != 0);
    TEST_ASSERT(setenv(env_name, "0", 1) == 0);
    TEST_ASSERT(test_restart_metal_lifecycle());
    ds4_gpu_test_laguna_route_counters_reset();
    TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                    heads, key_cache, value_cache, q, k, v, gate,
                    combined_pos, cache_cap, combined_key_start, key_count,
                    n_head, n_head_kv, head_dim, scale) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, on_heads, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, on_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, on_value, cache_bytes) != 0);
    {
        uint64_t ordinary = 0;
        uint64_t gqa3 = 0;
        uint64_t gqa9 = 0;
        uint64_t staged = 0;
        uint64_t global_grouped = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        TEST_ASSERT(ordinary == 1u && gqa3 == 0u && gqa9 == 0u &&
                    staged == 0u && global_grouped == 0u);
    }
    const test_float_compare_stats combined_stats =
        test_compare_float_bits(off_heads, on_heads, head_values);
    fprintf(stderr,
            "ds4-test: Laguna SWA GQA3 combined precedence "
            "mismatches=%zu max_ulp=%u max_abs=%g cache=%s/%s\n",
            combined_stats.mismatch_count, combined_stats.max_ulp,
            combined_stats.max_abs,
            memcmp(off_key, on_key, (size_t)cache_bytes) == 0
                ? "key-exact" : "key-drift",
            memcmp(off_value, on_value, (size_t)cache_bytes) == 0
                ? "value-exact" : "value-drift");
    TEST_ASSERT(combined_stats.mismatch_count == 0u);
    TEST_ASSERT(memcmp(off_key, on_key, (size_t)cache_bytes) == 0);
    TEST_ASSERT(memcmp(off_value, on_value, (size_t)cache_bytes) == 0);

cleanup:
    test_restore_env(gqa9_env_name, saved_gqa9_env);
    test_restore_env(staged_env_name, saved_staged_env);
    test_restore_env(env_name, saved_env);
    free(stale_reference);
    free(reference);
    free(on_heads);
    free(off_heads);
    free(heads_seed);
    free(gate_host);
    free(v_host);
    free(k_host);
    free(q_host);
    free(poisoned_value);
    free(poisoned_key);
    free(on_value);
    free(on_key);
    free(off_value);
    free(off_key);
    free(expected_value);
    free(expected_key);
    free(initial_value);
    free(initial_key);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
}

/* GQA9 sibling of the SWA grouped A/B: the production 72/8 ring has
 * heads_per_kv = 9, so the GQA9 kernel evaluates all 9 heads of a KV head in
 * one simdgroup instead of GQA3's 3.  The path is selected only by the
 * production flag; if its pipeline cannot be built the Metal entry point
 * returns failure rather than falling back.  Precedence is exercised by
 * construction: staged SWA > GQA9 > GQA3, so the main A/B isolates both
 * lower-precedence knobs, then two combined legs prove GQA9 suppresses GQA3
 * and staged SWA suppresses GQA9 (each bit-exact with the winning path
 * alone). */
static void test_metal_laguna_swa_gqa9_numeric_ab(void) {
    static const uint32_t positions[] = {
        511u, 512u, 513u, 516u, 767u, 1020u, 1023u, 1024u,
    };
    const char *env_name = "DS4_METAL_LAGUNA_SWA_GQA9";
    const char *staged_env_name = "DS4_METAL_LAGUNA_STAGED_SWA";
    const char *gqa3_env_name = "DS4_METAL_LAGUNA_SWA_GQA3";
    const uint32_t head_dim = 128u;
    const uint32_t n_head = 72u;
    const uint32_t n_head_kv = 8u;
    const uint32_t cache_cap = 512u;
    const uint32_t key_count = 512u;
    const uint32_t cache_width = n_head_kv * head_dim;
    const uint64_t head_values = (uint64_t)n_head * head_dim;
    const uint64_t cache_values = (uint64_t)cache_cap * cache_width;
    const uint64_t cache_bytes = cache_values * sizeof(uint16_t);
    const uint64_t heads_bytes = head_values * sizeof(float);
    const uint64_t current_kv_values = cache_width;
    const uint64_t current_kv_bytes = current_kv_values * sizeof(float);
    const uint64_t gate_bytes = (uint64_t)n_head * sizeof(float);
    const float scale = 1.0f / sqrtf((float)head_dim);

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *key_cache = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *value_cache = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(gate_bytes);
    uint16_t *initial_key = malloc((size_t)cache_bytes);
    uint16_t *initial_value = malloc((size_t)cache_bytes);
    uint16_t *poisoned_key = malloc((size_t)cache_bytes);
    uint16_t *poisoned_value = malloc((size_t)cache_bytes);
    uint16_t *expected_key = malloc((size_t)cache_bytes);
    uint16_t *expected_value = malloc((size_t)cache_bytes);
    uint16_t *off_key = malloc((size_t)cache_bytes);
    uint16_t *off_value = malloc((size_t)cache_bytes);
    uint16_t *on_key = malloc((size_t)cache_bytes);
    uint16_t *on_value = malloc((size_t)cache_bytes);
    float *q_host = malloc((size_t)heads_bytes);
    float *k_host = malloc((size_t)current_kv_bytes);
    float *v_host = malloc((size_t)current_kv_bytes);
    float *gate_host = malloc((size_t)gate_bytes);
    float *heads_seed = malloc((size_t)heads_bytes);
    float *off_heads = malloc((size_t)heads_bytes);
    float *on_heads = malloc((size_t)heads_bytes);
    float *reference = malloc((size_t)heads_bytes);
    float *stale_reference = malloc((size_t)heads_bytes);
    char *saved_env = test_save_env(env_name);
    char *saved_staged_env = test_save_env(staged_env_name);
    char *saved_gqa3_env = test_save_env(gqa3_env_name);

    TEST_ASSERT(heads && key_cache && value_cache && q && k && v && gate &&
                initial_key && initial_value && expected_key &&
                poisoned_key && poisoned_value &&
                expected_value && off_key && off_value && on_key && on_value &&
                q_host && k_host && v_host && gate_host && heads_seed &&
                off_heads && on_heads && reference && stale_reference);
    if (!heads || !key_cache || !value_cache || !q || !k || !v || !gate ||
        !initial_key || !initial_value || !poisoned_key || !poisoned_value ||
        !expected_key || !expected_value || !off_key || !off_value ||
        !on_key || !on_value || !q_host ||
        !k_host || !v_host || !gate_host || !heads_seed || !off_heads ||
        !on_heads || !reference || !stale_reference) {
        goto cleanup;
    }

    /* Isolate the GQA9 experiment from both lower-precedence knobs so the A/B
     * is purely GQA9-vs-ordinary; the staged and GQA3 precedence contracts
     * are covered separately below. */
    TEST_ASSERT(unsetenv(staged_env_name) == 0);
    TEST_ASSERT(unsetenv(gqa3_env_name) == 0);
    TEST_ASSERT(setenv(env_name, "1", 1) == 0);
    TEST_ASSERT(test_restart_metal_lifecycle());
    if (ds4_gpu_laguna_swa_gqa9_preflight(
            cache_cap, key_count, n_head, n_head_kv, head_dim) < 0) {
        fprintf(stderr,
                "ds4-test: Laguna SWA GQA9 A/B skipped; "
                "source does not provide the optional GQA9 route\n");
        goto cleanup;
    }

    float max_ab_abs = 0.0f;
    float max_ab_rms = 0.0f;
    uint32_t max_ab_ulp = 0u;
    size_t total_ab_mismatches = 0u;

    for (uint64_t i = 0; i < cache_values; i++) {
        const int key_value =
            (int)((i * 29u + (i >> 4u) * 17u + 13u) % 251u) - 125;
        const int value_value =
            (int)((i * 31u + (i >> 5u) * 11u + 7u) % 239u) - 119;
        initial_key[i] = test_float_to_f16((float)key_value / 192.0f);
        initial_value[i] = test_float_to_f16((float)value_value / 176.0f);
    }
    for (uint64_t i = 0; i < head_values; i++) {
        const int value =
            (int)((i * 37u + (i >> 3u) * 19u + 5u) % 257u) - 128;
        q_host[i] = (float)value / 192.0f;
        heads_seed[i] = 17.0f + (float)(i % 97u) / 32.0f;
    }
    for (uint64_t i = 0; i < current_kv_values; i++) {
        const int key_value =
            (int)((i * 41u + (i >> 4u) * 23u + 3u) % 233u) - 116;
        const int value_value =
            (int)((i * 43u + (i >> 3u) * 13u + 17u) % 227u) - 113;
        k_host[i] = (float)key_value / 168.0f;
        v_host[i] = (float)value_value / 156.0f;
    }
    for (uint32_t h = 0; h < n_head; h++) {
        gate_host[h] = ((float)((int)(h % 17u) - 8)) * 0.21875f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    k, 0, k_host, current_kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    v, 0, v_host, current_kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(gate, 0, gate_host, gate_bytes) != 0);

    for (size_t ci = 0; ci < sizeof(positions) / sizeof(positions[0]); ci++) {
        const uint32_t pos = positions[ci];
        const uint32_t key_start = pos + 1u >= key_count
            ? pos + 1u - key_count
            : 0u;
        const uint32_t cache_row = pos % cache_cap;
        const uint64_t row_base = (uint64_t)cache_row * cache_width;
        memcpy(poisoned_key, initial_key, (size_t)cache_bytes);
        memcpy(poisoned_value, initial_value, (size_t)cache_bytes);
        for (uint32_t i = 0; i < cache_width; i++) {
            poisoned_key[row_base + i] = test_float_to_f16(
                11.0f + (float)(i % 19u) / 16.0f);
            poisoned_value[row_base + i] = test_float_to_f16(
                37.0f + (float)(i % 23u) / 12.0f);
        }
        memcpy(expected_key, poisoned_key, (size_t)cache_bytes);
        memcpy(expected_value, poisoned_value, (size_t)cache_bytes);
        for (uint32_t i = 0; i < cache_width; i++) {
            expected_key[row_base + i] = test_float_to_f16(k_host[i]);
            expected_value[row_base + i] = test_float_to_f16(v_host[i]);
        }

        test_laguna_decode_cpu_reference(
            reference, q_host, gate_host, expected_key, expected_value,
            key_start, key_count, cache_cap, n_head, n_head_kv,
            head_dim, scale);
        test_laguna_decode_cpu_reference(
            stale_reference, q_host, gate_host, poisoned_key, poisoned_value,
            key_start, key_count, cache_cap, n_head, n_head_kv,
            head_dim, scale);

        /* Flag off: the ordinary full-ring path (ungrouped Flash vector). */
        TEST_ASSERT(ds4_gpu_tensor_write(
                        heads, 0, heads_seed, heads_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        key_cache, 0, poisoned_key, cache_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        value_cache, 0, poisoned_value, cache_bytes) != 0);
        TEST_ASSERT(setenv(env_name, "0", 1) == 0);
        TEST_ASSERT(test_restart_metal_lifecycle());
        ds4_gpu_test_laguna_route_counters_reset();
        TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                        heads, key_cache, value_cache, q, k, v, gate,
                        pos, cache_cap, key_start, key_count,
                        n_head, n_head_kv, head_dim, scale) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        heads, 0, off_heads, heads_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        key_cache, 0, off_key, cache_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        value_cache, 0, off_value, cache_bytes) != 0);
        {
            uint64_t ordinary = 0;
            uint64_t gqa3 = 0;
            uint64_t gqa9 = 0;
            uint64_t staged = 0;
            uint64_t global_grouped = 0;
            TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                            &ordinary, &gqa3, &gqa9, &staged,
                            &global_grouped) != 0);
            TEST_ASSERT(ordinary == 1u && gqa3 == 0u && gqa9 == 0u &&
                        staged == 0u && global_grouped == 0u);
        }

        /* Flag on: the GQA9 grouped pipeline is mandatory for this leg. */
        TEST_ASSERT(ds4_gpu_tensor_write(
                        heads, 0, heads_seed, heads_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        key_cache, 0, poisoned_key, cache_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        value_cache, 0, poisoned_value, cache_bytes) != 0);
        TEST_ASSERT(setenv(env_name, "1", 1) == 0);
        TEST_ASSERT(test_restart_metal_lifecycle());
        ds4_gpu_test_laguna_route_counters_reset();
        TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                        heads, key_cache, value_cache, q, k, v, gate,
                        pos, cache_cap, key_start, key_count,
                        n_head, n_head_kv, head_dim, scale) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        heads, 0, on_heads, heads_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        key_cache, 0, on_key, cache_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        value_cache, 0, on_value, cache_bytes) != 0);
        {
            uint64_t ordinary = 0;
            uint64_t gqa3 = 0;
            uint64_t gqa9 = 0;
            uint64_t staged = 0;
            uint64_t global_grouped = 0;
            TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                            &ordinary, &gqa3, &gqa9, &staged,
                            &global_grouped) != 0);
            TEST_ASSERT(ordinary == 0u && gqa3 == 0u && gqa9 == 1u &&
                        staged == 0u && global_grouped == 0u);
        }

        const test_float_compare_stats off_cpu =
            test_compare_float_bits(reference, off_heads, head_values);
        const test_float_compare_stats on_cpu =
            test_compare_float_bits(reference, on_heads, head_values);
        const test_float_compare_stats ab =
            test_compare_float_bits(off_heads, on_heads, head_values);
        const test_float_compare_stats stale =
            test_compare_float_bits(reference, stale_reference, head_values);
        double off_rms_sum = 0.0;
        double on_rms_sum = 0.0;
        double ab_rms_sum = 0.0;
        double stale_rms_sum = 0.0;
        size_t nonfinite = 0u;
        for (uint64_t i = 0; i < head_values; i++) {
            const float off_error = fabsf(off_heads[i] - reference[i]);
            const float on_error = fabsf(on_heads[i] - reference[i]);
            const float ab_error = fabsf(off_heads[i] - on_heads[i]);
            const float stale_error =
                fabsf(reference[i] - stale_reference[i]);
            off_rms_sum += (double)off_error * off_error;
            on_rms_sum += (double)on_error * on_error;
            ab_rms_sum += (double)ab_error * ab_error;
            stale_rms_sum += (double)stale_error * stale_error;
            if (!isfinite(off_heads[i]) || !isfinite(on_heads[i])) {
                nonfinite++;
            }
        }
        const float off_rms = sqrtf((float)(off_rms_sum / head_values));
        const float on_rms = sqrtf((float)(on_rms_sum / head_values));
        const float ab_rms = sqrtf((float)(ab_rms_sum / head_values));
        const float stale_rms =
            sqrtf((float)(stale_rms_sum / head_values));
        if (ab.max_abs > max_ab_abs) max_ab_abs = ab.max_abs;
        if (ab_rms > max_ab_rms) max_ab_rms = ab_rms;
        if (ab.max_ulp > max_ab_ulp) max_ab_ulp = ab.max_ulp;
        total_ab_mismatches += ab.mismatch_count;
        fprintf(stderr,
                "ds4-test: Laguna SWA GQA9 A/B pos=%u key_start=%u "
                "cpu_off=%zu/%u/%g/%g cpu_on=%zu/%u/%g/%g "
                "ab=%zu/%u/%g/%g stale=%zu/%u/%g/%g exact=%s "
                "cache=%s/%s nonfinite=%zu\n",
                pos, key_start,
                off_cpu.mismatch_count, off_cpu.max_ulp,
                off_cpu.max_abs, off_rms,
                on_cpu.mismatch_count, on_cpu.max_ulp,
                on_cpu.max_abs, on_rms,
                ab.mismatch_count, ab.max_ulp, ab.max_abs, ab_rms,
                stale.mismatch_count, stale.max_ulp,
                stale.max_abs, stale_rms,
                ab.mismatch_count == 0u ? "yes" : "no",
                memcmp(off_key, expected_key, (size_t)cache_bytes) == 0
                    ? "off-exact" : "off-drift",
                memcmp(on_key, expected_key, (size_t)cache_bytes) == 0
                    ? "on-exact" : "on-drift",
                nonfinite);

        TEST_ASSERT(nonfinite == 0u);
        TEST_ASSERT(off_cpu.max_abs < 5.0e-4f);
        TEST_ASSERT(off_rms < 1.0e-4f);
        TEST_ASSERT(on_cpu.max_abs < 5.0e-4f);
        TEST_ASSERT(on_rms < 1.0e-4f);
        TEST_ASSERT(stale.max_abs > 1.0e-3f);
        TEST_ASSERT(ab.max_abs < 1.0e-6f);
        TEST_ASSERT(ab_rms < 1.0e-7f);
        TEST_ASSERT(memcmp(off_key, expected_key, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(off_value, expected_value, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(on_key, expected_key, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(on_value, expected_value, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(off_key, on_key, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(off_value, on_value, (size_t)cache_bytes) == 0);
    }

    fprintf(stderr,
            "ds4-test: Laguna SWA GQA9 A/B aggregate cases=%zu "
            "mismatches=%zu max_ulp=%u max_abs=%g rms=%g "
            "bound_abs=1e-6 bound_rms=1e-7\n",
            sizeof(positions) / sizeof(positions[0]), total_ab_mismatches,
            max_ab_ulp, max_ab_abs, max_ab_rms);
    TEST_ASSERT(max_ab_abs < 1.0e-6f);
    TEST_ASSERT(max_ab_rms < 1.0e-7f);
    TEST_ASSERT(total_ab_mismatches > 0u);
    TEST_ASSERT(max_ab_abs > 1.0e-12f);

    /* Precedence 1 — GQA9 suppresses GQA3: exporting both (staged off) must
     * be bit-exact with GQA9 alone, proving the lower-precedence GQA3 knob
     * is ignored rather than mixed in. */
    const uint32_t combined_pos =
        positions[sizeof(positions) / sizeof(positions[0]) - 1u];
    const uint32_t combined_key_start = combined_pos + 1u - key_count;
    TEST_ASSERT(unsetenv(staged_env_name) == 0);
    TEST_ASSERT(setenv(env_name, "1", 1) == 0);
    TEST_ASSERT(setenv(gqa3_env_name, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, heads_seed, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, poisoned_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, poisoned_value, cache_bytes) != 0);
    TEST_ASSERT(test_restart_metal_lifecycle());
    ds4_gpu_test_laguna_route_counters_reset();
    TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                    heads, key_cache, value_cache, q, k, v, gate,
                    combined_pos, cache_cap, combined_key_start, key_count,
                    n_head, n_head_kv, head_dim, scale) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, on_heads, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, on_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, on_value, cache_bytes) != 0);

    TEST_ASSERT(setenv(gqa3_env_name, "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, heads_seed, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, poisoned_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, poisoned_value, cache_bytes) != 0);
    TEST_ASSERT(test_restart_metal_lifecycle());
    ds4_gpu_test_laguna_route_counters_reset();
    TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                    heads, key_cache, value_cache, q, k, v, gate,
                    combined_pos, cache_cap, combined_key_start, key_count,
                    n_head, n_head_kv, head_dim, scale) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, off_heads, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, off_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, off_value, cache_bytes) != 0);
    {
        const test_float_compare_stats s =
            test_compare_float_bits(off_heads, on_heads, head_values);
        fprintf(stderr,
                "ds4-test: Laguna SWA GQA9>GQA3 precedence "
                "mismatches=%zu max_ulp=%u max_abs=%g cache=%s/%s\n",
                s.mismatch_count, s.max_ulp, s.max_abs,
                memcmp(off_key, on_key, (size_t)cache_bytes) == 0
                    ? "key-exact" : "key-drift",
                memcmp(off_value, on_value, (size_t)cache_bytes) == 0
                    ? "value-exact" : "value-drift");
        TEST_ASSERT(s.mismatch_count == 0u);
        TEST_ASSERT(memcmp(off_key, on_key, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(off_value, on_value, (size_t)cache_bytes) == 0);
    }
    {
        uint64_t ordinary = 0;
        uint64_t gqa3 = 0;
        uint64_t gqa9 = 0;
        uint64_t staged = 0;
        uint64_t global_grouped = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        TEST_ASSERT(ordinary == 0u && gqa3 == 0u && gqa9 == 1u &&
                    staged == 0u && global_grouped == 0u);
    }

    /* Precedence 2 — staged SWA suppresses GQA9: exporting both must be
     * bit-exact with staged SWA alone (GQA9 off), proving the grouped knob
     * is ignored for the staged reduction.  Keep GQA3 enabled in both
     * winning legs so the staged counter records a real precedence decision
     * rather than relabeling an ordinary single-token decode. */
    TEST_ASSERT(setenv(staged_env_name, "1", 1) == 0);
    TEST_ASSERT(setenv(env_name, "1", 1) == 0);
    TEST_ASSERT(setenv(gqa3_env_name, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, heads_seed, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, poisoned_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, poisoned_value, cache_bytes) != 0);
    TEST_ASSERT(test_restart_metal_lifecycle());
    ds4_gpu_test_laguna_route_counters_reset();
    TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                    heads, key_cache, value_cache, q, k, v, gate,
                    combined_pos, cache_cap, combined_key_start, key_count,
                    n_head, n_head_kv, head_dim, scale) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, on_heads, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, on_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, on_value, cache_bytes) != 0);

    TEST_ASSERT(setenv(env_name, "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, heads_seed, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, poisoned_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, poisoned_value, cache_bytes) != 0);
    TEST_ASSERT(test_restart_metal_lifecycle());
    ds4_gpu_test_laguna_route_counters_reset();
    TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                    heads, key_cache, value_cache, q, k, v, gate,
                    combined_pos, cache_cap, combined_key_start, key_count,
                    n_head, n_head_kv, head_dim, scale) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, off_heads, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, off_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, off_value, cache_bytes) != 0);
    {
        const test_float_compare_stats s =
            test_compare_float_bits(off_heads, on_heads, head_values);
        fprintf(stderr,
                "ds4-test: Laguna SWA staged>GQA9 precedence "
                "mismatches=%zu max_ulp=%u max_abs=%g cache=%s/%s\n",
                s.mismatch_count, s.max_ulp, s.max_abs,
                memcmp(off_key, on_key, (size_t)cache_bytes) == 0
                    ? "key-exact" : "key-drift",
                memcmp(off_value, on_value, (size_t)cache_bytes) == 0
                    ? "value-exact" : "value-drift");
        TEST_ASSERT(s.mismatch_count == 0u);
        TEST_ASSERT(memcmp(off_key, on_key, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(off_value, on_value, (size_t)cache_bytes) == 0);
    }
    {
        uint64_t ordinary = 0;
        uint64_t gqa3 = 0;
        uint64_t gqa9 = 0;
        uint64_t staged = 0;
        uint64_t global_grouped = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        TEST_ASSERT(ordinary == 0u && gqa3 == 0u && gqa9 == 0u &&
                    staged == 1u && global_grouped == 0u);
    }

cleanup:
    test_restore_env(gqa3_env_name, saved_gqa3_env);
    test_restore_env(staged_env_name, saved_staged_env);
    test_restore_env(env_name, saved_env);
    free(stale_reference);
    free(reference);
    free(on_heads);
    free(off_heads);
    free(heads_seed);
    free(gate_host);
    free(v_host);
    free(k_host);
    free(q_host);
    free(poisoned_value);
    free(poisoned_key);
    free(on_value);
    free(on_key);
    free(off_value);
    free(off_key);
    free(expected_value);
    free(expected_key);
    free(initial_value);
    free(initial_key);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_cleanup();
}

/* The GQA9 selector is a lifecycle contract, not a per-dispatch hint.  This
 * test covers strict parsing, restart-only changes, exact shape boundaries,
 * and the old-source failure that previously stored KV before discovering the
 * missing GQA9 PSO.  The latter deliberately leaves an already-open command
 * batch around both end and discard so neither terminal path can commit a
 * feature mutation. */
static void test_metal_laguna_swa_gqa9_preflight_lifecycle(void) {
    const char *gqa9_env = "DS4_METAL_LAGUNA_SWA_GQA9";
    const char *gqa3_env = "DS4_METAL_LAGUNA_SWA_GQA3";
    const char *staged_env = "DS4_METAL_LAGUNA_STAGED_SWA";
    char *saved_gqa9 = test_save_env(gqa9_env);
    char *saved_gqa3 = test_save_env(gqa3_env);
    char *saved_staged = test_save_env(staged_env);
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *key_cache = NULL;
    ds4_gpu_tensor *value_cache = NULL;
    ds4_gpu_tensor *q = NULL;
    ds4_gpu_tensor *k = NULL;
    ds4_gpu_tensor *v = NULL;
    ds4_gpu_tensor *gate = NULL;
    uint16_t *key_before = NULL;
    uint16_t *value_before = NULL;
    uint16_t *key_after = NULL;
    uint16_t *value_after = NULL;
    float *heads_before = NULL;
    float *heads_after = NULL;
    const uint32_t cache_cap = 512u;
    const uint32_t key_count = 512u;
    const uint32_t n_head = 72u;
    const uint32_t n_head_kv = 8u;
    const uint32_t head_dim = 128u;
    const uint32_t cache_width = n_head_kv * head_dim;
    const uint64_t cache_values = (uint64_t)cache_cap * cache_width;
    const uint64_t head_values = (uint64_t)n_head * head_dim;
    const uint64_t cache_bytes = cache_values * sizeof(uint16_t);
    const uint64_t head_bytes = head_values * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)cache_width * sizeof(float);
    const uint64_t gate_bytes = (uint64_t)n_head * sizeof(float);
    const float scale = 1.0f / sqrtf((float)head_dim);

    TEST_ASSERT(unsetenv(gqa3_env) == 0);
    TEST_ASSERT(unsetenv(staged_env) == 0);

    /* Only unset/empty/0 are off; every other value is malformed. */
    TEST_ASSERT(setenv(gqa9_env, "", 1) == 0);
    ds4_gpu_cleanup();
    TEST_ASSERT(ds4_gpu_init() != 0);
    TEST_ASSERT(ds4_gpu_laguna_swa_gqa9_preflight(
                    512u, 512u, 72u, 8u, 128u) == 0);
    TEST_ASSERT(unsetenv(gqa9_env) == 0);
    ds4_gpu_cleanup();
    TEST_ASSERT(ds4_gpu_init() != 0);
    TEST_ASSERT(ds4_gpu_laguna_swa_gqa9_preflight(
                    512u, 512u, 72u, 8u, 128u) == 0);
    TEST_ASSERT(setenv(gqa9_env, "true", 1) == 0);
    ds4_gpu_cleanup();
    TEST_ASSERT(ds4_gpu_init() == 0);
    TEST_ASSERT(setenv(gqa9_env, "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_init() != 0);
    TEST_ASSERT(ds4_gpu_laguna_swa_gqa9_preflight(
                    cache_cap, key_count, n_head, n_head_kv, head_dim) == 0);

    /* A mid-lifecycle edit cannot hot-switch the selector. */
    TEST_ASSERT(setenv(gqa9_env, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_swa_gqa9_preflight(
                    cache_cap, key_count, n_head, n_head_kv, head_dim) == 0);

    /* A clean restart snapshots the opt-in and probes the current/override
     * source.  The old source branch below is the poisoned-cache regression. */
    TEST_ASSERT(test_restart_metal_lifecycle());
    const int enabled = ds4_gpu_laguna_swa_gqa9_preflight(
        cache_cap, key_count, n_head, n_head_kv, head_dim);
    TEST_ASSERT(enabled != 0);
    TEST_ASSERT(ds4_gpu_laguna_swa_gqa9_preflight(
                    cache_cap, 511u, n_head, n_head_kv, head_dim) < 0);
    TEST_ASSERT(ds4_gpu_laguna_swa_gqa9_preflight(
                    cache_cap, 513u, n_head, n_head_kv, head_dim) < 0);
    TEST_ASSERT(ds4_gpu_laguna_swa_gqa9_preflight(
                    513u, key_count, n_head, n_head_kv, head_dim) < 0);
    TEST_ASSERT(ds4_gpu_laguna_swa_gqa9_preflight(
                    cache_cap, key_count, 48u, 8u, head_dim) < 0);
    TEST_ASSERT(ds4_gpu_laguna_swa_gqa9_preflight(
                    cache_cap, key_count, 72u, 6u, head_dim) < 0);

    heads = ds4_gpu_tensor_alloc(head_bytes);
    key_cache = ds4_gpu_tensor_alloc(cache_bytes);
    value_cache = ds4_gpu_tensor_alloc(cache_bytes);
    q = ds4_gpu_tensor_alloc(head_bytes);
    k = ds4_gpu_tensor_alloc(kv_bytes);
    v = ds4_gpu_tensor_alloc(kv_bytes);
    gate = ds4_gpu_tensor_alloc(gate_bytes);
    key_before = malloc((size_t)cache_bytes);
    value_before = malloc((size_t)cache_bytes);
    key_after = malloc((size_t)cache_bytes);
    value_after = malloc((size_t)cache_bytes);
    heads_before = malloc((size_t)head_bytes);
    heads_after = malloc((size_t)head_bytes);
    TEST_ASSERT(heads && key_cache && value_cache && q && k && v && gate &&
                key_before && value_before && key_after && value_after &&
                heads_before && heads_after);
    if (!heads || !key_cache || !value_cache || !q || !k || !v || !gate ||
        !key_before || !value_before || !key_after || !value_after ||
        !heads_before || !heads_after) {
        goto cleanup;
    }
    for (uint64_t i = 0; i < cache_values; i++) {
        key_before[i] = (uint16_t)(0x2400u + (i % 0x300u));
        value_before[i] = (uint16_t)(0x5000u + (i % 0x300u));
    }
    for (uint64_t i = 0; i < head_values; i++) {
        heads_before[i] = 17.0f + (float)(i % 31u) * 0.03125f;
    }
    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, heads_before, head_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, key_before, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, value_before, cache_bytes) != 0);
    float *zeros_q = calloc((size_t)head_values, sizeof(float));
    float *zeros_kv = calloc((size_t)cache_width, sizeof(float));
    float *zeros_gate = calloc((size_t)n_head, sizeof(float));
    TEST_ASSERT(zeros_q && zeros_kv && zeros_gate);
    if (!zeros_q || !zeros_kv || !zeros_gate) {
        free(zeros_gate);
        free(zeros_kv);
        free(zeros_q);
        goto cleanup;
    }
    TEST_ASSERT(ds4_gpu_tensor_write(q, 0, zeros_q, head_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(k, 0, zeros_kv, kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(v, 0, zeros_kv, kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(gate, 0, zeros_gate, gate_bytes) != 0);
    free(zeros_gate);
    free(zeros_kv);
    free(zeros_q);

    ds4_gpu_test_laguna_route_counters_reset();
    if (enabled < 0) {
        TEST_ASSERT(ds4_gpu_begin_commands() != 0);
        TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                        heads, key_cache, value_cache, q, k, v, gate,
                        511u, cache_cap, 0u, key_count,
                        n_head, n_head_kv, head_dim, scale) == 0);
        TEST_ASSERT(ds4_gpu_commands_active());
        TEST_ASSERT(ds4_gpu_end_commands() != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        heads, 0, heads_after, head_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        key_cache, 0, key_after, cache_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        value_cache, 0, value_after, cache_bytes) != 0);
        TEST_ASSERT(memcmp(heads_after, heads_before, (size_t)head_bytes) == 0);
        TEST_ASSERT(memcmp(key_after, key_before, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(value_after, value_before, (size_t)cache_bytes) == 0);
        uint64_t ordinary = 0;
        uint64_t gqa3 = 0;
        uint64_t gqa9 = 0;
        uint64_t staged = 0;
        uint64_t global_grouped = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        TEST_ASSERT(ordinary == 0u && gqa3 == 0u && gqa9 == 0u &&
                    staged == 0u && global_grouped == 0u);

        TEST_ASSERT(ds4_gpu_tensor_write(
                        heads, 0, heads_before, head_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        key_cache, 0, key_before, cache_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        value_cache, 0, value_before, cache_bytes) != 0);
        ds4_gpu_test_laguna_route_counters_reset();
        TEST_ASSERT(ds4_gpu_begin_commands() != 0);
        TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                        heads, key_cache, value_cache, q, k, v, gate,
                        511u, cache_cap, 0u, key_count,
                        n_head, n_head_kv, head_dim, scale) == 0);
        TEST_ASSERT(ds4_gpu_discard_commands() != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        heads, 0, heads_after, head_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        key_cache, 0, key_after, cache_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        value_cache, 0, value_after, cache_bytes) != 0);
        TEST_ASSERT(memcmp(heads_after, heads_before, (size_t)head_bytes) == 0);
        TEST_ASSERT(memcmp(key_after, key_before, (size_t)cache_bytes) == 0);
        TEST_ASSERT(memcmp(value_after, value_before, (size_t)cache_bytes) == 0);
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        TEST_ASSERT(ordinary == 0u && gqa3 == 0u && gqa9 == 0u &&
                    staged == 0u && global_grouped == 0u);
        fprintf(stderr,
                "ds4-test: Laguna GQA9 old-source rollback exact "
                "end/discard cache/head/counters\n");
    } else {
        TEST_ASSERT(enabled > 0);
        ds4_gpu_test_laguna_route_counters_reset();
        TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                        heads, key_cache, value_cache, q, k, v, gate,
                        511u, cache_cap, 0u, key_count,
                        n_head, n_head_kv, head_dim, scale) != 0);
        uint64_t ordinary = 0;
        uint64_t gqa3 = 0;
        uint64_t gqa9 = 0;
        uint64_t staged = 0;
        uint64_t global_grouped = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        TEST_ASSERT(ordinary == 0u && gqa3 == 0u && gqa9 == 1u &&
                    staged == 0u && global_grouped == 0u);
        fprintf(stderr,
                "ds4-test: Laguna GQA9 current-source preflight/decode "
                "available=1 route=gqa9 completion=waited\n");
    }

cleanup:
    if (ds4_gpu_commands_active()) (void)ds4_gpu_discard_commands();
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_cleanup();
    free(heads_after);
    free(heads_before);
    free(value_after);
    free(key_after);
    free(value_before);
    free(key_before);
    test_restore_env(gqa9_env, saved_gqa9);
    test_restore_env(gqa3_env, saved_gqa3);
    test_restore_env(staged_env, saved_staged);
}

/* A 512-slot case can be either just short of a full production SWA ring or
 * a non-production 48/8 layer.  Keep both on the ordinary path when the
 * experiment flag is present. */
static void test_metal_laguna_swa_gqa3_scope_case(
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t pos,
        uint32_t key_count) {
    const char *env_name = "DS4_METAL_LAGUNA_SWA_GQA3";
    const char *gqa9_env_name = "DS4_METAL_LAGUNA_SWA_GQA9";
    const uint32_t head_dim = 128u;
    const uint32_t cache_cap = 512u;
    const uint32_t key_start = pos + 1u >= key_count
        ? pos + 1u - key_count
        : 0u;
    const uint32_t cache_width = n_head_kv * head_dim;
    const uint64_t head_values = (uint64_t)n_head * head_dim;
    const uint64_t cache_values = (uint64_t)cache_cap * cache_width;
    const uint64_t heads_bytes = head_values * sizeof(float);
    const uint64_t cache_bytes = cache_values * sizeof(uint16_t);
    const uint64_t current_kv_bytes =
        (uint64_t)cache_width * sizeof(float);
    const uint64_t gate_bytes = (uint64_t)n_head * sizeof(float);
    const float scale = 1.0f / sqrtf((float)head_dim);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *key_cache = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *value_cache = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(gate_bytes);
    uint16_t *initial_key = malloc((size_t)cache_bytes);
    uint16_t *initial_value = malloc((size_t)cache_bytes);
    uint16_t *off_key = malloc((size_t)cache_bytes);
    uint16_t *off_value = malloc((size_t)cache_bytes);
    uint16_t *on_key = malloc((size_t)cache_bytes);
    uint16_t *on_value = malloc((size_t)cache_bytes);
    float *q_host = malloc((size_t)heads_bytes);
    float *k_host = malloc((size_t)current_kv_bytes);
    float *v_host = malloc((size_t)current_kv_bytes);
    float *gate_host = malloc((size_t)gate_bytes);
    float *seed = malloc((size_t)heads_bytes);
    float *off_heads = malloc((size_t)heads_bytes);
    float *on_heads = malloc((size_t)heads_bytes);
    char *saved_env = test_save_env(env_name);
    char *saved_gqa9_env = test_save_env(gqa9_env_name);

    TEST_ASSERT(heads && key_cache && value_cache && q && k && v && gate &&
                initial_key && initial_value && off_key && off_value &&
                on_key && on_value && q_host && k_host && v_host &&
                gate_host && seed && off_heads && on_heads);
    TEST_ASSERT(unsetenv(gqa9_env_name) == 0);
    if (!heads || !key_cache || !value_cache || !q || !k || !v || !gate ||
        !initial_key || !initial_value || !off_key || !off_value || !on_key ||
        !on_value || !q_host || !k_host || !v_host || !gate_host || !seed ||
        !off_heads || !on_heads) {
        goto cleanup;
    }

    for (uint64_t i = 0; i < cache_values; i++) {
        initial_key[i] = test_float_to_f16(
            (float)((int)((i * 17u + (i >> 3u) * 7u) % 193u) - 96) /
            160.0f);
        initial_value[i] = test_float_to_f16(
            (float)((int)((i * 23u + (i >> 5u) * 11u) % 181u) - 90) /
            144.0f);
    }
    for (uint64_t i = 0; i < head_values; i++) {
        q_host[i] = (float)((int)((i * 31u + 9u) % 211u) - 105) / 176.0f;
        seed[i] = -11.0f - (float)(i % 41u) / 16.0f;
    }
    for (uint64_t i = 0; i < cache_width; i++) {
        k_host[i] = (float)((int)((i * 37u + 3u) % 199u) - 99) / 168.0f;
        v_host[i] = (float)((int)((i * 41u + 17u) % 197u) - 98) / 152.0f;
    }
    for (uint32_t h = 0; h < n_head; h++) {
        gate_host[h] = ((float)((int)(h % 11u) - 5)) * 0.25f;
    }
    TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    k, 0, k_host, current_kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    v, 0, v_host, current_kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(gate, 0, gate_host, gate_bytes) != 0);

    TEST_ASSERT(ds4_gpu_tensor_write(heads, 0, seed, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, initial_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, initial_value, cache_bytes) != 0);
    TEST_ASSERT(setenv(env_name, "0", 1) == 0);
    TEST_ASSERT(test_restart_metal_lifecycle());
    ds4_gpu_test_laguna_route_counters_reset();
    TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                    heads, key_cache, value_cache, q, k, v, gate,
                    pos, cache_cap, key_start, key_count,
                    n_head, n_head_kv, head_dim, scale) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, off_heads, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, off_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, off_value, cache_bytes) != 0);
    {
        uint64_t ordinary = 0;
        uint64_t gqa3 = 0;
        uint64_t gqa9 = 0;
        uint64_t staged = 0;
        uint64_t global_grouped = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        TEST_ASSERT(ordinary == 1u && gqa3 == 0u && gqa9 == 0u &&
                    staged == 0u && global_grouped == 0u);
    }

    TEST_ASSERT(ds4_gpu_tensor_write(heads, 0, seed, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, initial_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, initial_value, cache_bytes) != 0);
    TEST_ASSERT(setenv(env_name, "1", 1) == 0);
    TEST_ASSERT(test_restart_metal_lifecycle());
    ds4_gpu_test_laguna_route_counters_reset();
    TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                    heads, key_cache, value_cache, q, k, v, gate,
                    pos, cache_cap, key_start, key_count,
                    n_head, n_head_kv, head_dim, scale) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, on_heads, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, on_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, on_value, cache_bytes) != 0);
    {
        uint64_t ordinary = 0;
        uint64_t gqa3 = 0;
        uint64_t gqa9 = 0;
        uint64_t staged = 0;
        uint64_t global_grouped = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        TEST_ASSERT(ordinary == 1u && gqa3 == 0u && gqa9 == 0u &&
                    staged == 0u && global_grouped == 0u);
    }

    const test_float_compare_stats stats =
        test_compare_float_bits(off_heads, on_heads, head_values);
    fprintf(stderr,
            "ds4-test: Laguna SWA GQA3 scope GQA%u cap=%u flag-A/B "
            "mismatches=%zu max_ulp=%u max_abs=%g cache=%s/%s\n",
            n_head / n_head_kv, cache_cap, stats.mismatch_count,
            stats.max_ulp, stats.max_abs,
            memcmp(off_key, on_key, (size_t)cache_bytes) == 0
                ? "key-exact" : "key-drift",
            memcmp(off_value, on_value, (size_t)cache_bytes) == 0
                ? "value-exact" : "value-drift");
    TEST_ASSERT(stats.mismatch_count == 0u);
    TEST_ASSERT(memcmp(off_key, on_key, (size_t)cache_bytes) == 0);
    TEST_ASSERT(memcmp(off_value, on_value, (size_t)cache_bytes) == 0);

cleanup:
    test_restore_env(gqa9_env_name, saved_gqa9_env);
    test_restore_env(env_name, saved_env);
    free(on_heads);
    free(off_heads);
    free(seed);
    free(gate_host);
    free(v_host);
    free(k_host);
    free(q_host);
    free(on_value);
    free(on_key);
    free(off_value);
    free(off_key);
    free(initial_value);
    free(initial_key);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
}

static void test_metal_laguna_swa_gqa3_scope(void) {
    /* This is the production shape just before the first complete ring. */
    test_metal_laguna_swa_gqa3_scope_case(72u, 8u, 510u, 511u);
    /* A 48/8 512-slot layer remains on the ordinary path by shape. */
    test_metal_laguna_swa_gqa3_scope_case(48u, 8u, 512u, 512u);
    /* The head ratio is eligible, but the production KV-head count is not. */
    test_metal_laguna_swa_gqa3_scope_case(72u, 6u, 512u, 512u);
}

/* Global (non-SWA) Laguna layers carry a cache larger than the 512-slot ring,
 * so grouped-eligible shapes (the production 72/8 and 48/8 ratios) take the
 * grouped flash decode path by default.  Closing the 513-1023-key gap routes
 * these odd key counts through the same split-K grouped kernel that is
 * already the default at >=1024 keys, with no tail padding (the grouped
 * kernel strides keys and reads exactly key_count rows).  This case checks
 * the default grouped path against a double-precision reference and keeps a
 * stale-cache guard so an attention-before-store regression cannot hide. */
static void test_laguna_global_grouped_decode_numeric_case(
        uint32_t cache_cap,
        uint32_t key_start,
        uint32_t key_count,
        uint32_t n_head,
        uint32_t n_head_kv) {
    const uint32_t head_dim = 128u;
    const uint32_t pos = key_start + key_count - 1u;
    const uint32_t cache_width = n_head_kv * head_dim;
    const uint64_t head_values = (uint64_t)n_head * head_dim;
    const uint64_t cache_values = (uint64_t)cache_cap * cache_width;
    const uint64_t cache_bytes = cache_values * sizeof(uint16_t);
    const uint64_t heads_bytes = head_values * sizeof(float);
    const uint64_t current_kv_bytes = (uint64_t)cache_width * sizeof(float);
    const uint64_t gate_bytes = (uint64_t)n_head * sizeof(float);
    const float scale = 1.0f / sqrtf((float)head_dim);

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *key_cache = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *value_cache = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(gate_bytes);
    uint16_t *poisoned_key = malloc((size_t)cache_bytes);
    uint16_t *poisoned_value = malloc((size_t)cache_bytes);
    uint16_t *expected_key = malloc((size_t)cache_bytes);
    uint16_t *expected_value = malloc((size_t)cache_bytes);
    uint16_t *got_key = malloc((size_t)cache_bytes);
    uint16_t *got_value = malloc((size_t)cache_bytes);
    float *q_host = malloc((size_t)heads_bytes);
    float *k_host = malloc((size_t)current_kv_bytes);
    float *v_host = malloc((size_t)current_kv_bytes);
    float *gate_host = malloc((size_t)gate_bytes);
    float *heads_seed = malloc((size_t)heads_bytes);
    float *actual = malloc((size_t)heads_bytes);
    float *reference = malloc((size_t)heads_bytes);
    float *stale_reference = malloc((size_t)heads_bytes);

    TEST_ASSERT(cache_cap > 512u && key_count > 0u &&
                key_count <= cache_cap &&
                key_start <= cache_cap - key_count);
    TEST_ASSERT(heads && key_cache && value_cache && q && k && v && gate &&
                poisoned_key && poisoned_value && expected_key &&
                expected_value && got_key && got_value && q_host && k_host &&
                v_host && gate_host && heads_seed && actual && reference &&
                stale_reference);
    if (!heads || !key_cache || !value_cache || !q || !k || !v || !gate ||
        !poisoned_key || !poisoned_value || !expected_key || !expected_value ||
        !got_key || !got_value || !q_host || !k_host || !v_host || !gate_host ||
        !heads_seed || !actual || !reference || !stale_reference) {
        goto cleanup;
    }

    for (uint64_t i = 0; i < cache_values; i++) {
        const int key_value =
            (int)((i * 29u + (i >> 4u) * 17u + 13u) % 251u) - 125;
        const int value_value =
            (int)((i * 31u + (i >> 5u) * 11u + 7u) % 239u) - 119;
        poisoned_key[i] = test_float_to_f16((float)key_value / 192.0f);
        poisoned_value[i] = test_float_to_f16((float)value_value / 176.0f);
    }
    for (uint64_t i = 0; i < head_values; i++) {
        const int value =
            (int)((i * 37u + (i >> 3u) * 19u + 5u) % 257u) - 128;
        q_host[i] = (float)value / 192.0f;
        heads_seed[i] = 17.0f + (float)(i % 97u) / 32.0f;
    }
    for (uint64_t i = 0; i < cache_width; i++) {
        const int key_value =
            (int)((i * 41u + (i >> 4u) * 23u + 3u) % 233u) - 116;
        const int value_value =
            (int)((i * 43u + (i >> 3u) * 13u + 17u) % 227u) - 113;
        k_host[i] = (float)key_value / 168.0f;
        v_host[i] = (float)value_value / 156.0f;
    }
    for (uint32_t h = 0; h < n_head; h++) {
        gate_host[h] = ((float)((int)(h % 17u) - 8)) * 0.21875f;
    }

    const uint32_t cache_row = pos % cache_cap;
    const uint64_t row_base = (uint64_t)cache_row * cache_width;
    /* The poisoned row must be materially unlike the current K/V so a
     * stale attention-before-store leaves a visible output error. */
    for (uint32_t i = 0; i < cache_width; i++) {
        poisoned_key[row_base + i] = test_float_to_f16(
            11.0f + (float)(i % 19u) / 16.0f);
        poisoned_value[row_base + i] = test_float_to_f16(
            37.0f + (float)(i % 23u) / 12.0f);
    }
    memcpy(expected_key, poisoned_key, (size_t)cache_bytes);
    memcpy(expected_value, poisoned_value, (size_t)cache_bytes);
    for (uint32_t i = 0; i < cache_width; i++) {
        expected_key[row_base + i] = test_float_to_f16(k_host[i]);
        expected_value[row_base + i] = test_float_to_f16(v_host[i]);
    }

    test_laguna_decode_cpu_reference(
        reference, q_host, gate_host, expected_key, expected_value,
        key_start, key_count, cache_cap, n_head, n_head_kv, head_dim, scale);
    test_laguna_decode_cpu_reference(
        stale_reference, q_host, gate_host, poisoned_key, poisoned_value,
        key_start, key_count, cache_cap, n_head, n_head_kv, head_dim, scale);

    TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(k, 0, k_host, current_kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(v, 0, v_host, current_kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(gate, 0, gate_host, gate_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, heads_seed, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, poisoned_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, poisoned_value, cache_bytes) != 0);
    /* Default path: no experiment flag.  Grouped-eligible global shapes with
     * key_count in [512, cache_cap), plus aligned full-cap histories, take
     * the grouped flash decode kernel.  A nonzero key_start and an unaligned
     * full-cap history remain on the ordinary path. */
    ds4_gpu_test_laguna_route_counters_reset();
    TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                    heads, key_cache, value_cache, q, k, v, gate,
                    pos, cache_cap, key_start, key_count,
                    n_head, n_head_kv, head_dim, scale) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(heads, 0, actual, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(key_cache, 0, got_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(value_cache, 0, got_value, cache_bytes) != 0);
    {
        uint64_t ordinary = 0;
        uint64_t gqa3 = 0;
        uint64_t gqa9 = 0;
        uint64_t staged = 0;
        uint64_t global_grouped = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        const bool grouped =
            key_start == 0u &&
            (n_head % 3u) == 0u &&
            ((n_head / n_head_kv) % 3u) == 0u &&
            ((key_count >= 512u && key_count < cache_cap) ||
             (key_count >= 1024u && key_count % 32u == 0u));
        TEST_ASSERT(ordinary == (grouped ? 0u : 1u) &&
                    gqa3 == 0u && gqa9 == 0u && staged == 0u &&
                    global_grouped == (grouped ? 1u : 0u));
    }

    const test_float_compare_stats cpu =
        test_compare_float_bits(reference, actual, head_values);
    const test_float_compare_stats stale =
        test_compare_float_bits(reference, stale_reference, head_values);
    double rms_sum = 0.0;
    size_t nonfinite = 0u;
    for (uint64_t i = 0; i < head_values; i++) {
        const float error = fabsf(actual[i] - reference[i]);
        rms_sum += (double)error * error;
        if (!isfinite(actual[i])) nonfinite++;
    }
    const float rms = sqrtf((float)(rms_sum / head_values));
    fprintf(stderr,
            "ds4-test: Laguna global grouped decode numeric "
            "key_count=%u pos=%u cpu=%zu/%u/%g rms=%g stale=%zu/%u/%g "
            "cache=%s/%s nonfinite=%zu\n",
            key_count, pos, cpu.mismatch_count, cpu.max_ulp,
            cpu.max_abs, rms, stale.mismatch_count, stale.max_ulp,
            stale.max_abs,
            memcmp(got_key, expected_key, (size_t)cache_bytes) == 0
                ? "key-exact" : "key-drift",
            memcmp(got_value, expected_value, (size_t)cache_bytes) == 0
                ? "value-exact" : "value-drift",
            nonfinite);
    TEST_ASSERT(nonfinite == 0u);
    TEST_ASSERT(cpu.max_abs < 5.0e-4f);
    TEST_ASSERT(rms < 1.0e-4f);
    TEST_ASSERT(stale.max_abs > 1.0e-3f);
    TEST_ASSERT(memcmp(got_key, expected_key, (size_t)cache_bytes) == 0);
    TEST_ASSERT(memcmp(got_value, expected_value, (size_t)cache_bytes) == 0);

cleanup:
    free(stale_reference);
    free(reference);
    free(actual);
    free(heads_seed);
    free(gate_host);
    free(v_host);
    free(k_host);
    free(q_host);
    free(got_value);
    free(got_key);
    free(expected_value);
    free(expected_key);
    free(poisoned_value);
    free(poisoned_key);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
}

static void test_laguna_gqa3_decode_numeric(void) {
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    TEST_ASSERT(ds4_gpu_init() != 0);
#endif
    const uint32_t head_dim = 128;
    const uint32_t n_head = 6;
    const uint32_t n_head_kv = 2;
    const uint32_t cache_cap = 2048;
    const uint32_t key_count = 1024;
    const uint32_t pos = key_count - 1u;
    const uint32_t cache_width = n_head_kv * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const uint64_t heads_bytes =
        (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t kv_bytes =
        (uint64_t)cache_cap * cache_width * sizeof(uint16_t);
    const uint64_t q_bytes = heads_bytes;
    const uint64_t current_kv_bytes =
        (uint64_t)n_head_kv * head_dim * sizeof(float);
    const uint64_t gate_bytes = (uint64_t)n_head * sizeof(float);

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *key_cache = ds4_gpu_tensor_alloc(kv_bytes);
    ds4_gpu_tensor *value_cache = ds4_gpu_tensor_alloc(kv_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(gate_bytes);
    uint16_t *key_host = malloc((size_t)kv_bytes);
    uint16_t *value_host = malloc((size_t)kv_bytes);
    float *actual = malloc((size_t)heads_bytes);
    TEST_ASSERT(heads != NULL);
    TEST_ASSERT(key_cache != NULL);
    TEST_ASSERT(value_cache != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(k != NULL);
    TEST_ASSERT(v != NULL);
    TEST_ASSERT(gate != NULL);
    TEST_ASSERT(key_host != NULL);
    TEST_ASSERT(value_host != NULL);
    TEST_ASSERT(actual != NULL);

    if (heads && key_cache && value_cache && q && k && v && gate &&
        key_host && value_host && actual) {
        float q_host[n_head * head_dim];
        float k_host[n_head_kv * head_dim];
        float v_host[n_head_kv * head_dim];
        float gate_host[n_head];
        for (uint64_t i = 0; i < (uint64_t)cache_cap * cache_width; i++) {
            const int key_value =
                (int)((i * 29u + (i >> 3u) * 17u + 11u) % 193u) - 96;
            const int value_value =
                (int)((i * 31u + (i >> 5u) * 13u + 7u) % 181u) - 90;
            key_host[i] = test_float_to_f16((float)key_value / 160.0f);
            value_host[i] = test_float_to_f16((float)value_value / 144.0f);
        }
        for (uint32_t i = 0; i < n_head * head_dim; i++) {
            const int value =
                (int)((i * 37u + (i >> 2u) * 19u + 5u) % 211u) - 105;
            q_host[i] = (float)value / 176.0f;
        }
        for (uint32_t i = 0; i < n_head_kv * head_dim; i++) {
            const int key_value =
                (int)((i * 41u + (i >> 4u) * 23u + 3u) % 199u) - 99;
            const int value_value =
                (int)((i * 43u + (i >> 3u) * 11u + 17u) % 197u) - 98;
            k_host[i] = (float)key_value / 168.0f;
            v_host[i] = (float)value_value / 152.0f;
        }
        for (uint32_t h = 0; h < n_head; h++) {
            gate_host[h] = ((float)h - 2.5f) * 0.375f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(
                        key_cache, 0, key_host, kv_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        value_cache, 0, value_host, kv_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        k, 0, k_host, current_kv_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        v, 0, v_host, current_kv_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        gate, 0, gate_host, gate_bytes) != 0);
        TEST_ASSERT(ds4_gpu_laguna_store_attention_tensor(
                        heads, key_cache, value_cache, q, k, v, gate,
                        pos, cache_cap, 0, key_count,
                        n_head, n_head_kv, head_dim, scale) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        heads, 0, actual, heads_bytes) != 0);

        const uint64_t current_row = (uint64_t)pos * cache_width;
        for (uint32_t i = 0; i < n_head_kv * head_dim; i++) {
            key_host[current_row + i] = test_float_to_f16(k_host[i]);
            value_host[current_row + i] = test_float_to_f16(v_host[i]);
        }

        double sum_squared = 0.0;
        float max_abs = 0.0f;
        size_t nonfinite = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const uint32_t kv_head = h / (n_head / n_head_kv);
            double max_score = -DBL_MAX;
            for (uint32_t row = 0; row < key_count; row++) {
                const uint64_t kv_base =
                    (uint64_t)row * cache_width +
                    (uint64_t)kv_head * head_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += (double)q_host[h * head_dim + d] *
                        test_f16_to_f32(key_host[kv_base + d]);
                }
                const double score = dot * (double)scale;
                if (score > max_score) max_score = score;
            }

            double denominator = 0.0;
            double numerator[head_dim];
            memset(numerator, 0, sizeof(numerator));
            for (uint32_t row = 0; row < key_count; row++) {
                const uint64_t kv_base =
                    (uint64_t)row * cache_width +
                    (uint64_t)kv_head * head_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += (double)q_host[h * head_dim + d] *
                        test_f16_to_f32(key_host[kv_base + d]);
                }
                const double weight = exp(dot * (double)scale - max_score);
                denominator += weight;
                for (uint32_t d = 0; d < head_dim; d++) {
                    numerator[d] += weight *
                        test_f16_to_f32(value_host[kv_base + d]);
                }
            }
            const double gate_scale = log1p(exp((double)gate_host[h]));
            for (uint32_t d = 0; d < head_dim; d++) {
                const double reference =
                    numerator[d] / denominator * gate_scale;
                const float got = actual[h * head_dim + d];
                if (!isfinite(got)) {
                    nonfinite++;
                    continue;
                }
                const float error = fabsf(got - (float)reference);
                if (error > max_abs) max_abs = error;
                sum_squared += (double)error * error;
            }
        }
        const float rms = sqrtf((float)(sum_squared / (n_head * head_dim)));
        fprintf(stderr,
                "ds4-test: Laguna global GQA3 decode numeric "
                "max_abs=%g rms=%g nonfinite=%zu\n",
                max_abs, rms, nonfinite);
        TEST_ASSERT(nonfinite == 0);
        TEST_ASSERT(max_abs < 5.0e-4f);
        TEST_ASSERT(rms < 1.0e-4f);
    }

    free(actual);
    free(value_host);
    free(key_host);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
    /* The host-side admission floor is >=512: prove the exact 511/512/513
     * boundary against both the route counter and the numeric reference.
     * Keep the production 72/8 global ratio and the production 48/8 (GQA6)
     * ratio covered, then exercise the key-start and full-cap geometry gates.
     */
    test_laguna_global_grouped_decode_numeric_case(2048u, 0u, 511u, 72u, 8u);
    test_laguna_global_grouped_decode_numeric_case(2048u, 0u, 512u, 72u, 8u);
    test_laguna_global_grouped_decode_numeric_case(2048u, 0u, 513u, 72u, 8u);
    test_laguna_global_grouped_decode_numeric_case(2048u, 0u, 600u, 72u, 8u);
    test_laguna_global_grouped_decode_numeric_case(2048u, 0u, 777u, 72u, 8u);
    test_laguna_global_grouped_decode_numeric_case(2048u, 0u, 1023u, 72u, 8u);
    test_laguna_global_grouped_decode_numeric_case(2048u, 0u, 513u, 48u, 8u);
    test_laguna_global_grouped_decode_numeric_case(2048u, 1u, 512u, 72u, 8u);
    test_laguna_global_grouped_decode_numeric_case(1024u, 0u, 1024u, 72u, 8u);
    test_laguna_global_grouped_decode_numeric_case(1025u, 0u, 1025u, 72u, 8u);
    test_laguna_prefill_attention_numeric_case(48u, 8u);
    test_laguna_prefill_attention_numeric_case(72u, 8u);
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    ds4_gpu_cleanup();
#endif
}

#if defined(__APPLE__)
/*
 * DS4_METAL_LAGUNA_DIRECT_KV_PREFILL A/B.  The direct non-SWA path converts
 * the chunk straight into the cache slot and folds the commit away, so heads
 * and the committed ring must stay bit-identical to the staged path, chunk
 * after chunk.  A chunk that would wrap the ring must keep the staged path
 * even with the flag set.
 */
static void test_laguna_prefill_direct_kv_ab_case(
        uint32_t n_head,
        uint32_t n_head_kv) {
    const uint32_t head_dim = 128u;
    const uint32_t cache_cap = 1024u;
    const uint32_t cache_width = n_head_kv * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const uint32_t chunk_pos0[] = {0u, 17u, 990u, 1010u};
    const uint32_t chunk_tokens[] = {17u, 13u, 24u, 24u};
    const uint32_t n_chunks = 4u;
    const uint32_t max_tokens = 24u;
    const uint64_t q_values = (uint64_t)max_tokens * n_head * head_dim;
    const uint64_t kv_values = (uint64_t)max_tokens * cache_width;
    const uint64_t gate_values = (uint64_t)max_tokens * n_head;
    const uint64_t cache_values = (uint64_t)cache_cap * cache_width;

    float *q_host = malloc((size_t)q_values * sizeof(float));
    float *k_host = malloc((size_t)kv_values * sizeof(float));
    float *v_host = malloc((size_t)kv_values * sizeof(float));
    float *gate_host = malloc((size_t)gate_values * sizeof(float));
    float *heads_runs[2] = {NULL, NULL};
    uint16_t *key_runs[2] = {NULL, NULL};
    uint16_t *value_runs[2] = {NULL, NULL};
    const uint64_t all_heads_values =
        (uint64_t)n_chunks * q_values;
    heads_runs[0] = malloc((size_t)all_heads_values * sizeof(float));
    heads_runs[1] = malloc((size_t)all_heads_values * sizeof(float));
    key_runs[0] = malloc((size_t)cache_values * sizeof(uint16_t));
    key_runs[1] = malloc((size_t)cache_values * sizeof(uint16_t));
    value_runs[0] = malloc((size_t)cache_values * sizeof(uint16_t));
    value_runs[1] = malloc((size_t)cache_values * sizeof(uint16_t));
    TEST_ASSERT(q_host && k_host && v_host && gate_host &&
                heads_runs[0] && heads_runs[1] && key_runs[0] &&
                key_runs[1] && value_runs[0] && value_runs[1]);

    char *saved_direct = test_save_env("DS4_METAL_LAGUNA_DIRECT_KV_PREFILL");
    for (uint32_t run = 0; run < 2u; run++) {
        if (run == 1u) {
            setenv("DS4_METAL_LAGUNA_DIRECT_KV_PREFILL", "1", 1);
        } else {
            unsetenv("DS4_METAL_LAGUNA_DIRECT_KV_PREFILL");
        }
        /* The production selector is lifecycle-snapshotted at Metal init;
         * restart that lifecycle for the off/on A/B instead of hot-mutating
         * the environment between dispatches. */
        ds4_gpu_cleanup();
        TEST_ASSERT(ds4_gpu_init() != 0);
        ds4_gpu_test_laguna_route_counters_reset();
        ds4_gpu_tensor *heads =
            ds4_gpu_tensor_alloc(q_values * sizeof(float));
        ds4_gpu_tensor *key_cache =
            ds4_gpu_tensor_alloc(cache_values * sizeof(uint16_t));
        ds4_gpu_tensor *value_cache =
            ds4_gpu_tensor_alloc(cache_values * sizeof(uint16_t));
        ds4_gpu_tensor *staged_key =
            ds4_gpu_tensor_alloc(kv_values * sizeof(uint16_t));
        ds4_gpu_tensor *staged_value =
            ds4_gpu_tensor_alloc(kv_values * sizeof(uint16_t));
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_values * sizeof(float));
        ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(kv_values * sizeof(float));
        ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(kv_values * sizeof(float));
        ds4_gpu_tensor *gate =
            ds4_gpu_tensor_alloc(gate_values * sizeof(float));
        TEST_ASSERT(heads && key_cache && value_cache && staged_key &&
                    staged_value && q && k && v && gate);
        /* Seed the ring so stale-row reads cannot hide behind zeros. */
        for (uint64_t i = 0; i < cache_values; i++) {
            key_runs[run][i] = (uint16_t)(0x5A00u | (i % 251u));
            value_runs[run][i] = (uint16_t)(0xA500u | (i % 241u));
        }
        TEST_ASSERT(ds4_gpu_tensor_write(
                        key_cache, 0, key_runs[run],
                        cache_values * sizeof(uint16_t)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        value_cache, 0, value_runs[run],
                        cache_values * sizeof(uint16_t)) != 0);

        for (uint32_t chunk = 0; chunk < n_chunks; chunk++) {
            const uint32_t n_tokens = chunk_tokens[chunk];
            const uint64_t chunk_q = (uint64_t)n_tokens * n_head * head_dim;
            const uint64_t chunk_kv = (uint64_t)n_tokens * cache_width;
            const uint64_t chunk_gate = (uint64_t)n_tokens * n_head;
            for (uint64_t i = 0; i < chunk_q; i++) {
                const int value = (int)((i * 37u + (i >> 3u) * 11u +
                                         chunk * 29u + 5u) % 191u) - 95;
                q_host[i] = (float)value / 384.0f;
            }
            for (uint64_t i = 0; i < chunk_kv; i++) {
                const int key_value = (int)((i * 29u + (i >> 2u) * 17u +
                                             chunk * 31u + 7u) % 181u) - 90;
                const int value_value = (int)((i * 31u + (i >> 4u) * 13u +
                                               chunk * 23u + 3u) % 173u) - 86;
                k_host[i] = (float)key_value / 352.0f;
                v_host[i] = (float)value_value / 320.0f;
            }
            for (uint64_t i = 0; i < chunk_gate; i++) {
                gate_host[i] =
                    ((float)((int)((i + chunk) % 13u) - 6)) * 0.1875f;
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                            q, 0, q_host, chunk_q * sizeof(float)) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            k, 0, k_host, chunk_kv * sizeof(float)) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            v, 0, v_host, chunk_kv * sizeof(float)) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            gate, 0, gate_host,
                            chunk_gate * sizeof(float)) != 0);
            TEST_ASSERT(ds4_gpu_laguna_attention_prefill_tensor(
                            heads, key_cache, value_cache, staged_key,
                            staged_value, q, k, v, gate,
                            chunk_pos0[chunk], n_tokens, cache_cap,
                            n_head, n_head_kv, head_dim, scale, 0) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            heads, 0,
                            heads_runs[run] + chunk * q_values,
                            chunk_q * sizeof(float)) != 0);
        }
        TEST_ASSERT(ds4_gpu_tensor_read(
                        key_cache, 0, key_runs[run],
                        cache_values * sizeof(uint16_t)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        value_cache, 0, value_runs[run],
                        cache_values * sizeof(uint16_t)) != 0);

        ds4_gpu_tensor_free(gate);
        ds4_gpu_tensor_free(v);
        ds4_gpu_tensor_free(k);
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(staged_value);
        ds4_gpu_tensor_free(staged_key);
        ds4_gpu_tensor_free(value_cache);
        ds4_gpu_tensor_free(key_cache);
        ds4_gpu_tensor_free(heads);
    }
    test_restore_env("DS4_METAL_LAGUNA_DIRECT_KV_PREFILL", saved_direct);

    /* The wrapping tail chunk keeps the staged path under the flag, so all
     * four chunks and the whole ring must match bit for bit. */
    for (uint32_t chunk = 0; chunk < n_chunks; chunk++) {
        const size_t chunk_heads = (size_t)chunk_tokens[chunk] *
            n_head * head_dim;
        TEST_ASSERT(memcmp(heads_runs[0] + chunk * q_values,
                           heads_runs[1] + chunk * q_values,
                           chunk_heads * sizeof(float)) == 0);
    }
    TEST_ASSERT(memcmp(key_runs[0], key_runs[1],
                       (size_t)cache_values * sizeof(uint16_t)) == 0);
    TEST_ASSERT(memcmp(value_runs[0], value_runs[1],
                       (size_t)cache_values * sizeof(uint16_t)) == 0);
    {
        uint64_t direct_count = 0;
        uint64_t wrap_count = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_route_counters(
                        &direct_count, &wrap_count, NULL, NULL) != 0);
        TEST_ASSERT(direct_count == 3u);
        TEST_ASSERT(wrap_count == 1u);
        fprintf(stderr,
                "ds4-test: Laguna KV route counters direct=%llu wrap=%llu\n",
                (unsigned long long)direct_count,
                (unsigned long long)wrap_count);
    }
    fprintf(stderr,
            "ds4-test: Laguna direct KV prefill A/B bit-exact "
            "(heads=%u kv=%u)\n",
            n_head, n_head_kv);

    free(value_runs[1]);
    free(value_runs[0]);
    free(key_runs[1]);
    free(key_runs[0]);
    free(heads_runs[1]);
    free(heads_runs[0]);
    free(gate_host);
    free(v_host);
    free(k_host);
    free(q_host);
}

static void test_laguna_prefill_direct_kv_ab(void) {
    TEST_ASSERT(ds4_gpu_init() != 0);
    /* Plain GQA (ratio 4) and grouped GQA3 (ratio 3) staged-slot views. */
    test_laguna_prefill_direct_kv_ab_case(8u, 2u);
    test_laguna_prefill_direct_kv_ab_case(9u, 3u);
    /* Production global layers use six query heads per KV head. */
    test_laguna_prefill_direct_kv_ab_case(12u, 2u);
    ds4_gpu_cleanup();
}

/* A malformed direct-KV opt-in is rejected while taking the Metal lifecycle
 * snapshot.  The second half forces the low-level defensive guard and uses
 * poisoned cache rows to prove the split path cannot mutate KV before it
 * returns failure. */
static void test_laguna_prefill_direct_kv_invalid_mode(void) {
    const uint32_t n_head = 12u;
    const uint32_t n_head_kv = 2u;
    const uint32_t head_dim = 128u;
    const uint32_t cache_cap = 1024u;
    const uint32_t n_tokens = 4u;
    const uint32_t pos0 = 8u;
    const uint64_t q_values = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_values = (uint64_t)n_tokens * n_head_kv * head_dim;
    const uint64_t gate_values = (uint64_t)n_tokens * n_head;
    const uint64_t cache_values = (uint64_t)cache_cap * n_head_kv * head_dim;
    char *saved_direct = test_save_env("DS4_METAL_LAGUNA_DIRECT_KV_PREFILL");

    TEST_ASSERT(setenv("DS4_METAL_LAGUNA_DIRECT_KV_PREFILL", "01", 1) == 0);
    ds4_gpu_cleanup();
    /* Snapshot rejection happens before device, queue, or graph setup. */
    TEST_ASSERT(ds4_gpu_init() == 0);

    TEST_ASSERT(setenv("DS4_METAL_LAGUNA_DIRECT_KV_PREFILL", "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_init() != 0);
    ds4_gpu_test_laguna_route_counters_reset();
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_values * sizeof(float));
    ds4_gpu_tensor *key_cache =
        ds4_gpu_tensor_alloc(cache_values * sizeof(uint16_t));
    ds4_gpu_tensor *value_cache =
        ds4_gpu_tensor_alloc(cache_values * sizeof(uint16_t));
    ds4_gpu_tensor *staged_key =
        ds4_gpu_tensor_alloc(kv_values * sizeof(uint16_t));
    ds4_gpu_tensor *staged_value =
        ds4_gpu_tensor_alloc(kv_values * sizeof(uint16_t));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_values * sizeof(float));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(kv_values * sizeof(float));
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(kv_values * sizeof(float));
    ds4_gpu_tensor *gate =
        ds4_gpu_tensor_alloc(gate_values * sizeof(float));
    uint16_t *key_poison = malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *value_poison = malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *key_after = malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *value_after = malloc((size_t)cache_values * sizeof(uint16_t));
    TEST_ASSERT(heads && key_cache && value_cache && staged_key &&
                staged_value && q && k && v && gate && key_poison &&
                value_poison && key_after && value_after);
    if (!heads || !key_cache || !value_cache || !staged_key ||
        !staged_value || !q || !k || !v || !gate || !key_poison ||
        !value_poison || !key_after || !value_after) {
        goto invalid_cleanup;
    }
    for (uint64_t i = 0; i < cache_values; i++) {
        key_poison[i] = (uint16_t)(0x5100u | (i % 251u));
        value_poison[i] = (uint16_t)(0xA100u | (i % 241u));
    }
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, key_poison,
                    cache_values * sizeof(uint16_t)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, value_poison,
                    cache_values * sizeof(uint16_t)) != 0);

    /* This setter is test-only: it simulates a malformed captured mode after
     * a valid device lifecycle so the split branch itself is covered. */
    ds4_gpu_test_laguna_set_direct_kv_mode(-1);
    TEST_ASSERT(ds4_gpu_laguna_attention_prefill_tensor(
                    heads, key_cache, value_cache, staged_key, staged_value,
                    q, k, v, gate, pos0, n_tokens, cache_cap,
                    n_head, n_head_kv, head_dim,
                    1.0f / sqrtf((float)head_dim), 1) == 0);
    ds4_gpu_test_laguna_set_direct_kv_mode(0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, key_after,
                    cache_values * sizeof(uint16_t)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, value_after,
                    cache_values * sizeof(uint16_t)) != 0);
    TEST_ASSERT(memcmp(key_after, key_poison,
                       (size_t)cache_values * sizeof(uint16_t)) == 0);
    TEST_ASSERT(memcmp(value_after, value_poison,
                       (size_t)cache_values * sizeof(uint16_t)) == 0);
    {
        uint64_t direct_count = 0;
        uint64_t wrap_count = 0;
        uint64_t fused_count = 0;
        uint64_t stock_count = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_route_counters(
                        &direct_count, &wrap_count,
                        &fused_count, &stock_count) != 0);
        TEST_ASSERT(direct_count == 0u && wrap_count == 0u &&
                    fused_count == 0u && stock_count == 0u);
    }
    fprintf(stderr,
            "ds4-test: malformed Laguna direct-KV mode rejected before "
            "split dispatch; poisoned caches unchanged\n");

invalid_cleanup:
    free(value_after);
    free(key_after);
    free(value_poison);
    free(key_poison);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(staged_value);
    ds4_gpu_tensor_free(staged_key);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_cleanup();
    test_restore_env("DS4_METAL_LAGUNA_DIRECT_KV_PREFILL", saved_direct);
}

/*
 * Batched fused dense Q8 gate/up+SwiGLU against a double reference.  The
 * kernel stages activations as half and dequantizes Q8_0 to half exactly
 * like kernel_mul_mm_q8_0_f32, so the reference rounds both operands the
 * same way and accumulates in double; the kernel accumulates in float.
 */
static void test_laguna_dense_q8_batch_swiglu_case(uint32_t n_tok,
                                                   uint32_t out_dim) {
    const uint32_t in_dim = 512u;
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t gate_offset = 0u;
    const uint64_t up_offset = test_round_up_u64(weight_bytes, page);
    const uint64_t model_size = test_round_up_u64(up_offset + weight_bytes,
                                                  page);
    const uint64_t x_values = (uint64_t)n_tok * in_dim;
    const uint64_t mid_values = (uint64_t)n_tok * out_dim;

    void *model = NULL;
    TEST_ASSERT(posix_memalign(&model, (size_t)page,
                               (size_t)model_size) == 0);
    if (!model) return;
    memset(model, 0, (size_t)model_size);
    test_fill_q8_0_weights((uint8_t *)model + gate_offset, in_dim, out_dim,
                           3u);
    test_fill_q8_0_weights((uint8_t *)model + up_offset, in_dim, out_dim,
                           17u);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_values * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_values * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(mid_values * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mid_values * sizeof(float));
    ds4_gpu_tensor *mid_stock =
        ds4_gpu_tensor_alloc(mid_values * sizeof(float));
    float *x_host = malloc((size_t)x_values * sizeof(float));
    float *mid_host = malloc((size_t)mid_values * sizeof(float));
    float *mid_stock_host = malloc((size_t)mid_values * sizeof(float));
    TEST_ASSERT(x && mid && gate && up && mid_stock && x_host && mid_host &&
                mid_stock_host);
    if (!x || !mid || !gate || !up || !mid_stock || !x_host || !mid_host ||
        !mid_stock_host) {
        goto cleanup;
    }

    for (uint64_t i = 0; i < x_values; i++) {
        const int v = (int)((i * 13u + (i >> 4u) * 7u + 11u) % 61u) - 30;
        x_host[i] = (float)v / 128.0f;
    }
    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host,
                                     x_values * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model, model_size) != 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_batch_tensor(
                    mid, model, model_size, gate_offset, up_offset,
                    (uint64_t)INT32_MAX + 1u, out_dim, x, n_tok) == 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_batch_tensor(
                    mid, model, model_size, gate_offset, up_offset,
                    in_dim, (uint64_t)INT32_MAX + 1u, x, n_tok) == 0);
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_batch_tensor(
                    mid, model, model_size, gate_offset, up_offset,
                    in_dim, out_dim, x,
                    (uint64_t)INT32_MAX + 1u) == 0);
    const uint32_t mv_ext_ceiling = ds4_gpu_laguna_q8_mv_ext_max_tokens();
    const int expected_route =
        ds4_gpu_laguna_dense_q8_gate_up_swiglu_prefill_route(
            1, 0, n_tok, mv_ext_ceiling);
    TEST_ASSERT(expected_route ==
                (n_tok <= mv_ext_ceiling
                     ? DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_ORDINARY_PREFILL_STOCK
                     : DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_BATCH_FUSED));
    ds4_gpu_test_laguna_route_counters_reset();

    /* The caller pins quality mode.  For rows at or below the lifecycle
     * ceiling this deliberately exercises stock mul_mv_ext and never calls
     * the fused sibling; only rows strictly above the ceiling may compare
     * the fused arithmetic against the stock/reference result. */
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                    gate, model, model_size, gate_offset,
                    in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                    up, model, model_size, up_offset,
                    in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_swiglu_tensor(
                    mid_stock, gate, up,
                    (uint32_t)mid_values, 0.0f, 1.0f) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(mid_stock, 0, mid_stock_host,
                                    mid_values * sizeof(float)) != 0);
    if (expected_route == DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_BATCH_FUSED) {
        TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_batch_tensor(
                        mid, model, model_size, gate_offset, up_offset,
                        in_dim, out_dim, x, n_tok) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(mid, 0, mid_host,
                                        mid_values * sizeof(float)) != 0);
    } else {
        /* Keep the reference check below meaningful for stock-only boundary
         * rows without dispatching an ineligible fused kernel. */
        memcpy(mid_host, mid_stock_host,
               (size_t)mid_values * sizeof(float));
    }
    {
        uint64_t fused_count = 0;
        uint64_t stock_count = 0;
        uint64_t bco_false_count = 0;
        uint64_t bco_true_count = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_route_counters(
                        NULL, NULL, &fused_count, &stock_count) != 0);
        TEST_ASSERT(ds4_gpu_test_laguna_q8_bco_counters(
                        &bco_false_count, &bco_true_count) != 0);
        TEST_ASSERT(fused_count ==
                    (expected_route == DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_BATCH_FUSED
                         ? 1u : 0u));
        TEST_ASSERT(stock_count == 2u);
        const bool expected_bco = out_dim % 64u != 0u || n_tok % 32u != 0u;
        TEST_ASSERT(bco_false_count ==
                    (expected_route == DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_BATCH_FUSED &&
                     !expected_bco ? 1u : 0u));
        TEST_ASSERT(bco_true_count ==
                    (expected_route == DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_BATCH_FUSED &&
                     expected_bco ? 1u : 0u));
        fprintf(stderr,
                "ds4-test: Laguna Q8 route counters rows=%u out=%u fused=%llu "
                "stock=%llu bco_false=%llu bco_true=%llu\n",
                n_tok, out_dim,
                (unsigned long long)fused_count,
                (unsigned long long)stock_count,
                (unsigned long long)bco_false_count,
                (unsigned long long)bco_true_count);
    }
    TEST_ASSERT(memcmp(mid_host, mid_stock_host,
                       (size_t)mid_values * sizeof(float)) == 0);

    {
        const uint8_t *gate_w = (const uint8_t *)model + gate_offset;
        const uint8_t *up_w = (const uint8_t *)model + up_offset;
        double sum_squared = 0.0;
        double max_abs = 0.0;
        size_t nonfinite = 0u;
        for (uint32_t t = 0; t < n_tok; t++) {
            for (uint32_t o = 0; o < out_dim; o++) {
                double gate = 0.0;
                double up = 0.0;
                for (uint32_t b = 0; b < blocks; b++) {
                    uint16_t gate_bits;
                    uint16_t up_bits;
                    memcpy(&gate_bits,
                           gate_w + (uint64_t)o * row_bytes + b * 34u,
                           sizeof(gate_bits));
                    memcpy(&up_bits,
                           up_w + (uint64_t)o * row_bytes + b * 34u,
                           sizeof(up_bits));
                    const float gate_d = test_f16_to_f32(gate_bits);
                    const float up_d = test_f16_to_f32(up_bits);
                    const int8_t *gate_qs = (const int8_t *)(
                        gate_w + (uint64_t)o * row_bytes + b * 34u + 2u);
                    const int8_t *up_qs = (const int8_t *)(
                        up_w + (uint64_t)o * row_bytes + b * 34u + 2u);
                    for (uint32_t i = 0; i < 32u; i++) {
                        const double xv = (double)test_f16_to_f32(
                            test_float_to_f16(
                                x_host[(uint64_t)t * in_dim + b * 32u + i]));
                        const double gw = (double)test_f16_to_f32(
                            test_float_to_f16((float)gate_qs[i] * gate_d));
                        const double uw = (double)test_f16_to_f32(
                            test_float_to_f16((float)up_qs[i] * up_d));
                        gate += xv * gw;
                        up += xv * uw;
                    }
                }
                const double reference =
                    gate / (1.0 + exp(-gate)) * up;
                const float got = mid_host[(uint64_t)t * out_dim + o];
                if (!isfinite(got)) {
                    nonfinite++;
                    continue;
                }
                const double error = fabs((double)got - reference);
                if (error > max_abs) max_abs = error;
                sum_squared += error * error;
            }
        }
        const double rms = sqrt(sum_squared / (double)mid_values);
        fprintf(stderr,
                "ds4-test: Laguna batched fused Q8 gate/up+SwiGLU "
                "numeric rows=%u out=%u max_abs=%g rms=%g nonfinite=%zu\n",
                n_tok, out_dim, max_abs, rms, nonfinite);
        TEST_ASSERT(nonfinite == 0u);
        TEST_ASSERT(max_abs < 5.0e-3);
        TEST_ASSERT(rms < 1.0e-3);
    }

cleanup:
    free(mid_stock_host);
    free(mid_host);
    free(x_host);
    ds4_gpu_tensor_free(mid_stock);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(x);
    free(model);
}

static void test_laguna_dense_q8_batch_swiglu(void) {
    char *saved_mv_ceiling =
        test_save_env("DS4_METAL_Q8_MV_EXT_MAX_TOKENS");
    const char *dense_source = getenv("DS4_METAL_DENSE_SOURCE");
    const bool source_override = dense_source && dense_source[0];
    TEST_ASSERT(setenv("DS4_METAL_Q8_MV_EXT_MAX_TOKENS", "128", 1) == 0);
    /* The effective ceiling is captured at init, not reparsed per dispatch. */
    ds4_gpu_cleanup();
    TEST_ASSERT(ds4_gpu_init() != 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_mv_ext_max_tokens() == 128u);
    /* Enabled preflight is intentionally first: an old dense source must
     * fail here, before any stock, attention, or KV work is dispatched. */
    const int first_batch_available =
        ds4_gpu_laguna_dense_q8_gate_up_swiglu_batch_available();
    if (first_batch_available == 0) {
        /* A source override is allowed to predate this optional PSO.  The
         * default in-repo source must provide it, so only an explicit old
         * source gets a clean skip here; the selector-on graph preflight is
         * covered by the focused failure tests. */
        TEST_ASSERT(source_override);
        ds4_gpu_cleanup();
        test_restore_env("DS4_METAL_Q8_MV_EXT_MAX_TOKENS", saved_mv_ceiling);
        return;
    }
    /* A lifecycle override changes the stock boundary, so exercise both
     * sides of the effective 128-row ceiling before changing the env. */
    ds4_gpu_set_quality(true);
    test_laguna_dense_q8_batch_swiglu_case(128u, 160u);
    test_laguna_dense_q8_batch_swiglu_case(129u, 160u);
    /* Changing the environment cannot hot-switch the already initialized
     * lifecycle; only a clean init may make 16 effective. */
    TEST_ASSERT(setenv("DS4_METAL_Q8_MV_EXT_MAX_TOKENS", "16", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_mv_ext_max_tokens() == 128u);
    ds4_gpu_cleanup();
    TEST_ASSERT(ds4_gpu_init() != 0);
    TEST_ASSERT(ds4_gpu_laguna_q8_mv_ext_max_tokens() == 16u);
    /* Aligned bounded production shape: both dimensions select bco=false. */
    TEST_ASSERT(ds4_gpu_laguna_dense_q8_gate_up_swiglu_batch_available() != 0);
    test_laguna_dense_q8_batch_swiglu_case(32u, 192u);
    /* Boundary rows stay on stock mul_mv_ext; larger rows compare the fused
     * pass bit-for-bit with stock and with the independent reference. */
    test_laguna_dense_q8_batch_swiglu_case(1u, 160u);
    test_laguna_dense_q8_batch_swiglu_case(2u, 160u);
    test_laguna_dense_q8_batch_swiglu_case(16u, 160u);
    test_laguna_dense_q8_batch_swiglu_case(17u, 160u);
    test_laguna_dense_q8_batch_swiglu_case(64u, 160u);
    test_laguna_dense_q8_batch_swiglu_case(48u, 160u);
    test_laguna_dense_q8_batch_swiglu_case(33u, 160u);
    test_laguna_dense_q8_batch_swiglu_case(129u, 160u);
    ds4_gpu_set_quality(false);
    ds4_gpu_cleanup();
    test_restore_env("DS4_METAL_Q8_MV_EXT_MAX_TOKENS", saved_mv_ceiling);
}
#endif /* __APPLE__ */
#endif

#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} test_cuda_block_q4_K;

typedef struct {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
} test_cuda_block_q3_K;

static uint64_t test_align_u64(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static void test_fill_cuda_laguna_q4(
        test_cuda_block_q4_K *blocks,
        uint32_t n_expert,
        uint32_t n_rows,
        uint8_t q_base) {
    for (uint32_t expert = 0; expert < n_expert; expert++) {
        for (uint32_t row = 0; row < n_rows; row++) {
            test_cuda_block_q4_K *block =
                blocks + (uint64_t)expert * n_rows + row;
            memset(block, 0, sizeof(*block));
            block->d = (uint16_t)((7u + (expert & 1u)) << 10u);
            block->scales[0] = 1u;
            block->scales[1] = 1u;
            block->scales[2] = 1u;
            block->scales[3] = 1u;
            block->scales[8] = 1u;
            block->scales[9] = 1u;
            block->scales[10] = 1u;
            block->scales[11] = 1u;
            const uint8_t q = (uint8_t)(
                (q_base + expert) < 16u ? q_base + expert : 15u);
            memset(block->qs, (int)(q | (q << 4u)), sizeof(block->qs));
        }
    }
}

static void test_fill_cuda_laguna_q3(
        test_cuda_block_q3_K *blocks,
        uint32_t n_expert,
        uint32_t n_rows,
        uint8_t q_base) {
    for (uint32_t expert = 0; expert < n_expert; expert++) {
        for (uint32_t row = 0; row < n_rows; row++) {
            test_cuda_block_q3_K *block =
                blocks + (uint64_t)expert * n_rows + row;
            memset(block, 0, sizeof(*block));
            block->d = (uint16_t)((7u + (expert & 1u)) << 10u);
            memset(block->hmask, 0xff, sizeof(block->hmask));
            memset(block->scales, 0x11, 8u);
            memset(block->scales + 8u, 0xaa, 4u);
            const uint8_t q = (uint8_t)((q_base + expert) & 3u);
            memset(block->qs,
                   (int)(q | (q << 2u) | (q << 4u) | (q << 6u)),
                   sizeof(block->qs));
        }
    }
}

static void test_cuda_laguna_moe_decode_prefill_format(uint32_t type) {
    const uint32_t n_tokens = 32u;
    const uint32_t n_total_expert = 4u;
    const uint32_t n_expert = 2u;
    const uint32_t in_dim = 256u;
    const uint32_t mid_dim = 256u;
    const uint32_t out_dim = 256u;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t block_bytes =
        type == 12u ? sizeof(test_cuda_block_q4_K) :
                      sizeof(test_cuda_block_q3_K);
    const uint64_t row_bytes = block_bytes;
    const uint64_t expert_bytes = (uint64_t)mid_dim * row_bytes;
    const uint64_t tensor_bytes =
        (uint64_t)n_total_expert * expert_bytes;
    const uint64_t gate_offset = 0u;
    const uint64_t up_offset =
        test_align_u64(gate_offset + tensor_bytes, page);
    const uint64_t down_offset =
        test_align_u64(up_offset + tensor_bytes, page);
    const uint64_t model_size =
        test_align_u64(down_offset + tensor_bytes, page);
    void *model = NULL;
    TEST_ASSERT(posix_memalign(
                    &model, (size_t)page, (size_t)model_size) == 0);
    if (!model) return;
    memset(model, 0, (size_t)model_size);
    if (type == 12u) {
        test_fill_cuda_laguna_q4(
            (test_cuda_block_q4_K *)((uint8_t *)model + gate_offset),
            n_total_expert, mid_dim, 1u);
        test_fill_cuda_laguna_q4(
            (test_cuda_block_q4_K *)((uint8_t *)model + up_offset),
            n_total_expert, mid_dim, 2u);
        test_fill_cuda_laguna_q4(
            (test_cuda_block_q4_K *)((uint8_t *)model + down_offset),
            n_total_expert, out_dim, 3u);
    } else {
        test_fill_cuda_laguna_q3(
            (test_cuda_block_q3_K *)((uint8_t *)model + gate_offset),
            n_total_expert, mid_dim, 1u);
        test_fill_cuda_laguna_q3(
            (test_cuda_block_q3_K *)((uint8_t *)model + up_offset),
            n_total_expert, mid_dim, 2u);
        test_fill_cuda_laguna_q3(
            (test_cuda_block_q3_K *)((uint8_t *)model + down_offset),
            n_total_expert, out_dim, 3u);
    }
    TEST_ASSERT(ds4_gpu_set_model_map(model, model_size) != 0);

    const uint64_t x_bytes =
        (uint64_t)n_tokens * in_dim * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)n_tokens * n_expert * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)n_tokens * n_expert * sizeof(float);
    const uint64_t mid_bytes =
        (uint64_t)n_tokens * n_expert * mid_dim * sizeof(float);
    const uint64_t out_bytes =
        (uint64_t)n_tokens * out_dim * sizeof(float);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *x_one =
        ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *selected_one =
        ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(int32_t));
    ds4_gpu_tensor *weights_one =
        ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(float));
    ds4_gpu_tensor *mid_one =
        ds4_gpu_tensor_alloc(
            (uint64_t)n_expert * mid_dim * sizeof(float));
    ds4_gpu_tensor *out_one =
        ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    float *x_host = malloc((size_t)x_bytes);
    int32_t *selected_host = malloc((size_t)selected_bytes);
    float *weights_host = malloc((size_t)weights_bytes);
    float *batch_host = malloc((size_t)out_bytes);
    float *decode_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x && selected && weights && mid && out &&
                x_one && selected_one && weights_one && mid_one && out_one &&
                x_host && selected_host && weights_host &&
                batch_host && decode_host);
    if (!x || !selected || !weights || !mid || !out ||
        !x_one || !selected_one || !weights_one || !mid_one || !out_one ||
        !x_host || !selected_host || !weights_host ||
        !batch_host || !decode_host) {
        goto cleanup;
    }
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t col = 0; col < in_dim; col++) {
            x_host[(uint64_t)token * in_dim + col] =
                0.0625f +
                (float)((token * 17u + col * 13u) % 29u) / 256.0f;
        }
        selected_host[(uint64_t)token * n_expert] =
            (int32_t)(token & 3u);
        selected_host[(uint64_t)token * n_expert + 1u] =
            (int32_t)((token + 2u) & 3u);
        weights_host[(uint64_t)token * n_expert] = 0.625f;
        weights_host[(uint64_t)token * n_expert + 1u] = 0.375f;
    }
    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    selected, 0, selected_host, selected_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    weights, 0, weights_host, weights_bytes) != 0);
    TEST_ASSERT(ds4_gpu_glm_routed_moe_batch_tensor(
                    out, mid, model, model_size,
                    gate_offset, up_offset, down_offset,
                    type, type, type,
                    expert_bytes, row_bytes,
                    expert_bytes, row_bytes,
                    expert_bytes, row_bytes,
                    in_dim, mid_dim, out_dim,
                    selected, weights,
                    n_total_expert, n_expert, 0u, x, n_tokens,
                    n_expert * mid_dim, true) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    out, 0, batch_host, out_bytes) != 0);

    for (uint32_t token = 0; token < n_tokens; token++) {
        TEST_ASSERT(ds4_gpu_tensor_write(
                        x_one, 0,
                        x_host + (uint64_t)token * in_dim,
                        (uint64_t)in_dim * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        selected_one, 0,
                        selected_host + (uint64_t)token * n_expert,
                        (uint64_t)n_expert * sizeof(int32_t)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        weights_one, 0,
                        weights_host + (uint64_t)token * n_expert,
                        (uint64_t)n_expert * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_glm_routed_moe_batch_tensor(
                        out_one, mid_one, model, model_size,
                        gate_offset, up_offset, down_offset,
                        type, type, type,
                        expert_bytes, row_bytes,
                        expert_bytes, row_bytes,
                        expert_bytes, row_bytes,
                        in_dim, mid_dim, out_dim,
                        selected_one, weights_one,
                        n_total_expert, n_expert, 0u, x_one, 1u,
                        n_expert * mid_dim, true) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        out_one, 0,
                        decode_host + (uint64_t)token * out_dim,
                        (uint64_t)out_dim * sizeof(float)) != 0);
    }

    {
        float max_abs = 0.0f;
        float max_rel = 0.0f;
        size_t nonfinite = 0u;
        for (uint64_t i = 0; i < (uint64_t)n_tokens * out_dim; i++) {
            if (!isfinite(batch_host[i]) || !isfinite(decode_host[i])) {
                nonfinite++;
                continue;
            }
            const float error = fabsf(batch_host[i] - decode_host[i]);
            const float denom = fmaxf(fabsf(decode_host[i]), 1.0e-6f);
            if (error > max_abs) max_abs = error;
            if (error / denom > max_rel) max_rel = error / denom;
        }
        fprintf(stderr,
                "ds4-test: CUDA Laguna Q%u decode/prefill "
                "max_abs=%g max_rel=%g nonfinite=%zu\n",
                type == 12u ? 4u : 3u, max_abs, max_rel, nonfinite);
        TEST_ASSERT(nonfinite == 0u);
        TEST_ASSERT(max_abs < 2.0e-4f);
        TEST_ASSERT(max_rel < (type == 11u ? 1.0e-3f : 2.0e-4f));
    }

cleanup:
    free(decode_host);
    free(batch_host);
    free(weights_host);
    free(selected_host);
    free(x_host);
    ds4_gpu_tensor_free(out_one);
    ds4_gpu_tensor_free(mid_one);
    ds4_gpu_tensor_free(weights_one);
    ds4_gpu_tensor_free(selected_one);
    ds4_gpu_tensor_free(x_one);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(x);
    free(model);
}

static void test_cuda_laguna_moe_decode_prefill(void) {
    const char *tc_min_env = "DS4_CUDA_MOE_TC_MIN_TOKENS";
    TEST_ASSERT(sizeof(test_cuda_block_q4_K) == 144u);
    TEST_ASSERT(sizeof(test_cuda_block_q3_K) == 110u);
    TEST_ASSERT(ds4_gpu_init() != 0);
    test_metal_q8_0_prefill_matmul();
    test_cuda_laguna_moe_decode_prefill_format(12u);
    test_cuda_laguna_moe_decode_prefill_format(11u);
    char *saved_tc_min = test_save_env(tc_min_env);
    TEST_ASSERT(setenv(tc_min_env, "2", 1) == 0);
    test_cuda_laguna_moe_decode_prefill_format(11u);
    test_restore_env(tc_min_env, saved_tc_min);
    ds4_gpu_cleanup();
}
#endif

#if defined(__APPLE__)

static void test_fill_metal_glm_qmv_weights(
        uint8_t *weights,
        uint32_t type,
        uint32_t n_expert,
        uint32_t n_rows,
        uint32_t blocks,
        uint32_t seed) {
    const uint32_t row_bytes =
        type == 10u ? 84u * blocks :
        type == 11u ? 110u * blocks : 144u * blocks;
    for (uint32_t expert = 0; expert < n_expert; expert++) {
        for (uint32_t row = 0; row < n_rows; row++) {
            uint8_t *dst = weights +
                ((uint64_t)expert * n_rows + row) * row_bytes;
            for (uint32_t block = 0; block < blocks; block++) {
                uint8_t *b = dst + (uint64_t)block * (row_bytes / blocks);
                const uint32_t block_bytes = row_bytes / blocks;
                for (uint32_t i = 0; i < block_bytes; i++) {
                    b[i] = (uint8_t)(seed * 17u + expert * 29u +
                                     row * 7u + block * 11u + i * 13u);
                }
                if (type == 10u) {
                    const uint16_t d = 0x3c00u;
                    const uint16_t dmin = 0x3800u;
                    memcpy(b + 80u, &d, sizeof(d));
                    memcpy(b + 82u, &dmin, sizeof(dmin));
                } else if (type == 11u) {
                    const uint16_t d = 0x3c00u;
                    memcpy(b + 108u, &d, sizeof(d));
                } else {
                    const uint16_t d = 0x3c00u;
                    const uint16_t dmin = 0x3800u;
                    memcpy(b + 0u, &d, sizeof(d));
                    memcpy(b + 2u, &dmin, sizeof(dmin));
                }
            }
        }
    }
}

static void test_metal_glm_qmv_r1_case(uint32_t type) {
    const uint32_t n_total_expert = 2u;
    const uint32_t n_expert = 2u;
    const uint32_t in_dim = 512u;
    const uint32_t mid_dim = 512u;
    const uint32_t out_dim = 257u;
    const uint32_t blocks = in_dim / 256u;
    const uint32_t block_bytes =
        type == 10u ? 84u : type == 11u ? 110u : 144u;
    const uint64_t row_bytes = (uint64_t)blocks * block_bytes;
    const uint64_t gate_expert_bytes = (uint64_t)mid_dim * row_bytes;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t gate_offset = 0u;
    const uint64_t up_offset = test_round_up_u64(
        (uint64_t)n_total_expert * gate_expert_bytes, page);
    const uint64_t down_offset = test_round_up_u64(
        up_offset + (uint64_t)n_total_expert * gate_expert_bytes, page);
    const uint64_t model_size = test_round_up_u64(
        down_offset + (uint64_t)n_total_expert * down_expert_bytes, page);

    void *model = NULL;
    ds4_gpu_tensor *x = NULL;
    ds4_gpu_tensor *selected = NULL;
    ds4_gpu_tensor *weights = NULL;
    ds4_gpu_tensor *mid = NULL;
    ds4_gpu_tensor *out = NULL;
    float *x_host = NULL;
    int32_t *selected_host = NULL;
    float *weights_host = NULL;
    float *mid_init = NULL;
    float *out_init = NULL;
    float *mid_baseline = NULL;
    float *mid_r1 = NULL;
    float *out_baseline = NULL;
    float *out_r1 = NULL;
    char *saved_r1 = NULL;
    bool env_saved = false;

    TEST_ASSERT(posix_memalign(&model, (size_t)page, (size_t)model_size) == 0);
    if (!model) goto cleanup;
    memset(model, 0, (size_t)model_size);
    test_fill_metal_glm_qmv_weights(
        (uint8_t *)model + gate_offset, type, n_total_expert, mid_dim,
        blocks, 1u);
    test_fill_metal_glm_qmv_weights(
        (uint8_t *)model + up_offset, type, n_total_expert, mid_dim,
        blocks, 3u);
    test_fill_metal_glm_qmv_weights(
        (uint8_t *)model + down_offset, type, n_total_expert, out_dim,
        blocks, 5u);

    x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    selected = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(int32_t));
    weights = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(float));
    mid = ds4_gpu_tensor_alloc(
        (uint64_t)n_expert * mid_dim * sizeof(float));
    out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    x_host = malloc((size_t)in_dim * sizeof(float));
    selected_host = malloc((size_t)n_expert * sizeof(int32_t));
    weights_host = malloc((size_t)n_expert * sizeof(float));
    mid_init = malloc((size_t)n_expert * mid_dim * sizeof(float));
    out_init = malloc((size_t)out_dim * sizeof(float));
    mid_baseline = malloc((size_t)n_expert * mid_dim * sizeof(float));
    mid_r1 = malloc((size_t)n_expert * mid_dim * sizeof(float));
    out_baseline = malloc((size_t)out_dim * sizeof(float));
    out_r1 = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x && selected && weights && mid && out && x_host &&
                selected_host && weights_host && mid_init && out_init &&
                mid_baseline && mid_r1 && out_baseline && out_r1);
    if (!x || !selected || !weights || !mid || !out || !x_host ||
        !selected_host || !weights_host || !mid_init || !out_init ||
        !mid_baseline || !mid_r1 || !out_baseline || !out_r1) {
        goto cleanup;
    }

    for (uint32_t k = 0; k < in_dim; k++) {
        x_host[k] = 0.125f + (float)((k * 19u + (k >> 3u) * 7u) % 97u) /
                   128.0f;
    }
    /* Exercise two distinct resident experts with nontrivial route weights. */
    selected_host[0] = 0;
    selected_host[1] = 1;
    weights_host[0] = 0.625f;
    weights_host[1] = 0.375f;
    for (uint32_t i = 0; i < n_expert * mid_dim; i++) {
        mid_init[i] = 0.0f;
    }
    for (uint32_t i = 0; i < out_dim; i++) out_init[i] = 0.0f;

    TEST_ASSERT(ds4_gpu_set_model_map(model, model_size) != 0);
    saved_r1 = test_save_env("DS4_METAL_GLM_QMV_R1");
    env_saved = true;
    TEST_ASSERT(unsetenv("DS4_METAL_GLM_QMV_R1") == 0);

    TEST_ASSERT(ds4_gpu_tensor_write(
                    x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    selected, 0, selected_host,
                    (uint64_t)n_expert * sizeof(int32_t)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    weights, 0, weights_host,
                    (uint64_t)n_expert * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    mid, 0, mid_init,
                    (uint64_t)n_expert * mid_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    out, 0, out_init, (uint64_t)out_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_glm_routed_moe_one_tensor(
                    out, mid, model, model_size,
                    gate_offset, up_offset, down_offset,
                    type, type, type,
                    gate_expert_bytes, row_bytes,
                    gate_expert_bytes, row_bytes,
                    down_expert_bytes, row_bytes,
                    in_dim, mid_dim, out_dim,
                    selected, weights,
                    n_total_expert, n_expert, 0u, x, true) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    mid, 0, mid_baseline,
                    (uint64_t)n_expert * mid_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    out, 0, out_baseline,
                    (uint64_t)out_dim * sizeof(float)) != 0);

    TEST_ASSERT(setenv("DS4_METAL_GLM_QMV_R1", "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    mid, 0, mid_init,
                    (uint64_t)n_expert * mid_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    out, 0, out_init, (uint64_t)out_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_glm_routed_moe_one_tensor(
                    out, mid, model, model_size,
                    gate_offset, up_offset, down_offset,
                    type, type, type,
                    gate_expert_bytes, row_bytes,
                    gate_expert_bytes, row_bytes,
                    down_expert_bytes, row_bytes,
                    in_dim, mid_dim, out_dim,
                    selected, weights,
                    n_total_expert, n_expert, 0u, x, true) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    mid, 0, mid_r1,
                    (uint64_t)n_expert * mid_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    out, 0, out_r1,
                    (uint64_t)out_dim * sizeof(float)) != 0);

    {
        const test_float_compare_stats mid_stats = test_compare_float_bits(
            mid_baseline, mid_r1, (size_t)n_expert * mid_dim);
        const test_float_compare_stats out_stats = test_compare_float_bits(
            out_baseline, out_r1, out_dim);
        fprintf(stderr,
                "ds4-test: GLM QMV R1 Q%u exact mid=%zu/%u out=%zu/%u "
                "max_ulp=%u/%u max_abs=%g/%g\n",
                type == 10u ? 2u : type == 11u ? 3u : 4u,
                mid_stats.mismatch_count, n_expert * mid_dim,
                out_stats.mismatch_count, out_dim,
                mid_stats.max_ulp, out_stats.max_ulp,
                mid_stats.max_abs, out_stats.max_abs);
        TEST_ASSERT(mid_stats.mismatch_count == 0);
        TEST_ASSERT(out_stats.mismatch_count == 0);
    }

    /* Keep env_saved active until cleanup so an assertion in either this
     * section or the valid comparison still restores the caller's setting.
     * Invalid IDs are a separate bounds contract: pair kernels zero their
     * corresponding mid rows, and the down kernel ignores those slots.  Use
     * both sides of the valid range and nonzero sentinels so stale data cannot
     * make this check pass accidentally. */
    selected_host[0] = -1;
    selected_host[1] = (int32_t)n_total_expert;
    for (uint32_t i = 0; i < n_expert * mid_dim; i++) mid_init[i] = 17.25f;
    for (uint32_t i = 0; i < out_dim; i++) out_init[i] = -9.5f;
    TEST_ASSERT(ds4_gpu_tensor_write(
                    selected, 0, selected_host,
                    (uint64_t)n_expert * sizeof(int32_t)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    mid, 0, mid_init,
                    (uint64_t)n_expert * mid_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    out, 0, out_init, (uint64_t)out_dim * sizeof(float)) != 0);
    TEST_ASSERT(unsetenv("DS4_METAL_GLM_QMV_R1") == 0);
    TEST_ASSERT(ds4_gpu_glm_routed_moe_one_tensor(
                    out, mid, model, model_size,
                    gate_offset, up_offset, down_offset,
                    type, type, type,
                    gate_expert_bytes, row_bytes,
                    gate_expert_bytes, row_bytes,
                    down_expert_bytes, row_bytes,
                    in_dim, mid_dim, out_dim,
                    selected, weights,
                    n_total_expert, n_expert, 0u, x, true) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    mid, 0, mid_baseline,
                    (uint64_t)n_expert * mid_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    out, 0, out_baseline,
                    (uint64_t)out_dim * sizeof(float)) != 0);
    for (uint32_t i = 0; i < n_expert * mid_dim; i++) {
        TEST_ASSERT(mid_baseline[i] == 0.0f);
    }
    for (uint32_t i = 0; i < out_dim; i++) {
        TEST_ASSERT(out_baseline[i] == 0.0f);
    }

    TEST_ASSERT(setenv("DS4_METAL_GLM_QMV_R1", "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    mid, 0, mid_init,
                    (uint64_t)n_expert * mid_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    out, 0, out_init, (uint64_t)out_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_glm_routed_moe_one_tensor(
                    out, mid, model, model_size,
                    gate_offset, up_offset, down_offset,
                    type, type, type,
                    gate_expert_bytes, row_bytes,
                    gate_expert_bytes, row_bytes,
                    down_expert_bytes, row_bytes,
                    in_dim, mid_dim, out_dim,
                    selected, weights,
                    n_total_expert, n_expert, 0u, x, true) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    mid, 0, mid_r1,
                    (uint64_t)n_expert * mid_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    out, 0, out_r1,
                    (uint64_t)out_dim * sizeof(float)) != 0);
    for (uint32_t i = 0; i < n_expert * mid_dim; i++) {
        TEST_ASSERT(mid_r1[i] == 0.0f);
    }
    for (uint32_t i = 0; i < out_dim; i++) {
        TEST_ASSERT(out_r1[i] == 0.0f);
    }

cleanup:
    if (env_saved) test_restore_env("DS4_METAL_GLM_QMV_R1", saved_r1);
    free(out_r1);
    free(out_baseline);
    free(mid_r1);
    free(mid_baseline);
    free(out_init);
    free(mid_init);
    free(weights_host);
    free(selected_host);
    free(x_host);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(x);
    free(model);
}

static void test_metal_glm_qmv_r1_exact(void) {
    TEST_ASSERT(ds4_gpu_init() != 0);
    test_metal_glm_qmv_r1_case(10u);
    test_metal_glm_qmv_r1_case(11u);
    test_metal_glm_qmv_r1_case(12u);
}

static void test_metal_laguna_staged_swa_case(
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t pos0,
        uint32_t n_tokens) {
    const uint32_t head_dim = 128u;
    const uint32_t cache_cap = 512u;
    const uint32_t cache_width = n_head_kv * head_dim;
    const uint64_t cache_values = (uint64_t)cache_cap * cache_width;
    const uint64_t q_values = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_values = (uint64_t)n_tokens * cache_width;
    const uint64_t gate_values = (uint64_t)n_tokens * n_head;
    const uint64_t heads_bytes = q_values * sizeof(float);
    const uint64_t cache_bytes = cache_values * sizeof(uint16_t);
    const uint64_t current_kv_bytes = kv_values * sizeof(float);
    const uint64_t staged_bytes = kv_values * sizeof(uint16_t);
    const uint64_t gate_bytes = gate_values * sizeof(float);

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *key_cache = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *value_cache = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *staged_key = ds4_gpu_tensor_alloc(staged_bytes);
    ds4_gpu_tensor *staged_value = ds4_gpu_tensor_alloc(staged_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(current_kv_bytes);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(gate_bytes);
    uint16_t *initial_key = malloc((size_t)cache_bytes);
    uint16_t *initial_value = malloc((size_t)cache_bytes);
    uint16_t *baseline_key = malloc((size_t)cache_bytes);
    uint16_t *baseline_value = malloc((size_t)cache_bytes);
    uint16_t *staged_key_out = malloc((size_t)cache_bytes);
    uint16_t *staged_value_out = malloc((size_t)cache_bytes);
    uint16_t *staged_key_poison = malloc((size_t)staged_bytes);
    uint16_t *staged_value_poison = malloc((size_t)staged_bytes);
    float *q_host = malloc((size_t)heads_bytes);
    float *k_host = malloc((size_t)current_kv_bytes);
    float *v_host = malloc((size_t)current_kv_bytes);
    float *gate_host = malloc((size_t)gate_bytes);
    float *baseline_heads_seed = malloc((size_t)heads_bytes);
    float *staged_heads_seed = malloc((size_t)heads_bytes);
    float *baseline_heads = malloc((size_t)heads_bytes);
    float *staged_heads = malloc((size_t)heads_bytes);
    char *saved_staged_env = NULL;
    char *saved_require_env = NULL;
    char *saved_gqa3_env = NULL;
    char *saved_gqa9_env = NULL;
    bool staged_env_saved = false;
    bool require_env_saved = false;
    bool gqa3_env_saved = false;
    bool gqa9_env_saved = false;
    const char *staged_env_name = "DS4_METAL_LAGUNA_STAGED_SWA";
    const char *require_env_name = "DS4_METAL_LAGUNA_REQUIRE_STAGED_SWA";
    const char *gqa3_env_name = "DS4_METAL_LAGUNA_SWA_GQA3";
    const char *gqa9_env_name = "DS4_METAL_LAGUNA_SWA_GQA9";
    TEST_ASSERT(heads && key_cache && value_cache && staged_key &&
                staged_value && q && k && v && gate && initial_key &&
                initial_value && baseline_key && baseline_value &&
                staged_key_out && staged_value_out && staged_key_poison &&
                staged_value_poison && q_host && k_host &&
                v_host && gate_host && baseline_heads_seed &&
                staged_heads_seed && baseline_heads && staged_heads);
    if (!heads || !key_cache || !value_cache || !staged_key ||
        !staged_value || !q || !k || !v || !gate || !initial_key ||
        !initial_value || !baseline_key || !baseline_value ||
        !staged_key_out || !staged_value_out || !staged_key_poison ||
        !staged_value_poison || !q_host || !k_host ||
        !v_host || !gate_host || !baseline_heads_seed ||
        !staged_heads_seed || !baseline_heads || !staged_heads) {
        goto cleanup;
    }

    /* Each old slot is a sentinel with a distinct value.  New rows use much
     * larger, row-distinct K/V values so a future-row substitution changes the
     * first query visibly; the A/B comparison therefore catches leakage. */
    for (uint32_t slot = 0; slot < cache_cap; slot++) {
        for (uint32_t col = 0; col < cache_width; col++) {
            const float key_value = 0.03125f +
                (float)((slot * 17u + col * 11u) % 251u) / 64.0f;
            const float value_value = -0.0625f +
                (float)((slot * 23u + col * 7u) % 239u) / 72.0f;
            initial_key[(uint64_t)slot * cache_width + col] =
                test_float_to_f16(key_value);
            initial_value[(uint64_t)slot * cache_width + col] =
                test_float_to_f16(value_value);
        }
    }
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t col = 0; col < cache_width; col++) {
            /* Make future rows deliberately unlike row zero. */
            k_host[(uint64_t)token * cache_width + col] =
                3.0f + (float)token * 2.75f +
                (float)((col * 13u + token * 7u) % 31u) / 80.0f;
            v_host[(uint64_t)token * cache_width + col] =
                -4.0f - (float)token * 1.5f +
                (float)((col * 19u + token * 5u) % 37u) / 96.0f;
        }
    }
    for (uint64_t i = 0; i < q_values; i++) {
        const uint32_t token = (uint32_t)(i / ((uint64_t)n_head * head_dim));
        const uint32_t col = (uint32_t)(i % head_dim);
        const uint32_t head =
            (uint32_t)((i / head_dim) % n_head);
        q_host[i] = 0.015625f + (float)token * 0.75f +
            (float)head * 0.03125f + (float)(col % 17u) / 96.0f;
    }
    for (uint64_t i = 0; i < gate_values; i++) {
        gate_host[i] = -0.25f + (float)(i % 11u) / 37.0f;
    }
    for (uint64_t i = 0; i < kv_values; i++) {
        /* Sign-distinct NaN payloads make a missing stage write visible even
         * when equal-sized GQA cases reuse the same allocator storage. */
        staged_key_poison[i] = (uint16_t)(0x7c01u |
                                          ((i + pos0 + n_head) & 0x03ffu));
        staged_value_poison[i] = (uint16_t)(0xfc01u |
                                            ((i * 3u + pos0 + n_tokens) &
                                             0x03ffu));
    }
    for (uint64_t i = 0; i < q_values; i++) {
        baseline_heads_seed[i] = 1000.0f + (float)i * 0.001f;
        staged_heads_seed[i] = -2000.0f - (float)i * 0.001f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(k, 0, k_host, current_kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(v, 0, v_host, current_kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(gate, 0, gate_host, gate_bytes) != 0);

    const bool allow_staged_fallback =
        test_env_bool("DS4_TEST_LAGUNA_STAGED_SWA_ALLOW_FALLBACK");
    saved_staged_env = test_save_env(staged_env_name);
    staged_env_saved = true;
    saved_require_env = test_save_env(require_env_name);
    require_env_saved = true;
    saved_gqa3_env = test_save_env(gqa3_env_name);
    gqa3_env_saved = true;
    saved_gqa9_env = test_save_env(gqa9_env_name);
    gqa9_env_saved = true;
    TEST_ASSERT(unsetenv(staged_env_name) == 0);
    TEST_ASSERT(unsetenv(require_env_name) == 0);
    /* The ordinary-vs-staged exactness leg is independent of an exported
     * grouped-GQA3 flag.  The production-sized numeric test separately covers
     * the single-token precedence contract with both flags enabled. */
    TEST_ASSERT(unsetenv(gqa3_env_name) == 0);
    TEST_ASSERT(unsetenv(gqa9_env_name) == 0);
    TEST_ASSERT(test_restart_metal_lifecycle());
    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, baseline_heads_seed, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, initial_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, initial_value, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    staged_key, 0, staged_key_poison, staged_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    staged_value, 0, staged_value_poison, staged_bytes) != 0);
    TEST_ASSERT(ds4_gpu_laguna_attention_prefill_tensor(
                    heads, key_cache, value_cache, staged_key, staged_value,
                    q, k, v, gate, pos0, n_tokens, cache_cap,
                    n_head, n_head_kv, head_dim,
                    1.0f / sqrtf((float)head_dim), 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, baseline_heads, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, baseline_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, baseline_value, cache_bytes) != 0);

    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, staged_heads_seed, heads_bytes) != 0);
    TEST_ASSERT(setenv(staged_env_name, "1", 1) == 0);
    if (allow_staged_fallback) {
        TEST_ASSERT(unsetenv(require_env_name) == 0);
    } else {
        TEST_ASSERT(setenv(require_env_name, "1", 1) == 0);
    }
    TEST_ASSERT(test_restart_metal_lifecycle());
    ds4_gpu_test_laguna_route_counters_reset();
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, initial_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, initial_value, cache_bytes) != 0);
    /* Poison immediately before the required staged leg so a stage-kernel
     * no-write cannot be masked by allocator reuse or a prior test case. */
    TEST_ASSERT(ds4_gpu_tensor_write(
                    staged_key, 0, staged_key_poison, staged_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    staged_value, 0, staged_value_poison, staged_bytes) != 0);
    TEST_ASSERT(ds4_gpu_laguna_attention_prefill_tensor(
                    heads, key_cache, value_cache, staged_key, staged_value,
                    q, k, v, gate, pos0, n_tokens, cache_cap,
                    n_head, n_head_kv, head_dim,
                    1.0f / sqrtf((float)head_dim), 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, staged_heads, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, staged_key_out, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, staged_value_out, cache_bytes) != 0);

    if (!allow_staged_fallback) {
        uint64_t ordinary = 0;
        uint64_t gqa3 = 0;
        uint64_t gqa9 = 0;
        uint64_t staged = 0;
        uint64_t global_grouped = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        TEST_ASSERT(ordinary == 0u && gqa3 == 0u && gqa9 == 0u &&
                    staged == 1u && global_grouped == 0u);
    }

    TEST_ASSERT(memcmp(baseline_heads, staged_heads, (size_t)heads_bytes) == 0);
    TEST_ASSERT(memcmp(baseline_key, staged_key_out, (size_t)cache_bytes) == 0);
    TEST_ASSERT(memcmp(baseline_value, staged_value_out,
                       (size_t)cache_bytes) == 0);

    const uint32_t pos_mod = pos0 % cache_cap;
    for (uint32_t slot = 0; slot < cache_cap; slot++) {
        const uint32_t u = slot >= pos_mod ? slot - pos_mod :
            slot + cache_cap - pos_mod;
        const bool overwritten = u < n_tokens;
        for (uint32_t col = 0; col < cache_width; col++) {
            const uint16_t expected_key = overwritten ?
                test_float_to_f16(k_host[(uint64_t)u * cache_width + col]) :
                initial_key[(uint64_t)slot * cache_width + col];
            const uint16_t expected_value = overwritten ?
                test_float_to_f16(v_host[(uint64_t)u * cache_width + col]) :
                initial_value[(uint64_t)slot * cache_width + col];
            TEST_ASSERT(staged_key_out[(uint64_t)slot * cache_width + col] ==
                        expected_key);
            TEST_ASSERT(staged_value_out[(uint64_t)slot * cache_width + col] ==
                        expected_value);
            if (!overwritten) {
                TEST_ASSERT(staged_key_out[(uint64_t)slot * cache_width + col] ==
                            initial_key[(uint64_t)slot * cache_width + col]);
                TEST_ASSERT(staged_value_out[(uint64_t)slot * cache_width + col] ==
                            initial_value[(uint64_t)slot * cache_width + col]);
            }
        }
    }

    /* Re-run the staged multi-row path with both experiments exported.  The
     * staged pipeline must remain the exact counterpart of the ordinary
     * baseline; this catches regressions where an externally enabled grouped
     * flag changes the reference leg or leaks into staged execution. */
    TEST_ASSERT(setenv(gqa3_env_name, "1", 1) == 0);
    TEST_ASSERT(test_restart_metal_lifecycle());
    ds4_gpu_test_laguna_route_counters_reset();
    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, staged_heads_seed, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    key_cache, 0, initial_key, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    value_cache, 0, initial_value, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    staged_key, 0, staged_key_poison, staged_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    staged_value, 0, staged_value_poison, staged_bytes) != 0);
    TEST_ASSERT(ds4_gpu_laguna_attention_prefill_tensor(
                    heads, key_cache, value_cache, staged_key, staged_value,
                    q, k, v, gate, pos0, n_tokens, cache_cap,
                    n_head, n_head_kv, head_dim,
                    1.0f / sqrtf((float)head_dim), 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    heads, 0, staged_heads, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    key_cache, 0, staged_key_out, cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    value_cache, 0, staged_value_out, cache_bytes) != 0);
    if (!allow_staged_fallback) {
        uint64_t ordinary = 0;
        uint64_t gqa3 = 0;
        uint64_t gqa9 = 0;
        uint64_t staged = 0;
        uint64_t global_grouped = 0;
        TEST_ASSERT(ds4_gpu_test_laguna_decode_route_counters(
                        &ordinary, &gqa3, &gqa9, &staged,
                        &global_grouped) != 0);
        TEST_ASSERT(ordinary == 0u && gqa3 == 0u && gqa9 == 0u &&
                    staged == 1u && global_grouped == 0u);
    }
    TEST_ASSERT(memcmp(baseline_heads, staged_heads, (size_t)heads_bytes) == 0);
    TEST_ASSERT(memcmp(baseline_key, staged_key_out, (size_t)cache_bytes) == 0);
    TEST_ASSERT(memcmp(baseline_value, staged_value_out,
                       (size_t)cache_bytes) == 0);
    fprintf(stderr,
            "ds4-test: Laguna staged SWA combined-flags exact GQA%u "
            "pos=%u rows=%u\n",
            n_head / n_head_kv, pos0, n_tokens);

    fprintf(stderr,
            "ds4-test: Laguna staged SWA A/B exact GQA%u pos=%u rows=%u\n",
            n_head / n_head_kv, pos0, n_tokens);

    if (require_env_saved) {
        test_restore_env(require_env_name, saved_require_env);
        saved_require_env = NULL;
        require_env_saved = false;
    }
    if (staged_env_saved) {
        test_restore_env(staged_env_name, saved_staged_env);
        saved_staged_env = NULL;
        staged_env_saved = false;
    }
    if (gqa3_env_saved) {
        test_restore_env(gqa3_env_name, saved_gqa3_env);
        saved_gqa3_env = NULL;
        gqa3_env_saved = false;
    }
    if (gqa9_env_saved) {
        test_restore_env(gqa9_env_name, saved_gqa9_env);
        saved_gqa9_env = NULL;
        gqa9_env_saved = false;
    }

cleanup:
    if (require_env_saved) {
        test_restore_env(require_env_name, saved_require_env);
        saved_require_env = NULL;
        require_env_saved = false;
    }
    if (staged_env_saved) {
        test_restore_env(staged_env_name, saved_staged_env);
        saved_staged_env = NULL;
        staged_env_saved = false;
    }
    if (gqa3_env_saved) {
        test_restore_env(gqa3_env_name, saved_gqa3_env);
        saved_gqa3_env = NULL;
        gqa3_env_saved = false;
    }
    if (gqa9_env_saved) {
        test_restore_env(gqa9_env_name, saved_gqa9_env);
        saved_gqa9_env = NULL;
        gqa9_env_saved = false;
    }
    free(staged_heads);
    free(baseline_heads);
    free(staged_heads_seed);
    free(baseline_heads_seed);
    free(gate_host);
    free(v_host);
    free(k_host);
    free(q_host);
    free(staged_value_out);
    free(staged_key_out);
    free(staged_value_poison);
    free(staged_key_poison);
    free(baseline_value);
    free(baseline_key);
    free(initial_value);
    free(initial_key);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(staged_value);
    ds4_gpu_tensor_free(staged_key);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
}

static void test_metal_laguna_staged_swa_exact(void) {
    static const struct {
        uint32_t pos0;
        uint32_t n_tokens;
    } cases[] = {
        { 512u, 2u },
        { 516u, 4u },
        { 1020u, 16u },
        { 1024u, 16u },
    };
    TEST_ASSERT(ds4_gpu_init() != 0);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        test_metal_laguna_staged_swa_case(
            72u, 8u, cases[i].pos0, cases[i].n_tokens);
        test_metal_laguna_staged_swa_case(
            48u, 8u, cases[i].pos0, cases[i].n_tokens);
    }
}

static void test_metal_laguna_qk_norm_rope_pair_exact(void) {
    typedef struct {
        uint32_t n_tokens;
        uint32_t n_q_head;
        uint32_t n_k_head;
        uint32_t pos0;
        uint32_t n_rot;
        float ext_factor;
    } qk_case;
    static const qk_case cases[] = {
        /* Production SWA geometry: 72 Q / 8 K, head_dim=128, n_rot=128,
         * freq_base=10000, scale=1, and the full Laguna context (ext=0). */
        { 1, 72, 8,    37, 128, 0.0f },
        { 17, 72, 8,   510, 128, 0.0f },
        { 32, 72, 8,  1024, 128, 0.0f },
        /* Production global YaRN geometry: 48 Q / 8 K, head_dim=128,
         * n_rot=64, freq_base=500000, scale=1/32, original ctx=8192,
         * and beta_fast/beta_slow=32/1 (ext=1). */
        { 1,  48, 8, 65533,  64, 1.0f },
        /* Deliberately malformed geometry: proves env-on rejects a shape
         * outside either production layer family. */
        { 3,   7, 3,  2047,  64, 1.0f },
        { 64, 48, 8,  8191,  64, 1.0f },
    };
    const uint32_t head_dim = 128;
    const char *simd32_env_name =
        "DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32";
    char *saved_simd32_env = test_save_env(simd32_env_name);
    TEST_ASSERT(unsetenv(simd32_env_name) == 0);
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t k_weight_offset = page;
    const uint64_t model_size = 2u * page;
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_size) == 0);
    if (!model_raw) {
        test_restore_env(simd32_env_name, saved_simd32_env);
        return;
    }
    memset(model_raw, 0, (size_t)model_size);

    float *q_weight = model_raw;
    float *k_weight = (float *)((uint8_t *)model_raw + k_weight_offset);
    for (uint32_t i = 0; i < head_dim; i++) {
        q_weight[i] = 0.75f + (float)((i * 17u + 3u) % 29u) / 64.0f;
        k_weight[i] = 0.625f + (float)((i * 19u + 5u) % 31u) / 56.0f;
    }
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_size) != 0);

    size_t q_mismatches = 0;
    size_t k_mismatches = 0;
    size_t simd_q_mismatches = 0;
    size_t simd_k_mismatches = 0;
    size_t malformed_mismatches = 0;
    size_t rejected_mismatches = 0;
    uint32_t max_ulp = 0;
    uint64_t simd_encoded_dispatches_before =
        ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count();
    const uint64_t simd_completed_dispatches_before =
        ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count();
    const test_laguna_source_mode source_mode =
        test_laguna_source_mode_from_env();
    TEST_ASSERT(source_mode != TEST_LAGUNA_SOURCE_INVALID);
    size_t exact_cases = 0;
    bool simd32_available = false;
    bool simd32_capability_known = false;
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_env_mode() == 0);
    TEST_ASSERT(setenv(simd32_env_name, "", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_env_mode() == 0);
    TEST_ASSERT(setenv(simd32_env_name, "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_env_mode() == 0);
    TEST_ASSERT(setenv(simd32_env_name, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_env_mode() == 1);
    static const char *const malformed_env_values[] = {
        "true", "yes", "on", " 1", "01", "0 ", "-1",
        "1 ", "\t1", "\n1", "00", "false", "TRUE",
    };
    for (size_t mi = 0;
        mi < sizeof(malformed_env_values) / sizeof(malformed_env_values[0]);
         mi++) {
        TEST_ASSERT(setenv(simd32_env_name, malformed_env_values[mi], 1) == 0);
        TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
        TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_env_mode() < 0);
        TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                        48u, 8u, head_dim, 64u) < 0);
        TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached() == -1);
    }
    TEST_ASSERT(unsetenv(simd32_env_name) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
    /* A frozen disabled plan ignores later environment mutation until the
     * explicit lifecycle/test reset. */
    TEST_ASSERT(setenv(simd32_env_name, "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                    48u, 8u, head_dim, 64u) == 0);
    TEST_ASSERT(setenv(simd32_env_name, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                    48u, 8u, head_dim, 64u) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached() == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
    TEST_ASSERT(setenv(simd32_env_name, "1", 1) == 0);
    static const float adversarial_values[] = {
        0.0f, -0.0f,
        1.40129846e-45f, -1.40129846e-45f,
        1.17549435e-38f, -1.17549435e-38f,
        65504.0f, -65504.0f,
        1.0e18f, -1.0e18f,
    };
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const qk_case *c = &cases[ci];
        const bool exact_geometry =
            (c->n_q_head == 48u || c->n_q_head == 72u) &&
            c->n_k_head == 8u && c->n_rot <= head_dim &&
            (c->n_rot == 64u || c->n_rot == 128u);
        if (exact_geometry) exact_cases++;
        const uint64_t q_values =
            (uint64_t)c->n_tokens * c->n_q_head * head_dim;
        const uint64_t k_values =
            (uint64_t)c->n_tokens * c->n_k_head * head_dim;
        const uint64_t q_bytes = q_values * sizeof(float);
        const uint64_t k_bytes = k_values * sizeof(float);
        ds4_gpu_tensor *ref_q = ds4_gpu_tensor_alloc(q_bytes);
        ds4_gpu_tensor *ref_k = ds4_gpu_tensor_alloc(k_bytes);
        ds4_gpu_tensor *pair_q = ds4_gpu_tensor_alloc(q_bytes);
        ds4_gpu_tensor *pair_k = ds4_gpu_tensor_alloc(k_bytes);
        ds4_gpu_tensor *simd_q = ds4_gpu_tensor_alloc(q_bytes);
        ds4_gpu_tensor *simd_k = ds4_gpu_tensor_alloc(k_bytes);
        float *q_input = malloc((size_t)q_bytes);
        float *k_input = malloc((size_t)k_bytes);
        float *ref_q_host = malloc((size_t)q_bytes);
        float *ref_k_host = malloc((size_t)k_bytes);
        float *pair_q_host = malloc((size_t)q_bytes);
        float *pair_k_host = malloc((size_t)k_bytes);
        float *simd_q_host = malloc((size_t)q_bytes);
        float *simd_k_host = malloc((size_t)k_bytes);
        TEST_ASSERT(ref_q && ref_k && pair_q && pair_k &&
                    simd_q && simd_k && q_input && k_input &&
                    ref_q_host && ref_k_host && pair_q_host && pair_k_host &&
                    simd_q_host && simd_k_host);
        if (!ref_q || !ref_k || !pair_q || !pair_k ||
            !simd_q || !simd_k || !q_input || !k_input ||
            !ref_q_host || !ref_k_host || !pair_q_host || !pair_k_host ||
            !simd_q_host || !simd_k_host) {
            free(simd_k_host);
            free(simd_q_host);
            free(pair_k_host);
            free(pair_q_host);
            free(ref_k_host);
            free(ref_q_host);
            free(k_input);
            free(q_input);
            ds4_gpu_tensor_free(pair_k);
            ds4_gpu_tensor_free(pair_q);
            ds4_gpu_tensor_free(ref_k);
            ds4_gpu_tensor_free(ref_q);
            ds4_gpu_tensor_free(simd_k);
            ds4_gpu_tensor_free(simd_q);
            continue;
        }

        for (uint64_t i = 0; i < q_values; i++) {
            if (ci + 1u == sizeof(cases) / sizeof(cases[0])) {
                q_input[i] = adversarial_values[
                    (i * 7u + (i >> 4u)) %
                    (sizeof(adversarial_values) / sizeof(adversarial_values[0]))];
            } else {
                const int v = (int)((i * 37u + (i >> 3u) * 11u +
                                     ci * 13u + 7u) % 211u) - 105;
                q_input[i] = (float)v / 137.0f;
            }
        }
        for (uint64_t i = 0; i < k_values; i++) {
            if (ci + 1u == sizeof(cases) / sizeof(cases[0])) {
                k_input[i] = adversarial_values[
                    (i * 11u + (i >> 3u) + 3u) %
                    (sizeof(adversarial_values) / sizeof(adversarial_values[0]))];
            } else {
                const int v = (int)((i * 41u + (i >> 2u) * 17u +
                                     ci * 23u + 5u) % 199u) - 99;
                k_input[i] = (float)v / 149.0f;
            }
        }
        TEST_ASSERT(ds4_gpu_tensor_write(ref_q, 0, q_input, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(pair_q, 0, q_input, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(ref_k, 0, k_input, k_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(pair_k, 0, k_input, k_bytes) != 0);

        const bool global_rope = c->ext_factor != 0.0f;
        const float freq_base = global_rope ? 500000.0f : 10000.0f;
        const float freq_scale = global_rope ? 1.0f / 32.0f : 1.0f;
        const uint32_t n_ctx_orig = global_rope ? 8192u : 262144u;
        const float attn_factor = 1.0f;
        const float beta_fast = global_rope ? 32.0f : 0.0f;
        const float beta_slow = global_rope ? 1.0f : 0.0f;
        /* Reference calls must use the stock selector.  The opt-in SIMD32
         * plan is intentionally strict for single-K callers and would reject
         * this baseline route while it is frozen enabled. */
        TEST_ASSERT(setenv(simd32_env_name, "0", 1) == 0);
        TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
        TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_tensor(
                        ref_q, model_raw, model_size, 0,
                        c->n_tokens, c->n_q_head, head_dim, c->n_rot,
                        c->pos0, n_ctx_orig, freq_base, freq_scale,
                        c->ext_factor, attn_factor, beta_fast, beta_slow,
                        1e-6f) != 0);
        TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_tensor(
                        ref_k, model_raw, model_size, k_weight_offset,
                        c->n_tokens, c->n_k_head, head_dim, c->n_rot,
                        c->pos0, n_ctx_orig, freq_base, freq_scale,
                        c->ext_factor, attn_factor, beta_fast, beta_slow,
                        1e-6f) != 0);
        TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                        pair_q, pair_k, model_raw, model_size,
                        0, k_weight_offset, c->n_tokens,
                        c->n_q_head, c->n_k_head, head_dim, c->n_rot,
                        c->pos0, n_ctx_orig, freq_base, freq_scale,
                        c->ext_factor, attn_factor, beta_fast, beta_slow,
                        1e-6f) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_q, 0, ref_q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_k, 0, ref_k_host, k_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        pair_q, 0, pair_q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        pair_k, 0, pair_k_host, k_bytes) != 0);

        TEST_ASSERT(ds4_gpu_tensor_write(simd_q, 0, q_input, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(simd_k, 0, k_input, k_bytes) != 0);
        TEST_ASSERT(setenv(simd32_env_name, "1", 1) == 0);
        ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
        TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                        c->n_q_head, c->n_k_head, head_dim, c->n_rot) == 1 ||
                    !exact_geometry || source_mode == TEST_LAGUNA_SOURCE_OLD);
        const uint64_t dispatch_before_case =
            ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count();
        const uint64_t completed_before_case =
            ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count();
        const int simd_ok = ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
            simd_q, simd_k, model_raw, model_size,
            0, k_weight_offset, c->n_tokens,
            c->n_q_head, c->n_k_head, head_dim, c->n_rot,
            c->pos0, n_ctx_orig, freq_base, freq_scale,
            c->ext_factor, attn_factor, beta_fast, beta_slow, 1e-6f);
        if (exact_geometry) {
            if (!simd32_capability_known) {
                simd32_capability_known = true;
                simd32_available = simd_ok != 0;
            }
            TEST_ASSERT((simd_ok != 0) == simd32_available);
        }
        if (exact_geometry && simd32_available) {
            if (ds4_gpu_commands_active()) {
                TEST_ASSERT(ds4_gpu_end_commands() != 0);
            }
            TEST_ASSERT(ds4_gpu_wait_submitted_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() ==
                        dispatch_before_case + 1u);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count() ==
                        completed_before_case + 1u);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            simd_q, 0, simd_q_host, q_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            simd_k, 0, simd_k_host, k_bytes) != 0);
        } else if (!exact_geometry || !simd32_available) {
            /* Invalid geometry or an unavailable source pipeline are hard
             * failures under env-on; neither may silently execute stock PSOs. */
            if (!exact_geometry) TEST_ASSERT(simd_ok == 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() ==
                        dispatch_before_case);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count() ==
                        completed_before_case);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            simd_q, 0, simd_q_host, q_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            simd_k, 0, simd_k_host, k_bytes) != 0);
            rejected_mismatches +=
                test_compare_float_bits(q_input, simd_q_host,
                                        (size_t)q_values).mismatch_count +
                test_compare_float_bits(k_input, simd_k_host,
                                        (size_t)k_values).mismatch_count;
        }
        TEST_ASSERT(unsetenv(simd32_env_name) == 0);
        ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();

        const test_float_compare_stats q_stats =
            test_compare_float_bits(ref_q_host, pair_q_host, (size_t)q_values);
        const test_float_compare_stats k_stats =
            test_compare_float_bits(ref_k_host, pair_k_host, (size_t)k_values);
        q_mismatches += q_stats.mismatch_count;
        k_mismatches += k_stats.mismatch_count;
        if (q_stats.max_ulp > max_ulp) max_ulp = q_stats.max_ulp;
        if (k_stats.max_ulp > max_ulp) max_ulp = k_stats.max_ulp;
        if (exact_geometry && simd32_available) {
            const test_float_compare_stats simd_q_stats =
                test_compare_float_bits(pair_q_host, simd_q_host,
                                        (size_t)q_values);
            const test_float_compare_stats simd_k_stats =
                test_compare_float_bits(pair_k_host, simd_k_host,
                                        (size_t)k_values);
            simd_q_mismatches += simd_q_stats.mismatch_count;
            simd_k_mismatches += simd_k_stats.mismatch_count;
        }

        /* A malformed selector is fatal, not a safety fallback.  Re-run one
         * exact case from clean input and prove that no command or output
         * mutation occurred. */
        if (ci == 0u) {
            const uint64_t malformed_dispatches =
                ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count();
            const uint64_t malformed_completed_dispatches =
                ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count();
            TEST_ASSERT(setenv(simd32_env_name, "definitely-not-a-bool", 1) == 0);
            ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
            TEST_ASSERT(ds4_gpu_tensor_write(
                            simd_q, 0, q_input, q_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            simd_k, 0, k_input, k_bytes) != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            simd_q, simd_k, model_raw, model_size,
                            0, k_weight_offset, c->n_tokens,
                            c->n_q_head, c->n_k_head, head_dim, c->n_rot,
                            c->pos0, n_ctx_orig, freq_base, freq_scale,
                            c->ext_factor, attn_factor, beta_fast, beta_slow,
                            1e-6f) == 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() ==
                        malformed_dispatches);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count() ==
                        malformed_completed_dispatches);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            simd_q, 0, simd_q_host, q_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            simd_k, 0, simd_k_host, k_bytes) != 0);
            const test_float_compare_stats malformed_q_stats =
                test_compare_float_bits(q_input, simd_q_host,
                                        (size_t)q_values);
            const test_float_compare_stats malformed_k_stats =
                test_compare_float_bits(k_input, simd_k_host,
                                        (size_t)k_values);
            malformed_mismatches += malformed_q_stats.mismatch_count +
                malformed_k_stats.mismatch_count;
            TEST_ASSERT(unsetenv(simd32_env_name) == 0);
            ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
        }

        free(pair_k_host);
        free(pair_q_host);
        free(ref_k_host);
        free(ref_q_host);
        free(simd_k_host);
        free(simd_q_host);
        free(k_input);
        free(q_input);
        ds4_gpu_tensor_free(pair_k);
        ds4_gpu_tensor_free(pair_q);
        ds4_gpu_tensor_free(ref_k);
        ds4_gpu_tensor_free(ref_q);
        ds4_gpu_tensor_free(simd_k);
        ds4_gpu_tensor_free(simd_q);
    }

    fprintf(stderr,
            "ds4-test: Laguna paired Q/K norm/RoPE exact "
            "q_mismatches=%zu k_mismatches=%zu simd_q_mismatches=%zu "
            "simd_k_mismatches=%zu malformed_mismatches=%zu "
            "rejected_mismatches=%zu encoded_dispatches=%llu max_ulp=%u\n",
            q_mismatches, k_mismatches, simd_q_mismatches,
            simd_k_mismatches, malformed_mismatches, rejected_mismatches,
            (unsigned long long)(
                ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() -
                simd_encoded_dispatches_before),
            max_ulp);

    /* The selector is shape-gated as well as opt-in.  A head_dim=64 call
     * must fail under env-on and leave the output/counter untouched rather
     * than silently measuring the ordinary PSO. */
    const uint32_t fallback_head_dim = 64u;
    const uint64_t fallback_bytes =
        (uint64_t)fallback_head_dim * sizeof(float);
    ds4_gpu_tensor *fallback_simd_q = ds4_gpu_tensor_alloc(fallback_bytes);
    ds4_gpu_tensor *fallback_simd_k = ds4_gpu_tensor_alloc(fallback_bytes);
    float fallback_q_input[64];
    float fallback_k_input[64];
    float fallback_simd_q_host[64];
    float fallback_simd_k_host[64];
    TEST_ASSERT(fallback_simd_q && fallback_simd_k);
    size_t shape_fallback_mismatches = 0;
    if (fallback_simd_q && fallback_simd_k) {
        for (uint32_t i = 0; i < fallback_head_dim; i++) {
            fallback_q_input[i] = (float)((int)(i * 17u % 101u) - 50) / 64.0f;
            fallback_k_input[i] = (float)((int)(i * 23u % 97u) - 48) / 72.0f;
        }
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fallback_simd_q, 0, fallback_q_input, fallback_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fallback_simd_k, 0, fallback_k_input, fallback_bytes) != 0);
        TEST_ASSERT(setenv(simd32_env_name, "1", 1) == 0);
        ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
        const uint64_t shape_dispatches =
            ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count();
        TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                        fallback_simd_q, fallback_simd_k, model_raw, model_size,
                        0, k_weight_offset, 1, 1, 1, fallback_head_dim, 64,
                        9, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                        1e-6f) == 0);
        TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() ==
                    shape_dispatches);
        TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count() ==
                    simd_completed_dispatches_before +
                    (simd32_available ? exact_cases : 0u));
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fallback_simd_q, 0, fallback_simd_q_host, fallback_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fallback_simd_k, 0, fallback_simd_k_host, fallback_bytes) != 0);
        shape_fallback_mismatches =
            test_compare_float_bits(fallback_q_input, fallback_simd_q_host,
                                    fallback_head_dim).mismatch_count +
            test_compare_float_bits(fallback_k_input, fallback_simd_k_host,
                                    fallback_head_dim).mismatch_count;
        TEST_ASSERT(shape_fallback_mismatches == 0);
        TEST_ASSERT(unsetenv(simd32_env_name) == 0);
        ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
    }
    ds4_gpu_tensor_free(fallback_simd_k);
    ds4_gpu_tensor_free(fallback_simd_q);

    TEST_ASSERT(q_mismatches == 0);
    TEST_ASSERT(k_mismatches == 0);
    TEST_ASSERT(simd_q_mismatches == 0);
    TEST_ASSERT(simd_k_mismatches == 0);
    TEST_ASSERT(malformed_mismatches == 0);
    TEST_ASSERT(rejected_mismatches == 0);
    TEST_ASSERT(shape_fallback_mismatches == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() ==
                simd_encoded_dispatches_before +
                (simd32_available ? exact_cases : 0u));
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count() ==
                simd_completed_dispatches_before +
                (simd32_available ? exact_cases : 0u));
    TEST_ASSERT(simd32_capability_known);
    if (source_mode == TEST_LAGUNA_SOURCE_CURRENT ||
        source_mode == TEST_LAGUNA_SOURCE_BUILTIN) {
        TEST_ASSERT(simd32_available);
        TEST_ASSERT(exact_cases != 0u);
    } else if (source_mode == TEST_LAGUNA_SOURCE_OLD) {
        TEST_ASSERT(!simd32_available);
    }
    TEST_ASSERT(max_ulp == 0);

    /* False-pass guard for graph evidence: a support graph may have advanced
     * the low-level encoded counter before the target call.  A proof that
     * snapshots only after support would see two dispatches and could mistake
     * that movement for one target dispatch; the target-scoped delta below is
     * exactly one after the target batch is ended/waited. */
    uint64_t scoped_support_dispatches = 0;
    uint64_t scoped_completed_dispatches = 0;
    if (simd32_available) {
        const uint64_t q_bytes = (uint64_t)48u * head_dim * sizeof(float);
        const uint64_t k_bytes = (uint64_t)8u * head_dim * sizeof(float);
        ds4_gpu_tensor *support_q = ds4_gpu_tensor_alloc(q_bytes);
        ds4_gpu_tensor *support_k = ds4_gpu_tensor_alloc(k_bytes);
        float *support_q_host = calloc(1, (size_t)q_bytes);
        float *support_k_host = calloc(1, (size_t)k_bytes);
        TEST_ASSERT(support_q && support_k && support_q_host && support_k_host);
        if (support_q && support_k && support_q_host && support_k_host) {
            TEST_ASSERT(ds4_gpu_tensor_write(
                            support_q, 0, support_q_host, q_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            support_k, 0, support_k_host, k_bytes) != 0);
            TEST_ASSERT(setenv(simd32_env_name, "1", 1) == 0);
            ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
            const uint64_t support_route_before =
                ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count();
            TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_tensor(
                            support_k, model_raw, model_size,
                            k_weight_offset, 1, 8, head_dim, 128,
                            37, 262144u, 10000.0f, 1.0f,
                            0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) == 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() ==
                        support_route_before);
            TEST_ASSERT(setenv(simd32_env_name, "not-a-selector", 1) == 0);
            ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
            TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                            support_k, model_raw, model_size,
                            k_weight_offset, 1, 8, head_dim, 128,
                            37, 262144u, 10000.0f, 1.0f,
                            0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) == 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() ==
                        support_route_before);
            TEST_ASSERT(setenv(simd32_env_name, "1", 1) == 0);
            ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
            TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                            support_k, model_raw, model_size,
                            k_weight_offset, 1, 8, head_dim, 128,
                            37, 262144u, 10000.0f, 1.0f,
                            0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_wait_submitted_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() ==
                        support_route_before);
            const uint64_t support_before =
                ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count();
            const uint64_t support_completed_before =
                ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count();
            /* A target encoded in a discarded batch must not be promoted to
             * completion evidence.  The following successful re-encode is
             * isolated in its own command batch. */
            TEST_ASSERT(ds4_gpu_begin_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            support_q, support_k, model_raw, model_size,
                            0, k_weight_offset, 1, 48, 8, head_dim, 64,
                            65533, 8192u, 500000.0f, 1.0f / 32.0f,
                            1.0f, 1.0f, 32.0f, 1.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_discard_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count() ==
                        support_completed_before);
            TEST_ASSERT(ds4_gpu_begin_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            support_q, support_k, model_raw, model_size,
                            0, k_weight_offset, 1, 48, 8, head_dim, 64,
                            65533, 8192u, 500000.0f, 1.0f / 32.0f,
                            1.0f, 1.0f, 32.0f, 1.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_end_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count() ==
                        support_completed_before + 1u);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            support_q, support_k, model_raw, model_size,
                            0, k_weight_offset, 1, 48, 8, head_dim, 64,
                            65533, 8192u, 500000.0f, 1.0f / 32.0f,
                            1.0f, 1.0f, 32.0f, 1.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_wait_submitted_commands() != 0);
            const uint64_t target_before =
                ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count();
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            support_q, support_k, model_raw, model_size,
                            0, k_weight_offset, 1, 48, 8, head_dim, 64,
                            65533, 8192u, 500000.0f, 1.0f / 32.0f,
                            1.0f, 1.0f, 32.0f, 1.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_wait_submitted_commands() != 0);
            const uint64_t after_target =
                ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count();
            TEST_ASSERT(target_before > support_before);
            TEST_ASSERT(after_target - support_before == 4u);
            TEST_ASSERT(after_target - target_before == 1u);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count() ==
                        support_completed_before + 3u);
            scoped_support_dispatches = 4u;
            scoped_completed_dispatches = 3u;
            TEST_ASSERT(unsetenv(simd32_env_name) == 0);
        }
        free(support_k_host);
        free(support_q_host);
        ds4_gpu_tensor_free(support_k);
        ds4_gpu_tensor_free(support_q);
    }
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() ==
                simd_encoded_dispatches_before +
                (simd32_available ? exact_cases : 0u) +
                scoped_support_dispatches);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count() ==
                simd_completed_dispatches_before +
                (simd32_available ? exact_cases : 0u) +
                scoped_completed_dispatches);
    free(model_raw);
    test_restore_env(simd32_env_name, saved_simd32_env);
    ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
}
#endif

#if defined(__APPLE__)
static bool test_metal_laguna_rope_atlas_encode_target(
        ds4_gpu_tensor *const *global_q,
        ds4_gpu_tensor *const *global_k,
        ds4_gpu_tensor *const *swa_q,
        ds4_gpu_tensor *const *swa_k,
        void *model_raw,
        uint64_t model_size,
        uint64_t k_weight_offset) {
    bool ok = true;
    for (uint32_t i = 0; i < 12u; i++) {
        if (!ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                    global_q[i], global_k[i], model_raw, model_size,
                    0, k_weight_offset, 1u, 48u, 8u, 128u, 64u, 0u,
                    8192u, 500000.0f, 1.0f / 32.0f, 1.0f, 1.0f,
                    32.0f, 1.0f, 1e-6f)) {
            ok = false;
        }
    }
    for (uint32_t i = 0; i < 36u; i++) {
        if (!ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                    swa_q[i], swa_k[i], model_raw, model_size,
                    0, k_weight_offset, 1u, 72u, 8u, 128u, 128u, 0u,
                    262144u, 10000.0f, 1.0f, 0.0f, 1.0f,
                    0.0f, 0.0f, 1e-6f)) {
            ok = false;
        }
    }
    return ok;
}

static void test_metal_laguna_rope_atlas_reset_target_inputs(
        ds4_gpu_tensor *const *global_q,
        ds4_gpu_tensor *const *global_k,
        ds4_gpu_tensor *const *swa_q,
        ds4_gpu_tensor *const *swa_k,
        const float *global_q_input,
        const float *global_k_input,
        const float *swa_q_input,
        const float *swa_k_input,
        uint64_t global_q_bytes,
        uint64_t global_k_bytes,
        uint64_t swa_q_bytes,
        uint64_t swa_k_bytes) {
    for (uint32_t i = 0; i < 12u; i++) {
        TEST_ASSERT(ds4_gpu_tensor_write(global_q[i], 0, global_q_input,
                                         global_q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(global_k[i], 0, global_k_input,
                                         global_k_bytes) != 0);
    }
    for (uint32_t i = 0; i < 36u; i++) {
        TEST_ASSERT(ds4_gpu_tensor_write(swa_q[i], 0, swa_q_input,
                                         swa_q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(swa_k[i], 0, swa_k_input,
                                         swa_k_bytes) != 0);
    }
}

static FILE *test_metal_capture_stderr_start(int *saved_fd) {
    if (!saved_fd) return NULL;
    *saved_fd = -1;
    FILE *capture = tmpfile();
    if (!capture) return NULL;
    fflush(stderr);
    *saved_fd = dup(fileno(stderr));
    if (*saved_fd < 0 || dup2(fileno(capture), fileno(stderr)) < 0) {
        if (*saved_fd >= 0) close(*saved_fd);
        *saved_fd = -1;
        fclose(capture);
        return NULL;
    }
    return capture;
}

static char *test_metal_capture_stderr_finish(FILE *capture, int saved_fd) {
    if (!capture || saved_fd < 0) {
        if (capture) fclose(capture);
        if (saved_fd >= 0) close(saved_fd);
        return NULL;
    }
    fflush(stderr);
    const int stderr_fd = fileno(stderr);
    const int restore_ok = dup2(saved_fd, stderr_fd);
    close(saved_fd);
    if (restore_ok < 0 || fseek(capture, 0, SEEK_END) != 0) {
        fclose(capture);
        return NULL;
    }
    const long length = ftell(capture);
    if (length < 0 || fseek(capture, 0, SEEK_SET) != 0) {
        fclose(capture);
        return NULL;
    }
    char *text = malloc((size_t)length + 1u);
    if (!text) {
        fclose(capture);
        return NULL;
    }
    const size_t got = fread(text, 1u, (size_t)length, capture);
    text[got] = '\0';
    fclose(capture);
    return text;
}

static void test_metal_laguna_rope_atlas_production_counts(
        void *model_raw,
        uint64_t model_size,
        uint64_t k_weight_offset,
        const char *atlas_env_name,
        const char *simd_env_name) {
    enum { GLOBAL_CONSUMERS = 12, SWA_CONSUMERS = 36 };
    const uint32_t hd = 128u;
    const char *atlas_trace_env_name =
        "DS4_METAL_LAGUNA_ROPE_ATLAS_TRACE";
    char *saved_atlas_trace_env = test_save_env(atlas_trace_env_name);
    const uint64_t global_q_bytes = (uint64_t)48u * hd * sizeof(float);
    const uint64_t global_k_bytes = (uint64_t)8u * hd * sizeof(float);
    const uint64_t swa_q_bytes = (uint64_t)72u * hd * sizeof(float);
    const uint64_t swa_k_bytes = global_k_bytes;
    float *global_q_input = malloc((size_t)global_q_bytes);
    float *global_k_input = malloc((size_t)global_k_bytes);
    float *swa_q_input = malloc((size_t)swa_q_bytes);
    float *swa_k_input = malloc((size_t)swa_k_bytes);
    float *global_q_ref = malloc((size_t)global_q_bytes);
    float *global_k_ref = malloc((size_t)global_k_bytes);
    float *swa_q_ref = malloc((size_t)swa_q_bytes);
    float *swa_k_ref = malloc((size_t)swa_k_bytes);
    ds4_gpu_tensor *global_q[GLOBAL_CONSUMERS] = {0};
    ds4_gpu_tensor *global_k[GLOBAL_CONSUMERS] = {0};
    ds4_gpu_tensor *swa_q[SWA_CONSUMERS] = {0};
    ds4_gpu_tensor *swa_k[SWA_CONSUMERS] = {0};
    ds4_gpu_tensor *ref_q = NULL;
    ds4_gpu_tensor *ref_k = NULL;
    ds4_gpu_tensor *reg_q = NULL;
    ds4_gpu_tensor *reg_k = NULL;
    TEST_ASSERT(global_q_input && global_k_input && swa_q_input && swa_k_input &&
                global_q_ref && global_k_ref && swa_q_ref && swa_k_ref);
    if (!global_q_input || !global_k_input || !swa_q_input || !swa_k_input ||
        !global_q_ref || !global_k_ref || !swa_q_ref || !swa_k_ref) {
        goto cleanup;
    }
    for (uint64_t i = 0; i < (uint64_t)48u * hd; i++) {
        global_q_input[i] = (float)((int)(i * 17u % 257u) - 128) / 131.0f;
    }
    for (uint64_t i = 0; i < (uint64_t)8u * hd; i++) {
        global_k_input[i] = (float)((int)(i * 19u % 251u) - 125) / 127.0f;
        swa_k_input[i] = global_k_input[i];
    }
    for (uint64_t i = 0; i < (uint64_t)72u * hd; i++) {
        swa_q_input[i] = (float)((int)(i * 23u % 211u) - 105) / 137.0f;
    }

    /* Build one exact legacy reference for each target family. */
    TEST_ASSERT(setenv(atlas_env_name, "0", 1) == 0);
    TEST_ASSERT(setenv(simd_env_name, "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(48u, 8u, hd, 64u) == 0);
    ref_q = ds4_gpu_tensor_alloc(global_q_bytes);
    ref_k = ds4_gpu_tensor_alloc(global_k_bytes);
    TEST_ASSERT(ref_q && ref_k);
    TEST_ASSERT(ds4_gpu_tensor_write(ref_q, 0, global_q_input, global_q_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(ref_k, 0, global_k_input, global_k_bytes) != 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                    ref_q, ref_k, model_raw, model_size, 0, k_weight_offset,
                    1u, 48u, 8u, hd, 64u, 0u, 8192u, 500000.0f,
                    1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f, 1e-6f) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_q, 0, global_q_ref, global_q_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_k, 0, global_k_ref, global_k_bytes) != 0);
    ds4_gpu_tensor_free(ref_q);
    ds4_gpu_tensor_free(ref_k);
    ref_q = NULL;
    ref_k = NULL;
    ref_q = ds4_gpu_tensor_alloc(swa_q_bytes);
    ref_k = ds4_gpu_tensor_alloc(swa_k_bytes);
    TEST_ASSERT(ref_q && ref_k);
    TEST_ASSERT(ds4_gpu_tensor_write(ref_q, 0, swa_q_input, swa_q_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(ref_k, 0, swa_k_input, swa_k_bytes) != 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                    ref_q, ref_k, model_raw, model_size, 0, k_weight_offset,
                    1u, 72u, 8u, hd, 128u, 0u, 262144u, 10000.0f,
                    1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_q, 0, swa_q_ref, swa_q_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_k, 0, swa_k_ref, swa_k_bytes) != 0);
    ds4_gpu_tensor_free(ref_q);
    ds4_gpu_tensor_free(ref_k);
    ref_q = NULL;
    ref_k = NULL;

    /* First global consumer completes in its own CB; the remaining 47
     * consumers span a new CB and must reuse the completion-promoted key. */
    TEST_ASSERT(setenv(atlas_env_name, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(48u, 8u, hd, 64u) == 1);
    for (uint32_t i = 0; i < GLOBAL_CONSUMERS; i++) {
        global_q[i] = ds4_gpu_tensor_alloc(global_q_bytes);
        global_k[i] = ds4_gpu_tensor_alloc(global_k_bytes);
        TEST_ASSERT(global_q[i] && global_k[i]);
        TEST_ASSERT(ds4_gpu_tensor_write(global_q[i], 0, global_q_input,
                                         global_q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(global_k[i], 0, global_k_input,
                                         global_k_bytes) != 0);
    }
    for (uint32_t i = 0; i < SWA_CONSUMERS; i++) {
        swa_q[i] = ds4_gpu_tensor_alloc(swa_q_bytes);
        swa_k[i] = ds4_gpu_tensor_alloc(swa_k_bytes);
        TEST_ASSERT(swa_q[i] && swa_k[i]);
        TEST_ASSERT(ds4_gpu_tensor_write(swa_q[i], 0, swa_q_input,
                                         swa_q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(swa_k[i], 0, swa_k_input,
                                         swa_k_bytes) != 0);
    }
    const uint64_t generated_before =
        ds4_gpu_laguna_rope_atlas_completed_generated_count();
    const uint64_t consumed_before =
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
    const uint64_t family0_before =
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u);
    const uint64_t family1_before =
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u);
    const uint64_t encoded_before =
        ds4_gpu_laguna_rope_atlas_encoded_dispatch_count();
    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                    global_q[0], global_k[0], model_raw, model_size,
                    0, k_weight_offset, 1u, 48u, 8u, hd, 64u, 0u, 8192u,
                    500000.0f, 1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f,
                    1e-6f) != 0);
    TEST_ASSERT(ds4_gpu_end_commands() != 0);
    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    for (uint32_t i = 1; i < GLOBAL_CONSUMERS; i++) {
        TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                        global_q[i], global_k[i], model_raw, model_size,
                        0, k_weight_offset, 1u, 48u, 8u, hd, 64u, 0u,
                        8192u, 500000.0f, 1.0f / 32.0f, 1.0f, 1.0f,
                        32.0f, 1.0f, 1e-6f) != 0);
    }
    for (uint32_t i = 0; i < SWA_CONSUMERS; i++) {
        TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                        swa_q[i], swa_k[i], model_raw, model_size,
                        0, k_weight_offset, 1u, 72u, 8u, hd, 128u, 0u,
                        262144u, 10000.0f, 1.0f, 0.0f, 1.0f,
                        0.0f, 0.0f, 1e-6f) != 0);
    }
    TEST_ASSERT(ds4_gpu_end_commands() != 0);
    for (uint32_t i = 0; i < GLOBAL_CONSUMERS; i++) {
        float *got_q = malloc((size_t)global_q_bytes);
        float *got_k = malloc((size_t)global_k_bytes);
        TEST_ASSERT(got_q && got_k);
        TEST_ASSERT(ds4_gpu_tensor_read(global_q[i], 0, got_q, global_q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(global_k[i], 0, got_k, global_k_bytes) != 0);
        TEST_ASSERT(test_compare_float_bits(global_q_ref, got_q,
                                            (size_t)48u * hd).mismatch_count == 0);
        TEST_ASSERT(test_compare_float_bits(global_k_ref, got_k,
                                            (size_t)8u * hd).mismatch_count == 0);
        free(got_q);
        free(got_k);
    }
    for (uint32_t i = 0; i < SWA_CONSUMERS; i++) {
        float *got_q = malloc((size_t)swa_q_bytes);
        float *got_k = malloc((size_t)swa_k_bytes);
        TEST_ASSERT(got_q && got_k);
        TEST_ASSERT(ds4_gpu_tensor_read(swa_q[i], 0, got_q, swa_q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(swa_k[i], 0, got_k, swa_k_bytes) != 0);
        TEST_ASSERT(test_compare_float_bits(swa_q_ref, got_q,
                                            (size_t)72u * hd).mismatch_count == 0);
        TEST_ASSERT(test_compare_float_bits(swa_k_ref, got_k,
                                            (size_t)8u * hd).mismatch_count == 0);
        free(got_q);
        free(got_k);
    }
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_generated_count() ==
                generated_before + 1u);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() ==
                consumed_before + 48u);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(0u) ==
                family0_before + GLOBAL_CONSUMERS);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(1u) ==
                family1_before + SWA_CONSUMERS);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_encoded_dispatch_count() ==
                encoded_before + 1u);
    fprintf(stderr,
            "ds4-test: Laguna RoPE atlas production target "
            "generation_completed=1 consumers_completed=48 family0_completed=12 "
            "family1_completed=36 cross_cb_reuse=1\n");

    /* A completed target key is reusable even after the previous graph's
     * command buffer ended.  Reinitialize independent tensors so this second
     * unit proves both exact output and a zero-generation reuse, rather than
     * accidentally applying RoPE twice to the first unit's outputs. */
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_target_reuse_ready(1u, 0u) == 1);
    for (uint32_t i = 0; i < GLOBAL_CONSUMERS; i++) {
        TEST_ASSERT(ds4_gpu_tensor_write(global_q[i], 0, global_q_input,
                                         global_q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(global_k[i], 0, global_k_input,
                                         global_k_bytes) != 0);
    }
    for (uint32_t i = 0; i < SWA_CONSUMERS; i++) {
        TEST_ASSERT(ds4_gpu_tensor_write(swa_q[i], 0, swa_q_input,
                                         swa_q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(swa_k[i], 0, swa_k_input,
                                         swa_k_bytes) != 0);
    }
    const uint64_t reuse_generated_before =
        ds4_gpu_laguna_rope_atlas_completed_generated_count();
    const uint64_t reuse_consumed_before =
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
    const uint64_t reuse_family0_before =
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u);
    const uint64_t reuse_family1_before =
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u);
    const uint64_t reuse_encoded_before =
        ds4_gpu_laguna_rope_atlas_encoded_dispatch_count();
    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    for (uint32_t i = 0; i < GLOBAL_CONSUMERS; i++) {
        TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                        global_q[i], global_k[i], model_raw, model_size,
                        0, k_weight_offset, 1u, 48u, 8u, hd, 64u, 0u,
                        8192u, 500000.0f, 1.0f / 32.0f, 1.0f, 1.0f,
                        32.0f, 1.0f, 1e-6f) != 0);
    }
    for (uint32_t i = 0; i < SWA_CONSUMERS; i++) {
        TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                        swa_q[i], swa_k[i], model_raw, model_size,
                        0, k_weight_offset, 1u, 72u, 8u, hd, 128u, 0u,
                        262144u, 10000.0f, 1.0f, 0.0f, 1.0f,
                        0.0f, 0.0f, 1e-6f) != 0);
    }
    TEST_ASSERT(ds4_gpu_end_commands() != 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_generated_count() ==
                reuse_generated_before);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() ==
                reuse_consumed_before + 48u);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(0u) ==
                reuse_family0_before + GLOBAL_CONSUMERS);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(1u) ==
                reuse_family1_before + SWA_CONSUMERS);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_encoded_dispatch_count() ==
                reuse_encoded_before);
    for (uint32_t i = 0; i < GLOBAL_CONSUMERS; i++) {
        float *got_q = malloc((size_t)global_q_bytes);
        float *got_k = malloc((size_t)global_k_bytes);
        TEST_ASSERT(got_q && got_k);
        TEST_ASSERT(ds4_gpu_tensor_read(global_q[i], 0, got_q, global_q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(global_k[i], 0, got_k, global_k_bytes) != 0);
        TEST_ASSERT(test_compare_float_bits(global_q_ref, got_q,
                                            (size_t)48u * hd).mismatch_count == 0);
        TEST_ASSERT(test_compare_float_bits(global_k_ref, got_k,
                                            (size_t)8u * hd).mismatch_count == 0);
        free(got_q);
        free(got_k);
    }
    for (uint32_t i = 0; i < SWA_CONSUMERS; i++) {
        float *got_q = malloc((size_t)swa_q_bytes);
        float *got_k = malloc((size_t)swa_k_bytes);
        TEST_ASSERT(got_q && got_k);
        TEST_ASSERT(ds4_gpu_tensor_read(swa_q[i], 0, got_q, swa_q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(swa_k[i], 0, got_k, swa_k_bytes) != 0);
        TEST_ASSERT(test_compare_float_bits(swa_q_ref, got_q,
                                            (size_t)72u * hd).mismatch_count == 0);
        TEST_ASSERT(test_compare_float_bits(swa_k_ref, got_k,
                                            (size_t)8u * hd).mismatch_count == 0);
        free(got_q);
        free(got_k);
    }
    fprintf(stderr,
            "ds4-test: Laguna RoPE atlas production target "
            "generation_expected=0 generation_completed=0 consumers_completed=48 "
            "family0_completed=12 family1_completed=36 same_key_reuse=1\n");

    /* Exercise identity-scoped completion promotion.  A is committed and
     * left pending; B is encoded with a different key before A is drained.
     * Publishing A must not promote B's cache key.  B is then discarded and
     * must generate again on its next batch. */
    reg_q = ds4_gpu_tensor_alloc((uint64_t)2u * 48u * hd * sizeof(float));
    reg_k = ds4_gpu_tensor_alloc((uint64_t)2u * 8u * hd * sizeof(float));
    TEST_ASSERT(reg_q && reg_k);
    TEST_ASSERT(ds4_gpu_tensor_write(reg_q, 0, global_q_input,
                                     global_q_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(reg_q, global_q_bytes, global_q_input,
                                     global_q_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(reg_k, 0, global_k_input,
                                     global_k_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(reg_k, global_k_bytes, global_k_input,
                                     global_k_bytes) != 0);
    const uint64_t identity_gen_before =
        ds4_gpu_laguna_rope_atlas_completed_generated_count();
    const uint64_t identity_encoded_before =
        ds4_gpu_laguna_rope_atlas_encoded_dispatch_count();
    /* A: one row at a distinct position, committed without waiting. */
    TEST_ASSERT(ds4_gpu_tensor_write(global_q[0], 0, global_q_input,
                                     global_q_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(global_k[0], 0, global_k_input,
                                     global_k_bytes) != 0);
    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                    global_q[0], global_k[0], model_raw, model_size,
                    0, k_weight_offset, 1u, 48u, 8u, hd, 64u, 17u, 8192u,
                    500000.0f,
                    1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f, 1e-6f) != 0);
    TEST_ASSERT(ds4_gpu_flush_commands() != 0);
    TEST_ASSERT(ds4_gpu_diagnostic_pending_command_buffer_count() == 1);
    /* B: a different two-row key is encoded in the replacement CB. */
    TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                    reg_q, reg_k, model_raw, model_size, 0, k_weight_offset,
                    2u, 48u, 8u, hd, 64u, 1u, 8192u, 500000.0f,
                    1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f, 1e-6f) != 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_encoded_dispatch_count() ==
                identity_encoded_before + 2u);
    /* Drain only already-submitted A.  B remains active and uncompleted. */
    TEST_ASSERT(ds4_gpu_wait_submitted_commands() != 0);
    TEST_ASSERT(ds4_gpu_diagnostic_pending_command_buffer_count() == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_generated_count() ==
                identity_gen_before + 1u);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_valid_completed_for_test() == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_target_reuse_ready(2u, 1u) == 0);
    /* Discard B: its encoded generation must not publish completion. */
    TEST_ASSERT(ds4_gpu_discard_commands() != 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_generated_count() ==
                identity_gen_before + 1u);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_target_reuse_ready(2u, 1u) == 0);
    /* Re-encode B after discard; only this successful B completion is the
     * second completed generation in the identity-scoped sequence. */
    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                    reg_q, reg_k, model_raw, model_size, 0, k_weight_offset,
                    2u, 48u, 8u, hd, 64u, 1u, 8192u, 500000.0f,
                    1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f, 1e-6f) != 0);
    TEST_ASSERT(ds4_gpu_end_commands() != 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_generated_count() ==
                identity_gen_before + 2u);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_valid_completed_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_target_reuse_ready(2u, 1u) == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_encoded_dispatch_count() ==
                identity_encoded_before + 3u);

    /* Production-shaped DFlash support order: one independent support
     * generation, six paired Q/K consumers, then six same-key single-K
     * consumers across a flush boundary.  The two route families must all
     * complete and agree bit-for-bit with their stock references. */
    {
        enum { SUPPORT_CONSUMERS = 6 };
        const uint32_t support_tokens = 3u;
        const uint32_t support_pos = 511u;
        const uint64_t support_q_bytes =
            (uint64_t)support_tokens * 72u * hd * sizeof(float);
        const uint64_t support_k_bytes =
            (uint64_t)support_tokens * 8u * hd * sizeof(float);
        float *support_q_input = malloc((size_t)support_q_bytes);
        float *support_k_input = malloc((size_t)support_k_bytes);
        float *support_q_ref = malloc((size_t)support_q_bytes);
        float *support_k_ref = malloc((size_t)support_k_bytes);
        float *support_single_ref = malloc((size_t)support_k_bytes);
        ds4_gpu_tensor *support_pair_q[SUPPORT_CONSUMERS] = {0};
        ds4_gpu_tensor *support_pair_k[SUPPORT_CONSUMERS] = {0};
        ds4_gpu_tensor *support_single_k[SUPPORT_CONSUMERS] = {0};
        TEST_ASSERT(support_q_input && support_k_input && support_q_ref &&
                    support_k_ref && support_single_ref);
        if (support_q_input && support_k_input && support_q_ref &&
            support_k_ref && support_single_ref) {
            for (uint64_t i = 0; i < (uint64_t)support_tokens * 72u * hd; i++) {
                support_q_input[i] =
                    (float)((int)(i * 43u % 223u) - 111) / 139.0f;
            }
            for (uint64_t i = 0; i < (uint64_t)support_tokens * 8u * hd; i++) {
                support_k_input[i] =
                    (float)((int)(i * 47u % 227u) - 113) / 151.0f;
            }

            TEST_ASSERT(setenv(atlas_env_name, "0", 1) == 0);
            TEST_ASSERT(setenv(simd_env_name, "0", 1) == 0);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(72u, 8u, hd, 128u) == 0);
            ds4_gpu_tensor *support_ref_q =
                ds4_gpu_tensor_alloc(support_q_bytes);
            ds4_gpu_tensor *support_ref_k =
                ds4_gpu_tensor_alloc(support_k_bytes);
            ds4_gpu_tensor *support_ref_single =
                ds4_gpu_tensor_alloc(support_k_bytes);
            TEST_ASSERT(support_ref_q && support_ref_k && support_ref_single);
            if (support_ref_q && support_ref_k && support_ref_single) {
                TEST_ASSERT(ds4_gpu_tensor_write(
                                support_ref_q, 0, support_q_input,
                                support_q_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_write(
                                support_ref_k, 0, support_k_input,
                                support_k_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_write(
                                support_ref_single, 0, support_k_input,
                                support_k_bytes) != 0);
                TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                                support_ref_q, support_ref_k, model_raw,
                                model_size, 0, k_weight_offset,
                                support_tokens, 72u, 8u, hd, 128u,
                                support_pos, 262144u, 500000.0f, 1.0f,
                                0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) != 0);
                TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                                support_ref_single, model_raw, model_size,
                                k_weight_offset, support_tokens, 8u, hd,
                                128u, support_pos, 262144u, 500000.0f, 1.0f,
                                0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                                support_ref_q, 0, support_q_ref,
                                support_q_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                                support_ref_k, 0, support_k_ref,
                                support_k_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                                support_ref_single, 0, support_single_ref,
                                support_k_bytes) != 0);
            }
            ds4_gpu_tensor_free(support_ref_single);
            ds4_gpu_tensor_free(support_ref_k);
            ds4_gpu_tensor_free(support_ref_q);

            TEST_ASSERT(setenv(atlas_env_name, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(72u, 8u, hd, 128u) == 1);
            for (uint32_t i = 0; i < SUPPORT_CONSUMERS; i++) {
                support_pair_q[i] = ds4_gpu_tensor_alloc(support_q_bytes);
                support_pair_k[i] = ds4_gpu_tensor_alloc(support_k_bytes);
                support_single_k[i] = ds4_gpu_tensor_alloc(support_k_bytes);
                TEST_ASSERT(support_pair_q[i] && support_pair_k[i] &&
                            support_single_k[i]);
                if (support_pair_q[i] && support_pair_k[i] &&
                    support_single_k[i]) {
                    TEST_ASSERT(ds4_gpu_tensor_write(
                                    support_pair_q[i], 0, support_q_input,
                                    support_q_bytes) != 0);
                    TEST_ASSERT(ds4_gpu_tensor_write(
                                    support_pair_k[i], 0, support_k_input,
                                    support_k_bytes) != 0);
                    TEST_ASSERT(ds4_gpu_tensor_write(
                                    support_single_k[i], 0, support_k_input,
                                    support_k_bytes) != 0);
                }
            }
            const uint64_t support_prod_gen_before =
                ds4_gpu_laguna_rope_support_atlas_completed_generated_count();
            const uint64_t support_prod_cons_before =
                ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count();
            const uint64_t support_prod_encoded_gen_before =
                ds4_gpu_laguna_rope_support_atlas_encoded_dispatch_count();
            TEST_ASSERT(ds4_gpu_begin_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            support_pair_q[0], support_pair_k[0], model_raw,
                            model_size, 0, k_weight_offset, support_tokens,
                            72u, 8u, hd, 128u, support_pos, 262144u,
                            500000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                            1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_flush_commands() != 0);
            for (uint32_t i = 1; i < SUPPORT_CONSUMERS; i++) {
                TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                                support_pair_q[i], support_pair_k[i],
                                model_raw, model_size, 0, k_weight_offset,
                                support_tokens, 72u, 8u, hd, 128u,
                                support_pos, 262144u, 500000.0f, 1.0f,
                                0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) != 0);
            }
            for (uint32_t i = 0; i < SUPPORT_CONSUMERS; i++) {
                TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                                support_single_k[i], model_raw, model_size,
                                k_weight_offset, support_tokens, 8u, hd,
                                128u, support_pos, 262144u, 500000.0f,
                                1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                1e-6f) != 0);
            }
            TEST_ASSERT(ds4_gpu_end_commands() != 0);
            TEST_ASSERT(ds4_gpu_wait_submitted_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_completed_generated_count() ==
                        support_prod_gen_before + 1u);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count() ==
                        support_prod_cons_before + 12u);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_encoded_dispatch_count() ==
                        support_prod_encoded_gen_before + 1u);
            for (uint32_t i = 0; i < SUPPORT_CONSUMERS; i++) {
                float *got_q = malloc((size_t)support_q_bytes);
                float *got_k = malloc((size_t)support_k_bytes);
                TEST_ASSERT(got_q && got_k);
                if (got_q && got_k) {
                    TEST_ASSERT(ds4_gpu_tensor_read(
                                    support_pair_q[i], 0, got_q,
                                    support_q_bytes) != 0);
                    TEST_ASSERT(ds4_gpu_tensor_read(
                                    support_pair_k[i], 0, got_k,
                                    support_k_bytes) != 0);
                    TEST_ASSERT(test_compare_float_bits(
                                    support_q_ref, got_q,
                                    (size_t)support_tokens * 72u * hd
                                ).mismatch_count == 0);
                    TEST_ASSERT(test_compare_float_bits(
                                    support_k_ref, got_k,
                                    (size_t)support_tokens * 8u * hd
                                ).mismatch_count == 0);
                }
                free(got_k);
                free(got_q);
                got_k = malloc((size_t)support_k_bytes);
                TEST_ASSERT(got_k != NULL);
                if (got_k) {
                    TEST_ASSERT(ds4_gpu_tensor_read(
                                    support_single_k[i], 0, got_k,
                                    support_k_bytes) != 0);
                    TEST_ASSERT(test_compare_float_bits(
                                    support_single_ref, got_k,
                                    (size_t)support_tokens * 8u * hd
                                ).mismatch_count == 0);
                }
                free(got_k);
            }
            fprintf(stderr,
                    "ds4-test: Laguna RoPE atlas support production "
                    "generation_completed=1 paired_consumers=6 "
                    "single_consumers=6 cross_flush=1\n");
        }
        for (uint32_t i = 0; i < SUPPORT_CONSUMERS; i++) {
            ds4_gpu_tensor_free(support_single_k[i]);
            ds4_gpu_tensor_free(support_pair_k[i]);
            ds4_gpu_tensor_free(support_pair_q[i]);
        }
        free(support_single_ref);
        free(support_k_ref);
        free(support_q_ref);
        free(support_k_input);
        free(support_q_input);
    }

    /* Exercise the actual deferred graph evidence owner, including storage
     * before its target consumers encode and completion after the owning
     * command buffer waits.  The test hook is compiled only into this test
     * binary; production ds4.o has no corresponding symbol. */
    {
        TEST_ASSERT(setenv(atlas_trace_env_name, "0", 1) == 0);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(72u, 8u, hd, 64u) == 1);
        test_metal_laguna_rope_atlas_reset_target_inputs(
            global_q, global_k, swa_q, swa_k,
            global_q_input, global_k_input, swa_q_input, swa_k_input,
            global_q_bytes, global_k_bytes, swa_q_bytes, swa_k_bytes);
        /* Warm a completed key so the first deferred unit has expected=0. */
        TEST_ASSERT(ds4_gpu_begin_commands() != 0);
        TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                        global_q[0], global_k[0], model_raw, model_size,
                        0, k_weight_offset, 1u, 48u, 8u, hd, 64u, 0u,
                        8192u, 500000.0f, 1.0f / 32.0f, 1.0f, 1.0f,
                        32.0f, 1.0f, 1e-6f) != 0);
        TEST_ASSERT(ds4_gpu_end_commands() != 0);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_target_reuse_ready(1u, 0u) == 1);
        test_metal_laguna_rope_atlas_reset_target_inputs(
            global_q, global_k, swa_q, swa_k,
            global_q_input, global_k_input, swa_q_input, swa_k_input,
            global_q_bytes, global_k_bytes, swa_q_bytes, swa_k_bytes);
        const uint64_t deferred0_generated_before =
            ds4_gpu_laguna_rope_atlas_completed_generated_count();
        const uint64_t deferred0_consumed_before =
            ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
        const uint64_t deferred0_family0_before =
            ds4_gpu_laguna_rope_atlas_completed_family_count(0u);
        const uint64_t deferred0_family1_before =
            ds4_gpu_laguna_rope_atlas_completed_family_count(1u);
        TEST_ASSERT(ds4_gpu_begin_commands() != 0);
        TEST_ASSERT(ds4_laguna_test_rope_atlas_deferred_snapshot(
                        deferred0_generated_before, 0u,
                        deferred0_consumed_before,
                        deferred0_family0_before,
                        deferred0_family1_before, 1u) != 0);
        TEST_ASSERT(test_metal_laguna_rope_atlas_encode_target(
                        global_q, global_k, swa_q, swa_k,
                        model_raw, model_size, k_weight_offset));
        TEST_ASSERT(ds4_gpu_end_commands() != 0);
        int saved_stderr = -1;
        FILE *capture = test_metal_capture_stderr_start(&saved_stderr);
        TEST_ASSERT(capture != NULL);
        TEST_ASSERT(ds4_laguna_test_rope_atlas_deferred_complete() == 1);
        char *trace_off_evidence =
            test_metal_capture_stderr_finish(capture, saved_stderr);
        TEST_ASSERT(trace_off_evidence != NULL);
        if (trace_off_evidence) {
            TEST_ASSERT(strstr(trace_off_evidence, "waited_target") == NULL);
        }
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_generated_count() ==
                    deferred0_generated_before);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() ==
                    deferred0_consumed_before + 48u);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(0u) ==
                    deferred0_family0_before + 12u);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(1u) ==
                    deferred0_family1_before + 36u);
        free(trace_off_evidence);

        /* Reset validity while retaining the strict plan, forcing a fresh
         * generation whose expected=1 is stored before any target work. */
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(72u, 8u, hd, 64u) == 1);
        TEST_ASSERT(setenv(atlas_trace_env_name, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(72u, 8u, hd, 64u) == 1);
        test_metal_laguna_rope_atlas_reset_target_inputs(
            global_q, global_k, swa_q, swa_k,
            global_q_input, global_k_input, swa_q_input, swa_k_input,
            global_q_bytes, global_k_bytes, swa_q_bytes, swa_k_bytes);
        const uint64_t deferred1_generated_before =
            ds4_gpu_laguna_rope_atlas_completed_generated_count();
        const uint64_t deferred1_consumed_before =
            ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
        const uint64_t deferred1_family0_before =
            ds4_gpu_laguna_rope_atlas_completed_family_count(0u);
        const uint64_t deferred1_family1_before =
            ds4_gpu_laguna_rope_atlas_completed_family_count(1u);
        TEST_ASSERT(ds4_gpu_begin_commands() != 0);
        TEST_ASSERT(ds4_laguna_test_rope_atlas_deferred_snapshot(
                        deferred1_generated_before, 1u,
                        deferred1_consumed_before,
                        deferred1_family0_before,
                        deferred1_family1_before, 1u) != 0);
        TEST_ASSERT(test_metal_laguna_rope_atlas_encode_target(
                        global_q, global_k, swa_q, swa_k,
                        model_raw, model_size, k_weight_offset));
        TEST_ASSERT(ds4_gpu_end_commands() != 0);
        saved_stderr = -1;
        capture = test_metal_capture_stderr_start(&saved_stderr);
        TEST_ASSERT(capture != NULL);
        TEST_ASSERT(ds4_laguna_test_rope_atlas_deferred_complete() == 1);
        char *trace_on_evidence =
            test_metal_capture_stderr_finish(capture, saved_stderr);
        TEST_ASSERT(trace_on_evidence != NULL);
        if (trace_on_evidence) {
            TEST_ASSERT(strstr(trace_on_evidence, "waited_target") != NULL);
            TEST_ASSERT(strstr(trace_on_evidence,
                               "generation_expected=1") != NULL);
        }
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_generated_count() ==
                    deferred1_generated_before + 1u);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() ==
                    deferred1_consumed_before + 48u);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(0u) ==
                    deferred1_family0_before + 12u);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(1u) ==
                    deferred1_family1_before + 36u);
        free(trace_on_evidence);

        /* A completed same-key reuse has zero generation delta.  Deliberately
         * storing expected=1 must fail and emit the unconditional mismatch. */
        test_metal_laguna_rope_atlas_reset_target_inputs(
            global_q, global_k, swa_q, swa_k,
            global_q_input, global_k_input, swa_q_input, swa_k_input,
            global_q_bytes, global_k_bytes, swa_q_bytes, swa_k_bytes);
        const uint64_t wrong_generated_before =
            ds4_gpu_laguna_rope_atlas_completed_generated_count();
        const uint64_t wrong_consumed_before =
            ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
        const uint64_t wrong_family0_before =
            ds4_gpu_laguna_rope_atlas_completed_family_count(0u);
        const uint64_t wrong_family1_before =
            ds4_gpu_laguna_rope_atlas_completed_family_count(1u);
        TEST_ASSERT(ds4_gpu_begin_commands() != 0);
        TEST_ASSERT(ds4_laguna_test_rope_atlas_deferred_snapshot(
                        wrong_generated_before, 1u,
                        wrong_consumed_before,
                        wrong_family0_before,
                        wrong_family1_before, 1u) != 0);
        TEST_ASSERT(test_metal_laguna_rope_atlas_encode_target(
                        global_q, global_k, swa_q, swa_k,
                        model_raw, model_size, k_weight_offset));
        TEST_ASSERT(ds4_gpu_end_commands() != 0);
        saved_stderr = -1;
        capture = test_metal_capture_stderr_start(&saved_stderr);
        TEST_ASSERT(capture != NULL);
        TEST_ASSERT(ds4_laguna_test_rope_atlas_deferred_complete() == 0);
        char *mismatch_evidence =
            test_metal_capture_stderr_finish(capture, saved_stderr);
        TEST_ASSERT(mismatch_evidence != NULL);
        if (mismatch_evidence) {
            TEST_ASSERT(strstr(mismatch_evidence, "target proof failed") != NULL);
        }
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_generated_count() ==
                    wrong_generated_before);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() ==
                    wrong_consumed_before + 48u);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(0u) ==
                    wrong_family0_before + 12u);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(1u) ==
                    wrong_family1_before + 36u);
        free(mismatch_evidence);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
    }

cleanup:
    ds4_gpu_tensor_free(reg_k);
    ds4_gpu_tensor_free(reg_q);
    for (uint32_t i = 0; i < SWA_CONSUMERS; i++) {
        ds4_gpu_tensor_free(swa_k[i]);
        ds4_gpu_tensor_free(swa_q[i]);
    }
    for (uint32_t i = 0; i < GLOBAL_CONSUMERS; i++) {
        ds4_gpu_tensor_free(global_k[i]);
        ds4_gpu_tensor_free(global_q[i]);
    }
    ds4_gpu_tensor_free(ref_k);
    ds4_gpu_tensor_free(ref_q);
    free(swa_k_ref);
    free(swa_q_ref);
    free(global_k_ref);
    free(global_q_ref);
    free(swa_k_input);
    free(swa_q_input);
    free(global_k_input);
    free(global_q_input);
    test_restore_env(atlas_trace_env_name, saved_atlas_trace_env);
}

static void test_metal_laguna_rope_atlas_exact(void) {
    typedef struct {
        uint32_t n_tokens;
        uint32_t n_q_head;
        uint32_t n_k_head;
        uint32_t pos0;
        uint32_t n_rot;
    } atlas_case;
    static const atlas_case cases[] = {
        { 1u, 72u, 8u,    0u,  64u },
        { 3u, 72u, 8u,    1u,  64u },
        {17u, 48u, 8u,  511u,  64u },
        { 2u, 72u, 8u,  512u,  64u },
        { 4u, 48u, 8u, 8191u,  64u },
        { 2u, 72u, 8u, 8192u,  64u },
        {17u, 48u, 8u, 65533u, 128u },
        { 2u, 72u, 8u,  8192u, 128u },
    };
    const char *atlas_env_name = "DS4_METAL_LAGUNA_ROPE_ATLAS";
    const char *simd_env_name = "DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32";
    char *saved_atlas_env = test_save_env(atlas_env_name);
    char *saved_simd_env = test_save_env(simd_env_name);
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t k_weight_offset = page;
    const uint64_t model_size = 2u * page;
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_size) == 0);
    if (!model_raw) goto cleanup;
    const test_laguna_source_mode source_mode =
        test_laguna_source_mode_from_env();
    TEST_ASSERT(source_mode != TEST_LAGUNA_SOURCE_INVALID);
    memset(model_raw, 0, (size_t)model_size);
    float *q_weight = model_raw;
    float *k_weight = (float *)((uint8_t *)model_raw + k_weight_offset);
    for (uint32_t i = 0; i < 128u; i++) {
        q_weight[i] = 0.75f + (float)((i * 17u + 3u) % 29u) / 64.0f;
        k_weight[i] = 0.625f + (float)((i * 19u + 5u) % 31u) / 56.0f;
    }
    /* Exercise the pre-init cleanup boundary: a frozen disabled plan and a
     * malformed plan must not poison the corrected environment selected for
     * the subsequent Metal initialization. */
    TEST_ASSERT(setenv(atlas_env_name, "0", 1) == 0);
    TEST_ASSERT(setenv(simd_env_name, "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(72u, 8u, 128u, 64u) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                    48u, 8u, 128u, 64u) == 0);
    TEST_ASSERT(setenv(atlas_env_name, "malformed", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(72u, 8u, 128u, 64u) < 0);
    TEST_ASSERT(setenv(simd_env_name, "malformed", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                    48u, 8u, 128u, 64u) < 0);
    ds4_gpu_cleanup();
    TEST_ASSERT(setenv(atlas_env_name, "0", 1) == 0);
    TEST_ASSERT(setenv(simd_env_name, "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(72u, 8u, 128u, 64u) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                    48u, 8u, 128u, 64u) == 0);
    const bool metal_initialized = ds4_gpu_init() != 0;
    if (!metal_initialized) {
        /* Missing/unreadable explicit source is an exclusive, fresh-process
         * failure.  No nonempty override is accepted merely because it is
         * present in the environment. */
        TEST_ASSERT(source_mode == TEST_LAGUNA_SOURCE_MISSING);
        goto cleanup;
    }
    TEST_ASSERT(metal_initialized);
    TEST_ASSERT(source_mode != TEST_LAGUNA_SOURCE_MISSING);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_size) != 0);
    ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
    TEST_ASSERT(setenv(simd_env_name, "0", 1) == 0);
    ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();

    TEST_ASSERT(unsetenv(atlas_env_name) == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_env_mode() == 0);
    TEST_ASSERT(setenv(atlas_env_name, "", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_env_mode() == 0);
    TEST_ASSERT(setenv(atlas_env_name, "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_env_mode() == 0);
    static const char *const malformed[] = {
        "true", "yes", "on", " 1", "01", "0 ", "-1",
        "1 ", "\t1", "\n1", "00", "false", "TRUE",
    };
    for (size_t i = 0; i < sizeof(malformed) / sizeof(malformed[0]); i++) {
        TEST_ASSERT(setenv(atlas_env_name, malformed[i], 1) == 0);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_env_mode() < 0);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(
                        72u, 8u, 128u, 64u) < 0);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_mode_cached() == -1);
    }
    TEST_ASSERT(setenv(atlas_env_name, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
    const int atlas_preflight = ds4_gpu_laguna_rope_atlas_preflight(
        72u, 8u, 128u, 64u);
    if (atlas_preflight != 1) {
        /* An intentional old-source override must fail closed: it is useful
         * evidence for the compatibility screen, but cannot certify atlas
         * exactness because the atlas PSOs are absent by construction. */
        TEST_ASSERT(source_mode == TEST_LAGUNA_SOURCE_OLD);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_encoded_dispatch_count() == 0u);
        goto cleanup;
    }
    TEST_ASSERT(source_mode == TEST_LAGUNA_SOURCE_BUILTIN ||
                source_mode == TEST_LAGUNA_SOURCE_CURRENT);
    /* Support-first combined selection must not pass the K-only 8/0 shape
     * through the paired SIMD32 validator.  Atlas preflight freezes the
     * canonical Laguna 48/8 SIMD plan before validating this single route. */
    TEST_ASSERT(setenv(simd_env_name, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(
                    8u, 0u, 128u, 128u) == 1);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached() == 1);
    TEST_ASSERT(setenv(simd_env_name, "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(
                    72u, 8u, 128u, 64u) == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(
                    7u, 3u, 128u, 64u) < 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_mode_cached() == -1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(
                    72u, 8u, 128u, 64u) == 1);

    /* Atlas-only production planning must select the paired Q/K route for
     * every graph batch shape.  This pure seam covers ordinary prefill,
     * feature capture, verifier/decode-row, and GPU-draft routing without
     * depending on low-level dispatch counters. */
    TEST_ASSERT(ds4_laguna_graph_prefill_qk_norm_rope_paired_route(
                    0, 1, 0, 0, 0, 0) == 1);
    TEST_ASSERT(ds4_laguna_graph_prefill_qk_norm_rope_paired_route(
                    0, 1, 0, 0, 1, 0) == 1);
    TEST_ASSERT(ds4_laguna_graph_prefill_qk_norm_rope_paired_route(
                    0, 1, 1, 0, 1, 0) == 1);
    TEST_ASSERT(ds4_laguna_graph_prefill_qk_norm_rope_paired_route(
                    0, 1, 1, 1, 0, 0) == 1);
    TEST_ASSERT(ds4_laguna_graph_prefill_qk_norm_rope_paired_route(
                    0, 0, 0, 0, 0, 0) == 0);

    /* Exercise the standalone GPU generator before any consumer has used the
     * atlas.  The first production-shaped Q/K route below must reuse this
     * exact `(tokens=1,pos0=0)` generation rather than dispatching twice. */
    const uint64_t standalone_generation_before =
        ds4_gpu_laguna_rope_atlas_encoded_dispatch_count();
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_generate(1u, 0u) != 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_encoded_dispatch_count() ==
                standalone_generation_before + 1u);
    /* Reset is refused while an encoded generation is active/pending, and
     * succeeds only after the discard terminal boundary. */
    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_generate(2u, 1u) != 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 0);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 0);
    TEST_ASSERT(ds4_gpu_discard_commands() != 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
    TEST_ASSERT(setenv(simd_env_name, "0", 1) == 0);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(
                    72u, 8u, 128u, 64u) == 1);

    size_t pair_mismatches = 0;
    size_t single_mismatches = 0;
    size_t simd_mismatches = 0;
    size_t malformed_mismatches = 0;
    uint32_t max_ulp = 0;
    bool simd_available = false;
    bool simd_known = false;
    const uint64_t atlas_generation_before =
        ds4_gpu_laguna_rope_atlas_encoded_dispatch_count();

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const atlas_case *c = &cases[ci];
        const bool global = c->n_rot == 64u;
        const float freq_base = global ? 500000.0f : 10000.0f;
        const float freq_scale = global ? 1.0f / 32.0f : 1.0f;
        const float ext_factor = global ? 1.0f : 0.0f;
        const uint32_t rope_ctx = global ? 8192u : 262144u;
        const float beta_fast = global ? 32.0f : 0.0f;
        const float beta_slow = global ? 1.0f : 0.0f;
        const uint64_t q_values =
            (uint64_t)c->n_tokens * c->n_q_head * 128u;
        const uint64_t k_values =
            (uint64_t)c->n_tokens * c->n_k_head * 128u;
        const uint64_t q_bytes = q_values * sizeof(float);
        const uint64_t k_bytes = k_values * sizeof(float);
        ds4_gpu_tensor *legacy_q = ds4_gpu_tensor_alloc(q_bytes);
        ds4_gpu_tensor *legacy_k = ds4_gpu_tensor_alloc(k_bytes);
        ds4_gpu_tensor *atlas_q = ds4_gpu_tensor_alloc(q_bytes);
        ds4_gpu_tensor *atlas_k = ds4_gpu_tensor_alloc(k_bytes);
        ds4_gpu_tensor *single_q = ds4_gpu_tensor_alloc(q_bytes);
        ds4_gpu_tensor *single_k = ds4_gpu_tensor_alloc(k_bytes);
        ds4_gpu_tensor *simd_q = ds4_gpu_tensor_alloc(q_bytes);
        ds4_gpu_tensor *simd_k = ds4_gpu_tensor_alloc(k_bytes);
        float *q_input = malloc((size_t)q_bytes);
        float *k_input = malloc((size_t)k_bytes);
        float *legacy_q_host = malloc((size_t)q_bytes);
        float *legacy_k_host = malloc((size_t)k_bytes);
        float *atlas_q_host = malloc((size_t)q_bytes);
        float *atlas_k_host = malloc((size_t)k_bytes);
        float *single_q_host = malloc((size_t)q_bytes);
        float *single_k_host = malloc((size_t)k_bytes);
        float *simd_q_host = malloc((size_t)q_bytes);
        float *simd_k_host = malloc((size_t)k_bytes);
        TEST_ASSERT(legacy_q && legacy_k && atlas_q && atlas_k &&
                    single_q && single_k && simd_q && simd_k &&
                    q_input && k_input && legacy_q_host && legacy_k_host &&
                    atlas_q_host && atlas_k_host && single_q_host &&
                    single_k_host && simd_q_host && simd_k_host);
        if (!legacy_q || !legacy_k || !atlas_q || !atlas_k ||
            !single_q || !single_k || !simd_q || !simd_k ||
            !q_input || !k_input || !legacy_q_host || !legacy_k_host ||
            !atlas_q_host || !atlas_k_host || !single_q_host ||
            !single_k_host || !simd_q_host || !simd_k_host) {
            free(simd_k_host); free(simd_q_host); free(single_k_host);
            free(single_q_host); free(atlas_k_host); free(atlas_q_host);
            free(legacy_k_host); free(legacy_q_host); free(k_input); free(q_input);
            ds4_gpu_tensor_free(simd_k); ds4_gpu_tensor_free(simd_q);
            ds4_gpu_tensor_free(single_k); ds4_gpu_tensor_free(single_q);
            ds4_gpu_tensor_free(atlas_k); ds4_gpu_tensor_free(atlas_q);
            ds4_gpu_tensor_free(legacy_k); ds4_gpu_tensor_free(legacy_q);
            continue;
        }
        for (uint64_t i = 0; i < q_values; i++) {
            const int v = (int)((i * 37u + (i >> 3u) * 11u + ci * 13u + 7u) % 211u) - 105;
            q_input[i] = (float)v / 137.0f;
        }
        for (uint64_t i = 0; i < k_values; i++) {
            const int v = (int)((i * 41u + (i >> 2u) * 17u + ci * 23u + 5u) % 199u) - 99;
            k_input[i] = (float)v / 149.0f;
        }

        TEST_ASSERT(setenv(atlas_env_name, "0", 1) == 0);
        ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(
                        c->n_q_head, c->n_k_head, 128u, c->n_rot) == 0);
        TEST_ASSERT(setenv(simd_env_name, "0", 1) == 0);
        ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
        TEST_ASSERT(ds4_gpu_tensor_write(legacy_q, 0, q_input, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(legacy_k, 0, k_input, k_bytes) != 0);
        TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                        legacy_q, legacy_k, model_raw, model_size,
                        0, k_weight_offset, c->n_tokens, c->n_q_head,
                        c->n_k_head, 128u, c->n_rot, c->pos0, rope_ctx,
                        freq_base, freq_scale, ext_factor, 1.0f,
                        beta_fast, beta_slow, 1e-6f) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(legacy_q, 0, legacy_q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(legacy_k, 0, legacy_k_host, k_bytes) != 0);

        TEST_ASSERT(setenv(atlas_env_name, "1", 1) == 0);
        ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(
                        c->n_q_head, c->n_k_head, 128u, c->n_rot) == 1);
        TEST_ASSERT(ds4_gpu_tensor_write(atlas_q, 0, q_input, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(atlas_k, 0, k_input, k_bytes) != 0);
        TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                        atlas_q, atlas_k, model_raw, model_size,
                        0, k_weight_offset, c->n_tokens, c->n_q_head,
                        c->n_k_head, 128u, c->n_rot, c->pos0, rope_ctx,
                        freq_base, freq_scale, ext_factor, 1.0f,
                        beta_fast, beta_slow, 1e-6f) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(atlas_q, 0, atlas_q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(atlas_k, 0, atlas_k_host, k_bytes) != 0);
        test_float_compare_stats q_stats = test_compare_float_bits(
            legacy_q_host, atlas_q_host, (size_t)q_values);
        test_float_compare_stats k_stats = test_compare_float_bits(
            legacy_k_host, atlas_k_host, (size_t)k_values);
        pair_mismatches += q_stats.mismatch_count + k_stats.mismatch_count;
        if (q_stats.max_ulp > max_ulp) max_ulp = q_stats.max_ulp;
        if (k_stats.max_ulp > max_ulp) max_ulp = k_stats.max_ulp;

        /* Standalone generation followed by two single-tensor consumers proves
         * that the atlas is useful independently of paired Q/K dispatch. */
        const uint64_t generation_before_standalone =
            ds4_gpu_laguna_rope_atlas_encoded_dispatch_count();
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_generate(c->n_tokens, c->pos0) != 0);
        TEST_ASSERT(ds4_gpu_laguna_rope_atlas_encoded_dispatch_count() ==
                    generation_before_standalone);
        TEST_ASSERT(ds4_gpu_tensor_write(single_q, 0, q_input, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(single_k, 0, k_input, k_bytes) != 0);
        TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_tensor(
                        single_q, model_raw, model_size, 0, c->n_tokens,
                        c->n_q_head, 128u, c->n_rot, c->pos0, rope_ctx,
                        freq_base, freq_scale, ext_factor, 1.0f,
                        beta_fast, beta_slow, 1e-6f) != 0);
        TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_tensor(
                        single_k, model_raw, model_size, k_weight_offset,
                        c->n_tokens, c->n_k_head, 128u, c->n_rot, c->pos0,
                        rope_ctx, freq_base, freq_scale, ext_factor, 1.0f,
                        beta_fast, beta_slow, 1e-6f) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(single_q, 0, single_q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(single_k, 0, single_k_host, k_bytes) != 0);
        q_stats = test_compare_float_bits(legacy_q_host, single_q_host, (size_t)q_values);
        k_stats = test_compare_float_bits(legacy_k_host, single_k_host, (size_t)k_values);
        single_mismatches += q_stats.mismatch_count + k_stats.mismatch_count;

        /* The atlas and reviewed SIMD32 experiment must compose. */
        TEST_ASSERT(ds4_gpu_tensor_write(simd_q, 0, q_input, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(simd_k, 0, k_input, k_bytes) != 0);
        TEST_ASSERT(setenv(simd_env_name, "1", 1) == 0);
        ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
        TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                        c->n_q_head, c->n_k_head, 128u, c->n_rot) == 1);
        const int simd_ok = ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
            simd_q, simd_k, model_raw, model_size,
            0, k_weight_offset, c->n_tokens, c->n_q_head, c->n_k_head,
            128u, c->n_rot, c->pos0, rope_ctx, freq_base, freq_scale,
            ext_factor, 1.0f, beta_fast, beta_slow, 1e-6f);
        if (!simd_known) {
            simd_known = true;
            simd_available = simd_ok != 0;
        }
        TEST_ASSERT((simd_ok != 0) == simd_available);
        if (simd_ok) {
            TEST_ASSERT(ds4_gpu_tensor_read(simd_q, 0, simd_q_host, q_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(simd_k, 0, simd_k_host, k_bytes) != 0);
            q_stats = test_compare_float_bits(legacy_q_host, simd_q_host, (size_t)q_values);
            k_stats = test_compare_float_bits(legacy_k_host, simd_k_host, (size_t)k_values);
            simd_mismatches += q_stats.mismatch_count + k_stats.mismatch_count;
        }
        TEST_ASSERT(setenv(simd_env_name, "0", 1) == 0);
        ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();

        if (ci == 0u) {
            const uint64_t generation_before_shape =
                ds4_gpu_laguna_rope_atlas_encoded_dispatch_count();
            TEST_ASSERT(ds4_gpu_tensor_write(atlas_q, 0, q_input, q_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(atlas_k, 0, k_input, k_bytes) != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            atlas_q, atlas_k, model_raw, model_size,
                            0, k_weight_offset, c->n_tokens, 7u, 3u,
                            128u, 64u, c->pos0, 8192u, 500000.0f,
                            1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f,
                            1e-6f) == 0);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_encoded_dispatch_count() ==
                        generation_before_shape);
            TEST_ASSERT(ds4_gpu_tensor_read(atlas_q, 0, atlas_q_host, q_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(atlas_k, 0, atlas_k_host, k_bytes) != 0);
            malformed_mismatches += test_compare_float_bits(
                q_input, atlas_q_host, (size_t)q_values).mismatch_count;
            malformed_mismatches += test_compare_float_bits(
                k_input, atlas_k_host, (size_t)k_values).mismatch_count;

            const uint64_t generation_before_malformed =
                ds4_gpu_laguna_rope_atlas_encoded_dispatch_count();
            TEST_ASSERT(setenv(atlas_env_name, "not-a-selector", 1) == 0);
            /* Atlas mode is a graph/preflight plan, so refresh the plan at the
             * fail-closed boundary instead of relying on per-layer getenv. */
            ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(
                            c->n_q_head, c->n_k_head, 128u, c->n_rot) < 0);
            TEST_ASSERT(ds4_gpu_tensor_write(atlas_q, 0, q_input, q_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(atlas_k, 0, k_input, k_bytes) != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            atlas_q, atlas_k, model_raw, model_size,
                            0, k_weight_offset, c->n_tokens, c->n_q_head,
                            c->n_k_head, 128u, c->n_rot, c->pos0, rope_ctx,
                            freq_base, freq_scale, ext_factor, 1.0f,
                            beta_fast, beta_slow, 1e-6f) == 0);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_encoded_dispatch_count() ==
                        generation_before_malformed);
            TEST_ASSERT(ds4_gpu_tensor_read(atlas_q, 0, atlas_q_host, q_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(atlas_k, 0, atlas_k_host, k_bytes) != 0);
            malformed_mismatches += test_compare_float_bits(
                q_input, atlas_q_host, (size_t)q_values).mismatch_count;
            malformed_mismatches += test_compare_float_bits(
                k_input, atlas_k_host, (size_t)k_values).mismatch_count;
            TEST_ASSERT(setenv(atlas_env_name, "1", 1) == 0);
            ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
            ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(
                            c->n_q_head, c->n_k_head, 128u, c->n_rot) == 1);
        }

        free(simd_k_host); free(simd_q_host); free(single_k_host);
        free(single_q_host); free(atlas_k_host); free(atlas_q_host);
        free(legacy_k_host); free(legacy_q_host); free(k_input); free(q_input);
        ds4_gpu_tensor_free(simd_k); ds4_gpu_tensor_free(simd_q);
        ds4_gpu_tensor_free(single_k); ds4_gpu_tensor_free(single_q);
        ds4_gpu_tensor_free(atlas_k); ds4_gpu_tensor_free(atlas_q);
        ds4_gpu_tensor_free(legacy_k); ds4_gpu_tensor_free(legacy_q);
    }

    TEST_ASSERT(pair_mismatches == 0);
    TEST_ASSERT(single_mismatches == 0);
    TEST_ASSERT(simd_mismatches == 0);
    TEST_ASSERT(malformed_mismatches == 0);
    TEST_ASSERT(simd_known);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_consumed_family_count(0u) != 0u);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_consumed_family_count(1u) != 0u);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(0u) != 0u);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(1u) != 0u);
    TEST_ASSERT(ds4_gpu_laguna_rope_atlas_encoded_dispatch_count() >
                atlas_generation_before);

    /* DFlash's support family is deliberately a separate one-plane atlas.
     * Exercise exact bits, both flush orderings, the 512-row bound, discard
     * re-encode, wrap-adjacent positions, and poison dependency. */
    {
        const uint32_t hd = 128u, nq = 72u, nk = 8u, sn = 3u, tn = 2u;
        const uint64_t sqv = (uint64_t)sn * nq * hd;
        const uint64_t skv = (uint64_t)sn * nk * hd;
        const uint64_t tqv = (uint64_t)tn * nq * hd;
        const uint64_t tkv = (uint64_t)tn * nk * hd;
        const uint64_t sqb = sqv * sizeof(float), skb = skv * sizeof(float);
        const uint64_t tqb = tqv * sizeof(float), tkb = tkv * sizeof(float);
        const uint64_t sk_atlas_b = (uint64_t)512u * nk * hd * sizeof(float);
        ds4_gpu_tensor *sq = ds4_gpu_tensor_alloc(sqb);
        ds4_gpu_tensor *sk = ds4_gpu_tensor_alloc(skb);
        ds4_gpu_tensor *sq_atlas = ds4_gpu_tensor_alloc(sqb);
        ds4_gpu_tensor *sk_atlas = ds4_gpu_tensor_alloc(sk_atlas_b);
        ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(tqb);
        ds4_gpu_tensor *tk = ds4_gpu_tensor_alloc(tkb);
        ds4_gpu_tensor *tq_atlas = ds4_gpu_tensor_alloc(tqb);
        ds4_gpu_tensor *tk_atlas = ds4_gpu_tensor_alloc(tkb);
        /* Separate poison-probe tensors prevent a second in-place norm/RoPE
         * pass from masquerading as atlas dependence.  The first consumer
         * establishes a keyed atlas use; after the atlas is poisoned, the
         * fresh tensors receive exactly one consumer pass. */
        ds4_gpu_tensor *sk_poison = ds4_gpu_tensor_alloc(skb);
        ds4_gpu_tensor *tq_poison = ds4_gpu_tensor_alloc(tqb);
        ds4_gpu_tensor *tk_poison = ds4_gpu_tensor_alloc(tkb);
        float *sqi = malloc((size_t)sqb), *ski = malloc((size_t)skb);
        float *sqr = malloc((size_t)sqb), *skr = malloc((size_t)skb);
        float *sqo = malloc((size_t)sqb), *sko = malloc((size_t)skb);
        float *tqi = malloc((size_t)tqb), *tki = malloc((size_t)tkb);
        float *tqr = malloc((size_t)tqb), *tkr = malloc((size_t)tkb);
        float *tqo = malloc((size_t)tqb), *tko = malloc((size_t)tkb);
        TEST_ASSERT(sq && sk && sq_atlas && sk_atlas && tq && tk &&
                    tq_atlas && tk_atlas && sk_poison && tq_poison &&
                    tk_poison && sqi && ski && sqr && skr &&
                    sqo && sko && tqi && tki && tqr && tkr && tqo && tko);
        if (sq && sk && sq_atlas && sk_atlas && tq && tk && tq_atlas &&
            tk_atlas && sk_poison && tq_poison && tk_poison && sqi && ski &&
            sqr && skr && sqo && sko && tqi && tki && tqr && tkr && tqo &&
            tko) {
            for (uint64_t i = 0; i < sqv; i++) sqi[i] = (float)((int)(i * 29u % 257u) - 128) / 131.0f;
            for (uint64_t i = 0; i < skv; i++) ski[i] = (float)((int)(i * 31u % 251u) - 125) / 127.0f;
            for (uint64_t i = 0; i < tqv; i++) tqi[i] = (float)((int)(i * 37u % 211u) - 105) / 137.0f;
            for (uint64_t i = 0; i < tkv; i++) tki[i] = (float)((int)(i * 41u % 199u) - 99) / 149.0f;
            const uint32_t spos = 511u;
            TEST_ASSERT(setenv(atlas_env_name, "0", 1) == 0);
            ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(nq, nk, hd, 128u) == 0);
            TEST_ASSERT(ds4_gpu_tensor_write(sq, 0, sqi, sqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(sk, 0, ski, skb) != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            sq, sk, model_raw, model_size, 0, k_weight_offset,
                            sn, nq, nk, hd, 128u, spos, 262144u, 500000.0f,
                            1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(sq, 0, sqr, sqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(sk, 0, skr, skb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(tq, 0, tqi, tqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(tk, 0, tki, tkb) != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            tq, tk, model_raw, model_size, 0, k_weight_offset,
                            tn, nq, nk, hd, 64u, 0u, 8192u, 500000.0f,
                            1.0f / 32.0f, 1.0f, 1.0f, 32.0f, 1.0f,
                            1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(tq, 0, tqr, tqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(tk, 0, tkr, tkb) != 0);

            TEST_ASSERT(setenv(atlas_env_name, "1", 1) == 0);
            ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(nq, nk, hd, 128u) == 1);
            const uint64_t support_done_before =
                ds4_gpu_laguna_rope_support_atlas_completed_generated_count();
            const uint64_t support_consumed_before =
                ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count();
            const uint64_t target_done_before =
                ds4_gpu_laguna_rope_atlas_completed_generated_count();
            const uint64_t target_consumed_before =
                ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
            const uint64_t target_family0_before =
                ds4_gpu_laguna_rope_atlas_completed_family_count(0u);

            /* support -> flush -> target */
            TEST_ASSERT(ds4_gpu_tensor_write(sq_atlas, 0, sqi, sqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(sk_atlas, 0, ski, skb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(tq_atlas, 0, tqi, tqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(tk_atlas, 0, tki, tkb) != 0);
            TEST_ASSERT(ds4_gpu_begin_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            sq_atlas, sk_atlas, model_raw, model_size, 0,
                            k_weight_offset, sn, nq, nk, hd, 128u, spos,
                            262144u, 500000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_flush_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            tq_atlas, tk_atlas, model_raw, model_size, 0,
                            k_weight_offset, tn, nq, nk, hd, 64u, 0u, 8192u,
                            500000.0f, 1.0f / 32.0f, 1.0f, 1.0f, 32.0f,
                            1.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_end_commands() != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(sq_atlas, 0, sqo, sqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(sk_atlas, 0, sko, skb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(tq_atlas, 0, tqo, tqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(tk_atlas, 0, tko, tkb) != 0);
            TEST_ASSERT(test_compare_float_bits(sqr, sqo, (size_t)sqv).mismatch_count == 0);
            TEST_ASSERT(test_compare_float_bits(skr, sko, (size_t)skv).mismatch_count == 0);
            TEST_ASSERT(test_compare_float_bits(tqr, tqo, (size_t)tqv).mismatch_count == 0);
            TEST_ASSERT(test_compare_float_bits(tkr, tko, (size_t)tkv).mismatch_count == 0);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_completed_generated_count() ==
                        support_done_before + 1u);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count() ==
                        support_consumed_before + 1u);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_generated_count() ==
                        target_done_before + 1u);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() ==
                        target_consumed_before + 1u);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(0u) ==
                        target_family0_before + 1u);

            /* target -> flush -> support */
            /* Start this ordering with a fresh cache key.  The production
             * cross-CB test below separately certifies completion-promoted
             * reuse; this branch must prove that either generation order is
             * capable of producing both independent atlas families. */
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_plan_reset_for_test() == 1);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() == 1);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(nq, nk, hd, 128u) == 1);
            const uint64_t support_done_before_reverse =
                ds4_gpu_laguna_rope_support_atlas_completed_generated_count();
            const uint64_t support_consumed_before_reverse =
                ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count();
            const uint64_t target_done_before_reverse =
                ds4_gpu_laguna_rope_atlas_completed_generated_count();
            const uint64_t target_consumed_before_reverse =
                ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
            const uint64_t target_family0_before_reverse =
                ds4_gpu_laguna_rope_atlas_completed_family_count(0u);
            TEST_ASSERT(ds4_gpu_tensor_write(sq_atlas, 0, sqi, sqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(sk_atlas, 0, ski, skb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(tq_atlas, 0, tqi, tqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(tk_atlas, 0, tki, tkb) != 0);
            TEST_ASSERT(ds4_gpu_begin_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            tq_atlas, tk_atlas, model_raw, model_size, 0,
                            k_weight_offset, tn, nq, nk, hd, 64u, 0u, 8192u,
                            500000.0f, 1.0f / 32.0f, 1.0f, 1.0f, 32.0f,
                            1.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_flush_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            sq_atlas, sk_atlas, model_raw, model_size, 0,
                            k_weight_offset, sn, nq, nk, hd, 128u, spos,
                            262144u, 500000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_end_commands() != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(sq_atlas, 0, sqo, sqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(sk_atlas, 0, sko, skb) != 0);
            TEST_ASSERT(test_compare_float_bits(sqr, sqo, (size_t)sqv).mismatch_count == 0);
            TEST_ASSERT(test_compare_float_bits(skr, sko, (size_t)skv).mismatch_count == 0);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_completed_generated_count() ==
                        support_done_before_reverse + 1u);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count() ==
                        support_consumed_before_reverse + 1u);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_generated_count() ==
                        target_done_before_reverse + 1u);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() ==
                        target_consumed_before_reverse + 1u);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(0u) ==
                        target_family0_before_reverse + 1u);

            /* Shorten after discard/re-encode, and cover positions straddling
             * the common SWA wrap points with exact legacy comparisons. */
            const uint32_t positions[] = {511u, 512u, 8191u, 8192u, 262143u};
            for (size_t pi = 0; pi < sizeof(positions) / sizeof(positions[0]); pi++) {
                TEST_ASSERT(setenv(atlas_env_name, "0", 1) == 0);
                ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
                ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
                TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(nq, nk, hd, 128u) == 0);
                TEST_ASSERT(ds4_gpu_tensor_write(sk, 0, ski, skb) != 0);
                TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                                sk, model_raw, model_size, k_weight_offset,
                                sn, nk, hd, 128u, positions[pi], 262144u,
                                500000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                1e-6f) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(sk, 0, skr, skb) != 0);
                TEST_ASSERT(setenv(atlas_env_name, "1", 1) == 0);
                ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
                ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
                TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(nq, nk, hd, 128u) == 1);
                TEST_ASSERT(ds4_gpu_tensor_write(sk_atlas, 0, ski, skb) != 0);
                TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                                sk_atlas, model_raw, model_size, k_weight_offset,
                                sn, nk, hd, 128u, positions[pi], 262144u,
                                500000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                1e-6f) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(sk_atlas, 0, sko, skb) != 0);
                TEST_ASSERT(test_compare_float_bits(
                                skr, sko, (size_t)skv).mismatch_count == 0);
            }
            /* Restore the 511-row legacy reference used by the poison probe;
             * the final wrap case above intentionally leaves a different
             * position in skr. */
            TEST_ASSERT(setenv(atlas_env_name, "0", 1) == 0);
            ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
            ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(nq, nk, hd, 128u) == 0);
            TEST_ASSERT(ds4_gpu_tensor_write(sk, 0, ski, skb) != 0);
            TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                            sk, model_raw, model_size, k_weight_offset,
                            sn, nk, hd, 128u, spos, 262144u, 500000.0f,
                            1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(sk, 0, skr, skb) != 0);
            TEST_ASSERT(setenv(atlas_env_name, "1", 1) == 0);
            ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
            ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_preflight(nq, nk, hd, 128u) == 1);
            const uint64_t reencode_before =
                ds4_gpu_laguna_rope_support_atlas_encoded_dispatch_count();
            const uint64_t discard_completed_generated =
                ds4_gpu_laguna_rope_support_atlas_completed_generated_count();
            const uint64_t discard_completed_consumed =
                ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count();
            TEST_ASSERT(ds4_gpu_tensor_fill_f32(
                            sk_atlas, 0.0f, (uint64_t)512u * nk * hd) != 0);
            TEST_ASSERT(ds4_gpu_begin_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                            sk_atlas, model_raw, model_size, k_weight_offset,
                            512u, nk, hd, 128u, 511u, 262144u, 500000.0f,
                            1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_discard_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_completed_generated_count() ==
                        discard_completed_generated);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count() ==
                        discard_completed_consumed);
            TEST_ASSERT(ds4_gpu_begin_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                            sk_atlas, model_raw, model_size, k_weight_offset,
                            sn, nk, hd, 128u, 512u, 262144u, 500000.0f,
                            1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_end_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_encoded_dispatch_count() >=
                        reencode_before + 2u);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_completed_generated_count() ==
                        discard_completed_generated + 1u);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count() ==
                        discard_completed_consumed + 1u);

            /* Poison support after the first keyed consumer. */
            const uint64_t support_poison_encoded_before =
                ds4_gpu_laguna_rope_support_atlas_consumed_dispatch_count();
            const uint64_t support_poison_completed_before =
                ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count();
            TEST_ASSERT(ds4_gpu_tensor_write(sk_atlas, 0, ski, skb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(sk_poison, 0, ski, skb) != 0);
            TEST_ASSERT(ds4_gpu_begin_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                            sk_atlas, model_raw, model_size, k_weight_offset,
                            sn, nk, hd, 128u, spos, 262144u, 500000.0f,
                            1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_test_poison(1u) != 0);
            TEST_ASSERT(ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                            sk_poison, model_raw, model_size, k_weight_offset,
                            sn, nk, hd, 128u, spos, 262144u, 500000.0f,
                            1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_end_commands() != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(sk_poison, 0, sko, skb) != 0);
            bool support_poison_changed = false;
            for (uint64_t i = 0; i < skv; i++) {
                uint32_t bits = 0;
                memcpy(&bits, &sko[i], sizeof(bits));
                /* A zero rot128 atlas must rotate every support element to
                 * signed zero after the one consumer pass. */
                TEST_ASSERT((bits & 0x7fffffffu) == 0u);
                if (memcmp(&ski[i], &sko[i], sizeof(float)) != 0) {
                    support_poison_changed = true;
                }
            }
            TEST_ASSERT(support_poison_changed);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_consumed_dispatch_count() ==
                        support_poison_encoded_before + 2u);
            TEST_ASSERT(ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count() ==
                        support_poison_completed_before + 2u);

            /* Poison target after the first keyed consumer. */
            const uint64_t target_poison_encoded_before =
                ds4_gpu_laguna_rope_atlas_consumed_dispatch_count();
            const uint64_t target_poison_completed_before =
                ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
            const uint64_t target_poison_family0_before =
                ds4_gpu_laguna_rope_atlas_completed_family_count(0u);
            TEST_ASSERT(ds4_gpu_tensor_write(tq_atlas, 0, tqi, tqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(tk_atlas, 0, tki, tkb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(tq_poison, 0, tqi, tqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(tk_poison, 0, tki, tkb) != 0);
            TEST_ASSERT(ds4_gpu_begin_commands() != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            tq_atlas, tk_atlas, model_raw, model_size, 0,
                            k_weight_offset, tn, nq, nk, hd, 64u, 0u, 8192u,
                            500000.0f, 1.0f / 32.0f, 1.0f, 1.0f, 32.0f,
                            1.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_test_poison(0u) != 0);
            TEST_ASSERT(ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                            tq_poison, tk_poison, model_raw, model_size, 0,
                            k_weight_offset, tn, nq, nk, hd, 64u, 0u, 8192u,
                            500000.0f, 1.0f / 32.0f, 1.0f, 1.0f, 32.0f,
                            1.0f, 1e-6f) != 0);
            TEST_ASSERT(ds4_gpu_end_commands() != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(tq_poison, 0, tqo, tqb) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(tk_poison, 0, tko, tkb) != 0);
            bool target_poison_changed = false;
            for (uint64_t i = 0; i < tqv; i++) {
                const uint32_t head_offset = (uint32_t)(i % hd);
                if (head_offset < 64u) {
                    uint32_t bits = 0;
                    memcpy(&bits, &tqo[i], sizeof(bits));
                    TEST_ASSERT((bits & 0x7fffffffu) == 0u);
                } else {
                    TEST_ASSERT(memcmp(&tqr[i], &tqo[i], sizeof(float)) == 0);
                }
                if (memcmp(&tqi[i], &tqo[i], sizeof(float)) != 0) {
                    target_poison_changed = true;
                }
            }
            for (uint64_t i = 0; i < tkv; i++) {
                const uint32_t head_offset = (uint32_t)(i % hd);
                if (head_offset < 64u) {
                    uint32_t bits = 0;
                    memcpy(&bits, &tko[i], sizeof(bits));
                    TEST_ASSERT((bits & 0x7fffffffu) == 0u);
                } else {
                    TEST_ASSERT(memcmp(&tkr[i], &tko[i], sizeof(float)) == 0);
                }
                if (memcmp(&tki[i], &tko[i], sizeof(float)) != 0) {
                    target_poison_changed = true;
                }
            }
            TEST_ASSERT(target_poison_changed);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_consumed_dispatch_count() ==
                        target_poison_encoded_before + 2u);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() ==
                        target_poison_completed_before + 2u);
            TEST_ASSERT(ds4_gpu_laguna_rope_atlas_completed_family_count(0u) ==
                        target_poison_family0_before + 2u);
        }
        free(tko); free(tqo); free(tkr); free(tqr); free(tki); free(tqi);
        free(sko); free(sqo); free(skr); free(sqr); free(ski); free(sqi);
        ds4_gpu_tensor_free(tk_atlas); ds4_gpu_tensor_free(tq_atlas);
        ds4_gpu_tensor_free(tk_poison); ds4_gpu_tensor_free(tq_poison);
        ds4_gpu_tensor_free(tk); ds4_gpu_tensor_free(tq);
        ds4_gpu_tensor_free(sk_atlas); ds4_gpu_tensor_free(sq_atlas);
        ds4_gpu_tensor_free(sk_poison);
        ds4_gpu_tensor_free(sk); ds4_gpu_tensor_free(sq);
    }

    test_metal_laguna_rope_atlas_production_counts(
        model_raw, model_size, k_weight_offset, atlas_env_name, simd_env_name);

    fprintf(stderr,
            "ds4-test: Laguna RoPE atlas exact pair=%zu single=%zu simd=%zu "
            "malformed=%zu generation_encoded=%llu consumers_encoded=%llu "
            "family0_encoded=%llu family1_encoded=%llu "
            "family0_completed=%llu family1_completed=%llu "
            "max_ulp=%u\n",
            pair_mismatches, single_mismatches, simd_mismatches,
            malformed_mismatches,
            (unsigned long long)(ds4_gpu_laguna_rope_atlas_encoded_dispatch_count() -
                                 atlas_generation_before),
            (unsigned long long)ds4_gpu_laguna_rope_atlas_consumed_dispatch_count(),
            (unsigned long long)ds4_gpu_laguna_rope_atlas_consumed_family_count(0u),
            (unsigned long long)ds4_gpu_laguna_rope_atlas_consumed_family_count(1u),
            (unsigned long long)ds4_gpu_laguna_rope_atlas_completed_family_count(0u),
            (unsigned long long)ds4_gpu_laguna_rope_atlas_completed_family_count(1u),
            max_ulp);

cleanup:
    test_restore_env(atlas_env_name, saved_atlas_env);
    test_restore_env(simd_env_name, saved_simd_env);
    /* Do not let the focused test's frozen plan survive into a later
     * aggregate Metal test after restoring the caller's environment. */
    ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
    ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
    free(model_raw);
}

#endif

#if defined(__APPLE__)
static void test_metal_add_rms_norm_weight_rows_exact_case(
        uint32_t n,
        uint32_t n_rows,
        uint32_t seed) {
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_offset = page;
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint64_t activation_bytes = (uint64_t)n_rows * row_bytes;
    const uint64_t model_alloc = test_round_up_u64(
        weight_offset + row_bytes, page);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(activation_bytes);
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(activation_bytes);
    ds4_gpu_tensor *ref_sum = ds4_gpu_tensor_alloc(activation_bytes);
    ds4_gpu_tensor *fused_sum = ds4_gpu_tensor_alloc(activation_bytes);
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(activation_bytes);
    ds4_gpu_tensor *fused_norm = ds4_gpu_tensor_alloc(activation_bytes);
    float *a_host = malloc((size_t)activation_bytes);
    float *b_host = malloc((size_t)activation_bytes);
    float *ref_sum_host = malloc((size_t)activation_bytes);
    float *fused_sum_host = malloc((size_t)activation_bytes);
    float *ref_norm_host = malloc((size_t)activation_bytes);
    float *fused_norm_host = malloc((size_t)activation_bytes);

    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(a != NULL);
    TEST_ASSERT(b != NULL);
    TEST_ASSERT(ref_sum != NULL);
    TEST_ASSERT(fused_sum != NULL);
    TEST_ASSERT(ref_norm != NULL);
    TEST_ASSERT(fused_norm != NULL);
    TEST_ASSERT(a_host != NULL);
    TEST_ASSERT(b_host != NULL);
    TEST_ASSERT(ref_sum_host != NULL);
    TEST_ASSERT(fused_sum_host != NULL);
    TEST_ASSERT(ref_norm_host != NULL);
    TEST_ASSERT(fused_norm_host != NULL);

    const bool allocated = model_raw && a && b && ref_sum && fused_sum &&
        ref_norm && fused_norm && a_host && b_host && ref_sum_host &&
        fused_sum_host && ref_norm_host && fused_norm_host;
    test_float_compare_stats sum_stats = {0};
    test_float_compare_stats norm_stats = {0};
    if (allocated) {
        memset(model_raw, 0, (size_t)model_alloc);
        float *weight = (float *)((uint8_t *)model_raw + weight_offset);
        for (uint32_t d = 0; d < n; d++) {
            weight[d] = 0.25f +
                (float)((d * 29u + seed * 17u) % 113u) / 64.0f;
        }
        for (uint32_t row = 0; row < n_rows; row++) {
            for (uint32_t d = 0; d < n; d++) {
                const uint32_t key = d * 73u + row * 1009u + seed * 131u;
                const int av = (int)(key % 4093u) - 2046;
                const int bv = (int)((key * 7u + 19u) % 4093u) - 2046;
                const uint64_t i = (uint64_t)row * n + d;
                a_host[i] = (float)av / 1024.0f;
                b_host[i] = (float)bv / 2048.0f;
            }
        }

        TEST_ASSERT(ds4_gpu_tensor_write(
                        a, 0, a_host, activation_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        b, 0, b_host, activation_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
        TEST_ASSERT(ds4_gpu_add_tensor(
                        ref_sum, a, b, (uint32_t)((uint64_t)n * n_rows)) != 0);
        TEST_ASSERT(ds4_gpu_rms_norm_weight_rows_tensor(
                        ref_norm, ref_sum, model_raw, model_alloc,
                        weight_offset, n, n_rows, eps) != 0);
        TEST_ASSERT(ds4_gpu_add_rms_norm_weight_rows_tensor(
                        fused_norm, fused_sum, a, b, model_raw, model_alloc,
                        weight_offset, n, n_rows, eps) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_sum, 0, ref_sum_host, activation_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_sum, 0, fused_sum_host, activation_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_norm, 0, ref_norm_host, activation_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_norm, 0, fused_norm_host, activation_bytes) != 0);
        sum_stats = test_compare_float_bits(
            ref_sum_host, fused_sum_host, (size_t)n * n_rows);
        norm_stats = test_compare_float_bits(
            ref_norm_host, fused_norm_host, (size_t)n * n_rows);
    }

    fprintf(stderr,
            "ds4-test: add+RMSNorm rows exact n=%u rows=%u "
            "sum=%zu/%llu max_ulp=%u max_abs=%g "
            "norm=%zu/%llu max_ulp=%u max_abs=%g\n",
            n, n_rows,
            sum_stats.mismatch_count,
            (unsigned long long)((uint64_t)n * n_rows),
            sum_stats.max_ulp,
            sum_stats.max_abs,
            norm_stats.mismatch_count,
            (unsigned long long)((uint64_t)n * n_rows),
            norm_stats.max_ulp,
            norm_stats.max_abs);
    TEST_ASSERT(sum_stats.mismatch_count == 0);
    TEST_ASSERT(norm_stats.mismatch_count == 0);

    free(fused_norm_host);
    free(ref_norm_host);
    free(fused_sum_host);
    free(ref_sum_host);
    free(b_host);
    free(a_host);
    ds4_gpu_tensor_free(fused_norm);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(fused_sum);
    ds4_gpu_tensor_free(ref_sum);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    free(model_raw);
}

static void test_metal_add_rms_norm_weight_rows_exact(void) {
    test_metal_add_rms_norm_weight_rows_exact_case(4096, 1, 71);
    test_metal_add_rms_norm_weight_rows_exact_case(4096, 3, 83);
    test_metal_add_rms_norm_weight_rows_exact_case(3072, 1, 86);
    test_metal_add_rms_norm_weight_rows_exact_case(3072, 4, 87);
    test_metal_add_rms_norm_weight_rows_exact_case(7168, 5, 89);
}

static void test_metal_laguna_decode_residual_norm_env(void) {
    static const struct {
        const char *value;
        int expected;
    } cases[] = {
        { NULL, 0 },
        { "", 0 },
        { "0", 0 },
        { "1", 1 },
        { "01", -1 },
        { "true", -1 },
        { " 1", -1 },
        { "1 ", -1 },
        { "2", -1 },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        TEST_ASSERT(ds4_gpu_laguna_decode_residual_norm_env_mode(
                        cases[i].value) == cases[i].expected);
    }
}

static void test_metal_add3_rms_norm_rejects_partial_simd(void) {
    if (!ds4_gpu_laguna_decode_residual_norm_available()) {
        const char *override = getenv("DS4_METAL_NORM_SOURCE");
        if (override && override[0]) {
            fprintf(stderr,
                    "ds4-test: skipping partial-SIMD add3 rejection; "
                    "DS4_METAL_NORM_SOURCE lacks the opt-in kernel\n");
            return;
        }
        TEST_ASSERT(false);
        return;
    }

    const uint32_t n = 132u; /* n/4 = 33, one partial SIMD group. */
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    if (!model_raw) return;
    memset(model_raw, 0, (size_t)page);

    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *c = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *sum = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *norm = ds4_gpu_tensor_alloc(bytes);
    TEST_ASSERT(a && b && c && sum && norm);
    if (a && b && c && sum && norm) {
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        TEST_ASSERT(ds4_gpu_add3_rms_norm_weight_rows_tensor(
                        norm, sum, a, b, c, model_raw, page, 0,
                        n, 1, 1.0e-6f) == 0);
    }
    (void)ds4_gpu_wait_submitted_commands();
    ds4_gpu_tensor_free(norm);
    ds4_gpu_tensor_free(sum);
    ds4_gpu_tensor_free(c);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    free(model_raw);
}

static uint32_t test_laguna_residual_poison_word(uint32_t tag, uint64_t index) {
    uint32_t payload = (uint32_t)(
        (index * UINT64_C(2654435761) + (uint64_t)tag * UINT64_C(0x123457)) &
        UINT64_C(0x003fffff));
    if (payload == 0) payload = 1;
    return UINT32_C(0x7fc00000) | payload;
}

static void test_laguna_store_f32_bits(float *dst, uint32_t bits) {
    memcpy(dst, &bits, sizeof(bits));
}

static void test_metal_add3_rms_norm_weight_rows_exact_case(
        uint32_t n,
        uint32_t n_rows,
        uint32_t seed,
        bool exceptional_row) {
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_offset = page;
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint64_t activation_bytes = (uint64_t)n_rows * row_bytes;
    const uint64_t model_alloc = test_round_up_u64(
        weight_offset + row_bytes, page);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(activation_bytes);
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(activation_bytes);
    ds4_gpu_tensor *c = ds4_gpu_tensor_alloc(activation_bytes);
    ds4_gpu_tensor *ref_sum = ds4_gpu_tensor_alloc(activation_bytes);
    ds4_gpu_tensor *fused_sum = ds4_gpu_tensor_alloc(activation_bytes);
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(activation_bytes);
    ds4_gpu_tensor *fused_norm = ds4_gpu_tensor_alloc(activation_bytes);
    float *a_host = malloc((size_t)activation_bytes);
    float *b_host = malloc((size_t)activation_bytes);
    float *c_host = malloc((size_t)activation_bytes);
    float *ref_sum_host = malloc((size_t)activation_bytes);
    float *fused_sum_host = malloc((size_t)activation_bytes);
    float *ref_norm_host = malloc((size_t)activation_bytes);
    float *fused_norm_host = malloc((size_t)activation_bytes);
    uint32_t *poison = malloc((size_t)activation_bytes);

    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(a != NULL);
    TEST_ASSERT(b != NULL);
    TEST_ASSERT(c != NULL);
    TEST_ASSERT(ref_sum != NULL);
    TEST_ASSERT(fused_sum != NULL);
    TEST_ASSERT(ref_norm != NULL);
    TEST_ASSERT(fused_norm != NULL);
    TEST_ASSERT(a_host != NULL);
    TEST_ASSERT(b_host != NULL);
    TEST_ASSERT(c_host != NULL);
    TEST_ASSERT(ref_sum_host != NULL);
    TEST_ASSERT(fused_sum_host != NULL);
    TEST_ASSERT(ref_norm_host != NULL);
    TEST_ASSERT(fused_norm_host != NULL);
    TEST_ASSERT(poison != NULL);

    const bool allocated = model_raw && a && b && c && ref_sum && fused_sum &&
        ref_norm && fused_norm && a_host && b_host && c_host &&
        ref_sum_host && fused_sum_host && ref_norm_host && fused_norm_host &&
        poison;
    test_float_compare_stats sum_stats = {0};
    test_float_compare_stats norm_stats = {0};
    if (!allocated) goto cleanup;

    if (!ds4_gpu_laguna_decode_residual_norm_available()) {
        /* An explicitly supplied old norm source is allowed to omit this
         * opt-in kernel; the default in-repo source must provide it. */
        const char *override = getenv("DS4_METAL_NORM_SOURCE");
        if (override && override[0]) {
            fprintf(stderr,
                    "ds4-test: skipping add3+RMS exact case because "
                    "DS4_METAL_NORM_SOURCE lacks the opt-in kernel\n");
            goto cleanup;
        }
        TEST_ASSERT(false);
        goto cleanup;
    }

    memset(model_raw, 0, (size_t)model_alloc);
    float *weight = (float *)((uint8_t *)model_raw + weight_offset);
    for (uint32_t d = 0; d < n; d++) {
        const uint32_t key = d * 29u + seed * 17u;
        weight[d] = (d % 31u == 0u) ? -0.0f :
            0.25f + (float)(key % 113u) / 64.0f;
    }
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t d = 0; d < n; d++) {
            const uint32_t key = d * 73u + row * 1009u + seed * 131u;
            const uint64_t i = (uint64_t)row * n + d;
            if (exceptional_row && row + 1u == n_rows && d < 6u) {
                /* Keep exceptional values in one isolated row.  Metal's
                 * fast-math NaN/Inf payload behavior is not a portable
                 * source-level contract, so this row is a write/poison probe;
                 * the finite row below remains the exactness gate. */
                switch (d) {
                case 0:
                    a_host[i] = 0x1.fffffep+127f;
                    b_host[i] = -0x1.fffffep+127f;
                    c_host[i] = 1.0f;
                    break;
                case 1:
                    a_host[i] = 0x1.fffffep+127f;
                    b_host[i] = 0x1.fffffep+127f;
                    c_host[i] = 0.0f;
                    break;
                case 2:
                    test_laguna_store_f32_bits(&a_host[i], UINT32_C(0x7f800000));
                    b_host[i] = 1.0f;
                    c_host[i] = -1.0f;
                    break;
                case 3:
                    test_laguna_store_f32_bits(&a_host[i], UINT32_C(0x7fc00000));
                    b_host[i] = 1.0f;
                    c_host[i] = 2.0f;
                    break;
                case 4:
                    test_laguna_store_f32_bits(&a_host[i], UINT32_C(0xff800000));
                    test_laguna_store_f32_bits(&b_host[i], UINT32_C(0x7f800000));
                    c_host[i] = 0.0f;
                    break;
                default:
                    a_host[i] = 0x1p-149f;
                    b_host[i] = -0x1p-149f;
                    c_host[i] = 0x1p-149f;
                    break;
                }
            } else if (row == 0u && d == 7u) {
                /* A triple negative zero is the deterministic signed-zero
                 * case; (1 + -1) + +/-0 would round to +0. */
                const uint32_t neg_zero = UINT32_C(0x80000000);
                memcpy(&a_host[i], &neg_zero, sizeof(neg_zero));
                memcpy(&b_host[i], &neg_zero, sizeof(neg_zero));
                memcpy(&c_host[i], &neg_zero, sizeof(neg_zero));
            } else if ((key % 19u) == 0u) {
                /* Cancellation and an explicit signed zero exercise the
                 * stock left-associated add3 operation. */
                a_host[i] = 1.0f;
                b_host[i] = -1.0f;
                c_host[i] = (d & 1u) ? -0.0f : 0.0f;
            } else if ((key % 23u) == 0u) {
                a_host[i] = 1.0e20f;
                b_host[i] = -1.0e20f;
                c_host[i] = (float)((int)(key % 17u) - 8) / 4096.0f;
            } else if ((key % 29u) == 0u) {
                /* Include subnormals and max-finite operands whose
                 * left-associated residual remains finite.  The latter
                 * avoids making every tested row's RMS reduction Inf. */
                a_host[i] = (d & 1u) ? 0x1.fffffep+127f : 0x1p-149f;
                b_host[i] = (d & 1u) ? -0x1.fffffep+127f : -0x1p-149f;
                c_host[i] = (d & 1u) ? 1.0f : 0x1p-149f;
            } else {
                a_host[i] = (float)((int)(key % 4093u) - 2046) / 1024.0f;
                b_host[i] = (float)((int)((key * 7u + 19u) % 4093u) - 2046) /
                    2048.0f;
                c_host[i] = (float)((int)((key * 11u + 31u) % 4093u) - 2046) /
                    4096.0f;
            }
        }
    }

    TEST_ASSERT(ds4_gpu_tensor_write(a, 0, a_host, activation_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(b, 0, b_host, activation_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(c, 0, c_host, activation_bytes) != 0);
    memset(poison, 0xa5, (size_t)activation_bytes);
    const size_t value_count = (size_t)n * n_rows;
    for (size_t i = 0; i < value_count; i++) {
        poison[i] = test_laguna_residual_poison_word(1, i);
    }
    TEST_ASSERT(ds4_gpu_tensor_write(
                    ref_sum, 0, poison, activation_bytes) != 0);
    for (size_t i = 0; i < value_count; i++) {
        poison[i] = test_laguna_residual_poison_word(2, i);
    }
    TEST_ASSERT(ds4_gpu_tensor_write(
                    fused_sum, 0, poison, activation_bytes) != 0);
    for (size_t i = 0; i < value_count; i++) {
        poison[i] = test_laguna_residual_poison_word(3, i);
    }
    TEST_ASSERT(ds4_gpu_tensor_write(
                    ref_norm, 0, poison, activation_bytes) != 0);
    for (size_t i = 0; i < value_count; i++) {
        poison[i] = test_laguna_residual_poison_word(4, i);
    }
    TEST_ASSERT(ds4_gpu_tensor_write(
                    fused_norm, 0, poison, activation_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
    TEST_ASSERT(ds4_gpu_add3_tensor(
                    ref_sum, a, b, c,
                    (uint32_t)((uint64_t)n * n_rows)) != 0);
    TEST_ASSERT(ds4_gpu_rms_norm_weight_rows_tensor(
                    ref_norm, ref_sum, model_raw, model_alloc,
                    weight_offset, n, n_rows, eps) != 0);
    TEST_ASSERT(ds4_gpu_add3_rms_norm_weight_rows_tensor(
                    fused_norm, fused_sum, a, b, c,
                    model_raw, model_alloc, weight_offset, n, n_rows, eps) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    ref_sum, 0, ref_sum_host, activation_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    fused_sum, 0, fused_sum_host, activation_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    ref_norm, 0, ref_norm_host, activation_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    fused_norm, 0, fused_norm_host, activation_bytes) != 0);
    for (size_t i = 0; i < value_count; i++) {
        uint32_t bits = 0;
        memcpy(&bits, &ref_sum_host[i], sizeof(bits));
        TEST_ASSERT(bits != test_laguna_residual_poison_word(1, i));
        memcpy(&bits, &fused_sum_host[i], sizeof(bits));
        TEST_ASSERT(bits != test_laguna_residual_poison_word(2, i));
        memcpy(&bits, &ref_norm_host[i], sizeof(bits));
        TEST_ASSERT(bits != test_laguna_residual_poison_word(3, i));
        memcpy(&bits, &fused_norm_host[i], sizeof(bits));
        TEST_ASSERT(bits != test_laguna_residual_poison_word(4, i));
    }
    {
        const size_t signed_zero_index = 7u;
        uint32_t input_a_bits = 0;
        uint32_t input_b_bits = 0;
        uint32_t input_c_bits = 0;
        uint32_t ref_bits = 0;
        uint32_t fused_bits = 0;
        memcpy(&input_a_bits, &a_host[signed_zero_index], sizeof(input_a_bits));
        memcpy(&input_b_bits, &b_host[signed_zero_index], sizeof(input_b_bits));
        memcpy(&input_c_bits, &c_host[signed_zero_index], sizeof(input_c_bits));
        memcpy(&ref_bits, &ref_sum_host[signed_zero_index], sizeof(ref_bits));
        memcpy(&fused_bits, &fused_sum_host[signed_zero_index], sizeof(fused_bits));
        TEST_ASSERT(input_a_bits == UINT32_C(0x80000000));
        TEST_ASSERT(input_b_bits == UINT32_C(0x80000000));
        TEST_ASSERT(input_c_bits == UINT32_C(0x80000000));
        TEST_ASSERT(ref_bits == fused_bits);
    }
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t d = 0; d < n; d++) {
            const uint32_t key = d * 73u + row * 1009u + seed * 131u;
            /* The host sentinel checks intentionally stop at ordinary finite
             * values: the Metal fast-math contract flushes subnormal sums,
             * while the ref/fused GPU bit comparison above still covers the
             * key%29 subnormal/max-finite vectors. */
            if ((exceptional_row && row + 1u == n_rows) ||
                ((key % 19u) != 0u && (key % 23u) != 0u)) continue;
            const uint64_t i = (uint64_t)row * n + d;
            volatile float partial = a_host[i] + b_host[i];
            volatile float expected = partial + c_host[i];
            const float expected_value = expected;
            uint32_t expected_bits = 0;
            uint32_t actual_bits = 0;
            memcpy(&expected_bits, &expected_value, sizeof(expected_bits));
            memcpy(&actual_bits, &fused_sum_host[i], sizeof(actual_bits));
            TEST_ASSERT(expected_bits == actual_bits);
        }
    }
    const size_t exact_value_count = exceptional_row ? (size_t)n : value_count;
    sum_stats = test_compare_float_bits(
        ref_sum_host, fused_sum_host, exact_value_count);
    norm_stats = test_compare_float_bits(
        ref_norm_host, fused_norm_host, exact_value_count);
    fprintf(stderr,
            "ds4-test: add3+RMSNorm rows exact n=%u rows=%u "
            "sum=%zu/%llu max_ulp=%u norm=%zu/%llu max_ulp=%u"
            " exceptional=%s\n",
            n, n_rows,
            sum_stats.mismatch_count,
            (unsigned long long)exact_value_count,
            sum_stats.max_ulp,
            norm_stats.mismatch_count,
            (unsigned long long)exact_value_count,
            norm_stats.max_ulp,
            exceptional_row ? "write-probe" : "no");
    TEST_ASSERT(sum_stats.mismatch_count == 0);
    TEST_ASSERT(norm_stats.mismatch_count == 0);

cleanup:
    (void)ds4_gpu_wait_submitted_commands();
    free(poison);
    free(fused_norm_host);
    free(ref_norm_host);
    free(fused_sum_host);
    free(ref_sum_host);
    free(c_host);
    free(b_host);
    free(a_host);
    ds4_gpu_tensor_free(fused_norm);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(fused_sum);
    ds4_gpu_tensor_free(ref_sum);
    ds4_gpu_tensor_free(c);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    free(model_raw);
}

static void test_metal_add3_rms_norm_weight_rows_exact(void) {
    test_metal_add3_rms_norm_weight_rows_exact_case(4096, 1, 101, false);
    test_metal_add3_rms_norm_weight_rows_exact_case(4096, 3, 103, false);
    test_metal_add3_rms_norm_weight_rows_exact_case(3072, 1, 106, false);
    test_metal_add3_rms_norm_weight_rows_exact_case(3072, 4, 107, false);
    test_metal_add3_rms_norm_weight_rows_exact_case(7168, 5, 109, false);
    test_metal_add3_rms_norm_weight_rows_exact_case(3072, 2, 113, true);
}

static void test_metal_hc_split_weighted_sum_norm_batch_exact(void) {
    /* Compare the batched HC+RMSNorm fusion against the exact two-dispatch
     * sequence used by the reference path at DS4's production dimensions. */
    const uint32_t n_embd = 7168;
    const uint32_t n_hc = 4;
    const uint32_t n_rows = 3;
    const uint32_t sinkhorn_iters = 20;
    const float hc_eps = 1.0e-6f;
    const float norm_eps = 1.0e-6f;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t scale_offset = 0;
    const uint64_t base_offset = test_round_up_u64(3u * sizeof(float), page);
    const uint64_t norm_weight_offset =
        test_round_up_u64(base_offset + mix_hc * sizeof(float), page);
    const uint64_t model_alloc = test_round_up_u64(
        norm_weight_offset + (uint64_t)n_embd * sizeof(float), page);
    const uint64_t mix_bytes = (uint64_t)n_rows * mix_hc * sizeof(float);
    const uint64_t residual_bytes =
        (uint64_t)n_rows * n_hc * n_embd * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_rows * n_embd * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)model_alloc) == 0);
    if (!model_raw) return;
    memset(model_raw, 0, (size_t)model_alloc);

    float *scale = (float *)((uint8_t *)model_raw + scale_offset);
    float *base = (float *)((uint8_t *)model_raw + base_offset);
    float *norm_weight = (float *)((uint8_t *)model_raw + norm_weight_offset);
    scale[0] = 0.625f;
    scale[1] = -0.75f;
    scale[2] = 0.4375f;
    for (uint32_t i = 0; i < mix_hc; i++) {
        const int value = (int)((i * 17u + 5u) % 29u) - 14;
        base[i] = (float)value / 16.0f;
    }
    for (uint32_t i = 0; i < n_embd; i++) {
        norm_weight[i] = 0.5f + (float)((i * 13u + 7u) % 31u) / 32.0f;
    }

    ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(residual_bytes);
    ds4_gpu_tensor *ref_split = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *fused_split = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_norm = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(mix != NULL);
    TEST_ASSERT(residual != NULL);
    TEST_ASSERT(ref_split != NULL);
    TEST_ASSERT(fused_split != NULL);
    TEST_ASSERT(ref_out != NULL);
    TEST_ASSERT(fused_out != NULL);
    TEST_ASSERT(ref_norm != NULL);
    TEST_ASSERT(fused_norm != NULL);
    if (!mix || !residual || !ref_split || !fused_split ||
        !ref_out || !fused_out || !ref_norm || !fused_norm) {
        ds4_gpu_tensor_free(mix);
        ds4_gpu_tensor_free(residual);
        ds4_gpu_tensor_free(ref_split);
        ds4_gpu_tensor_free(fused_split);
        ds4_gpu_tensor_free(ref_out);
        ds4_gpu_tensor_free(fused_out);
        ds4_gpu_tensor_free(ref_norm);
        ds4_gpu_tensor_free(fused_norm);
        free(model_raw);
        return;
    }

    float *mix_host = malloc((size_t)mix_bytes);
    float *residual_host = malloc((size_t)residual_bytes);
    float *ref_split_host = malloc((size_t)mix_bytes);
    float *fused_split_host = malloc((size_t)mix_bytes);
    float *ref_out_host = malloc((size_t)out_bytes);
    float *fused_out_host = malloc((size_t)out_bytes);
    float *ref_norm_host = malloc((size_t)out_bytes);
    float *fused_norm_host = malloc((size_t)out_bytes);
    TEST_ASSERT(mix_host != NULL);
    TEST_ASSERT(residual_host != NULL);
    TEST_ASSERT(ref_split_host != NULL);
    TEST_ASSERT(fused_split_host != NULL);
    TEST_ASSERT(ref_out_host != NULL);
    TEST_ASSERT(fused_out_host != NULL);
    TEST_ASSERT(ref_norm_host != NULL);
    TEST_ASSERT(fused_norm_host != NULL);
    if (!mix_host || !residual_host || !ref_split_host || !fused_split_host ||
        !ref_out_host || !fused_out_host || !ref_norm_host || !fused_norm_host) {
        free(mix_host);
        free(residual_host);
        free(ref_split_host);
        free(fused_split_host);
        free(ref_out_host);
        free(fused_out_host);
        free(ref_norm_host);
        free(fused_norm_host);
        ds4_gpu_tensor_free(mix);
        ds4_gpu_tensor_free(residual);
        ds4_gpu_tensor_free(ref_split);
        ds4_gpu_tensor_free(fused_split);
        ds4_gpu_tensor_free(ref_out);
        ds4_gpu_tensor_free(fused_out);
        ds4_gpu_tensor_free(ref_norm);
        ds4_gpu_tensor_free(fused_norm);
        free(model_raw);
        return;
    }

    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t i = 0; i < mix_hc; i++) {
            const int value =
                (int)(((row + 1u) * 19u + i * 11u + (i ^ row) * 3u) % 47u) - 23;
            mix_host[(uint64_t)row * mix_hc + i] = (float)value / 9.0f;
        }
        for (uint32_t hc = 0; hc < n_hc; hc++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                const uint32_t key =
                    d * 37u + hc * 173u + row * 997u + ((d >> 3u) ^ (d * 7u));
                const int value = (int)(key % 2047u) - 1023;
                residual_host[((uint64_t)row * n_hc + hc) * n_embd + d] =
                    (float)value / 512.0f;
            }
        }
    }

    TEST_ASSERT(ds4_gpu_tensor_write(mix, 0, mix_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(residual, 0, residual_host, residual_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
    TEST_ASSERT(ds4_gpu_hc_split_weighted_sum_tensor(
        ref_out, ref_split, mix, residual,
        model_raw, model_alloc, scale_offset, base_offset,
        n_embd, n_hc, sinkhorn_iters, hc_eps) != 0);
    TEST_ASSERT(ds4_gpu_rms_norm_weight_rows_tensor(
        ref_norm, ref_out, model_raw, model_alloc, norm_weight_offset,
        n_embd, n_rows, norm_eps) != 0);
    TEST_ASSERT(ds4_gpu_hc_split_weighted_sum_norm_tensor(
        fused_out, fused_norm, fused_split, mix, residual,
        model_raw, model_alloc, scale_offset, base_offset, norm_weight_offset,
        n_embd, n_hc, sinkhorn_iters, hc_eps, norm_eps) != 0);

    TEST_ASSERT(ds4_gpu_tensor_read(ref_split, 0, ref_split_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_split, 0, fused_split_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_out, 0, ref_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_out, 0, fused_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_norm, 0, ref_norm_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_norm, 0, fused_norm_host, out_bytes) != 0);

    const test_float_compare_stats split_stats = test_compare_float_bits(
        ref_split_host, fused_split_host, (size_t)(n_rows * mix_hc));
    const test_float_compare_stats out_stats = test_compare_float_bits(
        ref_out_host, fused_out_host, (size_t)n_rows * n_embd);
    const test_float_compare_stats norm_stats = test_compare_float_bits(
        ref_norm_host, fused_norm_host, (size_t)n_rows * n_embd);
    fprintf(stderr,
            "ds4-test: batch HC+RMSNorm exactness rows=%u "
            "split=%zu/%llu max_ulp=%u max_abs=%g, "
            "collapse=%zu/%llu max_ulp=%u max_abs=%g, "
            "norm=%zu/%llu max_ulp=%u max_abs=%g\n",
            n_rows,
            split_stats.mismatch_count,
            (unsigned long long)(n_rows * mix_hc),
            split_stats.max_ulp,
            split_stats.max_abs,
            out_stats.mismatch_count,
            (unsigned long long)((uint64_t)n_rows * n_embd),
            out_stats.max_ulp,
            out_stats.max_abs,
            norm_stats.mismatch_count,
            (unsigned long long)((uint64_t)n_rows * n_embd),
            norm_stats.max_ulp,
            norm_stats.max_abs);
    TEST_ASSERT(split_stats.mismatch_count == 0);
    TEST_ASSERT(out_stats.mismatch_count == 0);
    TEST_ASSERT(norm_stats.mismatch_count == 0);

    free(mix_host);
    free(residual_host);
    free(ref_split_host);
    free(fused_split_host);
    free(ref_out_host);
    free(fused_out_host);
    free(ref_norm_host);
    free(fused_norm_host);
    ds4_gpu_tensor_free(mix);
    ds4_gpu_tensor_free(residual);
    ds4_gpu_tensor_free(ref_split);
    ds4_gpu_tensor_free(fused_split);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(fused_out);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(fused_norm);
    free(model_raw);
}

static void test_metal_output_hc_weights4_exact(void) {
    const uint32_t n_hc = 4;
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t scale_offset = 0;
    const uint64_t base_offset = page;
    const uint64_t model_alloc = 2u * page;
    const uint64_t bytes = n_hc * sizeof(float);
    const char *require_env = "DS4_METAL_REQUIRE_OUTPUT_HC_WEIGHTS4";
    char *saved_require = test_save_env(require_env);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *pre = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *candidate = ds4_gpu_tensor_alloc(bytes);
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(pre != NULL);
    TEST_ASSERT(reference != NULL);
    TEST_ASSERT(candidate != NULL);

    size_t total_mismatch = 0;
    uint32_t max_ulp = 0;
    const bool allocated = model_raw && pre && reference && candidate;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_alloc);
        float *scale = (float *)((uint8_t *)model_raw + scale_offset);
        float *base = (float *)((uint8_t *)model_raw + base_offset);

        for (uint32_t ci = 0; ci < 4; ci++) {
            float pre_host[4];
            float ref_host[4];
            float candidate_host[4];
            for (uint32_t i = 0; i < 4; i++) {
                pre_host[i] =
                    ((float)((int)(ci * 17u + i * 11u) - 23)) / 8.0f;
                base[i] =
                    ((float)((int)(ci * 13u + i * 7u) - 19)) / 16.0f;
            }
            scale[0] = 0.62500012f + (float)ci * 0.125f;

            if (ci == 1) {
                /* The first lane distinguishes the required two-rounding
                 * sequence from an illegally contracted multiply-add. */
                const uint32_t pre_bits = 0x4620a541u;
                const uint32_t scale_bits = 0x462483bau;
                const uint32_t base_bits = 0xccce790eu;
                memcpy(&pre_host[0], &pre_bits, sizeof(pre_bits));
                memcpy(&scale[0], &scale_bits, sizeof(scale_bits));
                memcpy(&base[0], &base_bits, sizeof(base_bits));
            } else if (ci == 2) {
                const uint32_t pre_bits[4] = {
                    0x00000000u, 0x80000000u,
                    0x00000001u, 0x80000001u,
                };
                const uint32_t base_bits[4] = {
                    0x80000000u, 0x00000000u,
                    0x00800000u, 0x80800000u,
                };
                for (uint32_t i = 0; i < 4; i++) {
                    memcpy(&pre_host[i], &pre_bits[i], sizeof(uint32_t));
                    memcpy(&base[i], &base_bits[i], sizeof(uint32_t));
                }
                scale[0] = -1.00000012f;
            } else if (ci == 3) {
                pre_host[0] = 100.0f;
                pre_host[1] = -100.0f;
                pre_host[2] = 88.0f;
                pre_host[3] = -88.0f;
                scale[0] = 1.0f;
                for (uint32_t i = 0; i < 4; i++) base[i] = 0.0f;
            }

            memset(ref_host, 0xa5, sizeof(ref_host));
            memset(candidate_host, 0xa5, sizeof(candidate_host));
            TEST_ASSERT(ds4_gpu_tensor_write(pre, 0, pre_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            reference, 0, ref_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            candidate, 0, candidate_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
            TEST_ASSERT(unsetenv(require_env) == 0);
            ds4_gpu_test_set_flags(0);
            ds4_gpu_set_quality(true);
            TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                            reference, pre, model_raw, model_alloc,
                            scale_offset, base_offset, n_hc, eps) != 0);

            ds4_gpu_set_quality(false);
            ds4_gpu_test_set_flags(DS4_GPU_TEST_OUTPUT_HC_WEIGHTS4);
            TEST_ASSERT(setenv(require_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                            candidate, pre, model_raw, model_alloc,
                            scale_offset, base_offset, n_hc, eps) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                            reference, 0, ref_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            candidate, 0, candidate_host, bytes) != 0);
            const test_float_compare_stats stats =
                test_compare_float_bits(ref_host, candidate_host, n_hc);
            fprintf(stderr,
                    "ds4-test: output HC weights4 exact case=%u "
                    "mismatch=%zu/%u max_ulp=%u max_abs=%g\n",
                    ci, stats.mismatch_count, n_hc,
                    stats.max_ulp, stats.max_abs);
            TEST_ASSERT(stats.mismatch_count == 0);
            total_mismatch += stats.mismatch_count;
            if (stats.max_ulp > max_ulp) max_ulp = stats.max_ulp;
        }

        /* Quality mode prevents strict selection of the fast path. */
        TEST_ASSERT(setenv(require_env, "1", 1) == 0);
        ds4_gpu_set_quality(true);
        TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                        candidate, pre, model_raw, model_alloc,
                        scale_offset, base_offset, n_hc, eps) == 0);
        ds4_gpu_set_quality(false);
        ds4_gpu_test_set_flags(0);
    }

    test_restore_env(require_env, saved_require);
    fprintf(stderr,
            "ds4-test: output HC weights4 total mismatch=%zu/16 max_ulp=%u\n",
            total_mismatch, max_ulp);
    TEST_ASSERT(total_mismatch == 0);
    TEST_ASSERT(max_ulp == 0);

    ds4_gpu_tensor_free(candidate);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(pre);
    free(model_raw);
}

static void test_metal_hc_rms_scale_project_f16_exact_shape(
        uint32_t in_dim,
        uint32_t seed) {
    const uint32_t out_dim = 24;
    /* One full 32-row matmul tile plus a tail row covers both load paths. */
    const uint32_t n_rows = 33;
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_offset = page;
    const uint64_t weight_bytes =
        (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t model_alloc = test_round_up_u64(
        weight_offset + weight_bytes, page);
    const uint64_t x_count = (uint64_t)in_dim * n_rows;
    const uint64_t out_count = (uint64_t)out_dim * n_rows;
    const uint64_t x_bytes = x_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);
    const uint64_t scale_bytes = (uint64_t)n_rows * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *scaled_out = ds4_gpu_tensor_alloc(out_bytes);
    /* Deliberately too small for the full-RMS fallback. */
    ds4_gpu_tensor *scale_scratch = ds4_gpu_tensor_alloc(scale_bytes);
    float *x_host = malloc((size_t)x_bytes);
    float *ref_norm_host = malloc((size_t)x_bytes);
    float *ref_out_host = malloc((size_t)out_bytes);
    float *scaled_out_host = malloc((size_t)out_bytes);
    float *scale_host = malloc((size_t)scale_bytes);
    float *expected_scale = malloc((size_t)scale_bytes);

    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref_norm != NULL);
    TEST_ASSERT(ref_out != NULL);
    TEST_ASSERT(scaled_out != NULL);
    TEST_ASSERT(scale_scratch != NULL);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref_norm_host != NULL);
    TEST_ASSERT(ref_out_host != NULL);
    TEST_ASSERT(scaled_out_host != NULL);
    TEST_ASSERT(scale_host != NULL);
    TEST_ASSERT(expected_scale != NULL);

    const bool allocated = model_raw && x && ref_norm && ref_out &&
        scaled_out && scale_scratch && x_host && ref_norm_host &&
        ref_out_host && scaled_out_host && scale_host && expected_scale;
    test_float_compare_stats scale_stats = {0};
    test_float_compare_stats out_stats = {0};
    if (allocated) {
        memset(model_raw, 0, (size_t)model_alloc);
        uint16_t *weights = (uint16_t *)((uint8_t *)model_raw + weight_offset);
        for (uint32_t o = 0; o < out_dim; o++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const uint32_t key = i * 37u + o * 1009u + seed * 53u +
                    ((i >> 4u) ^ (o * 17u));
                uint16_t bits;
                if (key % 257u == 0u) {
                    bits = (key & 1u) ? 0x8000u : 0x0000u;
                } else if (key % 263u == 0u) {
                    bits = (key & 1u) ? 0x8001u : 0x0001u;
                } else if (key % 269u == 0u) {
                    bits = (key & 1u) ? 0x8400u : 0x0400u;
                } else {
                    const int value = (int)(key % 127u) - 63;
                    bits = test_float_to_f16((float)value / 128.0f);
                }
                weights[(uint64_t)o * in_dim + i] = bits;
            }
        }

        static const uint32_t rounding_bits[] = {
            0x3f800fffu, 0x3f801000u, 0x3f801001u,
            0x3f802fffu, 0x3f803000u, 0x3f803001u,
            0xbf800fffu, 0xbf801000u, 0xbf801001u,
            0x3eaaaaabu, 0xbeaaaaabu,
        };
        static const int sentinel_exp[] = { -2, -10, 0, -4, -6, -1 };
        for (uint32_t row = 0; row < n_rows; row++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const uint32_t key = i * 131u + row * 977u + seed * 71u +
                    ((i >> 3u) ^ (row * 29u));
                const float sign = (key & 1u) ? -1.0f : 1.0f;
                float value;
                switch (row % 6u) {
                    case 0:
                        value = (float)((int)(key % 4093u) - 2046) / 512.0f;
                        break;
                    case 1:
                        value = sign * ldexpf(
                            (float)(1u + (key & 7u)) / 8.0f, -19);
                        break;
                    case 2:
                        value = i % 257u == row % 257u
                            ? sign * (16.0f + (float)(key & 7u))
                            : sign * (float)(1u + (key & 31u)) / 8192.0f;
                        break;
                    case 3: {
                        const uint32_t bits = rounding_bits[
                            key % (sizeof(rounding_bits) /
                                   sizeof(rounding_bits[0]))];
                        memcpy(&value, &bits, sizeof(value));
                        break;
                    }
                    case 4:
                        if ((key & 7u) == 0u) {
                            const uint32_t bits = (key & 8u) ? 0x80000000u : 0u;
                            memcpy(&value, &bits, sizeof(value));
                        } else {
                            value = sign * ldexpf(
                                1.0f + (float)(key & 3u) * 0.25f,
                                (int)((key >> 4u) % 14u) - 10);
                        }
                        break;
                    default:
                        value = sign * (float)(1u + (key % 251u)) / 256.0f;
                        break;
                }
                x_host[(uint64_t)row * in_dim + i] = value;
            }
            x_host[(uint64_t)row * in_dim] =
                ldexpf(1.0f, sentinel_exp[row % 6u]);
        }

        for (uint64_t i = 0; i < out_count; i++) {
            const uint32_t bits = 0x7fc01000u + (uint32_t)(i & 0xfffu);
            memcpy(scaled_out_host + i, &bits, sizeof(bits));
        }
        for (uint32_t row = 0; row < n_rows; row++) {
            const uint32_t bits = 0x7fc02000u + row;
            memcpy(scale_host + row, &bits, sizeof(bits));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        scaled_out, 0, scaled_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        scale_scratch, 0, scale_host, scale_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
        ds4_gpu_set_quality(false);

        int ref_begun = ds4_gpu_begin_commands();
        int ref_ok = ref_begun;
        if (ref_ok) ref_ok = ds4_gpu_rms_norm_plain_rows_tensor(
            ref_norm, x, in_dim, n_rows, eps);
        if (ref_ok) ref_ok = ds4_gpu_matmul_f16_tensor(
            ref_out, model_raw, model_alloc, weight_offset,
            in_dim, out_dim, ref_norm, n_rows);
        const int ref_end = ref_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(ref_ok != 0);
        TEST_ASSERT(ref_end != 0);

        int scaled_begun = ds4_gpu_begin_commands();
        int scaled_ok = scaled_begun;
        if (scaled_ok) scaled_ok = ds4_gpu_hc_rms_scale_project_f16_tensor(
            scaled_out, scale_scratch, model_raw, model_alloc, weight_offset,
            in_dim, out_dim, x, n_rows, eps);
        const int scaled_end = scaled_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(scaled_ok != 0);
        TEST_ASSERT(scaled_end != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_norm, 0, ref_norm_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_out, 0, ref_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        scaled_out, 0, scaled_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        scale_scratch, 0, scale_host, scale_bytes) != 0);

        for (uint32_t row = 0; row < n_rows; row++) {
            expected_scale[row] = ldexpf(
                ref_norm_host[(uint64_t)row * in_dim],
                -sentinel_exp[row % 6u]);
            TEST_ASSERT(isfinite(expected_scale[row]));
            TEST_ASSERT(isfinite(scale_host[row]));
        }
        scale_stats = test_compare_float_bits(
            expected_scale, scale_host, n_rows);
        out_stats = test_compare_float_bits(
            ref_out_host, scaled_out_host, (size_t)out_count);
    }

    fprintf(stderr,
            "ds4-test: HC RMS scale F16 projection exact K=%u rows=%u "
            "scale=%zu/%u max_ulp=%u projection=%zu/%llu max_ulp=%u\n",
            in_dim, n_rows,
            scale_stats.mismatch_count, n_rows, scale_stats.max_ulp,
            out_stats.mismatch_count, (unsigned long long)out_count,
            out_stats.max_ulp);
    TEST_ASSERT(scale_stats.mismatch_count == 0);
    TEST_ASSERT(scale_stats.max_ulp == 0);
    TEST_ASSERT(out_stats.mismatch_count == 0);
    TEST_ASSERT(out_stats.max_ulp == 0);

    free(expected_scale);
    free(scale_host);
    free(scaled_out_host);
    free(ref_out_host);
    free(ref_norm_host);
    free(x_host);
    ds4_gpu_tensor_free(scale_scratch);
    ds4_gpu_tensor_free(scaled_out);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_hc_rms_scale_project_f16_exact(void) {
    const char *disable_env = "DS4_METAL_DISABLE_HC_RMS_SCALE_PROJ";
    char *saved_disable = test_save_env(disable_env);

    TEST_ASSERT(unsetenv(disable_env) == 0);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_HC_RMS_SCALE_PROJ);
    test_metal_hc_rms_scale_project_f16_exact_shape(16384u, 59u);
    test_metal_hc_rms_scale_project_f16_exact_shape(28672u, 61u);
    ds4_gpu_test_set_flags(0);

    test_restore_env(disable_env, saved_disable);
}

/* The fused decode output-head dispatch must reproduce the stock
 * rms_norm_plain + matmul_f16 pair bit-for-bit on every shape whose
 * reduction trees it replicates, and fail closed outside that class. */
static void test_metal_f16_rms_norm_mv_exact_shape(
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t seed) {
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_offset = page;
    const uint64_t weight_bytes =
        (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t model_alloc = test_round_up_u64(
        weight_offset + weight_bytes, page);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_out = ds4_gpu_tensor_alloc(out_bytes);
    float *x_host = malloc((size_t)x_bytes);
    float *ref_out_host = malloc((size_t)out_bytes);
    float *fused_out_host = malloc((size_t)out_bytes);

    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref_norm != NULL);
    TEST_ASSERT(ref_out != NULL);
    TEST_ASSERT(fused_out != NULL);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref_out_host != NULL);
    TEST_ASSERT(fused_out_host != NULL);

    test_float_compare_stats out_stats = {0};
    const bool allocated = model_raw && x && ref_norm && ref_out &&
        fused_out && x_host && ref_out_host && fused_out_host;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_alloc);
        uint16_t *weights = (uint16_t *)((uint8_t *)model_raw + weight_offset);
        for (uint32_t o = 0; o < out_dim; o++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const uint32_t key = i * 37u + o * 1009u + seed * 53u +
                    ((i >> 4u) ^ (o * 17u));
                uint16_t bits;
                if (key % 257u == 0u) {
                    bits = (key & 1u) ? 0x8000u : 0x0000u;
                } else if (key % 269u == 0u) {
                    bits = (key & 1u) ? 0x8400u : 0x0400u;
                } else {
                    const int value = (int)(key % 127u) - 63;
                    bits = test_float_to_f16((float)value / 128.0f);
                }
                weights[(uint64_t)o * in_dim + i] = bits;
            }
        }
        static const uint32_t rounding_bits[] = {
            0x3f800fffu, 0x3f801000u, 0x3f801001u,
            0xbf800fffu, 0xbf801000u, 0xbf801001u,
            0x3eaaaaabu, 0xbeaaaaabu,
        };
        for (uint32_t i = 0; i < in_dim; i++) {
            const uint32_t key = i * 131u + seed * 71u + (i >> 3u);
            const float sign = (key & 1u) ? -1.0f : 1.0f;
            float value;
            switch (i % 4u) {
                case 0:
                    value = (float)((int)(key % 4093u) - 2046) / 512.0f;
                    break;
                case 1:
                    value = sign * ldexpf(
                        (float)(1u + (key & 7u)) / 8.0f, -19);
                    break;
                case 2: {
                    const uint32_t bits = rounding_bits[
                        key % (sizeof(rounding_bits) /
                               sizeof(rounding_bits[0]))];
                    memcpy(&value, &bits, sizeof(value));
                    break;
                }
                default:
                    value = sign * (float)(1u + (key % 251u)) / 256.0f;
                    break;
            }
            x_host[i] = value;
        }
        for (uint32_t o = 0; o < out_dim; o++) {
            const uint32_t bits = 0x7fc01000u + o;
            memcpy(fused_out_host + o, &bits, sizeof(bits));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_out, 0, fused_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
        ds4_gpu_set_quality(false);

        int ref_begun = ds4_gpu_begin_commands();
        int ref_ok = ref_begun;
        if (ref_ok) ref_ok = ds4_gpu_rms_norm_plain_tensor(
            ref_norm, x, in_dim, eps);
        if (ref_ok) ref_ok = ds4_gpu_matmul_f16_tensor(
            ref_out, model_raw, model_alloc, weight_offset,
            in_dim, out_dim, ref_norm, 1);
        const int ref_end = ref_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(ref_ok != 0);
        TEST_ASSERT(ref_end != 0);

        int fused_begun = ds4_gpu_begin_commands();
        int fused_ok = fused_begun;
        if (fused_ok) fused_ok = ds4_gpu_matmul_f16_rms_norm_mv_tensor(
            fused_out, model_raw, model_alloc, weight_offset,
            in_dim, out_dim, x, eps);
        const int fused_end = fused_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(fused_ok != 0);
        TEST_ASSERT(fused_end != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_out, 0, ref_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_out, 0, fused_out_host, out_bytes) != 0);
        out_stats = test_compare_float_bits(
            ref_out_host, fused_out_host, (size_t)out_dim);
    }

    fprintf(stderr,
            "ds4-test: fused RMS-norm F16 matvec exact in=%u out=%u "
            "projection=%zu/%u max_ulp=%u\n",
            in_dim, out_dim,
            out_stats.mismatch_count, out_dim, out_stats.max_ulp);
    TEST_ASSERT(out_stats.mismatch_count == 0);
    TEST_ASSERT(out_stats.max_ulp == 0);

    free(fused_out_host);
    free(ref_out_host);
    free(x_host);
    ds4_gpu_tensor_free(fused_out);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_f16_rms_norm_mv_exact(void) {
    /* Public dimensions become signed shader fields.  Oversized values must
     * fail during admission, before an optional PSO probe or command batch. */
    TEST_ASSERT(ds4_gpu_matmul_f16_rms_norm_mv_preflight(
                    (uint32_t)INT32_MAX + 1u, 4u) == 0);
    TEST_ASSERT(ds4_gpu_matmul_f16_rms_norm_mv_preflight(
                    16384u, (uint32_t)INT32_MAX + 1u) == 0);
    const char *output_selector =
        getenv("DS4_METAL_LAGUNA_OUTPUT_HEAD_NORM_FUSE");
    const bool output_selector_off =
        !output_selector || output_selector[0] == '\0' ||
        strcmp(output_selector, "0") == 0;
    const int output_pso_available =
        ds4_gpu_matmul_f16_rms_norm_mv_preflight(16384u, 4u);
    if (output_pso_available == 0) {
        /* A pre-feature DENSE source is valid for the default-off suite.  An
         * explicit selector still proves the optional PSO admission fails
         * closed; graph admission performs the same check before allocation. */
        TEST_ASSERT(output_selector_off ||
                    strcmp(output_selector, "1") == 0);
        fprintf(stderr,
                "ds4-test: skipping fused RMS-norm F16 exact suite; "
                "optional output PSO unavailable (selector=%s)\n",
                output_selector ? output_selector : "unset");
        return;
    }
    test_metal_f16_rms_norm_mv_exact_shape(16384u, 4u, 59u);
    test_metal_f16_rms_norm_mv_exact_shape(4096u, 3u, 61u);
    test_metal_f16_rms_norm_mv_exact_shape(2048u, 5u, 67u);
    test_metal_f16_rms_norm_mv_exact_shape(1024u, 2u, 71u);

    /* Outside the replicated reduction trees the fused dispatch must fail
     * closed: NSG<8 (896), non-power norm sweep (288 threads), and a
     * reduction dim the vectorized matvec cannot tile. */
    void *model_raw = NULL;
    const uint64_t page = (uint64_t)getpagesize();
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(928u * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(4u * sizeof(float));
    TEST_ASSERT(model_raw != NULL && x != NULL && out != NULL);
    if (model_raw && x && out) {
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        int begun = ds4_gpu_begin_commands();
        TEST_ASSERT(begun != 0);
        TEST_ASSERT(ds4_gpu_matmul_f16_rms_norm_mv_tensor(
                        out, model_raw, page, 0, 896u, 4u, x, 1.0e-6f) == 0);
        TEST_ASSERT(ds4_gpu_matmul_f16_rms_norm_mv_tensor(
                        out, model_raw, page, 0, 928u, 4u, x, 1.0e-6f) == 0);
        TEST_ASSERT(ds4_gpu_matmul_f16_rms_norm_mv_tensor(
                        out, model_raw, page, 0, 1000u, 4u, x, 1.0e-6f) == 0);
        TEST_ASSERT(ds4_gpu_end_commands() != 0);
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

/* The fused Laguna decode router must reproduce the stock
 * matmul_f32_decode_rows_exact + glm_router_select_batch pair bit-for-bit,
 * including the staged-logits device copy and the non-finite fallback. */
static void test_metal_laguna_router_fused_exact_case(
        uint32_t in_dim,
        uint32_t pattern,
        uint32_t seed) {
    const uint32_t n_expert = 256u;
    const uint32_t n_used = 10u;
    const float scale = 2.5f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_offset = page;
    const uint64_t weight_bytes =
        (uint64_t)n_expert * in_dim * sizeof(float);
    const uint64_t bias_offset = weight_offset + weight_bytes;
    const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t model_alloc = test_round_up_u64(
        bias_offset + bias_bytes, page);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t expert_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)n_used * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)n_used * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_logits = ds4_gpu_tensor_alloc(expert_bytes);
    ds4_gpu_tensor *fused_logits = ds4_gpu_tensor_alloc(expert_bytes);
    ds4_gpu_tensor *ref_probs = ds4_gpu_tensor_alloc(expert_bytes);
    ds4_gpu_tensor *fused_probs = ds4_gpu_tensor_alloc(expert_bytes);
    ds4_gpu_tensor *ref_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *fused_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *ref_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *fused_weights = ds4_gpu_tensor_alloc(weights_bytes);
    float *x_host = malloc((size_t)x_bytes);
    float *ref_logits_host = malloc((size_t)expert_bytes);
    float *fused_logits_host = malloc((size_t)expert_bytes);
    float *ref_probs_host = malloc((size_t)expert_bytes);
    float *fused_probs_host = malloc((size_t)expert_bytes);
    int32_t *ref_selected_host = malloc((size_t)selected_bytes);
    int32_t *fused_selected_host = malloc((size_t)selected_bytes);
    float *ref_weights_host = malloc((size_t)weights_bytes);
    float *fused_weights_host = malloc((size_t)weights_bytes);

    TEST_ASSERT(model_raw && x && ref_logits && fused_logits && ref_probs &&
                fused_probs && ref_selected && fused_selected &&
                ref_weights && fused_weights && x_host && ref_logits_host &&
                fused_logits_host && ref_probs_host && fused_probs_host &&
                ref_selected_host && fused_selected_host && ref_weights_host &&
                fused_weights_host);

    const bool allocated = model_raw && x && ref_logits && fused_logits &&
        ref_probs && fused_probs && ref_selected && fused_selected &&
        ref_weights && fused_weights && x_host && ref_logits_host &&
        fused_logits_host && ref_probs_host && fused_probs_host &&
        ref_selected_host && fused_selected_host && ref_weights_host &&
        fused_weights_host;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_alloc);
        float *w = (float *)((uint8_t *)model_raw + weight_offset);
        float *bias = (float *)((uint8_t *)model_raw + bias_offset);
        for (uint32_t r = 0; r < n_expert; r++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const uint32_t key =
                    i * 41u + r * 787u + seed * 29u + ((i >> 3u) ^ (r * 7u));
                float value =
                    (float)((int)(key % 251u) - 125) / 4096.0f;
                if (pattern == 2u && r == 3u) {
                    value = 1.0e38f;
                }
                w[(uint64_t)r * in_dim + i] = value;
                if (pattern == 3u && r == 17u && i == 0u) {
                    const uint32_t qnan = 0x7fc00011u;
                    memcpy(&w[(uint64_t)r * in_dim + i],
                           &qnan, sizeof(qnan));
                }
                if (pattern == 4u && r == 23u && i == 0u) {
                    const uint32_t pinf = 0x7f800000u;
                    memcpy(&w[(uint64_t)r * in_dim + i],
                           &pinf, sizeof(pinf));
                }
            }
            const uint32_t bkey = r * 53u + seed * 11u;
            bias[r] = (float)((int)(bkey % 61u) - 30) / 128.0f;
        }
        for (uint32_t i = 0; i < in_dim; i++) {
            const uint32_t key = i * 97u + seed * 13u + (i >> 4u);
            const float sign = (key & 1u) ? -1.0f : 1.0f;
            x_host[i] = pattern == 1u
                ? 0.0f
                : sign * (float)(1u + (key % 509u)) / 256.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
        ds4_gpu_set_quality(false);

        int ref_begun = ds4_gpu_begin_commands();
        int ref_ok = ref_begun;
        if (ref_ok) ref_ok = ds4_gpu_matmul_f32_decode_rows_exact_tensor(
            ref_logits, model_raw, model_alloc, weight_offset,
            in_dim, n_expert, x, 1u);
        if (ref_ok) ref_ok = ds4_gpu_glm_router_select_batch_tensor(
            ref_selected, ref_weights, ref_probs,
            model_raw, model_alloc, bias_offset, ref_logits,
            n_expert, n_used, scale, 1u);
        const int ref_end = ref_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(ref_ok != 0);
        TEST_ASSERT(ref_end != 0);

        int fused_begun = ds4_gpu_begin_commands();
        int fused_ok = fused_begun;
        if (fused_ok) fused_ok = ds4_gpu_laguna_router_decode_fused_tensor(
            fused_selected, fused_weights, fused_probs, fused_logits,
            model_raw, model_alloc, weight_offset, bias_offset,
            x, in_dim, n_expert, n_used, scale);
        const int fused_end = fused_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(fused_ok != 0);
        TEST_ASSERT(fused_end != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_logits, 0, ref_logits_host, expert_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_logits, 0, fused_logits_host, expert_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_probs, 0, ref_probs_host, expert_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_probs, 0, fused_probs_host, expert_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_selected, 0, ref_selected_host, selected_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_selected, 0, fused_selected_host, selected_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_weights, 0, ref_weights_host, weights_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_weights, 0, fused_weights_host, weights_bytes) != 0);

        const test_float_compare_stats logits_stats = test_compare_float_bits(
            ref_logits_host, fused_logits_host, (size_t)n_expert);
        const test_float_compare_stats probs_stats = test_compare_float_bits(
            ref_probs_host, fused_probs_host, (size_t)n_expert);
        const test_float_compare_stats weights_stats = test_compare_float_bits(
            ref_weights_host, fused_weights_host, (size_t)n_used);
        fprintf(stderr,
                "ds4-test: fused Laguna router exact in=%u pattern=%u "
                "logits=%zu/%u probs=%zu/%u selected=%s weights=%zu/%u\n",
                in_dim, pattern,
                logits_stats.mismatch_count, n_expert,
                probs_stats.mismatch_count, n_expert,
                memcmp(ref_selected_host, fused_selected_host,
                       (size_t)selected_bytes) == 0 ? "exact" : "MISMATCH",
                weights_stats.mismatch_count, n_used);
        TEST_ASSERT(logits_stats.mismatch_count == 0);
        TEST_ASSERT(logits_stats.max_ulp == 0);
        TEST_ASSERT(probs_stats.mismatch_count == 0);
        TEST_ASSERT(weights_stats.mismatch_count == 0);
        TEST_ASSERT(weights_stats.max_ulp == 0);
        TEST_ASSERT(memcmp(ref_selected_host, fused_selected_host,
                           (size_t)selected_bytes) == 0);
    }

    free(fused_weights_host);
    free(ref_weights_host);
    free(fused_selected_host);
    free(ref_selected_host);
    free(fused_probs_host);
    free(ref_probs_host);
    free(fused_logits_host);
    free(ref_logits_host);
    free(x_host);
    ds4_gpu_tensor_free(fused_weights);
    ds4_gpu_tensor_free(ref_weights);
    ds4_gpu_tensor_free(fused_selected);
    ds4_gpu_tensor_free(ref_selected);
    ds4_gpu_tensor_free(fused_probs);
    ds4_gpu_tensor_free(ref_probs);
    ds4_gpu_tensor_free(fused_logits);
    ds4_gpu_tensor_free(ref_logits);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_laguna_router_fused_exact(void) {
    /* The reference select must stay on the stock bitonic kernel here: the
     * fused dispatch carries the SIMD body, and the pair is certified
     * equivalent by the SIMD top-k suite. */
    const char *simd_env = "DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK";
    char *saved_simd = test_save_env(simd_env);
    TEST_ASSERT(unsetenv(simd_env) == 0);
    const int fused_available = ds4_gpu_laguna_router_decode_fused_preflight(
        3072u, 256u, 10u, 2.5f);
    if (fused_available == 0) {
        /* A pre-feature misc source is valid for the default suite.  Its
         * requested-on fail-closed behavior is exercised by the source
         * override test, while the current in-repo source must remain a
         * hard failure if its optional PSO is unexpectedly absent. */
        const char *source = getenv("DS4_METAL_DSV4_MISC_SOURCE");
        TEST_ASSERT(source && source[0]);
        fprintf(stderr,
                "ds4-test: skipping fused Laguna router exact suite; "
                "optional misc PSO unavailable under source override\n");
        test_restore_env(simd_env, saved_simd);
        return;
    }
    TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_preflight(
                    3072u, 256u, 9u, 2.5f) == 0);
    ds4_gpu_laguna_router_decode_fused_stats_reset();

    test_metal_laguna_router_fused_exact_case(4096u, 0u, 59u);
    test_metal_laguna_router_fused_exact_case(4096u, 1u, 61u);
    test_metal_laguna_router_fused_exact_case(4096u, 2u, 67u);
    test_metal_laguna_router_fused_exact_case(4096u, 3u, 69u);
    test_metal_laguna_router_fused_exact_case(2048u, 0u, 71u);
    /* Production Laguna router shape: finite, tie, Inf, and NaN poison
     * cases all exercise the normal one-token fused dispatch geometry. */
    test_metal_laguna_router_fused_exact_case(3072u, 0u, 73u);
    test_metal_laguna_router_fused_exact_case(3072u, 1u, 79u);
    test_metal_laguna_router_fused_exact_case(3072u, 4u, 83u);
    test_metal_laguna_router_fused_exact_case(3072u, 3u, 89u);

    TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_preflight(
                    (uint32_t)INT32_MAX + 1u, 256u, 10u, 2.5f) == 0);
    TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_preflight(
                    3072u, (uint32_t)INT32_MAX + 1u, 10u, 2.5f) == 0);
    TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_preflight(
                    3072u, 256u, (uint32_t)INT32_MAX + 1u, 2.5f) == 0);

    uint64_t valid_encoded_dispatches = 0;
    uint64_t valid_completed_dispatches = 0;
    TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_stats(
                    &valid_encoded_dispatches,
                    &valid_completed_dispatches) != 0);
    TEST_ASSERT(valid_encoded_dispatches == 9u);
    TEST_ASSERT(valid_completed_dispatches == valid_encoded_dispatches);

    /* Outside the replicated shape class the fused dispatch fails closed. */
    void *model_raw = NULL;
    const uint64_t page = (uint64_t)getpagesize();
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    const uint64_t x_bytes = 4096u * sizeof(float);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(256u * sizeof(float));
    ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(256u * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(10u * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(10u * sizeof(float));
    TEST_ASSERT(model_raw && x && logits && probs && selected && weights);
    if (model_raw && x && logits && probs && selected && weights) {
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        int begun = ds4_gpu_begin_commands();
        TEST_ASSERT(begun != 0);
        TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_tensor(
                        selected, weights, probs, logits,
                        model_raw, page, 0, 0,
                        x, 4096u, 255u, 10u, 2.5f) == 0);
        TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_tensor(
                        selected, weights, probs, logits,
                        model_raw, page, 0, 0,
                        x, 1000u, 256u, 10u, 2.5f) == 0);
        TEST_ASSERT(ds4_gpu_end_commands() != 0);
    }
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(probs);
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(x);
    free(model_raw);

    /* Failed preflights/tensor calls must not manufacture completion evidence
     * even when invoked inside a caller-owned command batch. */
    ds4_gpu_laguna_router_decode_fused_stats_reset();
    model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    x = ds4_gpu_tensor_alloc(x_bytes);
    logits = ds4_gpu_tensor_alloc(256u * sizeof(float));
    probs = ds4_gpu_tensor_alloc(256u * sizeof(float));
    selected = ds4_gpu_tensor_alloc(10u * sizeof(int32_t));
    weights = ds4_gpu_tensor_alloc(10u * sizeof(float));
    TEST_ASSERT(model_raw && x && logits && probs && selected && weights);
    if (model_raw && x && logits && probs && selected && weights) {
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        const int begun = ds4_gpu_begin_commands();
        TEST_ASSERT(begun != 0);
        TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_tensor(
                        selected, weights, probs, logits,
                        model_raw, page, 0, 0,
                        x, (uint32_t)INT32_MAX + 1u, 256u, 10u, 2.5f) == 0);
        TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_tensor(
                        selected, weights, probs, logits,
                        model_raw, page, 0, 0,
                        x, 3072u, (uint32_t)INT32_MAX + 1u, 10u, 2.5f) == 0);
        TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_tensor(
                        selected, weights, probs, logits,
                        model_raw, page, 0, 0,
                        x, 3072u, 256u,
                        (uint32_t)INT32_MAX + 1u, 2.5f) == 0);
        TEST_ASSERT(ds4_gpu_end_commands() != 0);
    }
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(probs);
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(x);
    free(model_raw);

    uint64_t encoded_dispatches = 0;
    uint64_t completed_dispatches = 0;
    TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_stats(
                    &encoded_dispatches, &completed_dispatches) != 0);
    TEST_ASSERT(encoded_dispatches == 0u);
    TEST_ASSERT(completed_dispatches == 0u);

    test_restore_env(simd_env, saved_simd);
}

static void test_metal_router_simd_finalize_exact(void) {
    typedef struct {
        const char *name;
        bool has_bias;
        uint32_t pattern;
    } router_case;
    static const router_case cases[] = {
        { "unique", false, 0 },
        { "bias", true, 0 },
        { "top6-ties", false, 1 },
        { "signed-zero-extremes", false, 2 },
        { "clamp-underflow", false, 3 },
        { "sum-rounding", true, 4 },
    };
    const uint32_t n_expert = 256;
    const uint32_t n_used = 6;
    const uint32_t modes = 4;
    const uint64_t probs_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)n_used * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)n_used * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();
    const char *disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_SIMD_FINALIZE";
    const char *weights_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_SIMD_WEIGHTS_FUSION";
    const char *transform_finalize_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_TRANSFORM_FINALIZE_FUSION";
    const char *select_disable_env =
        "DS4_METAL_DISABLE_ROUTER_SELECT_FUSION";

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(probs_bytes);
    ds4_gpu_tensor *ref_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *simd_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *ref_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *simd_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *ref_probs = ds4_gpu_tensor_alloc(probs_bytes);
    ds4_gpu_tensor *simd_probs = ds4_gpu_tensor_alloc(probs_bytes);
    float *logits_host = malloc((size_t)probs_bytes);
    float *ref_probs_host = malloc((size_t)probs_bytes);
    float *simd_probs_host = malloc((size_t)probs_bytes);
    int32_t ref_selected_host[6];
    int32_t simd_selected_host[6];
    int32_t unique_selected_host[6];
    float ref_weights_host[6];
    float simd_weights_host[6];
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(ref_selected != NULL);
    TEST_ASSERT(simd_selected != NULL);
    TEST_ASSERT(ref_weights != NULL);
    TEST_ASSERT(simd_weights != NULL);
    TEST_ASSERT(ref_probs != NULL);
    TEST_ASSERT(simd_probs != NULL);
    TEST_ASSERT(logits_host != NULL);
    TEST_ASSERT(ref_probs_host != NULL);
    TEST_ASSERT(simd_probs_host != NULL);

    char *saved_disable = test_save_env(disable_env);
    char *saved_weights_disable = test_save_env(weights_disable_env);
    char *saved_transform_finalize_disable =
        test_save_env(transform_finalize_disable_env);
    char *saved_select_disable = test_save_env(select_disable_env);
    size_t total_selected_mismatch = 0;
    size_t total_weights_mismatch = 0;
    size_t total_probs_mismatch = 0;
    const bool allocated = model_raw && logits && ref_selected &&
        simd_selected && ref_weights && simd_weights && ref_probs &&
        simd_probs && logits_host && ref_probs_host && simd_probs_host;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *bias = model_raw;
        for (uint32_t i = 0; i < n_used; i++) {
            bias[200u + i] = 16.0f - (float)i;
        }
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(setenv(transform_finalize_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(select_disable_env) == 0);

        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
            const router_case *c = &cases[ci];
            for (uint32_t i = 0; i < n_expert; i++) {
                const int value =
                    (int)((i * 47u + (i ^ (i >> 3u)) * 13u) % 257u) - 128;
                logits_host[i] = (float)value / 32.0f;
            }
            if (c->pattern == 1) {
                static const uint32_t tied[] = {
                    7u, 19u, 43u, 71u, 103u, 149u, 211u, 239u,
                };
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -7.0f - (float)(i % 17u) / 64.0f;
                }
                for (size_t i = 0; i < sizeof(tied) / sizeof(tied[0]); i++) {
                    logits_host[tied[i]] = 1.0f;
                }
            } else if (c->pattern == 2) {
                for (uint32_t i = 0; i < n_expert; i++) {
                    const uint32_t zero_bits = (i & 1u) ? 0x80000000u : 0u;
                    memcpy(&logits_host[i], &zero_bits, sizeof(zero_bits));
                }
                logits_host[3] = 80.0f;
                logits_host[17] = 40.0f;
                logits_host[61] = 20.0f;
                logits_host[127] = -20.0f;
                logits_host[193] = -40.0f;
                logits_host[251] = -80.0f;
            } else if (c->pattern == 3) {
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -30.0f - (float)(i % 11u);
                }
            } else if (c->pattern == 4) {
                static const float rounding_logits[6] = {
                    -0.6356699467f,
                    -0.8182631135f,
                    -2.7906901836f,
                    -3.4414808750f,
                    -3.4991359711f,
                    -3.1251864433f,
                };
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -20.0f;
                }
                for (uint32_t i = 0; i < n_used; i++) {
                    logits_host[200u + i] = rounding_logits[i];
                }
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits, 0, logits_host, probs_bytes) != 0);

            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_router_select_tensor(
                ref_selected, ref_weights, ref_probs,
                model_raw, page, 0, 0, 1, 0,
                n_expert, n_used, 1.5f, 1, 0,
                c->has_bias, false, logits) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_selected, 0, ref_selected_host, selected_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_weights, 0, ref_weights_host, weights_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_probs, 0, ref_probs_host, probs_bytes) != 0);

            if (ci == 0) {
                memcpy(unique_selected_host,
                       ref_selected_host,
                       sizeof(unique_selected_host));
            } else if (c->has_bias) {
                TEST_ASSERT(memcmp(unique_selected_host,
                                   ref_selected_host,
                                   sizeof(unique_selected_host)) != 0);
            }

            TEST_ASSERT(unsetenv(disable_env) == 0);
            for (uint32_t mode = 0; mode < modes; mode++) {
                const bool fused_weights = mode != 0;
                const bool fused_transform = mode >= 2;
                if (fused_weights) {
                    TEST_ASSERT(unsetenv(weights_disable_env) == 0);
                } else {
                    TEST_ASSERT(setenv(weights_disable_env, "1", 1) == 0);
                }
                if (fused_transform) {
                    TEST_ASSERT(unsetenv(transform_finalize_disable_env) == 0);
                } else {
                    TEST_ASSERT(setenv(
                        transform_finalize_disable_env, "1", 1) == 0);
                }
                for (uint32_t i = 0; i < n_expert; i++) {
                    const uint32_t poison_bits = 0x7fc00001u + i;
                    memcpy(&simd_probs_host[i],
                           &poison_bits,
                           sizeof(poison_bits));
                }
                TEST_ASSERT(ds4_gpu_tensor_write(
                    simd_probs, 0, simd_probs_host, probs_bytes) != 0);
                TEST_ASSERT(ds4_gpu_router_select_tensor(
                    simd_selected, simd_weights, simd_probs,
                    model_raw, page, 0, 0, 1, 0,
                    n_expert, n_used, 1.5f, 1, 0,
                    c->has_bias, false, logits) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_selected, 0, simd_selected_host,
                    selected_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_weights, 0, simd_weights_host,
                    weights_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_probs, 0, simd_probs_host, probs_bytes) != 0);

                size_t selected_mismatch = 0;
                for (uint32_t i = 0; i < n_used; i++) {
                    if (ref_selected_host[i] != simd_selected_host[i]) {
                        selected_mismatch++;
                    }
                }
                const test_float_compare_stats weights_stats =
                    test_compare_float_bits(
                        ref_weights_host, simd_weights_host, n_used);
                const test_float_compare_stats probs_stats =
                    test_compare_float_bits(
                        ref_probs_host, simd_probs_host, n_expert);
                fprintf(stderr,
                        "ds4-test: router SIMD finalize exactness "
                        "case=%s mode=%s rep=%u selected=%zu/%u weights=%zu/%u "
                        "probs=%zu/%u max_weight_ulp=%u max_prob_ulp=%u\n",
                        c->name,
                        fused_transform ? "fused-transform" :
                        fused_weights ? "fused-weights" : "split-weights",
                        fused_transform ? mode - 2u : 0u,
                        selected_mismatch,
                        n_used,
                        weights_stats.mismatch_count,
                        n_used,
                        probs_stats.mismatch_count,
                        n_expert,
                        weights_stats.max_ulp,
                        probs_stats.max_ulp);
                TEST_ASSERT(selected_mismatch == 0);
                TEST_ASSERT(weights_stats.mismatch_count == 0);
                TEST_ASSERT(probs_stats.mismatch_count == 0);
                total_selected_mismatch += selected_mismatch;
                total_weights_mismatch += weights_stats.mismatch_count;
                total_probs_mismatch += probs_stats.mismatch_count;
            }
        }
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(weights_disable_env, saved_weights_disable);
    test_restore_env(transform_finalize_disable_env,
                     saved_transform_finalize_disable);
    test_restore_env(select_disable_env, saved_select_disable);
    fprintf(stderr,
            "ds4-test: router SIMD finalize total selected=%zu/%zu "
            "weights=%zu/%zu probs=%zu/%zu\n",
            total_selected_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_used,
            total_weights_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_used,
            total_probs_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_expert);
    TEST_ASSERT(total_selected_mismatch == 0);
    TEST_ASSERT(total_weights_mismatch == 0);
    TEST_ASSERT(total_probs_mismatch == 0);

    free(simd_probs_host);
    free(ref_probs_host);
    free(logits_host);
    ds4_gpu_tensor_free(simd_probs);
    ds4_gpu_tensor_free(ref_probs);
    ds4_gpu_tensor_free(simd_weights);
    ds4_gpu_tensor_free(ref_weights);
    ds4_gpu_tensor_free(simd_selected);
    ds4_gpu_tensor_free(ref_selected);
    ds4_gpu_tensor_free(logits);
    free(model_raw);
}

/* An old DS4_METAL_DSV4_MISC_SOURCE is a deliberate source-level override,
 * not a promise that the opt-in Laguna selector exists in that source.  Keep
 * this check separate so the default --metal-kernels group never turns on a
 * kernel that an override does not provide. */
static void test_metal_glm_router_simd_topk_exact_suite(void);
static void test_metal_laguna_router_fused_exact(void);

/* The default-off route must remain usable even when an optional source
 * override predates both opt-in selectors.  This is intentionally a small
 * stock dispatch rather than a SIMD/fused availability probe. */
static void test_metal_glm_router_stock_default(void) {
    const uint32_t n_expert = 256u;
    const uint32_t n_used = 10u;
    const uint64_t page = (uint64_t)getpagesize();
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(
        (uint64_t)n_expert * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(
        (uint64_t)n_used * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(
        (uint64_t)n_used * sizeof(float));
    ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(
        (uint64_t)n_expert * sizeof(float));
    float *logits_host = malloc((size_t)n_expert * sizeof(float));
    if (model_raw && logits && selected && weights && probs && logits_host) {
        memset(model_raw, 0, (size_t)page);
        for (uint32_t i = 0; i < n_expert; i++) {
            logits_host[i] = (float)((int)(i * 17u % 101u) - 50) / 32.0f;
        }
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        logits, 0, logits_host,
                        (uint64_t)n_expert * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_glm_router_select_tensor(
                        selected, weights, probs,
                        model_raw, page, 0, logits,
                        n_expert, n_used, 2.5f) != 0);
    }
    free(logits_host);
    ds4_gpu_tensor_free(probs);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(logits);
    free(model_raw);
}

static void test_metal_glm_router_simd_topk_source_override(void) {
    const char *source = getenv("DS4_METAL_DSV4_MISC_SOURCE");
    if (!source || !source[0]) return;

    const char *enable_env = "DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK";
    const char *fused_enable_env = "DS4_METAL_LAGUNA_ROUTER_DECODE_FUSED";
    const char *trace_env = "DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK_TRACE";
    const char *legacy_enable_env = "DS4_METAL_ENABLE_GLM_ROUTER_SIMD_TOPK";
    const char *legacy_disable_env = "DS4_METAL_DISABLE_GLM_ROUTER_SIMD_TOPK";
    char *saved_enable = test_save_env(enable_env);
    char *saved_fused_enable = test_save_env(fused_enable_env);
    char *saved_trace = test_save_env(trace_env);
    char *saved_legacy_enable = test_save_env(legacy_enable_env);
    char *saved_legacy_disable = test_save_env(legacy_disable_env);

    TEST_ASSERT(unsetenv(enable_env) == 0);
    TEST_ASSERT(unsetenv(fused_enable_env) == 0);
    TEST_ASSERT(unsetenv(trace_env) == 0);
    TEST_ASSERT(unsetenv(legacy_enable_env) == 0);
    TEST_ASSERT(unsetenv(legacy_disable_env) == 0);
    /* The override must be loaded before probing for the optional function.
     * This also makes a current-source override distinguishable from an old
     * source that predates kernel_glm_router_select_one_simd. */
    TEST_ASSERT(ds4_gpu_init() != 0);
    TEST_ASSERT(ds4_gpu_laguna_router_simd_topk_preflight(
        256u, 10u, 2.5f) == 0);

    TEST_ASSERT(setenv(enable_env, "1", 1) == 0);
    const int mode = ds4_gpu_laguna_router_simd_topk_preflight(
        256u, 10u, 2.5f);
    /* Probe the fused PSO independently of the legacy SIMD selector. */
    const int fused_available = ds4_gpu_laguna_router_decode_fused_preflight(
        3072u, 256u, 10u, 2.5f);
    /* Classify the source by the fused router PSO independently.  An older
     * misc source may still carry the legacy SIMD top-k PSO; that must not
     * make its default-off stock route, or its fused opt-in failure, look
     * like the current source.  Conversely, a source advertising the fused
     * PSO without the legacy selector is not a current certified source. */
    const bool current_source = fused_available > 0;
    const bool old_source = fused_available == 0;
    TEST_ASSERT(current_source || old_source);
    if (current_source) {
        TEST_ASSERT(mode > 0);
        fprintf(stderr,
                "ds4-test: Laguna router current misc source override "
                "SIMD+fused PSOs available source=%s\n",
                source);
        TEST_ASSERT(setenv(fused_enable_env, "1", 1) == 0);
        /* Both optional kernels are independently exercised numerically. */
        test_metal_glm_router_simd_topk_exact_suite();
        test_metal_laguna_router_fused_exact();
    } else if (old_source) {
        TEST_ASSERT(setenv(fused_enable_env, "1", 1) == 0);
        ds4_gpu_laguna_router_decode_fused_stats_reset();
        /* The legacy selector was enabled for independent probing above.
         * Clear it before the default-off stock proof so an old source with
         * only the legacy PSO cannot accidentally exercise an optional path. */
        TEST_ASSERT(unsetenv(enable_env) == 0);
        test_metal_glm_router_stock_default();
        /* Explicit opt-in remains fail-closed, with no fused dispatch
         * evidence, even though the stock selector remains usable off. */
        const uint64_t page = (uint64_t)getpagesize();
        void *model_raw = NULL;
        TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(3072u * sizeof(float));
        ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(256u * sizeof(float));
        ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(256u * sizeof(float));
        ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(10u * sizeof(int32_t));
        ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(10u * sizeof(float));
        if (model_raw && x && logits && probs && selected && weights) {
            TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
            TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_tensor(
                            selected, weights, probs, logits,
                            model_raw, page, 0, 0, x,
                            3072u, 256u, 10u, 2.5f) == 0);
        }
        uint64_t encoded = 0;
        uint64_t completed = 0;
        TEST_ASSERT(ds4_gpu_laguna_router_decode_fused_stats(
                        &encoded, &completed) != 0);
        TEST_ASSERT(encoded == 0u && completed == 0u);
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(probs);
        ds4_gpu_tensor_free(logits);
        ds4_gpu_tensor_free(x);
        free(model_raw);
        fprintf(stderr,
                "ds4-test: Laguna router old misc source override "
                "stock=usable SIMD+fused=fail-closed source=%s\n",
                source);
    }

    test_restore_env(enable_env, saved_enable);
    test_restore_env(fused_enable_env, saved_fused_enable);
    test_restore_env(trace_env, saved_trace);
    test_restore_env(legacy_enable_env, saved_legacy_enable);
    test_restore_env(legacy_disable_env, saved_legacy_disable);
}

static void test_metal_glm_router_simd_topk_exact_suite(void) {
    typedef struct {
        const char *name;
        uint32_t n_expert;
        uint32_t n_used;
        uint32_t pattern;
        const char *enable_value;
        float expert_weight_scale;
        bool expect_failure;
    } router_case;
    static const router_case cases[] = {
        { "random-finite", 256u, 10u, 0u, "1",       2.5f, false },
        { "ties",          256u, 10u, 1u, "1",       2.5f, false },
        { "all-equal",     256u, 10u, 2u, "1",       2.5f, false },
        { "infinities",    256u, 10u, 3u, "1",       2.5f, false },
        { "nan-fallback",  256u, 10u, 4u, "1",       2.5f, false },
        { "bias-inf",      256u, 10u, 5u, "1",       2.5f, false },
        { "signed-zero",   256u, 10u, 6u, "1",       2.5f, false },
        { "shape-reject",  255u, 10u, 0u, "1",       2.5f, true },
        { "k-reject",      256u, 9u,  0u, "1",       2.5f, true },
        { "scale-reject",  256u, 10u, 0u, "1",       1.5f, true },
        { "literal-zero",  256u, 10u, 0u, "0",       2.5f, false },
        { "literal-empty", 256u, 10u, 0u, "",        2.5f, false },
        { "literal-true",  256u, 10u, 0u, "true",    2.5f, true },
        { "literal-yes",   256u, 10u, 0u, "yes",     2.5f, true },
        { "literal-bad",   256u, 10u, 0u, "enabled", 2.5f, true },
    };
    const uint32_t max_expert = 256u;
    const uint32_t max_used = 10u;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t logits_bytes = (uint64_t)max_expert * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)max_used * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)max_used * sizeof(float);
    const char *enable_env = "DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK";
    const char *trace_env = "DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK_TRACE";
    const char *legacy_enable_env = "DS4_METAL_ENABLE_GLM_ROUTER_SIMD_TOPK";
    const char *legacy_disable_env = "DS4_METAL_DISABLE_GLM_ROUTER_SIMD_TOPK";

    void *model_raw = NULL;
    ds4_gpu_tensor *logits = NULL;
    ds4_gpu_tensor *ref_selected = NULL;
    ds4_gpu_tensor *test_selected = NULL;
    ds4_gpu_tensor *ref_weights = NULL;
    ds4_gpu_tensor *test_weights = NULL;
    ds4_gpu_tensor *ref_probs = NULL;
    ds4_gpu_tensor *test_probs = NULL;
    float *logits_host = NULL;
    int32_t *ref_selected_host = NULL;
    int32_t *test_selected_host = NULL;
    float *ref_weights_host = NULL;
    float *test_weights_host = NULL;
    float *ref_probs_host = NULL;
    float *test_probs_host = NULL;

    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    logits = ds4_gpu_tensor_alloc(logits_bytes);
    ref_selected = ds4_gpu_tensor_alloc(selected_bytes);
    test_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ref_weights = ds4_gpu_tensor_alloc(weights_bytes);
    test_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ref_probs = ds4_gpu_tensor_alloc(logits_bytes);
    test_probs = ds4_gpu_tensor_alloc(logits_bytes);
    logits_host = malloc((size_t)logits_bytes);
    ref_selected_host = malloc((size_t)selected_bytes);
    test_selected_host = malloc((size_t)selected_bytes);
    ref_weights_host = malloc((size_t)weights_bytes);
    test_weights_host = malloc((size_t)weights_bytes);
    ref_probs_host = malloc((size_t)logits_bytes);
    test_probs_host = malloc((size_t)logits_bytes);
    TEST_ASSERT(model_raw && logits && ref_selected && test_selected &&
                ref_weights && test_weights && ref_probs && test_probs &&
                logits_host && ref_selected_host && test_selected_host &&
                ref_weights_host && test_weights_host && ref_probs_host &&
                test_probs_host);

    char *saved_enable = test_save_env(enable_env);
    char *saved_trace = test_save_env(trace_env);
    char *saved_legacy_enable = test_save_env(legacy_enable_env);
    char *saved_legacy_disable = test_save_env(legacy_disable_env);
    if (model_raw && logits && ref_selected && test_selected && ref_weights &&
        test_weights && ref_probs && test_probs && logits_host &&
        ref_selected_host && test_selected_host && ref_weights_host &&
        test_weights_host && ref_probs_host && test_probs_host) {
        memset(model_raw, 0, (size_t)page);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(setenv(trace_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(legacy_enable_env) == 0);
        TEST_ASSERT(unsetenv(legacy_disable_env) == 0);
        ds4_gpu_laguna_router_simd_topk_stats_reset();

        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
            const router_case *c = &cases[ci];
            float *bias = (float *)model_raw;
            for (uint32_t i = 0; i < max_expert; i++) {
                bias[i] = 0.0f;
                const int value =
                    (int)((i * 47u + (i ^ (i >> 3u)) * 13u) % 257u) - 128;
                logits_host[i] = (float)value / 32.0f;
            }
            if (c->pattern == 1u) {
                for (uint32_t i = 0; i < max_expert; i++) {
                    logits_host[i] = -4.0f - (float)(i % 17u) / 64.0f;
                }
                static const uint32_t tied[] = {
                    7u, 19u, 43u, 71u, 103u, 149u, 211u, 239u,
                };
                for (size_t i = 0; i < sizeof(tied) / sizeof(tied[0]); i++) {
                    logits_host[tied[i]] = 1.0f;
                }
            } else if (c->pattern == 2u) {
                for (uint32_t i = 0; i < max_expert; i++) logits_host[i] = 0.0f;
            } else if (c->pattern == 3u) {
                const uint32_t pinf = 0x7f800000u;
                const uint32_t ninf = 0xff800000u;
                memcpy(&logits_host[0], &pinf, sizeof(pinf));
                memcpy(&logits_host[1], &ninf, sizeof(ninf));
                memcpy(&logits_host[2], &pinf, sizeof(pinf));
                memcpy(&logits_host[3], &ninf, sizeof(ninf));
            } else if (c->pattern == 4u) {
                const uint32_t qnan = 0x7fc00000u;
                memcpy(&logits_host[17], &qnan, sizeof(qnan));
            } else if (c->pattern == 5u) {
                const uint32_t pinf = 0x7f800000u;
                memcpy(&bias[23], &pinf, sizeof(pinf));
            } else if (c->pattern == 6u) {
                for (uint32_t i = 0; i < max_expert; i++) {
                    const uint32_t zero = (i & 1u) ? 0x80000000u : 0u;
                    memcpy(&logits_host[i], &zero, sizeof(zero));
                }
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits, 0, logits_host, logits_bytes) != 0);

            TEST_ASSERT(unsetenv(enable_env) == 0);
            TEST_ASSERT(ds4_gpu_glm_router_select_tensor(
                ref_selected, ref_weights, ref_probs,
                model_raw, page, 0, logits,
                c->n_expert, c->n_used, c->expert_weight_scale) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_selected, 0, ref_selected_host, selected_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_weights, 0, ref_weights_host, weights_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_probs, 0, ref_probs_host, logits_bytes) != 0);

            TEST_ASSERT(setenv(enable_env, c->enable_value, 1) == 0);
            memset(test_selected_host, 0xa5, (size_t)selected_bytes);
            memset(test_weights_host, 0xa5, (size_t)weights_bytes);
            memset(test_probs_host, 0xa5, (size_t)logits_bytes);
            TEST_ASSERT(ds4_gpu_tensor_write(
                test_selected, 0, test_selected_host, selected_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                test_weights, 0, test_weights_host, weights_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                test_probs, 0, test_probs_host, logits_bytes) != 0);
            const int result = ds4_gpu_glm_router_select_tensor(
                test_selected, test_weights, test_probs,
                model_raw, page, 0, logits,
                c->n_expert, c->n_used, c->expert_weight_scale);
            if (c->expect_failure) {
                TEST_ASSERT(result == 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    test_selected, 0, test_selected_host, selected_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    test_weights, 0, test_weights_host, weights_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    test_probs, 0, test_probs_host, logits_bytes) != 0);
                for (uint32_t i = 0; i < max_used * sizeof(int32_t); i++) {
                    TEST_ASSERT(((const uint8_t *)test_selected_host)[i] == 0xa5);
                }
                for (uint32_t i = 0; i < max_used * sizeof(float); i++) {
                    TEST_ASSERT(((const uint8_t *)test_weights_host)[i] == 0xa5);
                }
                for (uint32_t i = 0; i < max_expert * sizeof(float); i++) {
                    TEST_ASSERT(((const uint8_t *)test_probs_host)[i] == 0xa5);
                }
                continue;
            }
            TEST_ASSERT(result != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                test_selected, 0, test_selected_host, selected_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                test_weights, 0, test_weights_host, weights_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                test_probs, 0, test_probs_host, logits_bytes) != 0);

            const size_t selected_mismatch = memcmp(
                ref_selected_host, test_selected_host,
                (size_t)c->n_used * sizeof(int32_t)) != 0;
            const test_float_compare_stats weight_stats = test_compare_float_bits(
                ref_weights_host, test_weights_host, c->n_used);
            const test_float_compare_stats prob_stats = test_compare_float_bits(
                ref_probs_host, test_probs_host, c->n_expert);
            fprintf(stderr,
                    "ds4-test: Laguna router SIMD top-k exactness case=%s "
                    "selected=%zu weights=%zu/%u probs=%zu/%u max_ulp=%u/%u\n",
                    c->name,
                    selected_mismatch,
                    weight_stats.mismatch_count, c->n_used,
                    prob_stats.mismatch_count, c->n_expert,
                    weight_stats.max_ulp, prob_stats.max_ulp);
            TEST_ASSERT(selected_mismatch == 0);
            TEST_ASSERT(weight_stats.mismatch_count == 0);
            TEST_ASSERT(prob_stats.mismatch_count == 0);
        }

        uint32_t optimized_rows = 0;
        uint32_t fallback_rows = 0;
        uint64_t encoded_rows = 0;
        uint64_t encoded_dispatches = 0;
        TEST_ASSERT(ds4_gpu_laguna_router_simd_topk_stats_after_wait(
            &optimized_rows, &fallback_rows,
            &encoded_rows, &encoded_dispatches) != 0);
        fprintf(stderr,
                "ds4-test: Laguna router SIMD top-k single stats "
                "optimized=%u fallback=%u encoded_rows=%llu "
                "encoded_dispatches=%llu\n",
                optimized_rows, fallback_rows,
                (unsigned long long)encoded_rows,
                (unsigned long long)encoded_dispatches);
        TEST_ASSERT(optimized_rows == 5u);
        TEST_ASSERT(fallback_rows == 2u);
        TEST_ASSERT(encoded_rows == 7u);
        TEST_ASSERT(optimized_rows + fallback_rows == encoded_rows);
        TEST_ASSERT(encoded_dispatches == 7u);

        /* A legacy flag is an explicit conflict, never a silent alias. */
        TEST_ASSERT(unsetenv(enable_env) == 0);
        TEST_ASSERT(setenv(legacy_enable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_glm_router_select_tensor(
            test_selected, test_weights, test_probs,
            model_raw, page, 0, logits, 256u, 10u, 2.5f) == 0);
        TEST_ASSERT(unsetenv(legacy_enable_env) == 0);
        TEST_ASSERT(setenv(legacy_disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_glm_router_select_tensor(
            test_selected, test_weights, test_probs,
            model_raw, page, 0, logits, 256u, 10u, 2.5f) == 0);
        TEST_ASSERT(unsetenv(legacy_disable_env) == 0);

        /* Exercise the public batch entry point with two finite rows and one
         * NaN row.  The extra bytes are guards against over-wide writes. */
        const uint32_t batch_tokens = 3u;
        const uint64_t guard = 64u;
        const uint64_t batch_logits_required =
            (uint64_t)batch_tokens * max_expert * sizeof(float);
        const uint64_t batch_selected_required =
            (uint64_t)batch_tokens * max_used * sizeof(int32_t);
        const uint64_t batch_weights_required =
            (uint64_t)batch_tokens * max_used * sizeof(float);
        ds4_gpu_tensor *batch_logits = ds4_gpu_tensor_alloc(
            batch_logits_required + guard);
        ds4_gpu_tensor *batch_ref_selected = ds4_gpu_tensor_alloc(
            batch_selected_required + guard);
        ds4_gpu_tensor *batch_test_selected = ds4_gpu_tensor_alloc(
            batch_selected_required + guard);
        ds4_gpu_tensor *batch_ref_weights = ds4_gpu_tensor_alloc(
            batch_weights_required + guard);
        ds4_gpu_tensor *batch_test_weights = ds4_gpu_tensor_alloc(
            batch_weights_required + guard);
        ds4_gpu_tensor *batch_ref_probs = ds4_gpu_tensor_alloc(
            batch_logits_required + guard);
        ds4_gpu_tensor *batch_test_probs = ds4_gpu_tensor_alloc(
            batch_logits_required + guard);
        uint8_t *batch_logits_host = malloc(
            (size_t)(batch_logits_required + guard));
        uint8_t *batch_ref_selected_host = malloc(
            (size_t)(batch_selected_required + guard));
        uint8_t *batch_test_selected_host = malloc(
            (size_t)(batch_selected_required + guard));
        uint8_t *batch_ref_weights_host = malloc(
            (size_t)(batch_weights_required + guard));
        uint8_t *batch_test_weights_host = malloc(
            (size_t)(batch_weights_required + guard));
        uint8_t *batch_ref_probs_host = malloc(
            (size_t)(batch_logits_required + guard));
        uint8_t *batch_test_probs_host = malloc(
            (size_t)(batch_logits_required + guard));
        TEST_ASSERT(batch_logits && batch_ref_selected && batch_test_selected &&
                    batch_ref_weights && batch_test_weights &&
                    batch_ref_probs && batch_test_probs &&
                    batch_logits_host && batch_ref_selected_host &&
                    batch_test_selected_host && batch_ref_weights_host &&
                    batch_test_weights_host && batch_ref_probs_host &&
                    batch_test_probs_host);
        if (batch_logits && batch_ref_selected && batch_test_selected &&
            batch_ref_weights && batch_test_weights && batch_ref_probs &&
            batch_test_probs && batch_logits_host && batch_ref_selected_host &&
            batch_test_selected_host && batch_ref_weights_host &&
            batch_test_weights_host && batch_ref_probs_host &&
            batch_test_probs_host) {
            float *batch_logits_f32 = (float *)batch_logits_host;
            for (uint32_t row = 0; row < batch_tokens; row++) {
                for (uint32_t expert = 0; expert < max_expert; expert++) {
                    const int value = (int)((row * 71u + expert * 29u) % 257u) - 128;
                    batch_logits_f32[(uint64_t)row * max_expert + expert] =
                        (float)value / 32.0f;
                }
            }
            const uint32_t qnan = 0x7fc00000u;
            memcpy(&batch_logits_f32[max_expert + 17u], &qnan, sizeof(qnan));
            memset(batch_logits_host + batch_logits_required, 0xa5, (size_t)guard);
            TEST_ASSERT(ds4_gpu_tensor_write(
                batch_logits, 0, batch_logits_host,
                batch_logits_required + guard) != 0);
            memset(batch_ref_selected_host, 0xa5,
                   (size_t)(batch_selected_required + guard));
            memset(batch_ref_weights_host, 0xa5,
                   (size_t)(batch_weights_required + guard));
            memset(batch_ref_probs_host, 0xa5,
                   (size_t)(batch_logits_required + guard));
            TEST_ASSERT(ds4_gpu_tensor_write(
                batch_ref_selected, 0, batch_ref_selected_host,
                batch_selected_required + guard) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                batch_ref_weights, 0, batch_ref_weights_host,
                batch_weights_required + guard) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                batch_ref_probs, 0, batch_ref_probs_host,
                batch_logits_required + guard) != 0);
            TEST_ASSERT(unsetenv(enable_env) == 0);
            TEST_ASSERT(ds4_gpu_glm_router_select_batch_tensor(
                batch_ref_selected, batch_ref_weights, batch_ref_probs,
                model_raw, page, 0, batch_logits, 256u, 10u, 2.5f,
                batch_tokens) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                batch_ref_selected, 0, batch_ref_selected_host,
                batch_selected_required + guard) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                batch_ref_weights, 0, batch_ref_weights_host,
                batch_weights_required + guard) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                batch_ref_probs, 0, batch_ref_probs_host,
                batch_logits_required + guard) != 0);

            memset(batch_test_selected_host, 0xa5,
                   (size_t)(batch_selected_required + guard));
            memset(batch_test_weights_host, 0xa5,
                   (size_t)(batch_weights_required + guard));
            memset(batch_test_probs_host, 0xa5,
                   (size_t)(batch_logits_required + guard));
            TEST_ASSERT(ds4_gpu_tensor_write(
                batch_test_selected, 0, batch_test_selected_host,
                batch_selected_required + guard) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                batch_test_weights, 0, batch_test_weights_host,
                batch_weights_required + guard) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                batch_test_probs, 0, batch_test_probs_host,
                batch_logits_required + guard) != 0);
            TEST_ASSERT(setenv(enable_env, "1", 1) == 0);
            ds4_gpu_laguna_router_simd_topk_stats_reset();
            TEST_ASSERT(ds4_gpu_glm_router_select_batch_tensor(
                batch_test_selected, batch_test_weights, batch_test_probs,
                model_raw, page, 0, batch_logits, 256u, 10u, 2.5f,
                batch_tokens) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                batch_test_selected, 0, batch_test_selected_host,
                batch_selected_required + guard) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                batch_test_weights, 0, batch_test_weights_host,
                batch_weights_required + guard) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                batch_test_probs, 0, batch_test_probs_host,
                batch_logits_required + guard) != 0);
            TEST_ASSERT(memcmp(batch_ref_selected_host, batch_test_selected_host,
                               (size_t)batch_selected_required) == 0);
            TEST_ASSERT(test_compare_float_bits(
                (const float *)batch_ref_weights_host,
                (const float *)batch_test_weights_host,
                (size_t)batch_tokens * max_used).mismatch_count == 0);
            TEST_ASSERT(test_compare_float_bits(
                (const float *)batch_ref_probs_host,
                (const float *)batch_test_probs_host,
                (size_t)batch_tokens * max_expert).mismatch_count == 0);
            for (uint64_t i = 0; i < guard; i++) {
                TEST_ASSERT(batch_test_selected_host[batch_selected_required + i] == 0xa5);
                TEST_ASSERT(batch_test_weights_host[batch_weights_required + i] == 0xa5);
                TEST_ASSERT(batch_test_probs_host[batch_logits_required + i] == 0xa5);
            }
            TEST_ASSERT(ds4_gpu_laguna_router_simd_topk_stats_after_wait(
                &optimized_rows, &fallback_rows,
                &encoded_rows, &encoded_dispatches) != 0);
            fprintf(stderr,
                    "ds4-test: Laguna router SIMD top-k batch stats "
                    "optimized=%u fallback=%u encoded_rows=%llu "
                    "encoded_dispatches=%llu scale=2.5\n",
                    optimized_rows, fallback_rows,
                    (unsigned long long)encoded_rows,
                    (unsigned long long)encoded_dispatches);
            TEST_ASSERT(optimized_rows == 2u);
            TEST_ASSERT(fallback_rows == 1u);
            TEST_ASSERT(encoded_rows == batch_tokens);
            TEST_ASSERT(optimized_rows + fallback_rows == encoded_rows);
            TEST_ASSERT(encoded_dispatches == 1u);
        }
        free(batch_test_probs_host);
        free(batch_ref_probs_host);
        free(batch_test_weights_host);
        free(batch_ref_weights_host);
        free(batch_test_selected_host);
        free(batch_ref_selected_host);
        free(batch_logits_host);
        ds4_gpu_tensor_free(batch_test_probs);
        ds4_gpu_tensor_free(batch_ref_probs);
        ds4_gpu_tensor_free(batch_test_weights);
        ds4_gpu_tensor_free(batch_ref_weights);
        ds4_gpu_tensor_free(batch_test_selected);
        ds4_gpu_tensor_free(batch_ref_selected);
        ds4_gpu_tensor_free(batch_logits);
    }

    test_restore_env(enable_env, saved_enable);
    test_restore_env(trace_env, saved_trace);
    test_restore_env(legacy_enable_env, saved_legacy_enable);
    test_restore_env(legacy_disable_env, saved_legacy_disable);
    free(test_probs_host);
    free(ref_probs_host);
    free(test_weights_host);
    free(ref_weights_host);
    free(test_selected_host);
    free(ref_selected_host);
    free(logits_host);
    ds4_gpu_tensor_free(test_probs);
    ds4_gpu_tensor_free(ref_probs);
    ds4_gpu_tensor_free(test_weights);
    ds4_gpu_tensor_free(ref_weights);
    ds4_gpu_tensor_free(test_selected);
    ds4_gpu_tensor_free(ref_selected);
    ds4_gpu_tensor_free(logits);
    free(model_raw);
}

static void test_metal_glm_router_simd_topk_exact(void) {
    const char *source = getenv("DS4_METAL_DSV4_MISC_SOURCE");
    if (source && source[0]) {
        test_metal_glm_router_simd_topk_source_override();
        return;
    }
    test_metal_glm_router_simd_topk_exact_suite();
}

static void test_metal_router_weights_batch_exact(void) {
    typedef struct {
        const char *name;
        uint32_t n_tokens;
        bool has_bias;
        bool hash_mode;
        uint32_t pattern;
    } router_batch_case;
    static const router_batch_case cases[] = {
        { "rows2-unique", 2, false, false, 0 },
        { "rows17-bias-ties", 17, true, false, 1 },
        { "rows129-hash-duplicates", 129, false, true, 2 },
        { "rows2048-typical", 2048, false, false, 0 },
        { "rows3-clamp-underflow", 3, false, false, 3 },
    };
    const uint32_t n_expert = 256;
    const uint32_t n_used = 6;
    const uint32_t max_tokens = 2048;
    const uint32_t hash_rows = 64;
    const uint32_t repeats = 2;
    const uint64_t logits_bytes =
        (uint64_t)max_tokens * n_expert * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)max_tokens * n_used * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)max_tokens * n_used * sizeof(float);
    const uint64_t tokens_bytes = (uint64_t)max_tokens * sizeof(int32_t);
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t bias_offset = 0;
    const uint64_t hash_offset = 2048;
    const char *disable_env =
        "DS4_METAL_DISABLE_ROUTER_WEIGHTS_BATCH_FUSION";
    const char *select_disable_env =
        "DS4_METAL_DISABLE_ROUTER_SELECT_FUSION";

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc(tokens_bytes);
    ds4_gpu_tensor *ref_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *batch_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *ref_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *batch_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *ref_probs = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *batch_probs = ds4_gpu_tensor_alloc(logits_bytes);
    float *logits_host = malloc((size_t)logits_bytes);
    int32_t *tokens_host = malloc((size_t)tokens_bytes);
    int32_t *ref_selected_host = malloc((size_t)selected_bytes);
    int32_t *batch_selected_host = malloc((size_t)selected_bytes);
    float *ref_weights_host = malloc((size_t)weights_bytes);
    float *batch_weights_host = malloc((size_t)weights_bytes);
    float *ref_probs_host = malloc((size_t)logits_bytes);
    float *batch_probs_host = malloc((size_t)logits_bytes);
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(tokens != NULL);
    TEST_ASSERT(ref_selected != NULL);
    TEST_ASSERT(batch_selected != NULL);
    TEST_ASSERT(ref_weights != NULL);
    TEST_ASSERT(batch_weights != NULL);
    TEST_ASSERT(ref_probs != NULL);
    TEST_ASSERT(batch_probs != NULL);
    TEST_ASSERT(logits_host != NULL);
    TEST_ASSERT(tokens_host != NULL);
    TEST_ASSERT(ref_selected_host != NULL);
    TEST_ASSERT(batch_selected_host != NULL);
    TEST_ASSERT(ref_weights_host != NULL);
    TEST_ASSERT(batch_weights_host != NULL);
    TEST_ASSERT(ref_probs_host != NULL);
    TEST_ASSERT(batch_probs_host != NULL);

    char *saved_disable = test_save_env(disable_env);
    char *saved_select_disable = test_save_env(select_disable_env);
    size_t total_selected_mismatch = 0;
    size_t total_weights_mismatch = 0;
    size_t total_probs_mismatch = 0;
    const bool allocated = model_raw && logits && tokens && ref_selected &&
        batch_selected && ref_weights && batch_weights && ref_probs &&
        batch_probs && logits_host && tokens_host && ref_selected_host &&
        batch_selected_host && ref_weights_host && batch_weights_host &&
        ref_probs_host && batch_probs_host;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *bias = (float *)((uint8_t *)model_raw + bias_offset);
        int32_t *hash = (int32_t *)((uint8_t *)model_raw + hash_offset);
        for (uint32_t i = 0; i < n_expert; i++) {
            bias[i] = (float)((int)((i * 29u) % 67u) - 33) / 16.0f;
        }
        for (uint32_t row = 0; row < hash_rows; row++) {
            const int32_t base = (int32_t)((row * 37u) % n_expert);
            hash[(uint64_t)row * n_used + 0u] = base;
            hash[(uint64_t)row * n_used + 1u] = base;
            hash[(uint64_t)row * n_used + 2u] = (base + 19) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 3u] = (base + 43) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 4u] = (base + 43) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 5u] = (base + 101) % (int32_t)n_expert;
        }
        for (uint32_t row = 0; row < max_tokens; row++) {
            tokens_host[row] = (int32_t)((row * 13u + 7u) % hash_rows);
        }

        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(ds4_gpu_tensor_write(
            tokens, 0, tokens_host, tokens_bytes) != 0);
        TEST_ASSERT(unsetenv(select_disable_env) == 0);

        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
            const router_batch_case *c = &cases[ci];
            const size_t prob_count = (size_t)c->n_tokens * n_expert;
            const size_t route_count = (size_t)c->n_tokens * n_used;
            const uint64_t case_probs_bytes = prob_count * sizeof(float);
            const uint64_t case_selected_bytes = route_count * sizeof(int32_t);
            const uint64_t case_weights_bytes = route_count * sizeof(float);
            for (uint32_t row = 0; row < c->n_tokens; row++) {
                for (uint32_t expert = 0; expert < n_expert; expert++) {
                    const size_t index = (size_t)row * n_expert + expert;
                    const int value =
                        (int)((row * 131u + expert * 47u +
                               (expert ^ (expert >> 3u)) * 13u) % 513u) -
                        256;
                    logits_host[index] = (float)value / 64.0f;
                    if (c->pattern == 1) {
                        logits_host[index] =
                            -7.0f - (float)((row + expert) % 17u) / 64.0f;
                    } else if (c->pattern == 3) {
                        logits_host[index] =
                            -30.0f - (float)((row * 17u + expert * 29u) % 11u);
                    }
                }
                if (c->pattern == 1) {
                    static const uint32_t tied[] = {
                        7u, 19u, 43u, 71u, 103u, 149u, 211u, 239u,
                    };
                    for (size_t i = 0; i < sizeof(tied) / sizeof(tied[0]); i++) {
                        logits_host[(size_t)row * n_expert + tied[i]] = 1.0f;
                    }
                }
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits, 0, logits_host, case_probs_bytes) != 0);

            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_router_select_batch_tensor(
                ref_selected, ref_weights, ref_probs,
                model_raw, page, bias_offset, hash_offset, hash_rows,
                1, 0, c->has_bias, c->hash_mode, logits, tokens,
                n_expert, n_used, 1.5f, c->n_tokens) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_selected, 0, ref_selected_host, case_selected_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_weights, 0, ref_weights_host, case_weights_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_probs, 0, ref_probs_host, case_probs_bytes) != 0);

            if (c->hash_mode) {
                for (uint32_t row = 0; row < c->n_tokens; row++) {
                    const uint32_t hash_row = (uint32_t)tokens_host[row];
                    for (uint32_t lane = 0; lane < n_used; lane++) {
                        TEST_ASSERT(
                            ref_selected_host[(size_t)row * n_used + lane] ==
                            hash[(uint64_t)hash_row * n_used + lane]);
                    }
                }
            }
            if (c->pattern == 3) {
                for (uint32_t row = 0; row < c->n_tokens; row++) {
                    float sum = 0.0f;
                    for (uint32_t lane = 0; lane < n_used; lane++) {
                        const int32_t expert =
                            ref_selected_host[(size_t)row * n_used + lane];
                        sum += ref_probs_host[(size_t)row * n_expert +
                                              (uint32_t)expert];
                    }
                    TEST_ASSERT(sum < 6.103515625e-5f);
                }
            }

            TEST_ASSERT(unsetenv(disable_env) == 0);
            for (uint32_t rep = 0; rep < repeats; rep++) {
                TEST_ASSERT(ds4_gpu_router_select_batch_tensor(
                    batch_selected, batch_weights, batch_probs,
                    model_raw, page, bias_offset, hash_offset, hash_rows,
                    1, 0, c->has_bias, c->hash_mode, logits, tokens,
                    n_expert, n_used, 1.5f, c->n_tokens) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_selected, 0, batch_selected_host,
                    case_selected_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_weights, 0, batch_weights_host,
                    case_weights_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_probs, 0, batch_probs_host,
                    case_probs_bytes) != 0);

                size_t selected_mismatch = 0;
                for (size_t i = 0; i < route_count; i++) {
                    if (ref_selected_host[i] != batch_selected_host[i]) {
                        selected_mismatch++;
                    }
                }
                const test_float_compare_stats weights_stats =
                    test_compare_float_bits(
                        ref_weights_host, batch_weights_host, route_count);
                const test_float_compare_stats probs_stats =
                    test_compare_float_bits(
                        ref_probs_host, batch_probs_host, prob_count);
                fprintf(stderr,
                        "ds4-test: router batch weights exactness "
                        "case=%s rep=%u selected=%zu/%zu weights=%zu/%zu "
                        "probs=%zu/%zu max_weight_ulp=%u max_prob_ulp=%u\n",
                        c->name,
                        rep,
                        selected_mismatch,
                        route_count,
                        weights_stats.mismatch_count,
                        route_count,
                        probs_stats.mismatch_count,
                        prob_count,
                        weights_stats.max_ulp,
                        probs_stats.max_ulp);
                TEST_ASSERT(selected_mismatch == 0);
                TEST_ASSERT(weights_stats.mismatch_count == 0);
                TEST_ASSERT(probs_stats.mismatch_count == 0);
                total_selected_mismatch += selected_mismatch;
                total_weights_mismatch += weights_stats.mismatch_count;
                total_probs_mismatch += probs_stats.mismatch_count;
            }
        }
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(select_disable_env, saved_select_disable);
    fprintf(stderr,
            "ds4-test: router batch weights total selected=%zu "
            "weights=%zu probs=%zu\n",
            total_selected_mismatch,
            total_weights_mismatch,
            total_probs_mismatch);
    TEST_ASSERT(total_selected_mismatch == 0);
    TEST_ASSERT(total_weights_mismatch == 0);
    TEST_ASSERT(total_probs_mismatch == 0);

    free(batch_probs_host);
    free(ref_probs_host);
    free(batch_weights_host);
    free(ref_weights_host);
    free(batch_selected_host);
    free(ref_selected_host);
    free(tokens_host);
    free(logits_host);
    ds4_gpu_tensor_free(batch_probs);
    ds4_gpu_tensor_free(ref_probs);
    ds4_gpu_tensor_free(batch_weights);
    ds4_gpu_tensor_free(ref_weights);
    ds4_gpu_tensor_free(batch_selected);
    ds4_gpu_tensor_free(ref_selected);
    ds4_gpu_tensor_free(tokens);
    ds4_gpu_tensor_free(logits);
    free(model_raw);
}
#endif

static void test_dflash_capture_nonfinite_sanitize(void) {
    enum {
        n_embd = 4,
        n_aux = 3,
        src_rows = 3,
        dst_rows = 4,
    };
    const float marker = 123.0f;
    float src_host[src_rows * n_embd] = {
        10.0f, 11.0f, 12.0f, 13.0f,
        0.0f, 0.0f, 0.0f, 1.25f,
        -0.0f, -2.5f, 65536.0f, -7.0f,
    };
    const uint32_t special_bits[3] = {
        0x7fc12345u, 0x7f800000u, 0xff800000u,
    };
    memcpy(&src_host[n_embd], special_bits, sizeof(special_bits));
    float dst_host[dst_rows * n_aux * n_embd];
    for (size_t i = 0; i < sizeof(dst_host) / sizeof(dst_host[0]); i++) {
        dst_host[i] = marker;
    }

    ds4_gpu_tensor *src = ds4_gpu_tensor_alloc(sizeof(src_host));
    ds4_gpu_tensor *dst = ds4_gpu_tensor_alloc(sizeof(dst_host));
    TEST_ASSERT(src != NULL);
    TEST_ASSERT(dst != NULL);
    if (!src || !dst) {
        ds4_gpu_tensor_free(src);
        ds4_gpu_tensor_free(dst);
        return;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(src, 0, src_host, sizeof(src_host)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(dst, 0, dst_host, sizeof(dst_host)) != 0);
    TEST_ASSERT(ds4_gpu_dflash_capture_rows_tensor(
        dst, src, 1, 1, 2, n_embd, n_aux, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(dst, 0, dst_host, sizeof(dst_host)) != 0);

    const float expected[2][n_embd] = {
        {0.0f, 65504.0f, -65504.0f, 1.25f},
        {-0.0f, -2.5f, 65536.0f, -7.0f},
    };
    for (uint32_t row = 0; row < 2; row++) {
        for (uint32_t col = 0; col < n_embd; col++) {
            const size_t index =
                ((size_t)(row + 1) * n_aux + 1) * n_embd + col;
            TEST_ASSERT(dst_host[index] == expected[row][col]);
            uint32_t bits = 0;
            memcpy(&bits, &dst_host[index], sizeof(bits));
            TEST_ASSERT((bits & 0x7f800000u) != 0x7f800000u);
        }
    }
    uint32_t negative_zero_bits = 0;
    memcpy(&negative_zero_bits,
           &dst_host[((size_t)2 * n_aux + 1) * n_embd],
           sizeof(negative_zero_bits));
    TEST_ASSERT(negative_zero_bits == 0x80000000u);
    TEST_ASSERT(dst_host[0] == marker);
    TEST_ASSERT(dst_host[((size_t)3 * n_aux + 2) * n_embd + 3] == marker);

    ds4_gpu_tensor_free(src);
    ds4_gpu_tensor_free(dst);
}

#if defined(__APPLE__)
static int test_laguna_argmax_host(const float *values, uint32_t n) {
    int best = 0;
    float best_value = -1.0e30f;
    for (uint32_t i = 0; i < n; i++) {
        const float value = values[i];
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        const bool is_nan =
            (bits & 0x7f800000u) == 0x7f800000u &&
            (bits & 0x007fffffu) != 0u;
        if (!is_nan && value > best_value) {
            best_value = value;
            best = (int)i;
        }
    }
    return best;
}

static void test_metal_laguna_gpu_argmax(void) {
    static const uint32_t sizes[] = {
        1u, 3u, 7u, 31u, 257u, 511u, 100003u, 100351u, 100352u,
    };
    static const uint32_t nan_bits[] = {
        0x7fc12345u, 0x7fa00001u, 0xffc12345u, 0xffa00001u,
    };
    const uint32_t pos_inf_bits = 0x7f800000u;
    const uint32_t neg_inf_bits = 0xff800000u;
    float nan_values[sizeof(nan_bits) / sizeof(nan_bits[0])];
    float pos_inf;
    float neg_inf;
    for (size_t i = 0; i < sizeof(nan_bits) / sizeof(nan_bits[0]); i++) {
        memcpy(&nan_values[i], &nan_bits[i], sizeof(nan_values[i]));
    }
    memcpy(&pos_inf, &pos_inf_bits, sizeof(pos_inf));
    memcpy(&neg_inf, &neg_inf_bits, sizeof(neg_inf));

    const int available = ds4_gpu_laguna_argmax_available();
    if (!available) {
        fprintf(stderr,
                "ds4-test: Laguna GPU argmax pipeline unavailable; skipped\n");
        return;
    }

    for (uint32_t ci = 0; ci < 7u; ci++) {
        for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
            const uint32_t n = sizes[si];
            const uint64_t bytes = (uint64_t)n * sizeof(float);
            float *values = malloc((size_t)bytes);
            ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(bytes);
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(int32_t));
            TEST_ASSERT(values != NULL);
            TEST_ASSERT(logits != NULL);
            TEST_ASSERT(out != NULL);
            if (!values || !logits || !out) {
                free(values);
                ds4_gpu_tensor_free(logits);
                ds4_gpu_tensor_free(out);
                continue;
            }

            for (uint32_t i = 0; i < n; i++) values[i] = -1.0e30f;
            if (ci == 0u && n > 1u) {
                values[n - 1u] = -1.0000001e30f;
            } else if (ci == 1u) {
                for (uint32_t i = 0; i < n; i++) values[i] = -17.0f;
                if (n > 2u) {
                    values[n / 3u] = 5.0f;
                    values[n - 1u] = 5.0f;
                } else {
                    values[n - 1u] = 5.0f;
                }
            } else if (ci == 2u) {
                for (uint32_t i = 0; i < n; i++) {
                    values[i] = nan_values[i %
                        (sizeof(nan_values) / sizeof(nan_values[0]))];
                }
                if (n > 1u) values[n - 1u] = 4.0f;
            } else if (ci == 3u) {
                for (uint32_t i = 0; i < n; i++) {
                    values[i] = nan_values[i %
                        (sizeof(nan_values) / sizeof(nan_values[0]))];
                }
            } else if (ci == 4u) {
                for (uint32_t i = 0; i < n; i++) values[i] = neg_inf;
                if (n > 2u) {
                    values[n / 3u] = pos_inf;
                    values[n - 1u] = pos_inf;
                } else {
                    values[n - 1u] = pos_inf;
                }
            } else if (ci == 5u) {
                for (uint32_t i = 0; i < n; i++) values[i] = neg_inf;
            } else if (ci == 6u) {
                uint32_t state = 0x9e3779b9u ^ (uint32_t)si * 0x45d9f3bu;
                for (uint32_t i = 0; i < n; i++) {
                    state = state * 1664525u + 1013904223u;
                    values[i] = ((float)(state >> 8) / 16777216.0f) * 200.0f - 100.0f;
                }
                if (n > 3u) {
                    values[1] = nan_values[si %
                        (sizeof(nan_values) / sizeof(nan_values[0]))];
                    values[n / 2u] = -1.0e30f;
                    values[n - 2u] = -1.0000001e30f;
                    values[n - 1u] = pos_inf;
                }
            }

            const int expected = test_laguna_argmax_host(values, n);
            int32_t actual = -99;
            TEST_ASSERT(ds4_gpu_tensor_write(logits, 0, values, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(out, 0, &actual, sizeof(actual)) != 0);
            TEST_ASSERT(ds4_gpu_laguna_argmax_tensor(out, logits, n) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(out, 0, &actual, sizeof(actual)) != 0);
            TEST_ASSERT(actual == expected);

            free(values);
            ds4_gpu_tensor_free(logits);
            ds4_gpu_tensor_free(out);
        }
    }
}

static void test_metal_laguna_decode_ladder_ordering_exact(void) {
    const uint32_t n = 257u;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    TEST_ASSERT(ds4_gpu_diagnostic_pending_command_buffer_count() == 0);
    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *c = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *d = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *e = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(bytes);
    float *a_host = malloc((size_t)bytes);
    float *b_host = malloc((size_t)bytes);
    float *poison = malloc((size_t)bytes);
    float *expected_c = malloc((size_t)bytes);
    float *expected_d = malloc((size_t)bytes);
    float *expected_e = malloc((size_t)bytes);
    float *expected_out = malloc((size_t)bytes);
    float *ref_c = malloc((size_t)bytes);
    float *ref_d = malloc((size_t)bytes);
    float *ref_e = malloc((size_t)bytes);
    float *ref_out = malloc((size_t)bytes);
    float *actual = malloc((size_t)bytes);
    TEST_ASSERT(a && b && c && d && e && out);
    TEST_ASSERT(a_host && b_host && poison && expected_c && expected_d &&
                expected_e && expected_out && ref_c && ref_d && ref_e &&
                ref_out && actual);
    if (!a || !b || !c || !d || !e || !out ||
        !a_host || !b_host || !poison || !expected_c || !expected_d ||
        !expected_e || !expected_out || !ref_c || !ref_d || !ref_e ||
        !ref_out || !actual) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < n; i++) {
        a_host[i] = ((float)((int)(i % 29u) - 14) * 0.125f) +
                    (float)(i % 7u) * 0.015625f;
        b_host[i] = ((float)((int)(i % 17u) - 8) * 0.0625f) -
                    (float)(i % 5u) * 0.03125f;
        const uint32_t poison_bits = 0x7fc10001u + (i & 0x3ffu);
        memcpy(&poison[i], &poison_bits, sizeof(poison_bits));

        /* Each input is dyadic.  Keep every dependent add as its own stored
         * stage so the baseline is checked against an independent host
         * reference before it is used to validate command-buffer splitting. */
        expected_c[i] = a_host[i] + b_host[i];
        expected_d[i] = expected_c[i] + a_host[i];
        expected_e[i] = expected_d[i] + b_host[i];
        expected_out[i] = expected_e[i] + expected_c[i];
    }
    TEST_ASSERT(ds4_gpu_tensor_write(a, 0, a_host, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(b, 0, b_host, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(c, 0, poison, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(d, 0, poison, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(e, 0, poison, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, poison, bytes) != 0);

    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    if (!ds4_gpu_commands_active()) goto cleanup;
    TEST_ASSERT(ds4_gpu_add_tensor(c, a, b, n) != 0);
    TEST_ASSERT(ds4_gpu_add_tensor(d, c, a, n) != 0);
    TEST_ASSERT(ds4_gpu_add_tensor(e, d, b, n) != 0);
    TEST_ASSERT(ds4_gpu_add_tensor(out, e, c, n) != 0);
    TEST_ASSERT(ds4_gpu_end_commands() != 0);
    TEST_ASSERT(ds4_gpu_diagnostic_pending_command_buffer_count() == 0);
    if (ds4_gpu_commands_active()) goto cleanup;
    TEST_ASSERT(ds4_gpu_tensor_read(c, 0, ref_c, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(d, 0, ref_d, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(e, 0, ref_e, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, ref_out, bytes) != 0);
    TEST_ASSERT(memcmp(ref_c, expected_c, (size_t)bytes) == 0);
    TEST_ASSERT(memcmp(ref_d, expected_d, (size_t)bytes) == 0);
    TEST_ASSERT(memcmp(ref_e, expected_e, (size_t)bytes) == 0);
    TEST_ASSERT(memcmp(ref_out, expected_out, (size_t)bytes) == 0);

    TEST_ASSERT(ds4_gpu_tensor_write(c, 0, poison, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(d, 0, poison, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(e, 0, poison, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, poison, bytes) != 0);

    TEST_ASSERT(ds4_gpu_diagnostic_pending_command_buffer_count() == 0);
    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    if (!ds4_gpu_commands_active()) goto cleanup;
    TEST_ASSERT(ds4_gpu_add_tensor(c, a, b, n) != 0);
    TEST_ASSERT(ds4_gpu_flush_commands() != 0);
    TEST_ASSERT(ds4_gpu_diagnostic_pending_command_buffer_count() == 1);
    TEST_ASSERT(ds4_gpu_add_tensor(d, c, a, n) != 0);
    TEST_ASSERT(ds4_gpu_flush_commands() != 0);
    TEST_ASSERT(ds4_gpu_diagnostic_pending_command_buffer_count() == 2);
    TEST_ASSERT(ds4_gpu_add_tensor(e, d, b, n) != 0);
    TEST_ASSERT(ds4_gpu_flush_commands() != 0);
    TEST_ASSERT(ds4_gpu_diagnostic_pending_command_buffer_count() == 3);
    TEST_ASSERT(ds4_gpu_add_tensor(out, e, c, n) != 0);
    TEST_ASSERT(ds4_gpu_end_commands() != 0);
    TEST_ASSERT(ds4_gpu_diagnostic_pending_command_buffer_count() == 0);
    if (ds4_gpu_commands_active()) goto cleanup;

    TEST_ASSERT(ds4_gpu_tensor_read(c, 0, actual, bytes) != 0);
    TEST_ASSERT(memcmp(actual, ref_c, (size_t)bytes) == 0);
    TEST_ASSERT(ds4_gpu_tensor_read(d, 0, actual, bytes) != 0);
    TEST_ASSERT(memcmp(actual, ref_d, (size_t)bytes) == 0);
    TEST_ASSERT(ds4_gpu_tensor_read(e, 0, actual, bytes) != 0);
    TEST_ASSERT(memcmp(actual, ref_e, (size_t)bytes) == 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, actual, bytes) != 0);
    TEST_ASSERT(memcmp(actual, ref_out, (size_t)bytes) == 0);

cleanup:
    if (ds4_gpu_commands_active()) {
        (void)ds4_gpu_discard_commands();
    } else {
        (void)ds4_gpu_wait_submitted_commands();
    }
    free(actual);
    free(ref_out);
    free(ref_e);
    free(ref_d);
    free(ref_c);
    free(expected_out);
    free(expected_e);
    free(expected_d);
    free(expected_c);
    free(poison);
    free(b_host);
    free(a_host);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(e);
    ds4_gpu_tensor_free(d);
    ds4_gpu_tensor_free(c);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
}

static void test_metal_parallel_ffn_terminal_lifecycle(void) {
    enum { n = 64 };
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    float a_host[n];
    float b_host[n];
    float expected[n];
    float poison[n];
    for (uint32_t i = 0; i < n; i++) {
        a_host[i] = (float)((int)i - 21) * 0.125f;
        b_host[i] = (float)((int)(i * 3u) - 17) * 0.0625f;
        expected[i] = a_host[i] + b_host[i];
        poison[i] = -777.0f;
    }

    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(bytes);
    TEST_ASSERT(a && b && out);
    if (!a || !b || !out) goto cleanup;
    TEST_ASSERT(ds4_gpu_tensor_write(a, 0, a_host, bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(b, 0, b_host, bytes) != 0);

    /* submit_commands() must reset the armed state before the command buffer
     * becomes terminal; query it before begin_commands() can reset anything. */
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, poison, bytes) != 0);
    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    TEST_ASSERT(ds4_gpu_parallel_ffn_test_arm_state() != 0);
    TEST_ASSERT(ds4_gpu_submit_commands() != 0);
    TEST_ASSERT(ds4_gpu_parallel_ffn_test_state_is_clean() != 0);
    TEST_ASSERT(!ds4_gpu_commands_active());
    TEST_ASSERT(ds4_gpu_wait_submitted_commands() != 0);

    /* A real ordinary batch after submit proves no concurrent encoder or
     * stale stage metadata leaks into the next command sequence. */
    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    TEST_ASSERT(ds4_gpu_add_tensor(out, a, b, n) != 0);
    TEST_ASSERT(ds4_gpu_end_commands() != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, poison, bytes) != 0);
    TEST_ASSERT(memcmp(poison, expected, (size_t)bytes) == 0);

    /* Repeat through discard, including its empty/aborted encoder boundary. */
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, poison, bytes) != 0);
    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    TEST_ASSERT(ds4_gpu_parallel_ffn_test_arm_state() != 0);
    TEST_ASSERT(ds4_gpu_discard_commands() != 0);
    TEST_ASSERT(ds4_gpu_parallel_ffn_test_state_is_clean() != 0);
    TEST_ASSERT(!ds4_gpu_commands_active());
    TEST_ASSERT(ds4_gpu_wait_submitted_commands() != 0);

    TEST_ASSERT(ds4_gpu_begin_commands() != 0);
    TEST_ASSERT(ds4_gpu_add_tensor(out, a, b, n) != 0);
    TEST_ASSERT(ds4_gpu_end_commands() != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, poison, bytes) != 0);
    TEST_ASSERT(memcmp(poison, expected, (size_t)bytes) == 0);

cleanup:
    if (ds4_gpu_commands_active()) {
        (void)ds4_gpu_discard_commands();
    } else {
        (void)ds4_gpu_wait_submitted_commands();
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
}
#endif

static void test_metal_kernel_group(void) {
    test_laguna_decode_ladder_parser();
    test_dflash_capture_nonfinite_sanitize();
    test_metal_f16_matvec_fast_nr0_4();
    test_metal_f16_prefill_matmul();
    test_metal_q8_0_prefill_matmul();
    test_metal_pack_slot_rows_f32();
    test_metal_store_raw_kv_batch_wrap();
    test_dspark_cache_window_crop();
    test_metal_q8_0_decode_pair_exact();
#if defined(__APPLE__)
    test_metal_q8_decode_lifecycle_snapshot();
    test_metal_laguna_decode_residual_norm_env();
    test_metal_add3_rms_norm_rejects_partial_simd();
    test_metal_laguna_gpu_argmax();
    test_metal_laguna_decode_ladder_ordering_exact();
    test_metal_laguna_q8_lmhead_screen_gates();
    test_metal_laguna_q8_lmhead_screen();
    test_metal_parallel_ffn_terminal_lifecycle();
    test_metal_laguna_dense_q8_gate_up_swiglu();
    test_metal_laguna_staged_swa_exact();
    test_metal_glm_qmv_r1_exact();
    test_metal_q8_0_output_nr4_exact();
    test_metal_f16_compressor_pair_state_store_exact();
    test_metal_compressor_ape_add_exact();
    test_metal_compressor_ratio4_pack_exact();
    test_metal_compressor_ratio4_replay_pack_exact();
    test_metal_compressor_ratio4_direct_pool_exact();
    test_metal_inplace_rope_pair_exact();
    test_metal_contiguous_f32_f16_roundtrip_exact();
    test_metal_gathered_kv_stage_exact();
    test_metal_contiguous_compressed_f16_attention_exact();
    test_metal_persistent_zero_attention_mask_exact();
    test_metal_zero_prefix_prefill_mask_cache_exact();
    test_metal_laguna_swa_gqa3_numeric_ab();
    test_metal_laguna_swa_gqa9_numeric_ab();
    test_metal_laguna_swa_gqa9_preflight_lifecycle();
    test_metal_laguna_swa_gqa3_scope();
    test_laguna_gqa3_decode_numeric();
    test_metal_laguna_qk_norm_rope_pair_exact();
    test_metal_add_rms_norm_weight_rows_exact();
    test_metal_add3_rms_norm_weight_rows_exact();
    test_metal_hc_split_weighted_sum_norm_batch_exact();
    test_metal_output_hc_weights4_exact();
    test_metal_hc_rms_scale_project_f16_exact();
    test_metal_f16_rms_norm_mv_exact();
    test_metal_laguna_router_fused_exact();
    test_metal_router_simd_finalize_exact();
    test_metal_glm_router_simd_topk_exact();
    test_metal_router_weights_batch_exact();
#endif
}

static void test_metal_short_prefill_ratio4(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    const int tokens[] = {
        ds4_token_user(engine),
        ds4_token_assistant(engine),
        ds4_token_eos(engine),
    };
    for (size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
        TEST_ASSERT(tokens[i] >= 0);
        if (tokens[i] < 0) return;
    }

    for (size_t n = 1; n <= 3; n++) {
        ds4_tokens prompt = {0};
        for (size_t i = 0; i < n; i++) {
            ds4_tokens_push(&prompt, tokens[i]);
        }
        TEST_ASSERT(prompt.len == (int)n);

        ds4_session *session = NULL;
        TEST_ASSERT(ds4_session_create(&session, engine, 2048) == 0);
        if (!session) {
            ds4_tokens_free(&prompt);
            return;
        }

        char err[160] = {0};
        const int rc = ds4_session_sync(session, &prompt, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "ds4-test: short prefill failed for %zu token(s): %s\n",
                    n, err);
        }
        TEST_ASSERT(rc == 0);

        ds4_session_free(session);
        ds4_tokens_free(&prompt);
    }
}

static char *test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *s = malloc((size_t)len + 1);
    if (!s) {
        fclose(fp);
        return NULL;
    }
    size_t nread = fread(s, 1, (size_t)len, fp);
    fclose(fp);
    if (nread != (size_t)len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

typedef struct {
    const char *name;
    int number;
} test_long_fact;

static const test_long_fact test_long_facts[] = {
    {"Bob", 34},
    {"Alice", 52},
    {"Clara", 71},
    {"Diego", 93},
    {"Elena", 16},
    {"Felix", 88},
    {"Greta", 47},
    {"Hugo", 29},
    {"Iris", 64},
    {"Jonas", 12},
    {"Kira", 81},
    {"Leo", 39},
    {"Marta", 76},
    {"Nadia", 23},
    {"Owen", 58},
    {"Priya", 97},
};

static bool test_is_name_boundary(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || !(isalnum(uc) || c == '_');
}

static bool test_parse_assignment_value(const char *p, int *value) {
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '=') return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (!isdigit((unsigned char)*p)) return false;

    int v = 0;
    while (isdigit((unsigned char)*p)) {
        v = v * 10 + (*p - '0');
        p++;
    }
    *value = v;
    return true;
}

static bool test_output_has_fact(const char *text, const test_long_fact *fact) {
    const size_t name_len = strlen(fact->name);
    const char *p = text;
    bool saw_wrong_assignment = false;
    int wrong_value = -1;

    while ((p = strstr(p, fact->name)) != NULL) {
        const bool before_ok = p == text || test_is_name_boundary(p[-1]);
        const bool after_ok = test_is_name_boundary(p[name_len]) ||
                              p[name_len] == ' ' ||
                              p[name_len] == '\t' ||
                              p[name_len] == '=';
        if (before_ok && after_ok) {
            int value = 0;
            if (test_parse_assignment_value(p + name_len, &value)) {
                if (value == fact->number) return true;
                saw_wrong_assignment = true;
                wrong_value = value;
            }
        }
        p += name_len;
    }

    if (saw_wrong_assignment) {
        fprintf(stderr,
                "ds4-test: long-context wrong assignment for %s: got %d expected %d\n",
                fact->name, wrong_value, fact->number);
    } else {
        fprintf(stderr,
                "ds4-test: long-context missing assignment for %s=%d\n",
                fact->name, fact->number);
    }
    return false;
}

static int test_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool test_hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    while (*hex && !isspace((unsigned char)*hex)) {
        int hi = test_hex_digit(hex[0]);
        int lo = test_hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static bool test_token_bytes_equal(ds4_engine *engine, int token,
                                   const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = ds4_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static void test_long_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 8192 == 0) {
        fprintf(stderr, "ds4-test: long-context prefill %d/%d\n", current, total);
    }
}

static void test_long_story_fact_recall(void) {
    const char *prompt_path = getenv("DS4_TEST_LONG_PROMPT");
    if (!prompt_path || !prompt_path[0]) {
        prompt_path = "tests/long_context_story_prompt.txt";
    }
    char *prompt_text = test_read_file(prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        free(prompt_text);
        return;
    }

    ds4_tokens prompt = {0};
    ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    TEST_ASSERT(prompt.len > 30000);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 100000) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        free(prompt_text);
        return;
    }

    char err[160];
    ds4_session_set_progress(session, test_long_prefill_progress, NULL);
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    ds4_session_set_progress(session, NULL, NULL);

    buf out = {0};
    uint64_t rng = 12345;
    int generated = 0;
    bool decode_ok = true;
    for (; generated < 350; generated++) {
        int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);

        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    const char *text = out.ptr ? out.ptr : "";
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(generated > 0);
    for (size_t i = 0; i < sizeof(test_long_facts) / sizeof(test_long_facts[0]); i++) {
        TEST_ASSERT(test_output_has_fact(text, &test_long_facts[i]));
    }

    buf_free(&out);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    free(prompt_text);
}

#define TEST_VEC_MAX_STEPS 16
#define TEST_VEC_MAX_TOP 32
#define TEST_VEC_MAX_TOKEN_BYTES 128

typedef struct {
    unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} test_vec_top;

typedef struct {
    unsigned char selected[TEST_VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    test_vec_top top[TEST_VEC_MAX_TOP];
} test_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    test_vec_step steps[TEST_VEC_MAX_STEPS];
} test_vec_case;

static char *test_trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static bool test_read_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            TEST_ASSERT(vc->nsteps > 0 && vc->nsteps <= TEST_VEC_MAX_STEPS);
            return true;
        }
        TEST_ASSERT(!"unexpected line before vector case");
    }
    return false;
}

static bool test_fill_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;

        if (!strncmp(p, "step ", 5)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                TEST_ASSERT(!"bad vector step line");
                return false;
            }
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(ntop >= 0 && ntop <= TEST_VEC_MAX_TOP);
            vc->steps[step_index].ntop = ntop;
            TEST_ASSERT(test_hex_to_bytes(hex,
                                          vc->steps[step_index].selected,
                                          TEST_VEC_MAX_TOKEN_BYTES,
                                          &vc->steps[step_index].selected_len));
            top_index = 0;
            continue;
        }

        if (!strncmp(p, "top ", 4)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index < vc->steps[step_index].ntop);
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) {
                TEST_ASSERT(!"bad vector top line");
                return false;
            }
            test_vec_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            TEST_ASSERT(test_hex_to_bytes(hex, top->bytes,
                                          TEST_VEC_MAX_TOKEN_BYTES, &top->len));
            continue;
        }

        TEST_ASSERT(!"unexpected vector line");
        return false;
    }

    TEST_ASSERT(!"unterminated vector case");
    return false;
}

static void test_logprob_vector_case(ds4_engine *engine, const test_vec_case *vc) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-test: vector %s prefill failed: %s\n", vc->id, err);
        TEST_ASSERT(false);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        return;
    }

    ds4_token_score scores[20];
    for (int i = 0; i < vc->nsteps; i++) {
        const test_vec_step *step = &vc->steps[i];
        int nscore = ds4_session_top_logprobs(session, scores, 20);
        int token = ds4_session_argmax(session);
        if (!test_token_bytes_equal(engine, token, step->selected, step->selected_len)) {
            fprintf(stderr, "ds4-test: vector %s step %d selected token mismatch\n",
                    vc->id, i);
            TEST_ASSERT(false);
        }

        for (int t = 0; t < step->ntop; t++) {
            bool found = false;
            float local_lp = 0.0f;
            for (int j = 0; j < nscore; j++) {
                if (scores[j].id < 0) continue;
                if (test_token_bytes_equal(engine, scores[j].id,
                                           step->top[t].bytes,
                                           step->top[t].len)) {
                    found = true;
                    local_lp = scores[j].logprob;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "ds4-test: vector %s step %d official top token missing locally\n",
                        vc->id, i);
                TEST_ASSERT(false);
            } else if (fabsf(local_lp - step->top[t].logprob) > 4.0f) {
                fprintf(stderr,
                        "ds4-test: vector %s step %d logprob delta too high: local=%g official=%g\n",
                        vc->id, i, local_lp, step->top[t].logprob);
                TEST_ASSERT(false);
            }
        }

        if (i + 1 < vc->nsteps) {
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-test: vector %s step %d eval failed: %s\n",
                        vc->id, i, err);
                TEST_ASSERT(false);
                break;
            }
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static bool test_logprob_vector_case_disabled(const char *path,
                                              const test_vec_case *vc) {
    /*
     * This one long-context vector currently matches the public DeepSeek API less
     * after adding the official Hadamard+FP4 indexer path.  The public official
     * implementation and the API appear to disagree here; the official graph has
     * slightly lower local perplexity on the A/B check we ran, so DS4 keeps that
     * implementation and only excludes this brittle API fixture for now.
     */
    return !strcmp(path, "tests/test-vectors/flash-pre-0731/official.vec") &&
           !strcmp(vc->id, "long_memory_archive");
}

static void test_official_logprob_vectors_run(const char *case_filter) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) {
        path = "tests/test-vectors/flash-0731/official.vec";
    }
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();
    setenv("DS4_METAL_PREFILL_CHUNK", "2048", 1);
    if (getenv("DS4_TEST_LOGPROB_AUTO_METAL") == NULL) {
        setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    } else {
        unsetenv("DS4_METAL_DISABLE_METAL4");
    }
    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_vec_case vc;
    int ran = 0;
    while (test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (case_filter && case_filter[0] && strcmp(vc.id, case_filter)) {
            continue;
        }
        if (test_logprob_vector_case_disabled(path, &vc)) {
            fprintf(stderr, "ds4-test: vector %s skipped (API/official graph mismatch)\n",
                    vc.id);
            continue;
        }
        fprintf(stderr, "ds4-test: vector %s\n", vc.id);
        test_logprob_vector_case(engine, &vc);
        ran++;
    }
    TEST_ASSERT(!case_filter || !case_filter[0] || ran == 1);
    ds4_engine_close(engine);
    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

static void test_official_logprob_vectors(void) {
    test_official_logprob_vectors_run(NULL);
}

static void test_metal_ssd_streaming_cache_pressure(void) {
#ifndef __APPLE__
    fprintf(stderr,
            "ds4-test: Metal SSD streaming cache-pressure repro skipped "
            "(Metal-only)\n");
#else
    /*
     * Regression repro for GitHub issue #384.
     *
     * The bug needs the Metal SSD-streaming decode layer-batch path and a small
     * routed-expert cache. Under pressure, a cache entry referenced by an
     * already-encoded-but-not-yet-executed layer can be reused for a later
     * layer in the same command buffer, producing deterministic wrong logits.
     */
    char *saved_streaming = test_save_env("DS4_TEST_SSD_STREAMING");
    char *saved_cache_gb = test_save_env("DS4_TEST_SSD_STREAMING_CACHE_GB");
    char *saved_cache_experts =
        test_save_env("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS");
    char *saved_disable_layer_batch =
        test_save_env("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH");
    char *saved_disable_static_decode =
        test_save_env("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP");
    char *saved_one_stage =
        test_save_env("DS4_METAL_MOE_ONE_STAGE_PROFILE");

    setenv("DS4_TEST_SSD_STREAMING", "1", 1);
    setenv("DS4_TEST_SSD_STREAMING_CACHE_GB", "16", 1);
    unsetenv("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS");
    unsetenv("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH");
    unsetenv("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP");
    unsetenv("DS4_METAL_MOE_ONE_STAGE_PROFILE");

    fprintf(stderr,
            "ds4-test: Metal SSD streaming cache-pressure repro "
            "(16GiB cache, layer-batched decode, short_code_completion)\n");
    test_official_logprob_vectors_run("short_code_completion");

    test_restore_env("DS4_METAL_MOE_ONE_STAGE_PROFILE", saved_one_stage);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP",
                     saved_disable_static_decode);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH",
                     saved_disable_layer_batch);
    test_restore_env("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS",
                     saved_cache_experts);
    test_restore_env("DS4_TEST_SSD_STREAMING_CACHE_GB", saved_cache_gb);
    test_restore_env("DS4_TEST_SSD_STREAMING", saved_streaming);
#endif
}

static void test_logits_topk(const float *logits, int n, int *out, int k);
static bool test_topk_contains(const int *top, int k, int id);

#define TEST_LOCAL_GOLDEN_MAX_TOP 128

typedef struct {
    int id;
    float logit;
} test_local_golden_top;

typedef struct {
    char id[96];
    char mode[16];
    char prompt_path[512];
    int ctx;
    int frontier;
    int ntop;
    test_local_golden_top top[TEST_LOCAL_GOLDEN_MAX_TOP];
} test_local_golden_case;

static bool test_read_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    memset(tc, 0, sizeof(*tc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %15s %d %d %511s %d",
                   tc->id, tc->mode, &tc->ctx, &tc->frontier,
                   tc->prompt_path, &tc->ntop) == 6) {
            TEST_ASSERT(tc->ctx > tc->frontier);
            TEST_ASSERT(tc->frontier > 0);
            TEST_ASSERT(tc->ntop > 0 && tc->ntop <= TEST_LOCAL_GOLDEN_MAX_TOP);
            return true;
        }
        TEST_ASSERT(!"unexpected line before local golden case");
        return false;
    }
    return false;
}

static bool test_fill_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    int seen = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) {
            TEST_ASSERT(seen == tc->ntop);
            return seen == tc->ntop;
        }
        int rank = -1;
        int id = -1;
        float logit = 0.0f;
        if (sscanf(p, "top %d %d %f", &rank, &id, &logit) != 3) {
            TEST_ASSERT(!"bad local golden top line");
            return false;
        }
        TEST_ASSERT(rank == seen);
        TEST_ASSERT(seen < tc->ntop);
        if (seen >= tc->ntop) return false;
        tc->top[seen].id = id;
        tc->top[seen].logit = logit;
        seen++;
    }
    TEST_ASSERT(!"unterminated local golden case");
    return false;
}

static int test_local_golden_overlap(const test_local_golden_case *tc,
                                     const int *cand_top,
                                     int n) {
    int overlap = 0;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        if (test_topk_contains(cand_top, n, tc->top[i].id)) overlap++;
    }
    return overlap;
}

static float test_local_golden_max_abs(const test_local_golden_case *tc,
                                       const float *cand_logits,
                                       int n) {
    float max_abs = 0.0f;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        const int id = tc->top[i].id;
        if (id < 0) continue;
        const float abs_delta = fabsf(cand_logits[id] - tc->top[i].logit);
        if (abs_delta > max_abs) max_abs = abs_delta;
    }
    return max_abs;
}

static void test_local_golden_case_run(ds4_engine *engine,
                                       const test_local_golden_case *tc) {
    char *prompt_text = test_read_file(tc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    if (!strcmp(tc->mode, "text")) {
        ds4_tokenize_text(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "rendered")) {
        ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "chat")) {
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    } else {
        TEST_ASSERT(!"unknown local golden prompt mode");
    }
    free(prompt_text);
    TEST_ASSERT(prompt.len >= tc->frontier);
    if (prompt.len < tc->frontier) {
        ds4_tokens_free(&prompt);
        return;
    }

    ds4_tokens prefix = {
        .v = prompt.v,
        .len = tc->frontier,
        .cap = tc->frontier,
    };

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);

    const int vocab = ds4_engine_vocab_size(engine);
    float *cand_logits = malloc((size_t)vocab * sizeof(cand_logits[0]));
    TEST_ASSERT(cand_logits != NULL);
    if (cand_logits &&
        ds4_session_copy_logits(session, cand_logits, vocab) == vocab) {
        int cand_top[TEST_LOCAL_GOLDEN_MAX_TOP];
        const int ntop = tc->ntop < TEST_LOCAL_GOLDEN_MAX_TOP ?
                         tc->ntop : TEST_LOCAL_GOLDEN_MAX_TOP;
        test_logits_topk(cand_logits, vocab, cand_top, ntop);

        const int top5_overlap = test_local_golden_overlap(tc, cand_top, 5);
        const int top20_overlap = test_local_golden_overlap(tc, cand_top, 20);
        const int top64_overlap = test_local_golden_overlap(tc, cand_top, 64);
        const float top20_max_abs =
            test_local_golden_max_abs(tc, cand_logits, 20);

        fprintf(stderr,
                "ds4-test: local golden %s top1 ref=%d cand=%d "
                "top5_overlap=%d/5 top20_overlap=%d/20 top64_overlap=%d/64 "
                "top20_max_abs=%g\n",
                tc->id, tc->top[0].id, cand_top[0],
                top5_overlap, top20_overlap, top64_overlap, top20_max_abs);

        /*
         * This is intentionally tolerant: it is meant to catch substantial
         * backend drift (wrong tiling, skipped work, bad dispatch), not tiny
         * floating-point differences from otherwise sane kernel changes.
         */
        TEST_ASSERT(cand_top[0] == tc->top[0].id);
        TEST_ASSERT(top5_overlap >= 4);
        TEST_ASSERT(top20_overlap >= 15);
        TEST_ASSERT(top64_overlap >= 40);
        TEST_ASSERT(top20_max_abs <= 8.0f);
    } else {
        TEST_ASSERT(false);
    }

    free(cand_logits);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static void test_local_golden_vectors(void) {
    const char *path = getenv("DS4_TEST_LOCAL_GOLDEN_FILE");
    if (!path || !path[0]) {
        path = "tests/test-vectors/flash-0731/local-golden.vec";
    }
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    char *saved_moe_tile_max = test_save_env("DS4_METAL_MOE_TILE_MAX");
    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();
    setenv("DS4_METAL_PREFILL_CHUNK", "4096", 1);
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    unsetenv("DS4_METAL_MOE_TILE_MAX");

    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_local_golden_case tc;
    while (test_read_local_golden_case(fp, &tc)) {
        if (!test_fill_local_golden_case(fp, &tc)) break;
        test_local_golden_case_run(engine, &tc);
    }

    ds4_engine_close(engine);
    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

#define TEST_MPP_EQ_MAX_CASES 8
#define TEST_MPP_EQ_TOPK 20
#define TEST_MPP_EQ_TOP5 5
#define TEST_MPP_EQ_DELTAS 5

typedef struct {
    char id[96];
    int ctx;
    int vocab_size;
    int gen_steps;
    ds4_tokens prompt;
    float *ref_logits;
    int ref_gen[TEST_VEC_MAX_STEPS];
    int ref_gen_len;
} test_mpp_eq_case;

typedef struct {
    int ref_top1;
    int cand_top1;
    int overlap;
    int top5_overlap;
    int max_rank_delta;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
    bool same_top1;
    bool pass;
} test_mpp_eq_result;

typedef struct {
    const char *label;
    int cases;
    int capture_failures;
    int logits_failures;
    int greedy_failures;
    int top1_mismatches;
    int min_overlap;
    int min_top5_overlap;
    int worst_rank_delta;
    float worst_rms;
    float worst_max_abs;
    float worst_top20_max_abs;
} test_mpp_eq_summary;

static void test_mpp_eq_case_free(test_mpp_eq_case *tc) {
    if (!tc) return;
    ds4_tokens_free(&tc->prompt);
    free(tc->ref_logits);
    memset(tc, 0, sizeof(*tc));
}

static void test_logits_topk(const float *logits, int n, int *out, int k) {
    for (int i = 0; i < k; i++) out[i] = -1;
    for (int id = 0; id < n; id++) {
        const float v = logits[id];
        if (!isfinite(v)) continue;
        for (int j = 0; j < k; j++) {
            if (out[j] < 0 || v > logits[out[j]]) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j] = id;
                break;
            }
        }
    }
}

static bool test_topk_contains(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return true;
    }
    return false;
}

static int test_topk_rank(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return i;
    }
    return -1;
}

static void test_note_delta(int *ids, float *ref_vals, float *cand_vals,
                            float *abs_vals, int id, float ref, float cand) {
    const float abs_delta = fabsf(cand - ref);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        if (ids[i] < 0 || abs_delta > abs_vals[i]) {
            for (int j = TEST_MPP_EQ_DELTAS - 1; j > i; j--) {
                ids[j] = ids[j - 1];
                ref_vals[j] = ref_vals[j - 1];
                cand_vals[j] = cand_vals[j - 1];
                abs_vals[j] = abs_vals[j - 1];
            }
            ids[i] = id;
            ref_vals[i] = ref;
            cand_vals[i] = cand;
            abs_vals[i] = abs_delta;
            return;
        }
    }
}

static float test_top_union_max_abs(const float *ref, const float *cand,
                                    const int *ref_top, const int *cand_top, int k) {
    float max_abs = 0.0f;
    for (int i = 0; i < k; i++) {
        if (ref_top[i] >= 0) {
            const float d = fabsf(cand[ref_top[i]] - ref[ref_top[i]]);
            if (d > max_abs) max_abs = d;
        }
        if (cand_top[i] >= 0 && !test_topk_contains(ref_top, k, cand_top[i])) {
            const float d = fabsf(cand[cand_top[i]] - ref[cand_top[i]]);
            if (d > max_abs) max_abs = d;
        }
    }
    return max_abs;
}

/*
 * Metal4/TensorOps equivalence is a smoke test, not a demand for bitwise local
 * logits.  Tensor kernels change precision and reduction order, so the useful
 * invariant here is: no NaNs, same first greedy token, and same short greedy
 * continuation.  Larger logit drift is still printed so it can be compared with
 * official API-vector and long-context recall gates.
 */
static test_mpp_eq_result test_compare_mpp_logits(const test_mpp_eq_case *tc,
                                                  const float *cand_logits,
                                                  bool assert_thresholds) {
    int ref_top[TEST_MPP_EQ_TOPK];
    int cand_top[TEST_MPP_EQ_TOPK];
    test_logits_topk(tc->ref_logits, tc->vocab_size, ref_top, TEST_MPP_EQ_TOPK);
    test_logits_topk(cand_logits, tc->vocab_size, cand_top, TEST_MPP_EQ_TOPK);

    int overlap = 0;
    int top5_overlap = 0;
    int max_rank_delta = 0;
    for (int i = 0; i < TEST_MPP_EQ_TOPK; i++) {
        const int cand_rank = test_topk_rank(cand_top, TEST_MPP_EQ_TOPK, ref_top[i]);
        if (ref_top[i] >= 0 && cand_rank >= 0) {
            overlap++;
            const int rank_delta = abs(cand_rank - i);
            if (rank_delta > max_rank_delta) max_rank_delta = rank_delta;
        }
        if (i < TEST_MPP_EQ_TOP5 &&
            ref_top[i] >= 0 &&
            test_topk_contains(cand_top, TEST_MPP_EQ_TOP5, ref_top[i])) {
            top5_overlap++;
        }
    }

    double sumsq = 0.0;
    float max_abs = 0.0f;
    int nonfinite = 0;
    int delta_ids[TEST_MPP_EQ_DELTAS];
    float delta_ref[TEST_MPP_EQ_DELTAS];
    float delta_cand[TEST_MPP_EQ_DELTAS];
    float delta_abs[TEST_MPP_EQ_DELTAS];
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        delta_ids[i] = -1;
        delta_ref[i] = 0.0f;
        delta_cand[i] = 0.0f;
        delta_abs[i] = 0.0f;
    }

    for (int i = 0; i < tc->vocab_size; i++) {
        if (!isfinite(tc->ref_logits[i]) || !isfinite(cand_logits[i])) {
            nonfinite++;
            continue;
        }
        const float delta = cand_logits[i] - tc->ref_logits[i];
        const float abs_delta = fabsf(delta);
        if (abs_delta > max_abs) max_abs = abs_delta;
        sumsq += (double)delta * (double)delta;
        test_note_delta(delta_ids, delta_ref, delta_cand, delta_abs,
                        (int)i, tc->ref_logits[i], cand_logits[i]);
    }

    const float rms = (float)sqrt(sumsq / (double)tc->vocab_size);
    const float top_abs = test_top_union_max_abs(tc->ref_logits, cand_logits,
                                                 ref_top, cand_top, TEST_MPP_EQ_TOPK);
    const bool same_top1 = ref_top[0] >= 0 && ref_top[0] == cand_top[0];
    test_mpp_eq_result result = {
        .ref_top1 = ref_top[0],
        .cand_top1 = cand_top[0],
        .overlap = overlap,
        .top5_overlap = top5_overlap,
        .max_rank_delta = max_rank_delta,
        .nonfinite = nonfinite,
        .rms = rms,
        .max_abs = max_abs,
        .top20_max_abs = top_abs,
        .same_top1 = same_top1,
        .pass = nonfinite == 0 && same_top1,
    };

    fprintf(stderr,
            "ds4-test: Tensor equivalence %s top1 ref=%d cand=%d top5_overlap=%d/%d overlap=%d/%d max_rank_delta=%d rms=%g max_abs=%g top20_max_abs=%g\n",
            tc->id, ref_top[0], cand_top[0],
            top5_overlap, TEST_MPP_EQ_TOP5,
            overlap, TEST_MPP_EQ_TOPK,
            max_rank_delta, rms, max_abs, top_abs);
    fprintf(stderr, "ds4-test: Tensor equivalence %s largest deltas:", tc->id);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS && delta_ids[i] >= 0; i++) {
        fprintf(stderr, " id=%d ref=%g cand=%g abs=%g",
                delta_ids[i], delta_ref[i], delta_cand[i], delta_abs[i]);
    }
    fputc('\n', stderr);

    if (assert_thresholds) {
        TEST_ASSERT(nonfinite == 0);
        TEST_ASSERT(same_top1);
    }
    return result;
}

static bool test_mpp_capture(ds4_engine *engine, const test_mpp_eq_case *tc,
                             float *logits, int *gen, int *gen_len) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    int n = 0;
    while (ok && n < tc->gen_steps) {
        const int token = ds4_session_argmax(session);
        gen[n++] = token;
        if (n < tc->gen_steps && ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            ok = false;
            TEST_ASSERT(false);
        }
    }
    *gen_len = n;

    ds4_session_free(session);
    return ok;
}

static bool test_mpp_capture_logits_only(ds4_engine *engine,
                                         const test_mpp_eq_case *tc,
                                         float *logits) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    ds4_session_free(session);
    return ok;
}

static bool test_mpp_eq_case_selected(const char *id) {
    const char *filter = getenv("DS4_TEST_MPP_EQ_CASE");
    if (!filter || !filter[0]) return true;

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", filter);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        tok = test_trim_line(tok);
        if (tok[0] && strstr(id, tok)) return true;
    }
    return false;
}

static int test_load_mpp_cases(ds4_engine *engine, test_mpp_eq_case *cases, int cap) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) {
        path = "tests/test-vectors/flash-0731/official.vec";
    }
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return 0;

    int ncase = 0;
    test_vec_case vc;
    while (ncase < cap && test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (!test_mpp_eq_case_selected(vc.id)) continue;
        char *prompt_text = test_read_file(vc.prompt_path);
        TEST_ASSERT(prompt_text != NULL);
        if (!prompt_text) continue;

        test_mpp_eq_case *tc = &cases[ncase++];
        snprintf(tc->id, sizeof(tc->id), "%s", vc.id);
        tc->ctx = vc.ctx;
        tc->vocab_size = ds4_engine_vocab_size(engine);
        tc->gen_steps = vc.nsteps < TEST_VEC_MAX_STEPS ? vc.nsteps : TEST_VEC_MAX_STEPS;
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &tc->prompt);
        free(prompt_text);
        TEST_ASSERT(tc->prompt.len > 0);
    }
    fclose(fp);
    return ncase;
}

static void test_mpp_summary_init(test_mpp_eq_summary *summary, const char *label) {
    memset(summary, 0, sizeof(*summary));
    summary->label = label;
    summary->min_overlap = TEST_MPP_EQ_TOPK;
    summary->min_top5_overlap = TEST_MPP_EQ_TOP5;
}

static void test_mpp_summary_note_logits(test_mpp_eq_summary *summary,
                                         const test_mpp_eq_result *result) {
    if (!result->pass) summary->logits_failures++;
    if (!result->same_top1) summary->top1_mismatches++;
    if (result->overlap < summary->min_overlap) summary->min_overlap = result->overlap;
    if (result->top5_overlap < summary->min_top5_overlap) {
        summary->min_top5_overlap = result->top5_overlap;
    }
    if (result->max_rank_delta > summary->worst_rank_delta) {
        summary->worst_rank_delta = result->max_rank_delta;
    }
    if (result->rms > summary->worst_rms) summary->worst_rms = result->rms;
    if (result->max_abs > summary->worst_max_abs) summary->worst_max_abs = result->max_abs;
    if (result->top20_max_abs > summary->worst_top20_max_abs) {
        summary->worst_top20_max_abs = result->top20_max_abs;
    }
}

static void test_mpp_summary_print(const test_mpp_eq_summary *summary) {
    fprintf(stderr,
            "ds4-test: Tensor summary route=%s cases=%d capture_fail=%d logits_fail=%d greedy_fail=%d top1_mismatch=%d min_top5_overlap=%d/%d min_overlap=%d/%d worst_rank_delta=%d worst_rms=%g worst_max_abs=%g worst_top20_max_abs=%g\n",
            summary->label,
            summary->cases,
            summary->capture_failures,
            summary->logits_failures,
            summary->greedy_failures,
            summary->top1_mismatches,
            summary->min_top5_overlap,
            TEST_MPP_EQ_TOP5,
            summary->min_overlap,
            TEST_MPP_EQ_TOPK,
            summary->worst_rank_delta,
            summary->worst_rms,
            summary->worst_max_abs,
            summary->worst_top20_max_abs);
}

static void test_run_mpp_candidate(const char *label,
                                   test_mpp_eq_case *cases,
                                   int ncase) {
    fprintf(stderr, "ds4-test: Tensor equivalence candidate route=%s\n", label);
    test_mpp_eq_summary summary;
    test_mpp_summary_init(&summary, label);
    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        const int vocab_size = ncase > 0 ? cases[0].vocab_size : 0;
        float *cand_logits = malloc((size_t)vocab_size * sizeof(cand_logits[0]));
        TEST_ASSERT(cand_logits != NULL);
        if (cand_logits) {
            for (int i = 0; i < ncase; i++) {
                test_mpp_eq_case *tc = &cases[i];
                if (!tc->ref_logits) continue;
                int cand_gen[TEST_VEC_MAX_STEPS] = {0};
                int cand_gen_len = 0;
                if (!test_mpp_capture(cand_engine, tc, cand_logits, cand_gen, &cand_gen_len)) {
                    summary.capture_failures++;
                    continue;
                }
                summary.cases++;
                test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_logits, true);
                test_mpp_summary_note_logits(&summary, &result);
                TEST_ASSERT(cand_gen_len == tc->ref_gen_len);
                if (cand_gen_len != tc->ref_gen_len) summary.greedy_failures++;
                for (int j = 0; j < tc->ref_gen_len && j < cand_gen_len; j++) {
                    if (cand_gen[j] != tc->ref_gen[j]) {
                        fprintf(stderr,
                                "ds4-test: Tensor equivalence %s greedy token mismatch step=%d ref=%d cand=%d\n",
                                tc->id, j, tc->ref_gen[j], cand_gen[j]);
                        summary.greedy_failures++;
                    }
                    TEST_ASSERT(cand_gen[j] == tc->ref_gen[j]);
                }
            }
            free(cand_logits);
        }
        ds4_engine_close(cand_engine);
    }
    test_mpp_summary_print(&summary);
}

static void test_metal_mpp_equivalence(void) {
    test_close_engines();

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);

    test_run_mpp_candidate("auto", cases, ncase);

    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

static void test_streaming_decode_prefill_correctness(void) {
    test_close_engines();
    if (!test_env_bool("DS4_TEST_SSD_STREAMING")) {
        fprintf(stderr,
                "ds4-test: streaming decode-prefill correctness skipped "
                "(set DS4_TEST_SSD_STREAMING=1 to enable)\n");
        return;
    }

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();

    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);

    unsetenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL");
    unsetenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR");

    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        for (int i = 0; i < ncase; i++) {
            test_mpp_eq_case *tc = &cases[i];
            if (!tc->ref_logits) continue;

            float *cand_cold = malloc((size_t)tc->vocab_size * sizeof(cand_cold[0]));
            float *cand_warm_a = malloc((size_t)tc->vocab_size * sizeof(cand_warm_a[0]));
            float *cand_warm_b = malloc((size_t)tc->vocab_size * sizeof(cand_warm_b[0]));
            TEST_ASSERT(cand_cold != NULL);
            TEST_ASSERT(cand_warm_a != NULL);
            TEST_ASSERT(cand_warm_b != NULL);
            if (!cand_cold || !cand_warm_a || !cand_warm_b) {
                free(cand_cold);
                free(cand_warm_a);
                free(cand_warm_b);
                continue;
            }

            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_cold));
            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_warm_a));
            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_warm_b));

            test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_cold, false);
            TEST_ASSERT(result.nonfinite == 0);
            TEST_ASSERT(result.top5_overlap >= 2);
            TEST_ASSERT(result.overlap >= 10);
            TEST_ASSERT(result.rms <= 4.0f);
            TEST_ASSERT(result.top20_max_abs <= 12.0f);

            int cold_warm_neq = 0;
            int warm_repeat_neq = 0;
            int repeat_nonfinite = 0;
            float cold_warm_max_abs = 0.0f;
            float warm_repeat_max_abs = 0.0f;
            for (int j = 0; j < tc->vocab_size; j++) {
                if (!isfinite(cand_cold[j]) ||
                    !isfinite(cand_warm_a[j]) ||
                    !isfinite(cand_warm_b[j])) {
                    repeat_nonfinite++;
                    continue;
                }
                const float cold_warm_d = fabsf(cand_cold[j] - cand_warm_a[j]);
                if (cold_warm_d != 0.0f) cold_warm_neq++;
                if (cold_warm_d > cold_warm_max_abs) cold_warm_max_abs = cold_warm_d;
                const float warm_repeat_d = fabsf(cand_warm_a[j] - cand_warm_b[j]);
                if (warm_repeat_d != 0.0f) warm_repeat_neq++;
                if (warm_repeat_d > warm_repeat_max_abs) {
                    warm_repeat_max_abs = warm_repeat_d;
                }
            }
            TEST_ASSERT(repeat_nonfinite == 0);
            TEST_ASSERT(cold_warm_neq == 0);
            TEST_ASSERT(warm_repeat_neq == 0);
            fprintf(stderr,
                    "ds4-test: streaming decode-prefill %s cold_warm_neq=%d "
                    "cold_warm_max_abs=%g warm_repeat_neq=%d "
                    "warm_repeat_max_abs=%g top1 canonical=%d decode=%d\n",
                    tc->id,
                    cold_warm_neq,
                    cold_warm_max_abs,
                    warm_repeat_neq,
                    warm_repeat_max_abs,
                    result.ref_top1,
                    result.cand_top1);

            free(cand_cold);
            free(cand_warm_a);
            free(cand_warm_b);
        }
        ds4_engine_close(cand_engine);
    }

    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

#define TEST_LIST_FILES_USER_PROMPT \
    "Use the list_files tool to list the current directory exactly once, " \
    "then report the listed files and stop."

#define TEST_LIST_FILES_TOOL_JSON \
    "{\"type\":\"function\",\"function\":{" \
        "\"name\":\"list_files\"," \
        "\"description\":\"List files in a directory.\"," \
        "\"parameters\":{\"type\":\"object\",\"properties\":{" \
            "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}" \
        "},\"required\":[\"path\"]}" \
    "}}"

#define TEST_LIST_FILES_RESULT "[\"README.md\",\"Makefile\",\"ds4.c\",\"metal\"]"

static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\""
            TEST_LIST_FILES_USER_PROMPT
        "\"}],"
        "\"tools\":[" TEST_LIST_FILES_TOOL_JSON "],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static char *test_tool_result_request_json(const char *assistant_content,
                                           const tool_call *call) {
    if (!call || !call->id || !call->id[0]) return NULL;

    buf b = {0};
    buf_puts(&b,
        "{\"model\":\"deepseek-v4-flash\",\"messages\":["
        "{\"role\":\"user\",\"content\":");
    json_escape(&b, TEST_LIST_FILES_USER_PROMPT);
    buf_puts(&b, "},{\"role\":\"assistant\",\"content\":");
    json_escape(&b, assistant_content ? assistant_content : "");
    buf_puts(&b, ",\"tool_calls\":[{\"id\":");
    json_escape(&b, call->id);
    buf_puts(&b, ",\"type\":\"function\",\"function\":{\"name\":");
    json_escape(&b, call->name ? call->name : "");
    buf_puts(&b, ",\"arguments\":");
    json_escape(&b, call->arguments ? call->arguments : "{}");
    buf_puts(&b, "}}]},{\"role\":\"tool\",\"tool_call_id\":");
    json_escape(&b, call->id);
    buf_puts(&b, ",\"content\":");
    json_escape(&b, TEST_LIST_FILES_RESULT);
    buf_puts(&b,
        "}],\"tools\":[" TEST_LIST_FILES_TOOL_JSON "],"
        "\"tool_choice\":\"auto\",\"think\":false,"
        "\"temperature\":0,\"max_tokens\":256,\"stream\":false}");
    return buf_take(&b);
}

/* A complete tool call inside unclosed reasoning is recovered directly. The
 * detector must wait for the complete block, and the parser must keep only the
 * preceding prose in reasoning_content. */
static void test_think_tool_recovery(void) {
    const char *generated =
        "The user wants a directory listing.\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"list_files\">\n"
        DS4_PARAM_START " name=\"path\" string=\"true\">." DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;

    buf text = {0};
    size_t scan_from = 0;
    bool complete = false;
    for (size_t i = 0; generated[i]; i++) {
        buf_append(&text, generated + i, 1);
        complete = complete_tool_call_inside_thinking(text.ptr, text.len,
                                                      &scan_from);
        TEST_ASSERT(complete == (generated[i + 1] == '\0'));
    }
    TEST_ASSERT(complete);

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr, true,
                                             &content, &reasoning, &calls);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));
    TEST_ASSERT(calls.v[0].arguments && strstr(calls.v[0].arguments, "\"path\": \".\""));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(reasoning && !strcmp(reasoning, "The user wants a directory listing."));

    fprintf(stderr,
            "ds4-test: think-tool-recovery complete=%d calls=%d name=%s\n",
            complete ? 1 : 0, calls.len, calls.len ? calls.v[0].name : "-");

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
}

typedef struct {
    char *raw;
    char *content;
    char *reasoning;
    tool_calls calls;
    const char *finish;
} test_chat_turn;

static void test_chat_turn_free(test_chat_turn *turn) {
    if (!turn) return;
    free(turn->raw);
    free(turn->content);
    free(turn->reasoning);
    tool_calls_free(&turn->calls);
    memset(turn, 0, sizeof(*turn));
}

/* Run the same greedy stop/tool-marker/response parse path needed by the server,
 * while keeping one session alive so the next request genuinely continues from
 * this sampled turn. */
static bool test_generate_chat_turn(ds4_engine *engine, ds4_session *session,
                                    const request *r, test_chat_turn *turn) {
    memset(turn, 0, sizeof(*turn));
    char err[160] = {0};
    if (ds4_session_sync(session, &r->prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-test: tool-call sync failed: %s\n", err);
        turn->finish = "error";
        return false;
    }

    buf text = {0};
    uint64_t rng = 123;
    const char *finish = "length";
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    bool decode_ok = true;

    for (int i = 0; i < r->max_tokens; i++) {
        int token = ds4_session_sample(session, r->temperature, r->top_k,
                                       r->top_p, r->min_p, &rng);
        if (ds4_token_is_stop_for_think_mode(engine, token, r->think_mode)) {
            finish = "stop";
            break;
        }
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            finish = "error";
            decode_ok = false;
            break;
        }

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        if (r->has_tools) {
            observe_tool_markers(text.ptr ? text.ptr : "",
                                 &saw_tool_start, &saw_tool_end, NULL);
            if (saw_tool_end) {
                finish = "tool_calls";
                break;
            }
        }
    }

    turn->raw = buf_take(&text);
    turn->finish = finish;
    if (!decode_ok) {
        fprintf(stderr, "ds4-test: tool-call decode failed: %s\n", err);
        return false;
    }

    bool recovered = false;
    bool parsed = parse_generated_message_for_response_for_syntax(
        r->model_syntax,
        turn->raw ? turn->raw : "",
        r->has_tools,
        saw_tool_start,
        ds4_think_mode_enabled(r->think_mode),
        &turn->finish,
        err,
        sizeof(err),
        &turn->content,
        &turn->reasoning,
        &turn->calls,
        &recovered);
    if (turn->calls.len > 0) turn->finish = "tool_calls";
    if (!parsed) {
        fprintf(stderr,
                "ds4-test: generated message parse failed: %s recovered=%d raw=%s\n",
                err, recovered ? 1 : 0, turn->raw ? turn->raw : "");
    }
    return parsed;
}

static void test_tool_call_quality_one(bool quality) {
    ds4_engine *engine = test_get_engine(quality);
    if (!engine) return;

    server s = {0};
    s.engine = engine;
    pthread_mutex_init(&s.tool_mu, NULL);

    request first_request = {0};
    request second_request = {0};
    ds4_session *session = NULL;
    test_chat_turn first = {0};
    test_chat_turn second = {0};
    char *second_body = NULL;
    char err[160] = {0};

    bool request_ok = parse_chat_request(engine, &s,
                                         test_tool_call_request_json(),
                                         512, 32768, &first_request,
                                         err, sizeof(err));
    TEST_ASSERT(request_ok);
    if (!request_ok) goto done;

    bool session_ok = ds4_session_create(&session, engine, 32768) == 0;
    TEST_ASSERT(session_ok);
    if (!session_ok) goto done;

    bool first_ok = test_generate_chat_turn(engine, session,
                                            &first_request, &first);
    TEST_ASSERT(first_ok);
    TEST_ASSERT(first.finish && !strcmp(first.finish, "tool_calls"));
    TEST_ASSERT(first.calls.len == 1);
    TEST_ASSERT(first.calls.len == 1 && first.calls.v[0].name &&
                !strcmp(first.calls.v[0].name, "list_files"));
    TEST_ASSERT(first.calls.raw_tool_text && first.calls.raw_tool_text[0]);
    if (!first_ok || first.calls.len != 1 ||
        !first.calls.v[0].name ||
        strcmp(first.calls.v[0].name, "list_files") ||
        !first.calls.raw_tool_text || !first.calls.raw_tool_text[0]) {
        goto done;
    }

    /* Use the real response-side id assignment and exact sampled-DSML memory.
     * The same id is serialized on both the assistant call and tool result. */
    assign_tool_call_ids(&s, &first.calls, API_OPENAI);
    TEST_ASSERT(first.calls.v[0].id && first.calls.v[0].id[0]);
    if (!first.calls.v[0].id || !first.calls.v[0].id[0]) goto done;
    tool_memory_remember(&s, &first.calls);

    second_body = test_tool_result_request_json(first.content,
                                                &first.calls.v[0]);
    TEST_ASSERT(second_body != NULL);
    if (!second_body) goto done;

    err[0] = '\0';
    request_ok = parse_chat_request(engine, &s, second_body,
                                    512, 32768, &second_request,
                                    err, sizeof(err));
    TEST_ASSERT(request_ok);
    if (!request_ok) goto done;
    TEST_ASSERT(second_request.tool_replay.mem == 1);
    TEST_ASSERT(second_request.tool_replay.disk == 0);
    TEST_ASSERT(second_request.tool_replay.canonical == 0);
    TEST_ASSERT(second_request.tool_replay.missing_ids == 0);

    bool second_ok = test_generate_chat_turn(engine, session,
                                             &second_request, &second);
    TEST_ASSERT(second_ok);
    TEST_ASSERT(second.finish && !strcmp(second.finish, "stop"));
    TEST_ASSERT(second.calls.len == 0);
    TEST_ASSERT(second.content && second.content[0]);

    fprintf(stderr,
            "ds4-test: post-tool-result turn1 finish_reason=%s tool_calls=%d "
            "turn2 finish_reason=%s tool_calls=%d replay_mem=%d\n",
            first.finish ? first.finish : "-", first.calls.len,
            second.finish ? second.finish : "-", second.calls.len,
            second_request.tool_replay.mem);

done:
    free(second_body);
    test_chat_turn_free(&second);
    test_chat_turn_free(&first);
    ds4_session_free(session);
    request_free(&second_request);
    request_free(&first_request);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_tool_call_quality(void) {
    fprintf(stderr, "ds4-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "ds4-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}

/* Greedy speculative decode: capture committed tokens and the largest accepted
 * chunk, so the caller can confirm the multi-row verify path actually ran. */
static bool test_mtp_capture_speculative(ds4_engine *engine, const ds4_tokens *prompt,
                                         int max_tokens, int *out, int *out_len,
                                         int *max_chunk) {
    *out_len = 0;
    *max_chunk = 0;
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);

    const int eos = ds4_token_eos(engine);
    int n = 0;
    bool stop = false;
    while (ok && !stop && n < max_tokens) {
        const int token = ds4_session_argmax(session);
        if (token == eos) break;

        int toks[17]; /* base token + draft depth, which the engine clamps to 16 */
        const int ntok = ds4_session_eval_speculative_argmax(
            session, token, max_tokens - n, eos, toks,
            (int)(sizeof(toks) / sizeof(toks[0])), err, sizeof(err));
        if (ntok < 0) { ok = false; TEST_ASSERT(false); break; }
        if (ntok > *max_chunk) *max_chunk = ntok;

        for (int j = 0; j < ntok; j++) {
            if (toks[j] == eos) { stop = true; break; }
            out[n++] = toks[j];
            if (n >= max_tokens) { stop = true; break; }
        }
    }

    *out_len = n;
    ds4_session_free(session);
    return ok;
}

/* Replay toks[] through plain decode and return the largest gap between a
 * position's argmax logit and the committed token's logit.  Correct speculation
 * commits (near-)argmax tokens (gap ~0); a mis-committed token gives a big gap. */
static bool test_mtp_worst_argmax_gap(ds4_engine *engine, const ds4_tokens *prompt,
                                      const int *toks, int n,
                                      float *worst_gap, int *worst_at) {
    *worst_gap = 0.0f;
    *worst_at = -1;
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);

    for (int i = 0; ok && i < n; i++) {
        ds4_token_score best, cur;
        ok = ds4_session_top_logprobs(session, &best, 1) >= 1 &&
             ds4_session_token_logprob(session, toks[i], &cur) == 1;
        TEST_ASSERT(ok);
        if (!ok) break;

        const float gap = best.logit - cur.logit;
        if (gap > *worst_gap) { *worst_gap = gap; *worst_at = i; }
        if (ds4_session_eval(session, toks[i], err, sizeof(err)) != 0) { ok = false; TEST_ASSERT(false); break; }
    }

    ds4_session_free(session);
    return ok;
}

/* Verbatim-copy task: keeps the model confident (a mis-committed token shows as
 * a large argmax gap) and draft acceptance high (so the multi-row verify path is
 * exercised across the generation). */
static const char *test_mtp_copy_prompt(void) {
    return
        "Reproduce the following C code EXACTLY, character for character, "
        "inside a single code block and output nothing else:\n\n"
        "```c\n"
        "static uint32_t clamp_u32(uint32_t v, uint32_t lo, uint32_t hi) {\n"
        "    if (v < lo) return lo;\n"
        "    if (v > hi) return hi;\n"
        "    return v;\n"
        "}\n"
        "\n"
        "static uint32_t ring_advance(uint32_t pos, uint32_t cap) {\n"
        "    uint32_t next = pos + 1u;\n"
        "    return next >= cap ? 0u : next;\n"
        "}\n"
        "\n"
        "static int scratch_init(scratch *s, uint32_t ctx_size) {\n"
        "    if (ctx_size == 0u) ctx_size = 1u;\n"
        "    s->ctx_size = ctx_size;\n"
        "    s->comp_cap = ctx_size / 4u + 2u;\n"
        "    s->rows = clamp_u32(s->comp_cap, 1u, 4096u);\n"
        "    s->head = 0u;\n"
        "    return s->rows > 0u ? 0 : -1;\n"
        "}\n"
        "```\n";
}

#define TEST_MTP_MAXGEN 256
#define TEST_DSPARK_MAXGEN 128

static ds4_engine *test_open_dspark_engine(const char *support_path) {
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = test_model_path(),
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = false,
        .ssd_streaming = test_env_bool("DS4_TEST_SSD_STREAMING"),
        .ssd_streaming_cold = test_env_bool("DS4_TEST_SSD_STREAMING_COLD"),
        .ssd_streaming_cache_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS"),
        .ssd_streaming_cache_bytes =
            test_env_gib("DS4_TEST_SSD_STREAMING_CACHE_GB"),
        .ssd_streaming_preload_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_PRELOAD_EXPERTS"),
        .mtp_path = support_path,
        .mtp_draft_tokens = 0,
        .dspark = true,
        .dspark_confidence_threshold = 0.9f,
        .dspark_confidence_threshold_set = true,
    };
    const int rc = ds4_engine_open(&engine, &opt);
    TEST_ASSERT(rc == 0);
    return rc == 0 ? engine : NULL;
}

/* Regression for the swapped top-k arguments in metal_graph_verify_suffix_tops
 * at draft depth > 2.  Replays the committed speculative tokens through plain
 * decode and requires each to be a (near-)argmax: that is the verify invariant,
 * and unlike comparing token streams it tolerates the near-greedy tie
 * divergences.  Needs an MTP head, so it self-skips without DS4_TEST_MTP. */
static void test_mtp_verify_depth(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_has_mtp(engine)) {
        fprintf(stderr, "ds4-test: mtp-verify-depth skipped (set DS4_TEST_MTP to an MTP GGUF)\n");
        return;
    }
    TEST_ASSERT(ds4_engine_mtp_draft_tokens(engine) > 2);

    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(prompt.len > 0);

    int *spec = malloc((size_t)TEST_MTP_MAXGEN * sizeof(*spec));
    TEST_ASSERT(spec != NULL);
    if (spec && prompt.len > 0) {
        int nspec = 0, max_chunk = 0;
        const bool ok_spec = test_mtp_capture_speculative(engine, &prompt, TEST_MTP_MAXGEN,
                                                          spec, &nspec, &max_chunk);
        TEST_ASSERT(ok_spec);
        TEST_ASSERT(max_chunk > 1);  /* multi-token chunks committed: the multi-row path ran */
        TEST_ASSERT(nspec > 128);    /* enough output to surface the bug, incl. a spurious-EOS truncation */

        float worst_gap = 0.0f;
        int worst_at = -1;
        const bool ok_check = test_mtp_worst_argmax_gap(engine, &prompt, spec, nspec,
                                                        &worst_gap, &worst_at);
        TEST_ASSERT(ok_check);
        fprintf(stderr, "ds4-test: mtp-verify-depth nspec=%d max_chunk=%d worst_argmax_gap=%.3f at=%d\n",
                nspec, max_chunk, worst_gap, worst_at);
        TEST_ASSERT(worst_gap <= 2.0f);  /* correct: ~0; bug: ~21 on the reference model */
    }

    free(spec);
    ds4_tokens_free(&prompt);
}

/* Same invariant as the MTP depth smoke, but for the DSpark support model.  This
 * is separate from the fixture because it teacher-forces every committed token
 * through normal decode and directly checks that DSpark never commits a token
 * that was not near the target argmax. */
static void test_dspark_verify_depth(void) {
    const char *support = getenv("DS4_TEST_DSPARK");
    if (!support || !support[0]) {
        fprintf(stderr, "ds4-test: dspark-verify-depth skipped (set DS4_TEST_DSPARK to a DSpark support GGUF)\n");
        return;
    }

    char *saved_scheduler = test_save_env("DS4_DSPARK_SCHEDULER");
    setenv("DS4_DSPARK_SCHEDULER", "0", 1);

    ds4_engine *engine = test_open_dspark_engine(support);
    ds4_tokens prompt = {0};
    int *spec = NULL;

    if (engine) {
        const int draft_depth = ds4_engine_mtp_draft_tokens(engine);
        TEST_ASSERT(draft_depth > 2);

        ds4_chat_begin(engine, &prompt);
        ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
        ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
        TEST_ASSERT(prompt.len > 0);

        spec = malloc((size_t)TEST_DSPARK_MAXGEN * sizeof(*spec));
        TEST_ASSERT(spec != NULL);
        if (draft_depth > 2 && spec && prompt.len > 0) {
            int nspec = 0, max_chunk = 0;
            const bool ok_spec = test_mtp_capture_speculative(engine, &prompt,
                                                              TEST_DSPARK_MAXGEN,
                                                              spec, &nspec,
                                                              &max_chunk);
            TEST_ASSERT(ok_spec);
            TEST_ASSERT(max_chunk > 1);
            TEST_ASSERT(nspec > 64);

            float worst_gap = 0.0f;
            int worst_at = -1;
            const bool ok_check = test_mtp_worst_argmax_gap(engine, &prompt,
                                                            spec, nspec,
                                                            &worst_gap,
                                                            &worst_at);
            TEST_ASSERT(ok_check);
            fprintf(stderr,
                    "ds4-test: dspark-verify-depth nspec=%d max_chunk=%d draft_depth=%d worst_argmax_gap=%.3f at=%d\n",
                    nspec, max_chunk, draft_depth, worst_gap, worst_at);
            TEST_ASSERT(worst_gap <= 2.0f);
        }
    }

    free(spec);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    test_restore_env("DS4_DSPARK_SCHEDULER", saved_scheduler);
}

#if defined(__APPLE__)
/* Model-independent coverage for the supported product slice.  Keep this
 * list deliberately separate from test_metal_kernel_group: that historical
 * umbrella also runs GLM/DSpark and optional-source compatibility cases. */
static void test_laguna_metal_core(void) {
    if (!ds4_gpu_init()) {
        TEST_ASSERT(false);
        return;
    }

    test_dflash_capture_nonfinite_sanitize();
    test_metal_f16_matvec_fast_nr0_4();
    test_metal_f16_prefill_matmul();
    test_metal_q8_0_prefill_matmul();
    test_metal_pack_slot_rows_f32();
    test_metal_store_raw_kv_batch_wrap();
    test_metal_q8_0_decode_pair_exact();
    test_metal_q8_0_output_nr4_exact();
    test_metal_q8_decode_lifecycle_snapshot();
    test_metal_add_rms_norm_weight_rows_exact();
    test_metal_laguna_decode_ladder_ordering_exact();
    test_metal_laguna_swa_gqa3_numeric_ab();
    test_metal_laguna_staged_swa_exact();
    test_metal_laguna_swa_gqa3_scope();
    test_laguna_gqa3_decode_numeric();
    test_metal_laguna_qk_norm_rope_pair_exact();
    test_metal_glm_qmv_r1_exact();
    test_metal_router_simd_finalize_exact();
    test_metal_router_weights_batch_exact();
}
#endif
#endif

static void test_server_unit_group(void) {
    ds4_server_unit_tests_run();
}

typedef void (*test_fn)(void);

typedef struct {
    const char *flag;
    const char *name;
    const char *desc;
    test_fn fn;
    bool explicit_only;
} ds4_test_entry;

static const ds4_test_entry test_entries[] = {
    {"--laguna-architecture", "laguna-architecture",
     "accept literal Laguna GGUF architecture and reject legacy/missing values",
     test_laguna_architecture_gate, false},
    {"--laguna-selector-parser", "laguna-selector-parser",
     "strict Laguna selector and prefill-route parser boundaries",
     test_laguna_selector_parser, false},
#ifndef DS4_NO_GPU
#if defined(__APPLE__)
    {"--laguna-metal-core", "laguna-metal-core",
     "model-independent Laguna Metal/DFlash kernel and topology regressions",
     test_laguna_metal_core, true},
#endif
    {"--laguna-attention-numeric", "laguna-attention-numeric",
     "Laguna decode attention against a double-precision reference",
     test_laguna_gqa3_decode_numeric, false},
#if defined(__APPLE__)
    {"--laguna-prefill-direct-kv-ab", "laguna-prefill-direct-kv-ab",
     "bit-exact A/B of the opt-in direct non-SWA prefill KV store",
     test_laguna_prefill_direct_kv_ab, false},
    {"--laguna-prefill-direct-kv-invalid", "laguna-prefill-direct-kv-invalid",
     "malformed direct-KV selector fails before split-row KV mutation",
     test_laguna_prefill_direct_kv_invalid_mode, false},
    {"--laguna-dense-q8-batch-swiglu", "laguna-dense-q8-batch-swiglu",
     "batched fused dense Q8 gate/up+SwiGLU against a double reference",
     test_laguna_dense_q8_batch_swiglu, false},
#endif
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    {"--cuda-laguna-moe", "cuda-laguna-moe",
     "CUDA Laguna Q8 signal and Q4/Q3 MoE prefill/decode numerics",
     test_cuda_laguna_moe_decode_prefill, false},
#endif
    {"--long-context", "long-context", "long-context story fact-recall regression", test_long_story_fact_recall, false},
    {"--tool-call-quality", "tool-call-quality", "model emits valid DSML tool calls", test_tool_call_quality, false},
    {"--think-tool-recovery", "think-tool-recovery", "recover a complete tool call emitted inside unclosed reasoning", test_think_tool_recovery, false},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison on the standard Metal path", test_official_logprob_vectors, false},
    {"--metal-ssd-streaming-cache-pressure", "metal-ssd-streaming-cache-pressure", "Metal SSD-streaming layer-batched decode cache-pressure repro for issue #384", test_metal_ssd_streaming_cache_pressure, false},
    {"--local-golden-vectors", "local-golden-vectors", "local top-k/logit drift regression for long Metal prefill", test_local_golden_vectors, false},
    {"--metal-short-prefill", "metal-short-prefill", "Metal ratio-4 short prefill regression", test_metal_short_prefill_ratio4, false},
    {"--metal-kernels", "metal-kernels", "isolated Metal kernel numeric regressions", test_metal_kernel_group, false},
#if defined(__APPLE__)
    {"--metal-graph-malformed-q8", "metal-graph-malformed-q8",
     "malformed Q8 graph admission fails before graph/KV/command mutation",
     test_metal_graph_malformed_q8_admission, true},
    {"--metal-graph-output-preflight", "metal-graph-output-preflight",
     "output-head selector/PSO failure leaves raw graph state untouched",
     test_metal_graph_output_preflight_failure, true},
    {"--metal-laguna-q8-lmhead-screen", "metal-laguna-q8-lmhead-screen",
     "certified Laguna Q8 lm-head top-1 screen (focused opt-in)",
     test_metal_laguna_q8_lmhead_screen_focused, true},
    {"--metal-laguna-rope-atlas", "metal-laguna-rope-atlas",
     "strict Laguna RoPE angle-atlas exactness and preflight (focused opt-in)",
     test_metal_laguna_rope_atlas_exact, true},
    {"--metal-glm-qmv-r1", "metal-glm-qmv-r1",
     "resident decode-only GLM QMV one-row-per-SIMD exactness",
     test_metal_glm_qmv_r1_exact, false},
    {"--metal-glm-router-simd-topk", "metal-glm-router-simd-topk",
     "exact finite-domain GLM/Laguna router SIMD top-k selector",
     test_metal_glm_router_simd_topk_exact, true},
    {"--metal-parallel-ffn-lifecycle", "metal-parallel-ffn-lifecycle",
     "submit/discard parallel-FFN lifecycle cleanup and follow-up batch",
     test_metal_parallel_ffn_terminal_lifecycle, true},
#endif
    {"--metal-tensor-equivalence", "metal-tensor-equivalence", "fast/quality Metal prompt-logit and greedy equivalence", test_metal_mpp_equivalence, false},
    {"--streaming-decode-prefill-correctness", "streaming-decode-prefill-correctness", "streaming decode-style cold prefill drift and repeatability", test_streaming_decode_prefill_correctness, false},
    {"--mtp-verify-depth", "mtp-verify-depth", "MTP speculative verify commits autoregressive-identical tokens at draft depth > 2", test_mtp_verify_depth, false},
    {"--dspark-verify-depth", "dspark-verify-depth", "DSpark speculative verify commits autoregressive-identical tokens at draft depth > 2", test_dspark_verify_depth, false},
#endif
    {"--server", "server", "server parser/rendering/cache unit tests", test_server_unit_group, false},
};

static void test_print_help(const char *prog) {
    printf("Usage: %s [--all | TEST...]\n\n", prog);
    puts("Tests:");
    puts("  --all");
    puts("      Run every non-explicit test. This is the default, ordered from slower to faster.");
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        printf("  %-20s %s%s\n",
               test_entries[i].flag,
               test_entries[i].desc,
               test_entries[i].explicit_only ? " [explicit-only]" : "");
    }
    puts("  --list");
    puts("      Print test names only.");
#ifndef DS4_NO_GPU
    puts("  --metal-mpp-equivalence");
    puts("      Compatibility alias for --metal-tensor-equivalence.");
#endif
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  DS4_TEST_MODEL=FILE        Model path. Default: ds4flash.gguf");
    puts("  DS4_TEST_BACKEND=cpu       Run model tests on CPU instead of Metal/CUDA.");
    puts("  DS4_TEST_SSD_STREAMING=1   Run model tests through Metal SSD streaming.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_GB=N  Streaming routed expert cache in GiB.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=N  Streaming routed expert cache count.");
    puts("  DS4_TEST_SSD_STREAMING_COLD=1  Skip streaming hot expert preload.");
    puts("  DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL=1  Force canonical streamed cold prefill.");
    puts("  DS4_METAL_GLM_QMV_R1=1  Enable resident decode-only one-row-per-SIMD GLM QMV.");
    puts("  DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK=1  Enable exact finite-domain Laguna router top-k SIMD selector.");
    puts("  DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK_TRACE=1  Collect optimized/fallback selector row counters.");
    puts("  DS4_METAL_LAGUNA_ROUTER_DECODE_FUSED=1  Enable the strict opt-in normal one-token Laguna fused router.");
    puts("  DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32=1  Require exact Laguna Q/K norm+RoPE SIMD32 (only literal 1 enables; invalid values fail).");
    puts("  DS4_METAL_LAGUNA_ROPE_ATLAS=1  Generate exact Laguna global/SWA RoPE angle families once per batch (only literal 1 enables; invalid values fail).");
    puts("  DS4_LAGUNA_PREFILL_QK_NORM_ROPE_PAIRED=1  Enable ordinary Laguna prefill paired Q/K norm/RoPE.");
    puts("  DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU=1  Enable opt-in Laguna leading-dense Q8 fused gate/up+SwiGLU.");
    puts("  DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_TRACE=1  Report each fused/stock route once after a waited graph.");
    puts("  DS4_TEST_LONG_PROMPT=FILE  Rendered long-context story fact prompt.");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Official fixture. Default: flash-0731/official.vec.");
    puts("  DS4_TEST_LOCAL_GOLDEN_FILE=FILE  Local fixture. Default: flash-0731/local-golden.vec.");
    puts("  DS4_TEST_MPP_EQ_CASE=NAME  Run only Tensor equivalence cases whose id contains NAME.");
    puts("  DS4_TEST_MTP=FILE         Legacy MTP support GGUF for --mtp-verify-depth.");
    puts("  DS4_TEST_DSPARK=FILE      DSpark support GGUF for --dspark-verify-depth.");
}

static const ds4_test_entry *test_find_entry(const char *arg) {
#ifndef DS4_NO_GPU
    if (!strcmp(arg, "--metal-mpp-equivalence")) {
        arg = "--metal-tensor-equivalence";
    }
#endif
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!strcmp(arg, test_entries[i].flag)) return &test_entries[i];
    }
    return NULL;
}

static void test_run_entry(const ds4_test_entry *entry) {
    int before = test_failures;
    fprintf(stderr, "%s:\n", entry->name);
    entry->fn();
    fprintf(stderr, "%s: ", entry->name);
    ds4_log(stderr,
            test_failures == before ? DS4_LOG_OK : DS4_LOG_ERROR,
            "%s",
            test_failures == before ? "OK" : "ERR");
    fputc('\n', stderr);
}

int main(int argc, char **argv) {
    bool run_all = argc == 1;
    bool selected[sizeof(test_entries) / sizeof(test_entries[0])] = {0};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) {
            run_all = true;
        } else if (!strcmp(argv[i], "--list")) {
            for (size_t j = 0; j < sizeof(test_entries) / sizeof(test_entries[0]); j++) {
                puts(test_entries[j].flag);
            }
            return 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            test_print_help(argv[0]);
            return 0;
        } else {
            const ds4_test_entry *entry = test_find_entry(argv[i]);
            if (!entry) {
                fprintf(stderr, "ds4-test: unknown test switch: %s\n", argv[i]);
                test_print_help(argv[0]);
                return 2;
            }
            selected[(size_t)(entry - test_entries)] = true;
        }
    }

    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!selected[i] || !test_entries[i].explicit_only) continue;
        for (size_t j = 0; j < sizeof(test_entries) / sizeof(test_entries[0]); j++) {
            if ((run_all && !test_entries[j].explicit_only) ||
                (selected[j] && j != i)) {
                fprintf(stderr,
                        "ds4-test: %s is explicit-only and must run alone "
                        "in a fresh process\n",
                        test_entries[i].flag);
                return 2;
            }
        }
    }

    if (run_all) {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            if (test_entries[i].explicit_only) continue;
            test_run_entry(&test_entries[i]);
        }
    } else {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            if (selected[i]) test_run_entry(&test_entries[i]);
        }
    }

#ifndef DS4_NO_GPU
    test_close_engines();
#endif

    if (test_failures) {
        fprintf(stderr, "ds4 tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4 tests: ok");
    return 0;
}
