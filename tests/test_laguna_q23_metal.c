#define _DARWIN_C_SOURCE

#include "lgn2_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* These values are the private Metal quant-type IDs in lgn2_metal.m. */
#define Q2_K_TYPE 10u
#define Q3_K_TYPE 11u
#define Q4_K_TYPE 12u

#define N_TOTAL_EXPERT 4u
#define N_EXPERT 3u
#define DIM 256u
#define Q4_PRODUCTION_TOTAL_EXPERT 256u
#define Q4_PRODUCTION_EXPERT 10u
#define Q4_PRODUCTION_TOKENS 4u
#define GROUPED_TOTAL_EXPERT 22u
#define GROUPED_EXPERT 2u
#define GROUPED_TOKENS 32u

bool lgn2_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

#ifdef LGN2_TEST_HOOKS
void lgn2_gpu_test_glm_grouped_moe_counters_reset(void);
int lgn2_gpu_test_glm_grouped_moe_counters(
        uint64_t *encoded_dispatches,
        uint64_t *completed_dispatches);
void lgn2_gpu_test_glm_exact_q4_counters_reset(void);
int lgn2_gpu_test_glm_exact_q4_counters(
        uint64_t *encoded_dispatches,
        uint64_t *completed_dispatches);
#endif

typedef struct {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
} block_q2_K_test;

typedef struct {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
} block_q3_K_test;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K_test;

typedef char q2_k_block_size_must_match[(sizeof(block_q2_K_test) == 84u) ? 1 : -1];
typedef char q3_k_block_size_must_match[(sizeof(block_q3_K_test) == 110u) ? 1 : -1];
typedef char q4_k_block_size_must_match[(sizeof(block_q4_K_test) == 144u) ? 1 : -1];

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static void fill_q2_matrix_n(void *storage,
                             uint32_t expert_count,
                             uint32_t salt) {
    block_q2_K_test *matrix = (block_q2_K_test *)storage;
    for (uint32_t expert = 0; expert < expert_count; expert++) {
        for (uint32_t row = 0; row < DIM; row++) {
            block_q2_K_test *block = matrix +
                (uint64_t)expert * DIM + row;
            for (uint32_t i = 0; i < 16u; i++) {
                block->scales[i] = (uint8_t)(0x11u +
                    ((salt + expert * 3u + row + i * 5u) & 0x7eu));
            }
            for (uint32_t i = 0; i < 64u; i++) {
                block->qs[i] = (uint8_t)(salt + expert * 11u +
                    row * 7u + i * 13u);
            }
            block->d = 0x3c00u;    /* 1.0 in IEEE FP16 */
            block->dmin = 0x3800u; /* 0.5 in IEEE FP16 */
        }
    }
}

static void fill_q2_matrix(void *storage, uint32_t salt) {
    fill_q2_matrix_n(storage, N_TOTAL_EXPERT, salt);
}

static void fill_q3_matrix_n(void *storage,
                             uint32_t expert_count,
                             uint32_t salt) {
    block_q3_K_test *matrix = (block_q3_K_test *)storage;
    for (uint32_t expert = 0; expert < expert_count; expert++) {
        for (uint32_t row = 0; row < DIM; row++) {
            block_q3_K_test *block = matrix +
                (uint64_t)expert * DIM + row;
            for (uint32_t i = 0; i < 32u; i++) {
                block->hmask[i] = (uint8_t)(salt + expert * 5u +
                    row * 3u + i * 7u);
            }
            for (uint32_t i = 0; i < 64u; i++) {
                block->qs[i] = (uint8_t)(salt + expert * 13u +
                    row * 9u + i * 5u);
            }
            for (uint32_t i = 0; i < 12u; i++) {
                block->scales[i] = (uint8_t)(salt + expert * 7u +
                    row * 11u + i * 3u);
            }
            block->d = 0x3c00u; /* 1.0 in IEEE FP16 */
        }
    }
}

static void fill_q3_matrix(void *storage, uint32_t salt) {
    fill_q3_matrix_n(storage, N_TOTAL_EXPERT, salt);
}

static void fill_q4_matrix_n(void *storage,
                             uint32_t expert_count,
                             uint32_t salt) {
    block_q4_K_test *matrix = (block_q4_K_test *)storage;
    for (uint32_t expert = 0; expert < expert_count; expert++) {
        for (uint32_t row = 0; row < DIM; row++) {
            block_q4_K_test *block = matrix +
                (uint64_t)expert * DIM + row;
            for (uint32_t i = 0; i < sizeof(block->scales); i++) {
                block->scales[i] = (uint8_t)(1u +
                    ((salt + expert * 3u + row + i * 5u) & 0x0fu));
            }
            for (uint32_t i = 0; i < sizeof(block->qs); i++) {
                block->qs[i] = (uint8_t)(salt + expert * 11u +
                    row * 7u + i * 13u);
            }
            block->d = 0x3c00u;    /* 1.0 in IEEE FP16 */
            block->dmin = 0x0000u; /* zero minimum for stable A/B values */
        }
    }
}

static unsigned quant_number(uint32_t quant_type) {
    return quant_type == Q2_K_TYPE ? 2u :
           quant_type == Q3_K_TYPE ? 3u : 4u;
}

static void fill_q2_grouped_matrix(void *storage,
                                   uint32_t expert_count,
                                   uint32_t high_expert,
                                   uint32_t variant) {
    block_q2_K_test *matrix = (block_q2_K_test *)storage;
    for (uint32_t expert = 0; expert < expert_count; expert++) {
        const uint8_t q_value = (uint8_t)(1u +
            (expert == high_expert ? variant : 0u));
        for (uint32_t row = 0; row < DIM; row++) {
            block_q2_K_test *block = matrix +
                (uint64_t)expert * DIM + row;
            memset(block->scales, 0x01, sizeof(block->scales));
            memset(block->qs, q_value, sizeof(block->qs));
            block->d = 0x3c00u;    /* 1.0 in IEEE FP16 */
            block->dmin = 0x0000u; /* zero minimum for stable A/B values */
        }
    }
}

static int run_case(const void *model,
                    uint64_t model_size,
                    uint32_t quant_type,
                    uint64_t gate_offset,
                    uint64_t up_offset,
                    uint64_t down_offset,
                    uint64_t expert_bytes,
                    uint64_t row_bytes,
                    uint32_t n_tokens) {
    const uint32_t per_token_mid = N_EXPERT * DIM;
    /* Exercise the caller-provided row stride rather than assuming a packed
     * mid tensor. The legacy path still binds each row independently. */
    const uint32_t mid_token_stride = per_token_mid + 16u;
    const uint64_t x_bytes = (uint64_t)n_tokens * DIM * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)n_tokens * N_EXPERT * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)n_tokens * N_EXPERT * sizeof(float);
    const uint64_t mid_bytes =
        ((uint64_t)(n_tokens - 1u) * mid_token_stride + per_token_mid) *
        sizeof(float);
    const uint64_t out_bytes =
        (uint64_t)n_tokens * DIM * sizeof(float);

    float *x_values = calloc((size_t)x_bytes, 1u);
    int32_t *selected_values = calloc((size_t)selected_bytes, 1u);
    float *weight_values = calloc((size_t)weights_bytes, 1u);
    uint8_t *legacy_mid = calloc((size_t)mid_bytes, 1u);
    uint8_t *multi_mid = calloc((size_t)mid_bytes, 1u);
    uint8_t *legacy_out = calloc((size_t)out_bytes, 1u);
    uint8_t *multi_out = calloc((size_t)out_bytes, 1u);
    if (!x_values || !selected_values || !weight_values ||
        !legacy_mid || !multi_mid || !legacy_out || !multi_out) {
        fprintf(stderr, "Q%u Metal A/B host allocation failed\n",
                quant_number(quant_type));
        free(x_values);
        free(selected_values);
        free(weight_values);
        free(legacy_mid);
        free(multi_mid);
        free(legacy_out);
        free(multi_out);
        return 0;
    }

    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t i = 0; i < DIM; i++) {
            x_values[(uint64_t)token * DIM + i] =
                (float)((int32_t)((i * 17u + token * 23u) % 97u) - 48) /
                32.0f;
        }
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            selected_values[(uint64_t)token * N_EXPERT + slot] =
                (int32_t)((token * 2u + slot * 3u + 1u) % N_TOTAL_EXPERT);
            weight_values[(uint64_t)token * N_EXPERT + slot] =
                0.125f + 0.03125f * (float)(token + slot);
        }
    }

    lgn2_gpu_tensor *x = lgn2_gpu_tensor_alloc(x_bytes);
    lgn2_gpu_tensor *selected = lgn2_gpu_tensor_alloc(selected_bytes);
    lgn2_gpu_tensor *weights = lgn2_gpu_tensor_alloc(weights_bytes);
    lgn2_gpu_tensor *mid = lgn2_gpu_tensor_alloc(mid_bytes);
    lgn2_gpu_tensor *out = lgn2_gpu_tensor_alloc(out_bytes);
    int ok = x && selected && weights && mid && out;
    ok = ok && lgn2_gpu_tensor_write(x, 0, x_values, x_bytes);
    ok = ok && lgn2_gpu_tensor_write(
        selected, 0, selected_values, selected_bytes);
    ok = ok && lgn2_gpu_tensor_write(weights, 0, weight_values, weights_bytes);

    const char *saved_env = getenv("LGN2_METAL_DISABLE_Q23_EXACT_MULTIROW");
    char *saved_env_copy = saved_env ? strdup(saved_env) : NULL;
    if (ok && setenv("LGN2_METAL_DISABLE_Q23_EXACT_MULTIROW", "1", 1) != 0) {
        ok = 0;
    }
    if (ok) {
        ok = lgn2_gpu_glm_routed_moe_batch_decode_exact_q2_q3_tensor(
            out, mid, model, model_size,
            gate_offset, up_offset, down_offset,
            quant_type, quant_type, quant_type,
            expert_bytes, row_bytes,
            expert_bytes, row_bytes,
            expert_bytes, row_bytes,
            DIM, DIM, DIM,
            selected, weights,
            N_TOTAL_EXPERT, N_EXPERT, 0u,
            x, n_tokens, mid_token_stride);
    }
    if (ok) {
        ok = lgn2_gpu_tensor_read(mid, 0, legacy_mid, mid_bytes) &&
             lgn2_gpu_tensor_read(out, 0, legacy_out, out_bytes);
    }
    if (ok && n_tokens == 1u) {
        /* The ordinary one-token helper must bind all three whole-model
         * ranges before dispatch. Compare its output with the exact Q2/Q3
         * verifier so a nil MTLBuffer cannot masquerade as success. */
        const char *saved_qmv = getenv("LGN2_METAL_LAGUNA_QMV_R1");
        char *saved_qmv_copy = saved_qmv ? strdup(saved_qmv) : NULL;
        lgn2_gpu_tensor *one_mid = lgn2_gpu_tensor_alloc(mid_bytes);
        lgn2_gpu_tensor *one_out = lgn2_gpu_tensor_alloc(out_bytes);
        int one_ok = !saved_qmv || saved_qmv_copy != NULL;
        if (one_ok) one_ok = unsetenv("LGN2_METAL_LAGUNA_QMV_R1") == 0;
        if (one_ok) {
            one_ok = one_mid && one_out &&
                lgn2_gpu_glm_routed_moe_one_tensor(
                    one_out, one_mid, model, model_size,
                    gate_offset, up_offset, down_offset,
                    quant_type, quant_type, quant_type,
                    expert_bytes, row_bytes,
                    expert_bytes, row_bytes,
                    expert_bytes, row_bytes,
                    DIM, DIM, DIM,
                    selected, weights,
                    N_TOTAL_EXPERT, N_EXPERT, 0u,
                    x, false);
        }
        if (one_ok) {
            one_ok = lgn2_gpu_tensor_read(
                one_mid, 0, multi_mid, mid_bytes) &&
                lgn2_gpu_tensor_read(one_out, 0, multi_out, out_bytes);
        }
        if (one_ok &&
            (memcmp(legacy_mid, multi_mid,
                    (size_t)per_token_mid * sizeof(float)) != 0 ||
             memcmp(legacy_out, multi_out, (size_t)out_bytes) != 0)) {
            fprintf(stderr, "Q%u Metal one-token mapped bind/output mismatch\n",
                    quant_number(quant_type));
            one_ok = 0;
        }
        if (saved_qmv_copy) {
            setenv("LGN2_METAL_LAGUNA_QMV_R1", saved_qmv_copy, 1);
        } else if (!saved_qmv) {
            unsetenv("LGN2_METAL_LAGUNA_QMV_R1");
        }
        free(saved_qmv_copy);
        lgn2_gpu_tensor_free(one_mid);
        lgn2_gpu_tensor_free(one_out);
        if (!one_ok) {
            fprintf(stderr, "Q%u Metal one-token mapped bind/output FAIL\n",
                    quant_number(quant_type));
            ok = 0;
        } else {
            fprintf(stderr, "Q%u Metal one-token mapped bind/output PASS\n",
                    quant_number(quant_type));
        }
    }
    /* Do not let a missing multirow write inherit the legacy result and make
     * the bitwise comparison pass accidentally. */
    if (ok) {
        memset(multi_mid, 0xa5, (size_t)mid_bytes);
        memset(multi_out, 0x5a, (size_t)out_bytes);
        ok = lgn2_gpu_tensor_write(mid, 0, multi_mid, mid_bytes) &&
             lgn2_gpu_tensor_write(out, 0, multi_out, out_bytes);
    }
    /* Force the production multirow path for the second half even if the
     * caller had the diagnostic switch enabled in its environment. */
    unsetenv("LGN2_METAL_DISABLE_Q23_EXACT_MULTIROW");

    if (ok) {
        ok = lgn2_gpu_glm_routed_moe_batch_decode_exact_q2_q3_tensor(
            out, mid, model, model_size,
            gate_offset, up_offset, down_offset,
            quant_type, quant_type, quant_type,
            expert_bytes, row_bytes,
            expert_bytes, row_bytes,
            expert_bytes, row_bytes,
            DIM, DIM, DIM,
            selected, weights,
            N_TOTAL_EXPERT, N_EXPERT, 0u,
            x, n_tokens, mid_token_stride);
    }
    if (ok) {
        ok = lgn2_gpu_tensor_read(mid, 0, multi_mid, mid_bytes) &&
             lgn2_gpu_tensor_read(out, 0, multi_out, out_bytes);
    }

    if (saved_env_copy) {
        setenv("LGN2_METAL_DISABLE_Q23_EXACT_MULTIROW", saved_env_copy, 1);
    } else {
        unsetenv("LGN2_METAL_DISABLE_Q23_EXACT_MULTIROW");
    }
    free(saved_env_copy);

    if (ok) {
        for (uint32_t token = 0; token < n_tokens; token++) {
            const uint64_t mid_offset =
                (uint64_t)token * mid_token_stride * sizeof(float);
            if (memcmp(legacy_mid + mid_offset, multi_mid + mid_offset,
                       (size_t)per_token_mid * sizeof(float)) != 0) {
                fprintf(stderr,
                        "Q%u Metal A/B mid mismatch rows=%u token=%u\n",
                        quant_number(quant_type),
                        n_tokens, token);
                ok = 0;
                break;
            }
        }
        if (ok && memcmp(legacy_out, multi_out, (size_t)out_bytes) != 0) {
            fprintf(stderr, "Q%u Metal A/B out mismatch rows=%u\n",
                    quant_number(quant_type), n_tokens);
            ok = 0;
        }
    }
    if (ok) {
        fprintf(stderr, "Q%u Metal A/B bit-exact rows=%u stride=%u\n",
                quant_number(quant_type),
                n_tokens, mid_token_stride);
    }

    lgn2_gpu_tensor_free(x);
    lgn2_gpu_tensor_free(selected);
    lgn2_gpu_tensor_free(weights);
    lgn2_gpu_tensor_free(mid);
    lgn2_gpu_tensor_free(out);
    free(x_values);
    free(selected_values);
    free(weight_values);
    free(legacy_mid);
    free(multi_mid);
    free(legacy_out);
    free(multi_out);
    return ok;
}

static int q4_values_finite_nonzero(const float *values, size_t count) {
    int nonzero = 0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(values[i])) return 0;
        if (fabsf(values[i]) > 1.0e-8f) nonzero = 1;
    }
    return nonzero;
}

static int q4_values_all_changed_from(const float *values,
                                      size_t count,
                                      float sentinel) {
    for (size_t i = 0; i < count; i++) {
        if (values[i] == sentinel) return 0;
    }
    return 1;
}

static float q4_max_scaled_diff(const float *left,
                                const float *right,
                                size_t count) {
    float max_scaled = 0.0f;
    for (size_t i = 0; i < count; i++) {
        const float scaled = fabsf(left[i] - right[i]) /
            (1.0f + fmaxf(fabsf(left[i]), fabsf(right[i])));
        if (scaled > max_scaled) max_scaled = scaled;
    }
    return max_scaled;
}

/* The exact-Q4 API must be independently checked against the ordinary
 * one-token route.  In particular, the Q4 exact API does not consult the
 * Q2/Q3 multirow diagnostic switch, so re-running that API for both A/B legs
 * would only prove that the same dispatch can write twice. */
static int run_q4_production_case(const void *model,
                                  uint64_t model_size,
                                  uint64_t gate_offset,
                                  uint64_t up_offset,
                                  uint64_t down_offset,
                                  uint64_t expert_bytes,
                                  uint64_t row_bytes) {
    const uint32_t n_tokens = Q4_PRODUCTION_TOKENS;
    const uint32_t per_token_mid = Q4_PRODUCTION_EXPERT * DIM;
    const uint32_t mid_token_stride = per_token_mid + 16u;
    const uint64_t x_count = (uint64_t)n_tokens * DIM;
    const uint64_t selected_count =
        (uint64_t)n_tokens * Q4_PRODUCTION_EXPERT;
    const uint64_t mid_count =
        (uint64_t)(n_tokens - 1u) * mid_token_stride + per_token_mid;
    const uint64_t out_count = (uint64_t)n_tokens * DIM;
    const uint64_t x_bytes = x_count * sizeof(float);
    const uint64_t selected_bytes = selected_count * sizeof(int32_t);
    const uint64_t weights_bytes = selected_count * sizeof(float);
    const uint64_t mid_bytes = mid_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);
    /* Keep the two legs visibly independent even when this test is built
     * with fast-math: finite sentinels make a missing write observable without
     * relying on NaN propagation or isfinite(). */
    const float batch_mid_sentinel = 12345.25f;
    const float batch_out_sentinel = -23456.5f;
    const float reference_mid_sentinel = -34567.75f;
    const float reference_out_sentinel = 45678.125f;
    float *x_values = calloc((size_t)x_count, sizeof(float));
    int32_t *selected_values = calloc((size_t)selected_count, sizeof(int32_t));
    float *weight_values = calloc((size_t)selected_count, sizeof(float));
    float *batch_mid_values = malloc((size_t)mid_bytes);
    float *batch_out_values = malloc((size_t)out_bytes);
    float *reference_mid_values = malloc((size_t)mid_bytes);
    float *reference_out_values = malloc((size_t)out_bytes);
    float *one_mid_values = malloc((size_t)per_token_mid * sizeof(float));
    float *one_out_values = malloc((size_t)DIM * sizeof(float));
    int ok = x_values && selected_values && weight_values &&
             batch_mid_values && batch_out_values &&
             reference_mid_values && reference_out_values &&
             one_mid_values && one_out_values;
    if (!ok) {
        fprintf(stderr, "Q4 production fixture host allocation failed\n");
        goto cleanup;
    }

    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t i = 0; i < DIM; i++) {
            x_values[(uint64_t)token * DIM + i] =
                (float)((int32_t)((i * 29u + token * 37u) % 127u) - 63) /
                40.0f;
        }
        selected_values[(uint64_t)token * Q4_PRODUCTION_EXPERT + 0u] = 0;
        selected_values[(uint64_t)token * Q4_PRODUCTION_EXPERT + 1u] =
            (int32_t)(Q4_PRODUCTION_TOTAL_EXPERT - 1u);
        for (uint32_t slot = 2; slot < Q4_PRODUCTION_EXPERT; slot++) {
            selected_values[(uint64_t)token * Q4_PRODUCTION_EXPERT + slot] =
                (int32_t)((token * 17u + slot * 11u) %
                          (Q4_PRODUCTION_TOTAL_EXPERT - 2u));
        }
        for (uint32_t slot = 0; slot < Q4_PRODUCTION_EXPERT; slot++) {
            weight_values[(uint64_t)token * Q4_PRODUCTION_EXPERT + slot] =
                0.03125f + 0.0078125f * (float)(token + slot);
        }
    }
    for (uint64_t i = 0; i < mid_count; i++) {
        batch_mid_values[i] = batch_mid_sentinel;
        reference_mid_values[i] = reference_mid_sentinel;
    }
    for (uint64_t i = 0; i < out_count; i++) {
        batch_out_values[i] = batch_out_sentinel;
        reference_out_values[i] = reference_out_sentinel;
    }

    /* Reinitialize against the production-shaped map so the exact leg starts
     * with no caller-owned command batch and therefore must take the owned
     * command-buffer path. */
    lgn2_gpu_cleanup();
    ok = lgn2_gpu_init() && lgn2_gpu_set_model_map(model, model_size);
    lgn2_gpu_set_quality(false);
    lgn2_gpu_tensor *batch_x = NULL;
    lgn2_gpu_tensor *batch_selected = NULL;
    lgn2_gpu_tensor *batch_weights = NULL;
    lgn2_gpu_tensor *batch_mid = NULL;
    lgn2_gpu_tensor *batch_out = NULL;
    lgn2_gpu_tensor *one_x = NULL;
    lgn2_gpu_tensor *one_selected = NULL;
    lgn2_gpu_tensor *one_weights = NULL;
    lgn2_gpu_tensor *one_mid = NULL;
    lgn2_gpu_tensor *one_out = NULL;
    if (!ok || lgn2_gpu_commands_active() != 0) {
        ok = 0;
        goto tensors_cleanup;
    }

    batch_x = lgn2_gpu_tensor_alloc(x_bytes);
    batch_selected = lgn2_gpu_tensor_alloc(selected_bytes);
    batch_weights = lgn2_gpu_tensor_alloc(weights_bytes);
    batch_mid = lgn2_gpu_tensor_alloc(mid_bytes);
    batch_out = lgn2_gpu_tensor_alloc(out_bytes);
    ok = batch_x && batch_selected && batch_weights && batch_mid && batch_out;
    ok = ok && lgn2_gpu_tensor_write(batch_x, 0, x_values, x_bytes);
    ok = ok && lgn2_gpu_tensor_write(
        batch_selected, 0, selected_values, selected_bytes);
    ok = ok && lgn2_gpu_tensor_write(
        batch_weights, 0, weight_values, weights_bytes);
    ok = ok && lgn2_gpu_tensor_write(batch_mid, 0, batch_mid_values, mid_bytes);
    ok = ok && lgn2_gpu_tensor_write(batch_out, 0, batch_out_values, out_bytes);
    if (ok && lgn2_gpu_commands_active() != 0) ok = 0;

#ifdef LGN2_TEST_HOOKS
    if (ok) lgn2_gpu_test_glm_exact_q4_counters_reset();
#endif
    if (ok) {
        ok = lgn2_gpu_glm_routed_moe_batch_decode_exact_q4_tensor(
            batch_out, batch_mid, model, model_size,
            gate_offset, up_offset, down_offset,
            Q4_K_TYPE, Q4_K_TYPE, Q4_K_TYPE,
            expert_bytes, row_bytes,
            expert_bytes, row_bytes,
            expert_bytes, row_bytes,
            DIM, DIM, DIM,
            batch_selected, batch_weights,
            Q4_PRODUCTION_TOTAL_EXPERT, Q4_PRODUCTION_EXPERT, 0u,
            batch_x, n_tokens, mid_token_stride);
    }
    if (ok && lgn2_gpu_commands_active() != 0) ok = 0;
#ifdef LGN2_TEST_HOOKS
    if (ok) {
        uint64_t encoded = 0;
        uint64_t completed = 0;
        ok = lgn2_gpu_test_glm_exact_q4_counters(&encoded, &completed) &&
             encoded == 1u && completed == 1u;
        if (!ok) {
            fprintf(stderr,
                    "Q4 production exact route evidence mismatch "
                    "encoded=%llu completed=%llu\n",
                    (unsigned long long)encoded,
                    (unsigned long long)completed);
        }
    }
#endif
    if (ok) {
        ok = lgn2_gpu_tensor_read(
            batch_mid, 0, batch_mid_values, mid_bytes) &&
             lgn2_gpu_tensor_read(batch_out, 0, batch_out_values, out_bytes);
    }
    int batch_mid_good = 0;
    int batch_mid_changed = 0;
    int batch_out_good = 0;
    int batch_out_changed = 0;
    if (ok) {
        batch_mid_good = 1;
        batch_mid_changed = 1;
        for (uint32_t token = 0; token < n_tokens; token++) {
            const float *token_mid = batch_mid_values +
                (uint64_t)token * mid_token_stride;
            if (!q4_values_finite_nonzero(token_mid, per_token_mid)) {
                batch_mid_good = 0;
            }
            if (!q4_values_all_changed_from(
                    token_mid, per_token_mid, batch_mid_sentinel)) {
                batch_mid_changed = 0;
            }
        }
        batch_out_good = q4_values_finite_nonzero(
            batch_out_values, (size_t)out_count);
        batch_out_changed = q4_values_all_changed_from(
            batch_out_values, (size_t)out_count, batch_out_sentinel);
    }
    if (ok &&
        (!batch_mid_good || !batch_mid_changed ||
         !batch_out_good || !batch_out_changed)) {
        fprintf(stderr,
                "Q4 production exact batch output was not changed and "
                "finite/nonzero\n");
        ok = 0;
    }

    const char *saved_qmv = getenv("LGN2_METAL_LAGUNA_QMV_R1");
    char *saved_qmv_copy = saved_qmv ? strdup(saved_qmv) : NULL;
    ok = ok && (!saved_qmv || saved_qmv_copy != NULL);
    if (ok) ok = unsetenv("LGN2_METAL_LAGUNA_QMV_R1") == 0;

    one_x = lgn2_gpu_tensor_alloc((uint64_t)DIM * sizeof(float));
    one_selected = lgn2_gpu_tensor_alloc(
        (uint64_t)Q4_PRODUCTION_EXPERT * sizeof(int32_t));
    one_weights = lgn2_gpu_tensor_alloc(
        (uint64_t)Q4_PRODUCTION_EXPERT * sizeof(float));
    one_mid = lgn2_gpu_tensor_alloc((uint64_t)per_token_mid * sizeof(float));
    one_out = lgn2_gpu_tensor_alloc((uint64_t)DIM * sizeof(float));
    ok = ok && one_x && one_selected && one_weights && one_mid && one_out;
    for (uint32_t token = 0; ok && token < n_tokens; token++) {
        for (uint32_t i = 0; i < per_token_mid; i++) {
            one_mid_values[i] = reference_mid_sentinel;
        }
        for (uint32_t i = 0; i < DIM; i++) {
            one_out_values[i] = reference_out_sentinel;
        }
        ok = lgn2_gpu_tensor_write(
            one_x, 0, x_values + (uint64_t)token * DIM, DIM * sizeof(float));
        ok = ok && lgn2_gpu_tensor_write(
            one_selected, 0,
            selected_values + (uint64_t)token * Q4_PRODUCTION_EXPERT,
            Q4_PRODUCTION_EXPERT * sizeof(int32_t));
        ok = ok && lgn2_gpu_tensor_write(
            one_weights, 0,
            weight_values + (uint64_t)token * Q4_PRODUCTION_EXPERT,
            Q4_PRODUCTION_EXPERT * sizeof(float));
        ok = ok && lgn2_gpu_tensor_write(
            one_mid, 0, one_mid_values,
            (uint64_t)per_token_mid * sizeof(float));
        ok = ok && lgn2_gpu_tensor_write(
            one_out, 0, one_out_values, DIM * sizeof(float));
        if (ok && lgn2_gpu_commands_active() != 0) ok = 0;
        if (ok) {
            ok = lgn2_gpu_glm_routed_moe_one_tensor(
                one_out, one_mid, model, model_size,
                gate_offset, up_offset, down_offset,
                Q4_K_TYPE, Q4_K_TYPE, Q4_K_TYPE,
                expert_bytes, row_bytes,
                expert_bytes, row_bytes,
                expert_bytes, row_bytes,
                DIM, DIM, DIM,
                one_selected, one_weights,
                Q4_PRODUCTION_TOTAL_EXPERT, Q4_PRODUCTION_EXPERT, 0u,
                one_x, false);
        }
        if (ok && lgn2_gpu_commands_active() != 0) ok = 0;
        if (ok) {
            ok = lgn2_gpu_tensor_read(
                one_mid, 0, one_mid_values,
                (uint64_t)per_token_mid * sizeof(float)) &&
                lgn2_gpu_tensor_read(one_out, 0, one_out_values,
                                    (uint64_t)DIM * sizeof(float));
        }
        if (ok &&
            (!q4_values_all_changed_from(
                 one_mid_values, per_token_mid, reference_mid_sentinel) ||
             !q4_values_all_changed_from(
                 one_out_values, DIM, reference_out_sentinel) ||
             !q4_values_finite_nonzero(one_mid_values, per_token_mid) ||
             !q4_values_finite_nonzero(one_out_values, DIM))) {
            fprintf(stderr,
                "Q4 production one-token reference output was not "
                "changed and finite/nonzero token=%u\n", token);
            ok = 0;
        }
        if (ok) {
            memcpy(reference_mid_values + (uint64_t)token * mid_token_stride,
                   one_mid_values, (size_t)per_token_mid * sizeof(float));
            memcpy(reference_out_values + (uint64_t)token * DIM,
                   one_out_values, (size_t)DIM * sizeof(float));
        }
    }
    if (saved_qmv_copy) {
        setenv("LGN2_METAL_LAGUNA_QMV_R1", saved_qmv_copy, 1);
    } else if (!saved_qmv) {
        unsetenv("LGN2_METAL_LAGUNA_QMV_R1");
    }
    free(saved_qmv_copy);

#ifdef LGN2_TEST_HOOKS
    if (ok) {
        uint64_t encoded = 0;
        uint64_t completed = 0;
        ok = lgn2_gpu_test_glm_exact_q4_counters(&encoded, &completed) &&
             encoded == 1u && completed == 1u;
    }
#endif
    if (ok) {
        float max_mid = 0.0f;
        float max_out = 0.0f;
        for (uint32_t token = 0; token < n_tokens; token++) {
            max_mid = fmaxf(max_mid,
                q4_max_scaled_diff(
                    batch_mid_values + (uint64_t)token * mid_token_stride,
                    reference_mid_values + (uint64_t)token * mid_token_stride,
                    per_token_mid));
            max_out = fmaxf(max_out,
                q4_max_scaled_diff(
                    batch_out_values + (uint64_t)token * DIM,
                    reference_out_values + (uint64_t)token * DIM,
                    DIM));
        }
        fprintf(stderr,
                "Q4 production exact-vs-one total=%u selected=%u "
                "ids=0,%u tokens=%u max_scaled_mid=%g max_scaled_out=%g\n",
                Q4_PRODUCTION_TOTAL_EXPERT, Q4_PRODUCTION_EXPERT,
                Q4_PRODUCTION_TOTAL_EXPERT - 1u, n_tokens,
                max_mid, max_out);
        ok = max_mid < 0.02f && max_out < 0.02f;
    }

tensors_cleanup:
    lgn2_gpu_tensor_free(one_out);
    lgn2_gpu_tensor_free(one_mid);
    lgn2_gpu_tensor_free(one_weights);
    lgn2_gpu_tensor_free(one_selected);
    lgn2_gpu_tensor_free(one_x);
    lgn2_gpu_tensor_free(batch_out);
    lgn2_gpu_tensor_free(batch_mid);
    lgn2_gpu_tensor_free(batch_weights);
    lgn2_gpu_tensor_free(batch_selected);
    lgn2_gpu_tensor_free(batch_x);
    lgn2_gpu_cleanup();
cleanup:
    free(one_out_values);
    free(one_mid_values);
    free(reference_out_values);
    free(reference_mid_values);
    free(batch_out_values);
    free(batch_mid_values);
    free(weight_values);
    free(selected_values);
    free(x_values);
    return ok;
}

static int run_grouped_dispatch(const void *model,
                                uint64_t model_size,
                                uint64_t gate_offset,
                                uint64_t up_offset,
                                uint64_t down_offset,
                                uint64_t expert_bytes,
                                uint64_t row_bytes,
                                const float *x_values,
                                const int32_t *selected_values,
                                const float *weight_values,
                                float *out_values) {
    const uint32_t mid_token_stride =
        GROUPED_EXPERT * DIM + 16u;
    const uint64_t x_bytes =
        (uint64_t)GROUPED_TOKENS * DIM * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)GROUPED_TOKENS * GROUPED_EXPERT * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)GROUPED_TOKENS * GROUPED_EXPERT * sizeof(float);
    const uint64_t mid_bytes =
        ((uint64_t)(GROUPED_TOKENS - 1u) * mid_token_stride +
         GROUPED_EXPERT * DIM) * sizeof(float);
    const uint64_t out_bytes =
        (uint64_t)GROUPED_TOKENS * DIM * sizeof(float);

    lgn2_gpu_tensor *x = lgn2_gpu_tensor_alloc(x_bytes);
    lgn2_gpu_tensor *selected = lgn2_gpu_tensor_alloc(selected_bytes);
    lgn2_gpu_tensor *weights = lgn2_gpu_tensor_alloc(weights_bytes);
    lgn2_gpu_tensor *mid = lgn2_gpu_tensor_alloc(mid_bytes);
    lgn2_gpu_tensor *out = lgn2_gpu_tensor_alloc(out_bytes);
    int ok = x && selected && weights && mid && out;
    if (ok) ok = lgn2_gpu_tensor_write(x, 0, x_values, x_bytes);
    if (ok) ok = lgn2_gpu_tensor_write(
        selected, 0, selected_values, selected_bytes);
    if (ok) ok = lgn2_gpu_tensor_write(weights, 0, weight_values, weights_bytes);
    if (ok) ok = lgn2_gpu_tensor_fill_f32(mid, 0.0f, mid_bytes / sizeof(float));
    if (ok) ok = lgn2_gpu_tensor_fill_f32(out, 0.0f, out_bytes / sizeof(float));
    if (ok) {
        ok = lgn2_gpu_glm_routed_moe_batch_tensor(
            out, mid, model, model_size,
            gate_offset, up_offset, down_offset,
            Q2_K_TYPE, Q2_K_TYPE, Q2_K_TYPE,
            expert_bytes, row_bytes,
            expert_bytes, row_bytes,
            expert_bytes, row_bytes,
            DIM, DIM, DIM,
            selected, weights,
            GROUPED_TOTAL_EXPERT, GROUPED_EXPERT, 0u,
            x, GROUPED_TOKENS, mid_token_stride, false);
    }
    /* A borrowed command-batch leg intentionally leaves the output pending
     * until its caller closes/submits the batch; the completion-evidence test
     * passes NULL here and reads no output before that boundary. */
    if (ok && out_values) {
        ok = lgn2_gpu_tensor_read(out, 0, out_values, out_bytes);
    }
    lgn2_gpu_tensor_free(x);
    lgn2_gpu_tensor_free(selected);
    lgn2_gpu_tensor_free(weights);
    lgn2_gpu_tensor_free(mid);
    lgn2_gpu_tensor_free(out);
    return ok;
}

/* Compiled grouped-MoE contract: the map must bind the complete 22-expert
 * tensor and preserve a valid high/tail ID (21) through route execution.  A
 * 96-token lifecycle threshold forces the existing row-loop batch path as an
 * oracle; a 32-token threshold then selects grouped mul_mm_id.  Grouped
 * router input is an upstream invariant: every selected ID is an integer in
 * [0, n_total_expert), so malformed IDs are not a promised zeroing behavior
 * of this grouped API and are deliberately not used as a success signal. */
static int run_grouped_full_map_case(const void *model,
                                     uint64_t model_size,
                                     uint64_t gate_offset,
                                     uint64_t up_offset,
                                     uint64_t down_offset,
                                     uint64_t expert_bytes,
                                     uint64_t row_bytes) {
    const uint64_t x_values = (uint64_t)GROUPED_TOKENS * DIM;
    const uint64_t route_values =
        (uint64_t)GROUPED_TOKENS * GROUPED_EXPERT;
    const uint64_t out_values = (uint64_t)GROUPED_TOKENS * DIM;
    float *x_values_host = calloc((size_t)x_values, sizeof(float));
    int32_t *mixed_ids = calloc((size_t)route_values, sizeof(int32_t));
    float *route_weights = calloc((size_t)route_values, sizeof(float));
    float *row_loop_out = calloc((size_t)out_values, sizeof(float));
    float *grouped_out = calloc((size_t)out_values, sizeof(float));
    const char *env_name = "LGN2_METAL_LAGUNA_GROUPED_MOE_MIN_TOKENS";
    const char *saved_env = getenv(env_name);
    char *saved_env_copy = saved_env ? strdup(saved_env) : NULL;
    int ok = x_values_host && mixed_ids && route_weights &&
             row_loop_out && grouped_out &&
             (!saved_env || saved_env_copy);
    if (!ok) goto cleanup;

    for (uint32_t token = 0; token < GROUPED_TOKENS; token++) {
        for (uint32_t i = 0; i < DIM; i++) {
            x_values_host[(uint64_t)token * DIM + i] =
                (float)((int32_t)((i * 19u + token * 31u) % 113u) - 56) /
                48.0f;
        }
        /* Both selected IDs are valid. Expert 21 is the high/tail end of the
         * full map, while expert 0 exercises the low end in the same batch. */
        mixed_ids[(uint64_t)token * GROUPED_EXPERT + 0u] = 0;
        mixed_ids[(uint64_t)token * GROUPED_EXPERT + 1u] =
            (int32_t)(GROUPED_TOTAL_EXPERT - 1u);
        route_weights[(uint64_t)token * GROUPED_EXPERT + 0u] = 0.375f;
        route_weights[(uint64_t)token * GROUPED_EXPERT + 1u] = 0.625f;
    }

    /* The cached threshold is lifecycle state, so each A/B leg gets a clean
     * Metal init just as production source/selector matrices do. */
    if (setenv(env_name, "96", 1) != 0) {
        ok = 0;
        goto cleanup;
    }
#ifdef LGN2_TEST_HOOKS
    lgn2_gpu_test_glm_grouped_moe_counters_reset();
#endif
    lgn2_gpu_cleanup();
    ok = lgn2_gpu_init() && lgn2_gpu_set_model_map(model, model_size);
    if (ok) {
        lgn2_gpu_set_quality(false);
        ok = run_grouped_dispatch(model, model_size,
                                  gate_offset, up_offset, down_offset,
                                  expert_bytes, row_bytes,
                                  x_values_host, mixed_ids, route_weights,
                                  row_loop_out);
    }
#ifdef LGN2_TEST_HOOKS
    if (ok) {
        uint64_t encoded = 0;
        uint64_t completed = 0;
        ok = lgn2_gpu_test_glm_grouped_moe_counters(
                 &encoded, &completed) && encoded == 0u && completed == 0u;
    }
#endif

    if (setenv(env_name, "32", 1) != 0) {
        ok = 0;
        goto cleanup;
    }
    lgn2_gpu_cleanup();
    if (ok) ok = lgn2_gpu_init() && lgn2_gpu_set_model_map(model, model_size);
    if (ok) {
        lgn2_gpu_set_quality(false);
        ok = run_grouped_dispatch(model, model_size,
                                  gate_offset, up_offset, down_offset,
                                  expert_bytes, row_bytes,
                                  x_values_host, mixed_ids, route_weights,
                                  grouped_out);
    }
#ifdef LGN2_TEST_HOOKS
    uint64_t grouped_encoded = 0;
    uint64_t grouped_completed = 0;
    if (ok) {
        ok = lgn2_gpu_test_glm_grouped_moe_counters(
                 &grouped_encoded, &grouped_completed) &&
             grouped_encoded == 1u && grouped_completed == 1u;
    }
#endif
    if (ok) {
        float max_rel = 0.0f;
        size_t nonfinite = 0;
        for (uint64_t i = 0; i < out_values; i++) {
            const float grouped = grouped_out[i];
            const float oracle = row_loop_out[i];
            if (!isfinite(grouped) || !isfinite(oracle)) {
                nonfinite++;
                continue;
            }
            const float rel = fabsf(grouped - oracle) /
                (1.0f + fmaxf(fabsf(oracle), fabsf(grouped)));
            if (rel > max_rel) max_rel = rel;
        }
        fprintf(stderr,
                "Q2 grouped full-map tokens=%u experts=%u high_id=%u "
                "max_scaled_diff=%g nonfinite=%zu\n",
                GROUPED_TOKENS, GROUPED_TOTAL_EXPERT,
                GROUPED_TOTAL_EXPERT - 1u, max_rel, nonfinite);
        ok = nonfinite == 0u && max_rel < 0.02f;
    }

cleanup:
    lgn2_gpu_cleanup();
    if (saved_env_copy) {
        setenv(env_name, saved_env_copy, 1);
    } else {
        unsetenv(env_name);
    }
    free(saved_env_copy);
    free(grouped_out);
    free(row_loop_out);
    free(route_weights);
    free(mixed_ids);
    free(x_values_host);
    return ok;
}

/* Completion evidence must follow the caller-owned command-buffer boundary,
 * not the grouped encoder's return.  Exercise the borrowed path directly:
 * encode is visible while the batch is open, completion is not, and only a
 * successful end promotes it.  Discard and an injected submitted-wait
 * failure must leave the encoded record unpromoted. */
static int run_grouped_borrowed_completion_case(
        const void *model,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t expert_bytes,
        uint64_t row_bytes) {
#ifndef LGN2_TEST_HOOKS
    (void)model;
    (void)model_size;
    (void)gate_offset;
    (void)up_offset;
    (void)down_offset;
    (void)expert_bytes;
    (void)row_bytes;
    return 1;
#else
    const char *env_name = "LGN2_METAL_LAGUNA_GROUPED_MOE_MIN_TOKENS";
    const char *saved_env = getenv(env_name);
    char *saved_env_copy = saved_env ? strdup(saved_env) : NULL;
    int ok = saved_env == NULL || saved_env_copy != NULL;
    if (!ok || setenv(env_name, "32", 1) != 0) {
        free(saved_env_copy);
        return 0;
    }

    const uint64_t x_count = (uint64_t)GROUPED_TOKENS * DIM;
    const uint64_t route_count =
        (uint64_t)GROUPED_TOKENS * GROUPED_EXPERT;
    float *x_values = calloc((size_t)x_count, sizeof(float));
    int32_t *selected_values = calloc((size_t)route_count, sizeof(int32_t));
    float *weight_values = calloc((size_t)route_count, sizeof(float));
    if (!x_values || !selected_values || !weight_values) {
        free(weight_values);
        free(selected_values);
        free(x_values);
        free(saved_env_copy);
        return 0;
    }
    for (uint32_t token = 0; token < GROUPED_TOKENS; token++) {
        for (uint32_t i = 0; i < DIM; i++) {
            x_values[(uint64_t)token * DIM + i] =
                (float)((int32_t)((i * 19u + token * 31u) % 113u) - 56) /
                48.0f;
        }
        selected_values[(uint64_t)token * GROUPED_EXPERT + 0u] = 0;
        selected_values[(uint64_t)token * GROUPED_EXPERT + 1u] =
            (int32_t)(GROUPED_TOTAL_EXPERT - 1u);
        weight_values[(uint64_t)token * GROUPED_EXPERT + 0u] = 0.375f;
        weight_values[(uint64_t)token * GROUPED_EXPERT + 1u] = 0.625f;
    }

    uint64_t encoded = 0;
    uint64_t completed = 0;

    /* A successful borrowed batch: completion remains zero until end. */
    lgn2_gpu_cleanup();
    ok = lgn2_gpu_init() && lgn2_gpu_set_model_map(model, model_size);
    lgn2_gpu_set_quality(false);
    lgn2_gpu_test_glm_grouped_moe_counters_reset();
    if (ok) ok = lgn2_gpu_begin_commands();
    if (ok) {
        ok = run_grouped_dispatch(
            model, model_size, gate_offset, up_offset, down_offset,
            expert_bytes, row_bytes, x_values, selected_values,
            weight_values, NULL);
    }
    if (ok) {
        ok = lgn2_gpu_test_glm_grouped_moe_counters(&encoded, &completed) &&
             encoded == 1u && completed == 0u;
    }
    if (ok) ok = lgn2_gpu_end_commands();
    if (ok) {
        ok = lgn2_gpu_test_glm_grouped_moe_counters(&encoded, &completed) &&
             encoded == 1u && completed == 1u;
    }

    /* A discarded borrowed batch must not promote its encoded record. */
    if (ok) {
        lgn2_gpu_cleanup();
        ok = lgn2_gpu_init() && lgn2_gpu_set_model_map(model, model_size);
        lgn2_gpu_set_quality(false);
        lgn2_gpu_test_glm_grouped_moe_counters_reset();
        if (ok) ok = lgn2_gpu_begin_commands();
        if (ok) {
            ok = run_grouped_dispatch(
                model, model_size, gate_offset, up_offset, down_offset,
                expert_bytes, row_bytes, x_values, selected_values,
                weight_values, NULL);
        }
        if (ok) {
            ok = lgn2_gpu_discard_commands();
            ok = ok && lgn2_gpu_test_glm_grouped_moe_counters(
                &encoded, &completed) && encoded == 1u && completed == 0u;
        }
    }

    /* A committed borrowed batch whose wait reports failure also stays
     * unpromoted, even though Metal has reached its terminal status. */
    if (ok) {
        lgn2_gpu_cleanup();
        ok = lgn2_gpu_init() && lgn2_gpu_set_model_map(model, model_size);
        lgn2_gpu_set_quality(false);
        lgn2_gpu_test_glm_grouped_moe_counters_reset();
        if (ok) ok = lgn2_gpu_begin_commands();
        if (ok) {
            ok = run_grouped_dispatch(
                model, model_size, gate_offset, up_offset, down_offset,
                expert_bytes, row_bytes, x_values, selected_values,
                weight_values, NULL);
        }
        if (ok) ok = lgn2_gpu_flush_commands();
        if (ok) {
            lgn2_gpu_test_inject_wait_submitted_failure();
            ok = lgn2_gpu_wait_submitted_commands() == 0;
        }
        if (ok) {
            ok = lgn2_gpu_test_glm_grouped_moe_counters(
                &encoded, &completed) && encoded == 1u && completed == 0u;
        }
    }

    if (lgn2_gpu_commands_active()) {
        (void)lgn2_gpu_discard_commands();
    } else {
        (void)lgn2_gpu_wait_submitted_commands();
    }
    lgn2_gpu_cleanup();
    if (saved_env_copy) {
        setenv(env_name, saved_env_copy, 1);
    } else {
        unsetenv(env_name);
    }
    free(saved_env_copy);
    free(weight_values);
    free(selected_values);
    free(x_values);
    if (!ok) {
        fprintf(stderr, "Q2 grouped borrowed completion evidence FAILED\n");
    } else {
        fprintf(stderr,
                "Q2 grouped borrowed completion evidence PASS "
                "(encode-before-end, end-promote, discard/failure-drop)\n");
    }
    return ok;
#endif
}

int main(void) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t q2_row_bytes = sizeof(block_q2_K_test);
    const uint64_t q3_row_bytes = sizeof(block_q3_K_test);
    const uint64_t q2_expert_bytes = DIM * q2_row_bytes;
    const uint64_t q3_expert_bytes = DIM * q3_row_bytes;
    const uint64_t q2_tensor_bytes = N_TOTAL_EXPERT * q2_expert_bytes;
    const uint64_t q3_tensor_bytes = N_TOTAL_EXPERT * q3_expert_bytes;
    const uint64_t q2_gate_offset = 0u;
    const uint64_t q2_up_offset = align_up(
        q2_gate_offset + q2_tensor_bytes, page);
    const uint64_t q2_down_offset = align_up(
        q2_up_offset + q2_tensor_bytes, page);
    const uint64_t q3_gate_offset = align_up(
        q2_down_offset + q2_tensor_bytes, page);
    const uint64_t q3_up_offset = align_up(
        q3_gate_offset + q3_tensor_bytes, page);
    const uint64_t q3_down_offset = align_up(
        q3_up_offset + q3_tensor_bytes, page);
    const uint64_t model_size = align_up(
        q3_down_offset + q3_tensor_bytes, page);

    void *model = NULL;
    if (posix_memalign(&model, (size_t)page, (size_t)model_size) != 0) {
        fprintf(stderr, "Q2/Q3 Metal A/B model allocation failed\n");
        return 1;
    }
    memset(model, 0, (size_t)model_size);
    fill_q2_matrix((uint8_t *)model + q2_gate_offset, 3u);
    fill_q2_matrix((uint8_t *)model + q2_up_offset, 17u);
    fill_q2_matrix((uint8_t *)model + q2_down_offset, 29u);
    fill_q3_matrix((uint8_t *)model + q3_gate_offset, 5u);
    fill_q3_matrix((uint8_t *)model + q3_up_offset, 19u);
    fill_q3_matrix((uint8_t *)model + q3_down_offset, 31u);

    int ok = lgn2_gpu_init() && lgn2_gpu_set_model_map(model, model_size);
    lgn2_gpu_set_quality(false);
    if (ok) {
        ok = run_case(model, model_size, Q2_K_TYPE,
                      q2_gate_offset, q2_up_offset, q2_down_offset,
                      q2_expert_bytes, q2_row_bytes, 1u);
    }
    if (ok) {
        ok = run_case(model, model_size, Q2_K_TYPE,
                      q2_gate_offset, q2_up_offset, q2_down_offset,
                      q2_expert_bytes, q2_row_bytes, 4u);
    }
    if (ok) {
        ok = run_case(model, model_size, Q3_K_TYPE,
                      q3_gate_offset, q3_up_offset, q3_down_offset,
                      q3_expert_bytes, q3_row_bytes, 1u);
    }
    if (ok) {
        ok = run_case(model, model_size, Q3_K_TYPE,
                      q3_gate_offset, q3_up_offset, q3_down_offset,
                      q3_expert_bytes, q3_row_bytes, 4u);
    }
    if (ok) {
        /* Use a full production-shaped Q4 map so the exact route exercises
         * both long-tail IDs (0 and 255), rather than a compact test map. */
        const uint64_t q4_row_bytes = sizeof(block_q4_K_test);
        const uint64_t q4_expert_bytes = DIM * q4_row_bytes;
        const uint64_t q4_tensor_bytes =
            Q4_PRODUCTION_TOTAL_EXPERT * q4_expert_bytes;
        const uint64_t q4_gate_offset = 0u;
        const uint64_t q4_up_offset = align_up(
            q4_gate_offset + q4_tensor_bytes, page);
        const uint64_t q4_down_offset = align_up(
            q4_up_offset + q4_tensor_bytes, page);
        const uint64_t q4_model_size = align_up(
            q4_down_offset + q4_tensor_bytes, page);
        void *q4_model = NULL;
        if (posix_memalign(&q4_model, (size_t)page,
                           (size_t)q4_model_size) != 0) {
            fprintf(stderr, "Q4 production model allocation failed\n");
            ok = 0;
        } else {
            memset(q4_model, 0, (size_t)q4_model_size);
            fill_q4_matrix_n((uint8_t *)q4_model + q4_gate_offset,
                             Q4_PRODUCTION_TOTAL_EXPERT, 7u);
            fill_q4_matrix_n((uint8_t *)q4_model + q4_up_offset,
                             Q4_PRODUCTION_TOTAL_EXPERT, 23u);
            fill_q4_matrix_n((uint8_t *)q4_model + q4_down_offset,
                             Q4_PRODUCTION_TOTAL_EXPERT, 37u);
            ok = run_q4_production_case(
                q4_model, q4_model_size,
                q4_gate_offset, q4_up_offset, q4_down_offset,
                q4_expert_bytes, q4_row_bytes);
            free(q4_model);
        }
    }

    if (ok) {
        /* A separate full-map allocation keeps the long-tail expert IDs out
         * of the compact four-expert exactness fixture above. */
        const uint64_t grouped_row_bytes = sizeof(block_q2_K_test);
        const uint64_t grouped_expert_bytes = DIM * grouped_row_bytes;
        const uint64_t grouped_tensor_bytes =
            GROUPED_TOTAL_EXPERT * grouped_expert_bytes;
        const uint64_t grouped_gate_offset = 0u;
        const uint64_t grouped_up_offset = align_up(
            grouped_gate_offset + grouped_tensor_bytes, page);
        const uint64_t grouped_down_offset = align_up(
            grouped_up_offset + grouped_tensor_bytes, page);
        const uint64_t grouped_model_size = align_up(
            grouped_down_offset + grouped_tensor_bytes, page);
        void *grouped_model = NULL;
        if (posix_memalign(&grouped_model, (size_t)page,
                           (size_t)grouped_model_size) != 0) {
            ok = 0;
        } else {
            memset(grouped_model, 0, (size_t)grouped_model_size);
            fill_q2_grouped_matrix(
                (uint8_t *)grouped_model + grouped_gate_offset,
                GROUPED_TOTAL_EXPERT, GROUPED_TOTAL_EXPERT - 1u, 1u);
            fill_q2_grouped_matrix(
                (uint8_t *)grouped_model + grouped_up_offset,
                GROUPED_TOTAL_EXPERT, GROUPED_TOTAL_EXPERT - 1u, 2u);
            fill_q2_grouped_matrix(
                (uint8_t *)grouped_model + grouped_down_offset,
                GROUPED_TOTAL_EXPERT, GROUPED_TOTAL_EXPERT - 1u, 3u);
            ok = run_grouped_full_map_case(
                grouped_model, grouped_model_size,
                grouped_gate_offset, grouped_up_offset, grouped_down_offset,
                grouped_expert_bytes, grouped_row_bytes);
            if (ok) {
                ok = run_grouped_borrowed_completion_case(
                    grouped_model, grouped_model_size,
                    grouped_gate_offset, grouped_up_offset, grouped_down_offset,
                    grouped_expert_bytes, grouped_row_bytes);
            }
            free(grouped_model);
        }
    }

    lgn2_gpu_cleanup();
    free(model);
    return ok ? 0 : 1;
}
