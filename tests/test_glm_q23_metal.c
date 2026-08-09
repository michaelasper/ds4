#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* These values are the private Metal quant-type IDs in ds4_metal.m. */
#define Q2_K_TYPE 10u
#define Q3_K_TYPE 11u

#define N_TOTAL_EXPERT 4u
#define N_EXPERT 3u
#define DIM 256u

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

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

typedef char q2_k_block_size_must_match[(sizeof(block_q2_K_test) == 84u) ? 1 : -1];
typedef char q3_k_block_size_must_match[(sizeof(block_q3_K_test) == 110u) ? 1 : -1];

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static void fill_q2_matrix(void *storage, uint32_t salt) {
    block_q2_K_test *matrix = (block_q2_K_test *)storage;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
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

static void fill_q3_matrix(void *storage, uint32_t salt) {
    block_q3_K_test *matrix = (block_q3_K_test *)storage;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
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
                quant_type == Q2_K_TYPE ? 2u : 3u);
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

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    int ok = x && selected && weights && mid && out;
    ok = ok && ds4_gpu_tensor_write(x, 0, x_values, x_bytes);
    ok = ok && ds4_gpu_tensor_write(
        selected, 0, selected_values, selected_bytes);
    ok = ok && ds4_gpu_tensor_write(weights, 0, weight_values, weights_bytes);

    const char *saved_env = getenv("DS4_METAL_DISABLE_Q23_EXACT_MULTIROW");
    char *saved_env_copy = saved_env ? strdup(saved_env) : NULL;
    if (ok && setenv("DS4_METAL_DISABLE_Q23_EXACT_MULTIROW", "1", 1) != 0) {
        ok = 0;
    }
    if (ok) {
        ok = ds4_gpu_glm_routed_moe_batch_decode_exact_q2_q3_tensor(
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
        ok = ds4_gpu_tensor_read(mid, 0, legacy_mid, mid_bytes) &&
             ds4_gpu_tensor_read(out, 0, legacy_out, out_bytes);
    }
    /* Do not let a missing multirow write inherit the legacy result and make
     * the bitwise comparison pass accidentally. */
    if (ok) {
        memset(multi_mid, 0xa5, (size_t)mid_bytes);
        memset(multi_out, 0x5a, (size_t)out_bytes);
        ok = ds4_gpu_tensor_write(mid, 0, multi_mid, mid_bytes) &&
             ds4_gpu_tensor_write(out, 0, multi_out, out_bytes);
    }
    /* Force the production multirow path for the second half even if the
     * caller had the diagnostic switch enabled in its environment. */
    unsetenv("DS4_METAL_DISABLE_Q23_EXACT_MULTIROW");

    if (ok) {
        ok = ds4_gpu_glm_routed_moe_batch_decode_exact_q2_q3_tensor(
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
        ok = ds4_gpu_tensor_read(mid, 0, multi_mid, mid_bytes) &&
             ds4_gpu_tensor_read(out, 0, multi_out, out_bytes);
    }

    if (saved_env_copy) {
        setenv("DS4_METAL_DISABLE_Q23_EXACT_MULTIROW", saved_env_copy, 1);
    } else {
        unsetenv("DS4_METAL_DISABLE_Q23_EXACT_MULTIROW");
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
                        quant_type == Q2_K_TYPE ? 2u : 3u,
                        n_tokens, token);
                ok = 0;
                break;
            }
        }
        if (ok && memcmp(legacy_out, multi_out, (size_t)out_bytes) != 0) {
            fprintf(stderr, "Q%u Metal A/B out mismatch rows=%u\n",
                    quant_type == Q2_K_TYPE ? 2u : 3u, n_tokens);
            ok = 0;
        }
    }
    if (ok) {
        fprintf(stderr, "Q%u Metal A/B bit-exact rows=%u stride=%u\n",
                quant_type == Q2_K_TYPE ? 2u : 3u,
                n_tokens, mid_token_stride);
    }

    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(out);
    free(x_values);
    free(selected_values);
    free(weight_values);
    free(legacy_mid);
    free(multi_mid);
    free(legacy_out);
    free(multi_out);
    return ok;
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

    int ok = ds4_gpu_init() && ds4_gpu_set_model_map(model, model_size);
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
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

    ds4_gpu_cleanup();
    free(model);
    return ok ? 0 : 1;
}
