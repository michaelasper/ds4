/* =========================================================================
 * ds4.c - DeepSeek V4 inference engine.
 * =========================================================================
 *
 * This file is deliberately vertical: it owns GGUF loading, the fixed
 * DeepSeek V4 tensor layouts, CPU reference kernels, the whole-model Metal
 * graph driver, and tokenizer wiring.  Model shape selection is intentionally
 * narrow: validation accepts the known Flash and Pro layouts and fails early
 * for anything else.
 *
 * Loading is mmap based.  The loader parses only the GGUF header, metadata
 * table, and tensor directory.  Tensor data stays in the kernel page cache
 * until inference touches it, or until Metal wraps slices of the mapping as
 * no-copy MTLBuffers.
 */

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

#include "ds4.h"
#include "lgn.h"
#include "lgn_dflash.h"
#include "lgn_dflash_exec.h"
#include "lgn_dflash_graph.h"
#include "lgn_graph.h"
#include "lgn_model.h"

#ifdef DS4_TEST_HOOKS
/* Counts only the main model-open boundary.  The lifecycle test uses this
 * process-local seam to prove pre-model option rejection without relying on a
 * deliberately missing model path. */
static uint64_t g_ds4_test_engine_model_open_calls;
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DS4_NEG_INF (-1.0e30f)
#define DS4_POS_INF ( 1.0e30f)
static const char DS4_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/* DeepSeek recommends Think Max only with at least a 384K-token context window.
 * Below that size we keep ordinary thinking to avoid injecting a prompt that
 * asks for a reasoning budget the allocated context is not meant to hold. */
#define DS4_THINK_MAX_MIN_CONTEXT 393216u

/* =========================================================================
 * Model Shape Profiles.
 * =========================================================================
 *
 * The weight binder and metadata validator admit one Laguna S2.1 profile.
 * Arrays retain the historical maximum dimensions so tensor/layout capacity
 * and serialized interfaces remain unchanged during the transition.
 */

enum {
    DS4_MAX_LAYER            = 79,
    DS4_MAX_EMBD             = 7168,
    DS4_MAX_VOCAB            = 154880,
    DS4_MAX_HEAD             = 128,
    DS4_MAX_HEAD_KV          = 8,
    DS4_MAX_HEAD_DIM         = 576,
    DS4_MAX_VALUE_DIM        = 512,
    DS4_MAX_ROT              = 128,
    DS4_MAX_OUT_GROUP        = 16,
    DS4_MAX_LORA_Q           = 2048,
    DS4_MAX_LORA_O           = 1024,
    DS4_MAX_EXPERT           = 384,
    DS4_MAX_EXPERT_USED      = 10,
    DS4_MAX_EXPERT_SHARED    = 1,
    DS4_MAX_FF_EXP           = 3072,
    DS4_MAX_HASH_LAYER       = 3,
    DS4_MAX_SWA              = 512,
    DS4_MAX_INDEXER_HEAD     = 64,
    DS4_MAX_INDEXER_HEAD_DIM = 128,
    DS4_MAX_INDEXER_TOP_K    = 2048,
    DS4_MAX_HC               = 4,
    DS4_MAX_HC_SINKHORN_ITER = 20,
};

static ds4_shape g_ds4_shape = {
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
    .n_out_group = 0,
    .n_lora_q = 0,
    .n_lora_o = 0,
    .n_expert = 256,
    .n_expert_used = 10,
    .n_expert_shared = 1,
    .n_ff_exp = 1024,
    .n_ff_shared = 1024,
    .n_ff_dense = 12288,
    .n_hash_layer = 0,
    .n_swa = 512,
    .n_indexer_head = 0,
    .n_indexer_head_dim = 0,
    .n_indexer_top_k = 0,
    .n_hc = 0,
    .n_hc_sinkhorn_iter = 0,
    .n_leading_dense = 1,
    .n_rot_swa = 128,
    .rms_eps = 1.0e-6f,
    .hc_eps = 0.0f,
    .expert_weight_scale = 2.5f,
    .swiglu_clamp_exp = 0.0f,
    .rope_freq_base = 500000.0f,
    .rope_scale_factor = 32.0f,
    .rope_yarn_beta_fast = 32.0f,
    .rope_yarn_beta_slow = 1.0f,
    .rope_yarn_attn_factor = 1.0f,
    .rope_freq_base_swa = 10000.0f,
    .context_length = UINT64_C(262144),
    .rope_orig_ctx = UINT64_C(8192),
};

#define DS4_MODEL_SHAPE_NAME          (g_ds4_shape.name)
#define DS4_MODEL_FAMILY              (g_ds4_shape.family)
#define DS4_MODEL_VARIANT             (g_ds4_shape.variant)
#define DS4_N_LAYER                   (g_ds4_shape.n_layer)
#define DS4_N_EMBD                    (g_ds4_shape.n_embd)
#define DS4_N_VOCAB                   (g_ds4_shape.n_vocab)
#define DS4_N_HEAD                    (g_ds4_shape.n_head)
#define DS4_N_HEAD_KV                 (g_ds4_shape.n_head_kv)
#define DS4_N_HEAD_DIM                (g_ds4_shape.n_head_dim)
#define DS4_N_VALUE_DIM               (g_ds4_shape.n_value_dim)
#define DS4_N_ROT                     (g_ds4_shape.n_rot)
#define DS4_N_OUT_GROUP               (g_ds4_shape.n_out_group)
#define DS4_N_LORA_Q                  (g_ds4_shape.n_lora_q)
#define DS4_N_LORA_O                  (g_ds4_shape.n_lora_o)
#define DS4_N_EXPERT                  (g_ds4_shape.n_expert)
#define DS4_N_EXPERT_USED             (g_ds4_shape.n_expert_used)
#define DS4_N_EXPERT_SHARED           (g_ds4_shape.n_expert_shared)
#define DS4_N_FF_EXP                  (g_ds4_shape.n_ff_exp)
#define DS4_N_FF_SHARED               (g_ds4_shape.n_ff_shared)
#define DS4_N_FF_DENSE                (g_ds4_shape.n_ff_dense)
#define DS4_N_HASH_LAYER              (g_ds4_shape.n_hash_layer)
#define DS4_N_SWA                     (g_ds4_shape.n_swa)
#define DS4_N_INDEXER_HEAD            (g_ds4_shape.n_indexer_head)
#define DS4_N_INDEXER_HEAD_DIM        (g_ds4_shape.n_indexer_head_dim)
#define DS4_N_INDEXER_TOP_K           (g_ds4_shape.n_indexer_top_k)
#define DS4_N_HC                      (g_ds4_shape.n_hc)
#define DS4_N_HC_SINKHORN_ITER        (g_ds4_shape.n_hc_sinkhorn_iter)
#define DS4_N_NEXTN_PREDICT           (g_ds4_shape.n_nextn_predict)
#define DS4_N_LEADING_DENSE           (g_ds4_shape.n_leading_dense)
#define DS4_N_KV_LORA                 (g_ds4_shape.n_kv_lora)
#define DS4_N_KEY_MLA                 (g_ds4_shape.n_key_mla)
#define DS4_N_VALUE_MLA               (g_ds4_shape.n_value_mla)
#define DS4_N_ROT_SWA                 (g_ds4_shape.n_rot_swa)
#define DS4_RMS_EPS                   (g_ds4_shape.rms_eps)
#define DS4_HC_EPS                    (g_ds4_shape.hc_eps)
#define DS4_EXPERT_WEIGHT_SCALE       (g_ds4_shape.expert_weight_scale)
#define DS4_SWIGLU_CLAMP_EXP          (g_ds4_shape.swiglu_clamp_exp)
#define DS4_ROPE_FREQ_BASE            (g_ds4_shape.rope_freq_base)
#define DS4_ROPE_SCALE_FACTOR         (g_ds4_shape.rope_scale_factor)
#define DS4_ROPE_YARN_BETA_FAST       (g_ds4_shape.rope_yarn_beta_fast)
#define DS4_ROPE_YARN_BETA_SLOW       (g_ds4_shape.rope_yarn_beta_slow)
#define DS4_ROPE_YARN_ATTN_FACTOR     (g_ds4_shape.rope_yarn_attn_factor)
#define DS4_ROPE_FREQ_BASE_SWA        (g_ds4_shape.rope_freq_base_swa)
#define DS4_COMPRESS_ROPE_FREQ_BASE   (g_ds4_shape.compress_rope_freq_base)
#define DS4_CONTEXT_LENGTH            (g_ds4_shape.context_length)
#define DS4_ROPE_ORIG_CTX             (g_ds4_shape.rope_orig_ctx)

/* Temporary source-compatible alias while Laguna shape ownership moves to
 * lgn_model.c.  Execution code can retain the established fixed-shape reads;
 * the immutable profile itself no longer lives in this orchestration unit. */
#define DS4_SHAPE_LAGUNA_S21           (*lgn_model_shape())

static int g_ds4_lock_fd = -1;

#if defined(__GNUC__) || defined(__clang__)
#define DS4_MAYBE_UNUSED __attribute__((unused))
#else
#define DS4_MAYBE_UNUSED
#endif

static const char DS4_RUNTIME_NAME[] DS4_MAYBE_UNUSED = "metal";

/* =========================================================================
 * GGUF Quant Block Formats.
 * =========================================================================
 *
 * These layouts and IQ2 tables match the GGUF quantized tensor format,
 * reduced to only the formats ds4.c currently reads or sizes:
 *   - Q2_K/Q3_K routed experts
 *   - Q4_K routed experts in the high-memory variant
 *   - Q5_K/Q6_K GLM routed experts
 *   - IQ2_XXS routed gate/up experts
 *   - MXFP4 routed experts preserved from native checkpoints
 *   - Q8_K temporary activation blocks for dot products
 */
#define QK_K 256
#define QK_MXFP4 32

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    uint8_t  hmask[QK_K / 8];
    uint8_t  qs[QK_K / 4];
    uint8_t  scales[12];
    uint16_t d;
} block_q3_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[QK_K / 8];
    uint8_t  qs[QK_K / 2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    uint16_t d;
} block_q6_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;

typedef struct {
    uint8_t e;
    uint8_t qs[QK_MXFP4 / 2];
} block_mxfp4;

#define DS4_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]
DS4_STATIC_ASSERT(ds4_block_q2_k_size, sizeof(block_q2_K) == 84);
DS4_STATIC_ASSERT(ds4_block_q3_k_size, sizeof(block_q3_K) == 110);
DS4_STATIC_ASSERT(ds4_block_q4_k_size, sizeof(block_q4_K) == 144);
DS4_STATIC_ASSERT(ds4_block_q5_k_size, sizeof(block_q5_K) == 176);
DS4_STATIC_ASSERT(ds4_block_q6_k_size, sizeof(block_q6_K) == 210);
DS4_STATIC_ASSERT(ds4_block_q8_k_size, sizeof(block_q8_K) == 292);
DS4_STATIC_ASSERT(ds4_block_iq2_xxs_size, sizeof(block_iq2_xxs) == 66);
DS4_STATIC_ASSERT(ds4_block_mxfp4_size, sizeof(block_mxfp4) == 17);

static const uint8_t kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static int8_t iq2xxs_signed_grid[256][128][8];
static int8_t iq2xxs_signs[128][8];
static pthread_once_t iq2xxs_signed_grid_once = PTHREAD_ONCE_INIT;

static void iq2xxs_signed_grid_init(void) {
    for (uint32_t s = 0; s < 128; s++) {
        const uint8_t signs = ksigns_iq2xs[s];
        for (uint32_t j = 0; j < 8; j++) {
            iq2xxs_signs[s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -1 : 1);
        }
    }

    for (uint32_t g = 0; g < 256; g++) {
        const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + g);
        for (uint32_t s = 0; s < 128; s++) {
            const uint8_t signs = ksigns_iq2xs[s];
            for (uint32_t j = 0; j < 8; j++) {
                const int v = (int)grid[j];
                iq2xxs_signed_grid[g][s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -v : v);
            }
        }
    }
}

static inline DS4_MAYBE_UNUSED int32_t dot_iq2_pair_16(const int8_t *grid0, const int8_t *grid1, const int8_t *q8) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int8x16_t gv = vcombine_s8(vld1_s8(grid0), vld1_s8(grid1));
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), gv, vld1q_s8(q8));
    return vaddvq_s32(acc);
#elif defined(__ARM_NEON)
    const int8x16_t gv = vcombine_s8(vld1_s8(grid0), vld1_s8(grid1));
    const int8x16_t qv = vld1q_s8(q8);
    const int16x8_t p0 = vmull_s8(vget_low_s8(gv), vget_low_s8(qv));
    const int16x8_t p1 = vmull_s8(vget_high_s8(gv), vget_high_s8(qv));
    return vaddvq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#else
    int32_t sum = 0;
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid0[i] * (int32_t)q8[i];
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid1[i] * (int32_t)q8[8 + i];
    return sum;
#endif
}

static inline DS4_MAYBE_UNUSED int32_t dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t packed = vld1q_u8(q2);
    uint8x16_t shifted;
    switch (shift) {
    case 0: shifted = packed; break;
    case 2: shifted = vshrq_n_u8(packed, 2); break;
    case 4: shifted = vshrq_n_u8(packed, 4); break;
    default: shifted = vshrq_n_u8(packed, 6); break;
    }
    const uint8x16_t vals_u = vandq_u8(shifted, vdupq_n_u8(3));
    const int8x16_t vals = vreinterpretq_s8_u8(vals_u);
    const int8x16_t q8v = vld1q_s8(q8);
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), q8v, vals);
    return vaddvq_s32(acc);
#elif defined(__ARM_NEON)
    uint8_t vals_tmp[16];
    for (uint32_t i = 0; i < 16; i++) vals_tmp[i] = (q2[i] >> shift) & 3;
    const int8x16_t vals = vreinterpretq_s8_u8(vld1q_u8(vals_tmp));
    const int8x16_t q8v = vld1q_s8(q8);
    const int16x8_t p0 = vmull_s8(vget_low_s8(q8v), vget_low_s8(vals));
    const int16x8_t p1 = vmull_s8(vget_high_s8(q8v), vget_high_s8(vals));
    const int32x4_t s0 = vpaddlq_s16(p0);
    const int32x4_t s1 = vpaddlq_s16(p1);
    return vaddvq_s32(vaddq_s32(s0, s1));
#else
    int32_t sum = 0;
    for (uint32_t i = 0; i < 16; i++) sum += (int32_t)q8[i] * (int32_t)((q2[i] >> shift) & 3);
    return sum;
#endif
}

/* =========================================================================
 * Shared Helpers, Allocation Guards, Threads, and Cursor Reads.
 * =========================================================================
 *
 * This section holds process-wide utilities used by all later stages:
 * fatal-error helpers, allocation wrappers, the persistent CPU worker pool,
 * and the small byte cursor used to parse GGUF metadata.
 */

#define DS4_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */
#define DS4_MAX_DIMS   8

typedef ds4_tokens token_vec;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} ds4_cursor;

static void ds4_die(const char *msg) {
    fprintf(stderr, "ds4: %s\n", msg);
    exit(1);
}

#ifndef DS4_NO_GPU
static uint32_t ds4_layer_head_count(uint32_t il) {
    if (il >= DS4_N_LAYER) ds4_die("layer index is outside the loaded model layout");
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA) {
        const uint32_t n = lgn_model_layer_head_count(il);
        if (n == 0) ds4_die("Laguna layer topology is not initialized");
        return n;
    }
    return DS4_N_HEAD;
}
#endif

#ifndef DS4_NO_GPU
static bool ds4_laguna_layer_is_swa(uint32_t il) {
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_LAGUNA) return false;
    (void)ds4_layer_head_count(il);
    return lgn_model_layer_is_swa(il);
}
#endif

static void ds4_die_errno(const char *what, const char *path) {
    fprintf(stderr, "ds4: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static bool ds4_streq(ds4_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}



static bool ds4_str_eq(ds4_str a, ds4_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

static uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static bool g_alloc_guard_enabled;
static const char *g_alloc_guard_phase;

static void ds4_alloc_guard_check(const char *op, size_t size) {
    if (!g_alloc_guard_enabled) return;
    fprintf(stderr,
            "ds4: internal allocation during %s: %s(%zu). "
            "guarded execution requires preallocated buffers.\n",
            g_alloc_guard_phase ? g_alloc_guard_phase : "guarded phase",
            op,
            size);
    exit(1);
}

static void *xcalloc(size_t n, size_t size) {
    ds4_alloc_guard_check("calloc", n * size);
    void *p = calloc(n, size);
    if (!p) ds4_die("out of memory");
    return p;
}

static void *xmalloc(size_t size) {
    ds4_alloc_guard_check("malloc", size);
    void *p = malloc(size);
    if (!p) ds4_die("out of memory");
    return p;
}

static char *ds4_strdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static void *xrealloc(void *ptr, size_t size) {
    ds4_alloc_guard_check("realloc", size);
    void *p = realloc(ptr, size);
    if (!p) ds4_die("out of memory");
    return p;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static const char *ds4_log_color_code(ds4_log_type type) {
    switch (type) {
    case DS4_LOG_PREFILL:
    case DS4_LOG_TIMING:
        return "\x1b[36m";
    case DS4_LOG_GENERATION:
    case DS4_LOG_OK:
        return "\x1b[32m";
    case DS4_LOG_KVCACHE:
        return "\x1b[33m";
    case DS4_LOG_TOOL:
        return "\x1b[90m";
    case DS4_LOG_WARNING:
        return "\x1b[38;5;208m";
    case DS4_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}

bool ds4_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

static void ds4_vlog(FILE *fp, ds4_log_type type, const char *fmt, va_list ap) {
    const bool colorize = type != DS4_LOG_DEFAULT && ds4_log_is_tty(fp);
    if (colorize) fputs(ds4_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
}

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ds4_vlog(fp, type, fmt, ap);
    va_end(ap);
}




typedef void (*ds4_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);

#define DS4_MAX_THREADS 32

typedef struct {
    pthread_t threads[DS4_MAX_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_cond;
    pthread_cond_t done_cond;
    uint32_t n_threads;
    uint32_t n_workers;
    uint32_t generation;
    uint32_t done;
    bool initialized;
    bool shutdown;
    ds4_parallel_fn fn;
    void *ctx;
    uint64_t n_rows;
} ds4_thread_pool;

static ds4_thread_pool g_pool;
static __thread int g_parallel_depth;
static uint32_t g_requested_threads;

static void *ds4_worker_main(void *arg) {
    const uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&g_pool.mutex);
        while (seen_generation == g_pool.generation && !g_pool.shutdown) {
            pthread_cond_wait(&g_pool.work_cond, &g_pool.mutex);
        }
        if (g_pool.shutdown) {
            pthread_mutex_unlock(&g_pool.mutex);
            return NULL;
        }

        seen_generation = g_pool.generation;
        ds4_parallel_fn fn = g_pool.fn;
        void *ctx = g_pool.ctx;
        const uint64_t n_rows = g_pool.n_rows;
        const uint32_t n_threads = g_pool.n_threads;
        pthread_mutex_unlock(&g_pool.mutex);

        const uint64_t rows_per_thread = (n_rows + n_threads - 1) / n_threads;
        const uint64_t row0 = (uint64_t)tid * rows_per_thread;
        uint64_t row1 = row0 + rows_per_thread;
        if (row1 > n_rows) row1 = n_rows;
        if (row0 < row1) {
            g_parallel_depth++;
            fn(ctx, row0, row1);
            g_parallel_depth--;
        }

        pthread_mutex_lock(&g_pool.mutex);
        g_pool.done++;
        if (g_pool.done == g_pool.n_workers) {
            pthread_cond_signal(&g_pool.done_cond);
        }
        pthread_mutex_unlock(&g_pool.mutex);
    }
}

/* Create the persistent CPU worker pool.  Decode reuses these threads instead
 * of creating pthreads in the token loop. */
static void ds4_threads_init(void) {
    if (g_pool.initialized) return;

    pthread_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init);

    uint32_t n_threads = 12;
    const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (online_cpus > 0) {
        n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
    }

    const char *env = getenv("DS4_THREADS");
    if (env && env[0]) {
        long v = strtol(env, NULL, 10);
        if (v > 0) n_threads = (uint32_t)v;
    }
    if (g_requested_threads > 0) n_threads = g_requested_threads;
    if (n_threads > DS4_MAX_THREADS) n_threads = DS4_MAX_THREADS;
    if (n_threads == 0) n_threads = 1;

    pthread_mutex_init(&g_pool.mutex, NULL);
    pthread_cond_init(&g_pool.work_cond, NULL);
    pthread_cond_init(&g_pool.done_cond, NULL);
    g_pool.n_threads = n_threads;
    g_pool.n_workers = n_threads > 0 ? n_threads - 1 : 0;
    g_pool.generation = 0;
    g_pool.done = 0;
    g_pool.shutdown = false;
    g_pool.initialized = true;

    for (uint32_t i = 1; i < n_threads; i++) {
        if (pthread_create(&g_pool.threads[i], NULL, ds4_worker_main, (void *)(uintptr_t)i) != 0) {
            ds4_die("failed to create worker thread");
        }
    }
}

static void ds4_threads_shutdown(void) {
    if (!g_pool.initialized) return;

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.shutdown = true;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);
    pthread_mutex_unlock(&g_pool.mutex);

    for (uint32_t i = 1; i < g_pool.n_threads; i++) {
        pthread_join(g_pool.threads[i], NULL);
    }

    pthread_cond_destroy(&g_pool.done_cond);
    pthread_cond_destroy(&g_pool.work_cond);
    pthread_mutex_destroy(&g_pool.mutex);
    memset(&g_pool, 0, sizeof(g_pool));
}

/* Run a row-parallel CPU kernel, falling back to serial execution for small
 * jobs or nested calls where spawning more work would only add latency. */
static void ds4_parallel_for_min_rows(uint64_t n_rows, ds4_parallel_fn fn, void *ctx, uint64_t min_parallel_rows) {
    ds4_threads_init();

    if (g_parallel_depth > 0 || g_pool.n_threads <= 1 || n_rows < min_parallel_rows) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.n_rows = n_rows;
    g_pool.done = 0;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);

    const uint64_t rows_per_thread = (n_rows + g_pool.n_threads - 1) / g_pool.n_threads;
    uint64_t main_row1 = rows_per_thread;
    if (main_row1 > n_rows) main_row1 = n_rows;
    pthread_mutex_unlock(&g_pool.mutex);

    if (main_row1 > 0) {
        g_parallel_depth++;
        fn(ctx, 0, main_row1);
        g_parallel_depth--;
    }

    pthread_mutex_lock(&g_pool.mutex);
    while (g_pool.done < g_pool.n_workers) {
        pthread_cond_wait(&g_pool.done_cond, &g_pool.mutex);
    }
    pthread_mutex_unlock(&g_pool.mutex);
}

static void ds4_lgn_dflash_parallel_for(void *parallel_ctx,
                                        uint64_t n_rows,
                                        lgn_dflash_range_fn fn,
                                        void *ctx,
                                        uint64_t min_parallel_rows) {
    (void)parallel_ctx;
    ds4_parallel_for_min_rows(n_rows, fn, ctx, min_parallel_rows);
}

static void ds4_parallel_for(uint64_t n_rows, ds4_parallel_fn fn, void *ctx) {
    ds4_parallel_for_min_rows(n_rows, fn, ctx, 512);
}

static void cursor_error(ds4_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(ds4_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(ds4_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(ds4_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(ds4_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(ds4_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(ds4_cursor *c, ds4_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

/* =========================================================================
 * GGUF Parsing and Model Mapping.
 * =========================================================================
 *
 * The loader maps the model once, records metadata/tensor descriptors, and
 * leaves tensor bytes in place.  Inference code accesses weights by adding
 * tensor offsets to the mapping instead of copying the GGUF into private
 * structures.
 */

enum {
    GGUF_VALUE_UINT8   = LGN_GGUF_VALUE_UINT8,
    GGUF_VALUE_INT8    = LGN_GGUF_VALUE_INT8,
    GGUF_VALUE_UINT16  = LGN_GGUF_VALUE_UINT16,
    GGUF_VALUE_INT16   = LGN_GGUF_VALUE_INT16,
    GGUF_VALUE_UINT32  = LGN_GGUF_VALUE_UINT32,
    GGUF_VALUE_INT32   = LGN_GGUF_VALUE_INT32,
    GGUF_VALUE_FLOAT32 = LGN_GGUF_VALUE_FLOAT32,
    GGUF_VALUE_BOOL    = LGN_GGUF_VALUE_BOOL,
    GGUF_VALUE_STRING  = LGN_GGUF_VALUE_STRING,
    GGUF_VALUE_ARRAY   = LGN_GGUF_VALUE_ARRAY,
    GGUF_VALUE_UINT64  = LGN_GGUF_VALUE_UINT64,
    GGUF_VALUE_INT64   = LGN_GGUF_VALUE_INT64,
    GGUF_VALUE_FLOAT64 = LGN_GGUF_VALUE_FLOAT64,
};

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

static const gguf_type_info gguf_types[] = {
    [0]  = {"f32",      1,   4},
    [1]  = {"f16",      1,   2},
    [2]  = {"q4_0",    32,  18},
    [3]  = {"q4_1",    32,  20},
    [6]  = {"q5_0",    32,  22},
    [7]  = {"q5_1",    32,  24},
    [8]  = {"q8_0",    32,  34},
    [9]  = {"q8_1",    32,  40},
    [10] = {"q2_k",   256,  84},
    [11] = {"q3_k",   256, 110},
    [12] = {"q4_k",   256, 144},
    [13] = {"q5_k",   256, 176},
    [14] = {"q6_k",   256, 210},
    [15] = {"q8_k",   256, 292},
    [16] = {"iq2_xxs",256,  66},
    [17] = {"iq2_xs", 256,  74},
    [18] = {"iq3_xxs",256,  98},
    [19] = {"iq1_s",  256, 110},
    [20] = {"iq4_nl", 256,  50},
    [21] = {"iq3_s",  256, 110},
    [22] = {"iq2_s",  256,  82},
    [23] = {"iq4_xs", 256, 136},
    [24] = {"i8",       1,   1},
    [25] = {"i16",      1,   2},
    [26] = {"i32",      1,   4},
    [27] = {"i64",      1,   8},
    [28] = {"f64",      1,   8},
    [29] = {"iq1_m",  256,  56},
    [30] = {"bf16",     1,   2},
    [39] = {"mxfp4",   32,  17},
};

enum {
    DS4_TENSOR_F32      = LGN_TENSOR_F32,
    DS4_TENSOR_F16      = LGN_TENSOR_F16,
    DS4_TENSOR_Q4_0     = LGN_TENSOR_Q4_0,
    DS4_TENSOR_Q8_0     = LGN_TENSOR_Q8_0,
    DS4_TENSOR_Q2_K     = LGN_TENSOR_Q2_K,
    DS4_TENSOR_Q3_K     = LGN_TENSOR_Q3_K,
    DS4_TENSOR_Q4_K     = LGN_TENSOR_Q4_K,
    DS4_TENSOR_Q5_K     = LGN_TENSOR_Q5_K,
    DS4_TENSOR_Q6_K     = LGN_TENSOR_Q6_K,
    DS4_TENSOR_Q8_K     = LGN_TENSOR_Q8_K,
    DS4_TENSOR_IQ2_XXS  = LGN_TENSOR_IQ2_XXS,
    DS4_TENSOR_I32      = LGN_TENSOR_I32,
    DS4_TENSOR_BF16     = LGN_TENSOR_BF16,
    DS4_TENSOR_MXFP4    = LGN_TENSOR_MXFP4,
};

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(ds4_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);

    if (type == GGUF_VALUE_STRING) {
        ds4_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;

        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;

        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

static const gguf_type_info *tensor_type(uint32_t type) {
    uint32_t n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}

static const char *tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0) return false;
    uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static ds4_cursor cursor_at(const ds4_model *m, uint64_t pos) {
    ds4_cursor c = {
        .base = m->map,
        .size = m->size,
        .pos = pos,
        .error = {0},
    };
    return c;
}

static ds4_kv *model_find_kv(const ds4_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (ds4_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

static bool model_get_string(const ds4_model *m, const char *key, ds4_str *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

static bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

static bool model_get_token_id(const ds4_model *m, const char *key, int *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;

    ds4_cursor c = cursor_at(m, kv->value_pos);
    switch (kv->type) {
    case GGUF_VALUE_UINT32: {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v) || v > (uint32_t)INT_MAX) return false;
        *out = (int)v;
        return true;
    }
    case GGUF_VALUE_INT32: {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v)) || v < 0) return false;
        *out = (int)v;
        return true;
    }
    case GGUF_VALUE_UINT64: {
        uint64_t v = 0;
        if (!cursor_u64(&c, &v) || v > (uint64_t)INT_MAX) return false;
        *out = (int)v;
        return true;
    }
    case GGUF_VALUE_INT64: {
        int64_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v)) || v < 0 || v > (int64_t)INT_MAX) return false;
        *out = (int)v;
        return true;
    }
    default:
        return false;
    }
}

static bool model_get_u64_compat(const ds4_model *m, const char *key, uint64_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) {
        return cursor_u64(&c, out);
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} ds4_array_ref;

static bool model_get_array(const ds4_model *m, const char *key, ds4_array_ref *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;

    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}

static void model_close(ds4_model *m) {
    if (!m) return;
    free(m->kv);
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

static void model_prefetch_cpu_mapping(const ds4_model *m) {
    if (!m || !m->map || m->size == 0) return;

    /*
     * CPU generation touches expert weights according to router decisions, so a
     * long decode can fault in model pages that the prompt never touched. On
     * current Darwin kernels we have seen those late file-backed faults trigger
     * an OS-level VM panic in map-count accounting. This hint does not copy or
     * pin the GGUF; it just asks the kernel to start bringing the read-only
     * mapping into the page cache before token generation reaches it.
     */
#if defined(POSIX_MADV_WILLNEED)
    const int rc = posix_madvise((void *)m->map, (size_t)m->size, POSIX_MADV_WILLNEED);
    if (rc != 0) {
        ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: POSIX_MADV_WILLNEED failed for CPU model mapping: %s\n",
                strerror(rc));
    }
#else
    (void)m;
#endif
}

/* Read the GGUF metadata table.  Values stay in the mmap; we store offsets so
 * later validation can decode only the keys it needs. */
static void parse_metadata(ds4_model *m, ds4_cursor *c) {
    /* n_kv comes from the header. Every entry consumes at least one byte in the
     * file, so a count larger than the bytes remaining cannot be real; reject it
     * before calloc so a tiny file can't request an enormous allocation. */
    if (m->n_kv > c->size - c->pos) ds4_die("GGUF metadata count exceeds file size");
    m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    if (!m->kv) ds4_die("out of memory while allocating metadata table");

    m->alignment = 32;

    for (uint64_t i = 0; i < m->n_kv; i++) {
        ds4_kv *kv = &m->kv[i];

        if (!cursor_string(c, &kv->key)) ds4_die(c->error);
        if (!cursor_u32(c, &kv->type)) ds4_die(c->error);

        kv->value_pos = c->pos;

        if (ds4_streq(kv->key, "general.alignment") &&
            kv->type == GGUF_VALUE_UINT32)
        {
            ds4_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t alignment;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                m->alignment = alignment;
            }
        }

        if (!skip_value(c, kv->type, 0)) ds4_die(c->error);
    }
}

/* Read the tensor directory and convert relative GGUF offsets to absolute
 * mmap offsets.  Tensor bytes are still never copied here. */
static void parse_tensors(ds4_model *m, ds4_cursor *c) {
    /* As in parse_metadata: each tensor directory entry needs at least one byte
     * in the file, so reject a count larger than the bytes remaining before the
     * allocation. */
    if (m->n_tensors > c->size - c->pos) ds4_die("GGUF tensor count exceeds file size");
    m->tensors = calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    if (!m->tensors) ds4_die("out of memory while allocating tensor table");

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];

        if (!cursor_string(c, &t->name)) ds4_die(c->error);
        if (!cursor_u32(c, &t->ndim)) ds4_die(c->error);
        if (t->ndim == 0 || t->ndim > DS4_MAX_DIMS) {
            ds4_die("tensor has an unsupported number of dimensions");
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) ds4_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                ds4_die("tensor element count overflow");
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) ds4_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) ds4_die(c->error);

        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: tensor %.*s has unsupported GGUF type %u\n",
                (int)t->name.len, t->name.ptr, t->type);
        }
    }

    m->tensor_data_pos = align_up(c->pos, m->alignment);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) {
            ds4_die("tensor offset overflow");
        }
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 &&
            (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset))
        {
            ds4_die("tensor points outside GGUF file");
        }
        if (t->bytes > m->max_tensor_bytes) {
            m->max_tensor_bytes = t->bytes;
        }
    }
}

/* Open and map the GGUF once. Metal needs a shared mapping for no-copy
 * MTLBuffers; tokenizer/host-only callers use a private read-only mapping.
 * They pass prefetch_cpu=false so inspecting tokens never walks the huge tensor
 * payload. */
static void model_open(ds4_model *m, const char *path, bool metal_mapping,
                       bool prefetch_cpu) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) ds4_die_errno("cannot open model", path);

    struct stat st;
    if (fstat(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
    if (st.st_size < 32) ds4_die("model file is too small to be GGUF");

    /* Metal wraps slices of this mapping as no-copy MTLBuffers, so the Metal
     * path keeps the file-backed shared mapping. Host-only readers use a private
     * read-only mapping to avoid inheriting Metal's VM policy. This also avoids
     * the Darwin VM map-count path observed with very large shared mappings. */
    const int mmap_flags = metal_mapping ? MAP_SHARED : MAP_PRIVATE;
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, mmap_flags, fd, 0);
    if (map == MAP_FAILED) ds4_die_errno("cannot mmap model", path);

    m->fd = fd;
    m->map = map;
    m->size = (uint64_t)st.st_size;

    ds4_cursor c = cursor_at(m, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic)) ds4_die(c.error);
    if (magic != DS4_GGUF_MAGIC) ds4_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) ds4_die(c.error);

    if (m->version != 3) ds4_die("only GGUF v3 is supported");

    parse_metadata(m, &c);
    parse_tensors(m, &c);

    if (!metal_mapping && prefetch_cpu) model_prefetch_cpu_mapping(m);
}

static void print_size(FILE *out, uint64_t bytes) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    fprintf(out, "%.2f GiB", (double)bytes / gib);
}

typedef enum {
    DS4_SUPPORT_NONE = 0,
    DS4_SUPPORT_DFLASH,
} ds4_support_kind;

static void model_summary_to(const ds4_model *m, FILE *out) {
    if (!m || !out) return;

    ds4_str name = {0};
    ds4_str arch = {0};
    uint32_t layers = 0;
    uint64_t ctx_train = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_swa = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint64_t tensor_bytes = 0;
    uint64_t params = 0;

    model_get_string(m, "general.name", &name);
    model_get_string(m, "general.architecture", &arch);
    lgn_model_summary_fields laguna = {0};
    if (lgn_model_is_laguna(m, NULL)) {
        /* The production target has already passed config validation before
         * its summary is printed.  DFlash support models take the metadata
         * branch below and must not borrow the target profile. */
        lgn_model_get_validated_summary(&laguna);
        layers = laguna.n_layer;
        ctx_train = laguna.context_length;
        n_head = laguna.n_head;
        n_head_kv = laguna.n_head_kv;
        head_dim = laguna.n_head_dim;
        n_swa = laguna.n_swa;
        n_expert = laguna.n_expert;
        n_expert_used = laguna.n_expert_used;
    } else if (ds4_streq(arch, "dflash")) {
        /* DFlash support models are summarized from their stable metadata;
         * they are not target Laguna profiles and must not borrow its shape. */
        model_get_u32(m, "dflash.block_count", &layers);
        model_get_u64_compat(m, "dflash.context_length", &ctx_train);
        model_get_u32(m, "dflash.attention.head_count", &n_head);
        model_get_u32(m, "dflash.attention.head_count_kv", &n_head_kv);
        model_get_u32(m, "dflash.attention.key_length", &head_dim);
        model_get_u32(m, "dflash.attention.sliding_window", &n_swa);
    }

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        tensor_bytes += m->tensors[i].bytes;
        params += m->tensors[i].elements;
    }

    fprintf(out, "model: %.*s\n", (int)name.len, name.ptr);
    fprintf(out, "arch:  %.*s\n", (int)arch.len, arch.ptr);
    fprintf(out, "gguf:  v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n",
        m->version, m->n_kv, m->n_tensors);
    if (layers) fprintf(out, "layers: %u\n", layers);
    if (ctx_train) fprintf(out, "train context: %" PRIu64 "\n", ctx_train);
    if (n_head || n_head_kv || head_dim || n_swa) {
        fprintf(out, "attention: heads=%u kv_heads=%u head_dim=%u swa=%u\n",
               n_head, n_head_kv, head_dim, n_swa);
    }
    if (n_expert || n_expert_used) {
        fprintf(out, "experts: count=%u used=%u\n", n_expert, n_expert_used);
    }
    fprintf(out, "file size: ");
    print_size(out, m->size);
    fprintf(out, "\n");
    fprintf(out, "tensor bytes described by GGUF: ");
    print_size(out, tensor_bytes);
    fprintf(out, "\n");
    fprintf(out, "logical parameters: %.2f B\n", (double)params / 1000000000.0);

    fprintf(out, "tensor types:\n");
    for (uint32_t type = 0; type < sizeof(gguf_types)/sizeof(gguf_types[0]); type++) {
        uint64_t count = 0;
        uint64_t bytes = 0;
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            if (m->tensors[i].type == type) {
                count++;
                bytes += m->tensors[i].bytes;
            }
        }
        if (count != 0) {
            fprintf(out, "  %-8s %5" PRIu64 " tensors, ", tensor_type_name(type), count);
            print_size(out, bytes);
            fprintf(out, "\n");
        }
    }

}

static void model_summary(const ds4_model *m) {
    model_summary_to(m, stdout);
}

#ifdef DS4_TEST_HOOKS
bool ds4_test_model_summary(const struct ds4_model *model, FILE *out) {
    if (!model || !out) return false;
    model_summary_to(model, out);
    return ferror(out) == 0;
}
#endif


static const char *support_kind_name(ds4_support_kind kind) {
    switch (kind) {
    case DS4_SUPPORT_DFLASH:     return "DFlash";
    case DS4_SUPPORT_NONE:       return "none";
    }
    return "unknown";
}

static ds4_support_kind support_model_detect(
        const ds4_model *m,
        uint32_t        *stages_out) {
    if (stages_out) *stages_out = 0;
    if (!m) return DS4_SUPPORT_NONE;

    ds4_str arch = {0};
    if (model_get_string(m, "general.architecture", &arch) &&
        ds4_streq(arch, "dflash")) {
        return DS4_SUPPORT_DFLASH;
    }

    /* Keep the detector fail-closed so a non-DFlash support file cannot be
     * loaded into an inert engine field. */
    return DS4_SUPPORT_NONE;
}

/* Return the in-place tensor payload inside the mapped GGUF. */
static const void *tensor_data(const ds4_model *m, const ds4_tensor *t) {
    return m->map + t->abs_offset;
}

/* Optional startup pass that touches tensor pages before timing generation. */
static void model_warm_weights(const ds4_model *m) {
    const uint64_t start = m->tensor_data_pos;
    const uint64_t end = m->size;
    if (start >= end) return;

    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint8_t *p = m->map;
    volatile uint64_t checksum = 0;
    const double t0 = now_sec();

    fprintf(stderr, "ds4: warming mapped tensor pages: %.2f GiB\n",
            (double)(end - start) / (1024.0 * 1024.0 * 1024.0));

#if defined(POSIX_MADV_WILLNEED)
    (void)posix_madvise((void *)(p + start), (size_t)(end - start), POSIX_MADV_WILLNEED);
#endif

    for (uint64_t off = start; off < end; off += page) {
        checksum += p[off];
    }
    checksum += p[end - 1];

    const double t1 = now_sec();
    fprintf(stderr, "ds4: warmed tensor pages in %.3fs (checksum=%llu)\n",
            t1 - t0, (unsigned long long)checksum);
}

/* =========================================================================
 * Scalar Conversion and Quantized Tensor Kernels.
 * =========================================================================
 *
 * These functions provide scalar conversion helpers for host utilities and
 * Metal diagnostics.  They implement only the tensor formats present in the
 * DeepSeek V4 Flash GGUF: F16, F32, Q8_0, Q2_K, IQ2_XXS, and Q8_K activation
 * blocks used for expert dot products.
 */

static inline float f16_to_f32(uint16_t h) {
#if defined(__ARM_NEON)
    const float16x4_t hv = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(hv), 0);
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

static inline uint16_t f32_to_f16(float f) {
#if defined(__ARM_NEON)
    const float32x4_t fv = vdupq_n_f32(f);
    const float16x4_t hv = vcvt_f16_f32(fv);
    return vget_lane_u16(vreinterpret_u16_f16(hv), 0);
#else
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

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



/* DeepSeek V4 stores the non-RoPE part of compressed KV through an E4M3-style
 * round trip.  Keeping this in the CPU reference makes cache values comparable
 * to the Metal graph's compressed-cache behavior. */





/* The official DeepSeek V4 graph rotates indexer activations with a 128-wide
 * Hadamard transform and immediately runs the FP4 activation-simulation
 * round trip. This applies to both indexer Q and the indexer compressor KV;
 * without it, the top-k compressed-row selection is not the model's graph. */


/* Quantize a float activation into Q8_K blocks so GGUF Q2_K/IQ2_XXS expert
 * kernels can reuse the same activation for many expert rows. */

static void ds4_vec_dot_q2_K_q8_K(int n, float *s, const block_q2_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m3 = vdupq_n_u8(0x03);
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * f16_to_f32(x[i].d);
        const float dmin = -y[i].d * f16_to_f32(x[i].dmin);

        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
        uint8_t scale_lanes[16];
        vst1q_u8(scale_lanes, scales);

        const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
        const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
        const int16x8x2_t mins16 = {{
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))),
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins))),
        }};
        const int32x4_t s0 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16.val[0]), vget_low_s16(q8sums.val[0])),
            vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
        const int32x4_t s1 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16.val[1]), vget_low_s16(q8sums.val[1])),
            vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
        sum += dmin * (float)vaddvq_s32(vaddq_s32(s0, s1));

        int isum = 0;
        int is = 0;
        for (int j = 0; j < QK_K / 128; j++) {
            const uint8x16x2_t q2bits = vld1q_u8_x2(q2);
            q2 += 32;

#define DS4_Q2_DOT_NOSHIFT(scale_index) do {                                           \
                const int8x16x2_t q8bytes = vld1q_s8_x2(q8);                           \
                q8 += 32;                                                              \
                const int8x16_t q2lo = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));\
                const int8x16_t q2hi = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));\
                isum += vaddvq_s32(vdotq_s32(zero, q2lo, q8bytes.val[0])) *            \
                        scale_lanes[is + (scale_index)];                               \
                isum += vaddvq_s32(vdotq_s32(zero, q2hi, q8bytes.val[1])) *            \
                        scale_lanes[is + 1 + (scale_index)];                           \
            } while (0)

#define DS4_Q2_DOT_SHIFT(shift, scale_index) do {                                      \
                const int8x16x2_t q8bytes = vld1q_s8_x2(q8);                           \
                q8 += 32;                                                              \
                const int8x16_t q2lo = vreinterpretq_s8_u8(                            \
                    vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3));                 \
                const int8x16_t q2hi = vreinterpretq_s8_u8(                            \
                    vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3));                 \
                isum += vaddvq_s32(vdotq_s32(zero, q2lo, q8bytes.val[0])) *            \
                        scale_lanes[is + (scale_index)];                               \
                isum += vaddvq_s32(vdotq_s32(zero, q2hi, q8bytes.val[1])) *            \
                        scale_lanes[is + 1 + (scale_index)];                           \
            } while (0)

            DS4_Q2_DOT_NOSHIFT(0);
            DS4_Q2_DOT_SHIFT(2, 2);
            DS4_Q2_DOT_SHIFT(4, 4);
            DS4_Q2_DOT_SHIFT(6, 6);
            is += 8;

#undef DS4_Q2_DOT_NOSHIFT
#undef DS4_Q2_DOT_SHIFT
        }

        sum += d * (float)isum;
    }

    *s = sum;
#else
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; j++) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * f16_to_f32(x[i].d);
        const float dmin = y[i].d * f16_to_f32(x[i].dmin);

        int isum = 0;
        int is = 0;
        for (int k = 0; k < QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                int isuml = dot_q2_16(q2, q8, shift);
                isum += d * isuml;

                d = sc[is++] & 0x0f;
                isuml = dot_q2_16(q2 + 16, q8 + 16, shift);
                isum += d * isuml;

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    *s = sumf;
#endif
}






static void ds4_vec_dot_q8_K_pair_q8_K(
        int n, float *s0, float *s1,
        const block_q8_K *x0, const block_q8_K *x1,
        const block_q8_K *y) {
    const int nb = n / QK_K;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    for (int i = 0; i < nb; i++) {
        int32x4_t isum0 = vdupq_n_s32(0);
        int32x4_t isum1 = vdupq_n_s32(0);
        for (int j = 0; j < QK_K; j += 16) {
            const int8x16_t yv = vld1q_s8(y[i].qs + j);
            isum0 = vdotq_s32(isum0, vld1q_s8(x0[i].qs + j), yv);
            isum1 = vdotq_s32(isum1, vld1q_s8(x1[i].qs + j), yv);
        }
        sum0 += x0[i].d * y[i].d * (float)vaddvq_s32(isum0);
        sum1 += x1[i].d * y[i].d * (float)vaddvq_s32(isum1);
    }
#else
    for (int i = 0; i < nb; i++) {
        int isum0 = 0;
        int isum1 = 0;
        for (int j = 0; j < QK_K; j++) {
            const int yv = (int)y[i].qs[j];
            isum0 += (int)x0[i].qs[j] * yv;
            isum1 += (int)x1[i].qs[j] * yv;
        }
        sum0 += x0[i].d * y[i].d * (float)isum0;
        sum1 += x1[i].d * y[i].d * (float)isum1;
    }
#endif
    *s0 = sum0;
    *s1 = sum1;
}

static DS4_MAYBE_UNUSED void ds4_vec_dot_iq2_xxs_q8_K(int n, float *s, const block_iq2_xxs *x, const block_q8_K *y) {
    const int nb = n / QK_K;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        float sumf1 = 0.0f;
        float sumf2 = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;

            uint32_t aux32[4];
            memcpy(aux32, q2, sizeof(aux32));
            q2 += 8;
            const uint8_t *aux8 = (const uint8_t *)aux32;

            int8x16_t q2u0 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[0])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[1])));
            int8x16_t q2u1 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[2])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[3])));
            int8x16_t q2u2 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[8])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[9])));
            int8x16_t q2u3 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[10])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[11])));

            const int8x16_t q2s0 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[1] >>  0) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[1] >>  7) & 127]));
            const int8x16_t q2s1 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[1] >> 14) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[1] >> 21) & 127]));
            const int8x16_t q2s2 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[3] >>  0) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[3] >>  7) & 127]));
            const int8x16_t q2s3 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[3] >> 14) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[3] >> 21) & 127]));

            q2u0 = vmulq_s8(q2u0, q2s0);
            q2u1 = vmulq_s8(q2u1, q2s1);
            q2u2 = vmulq_s8(q2u2, q2s2);
            q2u3 = vmulq_s8(q2u3, q2s3);

            const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u0, q8b.val[0]), q2u1, q8b.val[1]);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u2, q8b.val[2]), q2u3, q8b.val[3]);

            sumf1 += (float)vaddvq_s32(p1) * (0.5f + (float)(aux32[1] >> 28));
            sumf2 += (float)vaddvq_s32(p2) * (0.5f + (float)(aux32[3] >> 28));
        }

        sumf += d * (sumf1 + sumf2);
    }

    *s = 0.25f * sumf;
#else
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;

            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127;
                const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127;
                sumi += dot_iq2_pair_16(iq2xxs_signed_grid[aux8[l]][sign_idx0],
                                        iq2xxs_signed_grid[aux8[l + 1]][sign_idx1],
                                        q8);
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    *s = 0.125f * sumf;
#endif
}


/* =========================================================================
 * Fixed Weight Binding and Model Validation.
 * =========================================================================
 *
 * The GGUF tensor directory is converted into a DS4-specific pointer table.
 * After this section, the rest of the program addresses tensors by semantic
 * fields such as layer->attn_q_a or layer->ffn_gate_exps rather than by string
 * lookup.  Shape validation is intentionally strict.
 */

static DS4_MAYBE_UNUSED uint64_t routed_expert_block_bytes(uint32_t type) {
    switch (type) {
    case DS4_TENSOR_Q8_0:    return 34;
    case DS4_TENSOR_IQ2_XXS: return sizeof(block_iq2_xxs);
    case DS4_TENSOR_Q2_K:    return sizeof(block_q2_K);
    case DS4_TENSOR_Q3_K:    return sizeof(block_q3_K);
    case DS4_TENSOR_Q4_K:    return sizeof(block_q4_K);
    case DS4_TENSOR_Q5_K:    return sizeof(block_q5_K);
    case DS4_TENSOR_Q6_K:    return sizeof(block_q6_K);
    case DS4_TENSOR_MXFP4:   return sizeof(block_mxfp4);
    default:                 ds4_die("unsupported routed expert tensor type");
    }
    return 0;
}

static DS4_MAYBE_UNUSED uint64_t routed_expert_row_bytes(const ds4_tensor *t) {
    const gguf_type_info *info = tensor_type(t->type);
    if (!info || info->block_elems == 0) ds4_die("unsupported routed expert tensor type");
    if ((t->dim[0] % info->block_elems) != 0) ds4_die("routed expert row is not quant block aligned");
    return (t->dim[0] / info->block_elems) * routed_expert_block_bytes(t->type);
}

static uint64_t ds4_add_sat_u64(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

static double ds4_bytes_to_gib(uint64_t bytes) {
    return (double)bytes / 1073741824.0;
}

static bool weights_have_output_head(const ds4_weights *w) {
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA) {
        return lgn_weights_have_output_head(w);
    }
    return w &&
           w->output_hc_base &&
           w->output_hc_fn &&
           w->output_hc_scale &&
           w->output_norm &&
           w->output;
}

static void weights_validate_laguna_layout(
        const ds4_weights *w,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output) {
    lgn_weights_validate_layout(w,
                                layer_start,
                                layer_end,
                                require_token_embd,
                                require_output);
}

static void weights_validate_layout(
        const ds4_weights *w,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output) {
    weights_validate_laguna_layout(w,
                                   layer_start,
                                   layer_end,
                                   require_token_embd,
                                   require_output);
}

static void config_validate_laguna_model(const ds4_model *m) {
    g_ds4_shape = *lgn_model_shape();
    lgn_model_validate_config(m);
}

/* Architecture admission is deliberately side-effect free.  Keep this
 * predicate ahead of every family validator: those validators select global
 * shape state, and a non-Laguna GGUF must never reach the legacy branches. */
static void config_require_laguna_architecture(const ds4_model *m) {
    lgn_model_require_laguna_architecture(m);
}

static void config_validate_model(const ds4_model *m) {
    /* Fail closed before any legacy model-family selection or shape mutation. */
    config_require_laguna_architecture(m);
    config_validate_laguna_model(m);
}

/* Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers. */
static void weights_bind(
        ds4_weights     *w,
        const ds4_model *m) {
    lgn_weights_bind(w, m);
    weights_validate_layout(w, 0, DS4_N_LAYER - 1u, true, true);
}

static void weights_free(ds4_weights *w) {
    memset(w, 0, sizeof(*w));
}

/* Load one token embedding row and expand it to float activations. */

/* RMSNorm without a learned scale, used by hyper-connection control vectors. */

/* Standard DS4 RMSNorm with learned per-channel scale. */

/* Normalize each attention head independently after Q projection. */

typedef struct {
    float *out;
    const uint16_t *data;
    const float *x;
    uint64_t in_dim;
} matvec_f16_ctx;



/* Dense F16 matvec for small control projections such as HC and router heads. */


typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t row0;
    uint64_t blocks;
} matvec_q8_0_ctx;

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *data0;
    const uint8_t *data1;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
} matvec_q8_0_pair_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
    uint64_t rank;
} matvec_q8_0_grouped_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t n_groups;
    uint64_t group_dim;
    uint64_t blocks;
    uint64_t rank;
} matmul_q8_0_grouped_batch_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
} matmul_q8_0_batch_ctx;

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *data0;
    const uint8_t *data1;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
} matmul_q8_0_pair_batch_ctx;

typedef struct {
    const float *x;
    int8_t *xq;
    float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
} quantize_q8_0_batch_ctx;

static inline int32_t dot_i8_32(const int8_t *a, const int8_t *b, uint64_t n) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (n == 32) {
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, vld1q_s8(a),      vld1q_s8(b));
        acc = vdotq_s32(acc, vld1q_s8(a + 16), vld1q_s8(b + 16));
        return vaddvq_s32(acc);
    }
#endif
    int32_t sum = 0;
    for (uint64_t i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

static inline float dot_q8_0_row(
        const uint8_t *row,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t accv0 = vdupq_n_f32(0.0f);
        float32x4_t accv1 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t scale_bits0;
            uint16_t scale_bits1;
            memcpy(&scale_bits0, row + b * 34, sizeof(scale_bits0));
            memcpy(&scale_bits1, row + (b + 1) * 34, sizeof(scale_bits1));

            const int8_t *qs0 = (const int8_t *)(row + b * 34 + 2);
            const int8_t *qs1 = (const int8_t *)(row + (b + 1) * 34 + 2);
            const int8_t *xq0 = xq + b * 32;
            const int8_t *xq1 = xq + (b + 1) * 32;

            int32x4_t dot0 = vdupq_n_s32(0);
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0),      vld1q_s8(xq0));
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0 + 16), vld1q_s8(xq0 + 16));

            int32x4_t dot1 = vdupq_n_s32(0);
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1),      vld1q_s8(xq1));
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1 + 16), vld1q_s8(xq1 + 16));

            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot0), f16_to_f32(scale_bits0) * xscale[b]);
            accv1 = vfmaq_n_f32(accv1, vcvtq_f32_s32(dot1), f16_to_f32(scale_bits1) * xscale[b + 1]);
        }

        if (b < blocks) {
            uint16_t scale_bits;
            memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
            const int8_t *qs = (const int8_t *)(row + b * 34 + 2);
            const int8_t *xqb = xq + b * 32;
            int32x4_t dot = vdupq_n_s32(0);
            dot = vdotq_s32(dot, vld1q_s8(qs),      vld1q_s8(xqb));
            dot = vdotq_s32(dot, vld1q_s8(qs + 16), vld1q_s8(xqb + 16));
            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot), f16_to_f32(scale_bits) * xscale[b]);
        }

        return vaddvq_f32(vaddq_f32(accv0, accv1));
    }
#endif

    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
        const int8_t *qs = (const int8_t *)(row + b * 34 + 2);

        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc += f16_to_f32(scale_bits) * xscale[b] * (float)dot_i8_32(qs, xq + i0, n);
    }
    return acc;
}


static inline DS4_MAYBE_UNUSED void dot_q8_0_row_pair(
        const uint8_t *row0,
        const uint8_t *row1,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks,
        float         *out0,
        float         *out1) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t acc00 = vdupq_n_f32(0.0f);
        float32x4_t acc01 = vdupq_n_f32(0.0f);
        float32x4_t acc10 = vdupq_n_f32(0.0f);
        float32x4_t acc11 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t s00, s01, s10, s11;
            memcpy(&s00, row0 + b * 34, sizeof(s00));
            memcpy(&s01, row0 + (b + 1) * 34, sizeof(s01));
            memcpy(&s10, row1 + b * 34, sizeof(s10));
            memcpy(&s11, row1 + (b + 1) * 34, sizeof(s11));

            const int8_t *xq0 = xq + b * 32;
            const int8_t *xq1 = xq + (b + 1) * 32;
            const int8x16_t xv00 = vld1q_s8(xq0);
            const int8x16_t xv01 = vld1q_s8(xq0 + 16);
            const int8x16_t xv10 = vld1q_s8(xq1);
            const int8x16_t xv11 = vld1q_s8(xq1 + 16);

            const int8_t *q00 = (const int8_t *)(row0 + b * 34 + 2);
            const int8_t *q01 = (const int8_t *)(row0 + (b + 1) * 34 + 2);
            const int8_t *q10 = (const int8_t *)(row1 + b * 34 + 2);
            const int8_t *q11 = (const int8_t *)(row1 + (b + 1) * 34 + 2);

            int32x4_t d00 = vdupq_n_s32(0);
            d00 = vdotq_s32(d00, vld1q_s8(q00),      xv00);
            d00 = vdotq_s32(d00, vld1q_s8(q00 + 16), xv01);
            int32x4_t d01 = vdupq_n_s32(0);
            d01 = vdotq_s32(d01, vld1q_s8(q01),      xv10);
            d01 = vdotq_s32(d01, vld1q_s8(q01 + 16), xv11);
            int32x4_t d10 = vdupq_n_s32(0);
            d10 = vdotq_s32(d10, vld1q_s8(q10),      xv00);
            d10 = vdotq_s32(d10, vld1q_s8(q10 + 16), xv01);
            int32x4_t d11 = vdupq_n_s32(0);
            d11 = vdotq_s32(d11, vld1q_s8(q11),      xv10);
            d11 = vdotq_s32(d11, vld1q_s8(q11 + 16), xv11);

            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d00), f16_to_f32(s00) * xscale[b]);
            acc01 = vfmaq_n_f32(acc01, vcvtq_f32_s32(d01), f16_to_f32(s01) * xscale[b + 1]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d10), f16_to_f32(s10) * xscale[b]);
            acc11 = vfmaq_n_f32(acc11, vcvtq_f32_s32(d11), f16_to_f32(s11) * xscale[b + 1]);
        }

        if (b < blocks) {
            uint16_t s0, s1;
            memcpy(&s0, row0 + b * 34, sizeof(s0));
            memcpy(&s1, row1 + b * 34, sizeof(s1));
            const int8_t *xqb = xq + b * 32;
            const int8x16_t xv0 = vld1q_s8(xqb);
            const int8x16_t xv1 = vld1q_s8(xqb + 16);
            const int8_t *q0 = (const int8_t *)(row0 + b * 34 + 2);
            const int8_t *q1 = (const int8_t *)(row1 + b * 34 + 2);
            int32x4_t d0 = vdupq_n_s32(0);
            d0 = vdotq_s32(d0, vld1q_s8(q0),      xv0);
            d0 = vdotq_s32(d0, vld1q_s8(q0 + 16), xv1);
            int32x4_t d1 = vdupq_n_s32(0);
            d1 = vdotq_s32(d1, vld1q_s8(q1),      xv0);
            d1 = vdotq_s32(d1, vld1q_s8(q1 + 16), xv1);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d0), f16_to_f32(s0) * xscale[b]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d1), f16_to_f32(s1) * xscale[b]);
        }

        *out0 = vaddvq_f32(vaddq_f32(acc00, acc01));
        *out1 = vaddvq_f32(vaddq_f32(acc10, acc11));
        return;
    }
#endif

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t s0_bits;
        uint16_t s1_bits;
        memcpy(&s0_bits, row0 + b * 34, sizeof(s0_bits));
        memcpy(&s1_bits, row1 + b * 34, sizeof(s1_bits));
        const int8_t *q0 = (const int8_t *)(row0 + b * 34 + 2);
        const int8_t *q1 = (const int8_t *)(row1 + b * 34 + 2);
        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc0 += f16_to_f32(s0_bits) * xscale[b] * (float)dot_i8_32(q0, xq + i0, n);
        acc1 += f16_to_f32(s1_bits) * xscale[b] * (float)dot_i8_32(q1, xq + i0, n);
    }
    *out0 = acc0;
    *out1 = acc1;
}

static void quantize_q8_0_activation(const float *x, int8_t *xq, float *scale, uint64_t n) {
    const uint64_t blocks = (n + 31) / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = n - i0 < 32 ? n - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32 && i0 + i < blocks * 32; i++) {
            xq[i0 + i] = 0;
        }
    }
}



static void matvec_q8_0_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint64_t o = ctx->row0 + r;
        const uint8_t *row = ctx->data + o * ctx->blocks * 34;
        ctx->out[r] = dot_q8_0_row(row, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks);
    }
}






/* Multiply selected Q8_0 rows by an activation that has already been quantized
 * once.  This avoids repeated activation quantization for paired projections. */
static void matvec_q8_0_rows_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          row0,
        uint64_t          n_rows) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    if (row0 > out_dim || n_rows > out_dim - row0) ds4_die("Q8_0 row range is outside tensor");
    const uint64_t ctx_blocks = (in_dim + 31) / 32;

    matvec_q8_0_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .row0 = row0,
        .blocks = ctx_blocks,
    };
    ds4_parallel_for(n_rows, matvec_q8_0_worker, &ctx);
}

static DS4_MAYBE_UNUSED void matvec_q8_0_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale) {
    matvec_q8_0_rows_prequant(out, m, w, xq, xscale, 0, w->dim[1]);
}

static void matvec_q8_0_3d_slice_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          slice) {
    if (w->type != DS4_TENSOR_Q8_0 || w->ndim != 3) ds4_die("expected a 3D Q8_0 tensor");
    if (slice >= w->dim[2]) ds4_die("Q8_0 slice is outside tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t slice_bytes = out_dim * blocks * 34;
    const uint8_t *data = (const uint8_t *)tensor_data(m, w) + slice * slice_bytes;

    matvec_q8_0_ctx ctx = {
        .out = out,
        .data = data,
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .row0 = 0,
        .blocks = blocks,
    };
    ds4_parallel_for(out_dim, matvec_q8_0_worker, &ctx);
}

static DS4_MAYBE_UNUSED void matvec_q8_0_3d_slice(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint64_t          slice) {
    if (w->type != DS4_TENSOR_Q8_0 || w->ndim != 3) ds4_die("expected a 3D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)blocks * 32);
    float *xscale = xmalloc((size_t)blocks * sizeof(xscale[0]));

    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_3d_slice_prequant(out, m, w, xq, xscale, slice);

    free(xscale);
    free(xq);
}

/* Compute two Q8_0 projections from the same input, used by gate/up and
 * compressor kv/score pairs. */



/* Batched Q8_0 matmul for prefill: quantize all token activations, then scan
 * weight rows once per output channel. */



/* Single-token Q8_0 matvec, used heavily in decode. */


/* Decode scratch owns this temporary activation quantization so generation
 * can assert that the hot path performs no malloc. */







typedef struct {
    float *out;
    const float *data;
    const float *x;
    uint64_t in_dim;
} matvec_f32_ctx;



/* Dispatch for dense F32/F16/Q8_0 tensors used by auxiliary projections. */



/* Locate one expert's 2D matrix inside a 3D GGUF expert tensor. */
static const uint8_t *tensor_expert_bytes(
        const ds4_model  *m,
        const ds4_tensor *w,
        uint32_t          expert,
        uint64_t         *in_dim,
        uint64_t         *out_dim,
        uint64_t         *row_bytes) {
    if (w->ndim != 3) ds4_die("expected a 3D expert tensor");
    if (expert >= w->dim[2]) ds4_die("expert id is outside expert tensor");

    *in_dim = w->dim[0];
    *out_dim = w->dim[1];

    const gguf_type_info *info = tensor_type(w->type);
    if (!info || info->block_elems == 0) ds4_die("unsupported expert tensor type");
    const uint64_t blocks = (*in_dim + info->block_elems - 1) / info->block_elems;
    *row_bytes = blocks * info->block_bytes;

    const uint64_t expert_bytes = *out_dim * *row_bytes;
    return (const uint8_t *)tensor_data(m, w) + (uint64_t)expert * expert_bytes;
}

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *base0;
    const uint8_t *base1;
    const block_q8_K *xq;
    uint64_t in_dim;
    uint64_t row_bytes0;
    uint64_t row_bytes1;
} matvec_iq2_xxs_pair_ctx;


/* Project one routed expert's gate and up matrices.  Both are IQ2_XXS and
 * share the same Q8_K activation. */

static float silu(float x);

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_iq2_xxs_mid_ctx;

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const int8_t *xq;
    const float *xscale;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q8_0_mid_ctx;

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q8_k_mid_ctx;


static void matvec_q8_0_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_0_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const uint8_t *gate_row = ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot];
        const uint8_t *up_row = ctx->up_base[slot] + row * ctx->up_row_bytes[slot];
        dot_q8_0_row_pair(gate_row, up_row, ctx->xq, ctx->xscale,
                          ctx->in_dim, ctx->blocks, &gate, &up);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}

static void matvec_q8_k_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_k_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const block_q8_K *gate_row = (const block_q8_K *)(ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot]);
        const block_q8_K *up_row = (const block_q8_K *)(ctx->up_base[slot] + row * ctx->up_row_bytes[slot]);
        ds4_vec_dot_q8_K_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, ctx->xq);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}

/* Build all selected expert hidden vectors: IQ2_XXS gate/up, clamp, SwiGLU,
 * and router weight.  The down projection runs later on the quantized mids. */

static DS4_MAYBE_UNUSED void matvec_q8_0_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const int8_t     *xq,
        const float      *xscale,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != DS4_TENSOR_Q8_0 || up_w->type != DS4_TENSOR_Q8_0) {
        ds4_die("expected Q8_0 expert tensors");
    }
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_q8_0_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .xscale = xscale,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired Q8_0 expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("Q8_0 expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if ((in_dim0 % 32u) != 0) ds4_die("Q8_0 expert row is not QK8_0 aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ctx.blocks = in_dim0 / 32u;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_q8_0_mid_worker, &ctx);
}

static DS4_MAYBE_UNUSED void matvec_q8_k_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != DS4_TENSOR_Q8_K || up_w->type != DS4_TENSOR_Q8_K) {
        ds4_die("expected Q8_K expert tensors");
    }
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_q8_k_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired Q8_K expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("Q8_K expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q8_K expert row is not QK_K aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_q8_k_mid_worker, &ctx);
}

typedef struct {
    float *out;
    const uint8_t *base;
    const block_q8_K *xq;
    uint64_t in_dim;
    uint64_t row_bytes;
} matvec_q2_k_ctx;


/* Single expert Q2_K down projection, kept mostly for tracing and diagnostics. */

typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q2_k_accum_ctx;


/* Accumulate all selected experts' Q2_K down projections directly into the
 * 4096-wide MoE output. */

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q2_k_mid_ctx;



typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const int8_t *xq[DS4_MAX_EXPERT_USED];
    const float *xscale[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t blocks;
    int n_expert;
} matvec_q8_0_accum_ctx;



typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q8_k_accum_ctx;



typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_iq2_xxs_accum_ctx;



typedef struct {
    uint32_t token;
    uint32_t slot;
} ds4_expert_pair;

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT];
    const uint8_t *up_base[DS4_MAX_EXPERT];
    const block_q8_K *xq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT];
    uint64_t up_row_bytes[DS4_MAX_EXPERT];
    uint64_t xq_blocks;
} matvec_q2_k_batch_mid_ctx;


typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT];
    const uint8_t *up_base[DS4_MAX_EXPERT];
    const block_q8_K *xq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT];
    uint64_t up_row_bytes[DS4_MAX_EXPERT];
    uint64_t xq_blocks;
} matvec_iq2_xxs_batch_mid_ctx;

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT];
    const uint8_t *up_base[DS4_MAX_EXPERT];
    const block_q8_K *xq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t xq_blocks;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT];
    uint64_t up_row_bytes[DS4_MAX_EXPERT];
} matvec_q8_k_batch_mid_ctx;



typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT];
    const uint8_t *up_base[DS4_MAX_EXPERT];
    const int8_t *xq;
    const float *xscale;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT];
    uint64_t up_row_bytes[DS4_MAX_EXPERT];
} matvec_q8_0_batch_mid_ctx;


typedef struct {
    const float *mid;
    block_q8_K *midq;
    uint64_t down_in_dim;
    uint64_t down_blocks;
} quantize_mid_pairs_ctx;


typedef struct {
    float *down_pair;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q2_k_batch_down_ctx;

static DS4_MAYBE_UNUSED void matvec_q2_k_batch_down_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_q2_k_batch_down_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];
        const block_q2_K *br = (const block_q2_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
            ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim,
                                  ctx->down_pair + (uint64_t)pair_id * ctx->out_dim + row,
                                  br, xq);
        }
    }
}

typedef struct {
    float *moe;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q2_k_batch_accum_rows_ctx;


/* =========================================================================
 * Q4_K routed expert matrix-vector products.
 * ========================================================================= */

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q4_k_mid_ctx;



typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q4_k_accum_ctx;




typedef struct {
    float *out;
    const uint8_t *base;
    const block_q8_K *xq;
    uint64_t in_dim;
    uint64_t row_bytes;
    uint32_t type;
} matvec_q5_q6_k_ctx;


typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    uint32_t gate_type;
    uint32_t up_type;
    int n_expert;
} matvec_q5_q6_k_mid_ctx;




typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    uint32_t type;
    int n_expert;
} matvec_q5_q6_k_accum_ctx;



/* Q4_K batch mid worker: same structure as IQ2_XXS batch but uses Q4_K dot. */
typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT];
    const uint8_t *up_base[DS4_MAX_EXPERT];
    const block_q8_K *xq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT];
    uint64_t up_row_bytes[DS4_MAX_EXPERT];
    uint64_t xq_blocks;
} matvec_q4_k_batch_mid_ctx;


/* Q4_K batch down accum worker: same structure as Q2_K batch but uses Q4_K dot. */
typedef struct {
    float *moe;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q4_k_batch_accum_rows_ctx;


typedef struct {
    float *moe;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_iq2_xxs_batch_accum_rows_ctx;


/* Dispatch: call the right gate/up mid builder based on tensor type. */

/* Dispatch: call the right down-projection accumulator based on tensor type. */

/* Dispatch: single-expert gate/up pair for tracing. */

/* Dispatch: single-expert down projection for tracing. */

typedef struct {
    float *moe;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q8_k_batch_accum_rows_ctx;


typedef struct {
    float *moe;
    const uint8_t *base[DS4_MAX_EXPERT];
    const int8_t *midq;
    const float *midscale;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t blocks;
} matvec_q8_0_batch_accum_rows_ctx;


typedef struct {
    float *moe;
    const float *down_pair;
    uint32_t n_tok;
    uint64_t out_dim;
} sum_down_pairs_ctx;

static DS4_MAYBE_UNUSED void sum_down_pairs_worker(void *vctx, uint64_t row0, uint64_t row1) {
    sum_down_pairs_ctx *ctx = vctx;
    for (uint64_t idx = row0; idx < row1; idx++) {
        const uint32_t token = (uint32_t)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)token * ctx->out_dim;
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const uint64_t pair_id = (uint64_t)token * DS4_N_EXPERT_USED + slot;
            acc += ctx->down_pair[pair_id * ctx->out_dim + row];
        }
        ctx->moe[idx] = acc;
    }
}

/* =========================================================================
 * Hyper-Connection Transforms.
 * =========================================================================
 *
 * DeepSeek V4 Flash keeps four hyper-connection streams per token.  Before
 * attention or FFN, a learned small projection chooses how to reduce the HC
 * state into the 4096-wide sublayer input.  After the sublayer, the post and
 * combine weights expand the result back into the four-stream HC state.
 */

/* Decode the HC control projection.  The output contains pre weights, post
 * gates, and a small doubly-normalized combine matrix. */

/* Reduce the four HC streams into the plain embedding vector consumed by a
 * normal attention or FFN sublayer. */

/* HC pre step for one token.  It normalizes the HC state, projects the control
 * vector, runs the Sinkhorn split, and emits the sublayer input plus post data. */


/* The input embedding starts all HC streams with the same token vector. */

/* HC post step for one sublayer output.  It injects the new block output and
 * mixes the previous HC streams through the learned combine matrix. */

typedef struct {
    float       *out_hc;
    const float *block_out;
    const float *residual_hc;
    const float *post;
    const float *comb;
    uint64_t     hc_dim;
    uint32_t     n_embd;
    uint32_t     n_hc;
} hc_post_batch_ctx;



typedef struct {
    float       *out_hc;
    const float *moe;
    const float *shared;
    const float *residual_hc;
    const float *post;
    const float *comb;
    uint64_t     hc_dim;
    uint32_t     n_embd;
    uint32_t     n_hc;
} hc_post_sum_batch_ctx;



typedef struct {
    const ds4_model *model;
    const ds4_tensor *fn;
    const ds4_tensor *scale;
    const ds4_tensor *base;
    const ds4_tensor *norm_w;
    const float *inp_hc;
    float *residual_hc;
    float *cur;
    float *norm;
    float *post;
    float *comb;
    uint64_t hc_dim;
    uint32_t n_hc;
} hc_pre_norm_batch_ctx;


/* Batched HC pre plus RMSNorm.  Prefill uses this to keep the layer-major
 * token batch in contiguous arrays. */


/* =========================================================================
 * Attention Projections, RoPE, and Attention Output.
 * =========================================================================
 *
 * This block performs the attention half of a transformer layer: HC pre,
 * attention RMSNorm, Q and KV projections, layer-specific RoPE, sink-aware
 * attention over raw and compressed KV rows, and the grouped LoRA output
 * projection back to embedding width.
 */


/* KV projection has one KV head of width 512, followed by a learned RMSNorm. */






/* Apply DS4 RoPE only to the tail of each head.  Compressed layers use the
 * long-context frequency base and scale; inverse mode rotates attention output
 * back before the grouped output projection. */

/* Dense layers and compressed layers use different RoPE bases. */



typedef struct {
    float            *x;
    uint64_t          stride;
    uint32_t          n_head;
    uint32_t          head_dim;
    uint32_t          n_rot;
    uint32_t          pos0;
    uint32_t          il;
    bool              inverse;
} rope_tail_batch_ctx;






static float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = expf(x);
        return e / (1.0f + e);
    }
}

/* Sink-aware attention over a set of KV rows.  The learned sink logit is part
 * of the softmax denominator but contributes no value vector. */

/* Attention output projection is grouped: each group first maps its heads to
 * a 1024-rank low vector, then all groups are projected back to 4096. */



/* =========================================================================
 * Mixture-of-Experts FFN.
 * =========================================================================
 *
 * This is the FFN half of each layer.  It includes the shared expert, routed
 * expert selection, IQ2_XXS gate/up projections, SwiGLU, Q2_K down projection,
 * and the HC post step that returns the result to four-stream state.
 */

static float silu(float x) {
    return x * sigmoid_stable(x);
}



/* The shared expert is a normal Q8_0 SwiGLU MLP that runs for every token. */


typedef struct {
    float *mid;
    const float *gate;
    const float *up;
    uint64_t n;
    float clamp;
} swiglu_batch_ctx;



#ifndef DS4_NO_GPU
static int sample_argmax(const float *logits, uint32_t n_vocab);
#endif


typedef struct ds4_vocab ds4_vocab;

/* =========================================================================
 * Tokenizer and Chat Prompt Encoding.
 * =========================================================================
 *
 * DeepSeek V4 Flash stores a GPT-2 style byte-level BPE tokenizer in GGUF.
 * The implementation below is intentionally small.  It loads token strings
 * and merge ranks from the mmaped file, builds two open-addressed hash tables,
 * and applies BPE to user text.  Chat special tokens are inserted directly by
 * ID; user text goes through BPE.
 */

typedef struct {
    ds4_str key;
    int value;
    bool used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} str_i32_table;

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(str_i32_table *t, ds4_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (ds4_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return false;

    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;

    while (t->entry[i].used) {
        ds4_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

static void token_vec_push(token_vec *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

static void token_vec_free(token_vec *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

void ds4_tokens_push(ds4_tokens *tv, int token) {
    token_vec_push(tv, token);
}

void ds4_tokens_free(ds4_tokens *tv) {
    token_vec_free(tv);
}

void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) token_vec_push(dst, src->v[i]);
}

bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix) {
    if (prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}

struct ds4_vocab {
    ds4_str *token;
    int n_vocab;
    int bos_id;
    int eos_id;
    int eot_id;
    int system_id;
    int user_id;
    int assistant_id;
    int observation_id;
    int sop_id;
    int think_start_id;
    int think_end_id;
    int tool_call_start_id;
    int tool_call_end_id;
    int tool_response_start_id;
    int tool_response_end_id;
    int arg_key_start_id;
    int arg_key_end_id;
    int arg_value_start_id;
    int arg_value_end_id;
    int dsml_id;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
};

struct ds4_engine {
    ds4_model model;
    ds4_model dflash_model;
    ds4_vocab vocab;
    ds4_weights weights;
    ds4_dflash_weights dflash_weights;
    void *dflash_f16_map;
    uint64_t dflash_f16_map_size;
    ds4_support_kind support_kind;
    uint32_t support_stages;
    int dflash_draft_tokens;
    float dflash_p_min;
    uint64_t startup_model_span_bytes;
    bool quality;
    bool metal_ready;
    bool dflash_ready;
    size_t live_sessions;
    bool closing;

};

#ifdef DS4_TEST_HOOKS
typedef enum {
    DS4_ENGINE_CLOSE_BEGIN,
    DS4_ENGINE_CLOSE_GPU_DRAIN_BEGIN,
    DS4_ENGINE_CLOSE_GPU_DRAINED,
    DS4_ENGINE_CLOSE_GPU_CLEANUP_BEGIN,
    DS4_ENGINE_CLOSE_GPU_CLEANED,
    DS4_ENGINE_CLOSE_CPU_WORKERS_STOPPED,
    DS4_ENGINE_CLOSE_HOST_ALIASES_CLEARED,
    DS4_ENGINE_CLOSE_DFLASH_SHADOW_UNMAPPED,
    DS4_ENGINE_CLOSE_MODEL_MAPS_CLOSED,
    DS4_ENGINE_CLOSE_LOCK_RELEASED,
    DS4_ENGINE_CLOSE_ALLOCATIONS_RELEASING,
} ds4_engine_close_phase;

typedef struct {
    ds4_engine *engine;
    ds4_engine_close_phase phases[16];
    size_t phase_count;
    bool maps_live_through_gpu;
    bool aliases_cleared_before_unmap;
    bool dflash_shadow_unmapped_first;
    bool model_maps_unmapped;
    bool gpu_drain_reported_failure;
} ds4_test_engine_close_trace;

static ds4_test_engine_close_trace *g_ds4_test_engine_close_trace;

static bool ds4_test_engine_bytes_zero(const void *ptr, size_t size) {
    const unsigned char *bytes = ptr;
    for (size_t i = 0; i < size; i++) {
        if (bytes[i] != 0) return false;
    }
    return true;
}

static void ds4_engine_close_note(ds4_engine *e,
                                  ds4_engine_close_phase phase) {
    ds4_test_engine_close_trace *trace = g_ds4_test_engine_close_trace;
    if (!trace || trace->engine != e) return;
    if (trace->phase_count < sizeof(trace->phases) / sizeof(trace->phases[0])) {
        trace->phases[trace->phase_count++] = phase;
    }

    if (phase >= DS4_ENGINE_CLOSE_GPU_DRAIN_BEGIN &&
        phase <= DS4_ENGINE_CLOSE_GPU_CLEANED) {
        trace->maps_live_through_gpu =
            trace->maps_live_through_gpu &&
            e->model.map != NULL &&
            e->dflash_model.map != NULL && e->dflash_f16_map != NULL;
    } else if (phase == DS4_ENGINE_CLOSE_HOST_ALIASES_CLEARED) {
        trace->aliases_cleared_before_unmap =
            e->model.map != NULL &&
            e->dflash_model.map != NULL && e->dflash_f16_map != NULL &&
            ds4_test_engine_bytes_zero(&e->weights, sizeof(e->weights)) &&
            ds4_test_engine_bytes_zero(&e->dflash_weights,
                                       sizeof(e->dflash_weights));
    } else if (phase == DS4_ENGINE_CLOSE_DFLASH_SHADOW_UNMAPPED) {
        trace->dflash_shadow_unmapped_first =
            e->dflash_f16_map == NULL &&
            e->dflash_f16_map_size == 0 &&
            e->model.map != NULL &&
            e->dflash_model.map != NULL;
    } else if (phase == DS4_ENGINE_CLOSE_MODEL_MAPS_CLOSED) {
        trace->model_maps_unmapped =
            e->model.map == NULL &&
            e->dflash_model.map == NULL;
    }
}

#if !defined(DS4_NO_GPU)
static void ds4_engine_close_note_drain_result(ds4_engine *e, bool drained) {
    ds4_test_engine_close_trace *trace = g_ds4_test_engine_close_trace;
    if (!trace || trace->engine != e) return;
    trace->gpu_drain_reported_failure = !drained;
}
#else
#define ds4_engine_close_note_drain_result(engine, drained) \
    ((void)(engine), (void)(drained))
#endif
#else
#define ds4_engine_close_note(engine, phase) ((void)(engine))
#define ds4_engine_close_note_drain_result(engine, drained) \
    ((void)(engine), (void)(drained))
#endif

#if defined(__APPLE__)
static bool laguna_metal_router_simd_topk_preflight(
        const ds4_engine *engine,
        const char       *operation,
        char             *err,
        size_t            errlen);
static bool laguna_metal_swa_gqa9_preflight(
        const ds4_engine *engine,
        const char       *operation,
        char             *err,
        size_t            errlen);
static bool laguna_metal_router_simd_topk_trace_enabled(void);
static void laguna_metal_router_simd_topk_trace_reset(void);
static void laguna_metal_router_simd_topk_trace_report(const char *operation);
#endif

static void ds4_engine_print_startup_memory(
        const ds4_engine *e,
        int               ctx_size) {
    if (!e || ctx_size <= 0) return;

    const ds4_context_memory mem = ds4_context_memory_estimate(ctx_size);
    const uint64_t kv_bytes =
        ds4_add_sat_u64(mem.raw_bytes, mem.compressed_bytes);
    const uint64_t support_model_bytes =
        e->dflash_ready &&
        e->dflash_model.size > e->dflash_model.tensor_data_pos ?
            e->dflash_model.size - e->dflash_model.tensor_data_pos : 0;
    uint64_t total = kv_bytes;
    total = ds4_add_sat_u64(total, mem.scratch_bytes);
    total = ds4_add_sat_u64(total, e->startup_model_span_bytes);
    total = ds4_add_sat_u64(total, support_model_bytes);

    const bool color = ds4_log_is_tty(stderr);
    const char *green = color ? "\x1b[32m" : "";
    const char *bright_green = color ? "\x1b[1;32m" : "";
    const char *reset = color ? "\x1b[0m" : "";
    fprintf(stderr,
            "%sds4: memory: KV %.2f GiB (raw %.2f + compressed %.2f) "
            "+ buffers %.2f GiB + resident model %.2f GiB",
            green,
            ds4_bytes_to_gib(kv_bytes),
            ds4_bytes_to_gib(mem.raw_bytes),
            ds4_bytes_to_gib(mem.compressed_bytes),
            ds4_bytes_to_gib(mem.scratch_bytes),
            ds4_bytes_to_gib(e->startup_model_span_bytes));
    if (support_model_bytes != 0) {
        fprintf(stderr, " + support model %.2f GiB",
                ds4_bytes_to_gib(support_model_bytes));
    }
    fprintf(stderr, " = %s%.2f GiB planned%s\n",
            bright_green, ds4_bytes_to_gib(total), reset);
    fprintf(stderr,
            "%sds4: memory detail: ctx=%d prefill_cap=%u raw_kv_rows=%u "
            "compressed_kv_rows=%u backend=metal%s\n",
            green, ctx_size, mem.prefill_cap, mem.raw_cap, mem.comp_cap,
            reset);
}


static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }

    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

/* GPT-2 byte-level BPE first maps raw bytes to printable Unicode codepoints
 * so merges can operate on UTF-8 strings without losing byte identity. */
static char *byte_encode(ds4_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;

    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

/* Look up the merge rank for two adjacent BPE symbols. */
static int bpe_rank(const ds4_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return rank;
}

/* Apply byte-level BPE to one regex-like pre-tokenized piece and emit token ids. */
static void bpe_emit_piece(const ds4_vocab *vocab, ds4_str raw_piece, token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);

    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) {
            sym[j] = sym[j + 1];
        }
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            token_vec_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    token_vec_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
}

typedef struct {
    const ds4_vocab *vocab;
    token_vec       *out;
} bpe_emit_context;

static bool bpe_emit_lgn_piece(const char *piece,
                               size_t      piece_len,
                               void       *userdata) {
    bpe_emit_context *context = userdata;
    bpe_emit_piece(context->vocab,
                   (ds4_str){ piece, (uint64_t)piece_len },
                   context->out);
    return true;
}

static void bpe_tokenize_text(const ds4_vocab *vocab, const char *text, token_vec *out) {
    bpe_emit_context context = { vocab, out };
    if (!lgn_bpe_pretokenize(text, bpe_emit_lgn_piece, &context)) {
        ds4_die("Laguna BPE pre-tokenizer rejected its input");
    }
}

static int vocab_lookup(const ds4_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "ds4: required tokenizer token is missing: %s\n", text);
        exit(1);
    }
    return token;
}

/* Load token strings, special token ids, and merge ranks from GGUF metadata. */

static void vocab_load(ds4_vocab *vocab, const ds4_model *model) {
    memset(vocab, 0, sizeof(*vocab));

    ds4_array_ref tokens;
    ds4_array_ref merges;
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING ||
        tokens.len > INT32_MAX) {
        ds4_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) ||
        merges.type != GGUF_VALUE_STRING) {
        ds4_die("GGUF tokenizer merge table is missing or invalid");
    }

    vocab->n_vocab = (int)tokens.len;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);

    ds4_cursor c = cursor_at(model, tokens.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        if (!cursor_string(&c, &vocab->token[i])) ds4_die(c.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
    }

    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        ds4_str merge;
        if (!cursor_string(&c, &merge)) ds4_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }

    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA) {
        if (!model_get_token_id(model, "tokenizer.ggml.bos_token_id", &vocab->bos_id) ||
            !model_get_token_id(model, "tokenizer.ggml.eos_token_id", &vocab->eos_id)) {
            ds4_die("Laguna tokenizer is missing BOS/EOS token metadata");
        }
        if (!model_get_token_id(model, "tokenizer.ggml.eot_token_id", &vocab->eot_id)) {
            vocab->eot_id = vocab_lookup(vocab, "</assistant>");
        }
        vocab->system_id = -1;
        vocab->user_id = -1;
        vocab->assistant_id = vocab_lookup(vocab, "<assistant>");
        vocab->observation_id = -1;
        vocab->sop_id = -1;
        vocab->think_start_id = vocab_lookup(vocab, "<think>");
        vocab->think_end_id = vocab_lookup(vocab, "</think>");
        vocab->tool_call_start_id = vocab_lookup(vocab, "<tool_call>");
        vocab->tool_call_end_id = vocab_lookup(vocab, "</tool_call>");
        vocab->tool_response_start_id = -1;
        vocab->tool_response_end_id = -1;
        vocab->arg_key_start_id = -1;
        vocab->arg_key_end_id = -1;
        vocab->arg_value_start_id = -1;
        vocab->arg_value_end_id = -1;
        vocab->dsml_id = -1;
        return;
    }

    vocab->bos_id       = vocab_lookup(vocab, "<｜begin▁of▁sentence｜>");
    vocab->eos_id       = vocab_lookup(vocab, "<｜end▁of▁sentence｜>");
    vocab->system_id    = -1;
    vocab->user_id      = vocab_lookup(vocab, "<｜User｜>");
    vocab->assistant_id = vocab_lookup(vocab, "<｜Assistant｜>");
    vocab->observation_id = -1;
    vocab->sop_id = -1;
    vocab->think_start_id = vocab_lookup(vocab, "<think>");
    vocab->think_end_id = vocab_lookup(vocab, "</think>");
    vocab->tool_call_start_id = -1;
    vocab->tool_call_end_id = -1;
    vocab->tool_response_start_id = -1;
    vocab->tool_response_end_id = -1;
    vocab->arg_key_start_id = -1;
    vocab->arg_key_end_id = -1;
    vocab->arg_value_start_id = -1;
    vocab->arg_value_end_id = -1;
    vocab->dsml_id = vocab_lookup(vocab, "｜DSML｜");
    vocab->eot_id = -1;
}

static void vocab_free(ds4_vocab *vocab) {
    free(vocab->token);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

/* Build the DS4 chat prompt: BOS, optional system text, user prompt, assistant
 * marker, and either <think> or </think> depending on the requested mode.  Max
 * thinking is only a prompt prefix: the model still enters through <think>. */
static void chat_push_bos_sequence(const ds4_vocab *vocab, token_vec *out) {
    token_vec_push(out, vocab->bos_id);
}

static void chat_push_think_prefix(const ds4_vocab *vocab,
                                   ds4_think_mode   think_mode,
                                   token_vec       *out) {
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK4 &&
        think_mode == DS4_THINK_MAX) {
        bpe_tokenize_text(vocab, DS4_REASONING_EFFORT_MAX_PREFIX, out);
    }
}

static void tokenize_rendered_chat_vocab(const ds4_vocab *vocab,
                                         const char *text,
                                         token_vec *out);

static void laguna_chat_append_wrapped(const ds4_vocab *vocab,
                                       token_vec       *out,
                                       const char      *open,
                                       const char      *content,
                                       const char      *close) {
    if (!content) content = "";
    const size_t open_len = strlen(open);
    const size_t content_len = strlen(content);
    const size_t close_len = strlen(close);
    if (open_len > SIZE_MAX - content_len ||
        open_len + content_len > SIZE_MAX - close_len - 2u) {
        ds4_die("Laguna chat message is too large");
    }
    const size_t len = open_len + content_len + close_len + 1u;
    char *rendered = xmalloc(len + 1u);
    char *p = rendered;
    memcpy(p, open, open_len); p += open_len;
    memcpy(p, content, content_len); p += content_len;
    memcpy(p, close, close_len); p += close_len;
    *p++ = '\n';
    *p = '\0';
    /* The official template tokenizes each rendered message as contiguous
     * text.  Splitting at tag/content boundaries prevents valid BPE merges
     * such as `>You` and `.</`, changing the prompt despite identical text. */
    tokenize_rendered_chat_vocab(vocab, rendered, out);
    free(rendered);
}

static void encode_chat_prompt(
        const ds4_vocab *vocab,
        const char      *system,
        const char      *prompt,
        ds4_think_mode   think_mode,
        token_vec       *out) {
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA) {
        if (vocab->bos_id < 0 ||
            vocab->assistant_id < 0 ||
            vocab->think_start_id < 0 ||
            vocab->think_end_id < 0) {
            ds4_die("this tokenizer does not provide the Laguna chat markers; use raw prompt tokenization");
        }
        chat_push_bos_sequence(vocab, out);
        if (system && system[0]) {
            laguna_chat_append_wrapped(vocab, out, "<system>", system, "</system>");
        }
        laguna_chat_append_wrapped(vocab, out, "<user>", prompt, "</user>");
        token_vec_push(out, vocab->assistant_id);
        token_vec_push(out, ds4_think_mode_enabled(think_mode) ?
                       vocab->think_start_id : vocab->think_end_id);
        return;
    }

    const bool need_think_start = ds4_think_mode_enabled(think_mode);
    if (vocab->bos_id < 0 ||
        vocab->user_id < 0 ||
        vocab->assistant_id < 0 ||
        vocab->think_end_id < 0 ||
        (need_think_start && vocab->think_start_id < 0)) {
        ds4_die("this tokenizer does not provide the DeepSeek chat markers; use raw prompt tokenization");
    }

    chat_push_bos_sequence(vocab, out);
    chat_push_think_prefix(vocab, think_mode, out);
    if (system && system[0]) {
        bpe_tokenize_text(vocab, system, out);
    }
    token_vec_push(out, vocab->user_id);
    bpe_tokenize_text(vocab, prompt, out);
    token_vec_push(out, vocab->assistant_id);
    if (ds4_think_mode_enabled(think_mode)) {
        token_vec_push(out, vocab->think_start_id);
    } else {
        token_vec_push(out, vocab->think_end_id);
    }
}

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    bpe_tokenize_text(&e->vocab, text ? text : "", out);
}

static bool special_token_at(const ds4_vocab *vocab, const char *p, int *token, size_t *len) {
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"〈|EOS|〉",               vocab->eos_id},
        {"<｜begin▁of▁sentence｜>", vocab->bos_id},
        {"<｜end▁of▁sentence｜>",   vocab->eos_id},
        {"[gMASK]",                vocab->bos_id},
        {"<sop>",                  vocab->sop_id},
        {"<|system|>",             vocab->system_id},
        {"<｜User｜>",              vocab->user_id},
        {"<｜Assistant｜>",         vocab->assistant_id},
        {"<|user|>",               vocab->user_id},
        {"<|assistant|>",          vocab->assistant_id},
        {"<assistant>",            vocab->assistant_id},
        {"</assistant>",           vocab->eot_id},
        {"<|observation|>",        vocab->observation_id},
        {"<think>",                vocab->think_start_id},
        {"</think>",               vocab->think_end_id},
        {"<tool_call>",            vocab->tool_call_start_id},
        {"</tool_call>",           vocab->tool_call_end_id},
        {"<tool_response>",        vocab->tool_response_start_id},
        {"</tool_response>",       vocab->tool_response_end_id},
        {"<arg_key>",              vocab->arg_key_start_id},
        {"</arg_key>",             vocab->arg_key_end_id},
        {"<arg_value>",            vocab->arg_value_start_id},
        {"</arg_value>",           vocab->arg_value_end_id},
        {"｜DSML｜",                vocab->dsml_id},
    };

    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        if (specials[i].token < 0) continue;
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}

static void tokenize_span(const ds4_vocab *vocab, const char *p, size_t n, token_vec *out) {
    if (!n) return;
    char *tmp = xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
}




static void tokenize_rendered_chat_vocab(const ds4_vocab *vocab, const char *text,
                                         token_vec *out) {
    if (!text) text = "";

    const char *span = text;
    const char *p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (special_token_at(vocab, p, &token, &len)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            token_vec_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(p - span), out);
}

void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    tokenize_rendered_chat_vocab(&e->vocab, text, out);
}

void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens) {
    chat_push_bos_sequence(&e->vocab, tokens);
}

void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out) {
    encode_chat_prompt(&e->vocab, system, prompt ? prompt : "", think_mode, out);
}

void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens) {
    bpe_tokenize_text(&e->vocab, DS4_REASONING_EFFORT_MAX_PREFIX, tokens);
}

static void laguna_chat_append_tool_response(ds4_vocab *vocab,
                                             token_vec *out,
                                             const char *content) {
    static const char open[] = "<tool_response>";
    static const char close[] = "</tool_response>";
    if (!content) content = "";

    size_t escaped_len = strlen(content);
    const size_t close_len = sizeof(close) - 1u;
    for (const char *p = content; *p; p++) {
        if (!strncmp(p, close, close_len)) {
            if (escaped_len > SIZE_MAX - 3u)
                ds4_die("Laguna tool response is too large");
            escaped_len += 3u; /* '<' becomes "&lt;". */
        }
    }
    if (escaped_len > SIZE_MAX - sizeof(open) - sizeof(close))
        ds4_die("Laguna tool response is too large");

    const size_t rendered_len =
        (sizeof(open) - 1u) + escaped_len + close_len + 1u;
    char *rendered = xmalloc(rendered_len + 1u);
    char *dst = rendered;
    memcpy(dst, open, sizeof(open) - 1u);
    dst += sizeof(open) - 1u;
    for (const char *p = content; *p;) {
        if (!strncmp(p, close, close_len)) {
            memcpy(dst, "&lt;", 4u);
            dst += 4u;
            p++;
        } else {
            *dst++ = *p++;
        }
    }
    memcpy(dst, close, close_len);
    dst += close_len;
    *dst++ = '\n';
    *dst = '\0';

    /* These tags are ordinary Laguna BPE text.  Tokenize the complete span so
     * merges across tag/payload boundaries match the official template. */
    tokenize_rendered_chat_vocab(vocab, rendered, out);
    free(rendered);
}

void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content) {
    ds4_vocab *vocab = &e->vocab;
    if (!role) role = "user";
    if (!content) content = "";

    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA) {
        if (!strcmp(role, "system") || !strcmp(role, "developer")) {
            laguna_chat_append_wrapped(vocab, tokens, "<system>", content, "</system>");
        } else if (!strcmp(role, "assistant")) {
            token_vec_push(tokens, vocab->assistant_id);
            if (strncmp(content, "<think>", 7) != 0 &&
                strncmp(content, "</think>", 8) != 0) {
                token_vec_push(tokens, vocab->think_end_id);
            }
            tokenize_rendered_chat_vocab(vocab, content, tokens);
            token_vec_push(tokens, vocab->eot_id);
            bpe_tokenize_text(vocab, "\n", tokens);
        } else if (!strcmp(role, "tool") || !strcmp(role, "function")) {
            laguna_chat_append_tool_response(vocab, tokens, content);
        } else {
            laguna_chat_append_wrapped(vocab, tokens, "<user>", content, "</user>");
        }
        return;
    }
}


void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode) {
    token_vec_push(tokens, e->vocab.assistant_id);
    token_vec_push(tokens, ds4_think_mode_enabled(think_mode) ?
                   e->vocab.think_start_id : e->vocab.think_end_id);
}

void ds4_chat_append_assistant_end(ds4_engine *e, ds4_tokens *tokens) {
    if (!e || !tokens) return;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA) {
        token_vec_push(tokens, e->vocab.eot_id);
        bpe_tokenize_text(&e->vocab, "\n", tokens);
    } else {
        token_vec_push(tokens, e->vocab.eos_id);
    }
}

static void dump_tokens_fp(FILE *fp, const ds4_vocab *vocab, const token_vec *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) {
        if (i) fprintf(fp, ", ");
        fprintf(fp, "%d", tokens->v[i]);
    }
    fprintf(fp, "]\n");

    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i];
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%6d  %.*s\n", id, (int)vocab->token[id].len, vocab->token[id].ptr);
        }
    }
}

static void dump_tokens(const ds4_vocab *vocab, const token_vec *tokens) {
    dump_tokens_fp(stdout, vocab, tokens);
}

static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
            continue;
        }
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}

static bool vocab_token_is_literal_special(ds4_str s) {
    const unsigned char bar[] = {0xef, 0xbd, 0x9c}; /* U+FF5C fullwidth vertical bar. */
    if (s.len < sizeof(bar)) return false;
    for (uint64_t i = 0; i + sizeof(bar) <= s.len; i++) {
        if (!memcmp(s.ptr + i, bar, sizeof(bar))) return true;
    }
    return false;
}

static bool vocab_token_is_named_special(const ds4_vocab *vocab, int token) {
    const int ids[] = {
        vocab->bos_id,
        vocab->eos_id,
        vocab->system_id,
        vocab->user_id,
        vocab->assistant_id,
        vocab->observation_id,
        vocab->sop_id,
        vocab->think_start_id,
        vocab->think_end_id,
        vocab->tool_call_start_id,
        vocab->tool_call_end_id,
        vocab->tool_response_start_id,
        vocab->tool_response_end_id,
        vocab->arg_key_start_id,
        vocab->arg_key_end_id,
        vocab->arg_value_start_id,
        vocab->arg_value_end_id,
        vocab->dsml_id,
        vocab->eot_id,
    };
    for (size_t i = 0; i < sizeof(ids) / sizeof(ids[0]); i++) {
        if (ids[i] >= 0 && token == ids[i]) return true;
    }
    return false;
}

char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    ds4_vocab *vocab = &e->vocab;
    if (token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        char *out = xmalloc(1);
        out[0] = '\0';
        return out;
    }

    ds4_str s = vocab->token[token];
    char *out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_named_special(vocab, token) ||
        vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }

    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

void ds4_token_text_into(ds4_engine *e, int token, ds4_buf *b) {
    ds4_vocab *vocab = &e->vocab;
    if (!b || token < 0 || token >= vocab->n_vocab) return;

    ds4_str s = vocab->token[token];
    /* Decoding emits at most one byte per source byte, so one reserve covers
     * both the raw copy and the GPT-2 byte-mapping loop below. */
    if (b->len + (size_t)s.len + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->len + (size_t)s.len + 1) cap *= 2;
        b->ptr = xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    char *out = b->ptr + b->len;
    if (vocab_token_is_named_special(vocab, token) ||
        vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        b->len += (size_t)s.len;
    } else {
        size_t n = 0;
        uint64_t pos = 0;
        while (pos < s.len) {
            uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
            int byte = gpt2_codepoint_to_byte(cp);
            if (byte >= 0) out[n++] = (char)byte;
        }
        b->len += n;
    }
    b->ptr[b->len] = '\0';
}

static bool vocab_token_is_generation_stop(const ds4_vocab *vocab, int token) {
    if (!vocab || token < 0) return false;
    if (token == vocab->eos_id) return true;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA &&
        vocab->eot_id >= 0 && token == vocab->eot_id) {
        return true;
    }
    return false;
}

int ds4_token_eos(ds4_engine *e) {
    return e->vocab.eos_id;
}

bool ds4_token_is_stop(ds4_engine *e, int token) {
    return e ? vocab_token_is_generation_stop(&e->vocab, token) : false;
}

bool ds4_token_is_thinking_control(ds4_engine *e, int token) {
    if (!e || token < 0) return false;
    return (e->vocab.think_start_id >= 0 &&
            token == e->vocab.think_start_id) ||
           (e->vocab.think_end_id >= 0 &&
            token == e->vocab.think_end_id);
}

bool ds4_token_is_stop_for_think_mode(
        ds4_engine      *e,
        int              token,
        ds4_think_mode   mode) {
    if (ds4_token_is_stop(e, token)) return true;
    /* In no-thinking mode the prompt already supplied the protocol close tag.
     * If the model emits another thinking tag, do not print or feed it back:
     * it is a control marker, not assistant content. */
    if (!ds4_think_mode_enabled(mode) &&
        ds4_token_is_thinking_control(e, token)) {
        return true;
    }
    return false;
}

int ds4_token_user(ds4_engine *e) {
    return e->vocab.user_id;
}

int ds4_token_assistant(ds4_engine *e) {
    return e->vocab.assistant_id;
}

static inline void argmax_f32_unrolled8_range(
        const float *logits,
        uint32_t     begin,
        uint32_t     end,
        int         *best,
        float       *best_v) {
    uint32_t i = begin;
    int b0 = *best, b1 = *best, b2 = *best, b3 = *best;
    int b4 = *best, b5 = *best, b6 = *best, b7 = *best;
    float v0 = *best_v, v1 = *best_v, v2 = *best_v, v3 = *best_v;
    float v4 = *best_v, v5 = *best_v, v6 = *best_v, v7 = *best_v;

    while (end - i >= 8u) {
        const float x0 = logits[i + 0u];
        const float x1 = logits[i + 1u];
        const float x2 = logits[i + 2u];
        const float x3 = logits[i + 3u];
        const float x4 = logits[i + 4u];
        const float x5 = logits[i + 5u];
        const float x6 = logits[i + 6u];
        const float x7 = logits[i + 7u];
        if (x0 > v0) { v0 = x0; b0 = (int)(i + 0u); }
        if (x1 > v1) { v1 = x1; b1 = (int)(i + 1u); }
        if (x2 > v2) { v2 = x2; b2 = (int)(i + 2u); }
        if (x3 > v3) { v3 = x3; b3 = (int)(i + 3u); }
        if (x4 > v4) { v4 = x4; b4 = (int)(i + 4u); }
        if (x5 > v5) { v5 = x5; b5 = (int)(i + 5u); }
        if (x6 > v6) { v6 = x6; b6 = (int)(i + 6u); }
        if (x7 > v7) { v7 = x7; b7 = (int)(i + 7u); }
        i += 8u;
    }

#define DS4_ARGMAX_MERGE_LANE(b, v) \
    do { \
        if ((v) > *best_v || ((v) == *best_v && (b) < *best)) { \
            *best_v = (v); \
            *best = (b); \
        } \
    } while (0)
    DS4_ARGMAX_MERGE_LANE(b0, v0);
    DS4_ARGMAX_MERGE_LANE(b1, v1);
    DS4_ARGMAX_MERGE_LANE(b2, v2);
    DS4_ARGMAX_MERGE_LANE(b3, v3);
    DS4_ARGMAX_MERGE_LANE(b4, v4);
    DS4_ARGMAX_MERGE_LANE(b5, v5);
    DS4_ARGMAX_MERGE_LANE(b6, v6);
    DS4_ARGMAX_MERGE_LANE(b7, v7);
#undef DS4_ARGMAX_MERGE_LANE

    for (; i < end; i++) {
        const float v = logits[i];
        if (v > *best_v) {
            *best_v = v;
            *best = (int)i;
        }
    }
}

static int sample_argmax_unrolled8(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = DS4_NEG_INF;
    argmax_f32_unrolled8_range(logits, 0, n_vocab, &best, &best_v);
    return best;
}

static int argmax_f32_excluding_unrolled8(
        const float *logits,
        uint32_t     n,
        int          excluded_id) {
    const uint32_t first = excluded_id == 0 ? 1u : 0u;
    if (first >= n) return -1;

    int best = (int)first;
    float best_v = logits[first];
    const uint32_t begin = first + 1u;
    if (excluded_id >= 0 &&
        (uint32_t)excluded_id >= begin &&
        (uint32_t)excluded_id < n) {
        const uint32_t excluded = (uint32_t)excluded_id;
        argmax_f32_unrolled8_range(logits, begin, excluded, &best, &best_v);
        argmax_f32_unrolled8_range(logits, excluded + 1u, n, &best, &best_v);
    } else {
        argmax_f32_unrolled8_range(logits, begin, n, &best, &best_v);
    }
    return best;
}

static int sample_argmax(const float *logits, uint32_t n_vocab) {
    if (getenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == NULL) {
        return sample_argmax_unrolled8(logits, n_vocab);
    }
    int best = 0;
    float best_v = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

static DS4_MAYBE_UNUSED void logits_top2(const float *logits, uint32_t n_vocab,
                        int *top0, float *logit0,
                        int *top1, float *logit1) {
    int b0 = -1, b1 = -1;
    float v0 = DS4_NEG_INF, v1 = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > v0) {
            b1 = b0; v1 = v0;
            b0 = (int)i; v0 = v;
        } else if (v > v1) {
            b1 = (int)i; v1 = v;
        }
    }
    if (top0) *top0 = b0;
    if (logit0) *logit0 = v0;
    if (top1) *top1 = b1;
    if (logit1) *logit1 = v1;
}

static uint64_t sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}

static float sample_rng_f32(uint64_t *state) {
    const uint64_t x = sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

typedef struct {
    int id;
    float logit;
    float prob;
} sample_candidate;

/* Session-owned candidate arena: the full-vocab candidate list is reused
 * across tokens instead of a fresh multi-MB malloc/free per decode step. */
typedef struct {
    sample_candidate *v;
    size_t cap;
} sample_arena;

static sample_candidate *sample_arena_reserve(sample_arena *a, size_t n) {
    if (n > a->cap) {
        size_t cap = a->cap ? a->cap : 1024;
        while (cap < n) cap *= 2;
        a->v = xrealloc(a->v, cap * sizeof(a->v[0]));
        a->cap = cap;
    }
    return a->v;
}

#ifdef DS4_TEST_HOOKS
int ds4_test_sample_arena_lifecycle(void) {
    sample_arena arena = {0};
    sample_candidate *first = sample_arena_reserve(&arena, 17);
    const size_t first_cap = arena.cap;
    sample_candidate *reuse = sample_arena_reserve(&arena, 9);
    sample_candidate *grown = sample_arena_reserve(&arena, first_cap + 1);
    const bool ok = first != NULL && reuse == first && grown != NULL &&
                    arena.cap > first_cap;
    free(arena.v);
    return ok ? 0 : 1;
}
#endif

static int sample_candidate_cmp_desc(const void *a, const void *b) {
    const sample_candidate *ca = a;
    const sample_candidate *cb = b;
    const int logit_order =
        (cb->logit > ca->logit) - (cb->logit < ca->logit);
    if (logit_order != 0) return logit_order;
    return (ca->id > cb->id) - (ca->id < cb->id);
}

static bool sample_candidate_gt(sample_candidate a, sample_candidate b) {
    if (a.logit != b.logit) return a.logit > b.logit;
    return a.id < b.id;
}

static void sample_heap_sift_up(sample_candidate *heap, uint32_t idx) {
    while (idx > 0) {
        const uint32_t parent = (idx - 1u) / 2u;
        if (!sample_candidate_gt(heap[parent], heap[idx])) break;
        sample_candidate tmp = heap[parent];
        heap[parent] = heap[idx];
        heap[idx] = tmp;
        idx = parent;
    }
}

static void sample_heap_sift_down(sample_candidate *heap, uint32_t n, uint32_t idx) {
    for (;;) {
        const uint32_t left = idx * 2u + 1u;
        const uint32_t right = left + 1u;
        uint32_t smallest = idx;
        if (left < n && sample_candidate_gt(heap[smallest], heap[left])) {
            smallest = left;
        }
        if (right < n && sample_candidate_gt(heap[smallest], heap[right])) {
            smallest = right;
        }
        if (smallest == idx) break;
        sample_candidate tmp = heap[idx];
        heap[idx] = heap[smallest];
        heap[smallest] = tmp;
        idx = smallest;
    }
}

static bool sample_fast_top_p(
        const float *logits,
        uint32_t     n_vocab,
        uint32_t     finite,
        float        max_logit,
        int          best,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng,
        int         *token_out) {
    enum { SAMPLE_FAST_TOP_P_CAP = 512 };
    if (!logits || !rng || !token_out || finite == 0) return false;
    if (finite > SAMPLE_FAST_TOP_P_CAP && top_p >= 0.999f) return false;

    const uint32_t cap = finite < SAMPLE_FAST_TOP_P_CAP ?
        finite : (uint32_t)SAMPLE_FAST_TOP_P_CAP;
    sample_candidate heap[SAMPLE_FAST_TOP_P_CAP];
    uint32_t n = 0;
    float sum = 0.0f;
    float heap_sum = 0.0f;

    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        sum += p;
        sample_candidate cand = {.id = (int)i, .logit = v, .prob = p};
        if (n < cap) {
            heap[n] = cand;
            heap_sum += p;
            sample_heap_sift_up(heap, n);
            n++;
        } else if (sample_candidate_gt(cand, heap[0])) {
            heap_sum -= heap[0].prob;
            heap[0] = cand;
            heap_sum += p;
            sample_heap_sift_down(heap, n, 0);
        }
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        *token_out = best;
        return true;
    }

    if (n < finite && heap_sum < top_p * sum) {
        return false;
    }

    qsort(heap, n, sizeof(heap[0]), sample_candidate_cmp_desc);
    const float min_prob = (heap[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    const float min_prob_raw = heap[0].prob * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    bool stopped_by_min_p = false;
    for (uint32_t i = 0; i < n; i++) {
        const float p = heap[i].prob / sum;
        if (i > 0 && p < min_prob) {
            stopped_by_min_p = true;
            break;
        }
        filtered_sum += heap[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (n < finite &&
        stopped_by_min_p &&
        min_p > 0.0f &&
        heap[n - 1u].prob >= min_prob_raw) {
        return false;
    }
    if (filtered == 0) {
        *token_out = best;
        return true;
    }

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= heap[i].prob;
        if (r <= 0.0f) {
            *token_out = heap[i].id;
            return true;
        }
    }
    *token_out = heap[filtered - 1u].id;
    return true;
}

static int sample_full_vocab(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng,
        float       *prob_scratch,
        sample_arena *cands) {
    float max_logit = DS4_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    int fast_token = best;
    if (top_p < 1.0f &&
        sample_fast_top_p(logits,
                          n_vocab,
                          finite,
                          max_logit,
                          best,
                          temperature,
                          top_p,
                          min_p,
                          rng,
                          &fast_token)) {
        return fast_token;
    }

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        if (min_rel > 1.0f) return best;

        /* Find a conservative log-space rejection boundary using the same
         * expf implementation as the probability path. Values below this
         * boundary are guaranteed to fail min-p, avoiding an expf for the
         * overwhelming majority of a large vocabulary. Near-boundary values
         * still take the ordinary expf comparison. */
        float reject_scaled = DS4_NEG_INF;
        bool have_reject_scaled = false;
        if (min_rel > 0.0f && isfinite(min_rel)) {
            float cutoff = logf(min_rel);
            for (int i = 0; i < 8 && isfinite(cutoff); i++) {
                cutoff = nextafterf(cutoff, -FLT_MAX);
                if (expf(cutoff) < min_rel) {
                    reject_scaled = cutoff;
                    have_reject_scaled = true;
                    break;
                }
            }
        }

        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            prob_scratch[i] = -1.0f;
            if (!isfinite(v)) continue;
            const float scaled = (v - max_logit) / temperature;
            if (have_reject_scaled && scaled <= reject_scaled) continue;
            const float p = expf(scaled);
            if (p < min_rel) continue;
            prob_scratch[i] = p;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = sample_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float p = prob_scratch[i];
            if (p < 0.0f) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    uint32_t n = 0;
    float sum = 0.0f;
    sample_candidate *cand = NULL;
    if (min_p > 0.0f && min_p <= 1.0f) {
        /* The later min-p comparison is equivalent to
         * exp((logit-max)/temperature) >= min_p; its normalization cancels.
         * Still compute the full softmax sum in the original order, then sort
         * only candidates that can survive. This preserves the nucleus mass
         * and RNG semantics while avoiding a full-vocabulary qsort. */
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            prob_scratch[i] = -1.0f;
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            prob_scratch[i] = p;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;

        const float min_prob = (1.0f / sum) * min_p;
        cand = sample_arena_reserve(cands, finite);
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float p = prob_scratch[i];
            if (p < 0.0f || p / sum < min_prob) continue;
            cand[n++] = (sample_candidate){
                .id = (int)i, .logit = logits[i], .prob = p
            };
        }
        if (n == 0) return best;
    } else {
        cand = sample_arena_reserve(cands, finite);
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = p};
            sum += p;
        }
    }
    if (sum <= 0.0f || !isfinite(sum)) return best;

    qsort(cand, n, sizeof(cand[0]), sample_candidate_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) return best;

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) return cand[i].id;
    }
    return cand[filtered - 1].id;
}

static int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng,
        float       *prob_scratch,
        sample_arena *cands) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) {
        const bool owned_scratch = prob_scratch == NULL;
        if (owned_scratch) {
            prob_scratch = xmalloc((size_t)n_vocab * sizeof(prob_scratch[0]));
        }
        const int token = sample_full_vocab(logits, n_vocab, temperature,
                                            top_p, min_p, rng, prob_scratch,
                                            cands);
        if (owned_scratch) free(prob_scratch);
        return token;
    }
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    int ids[1024];
    float vals[1024];
    int n = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == top_k && v <= vals[n - 1]) continue;
        int j = n < top_k ? n++ : n - 1;
        while (j > 0 && vals[j - 1] < v) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = v;
        ids[j] = (int)i;
    }
    if (n == 0) return sample_argmax(logits, n_vocab);

    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = sample_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}

#ifdef DS4_TEST_HOOKS
int ds4_test_sample_logits(const float *logits, uint32_t n_vocab,
                           float temperature, int top_k,
                           float top_p, float min_p, uint64_t *rng,
                           float *prob_scratch) {
    if (!logits || !rng || n_vocab == 0) return -1;
    sample_arena cands = {0};
    const int token = sample_top_p_min_p(logits, n_vocab, temperature, top_k,
                                         top_p, min_p, rng, prob_scratch,
                                         &cands);
    free(cands.v);
    return token;
}

int ds4_test_argmax_excluding_logits(const float *logits, uint32_t n_vocab,
                                     int excluded_id) {
    if (!logits) return -1;
    if (getenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == NULL) {
        return argmax_f32_excluding_unrolled8(logits, n_vocab, excluded_id);
    }
    int best = -1;
    float best_v = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        if ((int)i == excluded_id) continue;
        const float v = logits[i];
        if (best < 0 || v > best_v) {
            best = (int)i;
            best_v = v;
        }
    }
    return best;
}
#endif

#ifndef DS4_NO_GPU
static void print_top_logits(
        FILE          * fp,
        const char    * label,
        const ds4_vocab * vocab,
        const float   * logits,
        uint32_t        n_vocab,
        int             k) {
    int best[16];
    if (k > 16) k = 16;
    for (int i = 0; i < k; i++) best[i] = -1;

    for (uint32_t i = 0; i < n_vocab; i++) {
        for (int j = 0; j < k; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int l = k - 1; l > j; l--) best[l] = best[l - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    fprintf(fp, "ds4: top logits %s:\n", label);
    for (int i = 0; i < k && best[i] >= 0; i++) {
        const int id = best[i];
        fprintf(fp, "  %2d %7d % .9g  ", i, id, logits[id]);
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%.*s", (int)vocab->token[id].len, vocab->token[id].ptr);
        }
        fputc('\n', fp);
    }
}
#endif

#ifndef DS4_NO_GPU

/* Laguna has a conventional residual stream and KV cache, but alternates
 * full-attention and sliding-window layers with different query geometry. */
typedef struct {
    ds4_gpu_tensor *features;
    const uint32_t *target_layers;
    uint32_t n_aux;
    uint32_t src_row0;
    uint32_t dst_row0;
    uint32_t n_rows;
} ds4_laguna_feature_capture;

#ifdef __APPLE__
static void laguna_graph_report_q8_lmhead_screen(
        const ds4_laguna_gpu_graph *g);
#endif

/* The lgn_graph module owns only the base target storage.  Keep extension
 * teardown here: speculative verifier scratch, the optional Metal lm-head
 * screen, and diagnostic/selector state all have command/lifetime semantics
 * that must remain coupled to the session scheduler. */
static void laguna_graph_free(ds4_laguna_gpu_graph *g) {
    if (!g) return;
#define DS4_LAGUNA_FREE(name) do { \
        ds4_gpu_tensor_free(g->name); \
        g->name = NULL; \
    } while (0)
#ifdef __APPLE__
    ds4_gpu_laguna_q8_lmhead_screen_destroy(g->lmhead_screen);
    g->lmhead_screen = NULL;
#endif
    DS4_LAGUNA_FREE(spec_output_norm);
    DS4_LAGUNA_FREE(spec_logits);
    DS4_LAGUNA_FREE(spec_argmax);
#undef DS4_LAGUNA_FREE
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(g->spec_key_backup[il]);
        ds4_gpu_tensor_free(g->spec_value_backup[il]);
        g->spec_key_backup[il] = NULL;
        g->spec_value_backup[il] = NULL;
    }
    lgn_graph_free(g);
    memset(g, 0, sizeof(*g));
}

static bool laguna_graph_alloc(ds4_laguna_gpu_graph *g, uint32_t ctx_size) {
    if (!g || ctx_size == 0 || ctx_size > DS4_CONTEXT_LENGTH ||
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_LAGUNA) {
        return false;
    }
#ifdef __APPLE__
    if (!laguna_metal_swa_gqa9_preflight(NULL,
                                         "Laguna graph allocation",
                                         NULL, 0)) return false;
#endif
    memset(g, 0, sizeof(*g));
    if (lgn_graph_alloc(g, ctx_size, &g_ds4_shape)) return true;

    fprintf(stderr, "ds4: failed to allocate Laguna GPU graph\n");
    laguna_graph_free(g);
    return false;
}

static bool laguna_graph_ensure_spec_scratch(ds4_laguna_gpu_graph *g) {
    if (!g) return false;
    const uint64_t rows = DS4_DFLASH_BLOCK_SIZE;
    if (!g->spec_output_norm) {
        g->spec_output_norm = ds4_gpu_tensor_alloc(
            rows * DS4_N_EMBD * sizeof(float));
    }
    if (!g->spec_logits) {
        g->spec_logits = ds4_gpu_tensor_alloc(
            rows * DS4_N_VOCAB * sizeof(float));
    }
    if (!g->spec_argmax) {
        g->spec_argmax =
            ds4_gpu_tensor_alloc(rows * sizeof(int32_t));
    }
    if (!g->spec_output_norm || !g->spec_logits || !g->spec_argmax) {
        fprintf(stderr,
                "ds4: failed to allocate Laguna speculative verifier scratch\n");
        return false;
    }

    const uint64_t row_bytes =
        (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
    const uint64_t backup_bytes = rows * row_bytes;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!ds4_laguna_layer_is_swa(il)) continue;
        if (!g->spec_key_backup[il]) {
            g->spec_key_backup[il] =
                ds4_gpu_tensor_alloc(backup_bytes);
        }
        if (!g->spec_value_backup[il]) {
            g->spec_value_backup[il] =
                ds4_gpu_tensor_alloc(backup_bytes);
        }
        if (!g->spec_key_backup[il] || !g->spec_value_backup[il]) {
            fprintf(stderr,
                    "ds4: failed to allocate Laguna speculative KV backup\n");
            return false;
        }
    }
    return true;
}

static bool laguna_graph_capture_feature(
        const ds4_laguna_feature_capture *capture,
        const ds4_gpu_tensor             *src,
        uint32_t                          target_layer) {
    if (!capture) return true;
    if (!capture->features || !capture->target_layers ||
        capture->n_aux == 0u || capture->n_rows == 0u) {
        return false;
    }
    for (uint32_t aux = 0; aux < capture->n_aux; aux++) {
        if (capture->target_layers[aux] != target_layer) continue;
        return ds4_gpu_dflash_capture_rows_tensor(
                   capture->features,
                   src,
                   capture->src_row0,
                   capture->dst_row0,
                   capture->n_rows,
                   DS4_N_EMBD,
                   capture->n_aux,
                   aux) != 0;
    }
    return true;
}

/* Copy a run of sliding-window ring rows in at most two transfers per layer
 * instead of one per row.  A speculative block is short and the ring wraps at
 * most once inside it, and the per-transfer cost dominates at 2 KiB a row:
 * doing this per row cost more than the draft pass it protects. */
static bool laguna_graph_spec_ring_copy(
        ds4_gpu_tensor *cache,
        ds4_gpu_tensor *backup,
        uint32_t        pos0,
        uint32_t        first_row,
        uint32_t        n_rows,
        uint32_t        cache_cap,
        uint64_t        row_bytes,
        bool            to_backup) {
    if (!cache || !backup || cache_cap == 0u) return false;
    uint32_t row = first_row;
    while (row < n_rows) {
        const uint32_t ring = (pos0 + row) % cache_cap;
        uint32_t run = n_rows - row;
        if (run > cache_cap - ring) run = cache_cap - ring;
        const uint64_t bytes = (uint64_t)run * row_bytes;
        const uint64_t cache_off = (uint64_t)ring * row_bytes;
        const uint64_t backup_off = (uint64_t)row * row_bytes;
        const int ok = to_backup ?
            ds4_gpu_tensor_copy(backup, backup_off, cache, cache_off, bytes) :
            ds4_gpu_tensor_copy(cache, cache_off, backup, backup_off, bytes);
        if (!ok) return false;
        row += run;
    }
    return true;
}

static bool laguna_graph_spec_snapshot(
        ds4_laguna_gpu_graph *g,
        uint32_t              pos0,
        uint32_t              n_rows) {
    if (!g || n_rows == 0u || n_rows > DS4_DFLASH_BLOCK_SIZE ||
        !laguna_graph_ensure_spec_scratch(g)) {
        return false;
    }
    const uint64_t row_bytes =
        (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (!ds4_laguna_layer_is_swa(il)) continue;
        ok = laguna_graph_spec_ring_copy(g->key_cache[il],
                                         g->spec_key_backup[il],
                                         pos0, 0, n_rows,
                                         g->cache_cap[il],
                                         row_bytes, true) &&
             laguna_graph_spec_ring_copy(g->value_cache[il],
                                         g->spec_value_backup[il],
                                         pos0, 0, n_rows,
                                         g->cache_cap[il],
                                         row_bytes, true);
    }
    /* The caller must commit this snapshot before allowing any speculative
     * batch to be discarded.  Otherwise a later restore could read a backup
     * whose copy was discarded together with the speculative work. */
    if (!ok && ds4_gpu_commands_active()) {
        (void)ds4_gpu_discard_commands();
    }
    return ok;
}

/* Low-level restore ownership is intentionally nonblocking on Apple: callers
 * that need a host-state certification must wait its submitted CB before
 * publishing state.  The DFlash product boundary below provides that proof. */
static bool laguna_graph_spec_restore(
        ds4_laguna_gpu_graph *g,
        uint32_t              pos0,
        uint32_t              first_row,
        uint32_t              n_rows) {
    if (!g || first_row > n_rows || n_rows > DS4_DFLASH_BLOCK_SIZE) {
        return false;
    }
    if (first_row == n_rows) return true;
    const uint64_t row_bytes =
        (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (!ds4_laguna_layer_is_swa(il)) continue;
        ok = laguna_graph_spec_ring_copy(g->key_cache[il],
                                         g->spec_key_backup[il],
                                         pos0, first_row, n_rows,
                                         g->cache_cap[il],
                                         row_bytes, false) &&
             laguna_graph_spec_ring_copy(g->value_cache[il],
                                         g->spec_value_backup[il],
                                         pos0, first_row, n_rows,
                                         g->cache_cap[il],
                                         row_bytes, false);
    }
    if (ds4_gpu_commands_active()) {
        if (ok) {
#if defined(__APPLE__)
            if (ds4_gpu_submit_commands() == 0) ok = false;
#else
            if (ds4_gpu_end_commands() == 0) ok = false;
#endif
        } else if (ds4_gpu_discard_commands() == 0) {
            ok = false;
        }
    }
    return ok;
}

static bool laguna_graph_read_spec_logits(
        const ds4_laguna_gpu_graph *g,
        uint32_t                    row,
        float                      *logits_out) {
    if (!g || !g->spec_logits || !logits_out ||
        row >= DS4_DFLASH_BLOCK_SIZE) {
        return false;
    }
    const uint64_t row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    return ds4_gpu_tensor_read(g->spec_logits,
                               (uint64_t)row * row_bytes,
                               logits_out,
                               row_bytes) != 0;
}

static bool laguna_graph_matmul(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        const ds4_tensor     *weight,
        const ds4_gpu_tensor *x,
        uint64_t              n_tokens) {
    if (!out || !model || !weight || !x || weight->ndim < 2) return false;
    if (weight->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out,
                                         model->map,
                                         model->size,
                                         weight->abs_offset,
                                         weight->dim[0],
                                         weight->dim[1],
                                         x,
                                         n_tokens) != 0;
    }
    if (weight->type == DS4_TENSOR_Q6_K) {
        return ds4_gpu_matmul_q6_K_tensor(out,
                                          model->map,
                                          model->size,
                                          weight->abs_offset,
                                          weight->dim[0],
                                          weight->dim[1],
                                          x,
                                          n_tokens) != 0;
    }
    return ds4_gpu_matmul_quant_tensor(out,
                                       model->map,
                                       model->size,
                                       weight->abs_offset,
                                       weight->type,
                                       weight->dim[0],
                                       weight->dim[1],
                                       x,
                                       n_tokens) != 0;
}

static bool laguna_graph_matmul_decode_rows(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        const ds4_tensor     *weight,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows,
        bool                  exact_q8_rows) {
    if (exact_q8_rows && weight && weight->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                   out,
                   model->map,
                   model->size,
                   weight->abs_offset,
                   weight->dim[0],
                   weight->dim[1],
                   x,
                   n_rows) != 0;
    }
    return laguna_graph_matmul(out, model, weight, x, n_rows);
}

/*
 * Laguna S 2.1 has one leading dense FFN layer. Its Q8_0 gate and up
 * projections consume the same normalized row, so Metal can derive the
 * SwiGLU mid row while it is still in the reduction threadgroup. Keep this
 * arm opt-in until an M5/M3 benchmark certifies it; in particular, do not
 * let an unrelated environment value change the ordinary decode path.
 *
 * The selector is intentionally strict. Unset, empty, and "0" mean off;
 * only the literal "1" requests the fused route. Any other value is an
 * error and is rejected before a graph command or KV mutation is opened.
 */
#define DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU \
    "DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU"
#define DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_TRACE \
    "DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_TRACE"

static int laguna_metal_dense_q8_gate_up_swiglu_mode(void) {
    const char *value =
        getenv(DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU);
    const int mode = ds4_gpu_laguna_dense_q8_gate_up_swiglu_env_mode(value);
    if (mode >= 0) return mode;
    fprintf(stderr,
            "ds4: invalid %s='%s'; expected unset, empty, 0, or literal 1\n",
            DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU,
            value ? value : "");
    return -1;
}

static bool laguna_dense_q8_gate_up_swiglu_weights_eligible(
        const ds4_weights *weights) {
    if (!weights || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_LAGUNA ||
        DS4_N_LEADING_DENSE == 0u) {
        return false;
    }
    for (uint32_t il = 0; il < DS4_N_LEADING_DENSE; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!layer->ffn_gate || !layer->ffn_up ||
            layer->ffn_gate->type != DS4_TENSOR_Q8_0 ||
            layer->ffn_up->type != DS4_TENSOR_Q8_0 ||
            layer->ffn_gate->ndim < 2 || layer->ffn_up->ndim < 2 ||
            layer->ffn_gate->dim[0] != DS4_N_EMBD ||
            layer->ffn_gate->dim[1] != DS4_N_FF_DENSE ||
            layer->ffn_up->dim[0] != DS4_N_EMBD ||
            layer->ffn_up->dim[1] != DS4_N_FF_DENSE ||
            layer->ffn_gate->dim[0] != 3072u ||
            layer->ffn_gate->dim[1] != 12288u ||
            layer->ffn_up->dim[0] != 3072u ||
            layer->ffn_up->dim[1] != 12288u) {
            return false;
        }
    }
    return true;
}

static bool laguna_dense_q8_gate_up_swiglu_model_ranges_valid(
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!model || !model->map || model->size == 0 || !weights) return false;
    if ((DS4_N_EMBD % 32u) != 0u || DS4_N_FF_DENSE == 0u) return false;

    const uint64_t blocks = (uint64_t)DS4_N_EMBD / 32u;
    if (blocks > UINT64_MAX / 34u) return false;
    const uint64_t row_bytes = blocks * 34u;
    if ((uint64_t)DS4_N_FF_DENSE > UINT64_MAX / row_bytes) return false;
    const uint64_t weight_bytes = (uint64_t)DS4_N_FF_DENSE * row_bytes;

    for (uint32_t il = 0; il < DS4_N_LEADING_DENSE; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        const ds4_tensor *tensors[] = {layer->ffn_gate, layer->ffn_up};
        for (size_t ti = 0; ti < sizeof(tensors) / sizeof(tensors[0]); ti++) {
            const ds4_tensor *tensor = tensors[ti];
            if (!tensor || tensor->bytes != weight_bytes ||
                tensor->abs_offset > model->size ||
                weight_bytes > model->size - tensor->abs_offset) {
                return false;
            }
        }
    }
    return true;
}

/* Trace is intentionally independent from the broad profiling environment.
 * Resolve it only after the explicit fused selector has requested a
 * production route; selector-off inference therefore performs no trace env
 * lookup, and timed enabled inference pays one literal lookup per process. */
static bool laguna_dense_q8_gate_up_swiglu_trace_requested(void) {
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized) {
        const char *value = getenv(
            DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_TRACE);
        enabled = value && value[0] == '1' && value[1] == '\0';
        initialized = 1;
    }
    return enabled;
}

static bool laguna_dense_q8_gate_up_swiglu_trace_enabled;

static bool laguna_dense_q8_gate_up_swiglu_preflight(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool              *enabled_out) {
    if (enabled_out) *enabled_out = false;
    const int mode = laguna_metal_dense_q8_gate_up_swiglu_mode();
    if (mode < 0) {
        laguna_dense_q8_gate_up_swiglu_trace_enabled = false;
        return false;
    }
    if (mode == 0) {
        laguna_dense_q8_gate_up_swiglu_trace_enabled = false;
        return true;
    }

    laguna_dense_q8_gate_up_swiglu_trace_enabled =
        laguna_dense_q8_gate_up_swiglu_trace_requested();

    const bool weights_eligible =
        laguna_dense_q8_gate_up_swiglu_weights_eligible(weights);
    if (!weights_eligible) {
        fprintf(stderr,
                "ds4: %s requested but Laguna leading dense gate/up "
                "weights are not Q8_0 [3072,12288]; refusing stock fallback\n",
                DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU);
        return false;
    }
#if defined(__APPLE__)
    ds4_gpu_q8_decode_config q8_config;
    const int q8_config_valid =
        ds4_gpu_q8_decode_config_snapshot(&q8_config);
    const int q8_rows_mode = q8_config_valid < 0 ? -1 : q8_config.q8_mv_rows;
    const char *q8_rows_value = q8_config_valid < 0 ? "invalid snapshot" :
        (q8_rows_mode == 2 ? "2" : "4");
#else
    const char *q8_rows_value = getenv("DS4_METAL_Q8_MV_ROWS");
    const int q8_rows_mode =
        ds4_gpu_laguna_dense_q8_gate_up_swiglu_rows_env_mode(q8_rows_value);
#endif
    if (q8_rows_mode != 2) {
        fprintf(stderr,
                "ds4: %s requires DS4_METAL_Q8_MV_ROWS unset/2 for "
                "the fused decode topology (got '%s'); refusing stock "
                "fallback\n",
                DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU,
                q8_rows_value ? q8_rows_value : "");
        return false;
    }
    const bool ranges_valid =
        laguna_dense_q8_gate_up_swiglu_model_ranges_valid(model, weights);
    if (!ranges_valid) {
        fprintf(stderr,
                "ds4: %s requested but a Laguna leading dense Q8_0 mapped "
                "range is missing or outside the model; refusing stock fallback\n",
                DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU);
        return false;
    }
#ifdef __APPLE__
    const bool mid_ready =
        ds4_gpu_shared_mid_swiglu_q8_0_available() != 0;
#else
    const bool mid_ready = false;
#endif
    const int decision =
        ds4_gpu_laguna_dense_q8_gate_up_swiglu_preflight_decision(
            mode,
            weights_eligible ? 1 : 0,
            ranges_valid ? 1 : 0,
            q8_rows_mode,
            mid_ready ? 1 : 0);
#ifdef __APPLE__
    if (decision != 1) {
        fprintf(stderr,
                "ds4: %s requested but the Metal Q8 gate/up+SwiGLU "
                "decode pipeline is unavailable (mid=%d); "
                "refusing stock fallback\n",
                DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU,
                mid_ready ? 1 : 0);
        return false;
    }
#else
    (void)decision;
    fprintf(stderr,
            "ds4: %s requested but this build has no Metal fused "
            "gate/up+SwiGLU pipeline\n",
            DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU);
    return false;
#endif
    if (enabled_out) *enabled_out = true;
    return true;
}

#ifdef __APPLE__
/*
 * Opt-in batched sibling of the fused decode route for ordinary prefill.
 * The batched kernel computes both dense Q8 projections for the whole row
 * block in one tiled pass and applies SwiGLU in the tile epilogue, so the
 * normed rows are read once and the gate/up rows never reach device memory.
 * Same strict selector discipline as the decode route: only a literal "1"
 * requests it, exact verifier rows stay on the stock path, and an explicit
 * request that cannot be honored fails before any graph mutation.
 */
#define DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_BATCH \
    "DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_BATCH"

static bool laguna_dense_q8_gate_up_swiglu_batch_preflight(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool              *enabled_out) {
    if (enabled_out) *enabled_out = false;
    const char *value =
        getenv(DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_BATCH);
    const int mode = ds4_gpu_laguna_dense_q8_gate_up_swiglu_env_mode(value);
    if (mode < 0) {
        fprintf(stderr,
                "ds4: invalid %s='%s'; expected unset, empty, 0, or literal 1\n",
                DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_BATCH,
                value ? value : "");
        return false;
    }
    if (mode == 0) return true;
    if (!laguna_dense_q8_gate_up_swiglu_weights_eligible(weights) ||
        !laguna_dense_q8_gate_up_swiglu_model_ranges_valid(model, weights)) {
        fprintf(stderr,
                "ds4: %s requested but Laguna leading dense gate/up "
                "weights are not mapped Q8_0 [3072,12288]; refusing stock "
                "fallback\n",
                DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_BATCH);
        return false;
    }
#ifdef __APPLE__
    if (ds4_gpu_laguna_dense_q8_gate_up_swiglu_batch_available() == 0) {
        fprintf(stderr,
                "ds4: %s requested but the Metal batched Q8 gate/up+SwiGLU "
                "pipeline is unavailable; refusing stock fallback\n",
                DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_BATCH);
        return false;
    }
    if (enabled_out) *enabled_out = true;
    return true;
#else
    fprintf(stderr,
            "ds4: %s requested but this build has no Metal batched fused "
            "gate/up+SwiGLU pipeline\n",
            DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_BATCH);
    return false;
#endif
}
#endif /* __APPLE__ */

static laguna_dense_q8_gate_up_swiglu_counters
    laguna_dense_q8_gate_up_swiglu_completed;
static laguna_dense_q8_gate_up_swiglu_counters
    laguna_dense_q8_gate_up_swiglu_reported;

static void laguna_dense_q8_gate_up_swiglu_pending_clear(
        ds4_laguna_gpu_graph *g) {
    if (!g || !laguna_dense_q8_gate_up_swiglu_trace_enabled) return;
    if (g->dense_q8_pending.decode_mid_fused == 0u &&
        g->dense_q8_pending.ordinary_prefill_stock == 0u) {
        return;
    }
    memset(&g->dense_q8_pending, 0, sizeof(g->dense_q8_pending));
}

static void laguna_dense_q8_gate_up_swiglu_note_decode_mid(
        ds4_laguna_gpu_graph *g) {
    if (!laguna_dense_q8_gate_up_swiglu_trace_enabled) return;
    if (!g) return;
    g->dense_q8_pending.decode_mid_fused++;
}

static void laguna_dense_q8_gate_up_swiglu_note_prefill_stock(
        ds4_laguna_gpu_graph *g) {
    if (!laguna_dense_q8_gate_up_swiglu_trace_enabled) return;
    if (!g) return;
    g->dense_q8_pending.ordinary_prefill_stock++;
}

/* The backend APIs return after encoding into an owned graph command buffer;
 * the caller may still have an active/pending Metal graph.  Only call this
 * after a waited graph boundary, and say so explicitly in the diagnostic. */
static void laguna_dense_q8_gate_up_swiglu_report_waited(
        ds4_laguna_gpu_graph *g) {
    if (!laguna_dense_q8_gate_up_swiglu_trace_enabled) return;
    if (!g) return;
    const laguna_dense_q8_gate_up_swiglu_counters pending =
        g->dense_q8_pending;
    if (pending.decode_mid_fused == 0u &&
        pending.ordinary_prefill_stock == 0u) {
        return;
    }
    laguna_dense_q8_gate_up_swiglu_pending_clear(g);
    laguna_dense_q8_gate_up_swiglu_completed.decode_mid_fused +=
        pending.decode_mid_fused;
    laguna_dense_q8_gate_up_swiglu_completed.ordinary_prefill_stock +=
        pending.ordinary_prefill_stock;
    const laguna_dense_q8_gate_up_swiglu_counters current =
        laguna_dense_q8_gate_up_swiglu_completed;
    const laguna_dense_q8_gate_up_swiglu_counters reported =
        laguna_dense_q8_gate_up_swiglu_reported;
    const bool new_decode = current.decode_mid_fused != 0u &&
        reported.decode_mid_fused == 0u;
    const bool new_prefill = current.ordinary_prefill_stock != 0u &&
        reported.ordinary_prefill_stock == 0u;
    if (!new_decode && !new_prefill) {
        return;
    }
    fprintf(stderr,
            "ds4: Laguna dense Q8 path counters "
            "decode_mid_fused=%llu ordinary_prefill_stock=%llu route=%s "
            "completion=waited "
            "encoded=successful\n",
            (unsigned long long)current.decode_mid_fused,
            (unsigned long long)current.ordinary_prefill_stock,
            new_decode ? "decode_mid_fused" : "ordinary_prefill_stock");
    laguna_dense_q8_gate_up_swiglu_reported = current;
}

static bool laguna_graph_routed_moe_decode_rows(
        ds4_laguna_gpu_graph   *g,
        const ds4_model        *model,
        const ds4_layer_weights *l,
        uint32_t                layer,
        uint32_t                n_rows,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes) {
    if (!g || !model || !l || n_rows == 0u) return false;
    const uint64_t embd_bytes =
        (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t mid_elems =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t mid_bytes = mid_elems * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t weight_bytes =
        (uint64_t)DS4_N_EXPERT_USED * sizeof(float);

#if defined(__APPLE__)
    if (l->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        l->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
        l->ffn_down_exps->type == DS4_TENSOR_Q4_K) {
        return ds4_gpu_glm_routed_moe_batch_decode_exact_q4_tensor(
                   g->ffn_out,
                   g->routed_mid,
                   model->map,
                   model->size,
                   l->ffn_gate_exps->abs_offset,
                   l->ffn_up_exps->abs_offset,
                   l->ffn_down_exps->abs_offset,
                   l->ffn_gate_exps->type,
                   l->ffn_up_exps->type,
                   l->ffn_down_exps->type,
                   gate_expert_bytes,
                   gate_row_bytes,
                   up_expert_bytes,
                   up_row_bytes,
                   down_expert_bytes,
                   down_row_bytes,
                   DS4_N_EMBD,
                   DS4_N_FF_EXP,
                   DS4_N_EMBD,
                   g->router_selected,
                   g->router_weights,
                   DS4_N_EXPERT,
                   DS4_N_EXPERT_USED,
                   layer,
                   g->ffn_norm,
                   n_rows,
                   (uint32_t)mid_elems) != 0;
    }
    if ((l->ffn_gate_exps->type == DS4_TENSOR_Q2_K ||
         l->ffn_gate_exps->type == DS4_TENSOR_Q3_K) &&
        l->ffn_up_exps->type == l->ffn_gate_exps->type &&
        l->ffn_down_exps->type == l->ffn_gate_exps->type &&
        n_rows <= DS4_DFLASH_BLOCK_SIZE) {
        /*
         * The Q2_K/Q3_K batch dispatch uses the same independent per-token
         * kernels as one-token decode. Encoding all verifier rows together
         * removes repeated encoder boundaries without changing reductions.
         */
        return ds4_gpu_glm_routed_moe_batch_decode_exact_q2_q3_tensor(
                   g->ffn_out,
                   g->routed_mid,
                   model->map,
                   model->size,
                   l->ffn_gate_exps->abs_offset,
                   l->ffn_up_exps->abs_offset,
                   l->ffn_down_exps->abs_offset,
                   l->ffn_gate_exps->type,
                   l->ffn_up_exps->type,
                   l->ffn_down_exps->type,
                   gate_expert_bytes,
                   gate_row_bytes,
                   up_expert_bytes,
                   up_row_bytes,
                   down_expert_bytes,
                   down_row_bytes,
                   DS4_N_EMBD,
                   DS4_N_FF_EXP,
                   DS4_N_EMBD,
                   g->router_selected,
                   g->router_weights,
                   DS4_N_EXPERT,
                   DS4_N_EXPERT_USED,
                   layer,
                   g->ffn_norm,
                   n_rows,
                   (uint32_t)mid_elems) != 0;
    }
#endif
    for (uint32_t row = 0; row < n_rows; row++) {
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(
            g->ffn_out, (uint64_t)row * embd_bytes, embd_bytes);
        ds4_gpu_tensor *mid = ds4_gpu_tensor_view(
            g->routed_mid, (uint64_t)row * mid_bytes, mid_bytes);
        ds4_gpu_tensor *selected = ds4_gpu_tensor_view(
            g->router_selected,
            (uint64_t)row * selected_bytes,
            selected_bytes);
        ds4_gpu_tensor *weights = ds4_gpu_tensor_view(
            g->router_weights,
            (uint64_t)row * weight_bytes,
            weight_bytes);
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(
            g->ffn_norm, (uint64_t)row * embd_bytes, embd_bytes);
        const int ok =
            out && mid && selected && weights && x &&
            ds4_gpu_glm_routed_moe_one_tensor(
                out,
                mid,
                model->map,
                model->size,
                l->ffn_gate_exps->abs_offset,
                l->ffn_up_exps->abs_offset,
                l->ffn_down_exps->abs_offset,
                l->ffn_gate_exps->type,
                l->ffn_up_exps->type,
                l->ffn_down_exps->type,
                gate_expert_bytes,
                gate_row_bytes,
                up_expert_bytes,
                up_row_bytes,
                down_expert_bytes,
                down_row_bytes,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                DS4_N_EMBD,
                selected,
                weights,
                DS4_N_EXPERT,
                DS4_N_EXPERT_USED,
                layer,
                x,
                true);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(mid);
        ds4_gpu_tensor_free(out);
        if (!ok) return false;
    }
    return true;
}

/* Opt-in fused decode router (logits matvec + SIMD top-k in one dispatch).
 * Literal-1 opt-in like the other DS4_METAL_LAGUNA_* routes; cached
 * process-wide since the router runs once per layer per token.  Returns -1
 * on an invalid value. */
static int laguna_metal_router_decode_fused_mode(void) {
    static int cache = -2;
    if (cache == -2) {
#if defined(__APPLE__)
        const char *value = getenv("DS4_METAL_LAGUNA_ROUTER_DECODE_FUSED");
        if (!value || value[0] == '\0' || strcmp(value, "0") == 0) {
            cache = 0;
        } else if (strcmp(value, "1") == 0) {
            cache = 1;
        } else {
            fprintf(stderr,
                    "ds4: invalid DS4_METAL_LAGUNA_ROUTER_DECODE_FUSED='%s'; "
                    "expected unset, empty, 0, or literal 1\n", value);
            cache = -1;
        }
#else
        cache = 0;
#endif
    }
    return cache;
}

static bool laguna_graph_router_decode_rows(
        ds4_laguna_gpu_graph    *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 n_rows,
        bool                     allow_fused) {
    if (!g || !model || !l || n_rows == 0u) return false;
#if defined(__APPLE__)
    if (allow_fused) {
        /* The graph boundary already froze and preflighted this selector;
         * keep the per-layer helper free of another getenv/mode admission. */
        const int fused_router = laguna_metal_router_decode_fused_mode();
        if (fused_router < 0) return false;
        if (fused_router > 0 && n_rows == 1u) {
            return ds4_gpu_laguna_router_decode_fused_tensor(
                       g->router_selected,
                       g->router_weights,
                       g->router_probs,
                       g->router_logits,
                       model->map,
                       model->size,
                       l->ffn_gate_inp->abs_offset,
                       l->ffn_exp_probs_b->abs_offset,
                       g->ffn_norm,
                       (uint32_t)DS4_N_EMBD,
                       DS4_N_EXPERT,
                       DS4_N_EXPERT_USED,
                       DS4_EXPERT_WEIGHT_SCALE) != 0;
        }
    }
#endif
#if defined(__APPLE__)
    return ds4_gpu_matmul_f32_decode_rows_exact_tensor(
               g->router_logits,
               model->map,
               model->size,
               l->ffn_gate_inp->abs_offset,
               DS4_N_EMBD,
               DS4_N_EXPERT,
               g->ffn_norm,
               n_rows) != 0 &&
           ds4_gpu_glm_router_select_batch_tensor(
               g->router_selected,
               g->router_weights,
               g->router_probs,
               model->map,
               model->size,
               l->ffn_exp_probs_b->abs_offset,
               g->router_logits,
               DS4_N_EXPERT,
               DS4_N_EXPERT_USED,
               DS4_EXPERT_WEIGHT_SCALE,
               n_rows) != 0;
#endif
    const uint64_t embd_bytes =
        (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t expert_f32_bytes =
        (uint64_t)DS4_N_EXPERT * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t selected_weight_bytes =
        (uint64_t)DS4_N_EXPERT_USED * sizeof(float);

    for (uint32_t row = 0; row < n_rows; row++) {
        ds4_gpu_tensor *logits = ds4_gpu_tensor_view(
            g->router_logits,
            (uint64_t)row * expert_f32_bytes,
            expert_f32_bytes);
        ds4_gpu_tensor *probs = ds4_gpu_tensor_view(
            g->router_probs,
            (uint64_t)row * expert_f32_bytes,
            expert_f32_bytes);
        ds4_gpu_tensor *selected = ds4_gpu_tensor_view(
            g->router_selected,
            (uint64_t)row * selected_bytes,
            selected_bytes);
        ds4_gpu_tensor *weights = ds4_gpu_tensor_view(
            g->router_weights,
            (uint64_t)row * selected_weight_bytes,
            selected_weight_bytes);
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(
            g->ffn_norm, (uint64_t)row * embd_bytes, embd_bytes);
        const int ok =
            logits && probs && selected && weights && x &&
            ds4_gpu_matmul_f32_tensor(
                logits,
                model->map,
                model->size,
                l->ffn_gate_inp->abs_offset,
                DS4_N_EMBD,
                DS4_N_EXPERT,
                x,
                1u) &&
            ds4_gpu_glm_router_select_tensor(
                selected,
                weights,
                probs,
                model->map,
                model->size,
                l->ffn_exp_probs_b->abs_offset,
                logits,
                DS4_N_EXPERT,
                DS4_N_EXPERT_USED,
                DS4_EXPERT_WEIGHT_SCALE);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(probs);
        ds4_gpu_tensor_free(logits);
        if (!ok) return false;
    }
    return true;
}

static void dflash_graph_free(ds4_dflash_gpu_graph *g) {
    lgn_dflash_graph_free(g);
}

static bool dflash_graph_alloc(ds4_dflash_gpu_graph *g) {
    if (!g || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_LAGUNA) return false;
#ifdef __APPLE__
    if (!laguna_metal_swa_gqa9_preflight(NULL,
                                         "Laguna DFlash graph allocation",
                                         NULL, 0)) return false;
#endif
    memset(g, 0, sizeof(*g));
    if (lgn_dflash_graph_alloc(g)) return true;
    fprintf(stderr, "ds4: failed to allocate DFlash GPU graph\n");
    dflash_graph_free(g);
    return false;
}

static lgn_dflash_exec_context dflash_graph_exec_context(
        const ds4_engine *e) {
    lgn_dflash_exec_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    if (!e) return ctx;
    ctx.support_map = e->dflash_model.map;
    ctx.support_map_size = e->dflash_model.size;
    ctx.f16_map = e->dflash_f16_map;
    ctx.f16_map_size = e->dflash_f16_map_size;
    ctx.support_weights = &e->dflash_weights;
    ctx.target_context_length = LGN_DFLASH_TARGET_CONTEXT_LENGTH;
    return ctx;
}

/* ds4.c owns admission and the terminal command boundary; the execution
 * module only records support work into the active transaction. */
static bool dflash_graph_encode_inject(
        ds4_dflash_gpu_graph *g,
        const ds4_engine     *e,
        uint32_t              pos0,
        uint32_t              n_rows) {
    if (!g || !e || !e->dflash_ready || n_rows == 0u ||
        n_rows > g->feature_cap) {
        return false;
    }
    const lgn_dflash_exec_context ctx = dflash_graph_exec_context(e);
    if (!lgn_dflash_exec_context_valid(&ctx)) return false;

    bool ok = ds4_gpu_commands_active() != 0;
    if (!ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = lgn_dflash_exec_encode_record(g, &ctx, pos0, n_rows);
    }
    /* The wrapper retains the former admission/terminal behavior.  The
     * record-only module never closes or discards this caller-owned batch. */
    if (ds4_gpu_commands_active()) {
        if (ok) {
            if (ds4_gpu_end_commands() == 0) ok = false;
        } else if (ds4_gpu_discard_commands() == 0) {
            ok = false;
        }
    }
    return ok;
}

#ifdef __APPLE__
static bool laguna_metal_qk_norm_rope_simd32_preflight(void);
static void laguna_metal_qk_norm_rope_simd32_trace_route(
        const char *route,
        uint32_t    n_tokens,
        bool        capture,
        bool        verifier);
static void laguna_metal_qk_norm_rope_simd32_target_evidence_discard(
        ds4_laguna_gpu_graph *g);
static bool laguna_metal_qk_norm_rope_simd32_target_evidence_complete(
        ds4_laguna_gpu_graph *g);
#endif

static bool dflash_graph_commands_active(void) {
    return ds4_gpu_commands_active() != 0;
}

static void dflash_graph_restore_cursors(
        ds4_dflash_gpu_graph *g,
        ds4_gpu_tensor       *saved_cur,
        ds4_gpu_tensor       *saved_next) {
    if (!g) return;
    g->cur = saved_cur;
    g->next = saved_next;
}

static bool dflash_graph_draft_block(
        ds4_dflash_gpu_graph *g,
        const ds4_engine     *e,
        int                   first_token,
        uint32_t              pos0,
        uint32_t              n_draft) {
    if (!g || !e || !e->dflash_ready ||
        first_token < 0 || first_token >= (int)DS4_N_VOCAB ||
        n_draft == 0u || n_draft + 1u > g->block_cap) {
        return false;
    }
    /* The verifier snapshot owns this batch.  Draft recording only appends
     * work to it; it must never create, submit, or discard a caller-owned
     * transaction behind the caller's back. */
    if (!dflash_graph_commands_active()) return false;
    const lgn_dflash_exec_context exec_ctx = dflash_graph_exec_context(e);
    if (!lgn_dflash_exec_context_valid(&exec_ctx)) return false;
    ds4_gpu_tensor *saved_cur = g->cur;
    ds4_gpu_tensor *saved_next = g->next;
    const uint32_t n_rows = n_draft + 1u;
    uint32_t token_ids[DS4_DFLASH_BLOCK_SIZE];
    token_ids[0] = (uint32_t)first_token;
    for (uint32_t row = 1; row < n_rows; row++) {
        token_ids[row] = e->dflash_weights.mask_token_id;
    }
    if (!ds4_gpu_tensor_write(g->tokens,
                              0,
                              token_ids,
                              (uint64_t)n_rows * sizeof(token_ids[0]))) {
        return false;
    }

    const ds4_dflash_weights *w = exec_ctx.support_weights;
    const uint32_t embd = DS4_SHAPE_LAGUNA_S21.n_embd;
    const uint32_t n_head = DS4_SHAPE_LAGUNA_S21.n_head;
    const uint32_t n_head_kv = DS4_SHAPE_LAGUNA_S21.n_head_kv;
    const uint32_t head_dim = DS4_SHAPE_LAGUNA_S21.n_head_dim;
    const uint32_t q_dim = n_head * head_dim;
    const uint32_t kv_dim = n_head_kv * head_dim;
    const uint32_t ff = DS4_SHAPE_LAGUNA_S21.n_ff_dense;
    const void *weight_map = exec_ctx.f16_map ?
        exec_ctx.f16_map : exec_ctx.support_map;
    const uint64_t weight_map_size = exec_ctx.f16_map ?
        exec_ctx.f16_map_size : exec_ctx.support_map_size;

    bool ok = true;
    if (ok) {
        ok = ds4_gpu_embed_tokens_quant_tensor(
                 g->cur,
                 g->tokens,
                 e->model.map,
                 e->model.size,
                 e->weights.token_embd->abs_offset,
                 e->weights.token_embd->type,
                 DS4_N_VOCAB,
                 n_rows,
                 embd) != 0;
    }
    for (uint32_t il = 0; ok && il < DS4_DFLASH_N_LAYER; il++) {
        const ds4_dflash_layer_weights *l = &w->layer[il];
        ok = ds4_gpu_rms_norm_weight_rows_tensor(
                 g->norm,
                 g->cur,
                 weight_map,
                 weight_map_size,
                 l->attn_norm->abs_offset,
                 embd,
                 n_rows,
                 DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
        if (ok) {
            ok = lgn_dflash_exec_matmul(g->q, &exec_ctx, l->attn_q,
                                        g->norm, n_rows) &&
                 lgn_dflash_exec_matmul(g->k, &exec_ctx, l->attn_k,
                                        g->norm, n_rows) &&
                 lgn_dflash_exec_matmul(g->v, &exec_ctx, l->attn_v,
                                        g->norm, n_rows) &&
                 lgn_dflash_exec_matmul(g->gate, &exec_ctx, l->attn_gate,
                                        g->norm, n_rows);
        }
        if (ok) {
            ok = ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                     g->q,
                     g->k,
                     weight_map,
                     weight_map_size,
                     l->attn_q_norm->abs_offset,
                     l->attn_k_norm->abs_offset,
                     n_rows,
                     n_head,
                     n_head_kv,
                     head_dim,
                     DS4_SHAPE_LAGUNA_S21.n_rot_swa,
                     pos0,
                     DS4_CONTEXT_LENGTH,
                     500000.0f,
                     1.0f,
                     0.0f,
                     1.0f,
                     0.0f,
                     0.0f,
                     DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
        }
        if (ok) {
            ok = ds4_gpu_laguna_attention_prefill_tensor(
                     g->heads,
                     g->key_cache[il],
                     g->value_cache[il],
                     g->staged_key,
                     g->staged_value,
                     g->q,
                     g->k,
                     g->v,
                     g->gate,
                     pos0,
                     n_rows,
                     g->cache_cap,
                     n_head,
                     n_head_kv,
                     head_dim,
                     1.0f / sqrtf((float)head_dim),
                     1) != 0;
        }
        if (ok) {
            ok = lgn_dflash_exec_matmul(g->attn_out, &exec_ctx,
                                        l->attn_output, g->heads,
                                        n_rows) &&
                 ds4_gpu_add_rms_norm_weight_rows_tensor(
                     g->ffn_norm,
                     g->after_attn,
                     g->cur,
                     g->attn_out,
                     weight_map,
                     weight_map_size,
                     l->ffn_norm->abs_offset,
                     embd,
                     n_rows,
                     DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
        }
        if (ok) {
            ok = lgn_dflash_exec_matmul(g->ffn_gate, &exec_ctx,
                                        l->ffn_gate, g->ffn_norm,
                                        n_rows) &&
                 lgn_dflash_exec_matmul(g->ffn_up, &exec_ctx,
                                        l->ffn_up, g->ffn_norm,
                                        n_rows);
        }
        if (ok) {
            ok = ds4_gpu_swiglu_tensor(g->ffn_mid,
                                        g->ffn_gate,
                                        g->ffn_up,
                                        (uint64_t)n_rows * ff,
                                        0.0f,
                                        1.0f) != 0;
        }
        if (ok) {
            ok = lgn_dflash_exec_matmul(g->ffn_out, &exec_ctx,
                                        l->ffn_down, g->ffn_mid,
                                        n_rows) &&
                 ds4_gpu_add_tensor(g->next,
                                    g->after_attn,
                                    g->ffn_out,
                                    (uint64_t)n_rows * embd) != 0;
        }
        if (ok) {
            ds4_gpu_tensor *tmp = g->cur;
            g->cur = g->next;
            g->next = tmp;
        }
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(
                 g->output_norm,
                 g->cur,
                 weight_map,
                 weight_map_size,
                 w->output_norm->abs_offset,
                 embd,
                 n_rows,
                 DS4_SHAPE_LAGUNA_S21.rms_eps) != 0;
    }
    if (ok) {
#ifdef __APPLE__
        if (e->weights.output->type == DS4_TENSOR_Q8_0) {
            ok = ds4_gpu_matmul_q8_0_dflash_tensor(
                     g->logits,
                     e->model.map,
                     e->model.size,
                     e->weights.output->abs_offset,
                     e->weights.output->dim[0],
                     e->weights.output->dim[1],
                     g->output_norm,
                     n_rows) != 0;
        } else
#endif
        {
            ok = laguna_graph_matmul(g->logits,
                                     &e->model,
                                     e->weights.output,
                                     g->output_norm,
                                     n_rows);
        }
    }
    if (ok) {
        ok = ds4_gpu_indexer_topk_tensor(g->argmax,
                                          g->logits,
                                          DS4_N_VOCAB,
                                          n_rows,
                                          1) != 0;
    }
    if (ok && e->dflash_p_min > 0.0f) {
        ok = ds4_gpu_dflash_probabilities_tensor(
                 g->probabilities,
                 g->logits,
                 g->argmax,
                 n_rows,
                 DS4_N_VOCAB) != 0;
    }
    if (!ok) {
        /* The caller will discard the active batch.  Restore the host-side
         * ping-pong state before it does so, so a subsequent fallback cannot
         * observe a half-recorded support graph. */
        dflash_graph_restore_cursors(g, saved_cur, saved_next);
    }
    (void)q_dim;
    (void)kv_dim;
    return ok;
}

#ifdef __APPLE__
/* The explicit Q/K SIMD32 and RoPE-atlas selectors are graph contracts, not
 * kernel hints. Validate them before any graph scratch, command buffer, or KV
 * mutation. The Laguna S 2.1 model alternates 48- and 72-query-head layers,
 * with 8 KV heads and 64/128 rotary dimensions for global/SWA layers. */
static bool laguna_metal_qk_norm_rope_simd32_preflight(void) {
    int mode = ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached();
    if (mode == -2) {
        mode = ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                48u, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, 64u);
        mode = ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached();
    }
    int atlas_mode = ds4_gpu_laguna_rope_atlas_plan_mode_cached();
    if (atlas_mode == -2) {
        /* Select and allocate the immutable atlas graph plan once.  The
         * backend owns strict environment parsing; later layer calls consume
         * this cached result instead of reparsing getenv. */
        if (ds4_gpu_laguna_rope_atlas_preflight(
                    48u, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, 64u) < 0) {
            return false;
        }
        atlas_mode = ds4_gpu_laguna_rope_atlas_plan_mode_cached();
    }
    if (mode < 0) return false;
    if (atlas_mode < 0) return false;
    if (mode == 0 && atlas_mode == 0) return true;
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_LAGUNA) {
        fprintf(stderr,
                "ds4: Laguna Q/K norm/RoPE experiment requested outside the "
                "Laguna S 2.1 graph\n");
        return false;
    }
    static const uint32_t q_heads[] = { 48u, 72u };
    static const uint32_t n_rots[] = { 64u, 128u };
    for (size_t hi = 0; hi < sizeof(q_heads) / sizeof(q_heads[0]); hi++) {
        for (size_t ri = 0; ri < sizeof(n_rots) / sizeof(n_rots[0]); ri++) {
            if (mode > 0 &&
                ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                    q_heads[hi], DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                    n_rots[ri]) != 1) {
                return false;
            }
            if (atlas_mode > 0 &&
                ds4_gpu_laguna_rope_atlas_preflight(
                    q_heads[hi], DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                    n_rots[ri]) != 1) {
                return false;
            }
        }
    }
    return true;
}

static void laguna_metal_qk_norm_rope_simd32_trace_route(
        const char *route,
        uint32_t    n_tokens,
        bool        capture,
        bool        verifier) {
    if (!ds4_gpu_laguna_qk_head_norm_rope_simd32_trace_enabled()) return;
    fprintf(stderr,
            "ds4: Laguna Q/K norm/RoPE SIMD32 route=%s tokens=%u "
            "capture=%d verifier=%d\n",
            route ? route : "unknown", n_tokens, capture ? 1 : 0,
            verifier ? 1 : 0);
}

/* Target proof uses an encoded-count delta isolated immediately before the
 * target plus command_waited.  The backend's completion-scoped counter is
 * retained as lifecycle evidence/diagnostics, but a DFlash support command
 * may complete between the target snapshot and target completion.  Support
 * injection is K-only and does not increment the Q/K encoded counter. */
static bool laguna_metal_qk_norm_rope_simd32_target_evidence(
        uint64_t    encoded_before,
        uint64_t    completed_before,
        uint32_t    n_tokens,
        bool        capture,
        bool        verifier,
        const char *route,
        bool        command_waited) {
    if (!command_waited) return false;
    const uint64_t encoded_after =
        ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count();
    const uint64_t encoded_delta = encoded_after >= encoded_before ?
        encoded_after - encoded_before : UINT64_MAX;
    const uint64_t completed_after =
        ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count();
    const uint64_t completed_delta = completed_after >= completed_before ?
        completed_after - completed_before : UINT64_MAX;
    /* The global completion counter is diagnostic only here: a DFlash
     * support command may complete between the target snapshot and target
     * completion.  The encoded target delta remains isolated by the snapshot
     * immediately before this graph's target, while command_waited proves the
     * owning command completed successfully. */
    const bool exact = encoded_delta == 48u;
    if (ds4_gpu_laguna_qk_head_norm_rope_simd32_trace_enabled()) {
        fprintf(stderr,
                "ds4: Laguna Q/K norm/RoPE SIMD32 target_evidence "
                "route=%s tokens=%u capture=%d verifier=%d "
                "expected_target_dispatches=48 encoded=%llu completed=%llu "
                "completion=waited status=%s\n",
                route ? route : "unknown",
                n_tokens,
                capture ? 1 : 0,
                verifier ? 1 : 0,
                (unsigned long long)encoded_delta,
                (unsigned long long)completed_delta,
                exact ? "ok" : "mismatch");
    }
    if (!exact) {
        if (encoded_delta == UINT64_MAX) {
            fprintf(stderr,
                    "ds4: Laguna Q/K norm/RoPE SIMD32 target proof failed: "
                    "encoded counter moved backwards after waited %s\n",
                    route ? route : "target");
        } else {
            fprintf(stderr,
                    "ds4: Laguna Q/K norm/RoPE SIMD32 target proof failed: "
                    "expected 48 encoded layer dispatches after waited %s, "
                    "actual=%llu (global completed delta=%llu)\n",
                    route ? route : "target",
                    (unsigned long long)encoded_delta,
                    (unsigned long long)completed_delta);
        }
    }
    return exact;
}

static void laguna_metal_qk_norm_rope_simd32_target_evidence_discard(
        ds4_laguna_gpu_graph *g) {
    if (!g) return;
    g->qk_simd32_target_encoded_before = 0;
    g->qk_simd32_target_completed_before = 0;
    g->qk_simd32_target_n_tokens = 0;
    g->qk_simd32_target_capture = false;
    g->qk_simd32_target_verifier = false;
    g->qk_simd32_target_evidence_pending = false;
}

static bool laguna_metal_qk_norm_rope_simd32_target_evidence_complete(
        ds4_laguna_gpu_graph *g) {
    if (!g || !g->qk_simd32_target_evidence_pending) return true;
    const bool ok = laguna_metal_qk_norm_rope_simd32_target_evidence(
        g->qk_simd32_target_encoded_before,
        g->qk_simd32_target_completed_before,
        g->qk_simd32_target_n_tokens,
        g->qk_simd32_target_capture,
        g->qk_simd32_target_verifier,
        g->qk_simd32_target_verifier ? "speculative-verifier" : "prefill",
        true);
    laguna_metal_qk_norm_rope_simd32_target_evidence_discard(g);
    return ok;
}

/* Atlas target proof is deliberately completion-scoped.  Support-model
 * injection has its own counters and may complete before a deferred target;
 * using the global encoded counters here would either certify discarded work
 * or mix support with target.  A production Laguna graph consumes the two
 * fixed families across all 48 layers (12 global64, 36 SWA128), and encodes
 * exactly one target generation unless a completed same-key atlas is reused. */
static bool laguna_metal_rope_atlas_target_evidence(
        uint64_t generated_before,
        uint64_t expected_generated,
        uint64_t consumed_before,
        uint64_t family0_before,
        uint64_t family1_before,
        uint32_t n_tokens,
        bool capture,
        bool verifier,
        const char *route,
        bool command_waited) {
    const uint64_t generated_after =
        ds4_gpu_laguna_rope_atlas_completed_generated_count();
    const uint64_t consumed_after =
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
    const uint64_t family0_after =
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u);
    const uint64_t family1_after =
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u);
    const uint64_t generated_delta = generated_after >= generated_before ?
        generated_after - generated_before : UINT64_MAX;
    const uint64_t consumed_delta = consumed_after >= consumed_before ?
        consumed_after - consumed_before : UINT64_MAX;
    const uint64_t family0_delta = family0_after >= family0_before ?
        family0_after - family0_before : UINT64_MAX;
    const uint64_t family1_delta = family1_after >= family1_before ?
        family1_after - family1_before : UINT64_MAX;
    const bool exact = command_waited && generated_delta == expected_generated &&
        consumed_delta == 48u && family0_delta == 12u && family1_delta == 36u;
    if (exact && ds4_gpu_laguna_rope_atlas_trace_enabled()) {
        fprintf(stderr,
                "ds4: Laguna RoPE atlas waited_target route=%s tokens=%u "
                "capture=%d verifier=%d generation_expected=%llu "
                "generation_completed=%llu consumers_completed=%llu "
                "family0_completed=%llu family1_completed=%llu "
                "completion=%s status=%s\n",
                route ? route : "unknown",
                n_tokens,
                capture ? 1 : 0,
                verifier ? 1 : 0,
                (unsigned long long)expected_generated,
                (unsigned long long)generated_delta,
                (unsigned long long)consumed_delta,
                (unsigned long long)family0_delta,
                (unsigned long long)family1_delta,
                command_waited ? "waited" : "not-waited",
                exact ? "ok" : "mismatch");
    }
    if (!exact) {
        fprintf(stderr,
                "ds4: Laguna RoPE atlas target proof failed: expected "
                "completed generation=%llu consumers=48 family0=12 family1=36 "
                "after waited %s (actual generation=%llu consumers=%llu "
                "family0=%llu family1=%llu)\n",
                (unsigned long long)expected_generated,
                route ? route : "target",
                (unsigned long long)generated_delta,
                (unsigned long long)consumed_delta,
                (unsigned long long)family0_delta,
                (unsigned long long)family1_delta);
    }
    return exact;
}

static void laguna_metal_rope_atlas_target_evidence_discard(
        ds4_laguna_gpu_graph *g) {
    if (!g) return;
    g->rope_atlas_target_generated_before = 0;
    g->rope_atlas_target_consumed_before = 0;
    g->rope_atlas_target_family0_before = 0;
    g->rope_atlas_target_family1_before = 0;
    g->rope_atlas_target_generation_expected = 0;
    g->rope_atlas_target_n_tokens = 0;
    g->rope_atlas_target_capture = false;
    g->rope_atlas_target_verifier = false;
    g->rope_atlas_target_evidence_pending = false;
}

static void laguna_metal_rope_atlas_target_evidence_store(
        ds4_laguna_gpu_graph *g,
        uint64_t generated_before,
        uint64_t expected_generated,
        uint64_t consumed_before,
        uint64_t family0_before,
        uint64_t family1_before,
        uint32_t n_tokens,
        bool capture,
        bool verifier) {
    if (!g) return;
    g->rope_atlas_target_generated_before = generated_before;
    g->rope_atlas_target_generation_expected = expected_generated;
    g->rope_atlas_target_consumed_before = consumed_before;
    g->rope_atlas_target_family0_before = family0_before;
    g->rope_atlas_target_family1_before = family1_before;
    g->rope_atlas_target_n_tokens = n_tokens;
    g->rope_atlas_target_capture = capture;
    g->rope_atlas_target_verifier = verifier;
    g->rope_atlas_target_evidence_pending = true;
}

static bool laguna_metal_rope_atlas_target_evidence_complete(
        ds4_laguna_gpu_graph *g) {
    if (!g || !g->rope_atlas_target_evidence_pending) return true;
    const bool ok = laguna_metal_rope_atlas_target_evidence(
        g->rope_atlas_target_generated_before,
        g->rope_atlas_target_generation_expected,
        g->rope_atlas_target_consumed_before,
        g->rope_atlas_target_family0_before,
        g->rope_atlas_target_family1_before,
        g->rope_atlas_target_n_tokens,
        g->rope_atlas_target_capture,
        g->rope_atlas_target_verifier,
        g->rope_atlas_target_verifier ? "speculative-verifier" : "prefill",
        true);
    laguna_metal_rope_atlas_target_evidence_discard(g);
    return ok;
}

#ifdef DS4_TEST_HOOKS
/* Test-only owner for the same deferred snapshot/complete state used by a
 * speculative graph.  It is compiled into the focused test object only. */
static ds4_laguna_gpu_graph g_laguna_test_rope_atlas_evidence;
static bool g_laguna_test_rope_atlas_evidence_live;

int ds4_laguna_test_rope_atlas_deferred_snapshot(
        uint64_t generated_before,
        uint64_t expected_generated,
        uint64_t consumed_before,
        uint64_t family0_before,
        uint64_t family1_before,
        uint32_t n_tokens) {
    if (g_laguna_test_rope_atlas_evidence_live ||
        !ds4_gpu_commands_active()) {
        return 0;
    }
    memset(&g_laguna_test_rope_atlas_evidence, 0,
           sizeof(g_laguna_test_rope_atlas_evidence));
    laguna_metal_rope_atlas_target_evidence_store(
        &g_laguna_test_rope_atlas_evidence,
        generated_before,
        expected_generated,
        consumed_before,
        family0_before,
        family1_before,
        n_tokens,
        false,
        false);
    g_laguna_test_rope_atlas_evidence_live = true;
    return 1;
}

int ds4_laguna_test_rope_atlas_deferred_complete(void) {
    if (!g_laguna_test_rope_atlas_evidence_live ||
        ds4_gpu_commands_active()) {
        return 0;
    }
    const bool ok = laguna_metal_rope_atlas_target_evidence_complete(
        &g_laguna_test_rope_atlas_evidence);
    memset(&g_laguna_test_rope_atlas_evidence, 0,
           sizeof(g_laguna_test_rope_atlas_evidence));
    g_laguna_test_rope_atlas_evidence_live = false;
    return ok ? 1 : 0;
}
#endif

static void laguna_metal_target_evidence_discard(ds4_laguna_gpu_graph *g) {
    laguna_metal_qk_norm_rope_simd32_target_evidence_discard(g);
    laguna_metal_rope_atlas_target_evidence_discard(g);
}

static int laguna_metal_decode_residual_norm_mode(void);
static bool laguna_metal_decode_residual_norm_preflight(void);
static bool laguna_metal_router_decode_fused_preflight(
        const ds4_model   *model,
        const ds4_weights *weights);
#endif

static bool laguna_graph_forward_token(
        ds4_laguna_gpu_graph *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        int                   token,
        uint32_t              pos,
        const ds4_laguna_feature_capture *capture,
        float                *logits_out) {
#ifdef __APPLE__
    if (!laguna_metal_swa_gqa9_preflight(
            NULL, "Laguna decode", NULL, 0)) return false;
    if (!laguna_metal_router_simd_topk_preflight(
            NULL, "Laguna decode", NULL, 0)) return false;
#endif
    if (!g || !model || !weights || token < 0 ||
        token >= (int)DS4_N_VOCAB || pos >= g->ctx_size) {
        return false;
    }
#ifdef __APPLE__
    const int fused_router_mode = laguna_metal_router_decode_fused_mode();
    if (fused_router_mode < 0 ||
        !laguna_metal_router_decode_fused_preflight(model, weights)) {
        return false;
    }
#endif

    bool dense_q8_gate_up_fusion = false;
    if (!g->dense_q8_fusion_decision_valid) {
        if (!laguna_dense_q8_gate_up_swiglu_preflight(
                    model, weights, &dense_q8_gate_up_fusion)) {
            return false;
        }
        g->dense_q8_fusion_enabled = dense_q8_gate_up_fusion;
        g->dense_q8_fusion_decision_valid = true;
    } else {
        dense_q8_gate_up_fusion = g->dense_q8_fusion_enabled;
    }
#ifdef __APPLE__
    /* Freeze the ordinary-prefill selector at the same graph lifecycle even
     * when this first call is one-token decode.  A later env change must not
     * create a cross-feature epoch where decode and prefill disagree. */
    if (!g->dense_q8_batch_decision_valid) {
        bool dense_q8_gate_up_batch = false;
        if (!laguna_dense_q8_gate_up_swiglu_batch_preflight(
                    model, weights, &dense_q8_gate_up_batch)) {
            return false;
        }
        g->dense_q8_batch_enabled = dense_q8_gate_up_batch;
        g->dense_q8_batch_decision_valid = true;
    }
#endif
    const int dense_q8_route =
        ds4_gpu_laguna_dense_q8_gate_up_swiglu_route(
            dense_q8_gate_up_fusion ? 1 : 0, 1, 0);
#ifdef __APPLE__
    if (!laguna_metal_qk_norm_rope_simd32_preflight()) return false;
    const bool qk_simd32_target_evidence =
        ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached() > 0;
    const uint64_t qk_simd32_target_encoded_before =
        qk_simd32_target_evidence ?
        ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() : 0;
    const uint64_t qk_simd32_target_completed_before =
        qk_simd32_target_evidence ?
        ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count() : 0;
    const bool rope_atlas_target_evidence =
        ds4_gpu_laguna_rope_atlas_plan_mode_cached() > 0;
    const uint64_t rope_atlas_target_generation_expected =
        rope_atlas_target_evidence &&
        !ds4_gpu_laguna_rope_atlas_target_reuse_ready(1u, pos) ? 1u : 0u;
    const uint64_t rope_atlas_target_generated_before =
        rope_atlas_target_evidence ?
        ds4_gpu_laguna_rope_atlas_completed_generated_count() : 0;
    const uint64_t rope_atlas_target_consumed_before =
        rope_atlas_target_evidence ?
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() : 0;
    const uint64_t rope_atlas_target_family0_before =
        rope_atlas_target_evidence ?
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u) : 0;
    const uint64_t rope_atlas_target_family1_before =
        rope_atlas_target_evidence ?
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u) : 0;
    if (qk_simd32_target_evidence) {
        laguna_metal_qk_norm_rope_simd32_trace_route(
            "decode", 1u, capture != NULL, false);
    }
#endif

#ifdef __APPLE__
    const bool router_simd_topk_trace =
        laguna_metal_router_simd_topk_trace_enabled();
    bool decode_residual_norm_completion_waited = false;
    bool qk_simd32_target_command_waited = false;
    bool rope_atlas_target_command_waited = false;
    const int decode_residual_norm_mode =
        laguna_metal_decode_residual_norm_mode();
    if (decode_residual_norm_mode < 0) return false;
    const bool decode_residual_fusion = decode_residual_norm_mode != 0;
    if (decode_residual_fusion &&
        !ds4_gpu_laguna_decode_residual_norm_available()) {
        fprintf(stderr,
                "ds4: Laguna decode residual fusion requested but "
                "kernel_add3_rms_norm_mul_f32_4 is unavailable\n");
        return false;
    }
    uint64_t decode_ladder_mask = 0;
    uint32_t decode_ladder_flushes = 0;
    static bool decode_ladder_reported;
    static bool decode_residual_norm_reported;
    uint32_t fused_add2_count = 0;
    uint32_t fused_add3_count = 0;
    const char *decode_ladder_value =
        getenv("DS4_METAL_LAGUNA_DECODE_LADDER");
    if (!lgn_decode_ladder_parse(decode_ladder_value,
                                 (uint32_t)DS4_N_LAYER,
                                 &decode_ladder_mask)) {
        fprintf(stderr,
                "ds4: invalid DS4_METAL_LAGUNA_DECODE_LADDER='%s'; "
                "expected a strictly increasing comma-separated list of "
                "layer indices in [0,%u]\n",
                decode_ladder_value ? decode_ladder_value : "",
                (unsigned)(DS4_N_LAYER ? DS4_N_LAYER - 1u : 0u));
        return false;
    }
#endif /* __APPLE__ decode ladder parsing */

    /* A new graph call owns a new evidence unit.  Keep this graph-local
     * cleanup after every enabled selector preflight so a rejected request
     * cannot mutate pending state before failing closed. */
    laguna_dense_q8_gate_up_swiglu_pending_clear(g);
#ifdef __APPLE__
    laguna_metal_router_simd_topk_trace_reset();
#endif

#ifdef __APPLE__
    const bool q8_lmhead_screen =
        g->lmhead_screen != NULL && logits_out == NULL;
    g->q8_lmhead_screen_dispatched = false;
#else
    const bool q8_lmhead_screen = false;
#endif

#ifdef __APPLE__
    bool decode_attn_norm_ready = false;
    bool decode_output_norm_ready = false;
#endif

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_embed_token_quant_tensor(g->cur,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              weights->token_embd->type,
                                              DS4_N_VOCAB,
                                              (uint32_t)token,
                                              DS4_N_EMBD) != 0;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t n_head = ds4_layer_head_count(il);
        const uint32_t q_dim = n_head * DS4_N_HEAD_DIM;
        const bool is_swa = ds4_laguna_layer_is_swa(il);
        const uint32_t n_rot = is_swa ? DS4_N_ROT_SWA : DS4_N_ROT;
        const float freq_base = is_swa ? DS4_ROPE_FREQ_BASE_SWA : DS4_ROPE_FREQ_BASE;
        const float freq_scale = is_swa ? 1.0f : 1.0f / DS4_ROPE_SCALE_FACTOR;
        const float ext_factor = is_swa ? 0.0f : 1.0f;
        /* rope_yarn applies the YaRN magnitude multiplier internally.  The
         * reference runtime passes the configured attention factor here; its
         * context-level adjustment first constructs, then cancels, that same
         * internal multiplier.  Dividing once more suppresses YaRN mscale. */
        const float attn_factor = is_swa ? 1.0f :
            DS4_ROPE_YARN_ATTN_FACTOR;
        const float beta_fast = is_swa ? 0.0f : DS4_ROPE_YARN_BETA_FAST;
        const float beta_slow = is_swa ? 0.0f : DS4_ROPE_YARN_BETA_SLOW;
        const uint32_t rope_ctx = is_swa ?
            (uint32_t)DS4_CONTEXT_LENGTH : (uint32_t)DS4_ROPE_ORIG_CTX;
        bool ffn_norm_ready = false;

        ok = laguna_graph_capture_feature(capture, g->cur, il);
        if (!ok) break;
#ifdef __APPLE__
        if (decode_attn_norm_ready) {
            decode_attn_norm_ready = false;
        } else
#endif
        {
            ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm,
                                                 g->cur,
                                                 model->map,
                                                 model->size,
                                                 l->attn_norm->abs_offset,
                                                 DS4_N_EMBD,
                                                 DS4_RMS_EPS) != 0;
        }
        if (ok) {
            if (l->attn_q->type == DS4_TENSOR_F16) {
                ok = ds4_gpu_laguna_qkvg_f16_tensor(
                        g->q,
                        g->k,
                        g->v,
                        g->gate,
                        model->map,
                        model->size,
                        l->attn_q->abs_offset,
                        l->attn_k->abs_offset,
                        l->attn_v->abs_offset,
                        l->attn_gate->abs_offset,
                        DS4_N_EMBD,
                        q_dim,
                        DS4_N_HEAD_KV * DS4_N_HEAD_DIM,
                        n_head,
                        g->attn_norm) != 0;
            } else {
                ok = ds4_gpu_matmul_q8_0_pair_tensor(
                        g->q,
                        g->k,
                        model->map,
                        model->size,
                        l->attn_q->abs_offset,
                        l->attn_k->abs_offset,
                        DS4_N_EMBD,
                        q_dim,
                        DS4_N_HEAD_KV * DS4_N_HEAD_DIM,
                        g->attn_norm,
                        1) != 0 &&
                     ds4_gpu_matmul_q8_0_pair_tensor(
                        g->v,
                        g->gate,
                        model->map,
                        model->size,
                        l->attn_v->abs_offset,
                        l->attn_gate->abs_offset,
                        DS4_N_EMBD,
                        DS4_N_HEAD_KV * DS4_N_HEAD_DIM,
                        n_head,
                        g->attn_norm,
                        1) != 0;
            }
        }
        if (ok) {
            ok = ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                    g->q,
                    g->k,
                    model->map,
                    model->size,
                    l->attn_q_norm->abs_offset,
                    l->attn_k_norm->abs_offset,
                    1,
                    n_head,
                    DS4_N_HEAD_KV,
                    DS4_N_HEAD_DIM,
                    n_rot,
                    pos,
                    rope_ctx,
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    beta_fast,
                    beta_slow,
                    DS4_RMS_EPS) != 0;
        }
        uint32_t key_count = pos + 1u;
        if (key_count > g->cache_cap[il]) key_count = g->cache_cap[il];
        const uint32_t key_start = pos + 1u - key_count;
        if (ok) {
            ok = ds4_gpu_laguna_store_attention_tensor(
                    g->heads,
                    g->key_cache[il],
                    g->value_cache[il],
                    g->q,
                    g->k,
                    g->v,
                    g->gate,
                    pos,
                    g->cache_cap[il],
                    key_start,
                    key_count,
                    n_head,
                    DS4_N_HEAD_KV,
                    DS4_N_HEAD_DIM,
                    1.0f / sqrtf((float)DS4_N_HEAD_DIM)) != 0;
        }
        if (ok) {
            if (l->attn_output->type == DS4_TENSOR_F16) {
                ok = ds4_gpu_laguna_attn_output_residual_f16_tensor(
                        g->after_attn,
                        model->map,
                        model->size,
                        l->attn_output->abs_offset,
                        q_dim,
                        DS4_N_EMBD,
                        g->heads,
                        g->cur) != 0;
            } else {
                ok = laguna_graph_matmul(g->attn_out,
                                         model,
                                         l->attn_output,
                                         g->heads,
                                         1) &&
                     ds4_gpu_add_rms_norm_weight_rows_tensor(
                             g->ffn_norm,
                             g->after_attn,
                             g->cur,
                             g->attn_out,
                             model->map,
                             model->size,
                             l->ffn_norm->abs_offset,
                             DS4_N_EMBD,
                             1,
                             DS4_RMS_EPS) != 0;
                ffn_norm_ready = ok;
            }
        }
        if (ok && !ffn_norm_ready) {
            ok = ds4_gpu_rms_norm_weight_tensor(g->ffn_norm,
                                                 g->after_attn,
                                                 model->map,
                                                 model->size,
                                                 l->ffn_norm->abs_offset,
                                                 DS4_N_EMBD,
                                                 DS4_RMS_EPS) != 0;
        }
        if (ok && il < DS4_N_LEADING_DENSE) {
            if (dense_q8_route ==
                    DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_DECODE_MID) {
#ifdef __APPLE__
                ok = ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                         g->ffn_mid,
                         model->map,
                         model->size,
                         l->ffn_gate->abs_offset,
                         l->ffn_up->abs_offset,
                         DS4_N_EMBD,
                         DS4_N_FF_DENSE,
                         g->ffn_norm,
                         0.0f) != 0;
                if (ok) laguna_dense_q8_gate_up_swiglu_note_decode_mid(g);
                if (!ok) {
                    fprintf(stderr,
                            "ds4: Laguna dense Q8 gate/up+SwiGLU fused "
                            "dispatch failed; stock fallback is disabled\n");
                }
#else
                ok = false;
#endif
            } else {
                ok = laguna_graph_matmul(g->ffn_gate,
                                         model,
                                         l->ffn_gate,
                                         g->ffn_norm,
                                         1) &&
                     laguna_graph_matmul(g->ffn_up,
                                         model,
                                         l->ffn_up,
                                         g->ffn_norm,
                                         1);
                if (ok) {
                    ok = ds4_gpu_swiglu_tensor(g->ffn_mid,
                                                g->ffn_gate,
                                                g->ffn_up,
                                                DS4_N_FF_DENSE,
                                                0.0f,
                                                1.0f) != 0;
                }
            }
            if (ok) {
                ok = laguna_graph_matmul(g->ffn_out,
                                         model,
                                         l->ffn_down,
                                         g->ffn_mid,
                                         1);
            }
            if (ok) {
#ifdef __APPLE__
                const bool fuse_residual_norm =
                    decode_residual_fusion &&
                    (il + 1u < (uint32_t)DS4_N_LAYER ||
                     logits_out != NULL || g->gpu_argmax_enabled);
                if (fuse_residual_norm) {
                    ds4_gpu_tensor *norm_out = il + 1u < (uint32_t)DS4_N_LAYER ?
                        g->attn_norm : g->output_norm;
                    const uint64_t norm_offset = il + 1u < (uint32_t)DS4_N_LAYER ?
                        weights->layer[il + 1u].attn_norm->abs_offset :
                        weights->output_norm->abs_offset;
                    ok = ds4_gpu_add_rms_norm_weight_rows_tensor(
                             norm_out,
                             g->next,
                             g->after_attn,
                             g->ffn_out,
                             model->map,
                             model->size,
                             norm_offset,
                             DS4_N_EMBD,
                             1,
                             DS4_RMS_EPS) != 0;
                    if (ok) {
                        if (il + 1u < (uint32_t)DS4_N_LAYER) {
                            decode_attn_norm_ready = true;
                        } else {
                            decode_output_norm_ready = true;
                        }
                        fused_add2_count++;
                    }
                } else
#endif
                {
                    ok = ds4_gpu_add_tensor(g->next,
                                            g->after_attn,
                                            g->ffn_out,
                                            DS4_N_EMBD) != 0;
                }
            }
        } else if (ok) {
#ifdef __APPLE__
            if (fused_router_mode > 0) {
                /* This is the normal one-token Laguna decode hot path.  The
                 * fused helper owns both router projection and top-k; stock
                 * projection/selection remains the opt-out default. */
                ok = ds4_gpu_laguna_router_decode_fused_tensor(
                        g->router_selected,
                        g->router_weights,
                        g->router_probs,
                        g->router_logits,
                        model->map,
                        model->size,
                        l->ffn_gate_inp->abs_offset,
                        l->ffn_exp_probs_b->abs_offset,
                        g->ffn_norm,
                        (uint32_t)DS4_N_EMBD,
                        DS4_N_EXPERT,
                        DS4_N_EXPERT_USED,
                        DS4_EXPERT_WEIGHT_SCALE) != 0;
            } else
#endif
            {
                ok = ds4_gpu_matmul_f32_tensor(g->router_logits,
                                                model->map,
                                                model->size,
                                                l->ffn_gate_inp->abs_offset,
                                                DS4_N_EMBD,
                                                DS4_N_EXPERT,
                                                g->ffn_norm,
                                                1) != 0;
                if (ok) {
                    ok = ds4_gpu_glm_router_select_tensor(
                            g->router_selected,
                            g->router_weights,
                            g->router_probs,
                            model->map,
                            model->size,
                            l->ffn_exp_probs_b->abs_offset,
                            g->router_logits,
                            DS4_N_EXPERT,
                            DS4_N_EXPERT_USED,
                            DS4_EXPERT_WEIGHT_SCALE) != 0;
                }
            }

            const uint64_t gate_row_bytes =
                routed_expert_row_bytes(l->ffn_gate_exps);
            const uint64_t up_row_bytes =
                routed_expert_row_bytes(l->ffn_up_exps);
            const uint64_t down_row_bytes =
                routed_expert_row_bytes(l->ffn_down_exps);
            const uint64_t gate_expert_bytes =
                l->ffn_gate_exps->dim[1] * gate_row_bytes;
            const uint64_t up_expert_bytes =
                l->ffn_up_exps->dim[1] * up_row_bytes;
            const uint64_t down_expert_bytes =
                l->ffn_down_exps->dim[1] * down_row_bytes;
            const uint64_t shared_gate_row_bytes =
                routed_expert_row_bytes(l->ffn_gate_shexp);
            const uint64_t shared_up_row_bytes =
                routed_expert_row_bytes(l->ffn_up_shexp);
            const uint64_t shared_down_row_bytes =
                routed_expert_row_bytes(l->ffn_down_shexp);
            const ds4_gpu_laguna_moe_desc routed_moe = {
                .gate_offset = l->ffn_gate_exps->abs_offset,
                .up_offset = l->ffn_up_exps->abs_offset,
                .down_offset = l->ffn_down_exps->abs_offset,
                .gate_type = l->ffn_gate_exps->type,
                .up_type = l->ffn_up_exps->type,
                .down_type = l->ffn_down_exps->type,
                .gate_expert_bytes = gate_expert_bytes,
                .gate_row_bytes = gate_row_bytes,
                .up_expert_bytes = up_expert_bytes,
                .up_row_bytes = up_row_bytes,
                .down_expert_bytes = down_expert_bytes,
                .down_row_bytes = down_row_bytes,
            };
            const ds4_gpu_laguna_moe_desc shared_moe = {
                .gate_offset = l->ffn_gate_shexp->abs_offset,
                .up_offset = l->ffn_up_shexp->abs_offset,
                .down_offset = l->ffn_down_shexp->abs_offset,
                .gate_type = l->ffn_gate_shexp->type,
                .up_type = l->ffn_up_shexp->type,
                .down_type = l->ffn_down_shexp->type,
                .gate_expert_bytes =
                    l->ffn_gate_shexp->dim[1] * shared_gate_row_bytes,
                .gate_row_bytes = shared_gate_row_bytes,
                .up_expert_bytes =
                    l->ffn_up_shexp->dim[1] * shared_up_row_bytes,
                .up_row_bytes = shared_up_row_bytes,
                .down_expert_bytes =
                    l->ffn_down_shexp->dim[1] * shared_down_row_bytes,
                .down_row_bytes = shared_down_row_bytes,
            };
            /* The legacy recipe can co-dispatch routed and shared Q4 work.
             * Revised layouts keep one routed quantization but make the
             * shared expert Q8, so each uses its native fast kernel family. */
            if (ok && l->ffn_gate_shexp->type == DS4_TENSOR_Q4_K) {
                ok = ds4_gpu_laguna_routed_shared_moe_one_tensor(
                        g->ffn_out,
                        g->routed_mid,
                        g->shared_out,
                        g->ffn_mid,
                        model->map,
                        model->size,
                        &routed_moe,
                        &shared_moe,
                        DS4_N_EMBD,
                        DS4_N_FF_EXP,
                        DS4_N_EMBD,
                        g->router_selected,
                        g->router_weights,
                        DS4_N_EXPERT,
                        DS4_N_EXPERT_USED,
                        g->shared_selected,
                        g->shared_weight,
                        g->ffn_norm) != 0;
            } else if (ok) {
                ok = ds4_gpu_glm_routed_moe_one_tensor(
                        g->ffn_out,
                        g->routed_mid,
                        model->map,
                        model->size,
                        l->ffn_gate_exps->abs_offset,
                        l->ffn_up_exps->abs_offset,
                        l->ffn_down_exps->abs_offset,
                        l->ffn_gate_exps->type,
                        l->ffn_up_exps->type,
                        l->ffn_down_exps->type,
                        gate_expert_bytes,
                        gate_row_bytes,
                        up_expert_bytes,
                        up_row_bytes,
                        down_expert_bytes,
                        down_row_bytes,
                        DS4_N_EMBD,
                        DS4_N_FF_EXP,
                        DS4_N_EMBD,
                        g->router_selected,
                        g->router_weights,
                        DS4_N_EXPERT,
                        DS4_N_EXPERT_USED,
                        il,
                        g->ffn_norm,
                        true) != 0;
                if (ok) {
                    ok = ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                            g->ffn_mid,
                            model->map,
                            model->size,
                            l->ffn_gate_shexp->abs_offset,
                            l->ffn_up_shexp->abs_offset,
                            DS4_N_EMBD,
                            DS4_N_FF_SHARED,
                            g->ffn_norm,
                            0.0f) != 0;
                }
                if (ok) {
                    ok = laguna_graph_matmul(g->shared_out,
                                             model,
                                             l->ffn_down_shexp,
                                             g->ffn_mid,
                                             1);
                }
            }
            if (ok) {
#ifdef __APPLE__
                const bool fuse_residual_norm =
                    decode_residual_fusion &&
                    (il + 1u < (uint32_t)DS4_N_LAYER ||
                     logits_out != NULL || g->gpu_argmax_enabled);
                if (fuse_residual_norm) {
                    ds4_gpu_tensor *norm_out = il + 1u < (uint32_t)DS4_N_LAYER ?
                        g->attn_norm : g->output_norm;
                    const uint64_t norm_offset = il + 1u < (uint32_t)DS4_N_LAYER ?
                        weights->layer[il + 1u].attn_norm->abs_offset :
                        weights->output_norm->abs_offset;
                    ok = ds4_gpu_add3_rms_norm_weight_rows_tensor(
                             norm_out,
                             g->next,
                             g->after_attn,
                             g->ffn_out,
                             g->shared_out,
                             model->map,
                             model->size,
                             norm_offset,
                             DS4_N_EMBD,
                             1,
                             DS4_RMS_EPS) != 0;
                    if (ok) {
                        if (il + 1u < (uint32_t)DS4_N_LAYER) {
                            decode_attn_norm_ready = true;
                        } else {
                            decode_output_norm_ready = true;
                        }
                        fused_add3_count++;
                    }
                } else
#endif
                {
                    ok = ds4_gpu_add3_tensor(g->next,
                                             g->after_attn,
                                             g->ffn_out,
                                             g->shared_out,
                                             DS4_N_EMBD) != 0;
                }
            }
        }

        if (ok) {
            ds4_gpu_tensor *tmp = g->cur;
            g->cur = g->next;
            g->next = tmp;
        }
#ifdef __APPLE__
        if (ok && decode_ladder_mask != 0 && il < 64u &&
            (decode_ladder_mask & (UINT64_C(1) << il)) != 0) {
            if (ds4_gpu_flush_commands() == 0) {
                ok = false;
            } else {
                decode_ladder_flushes++;
            }
        }
#endif
    }

    if (ok) {
        ok = laguna_graph_capture_feature(capture, g->cur, DS4_N_LAYER);
    }
    if (ok && (logits_out || g->gpu_argmax_enabled)) {
#ifdef __APPLE__
        if (decode_output_norm_ready) {
            /* The final residual fused directly into g->output_norm. */
        } else
#endif
        {
        ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                             g->cur,
                                             model->map,
                                             model->size,
                                             weights->output_norm->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_RMS_EPS) != 0;
        }
        if (ok) {
            if (q8_lmhead_screen) {
#ifdef __APPLE__
                ok = ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    g->lmhead_screen,
                    g->argmax,
                    NULL,
                    model->map,
                    model->size,
                    weights->output->abs_offset,
                    g->output_norm) != 0;
                if (ok) g->q8_lmhead_screen_dispatched = true;
#endif
            } else {
                ok = laguna_graph_matmul(g->logits,
                                         model,
                                         weights->output,
                                         g->output_norm,
                                         1);
            }
        }
    }
#ifdef __APPLE__
    if (ok && g->gpu_argmax_enabled && !q8_lmhead_screen) {
        ok = ds4_gpu_laguna_argmax_tensor(g->argmax,
                                          g->logits,
                                          DS4_N_VOCAB) != 0;
    }
#endif
#ifdef __APPLE__
    if (ds4_gpu_commands_active()) {
        if (ok) {
            if (ds4_gpu_end_commands() == 0) {
                ok = false;
            } else {
                decode_residual_norm_completion_waited = true;
                qk_simd32_target_command_waited = true;
                rope_atlas_target_command_waited = true;
                laguna_dense_q8_gate_up_swiglu_report_waited(g);
            }
        } else {
            /* Never commit KV/attention work recorded before a later graph
             * stage failed.  A failed command batch is a transaction: drop
             * it so the cache remains unchanged and route evidence cannot
             * promote. */
            if (ds4_gpu_discard_commands() == 0) ok = false;
            laguna_dense_q8_gate_up_swiglu_pending_clear(g);
        }
    }
    if (ok && qk_simd32_target_evidence) {
        ok = laguna_metal_qk_norm_rope_simd32_target_evidence(
            qk_simd32_target_encoded_before,
            qk_simd32_target_completed_before,
            1u,
            capture != NULL,
            false,
            "decode",
            qk_simd32_target_command_waited);
    }
    if (ok && rope_atlas_target_evidence) {
        ok = laguna_metal_rope_atlas_target_evidence(
            rope_atlas_target_generated_before,
            rope_atlas_target_generation_expected,
            rope_atlas_target_consumed_before,
            rope_atlas_target_family0_before,
            rope_atlas_target_family1_before,
            1u,
            capture != NULL,
            false,
            "decode",
            rope_atlas_target_command_waited);
    }
    if (ok && decode_residual_fusion &&
        decode_residual_norm_completion_waited &&
        !decode_residual_norm_reported &&
        (logits_out != NULL || g->gpu_argmax_enabled) &&
        (fused_add2_count != 0 || fused_add3_count != 0)) {
        fprintf(stderr,
                "ds4: Laguna decode residual+RMS fusion enabled "
                "(add2=%u add3=%u completion=waited)\n",
                fused_add2_count,
                fused_add3_count);
        decode_residual_norm_reported = true;
    }
    if (ok && decode_ladder_flushes != 0 && !decode_ladder_reported) {
        char canonical[256];
        if (lgn_decode_ladder_format(decode_ladder_mask,
                                     (uint32_t)DS4_N_LAYER,
                                     canonical,
                                     sizeof(canonical))) {
            fprintf(stderr,
                    "ds4: Laguna decode command-buffer ladder enabled "
                    "(layers=%s; flushes=%u; completion=waited)\n",
                    canonical,
                    decode_ladder_flushes);
        } else {
            /* The fixed 64-bit mask always fits in the local buffer; keep a
             * diagnostic if that invariant is ever changed. */
            fprintf(stderr,
                    "ds4: Laguna decode command-buffer ladder enabled "
                    "(flushes=%u; completion=waited)\n",
                    decode_ladder_flushes);
        }
        decode_ladder_reported = true;
    }
#else
    if (ds4_gpu_commands_active()) {
        if (ok) {
            if (ds4_gpu_end_commands() == 0) ok = false;
        } else if (ds4_gpu_discard_commands() == 0) {
            ok = false;
        }
    }
#endif
    if (ok && g->gpu_argmax_enabled) {
        ok = ds4_gpu_tensor_read(g->argmax,
                                 0,
                                 &g->gpu_argmax_result,
                                 sizeof(g->gpu_argmax_result)) != 0;
    }
#ifdef __APPLE__
    if (ok && g->q8_lmhead_screen_dispatched) {
        laguna_graph_report_q8_lmhead_screen(g);
    }
#endif
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits,
                                 0,
                                 logits_out,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
#ifdef __APPLE__
    if (router_simd_topk_trace) {
        laguna_metal_router_simd_topk_trace_report("Laguna decode");
    }
#endif
    if (!ok) laguna_dense_q8_gate_up_swiglu_pending_clear(g);
    return ok;
}

static void laguna_graph_report_prefill_display_progress(
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                layer_done,
        int                     total,
        bool                    allow_complete) {
    if (!display_progress || n_tokens == 0) return;
    if (layer_done > (uint32_t)DS4_N_LAYER) {
        layer_done = (uint32_t)DS4_N_LAYER;
    }

    uint64_t done = (uint64_t)n_tokens * layer_done /
                    (uint32_t)DS4_N_LAYER;
    if (layer_done == (uint32_t)DS4_N_LAYER) done = n_tokens;
    if (!allow_complete && done >= n_tokens) done = n_tokens - 1u;
    display_progress(display_progress_ud,
                     "prefill_display",
                     (int)((uint64_t)pos0 + done),
                     total);
}

static bool laguna_graph_capture_final_feature(
        ds4_laguna_gpu_graph             *g,
        const ds4_model                  *model,
        const ds4_layer_weights          *layer,
        uint32_t                          n_tokens,
        const ds4_laguna_feature_capture *capture) {
    if (!capture) return true;
    if (!g || !model || !layer ||
        capture->src_row0 > n_tokens ||
        capture->n_rows > n_tokens - capture->src_row0) {
        return false;
    }

    /*
     * The fast Q8 batch kernel stages FP32 activations as FP16. Laguna's final
     * shared-expert SwiGLU can legitimately exceed the FP16 range on early
     * prompt rows; normal inference does not consume those final hidden rows,
     * but DFlash does. Recompute only the rows DFlash captures with the
     * decode-row reduction, which reads FP32 directly. The ordinary target
     * result in g->cur remains untouched, so enabling DFlash cannot alter
     * target logits.
     */
    if (n_tokens <= 16u ||
        layer->ffn_down_shexp->type != DS4_TENSOR_Q8_0) {
        return laguna_graph_capture_feature(capture,
                                            g->cur,
                                            DS4_N_LAYER);
    }

    const uint64_t rows = capture->n_rows;
    const uint64_t embd_bytes = rows * DS4_N_EMBD * sizeof(float);
    const uint64_t shared_bytes =
        rows * DS4_N_FF_SHARED * sizeof(float);
    const uint64_t src_embd_off =
        (uint64_t)capture->src_row0 * DS4_N_EMBD * sizeof(float);
    const uint64_t src_shared_off =
        (uint64_t)capture->src_row0 * DS4_N_FF_SHARED * sizeof(float);

    ds4_gpu_tensor *mid = ds4_gpu_tensor_view(g->ffn_mid,
                                               src_shared_off,
                                               shared_bytes);
    ds4_gpu_tensor *after_attn = ds4_gpu_tensor_view(g->after_attn,
                                                     src_embd_off,
                                                     embd_bytes);
    ds4_gpu_tensor *routed = ds4_gpu_tensor_view(g->ffn_out,
                                                 src_embd_off,
                                                 embd_bytes);
    ds4_gpu_tensor *shared = ds4_gpu_tensor_view(g->attn_out,
                                                 0,
                                                 embd_bytes);
    ds4_gpu_tensor *final = ds4_gpu_tensor_view(g->q,
                                                0,
                                                embd_bytes);
    bool ok = mid && after_attn && routed && shared && final;
    if (ok) {
        ok = laguna_graph_matmul_decode_rows(
                 shared,
                 model,
                 layer->ffn_down_shexp,
                 mid,
                 capture->n_rows,
                 true);
    }
    if (ok) {
        ok = ds4_gpu_add3_tensor(final,
                                 after_attn,
                                 routed,
                                 shared,
                                 rows * DS4_N_EMBD) != 0;
    }
    if (ok) {
        ds4_laguna_feature_capture adjusted = *capture;
        adjusted.src_row0 = 0;
        ok = laguna_graph_capture_feature(&adjusted,
                                          final,
                                          DS4_N_LAYER);
    }
    ds4_gpu_tensor_free(final);
    ds4_gpu_tensor_free(shared);
    ds4_gpu_tensor_free(routed);
    ds4_gpu_tensor_free(after_attn);
    ds4_gpu_tensor_free(mid);
    return ok;
}

/* Keep the historical stock prefill Q/K norm+RoPE dispatch choice opt-in.
 * A frozen SIMD32 or RoPE-atlas plan forces the paired route for every graph
 * batch shape; otherwise only the explicit literal selector enables paired
 * ordinary prefill. */
static bool laguna_graph_prefill_qk_norm_rope_paired_requested(void) {
    const char *env = getenv("DS4_LAGUNA_PREFILL_QK_NORM_ROPE_PAIRED");
    return env && strcmp(env, "1") == 0;
}

int ds4_laguna_graph_prefill_qk_norm_rope_paired_route(
        int simd32_plan_mode,
        int atlas_plan_mode,
        int exact_q8_rows,
        int gpu_draft_tokens,
        int capture,
        int explicit_request) {
    if (simd32_plan_mode > 0 || atlas_plan_mode > 0) return 1;
    return !exact_q8_rows && !gpu_draft_tokens && !capture &&
           explicit_request != 0;
}

static bool laguna_graph_forward_batch(
        ds4_laguna_gpu_graph *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        const int            *tokens,
        const ds4_gpu_tensor *gpu_draft_tokens,
        uint32_t              n_tokens,
        uint32_t              pos0,
        float                *logits_out,
        int                  *row_argmax_out,
        const ds4_laguna_feature_capture *capture,
        ds4_session_progress_fn display_progress,
        void                 *display_progress_ud,
        int                   display_total) {
#ifdef __APPLE__
    if (!laguna_metal_swa_gqa9_preflight(
            NULL, "Laguna prefill/speculative batch", NULL, 0)) return false;
    if (!laguna_metal_router_simd_topk_preflight(
            NULL, "Laguna prefill/speculative batch", NULL, 0)) return false;
#endif
    if (!g || !model || !weights || !tokens || n_tokens == 0 ||
        n_tokens > g->prefill_cap || pos0 > g->ctx_size - n_tokens) {
        return false;
    }
    const bool caller_owned_draft_batch = gpu_draft_tokens != NULL;
#ifdef __APPLE__
    const int fused_router_mode = laguna_metal_router_decode_fused_mode();
    if (fused_router_mode < 0 ||
        !laguna_metal_router_decode_fused_preflight(model, weights)) {
        return false;
    }
    const bool fused_router_batch = fused_router_mode > 0 && n_tokens == 1u;
#else
    const bool fused_router_batch = false;
#endif
#ifdef __APPLE__
    const bool router_simd_topk_trace =
        laguna_metal_router_simd_topk_trace_enabled();
#endif
    bool dense_q8_gate_up_fusion = false;
    if (!g->dense_q8_fusion_decision_valid) {
        if (!laguna_dense_q8_gate_up_swiglu_preflight(
                    model, weights, &dense_q8_gate_up_fusion)) {
            return false;
        }
        g->dense_q8_fusion_enabled = dense_q8_gate_up_fusion;
        g->dense_q8_fusion_decision_valid = true;
    } else {
        dense_q8_gate_up_fusion = g->dense_q8_fusion_enabled;
    }
#ifdef __APPLE__
    bool dense_q8_gate_up_batch = false;
    if (!g->dense_q8_batch_decision_valid) {
        if (!laguna_dense_q8_gate_up_swiglu_batch_preflight(
                    model, weights, &dense_q8_gate_up_batch)) {
            return false;
        }
        g->dense_q8_batch_enabled = dense_q8_gate_up_batch;
        g->dense_q8_batch_decision_valid = true;
    } else {
        dense_q8_gate_up_batch = g->dense_q8_batch_enabled;
    }
#endif
    /* Freeze and validate all enabled route selectors before touching graph
     * scratch/pending state or beginning any GPU work. */
    laguna_dense_q8_gate_up_swiglu_pending_clear(g);
#ifdef __APPLE__
    laguna_metal_router_simd_topk_trace_reset();
    if (!laguna_metal_qk_norm_rope_simd32_preflight()) return false;
    const bool qk_simd32_target_evidence =
        ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached() > 0;
    const uint64_t qk_simd32_target_encoded_before =
        qk_simd32_target_evidence ?
        ds4_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count() : 0;
    const uint64_t qk_simd32_target_completed_before =
        qk_simd32_target_evidence ?
        ds4_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count() : 0;
    const bool rope_atlas_target_evidence =
        ds4_gpu_laguna_rope_atlas_plan_mode_cached() > 0;
    const uint64_t rope_atlas_target_generation_expected =
        rope_atlas_target_evidence &&
        !ds4_gpu_laguna_rope_atlas_target_reuse_ready(n_tokens, pos0) ? 1u : 0u;
    const uint64_t rope_atlas_target_generated_before =
        rope_atlas_target_evidence ?
        ds4_gpu_laguna_rope_atlas_completed_generated_count() : 0;
    const uint64_t rope_atlas_target_consumed_before =
        rope_atlas_target_evidence ?
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() : 0;
    const uint64_t rope_atlas_target_family0_before =
        rope_atlas_target_evidence ?
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u) : 0;
    const uint64_t rope_atlas_target_family1_before =
        rope_atlas_target_evidence ?
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u) : 0;
#endif
    if (row_argmax_out &&
        (n_tokens > DS4_DFLASH_BLOCK_SIZE ||
         !laguna_graph_ensure_spec_scratch(g))) {
        return false;
    }

    bool ok = true;
    bool dense_q8_completion_waited = false;
    const bool gpu_draft_pipeline_ready =
        ds4_gpu_commands_active() != 0;
    if (gpu_draft_tokens) {
        /* Copy proposals on-GPU so drafting does not add an intermediate
         * completion and CPU readback. Metal keeps both graphs in one command
         * buffer, so the existing command boundary provides the ordering. */
        if (!gpu_draft_pipeline_ready ||
            tokens[0] < 0 || tokens[0] >= (int)DS4_N_VOCAB ||
            ds4_gpu_tensor_bytes(gpu_draft_tokens) <
                (uint64_t)n_tokens * sizeof(uint32_t)) {
            return false;
        }
        const uint32_t first_token = (uint32_t)tokens[0];
        ok = ds4_gpu_tensor_write(g->tokens,
                                  0,
                                  &first_token,
                                  sizeof(first_token)) != 0;
        if (ok && n_tokens > 1u) {
            ok = ds4_gpu_tensor_copy(
                     g->tokens,
                     sizeof(uint32_t),
                     gpu_draft_tokens,
                     sizeof(uint32_t),
                     (uint64_t)(n_tokens - 1u) * sizeof(uint32_t)) != 0;
        }
    } else {
        uint32_t *token_ids =
            xmalloc((size_t)n_tokens * sizeof(*token_ids));
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (tokens[i] < 0 || tokens[i] >= (int)DS4_N_VOCAB) {
                free(token_ids);
                return false;
            }
            token_ids[i] = (uint32_t)tokens[i];
        }
        ok = ds4_gpu_tensor_write(
                 g->tokens,
                 0,
                 token_ids,
                 (uint64_t)n_tokens * sizeof(*token_ids)) != 0;
        free(token_ids);
    }
    if (!ok) return false;

    laguna_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  pos0,
                                                  n_tokens,
                                                  0,
                                                  display_total,
                                                  true);

    /* Metal 4 Tensor matmuls introduce enough low-bit accumulation drift in
     * Laguna prefill to change sliding-attention routes after a few hundred
     * tokens. The legacy batched kernels are both stable against the Poolside
     * reference and slightly faster on these shapes. */
    ds4_gpu_set_tensor_matmul_suppressed(true);
    /* A long Laguna prefill otherwise lives in one command buffer and cannot
     * report real progress until the whole chunk completes.  When a frontend
     * asks for display progress, finish one layer at a time so each callback
     * reflects GPU work that has actually completed.  Tiny suffixes keep the
     * lower-overhead single-command path. */
    const bool live_progress = display_progress != NULL && n_tokens >= 32u;
    const bool exact_q8_rows = row_argmax_out != NULL;
#ifdef __APPLE__
    bool qk_simd32_target_command_waited = false;
    bool rope_atlas_target_command_waited = false;
#endif
    const int dense_q8_route =
        ds4_gpu_laguna_dense_q8_gate_up_swiglu_route(
            dense_q8_gate_up_fusion ? 1 : 0,
            0,
            exact_q8_rows ? 1 : 0);
#ifdef __APPLE__
    const int dense_q8_prefill_route =
        ds4_gpu_laguna_dense_q8_gate_up_swiglu_prefill_route(
            dense_q8_gate_up_batch ? 1 : 0,
            exact_q8_rows ? 1 : 0,
            n_tokens,
            ds4_gpu_laguna_q8_mv_ext_max_tokens());
#else
    const int dense_q8_prefill_route =
        DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_ORDINARY_PREFILL_STOCK;
#endif
    /* DFlash feature capture and GPU draft-token verification have their own
     * cache/injection sequencing, but an enabled frozen SIMD32/atlas plan
     * still forces the paired route so all production shapes are certified. */
#ifdef __APPLE__
    const int simd32_plan_mode =
        ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached();
    const int rope_atlas_plan_mode =
        ds4_gpu_laguna_rope_atlas_plan_mode_cached();
#else
    const int simd32_plan_mode = 0;
    const int rope_atlas_plan_mode = 0;
#endif
    const bool paired_qk_norm_rope =
        ds4_laguna_graph_prefill_qk_norm_rope_paired_route(
            simd32_plan_mode,
            rope_atlas_plan_mode,
            exact_q8_rows ? 1 : 0,
            gpu_draft_tokens != NULL ? 1 : 0,
            capture != NULL ? 1 : 0,
            laguna_graph_prefill_qk_norm_rope_paired_requested() ? 1 : 0) != 0;
#ifdef __APPLE__
    if (simd32_plan_mode > 0) {
        laguna_metal_qk_norm_rope_simd32_trace_route(
                row_argmax_out ? "speculative-verifier" : "prefill",
                n_tokens, capture != NULL, row_argmax_out != NULL);
    }
#endif
    if (gpu_draft_tokens) {
        ok = gpu_draft_pipeline_ready;
    } else {
        ok = ds4_gpu_begin_commands() != 0;
    }
    if (ok) {
        ok = ds4_gpu_embed_tokens_quant_tensor(g->cur,
                                                g->tokens,
                                                model->map,
                                                model->size,
                                                weights->token_embd->abs_offset,
                                                weights->token_embd->type,
                                                DS4_N_VOCAB,
                                                n_tokens,
                                                DS4_N_EMBD) != 0;
    }

    uint32_t completed_layers = 0;
    const char *failed_stage = "embedding";
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t n_head = ds4_layer_head_count(il);
        const bool is_swa = ds4_laguna_layer_is_swa(il);
        const uint32_t n_rot = is_swa ? DS4_N_ROT_SWA : DS4_N_ROT;
        const float freq_base = is_swa ?
            DS4_ROPE_FREQ_BASE_SWA : DS4_ROPE_FREQ_BASE;
        const float freq_scale = is_swa ?
            1.0f : 1.0f / DS4_ROPE_SCALE_FACTOR;
        const float ext_factor = is_swa ? 0.0f : 1.0f;
        const float attn_factor = is_swa ? 1.0f : DS4_ROPE_YARN_ATTN_FACTOR;
        const float beta_fast = is_swa ? 0.0f : DS4_ROPE_YARN_BETA_FAST;
        const float beta_slow = is_swa ? 0.0f : DS4_ROPE_YARN_BETA_SLOW;
        const uint32_t rope_ctx = is_swa ?
            (uint32_t)DS4_CONTEXT_LENGTH : (uint32_t)DS4_ROPE_ORIG_CTX;

        failed_stage = "attention norm";
        ok = laguna_graph_capture_feature(capture, g->cur, il);
        if (!ok) {
            failed_stage = "DFlash feature capture";
            break;
        }
        ok = ds4_gpu_rms_norm_weight_rows_tensor(
                g->attn_norm,
                g->cur,
                model->map,
                model->size,
                l->attn_norm->abs_offset,
                DS4_N_EMBD,
                n_tokens,
                DS4_RMS_EPS) != 0;
        if (ok) {
            failed_stage = "Q/K projection";
            ok = laguna_graph_matmul_decode_rows(g->q,
                                                 model,
                                                 l->attn_q,
                                                 g->attn_norm,
                                                 n_tokens,
                                                 exact_q8_rows) &&
                 laguna_graph_matmul_decode_rows(g->k,
                                                 model,
                                                 l->attn_k,
                                                 g->attn_norm,
                                                 n_tokens,
                                                 exact_q8_rows);
        }
        if (ok) {
            failed_stage = "V/gate projection";
            ok = laguna_graph_matmul_decode_rows(g->v,
                                                 model,
                                                 l->attn_v,
                                                 g->attn_norm,
                                                 n_tokens,
                                                 exact_q8_rows) &&
                 laguna_graph_matmul_decode_rows(g->gate,
                                                 model,
                                                 l->attn_gate,
                                                 g->attn_norm,
                                                 n_tokens,
                                                 exact_q8_rows);
        }
        if (ok && (exact_q8_rows || paired_qk_norm_rope)) {
            failed_stage = "Q/K norm/RoPE";
            ok = ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                    g->q,
                    g->k,
                    model->map,
                    model->size,
                    l->attn_q_norm->abs_offset,
                    l->attn_k_norm->abs_offset,
                    n_tokens,
                    n_head,
                    DS4_N_HEAD_KV,
                    DS4_N_HEAD_DIM,
                    n_rot,
                    pos0,
                    rope_ctx,
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    beta_fast,
                    beta_slow,
                    DS4_RMS_EPS) != 0;
        } else {
            if (ok) {
                failed_stage = "Q norm/RoPE";
                ok = ds4_gpu_laguna_head_rms_norm_rope_tensor(
                        g->q,
                        model->map,
                        model->size,
                        l->attn_q_norm->abs_offset,
                        n_tokens,
                        n_head,
                        DS4_N_HEAD_DIM,
                        n_rot,
                        pos0,
                        rope_ctx,
                        freq_base,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        beta_fast,
                        beta_slow,
                        DS4_RMS_EPS) != 0;
            }
            if (ok) {
                failed_stage = "K norm/RoPE";
                ok = ds4_gpu_laguna_head_rms_norm_rope_tensor(
                        g->k,
                        model->map,
                        model->size,
                        l->attn_k_norm->abs_offset,
                        n_tokens,
                        DS4_N_HEAD_KV,
                        DS4_N_HEAD_DIM,
                        n_rot,
                        pos0,
                        rope_ctx,
                        freq_base,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        beta_fast,
                        beta_slow,
                        DS4_RMS_EPS) != 0;
            }
        }
        if (ok) {
            failed_stage = "causal attention";
            /* Ask the backend to replay decode's split-key attention per
             * verifier row for long-context Metal throughput. */
            const int split_decode_rows = row_argmax_out != NULL &&
                (uint64_t)pos0 + n_tokens > 256u &&
                true;
            ok = ds4_gpu_laguna_attention_prefill_tensor(
                    g->heads,
                    g->key_cache[il],
                    g->value_cache[il],
                    g->staged_key,
                    g->staged_value,
                    g->q,
                    g->k,
                    g->v,
                    g->gate,
                    pos0,
                    n_tokens,
                    g->cache_cap[il],
                    n_head,
                    DS4_N_HEAD_KV,
                    DS4_N_HEAD_DIM,
                    1.0f / sqrtf((float)DS4_N_HEAD_DIM),
                    split_decode_rows) != 0;
        }
        if (ok) {
            failed_stage = "attention output projection";
            ok = laguna_graph_matmul_decode_rows(g->attn_out,
                                                 model,
                                                 l->attn_output,
                                                 g->heads,
                                                 n_tokens,
                                                 exact_q8_rows);
        }
        if (ok) {
            failed_stage = "attention residual + FFN norm";
            ok = ds4_gpu_add_rms_norm_weight_rows_tensor(
                    g->ffn_norm,
                    g->after_attn,
                    g->cur,
                    g->attn_out,
                    model->map,
                    model->size,
                    l->ffn_norm->abs_offset,
                    DS4_N_EMBD,
                    n_tokens,
                    DS4_RMS_EPS) != 0;
        }
        if (ok && il < DS4_N_LEADING_DENSE) {
            /* Exact verifier rows stay on the stock row-wise matmuls.  The
             * fused decode route stays one-token; ordinary prefill may opt
             * into the batched fused gate/up+SwiGLU pass, which reads the
             * normed rows once and never materializes gate/up. */
            if (dense_q8_prefill_route ==
                    DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_BATCH_FUSED) {
#ifdef __APPLE__
                failed_stage = "dense FFN batched gate/up+SwiGLU";
                ok = ds4_gpu_laguna_dense_q8_gate_up_swiglu_batch_tensor(
                        g->ffn_mid,
                        model->map,
                        model->size,
                        l->ffn_gate->abs_offset,
                        l->ffn_up->abs_offset,
                        DS4_N_EMBD,
                        DS4_N_FF_DENSE,
                        g->ffn_norm,
                        n_tokens) != 0;
#else
                /* The batch preflight only enables the route on Metal, so
                 * this arm is unreachable elsewhere; keep the link clean. */
                ok = false;
#endif
            } else {
                failed_stage = "dense FFN gate/up";
                ok = laguna_graph_matmul_decode_rows(g->ffn_gate,
                                                     model,
                                                     l->ffn_gate,
                                                     g->ffn_norm,
                                                     n_tokens,
                                                     exact_q8_rows) &&
                     laguna_graph_matmul_decode_rows(g->ffn_up,
                                                     model,
                                                     l->ffn_up,
                                                     g->ffn_norm,
                                                     n_tokens,
                                                     exact_q8_rows);
                if (ok) {
                    failed_stage = "dense FFN SwiGLU";
                    ok = ds4_gpu_swiglu_tensor(
                            g->ffn_mid,
                            g->ffn_gate,
                            g->ffn_up,
                            (uint64_t)n_tokens * DS4_N_FF_DENSE,
                            0.0f,
                            1.0f) != 0;
                }
                if (ok && dense_q8_route ==
                               DS4_GPU_LAGUNA_DENSE_Q8_ROUTE_ORDINARY_PREFILL_STOCK) {
                    laguna_dense_q8_gate_up_swiglu_note_prefill_stock(g);
                }
            }
            if (ok) {
                failed_stage = "dense FFN down";
                ok = laguna_graph_matmul_decode_rows(g->ffn_out,
                                                     model,
                                                     l->ffn_down,
                                                     g->ffn_mid,
                                                     n_tokens,
                                                     exact_q8_rows);
            }
            if (ok) {
                failed_stage = "dense FFN residual";
                ok = ds4_gpu_add_tensor(g->next,
                                        g->after_attn,
                                        g->ffn_out,
                                        (uint64_t)n_tokens * DS4_N_EMBD) != 0;
            }
        } else if (ok) {
            if (exact_q8_rows) {
                failed_stage = "decode-row router";
                ok = laguna_graph_router_decode_rows(
                        g, model, l, n_tokens, false);
            } else if (fused_router_batch) {
                /* A one-row ordinary graph uses the same production fused
                 * dispatch as laguna_graph_forward_token.  Exact verifier
                 * batches deliberately stay on their row-replay path above. */
                failed_stage = "fused router";
                ok = ds4_gpu_laguna_router_decode_fused_tensor(
                         g->router_selected,
                         g->router_weights,
                         g->router_probs,
                         g->router_logits,
                         model->map,
                         model->size,
                         l->ffn_gate_inp->abs_offset,
                         l->ffn_exp_probs_b->abs_offset,
                         g->ffn_norm,
                         (uint32_t)DS4_N_EMBD,
                         DS4_N_EXPERT,
                         DS4_N_EXPERT_USED,
                         DS4_EXPERT_WEIGHT_SCALE) != 0;
            } else {
                failed_stage = "router projection";
                ok = ds4_gpu_matmul_f32_tensor(
                        g->router_logits,
                        model->map,
                        model->size,
                        l->ffn_gate_inp->abs_offset,
                        DS4_N_EMBD,
                        DS4_N_EXPERT,
                        g->ffn_norm,
                        n_tokens) != 0;
            }
            if (ok && !exact_q8_rows && !fused_router_batch) {
                failed_stage = "router selection";
                ok = ds4_gpu_glm_router_select_batch_tensor(
                        g->router_selected,
                        g->router_weights,
                        g->router_probs,
                        model->map,
                        model->size,
                        l->ffn_exp_probs_b->abs_offset,
                        g->router_logits,
                        DS4_N_EXPERT,
                        DS4_N_EXPERT_USED,
                        DS4_EXPERT_WEIGHT_SCALE,
                        n_tokens) != 0;
            }

            const uint64_t gate_row_bytes =
                routed_expert_row_bytes(l->ffn_gate_exps);
            const uint64_t up_row_bytes =
                routed_expert_row_bytes(l->ffn_up_exps);
            const uint64_t down_row_bytes =
                routed_expert_row_bytes(l->ffn_down_exps);
            const uint64_t gate_expert_bytes =
                l->ffn_gate_exps->dim[1] * gate_row_bytes;
            const uint64_t up_expert_bytes =
                l->ffn_up_exps->dim[1] * up_row_bytes;
            const uint64_t down_expert_bytes =
                l->ffn_down_exps->dim[1] * down_row_bytes;
            if (ok) {
                failed_stage = "routed experts";
                if (exact_q8_rows) {
                    ok = laguna_graph_routed_moe_decode_rows(
                            g,
                            model,
                            l,
                            il,
                            n_tokens,
                            gate_expert_bytes,
                            gate_row_bytes,
                            up_expert_bytes,
                            up_row_bytes,
                            down_expert_bytes,
                            down_row_bytes);
                } else {
                    ok = ds4_gpu_glm_routed_moe_batch_tensor(
                            g->ffn_out,
                            g->routed_mid,
                            model->map,
                            model->size,
                            l->ffn_gate_exps->abs_offset,
                            l->ffn_up_exps->abs_offset,
                            l->ffn_down_exps->abs_offset,
                            l->ffn_gate_exps->type,
                            l->ffn_up_exps->type,
                            l->ffn_down_exps->type,
                            gate_expert_bytes,
                            gate_row_bytes,
                            up_expert_bytes,
                            up_row_bytes,
                            down_expert_bytes,
                            down_row_bytes,
                            DS4_N_EMBD,
                            DS4_N_FF_EXP,
                            DS4_N_EMBD,
                            g->router_selected,
                            g->router_weights,
                            DS4_N_EXPERT,
                            DS4_N_EXPERT_USED,
                            il,
                            g->ffn_norm,
                            n_tokens,
                            DS4_N_EXPERT_USED * DS4_N_FF_EXP,
                            true) != 0;
                }
            }
            if (ok) {
                failed_stage = "shared expert gate/up";
                ok = laguna_graph_matmul_decode_rows(
                         g->ffn_gate,
                         model,
                         l->ffn_gate_shexp,
                         g->ffn_norm,
                         n_tokens,
                         exact_q8_rows) &&
                     laguna_graph_matmul_decode_rows(
                         g->ffn_up,
                         model,
                         l->ffn_up_shexp,
                         g->ffn_norm,
                         n_tokens,
                         exact_q8_rows);
                if (ok) {
                    failed_stage = "shared expert SwiGLU";
                    ok = ds4_gpu_swiglu_tensor(
                            g->ffn_mid,
                            g->ffn_gate,
                            g->ffn_up,
                            (uint64_t)n_tokens * DS4_N_FF_SHARED,
                            0.0f,
                            1.0f) != 0;
                }
            }
            if (ok) {
                failed_stage = "shared expert down";
                ok = laguna_graph_matmul_decode_rows(
                         g->shared_out,
                         model,
                         l->ffn_down_shexp,
                         g->ffn_mid,
                         n_tokens,
                         exact_q8_rows);
            }
            if (ok) {
                failed_stage = "MoE residual";
                ok = ds4_gpu_add3_tensor(
                        g->next,
                        g->after_attn,
                        g->ffn_out,
                        g->shared_out,
                        (uint64_t)n_tokens * DS4_N_EMBD) != 0;
            }
        }

        if (ok) {
            ds4_gpu_tensor *tmp = g->cur;
            g->cur = g->next;
            g->next = tmp;
            completed_layers = il + 1u;
        }
        if (ok && live_progress && !caller_owned_draft_batch) {
            ok = ds4_gpu_end_commands() != 0;
#ifdef __APPLE__
            if (ok) {
                qk_simd32_target_command_waited = true;
                rope_atlas_target_command_waited = true;
            }
#endif
            if (ok) {
                const bool layer_is_all_work =
                    completed_layers == (uint32_t)DS4_N_LAYER &&
                    logits_out == NULL &&
                    !g->gpu_argmax_enabled;
                laguna_graph_report_prefill_display_progress(
                        display_progress,
                        display_progress_ud,
                        pos0,
                        n_tokens,
                        completed_layers,
                        display_total,
                        layer_is_all_work);
            }
            if (ok && (completed_layers < (uint32_t)DS4_N_LAYER ||
                       logits_out != NULL || g->gpu_argmax_enabled)) {
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
    }

    if (!ok) {
        fprintf(stderr,
                "ds4: Laguna batch prefill failed in %s after %u/%u layers\n",
                failed_stage,
                completed_layers,
                (unsigned)DS4_N_LAYER);
    }

    if (ok) {
        ok = laguna_graph_capture_final_feature(
                g,
                model,
                &weights->layer[DS4_N_LAYER - 1u],
                n_tokens,
                capture);
        if (!ok) failed_stage = "DFlash final feature capture";
    }

    if (ok && row_argmax_out) {
        failed_stage = "speculative output norm";
        ok = ds4_gpu_rms_norm_weight_rows_tensor(
                g->spec_output_norm,
                g->cur,
                model->map,
                model->size,
                weights->output_norm->abs_offset,
                DS4_N_EMBD,
                n_tokens,
                DS4_RMS_EPS) != 0;
        if (ok) {
            failed_stage = "speculative output projection";
            ok = laguna_graph_matmul_decode_rows(g->spec_logits,
                                                 model,
                                                 weights->output,
                                                 g->spec_output_norm,
                                                 n_tokens,
                                                 exact_q8_rows);
        }
        if (ok) {
            failed_stage = "speculative row argmax";
            ok = ds4_gpu_indexer_topk_tensor(g->spec_argmax,
                                              g->spec_logits,
                                              DS4_N_VOCAB,
                                              n_tokens,
                                              1) != 0;
        }
    }

    ds4_gpu_tensor *last = NULL;
#ifdef __APPLE__
    const bool q8_lmhead_screen =
        g->lmhead_screen != NULL && logits_out == NULL;
    g->q8_lmhead_screen_dispatched = false;
#else
    const bool q8_lmhead_screen = false;
#endif
    if (ok && (logits_out || g->gpu_argmax_enabled)) {
        last = ds4_gpu_tensor_view(
                g->cur,
                (uint64_t)(n_tokens - 1u) * DS4_N_EMBD * sizeof(float),
                (uint64_t)DS4_N_EMBD * sizeof(float));
        ok = last != NULL;
        if (ok) {
            ok = ds4_gpu_rms_norm_weight_tensor(
                    g->output_norm,
                    last,
                    model->map,
                    model->size,
                    weights->output_norm->abs_offset,
                    DS4_N_EMBD,
                    DS4_RMS_EPS) != 0;
        }
        if (ok) {
            if (q8_lmhead_screen) {
#ifdef __APPLE__
                failed_stage = "Q8 lm-head screen";
                ok = ds4_gpu_laguna_q8_lmhead_screen_tensor(
                    g->lmhead_screen,
                    g->argmax,
                    NULL,
                    model->map,
                    model->size,
                    weights->output->abs_offset,
                    g->output_norm) != 0;
                if (ok) g->q8_lmhead_screen_dispatched = true;
#endif
            } else {
                ok = laguna_graph_matmul(g->logits,
                                         model,
                                         weights->output,
                                         g->output_norm,
                                         1);
            }
        }
    }
#ifdef __APPLE__
    if (ok && g->gpu_argmax_enabled && !q8_lmhead_screen) {
        failed_stage = "GPU argmax";
        ok = ds4_gpu_laguna_argmax_tensor(g->argmax,
                                          g->logits,
                                          DS4_N_VOCAB) != 0;
    }
#endif
    /* A speculative cycle appends support-cache injection before completing
     * the shared snapshot/draft/verify command stream. */
    const bool defer_completion = ok && caller_owned_draft_batch;
    if (!caller_owned_draft_batch && ds4_gpu_commands_active()) {
        if (ok) {
            if (ds4_gpu_end_commands() == 0) {
                ok = false;
            } else {
                dense_q8_completion_waited = true;
                qk_simd32_target_command_waited = true;
                rope_atlas_target_command_waited = true;
            }
        } else {
            /* Never commit KV/attention work recorded before a later graph
             * stage failed.  Drop the shared batch transaction instead. */
            if (ds4_gpu_discard_commands() == 0) ok = false;
        }
    }
    if (ok && dense_q8_completion_waited) {
        laguna_dense_q8_gate_up_swiglu_report_waited(g);
    }
#ifdef __APPLE__
    if (ok && qk_simd32_target_evidence && !defer_completion) {
        ok = laguna_metal_qk_norm_rope_simd32_target_evidence(
            qk_simd32_target_encoded_before,
            qk_simd32_target_completed_before,
            n_tokens,
            capture != NULL,
            row_argmax_out != NULL,
            row_argmax_out != NULL ? "speculative-verifier" : "prefill",
            qk_simd32_target_command_waited);
    }
    if (ok && rope_atlas_target_evidence && !defer_completion) {
        ok = laguna_metal_rope_atlas_target_evidence(
            rope_atlas_target_generated_before,
            rope_atlas_target_generation_expected,
            rope_atlas_target_consumed_before,
            rope_atlas_target_family0_before,
            rope_atlas_target_family1_before,
            n_tokens,
            capture != NULL,
            row_argmax_out != NULL,
            row_argmax_out != NULL ? "speculative-verifier" : "prefill",
            rope_atlas_target_command_waited);
    }
    if (ok && qk_simd32_target_evidence && defer_completion) {
        g->qk_simd32_target_encoded_before =
            qk_simd32_target_encoded_before;
        g->qk_simd32_target_completed_before =
            qk_simd32_target_completed_before;
        g->qk_simd32_target_n_tokens = n_tokens;
        g->qk_simd32_target_capture = capture != NULL;
        g->qk_simd32_target_verifier = row_argmax_out != NULL;
        g->qk_simd32_target_evidence_pending = true;
    }
    if (ok && rope_atlas_target_evidence && defer_completion) {
        laguna_metal_rope_atlas_target_evidence_store(
            g,
            rope_atlas_target_generated_before,
            rope_atlas_target_generation_expected,
            rope_atlas_target_consumed_before,
            rope_atlas_target_family0_before,
            rope_atlas_target_family1_before,
            n_tokens,
            capture != NULL,
            row_argmax_out != NULL);
    }
    if (!ok) laguna_metal_target_evidence_discard(g);
#endif
    ds4_gpu_tensor_free(last);
    if (ok && g->gpu_argmax_enabled && !defer_completion) {
        ok = ds4_gpu_tensor_read(g->argmax,
                                 0,
                                 &g->gpu_argmax_result,
                                 sizeof(g->gpu_argmax_result)) != 0;
    }
#ifdef __APPLE__
    if (ok && g->q8_lmhead_screen_dispatched && !defer_completion) {
        laguna_graph_report_q8_lmhead_screen(g);
    }
#endif
    if (ok && row_argmax_out && !defer_completion) {
        ok = ds4_gpu_tensor_read(g->spec_argmax,
                                 0,
                                 row_argmax_out,
                                 (uint64_t)n_tokens *
                                     sizeof(row_argmax_out[0])) != 0;
    }
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits,
                                 0,
                                 logits_out,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok && (!live_progress || logits_out != NULL ||
               (g->gpu_argmax_enabled && !defer_completion))) {
        laguna_graph_report_prefill_display_progress(display_progress,
                                                      display_progress_ud,
                                                      pos0,
                                                      n_tokens,
                                                      DS4_N_LAYER,
                                                      display_total,
                                                      true);
    }
    ds4_gpu_set_tensor_matmul_suppressed(false);
#ifdef __APPLE__
    if (router_simd_topk_trace && !defer_completion) {
        laguna_metal_router_simd_topk_trace_report(
            "Laguna prefill/speculative batch");
    }
#endif
    if (!ok) laguna_dense_q8_gate_up_swiglu_pending_clear(g);
    return ok;
}

#if defined(__APPLE__)
static bool laguna_metal_gpu_argmax_requested(void) {
    const char *env = getenv("DS4_METAL_LAGUNA_GPU_ARGMAX");
    return env && strcmp(env, "1") == 0;
}

static bool laguna_metal_gpu_argmax_debug_forces_full_logits(void) {
    /* These diagnostics inspect or dump the complete row, so retaining the
     * host logits buffer is part of their contract. Presence matches the
     * existing DS4_TRACE_TOP behavior; dump paths require a non-empty value. */
    static const char *const presence[] = {
        "DS4_TRACE_TOP",
        "DS4_METAL_GRAPH_DUMP_PREFIX",
        "DS4_METAL_GRAPH_DUMP_NAME",
        "DS4_METAL_GRAPH_DUMP_LAYER",
        "DS4_METAL_GRAPH_DUMP_POS",
        "DS4_METAL_GRAPH_DUMP_TRACE",
        "DS4_METAL_GRAPH_TRACE_CACHE",
        "DS4_METAL_GRAPH_TRACE_COMP",
        "DS4_METAL_GRAPH_TRACE_LAYERS",
        "DS4_METAL_GRAPH_TRACE_STAGE_LAYER",
    };
    static const char *const dumps[] = {
        "DS4_METAL_DUMP_PREFILL_LOGITS",
        "DS4_METAL_GRAPH_DUMP_LOGITS",
        "DS4_CPU_DUMP_LOGITS",
        "DS4_CPU_DUMP_PREFILL_LOGITS",
    };
    for (size_t i = 0; i < sizeof(presence) / sizeof(presence[0]); i++) {
        if (getenv(presence[i]) != NULL) return true;
    }
    for (size_t i = 0; i < sizeof(dumps) / sizeof(dumps[0]); i++) {
        const char *value = getenv(dumps[i]);
        if (value && value[0]) return true;
    }
    return false;
}

static int laguna_metal_q8_lmhead_screen_v2_mode(void) {
    const char *v2 = getenv("DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2");
    /* v2 is deliberately a strict selector: it can opt in the screen on its
     * own, while malformed values fail before graph/KV state is touched. */
    if (!v2 || v2[0] == '\0' || strcmp(v2, "0") == 0) return 0;
    if (strcmp(v2, "1") == 0) return 1;
    {
        fprintf(stderr,
                "ds4: invalid DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2='%s'; "
                "expected unset, empty, 0, or literal 1\n",
                v2);
        return -1;
    }
}

/* Laguna environment switches use the historical boolean contract: an
 * unset or empty value keeps the caller's default, while any non-empty value
 * other than the exact string "0" enables the switch. */
static bool laguna_graph_env_flag(const char *name, bool dflt) {
    const char *value = name ? getenv(name) : NULL;
    if (!value || value[0] == '\0') return dflt;
    return strcmp(value, "0") != 0;
}

static bool laguna_metal_q8_lmhead_screen_requested(void) {
    const int v2_mode = laguna_metal_q8_lmhead_screen_v2_mode();
    if (v2_mode < 0) return false;
    return laguna_graph_env_flag(
               "DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN", false) ||
           v2_mode > 0;
}

/* The lm-head screen shares the dense Q8 matvec dispatch environment.  An
 * explicit alternate row topology is a real cross-feature conflict, so
 * reject it alongside the other generation preflights before allocating the
 * raw Laguna graph. */
static bool laguna_metal_q8_lmhead_screen_dispatch_env_preflight(void) {
    ds4_gpu_q8_decode_config q8_config;
    const int q8_config_valid =
        ds4_gpu_q8_decode_config_snapshot(&q8_config);
    const int rows_mode = q8_config_valid < 0 ? -1 : q8_config.q8_mv_rows;
    const char *rows_value = q8_config_valid < 0 ?
        "invalid snapshot" : (rows_mode == 2 ? "2" : "4");
    if (rows_mode != 2) {
        fprintf(stderr,
                "ds4: Laguna Q8 lm-head screen requires "
                "DS4_METAL_Q8_MV_ROWS unset/empty or literal 2 "
                "(got '%s'); refusing graph allocation\n",
                rows_value ? rows_value : "");
        return false;
    }
    if (getenv("DS4_METAL_Q8_DECODE_MPP") != NULL ||
        getenv("DS4_METAL_ENABLE_OUTPUT_Q8_NR4") != NULL) {
        fprintf(stderr,
                "ds4: Laguna Q8 lm-head screen has an incompatible "
                "Q8 dispatch environment; refusing graph allocation\n");
        return false;
    }
    return true;
}

/* The decode residual fusion is deliberately stricter than the other
 * experiment switches: unset, empty, and "0" are off; only a literal "1"
 * enables it.  Returning -1 lets every caller reject malformed values before
 * opening a command batch or touching a Laguna graph/KV cache. */
static int laguna_metal_decode_residual_norm_mode(void) {
    const char *env = getenv("DS4_METAL_LAGUNA_DECODE_RESIDUAL_NORM");
    const int mode = ds4_gpu_laguna_decode_residual_norm_env_mode(env);
    if (mode >= 0) return mode;
    fprintf(stderr,
            "ds4: invalid DS4_METAL_LAGUNA_DECODE_RESIDUAL_NORM='%s'; "
            "expected unset, empty, 0, or literal 1\n",
            env ? env : "");
    return mode;
}

static bool laguna_metal_decode_residual_norm_preflight(void) {
    const int mode = laguna_metal_decode_residual_norm_mode();
    if (mode < 0) return false;
    if (mode == 0) return true;
    if (!ds4_gpu_laguna_decode_residual_norm_available()) {
        fprintf(stderr,
                "ds4: Laguna decode residual fusion requested but "
                "kernel_add3_rms_norm_mul_f32_4 is unavailable\n");
        return false;
    }
    return true;
}

/* Public Laguna graph/session boundaries call this before allocating graph
 * scratch, opening a command batch, or changing session/cache state.  The
 * Metal entry point repeats the same probe defensively for raw callers. */
static bool laguna_metal_swa_gqa9_preflight(
        const ds4_engine *engine,
        const char       *operation,
        char             *err,
        size_t            errlen) {
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_LAGUNA) return true;
    const int mode = ds4_gpu_laguna_swa_gqa9_preflight(
        DS4_SHAPE_LAGUNA_S21.n_swa,
        DS4_SHAPE_LAGUNA_S21.n_swa,
        DS4_SHAPE_LAGUNA_S21.n_head,
        DS4_SHAPE_LAGUNA_S21.n_head_kv,
        DS4_SHAPE_LAGUNA_S21.n_head_dim);
    if (mode < 0) {
        if (err && errlen) {
            snprintf(err, errlen,
                     "%s Laguna SWA GQA9 preflight failed",
                     operation ? operation : "graph operation");
        }
        return false;
    }
    if (mode > 0 && engine && !engine->metal_ready) {
        fprintf(stderr,
                "ds4: Laguna SWA GQA9 requires a ready Metal runtime; "
                "refusing explicit opt-in fallback\n");
        if (err && errlen) {
            snprintf(err, errlen,
                     "%s Laguna SWA GQA9 requires a ready Metal runtime",
                     operation ? operation : "graph operation");
        }
        return false;
    }
    return true;
}

/* Graph-level gate for the opt-in Laguna router selector.  This must run at
 * the public graph boundary, before a command batch, capture, KV/cache write,
 * or session timeline mutation can happen.  The low-level router entry points
 * repeat the check defensively because diagnostics can call them directly. */
static bool laguna_metal_router_simd_topk_preflight(
        const ds4_engine *engine,
        const char       *operation,
        char             *err,
        size_t            errlen) {
    const int mode = ds4_gpu_laguna_router_simd_topk_preflight(
        DS4_N_EXPERT,
        DS4_N_EXPERT_USED,
        DS4_EXPERT_WEIGHT_SCALE);
    if (mode < 0) {
        if (err && errlen) {
            snprintf(err, errlen,
                     "%s Laguna router SIMD top-k preflight failed",
                     operation ? operation : "graph operation");
        }
        return false;
    }
    if (mode > 0 && engine && !engine->metal_ready) {
        fprintf(stderr,
                "ds4: Laguna router SIMD top-k requires a ready Metal runtime; "
                "refusing explicit opt-in fallback\n");
        if (err && errlen) {
            snprintf(err, errlen,
                     "%s Laguna router SIMD top-k requires a ready Metal runtime",
                     operation ? operation : "graph operation");
        }
        return false;
    }
    return true;
}

/* Admission certificate for the normal one-token fused Laguna router.  The
 * selector is frozen by laguna_metal_router_decode_fused_mode(); weight
 * layouts/ranges are checked here for every sparse layer, while the Metal
 * backend certifies the optional source PSO, TEW, and threadgroup geometry.
 * This function intentionally runs before command/KV/capture mutation at each
 * public graph boundary. */
static bool laguna_metal_router_decode_fused_preflight(
        const ds4_model   *model,
        const ds4_weights *weights) {
    const int mode = laguna_metal_router_decode_fused_mode();
    if (mode < 0) return false;
    if (mode == 0) return true;
    if (!model || !model->map || !weights) {
        fprintf(stderr,
                "ds4: Laguna fused router requested without a mapped model "
                "and bound weights\n");
        return false;
    }

    const uint64_t gate_bytes =
        (uint64_t)DS4_N_EMBD * (uint64_t)DS4_N_EXPERT * sizeof(float);
    const uint64_t bias_bytes = (uint64_t)DS4_N_EXPERT * sizeof(float);
    for (uint32_t il = DS4_N_LEADING_DENSE;
         il < (uint32_t)DS4_N_LAYER;
         il++) {
        const ds4_layer_weights *l = &weights->layer[il];
        const ds4_tensor *gate = l->ffn_gate_inp;
        const ds4_tensor *bias = l->ffn_exp_probs_b;
        if (!gate || !bias || gate->type != DS4_TENSOR_F32 ||
            gate->ndim < 2u || gate->dim[0] != DS4_N_EMBD ||
            gate->dim[1] != DS4_N_EXPERT ||
            bias->type != DS4_TENSOR_F32 || bias->ndim < 1u ||
            bias->dim[0] != DS4_N_EXPERT ||
            gate->bytes < gate_bytes || bias->bytes < bias_bytes ||
            gate->abs_offset > model->size ||
            gate->bytes > model->size - gate->abs_offset ||
            bias->abs_offset > model->size ||
            bias->bytes > model->size - bias->abs_offset) {
            fprintf(stderr,
                    "ds4: Laguna fused router requested but sparse layer "
                    "%u has unavailable or incompatible F32 router weights\n",
                    il);
            return false;
        }
    }
    return ds4_gpu_laguna_router_decode_fused_preflight(
               (uint32_t)DS4_N_EMBD,
               DS4_N_EXPERT,
               DS4_N_EXPERT_USED,
               DS4_EXPERT_WEIGHT_SCALE) != 0;
}

/* The router trace is intentionally opt-in and is kept out of timed runs.
 * The Metal kernel only enables its atomic row counters for the same literal
 * value, so a trace-off run pays neither the atomic updates nor these waits. */
static bool laguna_metal_router_simd_topk_trace_enabled(void) {
    const char *value = getenv("DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK_TRACE");
    return value && strcmp(value, "1") == 0;
}

static void laguna_metal_router_simd_topk_trace_reset(void) {
    if (laguna_metal_router_simd_topk_trace_enabled()) {
        ds4_gpu_laguna_router_simd_topk_stats_reset();
    }
}

static void laguna_metal_router_simd_topk_trace_report(const char *operation) {
    if (!laguna_metal_router_simd_topk_trace_enabled()) return;

    uint32_t optimized_rows = 0;
    uint32_t fallback_rows = 0;
    uint64_t encoded_rows = 0;
    uint64_t encoded_dispatches = 0;
    if (!ds4_gpu_laguna_router_simd_topk_stats_after_wait(
            &optimized_rows,
            &fallback_rows,
            &encoded_rows,
            &encoded_dispatches)) {
        fprintf(stderr,
                "ds4: %s Laguna router SIMD top-k trace read failed\n",
                operation ? operation : "graph operation");
        return;
    }
    fprintf(stderr,
            "ds4: %s Laguna router SIMD top-k trace "
            "optimized=%u fallback=%u encoded_rows=%llu "
            "encoded_dispatches=%llu\n",
            operation ? operation : "graph operation",
            optimized_rows,
            fallback_rows,
            (unsigned long long)encoded_rows,
            (unsigned long long)encoded_dispatches);
}

#endif

static bool laguna_graph_enable_gpu_argmax(ds4_laguna_gpu_graph *g) {
    if (!g || g->gpu_argmax_enabled) return g != NULL;
    g->argmax = ds4_gpu_tensor_alloc(sizeof(int32_t));
    if (!g->argmax) return false;
    g->scratch_bytes += sizeof(int32_t);
    g->gpu_argmax_enabled = true;
    return true;
}

#ifdef __APPLE__
static void laguna_graph_report_q8_lmhead_screen(
        const ds4_laguna_gpu_graph *g) {
    if (!g || !g->lmhead_screen) {
        return;
    }
    uint32_t candidate_rows = 0;
    uint32_t candidate_row_blocks = 0;
    uint32_t coarse_nonfinite = 0;
    uint64_t screen_calls = 0;
    uint64_t packed_bytes = 0;
    double sidecopy_init_ms = 0.0;
    uint32_t exact_row_blocks = 0;
    uint32_t compact_pair_count = 0;
    uint32_t exact_dispatch_groups = 0;
    int32_t winner_index = -1;
    float winner_value = 0.0f;
    if (ds4_gpu_laguna_q8_lmhead_screen_stats(
            g->lmhead_screen,
            &candidate_rows,
            &candidate_row_blocks,
            &coarse_nonfinite,
            &screen_calls,
            &packed_bytes,
            &sidecopy_init_ms,
            &exact_row_blocks,
            &winner_index,
            &winner_value)) {
        const int v2 = ds4_gpu_laguna_q8_lmhead_screen_v2_enabled(
            g->lmhead_screen);
        if (v2) {
            (void)ds4_gpu_laguna_q8_lmhead_screen_stats_v2(
                g->lmhead_screen,
                &compact_pair_count,
                &exact_dispatch_groups);
        }
        uint32_t winner_bits = 0;
        memcpy(&winner_bits, &winner_value, sizeof(winner_bits));
        fprintf(stderr,
                "ds4: Laguna Q8 lm-head screen stats screen_calls=%llu "
                "candidates=%u row_blocks=%u exact_row_blocks=%u "
                "nonfinite=%u packed_bytes=%llu sidecopy_init_ms=%.3f "
                "winner=%d value=%g winner_bits=0x%08x",
                (unsigned long long)screen_calls,
                candidate_rows,
                candidate_row_blocks,
                exact_row_blocks,
                coarse_nonfinite,
                (unsigned long long)packed_bytes,
                sidecopy_init_ms,
                winner_index,
                winner_value,
                winner_bits);
        if (v2) {
            fprintf(stderr,
                    " compact_pairs=%u exact_dispatch_groups=%u mode=v2-indirect",
                    compact_pair_count, exact_dispatch_groups);
        }
        fputc('\n', stderr);
    }
}
#endif

static int generate_laguna_metal_argmax(
        const ds4_model   *model,
        const ds4_vocab   *vocab,
        const ds4_weights *weights,
        const token_vec   *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud) {
    if (!prompt || prompt->len <= 0 || prompt->len >= ctx_size) {
        fprintf(stderr, "ds4: Laguna prompt is empty or leaves no context room\n");
        return 1;
    }
    if (!laguna_dense_q8_gate_up_swiglu_preflight(model, weights, NULL)) return 1;
#if defined(__APPLE__)
    if (!laguna_metal_swa_gqa9_preflight(
            NULL, "Laguna generation", NULL, 0)) return 1;
    if (!laguna_metal_router_simd_topk_preflight(
            NULL, "Laguna generation", NULL, 0)) return 1;
    if (!laguna_metal_router_decode_fused_preflight(model, weights)) return 1;
    if (laguna_metal_q8_lmhead_screen_v2_mode() < 0) return 1;
    if (!laguna_metal_decode_residual_norm_preflight()) return 1;
    if (!laguna_metal_qk_norm_rope_simd32_preflight()) return 1;
    const bool gpu_argmax_requested = laguna_metal_gpu_argmax_requested();
    const bool lmhead_screen_requested =
        laguna_metal_q8_lmhead_screen_requested();
    if (lmhead_screen_requested &&
        !laguna_metal_q8_lmhead_screen_dispatch_env_preflight()) return 1;
    if (lmhead_screen_requested && !gpu_argmax_requested) {
        fprintf(stderr,
                "ds4: Laguna Q8 lm-head screen requires "
                "DS4_METAL_LAGUNA_GPU_ARGMAX=1\n");
        return 1;
    }
    const bool debug_forces_full_logits =
        laguna_metal_gpu_argmax_debug_forces_full_logits();
    const bool gpu_argmax =
        gpu_argmax_requested && !debug_forces_full_logits;
    if (lmhead_screen_requested && gpu_argmax &&
        (!weights || !weights->output ||
         weights->output->type != DS4_TENSOR_Q8_0 ||
         weights->output->ndim < 2 ||
         weights->output->dim[0] != 3072u ||
         weights->output->dim[1] != 100352u)) {
        fprintf(stderr,
                "ds4: Laguna Q8 lm-head screen requires untied Q8_0 "
                "output[100352,3072]\n");
        return 1;
    }
    if (lmhead_screen_requested && !gpu_argmax &&
        debug_forces_full_logits) {
        fprintf(stderr,
                "ds4: Laguna Q8 lm-head screen suppressed because full-logit "
                "debug precedence is active\n");
    }
    if (gpu_argmax && !ds4_gpu_laguna_argmax_available()) {
        /* This check runs before the first graph dispatch/KV mutation. A
         * missing prerequisite is an explicit failure; later kernel failures
         * are propagated and never retried through the old path. */
        fprintf(stderr,
                "ds4: Laguna GPU argmax requested but its Metal pipeline is unavailable\n");
        return 1;
    }
#else
    const bool gpu_argmax = false;
#endif
    ds4_laguna_gpu_graph g = {0};
    if (!laguna_graph_alloc(&g, (uint32_t)ctx_size)) return 1;
    if (gpu_argmax) {
        if (!laguna_graph_enable_gpu_argmax(&g)) {
            fprintf(stderr, "ds4: failed to allocate Laguna GPU argmax output\n");
            laguna_graph_free(&g);
            return 1;
        }
        fprintf(stderr, "ds4: Laguna GPU argmax enabled (single-dispatch)\n");
#ifdef __APPLE__
        if (lmhead_screen_requested) {
            g.lmhead_screen = ds4_gpu_laguna_q8_lmhead_screen_create(
                model->map,
                model->size,
                weights->output->abs_offset,
                weights->output->dim[0],
                weights->output->dim[1]);
            if (!g.lmhead_screen) {
                fprintf(stderr,
                        "ds4: Laguna Q8 lm-head screen requested but could not be prepared\n");
                laguna_graph_free(&g);
                return 1;
            }
            fprintf(stderr,
                    "ds4: Laguna Q8 lm-head screen enabled (exact top-1 only)\n");
        }
#endif
    }
    float *logits = gpu_argmax ? NULL :
        xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    bool ok = true;
    const double prefill_t0 = now_sec();
    for (int i = 0; ok && i < prompt->len;) {
        uint32_t n = (uint32_t)(prompt->len - i);
        if (n > g.prefill_cap) n = g.prefill_cap;
        g.gpu_argmax_enabled = gpu_argmax &&
            i + (int)n == prompt->len;
        ok = laguna_graph_forward_batch(&g,
                                        model,
                                        weights,
                                        prompt->v + i,
                                        NULL,
                                        n,
                                        (uint32_t)i,
                                        !g.gpu_argmax_enabled &&
                                        i + (int)n == prompt->len ?
                                            logits : NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        prompt->len);
        i += (int)n;
        if (progress) progress(progress_ud, "prefill_chunk", i, prompt->len);
    }
    g.gpu_argmax_enabled = gpu_argmax;
    const double prefill_t1 = now_sec();

    int generated = 0;
    int successful_decode_evals = 0;
    const char *bench_stats_env = getenv("DS4_LAGUNA_BENCH_STATS");
    const bool bench_stats = bench_stats_env && !strcmp(bench_stats_env, "1");
    uint32_t pos = (uint32_t)prompt->len;
    const double decode_t0 = now_sec();
    for (int i = 0; ok && i < n_predict && pos < (uint32_t)ctx_size; i++) {
        if (!gpu_argmax && getenv("DS4_TRACE_TOP") != NULL) {
            char label[64];
            snprintf(label, sizeof(label), "Laguna step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }
        const int token = gpu_argmax ? (int)g.gpu_argmax_result :
            sample_argmax(logits, DS4_N_VOCAB);
        if (gpu_argmax &&
            (token < 0 || token >= (int)DS4_N_VOCAB)) {
            fprintf(stderr,
                    "ds4: Laguna GPU argmax returned invalid token %d\n",
                    token);
            ok = false;
            break;
        }
        if (vocab_token_is_generation_stop(vocab, token)) break;
        if (emit) emit(emit_ud, token);
        generated++;
        if (i + 1 == n_predict || pos + 1u >= (uint32_t)ctx_size) break;
        ok = laguna_graph_forward_token(
            &g, model, weights, token, pos, NULL,
            gpu_argmax ? NULL : logits);
        if (bench_stats && ok) successful_decode_evals++;
        pos++;
    }
    const double decode_t1 = now_sec();
    if (done) done(emit_ud);
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: Laguna prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_t1 > prefill_t0 ?
                (double)prompt->len / (prefill_t1 - prefill_t0) : 0.0,
            decode_t1 > decode_t0 ?
                (double)generated / (decode_t1 - decode_t0) : 0.0);
    if (bench_stats) {
        fprintf(stderr,
                "ds4: Laguna decode stats generated=%d requested=%d evals=%d\n",
                generated,
                n_predict,
                successful_decode_evals);
    }
    free(logits);
    laguna_graph_free(&g);
    return ok ? 0 : 1;
}

#endif

ds4_context_memory ds4_context_memory_estimate(int ctx_size) {
    /* Derive the values from the same dimensions and per-layer cache caps as
     * lgn_graph_alloc().  The model is always whole-mmap backed. */
    ds4_context_memory m = {0};
    const ds4_shape *shape = lgn_model_shape();
    if (!shape || ctx_size <= 0 ||
        (uint64_t)ctx_size > shape->context_length) {
        return m;
    }
    const uint32_t ctx = (uint32_t)ctx_size;
    const uint32_t prefill_cap = ctx < 16384u ? ctx : 16384u;
    const uint32_t swa_cap = ctx < shape->n_swa ? ctx : shape->n_swa;
    const uint64_t f32 = sizeof(float);
    const uint64_t q_dim = (uint64_t)shape->n_head * shape->n_head_dim;
    const uint64_t kv_dim = (uint64_t)shape->n_head_kv * shape->n_head_dim;
    const uint64_t ffn_max = shape->n_ff_dense >
        (uint64_t)shape->n_expert_used * shape->n_ff_exp ?
        shape->n_ff_dense : (uint64_t)shape->n_expert_used * shape->n_ff_exp;
    const uint64_t kv_row_bytes = 2u * kv_dim * sizeof(uint16_t);

    m.prefill_cap = prefill_cap;
    /* These legacy names now report the largest full-attention and SWA cache
     * capacities, respectively.  Laguna has no compressed KV allocation. */
    m.raw_cap = ctx;
    m.comp_cap = swa_cap;
    for (uint32_t il = 0; il < shape->n_layer; il++) {
        const uint32_t cap = lgn_model_layer_is_swa(il) ? swa_cap : ctx;
        m.raw_bytes += (uint64_t)cap * kv_row_bytes;
    }

    const uint64_t rows = prefill_cap;
    /* Keep this list in lockstep with lgn_graph_alloc().  In particular,
     * q/heads and the eight embedding-sized activation tensors are distinct
     * allocations even when their lifetimes do not overlap. */
    m.scratch_bytes = rows * sizeof(uint32_t) +
                      rows * shape->n_embd * f32 +              /* attn_norm */
                      rows * shape->n_embd * f32 * 2u +         /* cur/next */
                      rows * q_dim * f32 * 2u +                 /* q/heads */
                      rows * kv_dim * f32 * 2u +                /* k/v */
                      rows * shape->n_head * f32 +              /* gate */
                      rows * shape->n_embd * f32 * 5u +         /* attn_out..shared_out */
                      rows * ffn_max * f32 * 3u +               /* gate/up/mid */
                      rows * shape->n_expert_used * shape->n_ff_exp * f32 +
                      rows * shape->n_expert * f32 * 2u +       /* router logits/probs */
                      rows * shape->n_expert_used * sizeof(int32_t) +
                      rows * shape->n_expert_used * f32 +
                      sizeof(int32_t) + sizeof(float) +
                      rows * kv_dim * sizeof(uint16_t) * 2u +
                      shape->n_embd * f32 +
                      shape->n_vocab * f32;
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

#ifdef DS4_TEST_HOOKS
bool ds4_test_laguna_context_memory_estimator(void) {
    const ds4_shape *shape = lgn_model_shape();
    static const struct {
        int ctx;
        uint32_t prefill_cap;
        uint32_t raw_cap;
        uint32_t comp_cap;
        uint64_t raw_bytes;
        uint64_t scratch_bytes;
        uint64_t total_bytes;
    } expected[] = {
        { 1,      1,     1,      1,   196608u,      788860u,      985468u },
        { 512,    512,   512,    512, 100663296u,   192493576u,   293156872u },
        { 513,    513,   513,    512, 100712448u,  192868732u,   293581180u },
        { 16384,  16384, 16384, 512, 880803840u,   6146969608u,  7027773448u },
        { 262144, 16384, 262144, 512, 12960399360u, 6146969608u, 19107368968u },
    };
    if (!shape || shape->context_length != 262144u ||
        shape->n_swa != 512u) {
        return false;
    }
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); i++) {
        const ds4_context_memory m =
            ds4_context_memory_estimate(expected[i].ctx);
        if (m.prefill_cap != expected[i].prefill_cap ||
            m.raw_cap != expected[i].raw_cap ||
            m.comp_cap != expected[i].comp_cap ||
            m.raw_bytes != expected[i].raw_bytes ||
            m.scratch_bytes != expected[i].scratch_bytes ||
            m.total_bytes != expected[i].total_bytes ||
            m.compressed_bytes != 0u ||
            m.total_bytes != m.raw_bytes + m.scratch_bytes) {
            return false;
        }
    }
    const ds4_context_memory invalid_zero =
        ds4_context_memory_estimate(0);
    const ds4_context_memory invalid_high =
        ds4_context_memory_estimate(262145);
    return invalid_zero.total_bytes == 0u && invalid_high.total_bytes == 0u;
}

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
bool ds4_test_laguna_graph_env_flag(void) {
    static const char name[] = "DS4_TEST_LAGUNA_GRAPH_ENV_FLAG";
    const char *old = getenv(name);
    char *saved = old ? ds4_strdup(old) : NULL;
    bool ok = true;

    unsetenv(name);
    ok = ok && !laguna_graph_env_flag(name, false);
    ok = ok && laguna_graph_env_flag(name, true);
    setenv(name, "", 1);
    ok = ok && !laguna_graph_env_flag(name, false);
    ok = ok && laguna_graph_env_flag(name, true);
    setenv(name, "0", 1);
    ok = ok && !laguna_graph_env_flag(name, true);
    setenv(name, "00", 1);
    ok = ok && laguna_graph_env_flag(name, false);
    setenv(name, "false", 1);
    ok = ok && laguna_graph_env_flag(name, false);

    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
    return ok;
}
#endif
#endif

/* =========================================================================
 * Engine API and Process Lock.
 * =========================================================================
 *
 * The public entry points acquire the single instance lock, open the GGUF with
 * the backend-appropriate mmap policy, and expose tokenized prompt operations
 * to the CLI and server.
 */

bool ds4_think_mode_enabled(ds4_think_mode mode) {
    return mode == DS4_THINK_HIGH || mode == DS4_THINK_MAX;
}

const char *ds4_think_mode_name(ds4_think_mode mode) {
    switch (mode) {
    case DS4_THINK_NONE: return "none";
    case DS4_THINK_HIGH: return "high";
    case DS4_THINK_MAX:  return "max";
    }
    return "unknown";
}

const char *ds4_think_max_prefix(void) {
    return DS4_REASONING_EFFORT_MAX_PREFIX;
}

uint32_t ds4_think_max_min_context(void) {
    return DS4_THINK_MAX_MIN_CONTEXT;
}

ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size) {
    if (mode == DS4_THINK_MAX && (uint32_t)(ctx_size > 0 ? ctx_size : 0) < DS4_THINK_MAX_MIN_CONTEXT) {
        return DS4_THINK_HIGH;
    }
    return mode;
}

static void ds4_release_instance_lock(void) {
    if (g_ds4_lock_fd >= 0) {
        close(g_ds4_lock_fd);
        g_ds4_lock_fd = -1;
    }
}

/* Refuse to start a second ds4 process.  The model can map tens of GiB, so a
 * stale accidental second run is more dangerous than a normal CLI error. */
static void ds4_acquire_instance_lock(void) {
    const char *path = getenv("DS4_LOCK_FILE");
    if (!path || !path[0]) path = "/tmp/ds4.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "ds4: failed to open lock file %s: %s\n", path, strerror(errno));
        exit(2);
    }
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                char *end = NULL;
                owner = strtol(buf, &end, 10);
            }
            if (owner > 0) {
                fprintf(stderr, "ds4: another ds4 process is already running (pid %ld); refusing to start\n", owner);
            } else {
                fprintf(stderr, "ds4: another ds4 process is already running; refusing to start\n");
            }
            close(fd);
            exit(2);
        }
        fprintf(stderr, "ds4: failed to lock %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }

    if (ftruncate(fd, 0) != 0) {
        fprintf(stderr, "ds4: failed to truncate lock file %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }
    dprintf(fd, "%ld\n", (long)getpid());
    g_ds4_lock_fd = fd;
    atexit(ds4_release_instance_lock);
}

struct ds4_session {
    ds4_engine *engine;
    bool speculative_enabled;
    bool registered_with_engine;
#ifndef DS4_NO_GPU
    ds4_laguna_gpu_graph laguna_graph;
    bool laguna_graph_ready;
    ds4_dflash_gpu_graph dflash_graph;
    bool dflash_graph_ready;
    bool dflash_synced;
    uint64_t dflash_cycles;
    uint64_t dflash_proposed;
    uint64_t dflash_pruned;
    uint64_t dflash_accepted;
    double dflash_draft_ms;
    double dflash_verify_ms;
    double dflash_baseline_ms;
    uint32_t dflash_baseline_tokens;
    uint32_t dflash_cycles_since_baseline;
    double dflash_window_ms;
    uint64_t dflash_window_tokens;
    uint32_t dflash_window_cycles;
    uint32_t dflash_slow_windows;
    uint32_t dflash_active_draft;
    uint32_t dflash_deferred_rows;
    uint32_t dflash_deferred_pos0;
    bool dflash_defer_inject;
    uint32_t dflash_best_draft;
    double dflash_best_ms_per_token;
    uint32_t dflash_stage_full_accepts;
    bool dflash_suspended;
    bool dflash_guard_decided;
#endif
    token_vec checkpoint;
    float *logits;
    float *sample_probs;
    sample_arena sample_cands;
    /* Logprob observers share one logsumexp per logits state.  logits_gen is
     * bumped by every public entry point that can replace the s->logits
     * content (eval/sync/speculative-commit/payload-load/set); the cache is
     * recomputed by the first observer call after a bump. */
    uint64_t logits_gen;
    uint64_t logsumexp_gen;
    double logsumexp;
    bool logsumexp_valid;
    bool logsumexp_ok;
    double last_sample_ms;
    ds4_session_progress_fn progress;
    void *progress_ud;
    ds4_session_progress_fn display_progress;
    void *display_progress_ud;
    ds4_session_cancel_fn cancel;
    void *cancel_ud;
    uint32_t prefill_cap;
    int ctx_size;
    bool checkpoint_valid;
};

#ifdef DS4_TEST_HOOKS
/* Model-independent routing seam: the public mixed-prefill API still owns
 * validation and ordering, while these hooks substitute only the Laguna
 * sync/eval leaves.  Production builds contain neither the hooks nor their
 * counters. */
typedef int (*ds4_test_route_sync_fn)(
        ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
typedef int (*ds4_test_route_eval_fn)(
        ds4_session *s, int token, char *err, size_t errlen);
static ds4_test_route_sync_fn g_ds4_test_route_sync_fn;
static ds4_test_route_eval_fn g_ds4_test_route_eval_fn;
static uint32_t g_ds4_test_route_sync_calls;
static uint32_t g_ds4_test_route_eval_calls;
static uint32_t g_ds4_test_route_generic_eval_calls;
static DS4_MAYBE_UNUSED bool g_ds4_test_route_generic_access_forbidden;
#endif

/* Marks the session's logits content as replaced.  Every public entry point
 * that can write s->logits calls this on entry, which stales the shared
 * logsumexp cache below.  Over-marking is harmless (it only forces a
 * recompute), so entry points mark unconditionally. */
static void ds4_session_note_logits_dirty(ds4_session *s) {
    if (s) s->logits_gen++;
}

#ifdef DS4_TEST_HOOKS
static ds4_test_logprob_stats g_ds4_test_logprob_stats;

void ds4_test_logprob_stats_reset(void) {
    memset(&g_ds4_test_logprob_stats, 0,
           sizeof(g_ds4_test_logprob_stats));
}

void ds4_test_logprob_stats_get(ds4_test_logprob_stats *out) {
    if (out) *out = g_ds4_test_logprob_stats;
}
#endif

/* =========================================================================
 * Session Snapshot Payloads.
 * =========================================================================
 *
 * The server disk cache stores a high-level file header, then delegates the
 * graph-specific payload below to the engine.  This payload is intentionally
 * not mmaped: restoring a checkpoint copies bytes back into the already
 * allocated Metal tensors, preserving the same live graph buffers used by
 * normal prefill/decode.  Laguna's circular F16 key/value caches are
 * serialized in logical token order, with each layer retaining only the rows
 * live at the checkpoint.  A restore can therefore use a different physical
 * ring position without changing the next-token state.
 *
 * The payload is model-specific rather than self-describing.  The fixed header
 * records enough shape information to reject a file written for a different
 * Laguna runtime, then the body writes checkpoint tokens, last logits, and
 * per-layer key/value rows.  That is the minimum state needed for the next
 * token to match a session that had just prefetched the prefix.
 */

#define DS4_SESSION_IO_CHUNK (8u * 1024u * 1024u)

static void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}

static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}

static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

static DS4_MAYBE_UNUSED int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static DS4_MAYBE_UNUSED int payload_skip_bytes(FILE *fp, uint64_t bytes,
                                               uint8_t *buf, size_t cap,
                                               uint64_t *remaining,
                                               char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (!buf || cap == 0) {
        payload_set_err(err, errlen, "session payload skip buffer is missing");
        return 1;
    }
    const uint64_t original = bytes;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)cap ? cap : (size_t)bytes;
        if (fread(buf, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to skip session payload");
            return 1;
        }
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static DS4_MAYBE_UNUSED int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

static DS4_MAYBE_UNUSED int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}

static int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes, char *err, size_t errlen) {
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    while (bytes != 0) {
        const size_t n = bytes > DS4_SESSION_IO_CHUNK ? DS4_SESSION_IO_CHUNK : (size_t)bytes;
        if (fread(buf, 1, n, src) != n) {
            payload_set_err(err, errlen, "failed to read staged session payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            payload_set_err(err, errlen, "failed to write staged session payload");
            rc = 1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}

#ifndef DS4_NO_GPU
static uint32_t session_laguna_layer_live_rows(
        const ds4_laguna_gpu_graph *g,
        uint32_t                    layer,
        uint32_t                    checkpoint_len) {
    if (!g || layer >= DS4_N_LAYER || layer >= DS4_MAX_LAYER) return 0;
    uint32_t rows = g->cache_cap[layer];
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}

static uint64_t session_laguna_payload_live_tensor_bytes(
        const ds4_laguna_gpu_graph *g,
        uint32_t                    checkpoint_len) {
    const uint64_t row_bytes =
        (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint64_t rows =
            session_laguna_layer_live_rows(g, il, checkpoint_len);
        if (rows > (UINT64_MAX - bytes) / (2u * row_bytes)) return 0;
        bytes += rows * 2u * row_bytes;
    }
    return bytes;
}

/* Accelerator tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
static int payload_write_tensor_span(FILE *fp, const ds4_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (ds4_gpu_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read accelerator session tensor");
            return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

static int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (ds4_gpu_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore accelerator session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

/* Persist a circular F16 cache in logical token order.  A restored session can
 * use a different context allocation while retaining the same last-window
 * state because the file layout does not expose the source ring position. */
static int payload_write_laguna_ring(FILE *fp,
                                     const ds4_gpu_tensor *tensor,
                                     uint32_t cap, uint32_t logical_first,
                                     uint32_t rows, uint64_t row_bytes,
                                     uint8_t *buf, size_t buf_cap,
                                     char *err, size_t errlen) {
    if (rows == 0) return 0;
    if (!tensor || cap == 0 || rows > cap) {
        payload_set_err(err, errlen, "invalid Laguna KV ring snapshot");
        return 1;
    }
    const uint32_t physical_first = logical_first % cap;
    uint32_t first_rows = cap - physical_first;
    if (first_rows > rows) first_rows = rows;
    int rc = payload_write_tensor_span(fp,
                                       tensor,
                                       (uint64_t)physical_first * row_bytes,
                                       (uint64_t)first_rows * row_bytes,
                                       buf,
                                       buf_cap,
                                       err,
                                       errlen);
    if (rc == 0 && first_rows < rows) {
        rc = payload_write_tensor_span(fp,
                                       tensor,
                                       0,
                                       (uint64_t)(rows - first_rows) * row_bytes,
                                       buf,
                                       buf_cap,
                                       err,
                                       errlen);
    }
    return rc;
}

static int payload_read_laguna_ring(FILE *fp,
                                    ds4_gpu_tensor *tensor,
                                    uint32_t cap, uint32_t logical_first,
                                    uint32_t rows, uint64_t row_bytes,
                                    uint8_t *buf, size_t buf_cap,
                                    uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (rows == 0) return 0;
    if (!tensor || cap == 0 || rows > cap) {
        payload_set_err(err, errlen, "invalid Laguna KV ring restore");
        return 1;
    }
    const uint32_t physical_first = logical_first % cap;
    uint32_t first_rows = cap - physical_first;
    if (first_rows > rows) first_rows = rows;
    int rc = payload_read_tensor_span(fp,
                                      tensor,
                                      (uint64_t)physical_first * row_bytes,
                                      (uint64_t)first_rows * row_bytes,
                                      buf,
                                      buf_cap,
                                      remaining,
                                      err,
                                      errlen);
    if (rc == 0 && first_rows < rows) {
        rc = payload_read_tensor_span(fp,
                                      tensor,
                                      0,
                                      (uint64_t)(rows - first_rows) * row_bytes,
                                      buf,
                                      buf_cap,
                                      remaining,
                                      err,
                                      errlen);
    }
    return rc;
}

static DS4_MAYBE_UNUSED int payload_write_tensor_span_f16_as_f32(FILE *fp, const ds4_gpu_tensor *tensor,
                                                                 uint64_t offset_f16, uint64_t count,
                                                                 uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > ds4_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > ds4_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (ds4_gpu_tensor_read(tensor, offset_f16 + done * sizeof(uint16_t),
                                h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to read Metal F16 session tensor");
            return 1;
        }
        for (size_t i = 0; i < n; i++) f[i] = f16_to_f32(h[i]);
        if (payload_write_bytes(fp, f, (uint64_t)n * sizeof(float), err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

static DS4_MAYBE_UNUSED int payload_read_tensor_span_f32_as_f16(FILE *fp, ds4_gpu_tensor *tensor,
                                                                uint64_t offset_f16, uint64_t count,
                                                                uint8_t *buf, size_t cap, uint64_t *remaining,
                                                                char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > ds4_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > ds4_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (payload_read_bytes(fp, f, (uint64_t)n * sizeof(float), remaining, err, errlen) != 0) return 1;
        for (size_t i = 0; i < n; i++) h[i] = f32_to_f16(f[i]);
        if (ds4_gpu_tensor_write(tensor, offset_f16 + done * sizeof(uint16_t),
                                 h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to restore Metal F16 session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

#endif

static bool ds4_session_is_laguna(const ds4_session *s) {
    return s && s->engine && DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA;
}

#ifndef DS4_NO_GPU
/* The DFlash support graph owns a separate KV/feature history from the
 * Laguna target graph.  Any operation that invalidates the target timeline
 * must invalidate this history as one unit too; in particular, a deferred
 * feature injection must never survive across a target payload restore. */
static void ds4_session_dflash_invalidate(ds4_session *s) {
    if (!s) return;
    s->dflash_synced = false;
    s->dflash_deferred_rows = 0;
    s->dflash_deferred_pos0 = 0;
    s->dflash_defer_inject = false;
}

#endif

static uint32_t ds4_model_normal_layer_count(void) {
    return DS4_N_LAYER <= DS4_MAX_LAYER ? (uint32_t)DS4_N_LAYER : 0;
}

static bool ds4_layer_payload_range_valid(uint32_t layer_start, uint32_t layer_end) {
    const uint32_t n_layers = ds4_model_normal_layer_count();
    return n_layers != 0 && layer_start <= layer_end && layer_end < n_layers;
}

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end) {
    if (!s || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end))
        return 0;
    if (ds4_session_is_laguna(s)) return 0;
    return 0;
}

int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload save");
        return 1;
    }
    if (ds4_session_is_laguna(s)) {
        payload_set_err(err, errlen,
                        "Laguna layer snapshots are not supported");
        return 1;
    }
    payload_set_err(err, errlen, "generic graph layer payloads are unsupported");
    return 1;
}

int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !tokens ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload load");
        return 1;
    }
    if (ds4_session_is_laguna(s)) {
        payload_set_err(err, errlen,
                        "Laguna layer restores are not supported");
        return 1;
    }
    (void)fp;
    (void)payload_bytes;
    (void)tokens;
    (void)n_tokens;
    payload_set_err(err, errlen, "generic graph layer payloads are unsupported");
    return 1;
}

int ds4_engine_routed_quant_bits(ds4_engine *e) {
    if (!e) return 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_tensor *gate = e->weights.layer[il].ffn_gate_exps;
        if (!gate) continue;
        return gate->type == DS4_TENSOR_Q4_K ? 4 : 2;
    }
    return 0;
}

bool ds4_engine_has_output_head(ds4_engine *e) {
    return e && weights_have_output_head(&e->weights);
}

bool ds4_engine_has_dflash(ds4_engine *e) {
    return e && e->dflash_ready;
}

int ds4_engine_dflash_draft_tokens(ds4_engine *e) {
    return ds4_engine_has_dflash(e) ? e->dflash_draft_tokens : 0;
}

const ds4_tokens *ds4_session_tokens(ds4_session *s) {
    return s ? &s->checkpoint : NULL;
}


uint64_t ds4_session_payload_bytes(ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    if (ds4_session_is_laguna(s)) {
#ifdef DS4_NO_GPU
        return 0;
#else
        if (!s->laguna_graph_ready) return 0;
        uint64_t bytes =
            (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
        bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
        const uint64_t kv_bytes = session_laguna_payload_live_tensor_bytes(
            &s->laguna_graph, (uint32_t)s->checkpoint.len);
        if (kv_bytes == 0 && s->checkpoint.len != 0) return 0;
        if (bytes > UINT64_MAX - kv_bytes) return 0;
        return bytes + kv_bytes;
#endif
    }
    return 0;
}

int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        payload_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        payload_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
    return rc;
}

void ds4_session_payload_file_free(ds4_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}

int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen) {
    if (!out) {
        payload_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    char tmpl[] = "/tmp/ds4-session-payload.XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        payload_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        if (errlen) snprintf(err, errlen, "failed to open staged session payload: %s",
                             strerror(saved));
        return 1;
    }

    int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        payload_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            payload_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        return 1;
    }
    out->path = ds4_strdup(tmpl);
    out->bytes = (uint64_t)pos;
    return 0;
}

int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (ds4_session_is_laguna(s)) {
#ifdef DS4_NO_GPU
        payload_set_err(err, errlen, "graph backend support is not compiled in");
        return 1;
#else
        if (!s->laguna_graph_ready) {
            payload_set_err(err, errlen, "Laguna graph is not ready for snapshot");
            return 1;
        }
        if (ds4_gpu_synchronize() == 0) {
            payload_set_err(err, errlen,
                            "failed to synchronize accelerator before Laguna snapshot");
            return 1;
        }

        ds4_laguna_gpu_graph *g = &s->laguna_graph;
        const uint32_t checkpoint_len = (uint32_t)s->checkpoint.len;
        /* Laguna payload fields identify its heterogeneous attention layout:
         * context, SWA window, full/SWA query heads, KV heads, token count,
         * layers, head width, rotary width, vocabulary, and KV element bytes. */
        uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
            DS4_SESSION_PAYLOAD_MAGIC,
            DS4_SESSION_PAYLOAD_VERSION,
            (uint32_t)s->ctx_size,
            DS4_N_SWA,
            ds4_layer_head_count(0),
            ds4_layer_head_count(1),
            DS4_N_HEAD_KV,
            checkpoint_len,
            DS4_N_LAYER,
            DS4_N_HEAD_DIM,
            DS4_N_ROT,
            DS4_N_VOCAB,
            (uint32_t)sizeof(uint16_t),
        };
        for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
            if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
        }
        for (int i = 0; i < s->checkpoint.len; i++) {
            if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i],
                                  err, errlen) != 0) return 1;
        }
        if (payload_write_bytes(fp,
                                s->logits,
                                (uint64_t)DS4_N_VOCAB * sizeof(float),
                                err,
                                errlen) != 0) return 1;

        const uint64_t row_bytes =
            (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
        uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
        int rc = 0;
        for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
            const uint32_t live =
                session_laguna_layer_live_rows(g, il, checkpoint_len);
            const uint32_t logical_first = checkpoint_len - live;
            rc = payload_write_laguna_ring(fp,
                                           g->key_cache[il],
                                           g->cache_cap[il],
                                           logical_first,
                                           live,
                                           row_bytes,
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
            if (rc == 0) {
                rc = payload_write_laguna_ring(fp,
                                               g->value_cache[il],
                                               g->cache_cap[il],
                                               logical_first,
                                               live,
                                               row_bytes,
                                               buf,
                                               DS4_SESSION_IO_CHUNK,
                                               err,
                                               errlen);
            }
        }
        free(buf);
        return rc;
#endif
    }
    payload_set_err(err, errlen, "generic graph payload is unsupported");
    return 1;
}

int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!s) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
#ifndef DS4_NO_GPU
    /* Payload restore replaces the Laguna target timeline.  Invalidate the
     * independent DFlash support history before reading or validating any
     * payload bytes so every failed load attempt is safe too. */
    if (ds4_session_is_laguna(s)) ds4_session_dflash_invalidate(s);
#endif
    if (!fp) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    ds4_session_note_logits_dirty(s);
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    if (ds4_session_is_laguna(s)) {
#ifdef DS4_NO_GPU
        payload_set_err(err, errlen, "graph backend support is not compiled in");
        return 1;
#else
        if (!s->laguna_graph_ready) {
            payload_set_err(err, errlen, "Laguna graph is not ready for KV restore");
            return 1;
        }
        const uint32_t saved_ctx = h[2];
        const uint32_t saved_tokens = h[7];
        if (saved_ctx > (uint32_t)s->ctx_size ||
            saved_tokens >= (uint32_t)s->ctx_size ||
            saved_tokens >= saved_ctx) {
            payload_set_err(err, errlen,
                            "Laguna KV checkpoint does not fit current context");
            return 1;
        }
        if (h[3] != DS4_N_SWA ||
            h[4] != ds4_layer_head_count(0) ||
            h[5] != ds4_layer_head_count(1) ||
            h[6] != DS4_N_HEAD_KV ||
            h[8] != DS4_N_LAYER ||
            h[9] != DS4_N_HEAD_DIM ||
            h[10] != DS4_N_ROT ||
            h[11] != DS4_N_VOCAB ||
            h[12] != sizeof(uint16_t)) {
            payload_set_err(err, errlen,
                            "KV checkpoint was written for a different Laguna layout");
            return 1;
        }

        token_vec new_checkpoint = {0};
        for (uint32_t i = 0; i < saved_tokens; i++) {
            uint32_t tok = 0;
            if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (tok >= DS4_N_VOCAB) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen,
                                "Laguna KV checkpoint contains an invalid token");
                return 1;
            }
            token_vec_push(&new_checkpoint, (int)tok);
        }
        if (payload_read_bytes(fp,
                               s->logits,
                               (uint64_t)DS4_N_VOCAB * sizeof(float),
                               &remaining,
                               err,
                               errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (ds4_gpu_synchronize() == 0) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen,
                            "failed to synchronize accelerator before Laguna KV restore");
            return 1;
        }

        s->checkpoint_valid = false;
        ds4_laguna_gpu_graph *g = &s->laguna_graph;
        const uint64_t row_bytes =
            (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
        uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
        int rc = 0;
        for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
            const uint32_t live =
                session_laguna_layer_live_rows(g, il, saved_tokens);
            const uint32_t logical_first = saved_tokens - live;
            rc = payload_read_laguna_ring(fp,
                                          g->key_cache[il],
                                          g->cache_cap[il],
                                          logical_first,
                                          live,
                                          row_bytes,
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) {
                rc = payload_read_laguna_ring(fp,
                                              g->value_cache[il],
                                              g->cache_cap[il],
                                              logical_first,
                                              live,
                                              row_bytes,
                                              buf,
                                              DS4_SESSION_IO_CHUNK,
                                              &remaining,
                                              err,
                                              errlen);
            }
        }
        free(buf);
        if (rc != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (remaining != 0) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen,
                            "Laguna KV checkpoint has trailing payload bytes");
            return 1;
        }
        if (ds4_gpu_synchronize() == 0) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen,
                            "failed to synchronize accelerator after Laguna KV restore");
            return 1;
        }

        token_vec_free(&s->checkpoint);
        s->checkpoint = new_checkpoint;
        s->checkpoint_valid = true;
        return 0;
#endif
    }
    payload_set_err(err, errlen, "generic graph payload is unsupported");
    return 1;
}

int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap) {
        payload_set_err(err, errlen, "invalid session snapshot save");
        return 1;
    }
    const uint64_t bytes = ds4_session_payload_bytes(s);
    if (bytes == 0) {
        payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    if (bytes > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    if (snap->cap < bytes) {
        uint8_t *p = realloc(snap->ptr, (size_t)bytes);
        if (!p) {
            payload_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = bytes;
    }

    FILE *fp = fmemopen(snap->ptr, (size_t)bytes, "wb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot");
        return 1;
    }
    const int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to finalize memory session snapshot");
        return 1;
    }
    if (rc != 0) return 1;
    snap->len = bytes;
    return 0;
}

int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap || !snap->ptr || snap->len == 0) {
        payload_set_err(err, errlen, "invalid session snapshot load");
        return 1;
    }
    ds4_session_note_logits_dirty(s);
    if (snap->len > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }

    FILE *fp = fmemopen((void *)snap->ptr, (size_t)snap->len, "rb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot restore");
        return 1;
    }
    const int rc = ds4_session_load_payload(s, fp, snap->len, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}

void ds4_session_snapshot_free(ds4_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

#ifdef DS4_TEST_HOOKS
#ifndef DS4_NO_GPU
typedef struct {
    ds4_shape saved_shape;
} ds4_test_laguna_shape_scope;

static void ds4_test_laguna_shape_scope_begin(
        ds4_test_laguna_shape_scope *scope) {
    if (!scope) return;
    scope->saved_shape = g_ds4_shape;
    /* The malformed payload is rejected before any layout fields are read;
     * the family gate is the minimum Laguna state needed by this seam. */
    g_ds4_shape.family = DS4_MODEL_FAMILY_LAGUNA;
}

static void ds4_test_laguna_shape_scope_end(
        const ds4_test_laguna_shape_scope *scope) {
    if (!scope) return;
    g_ds4_shape = scope->saved_shape;
}

/* Exercise the payload-restore boundary without opening a model or allocating
 * the production Laguna/DFlash graphs.  A malformed payload is sufficient:
 * the loader must invalidate already-synced/deferred support state before it
 * can reject the header, and snapshot restore must reach that same loader. */
bool ds4_test_dflash_payload_invalidation(void) {
    ds4_test_laguna_shape_scope shape_scope;
    ds4_test_laguna_shape_scope_begin(&shape_scope);

    ds4_engine engine;
    memset(&engine, 0, sizeof(engine));

    ds4_session session;
    memset(&session, 0, sizeof(session));
    session.engine = &engine;

    const uint32_t bad_payload[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC,
        DS4_SESSION_PAYLOAD_VERSION + 1u,
    };
    char err[128] = {0};
    FILE *payload_fp = NULL;
    bool payload_cleared = false;
    bool snapshot_cleared = false;
    int payload_rc = 1;
    int snapshot_rc = 1;

    session.dflash_synced = true;
    session.dflash_deferred_rows = 3u;
    session.dflash_deferred_pos0 = 41u;
    session.dflash_defer_inject = true;
    payload_fp = fmemopen((void *)bad_payload, sizeof(bad_payload), "rb");
    if (!payload_fp) goto cleanup;
    payload_rc = ds4_session_load_payload(
        &session, payload_fp, sizeof(bad_payload), err, sizeof(err));
    payload_cleared =
        payload_rc != 0 &&
        !session.dflash_synced &&
        session.dflash_deferred_rows == 0u &&
        session.dflash_deferred_pos0 == 0u &&
        !session.dflash_defer_inject;
    fclose(payload_fp);
    payload_fp = NULL;

    session.dflash_synced = true;
    session.dflash_deferred_rows = 2u;
    session.dflash_deferred_pos0 = 17u;
    session.dflash_defer_inject = true;
    ds4_session_snapshot snap = {
        .ptr = (uint8_t *)(uintptr_t)bad_payload,
        .len = sizeof(bad_payload),
        .cap = sizeof(bad_payload),
    };
    memset(err, 0, sizeof(err));
    snapshot_rc = ds4_session_load_snapshot(
        &session, &snap, err, sizeof(err));
    snapshot_cleared =
        snapshot_rc != 0 &&
        !session.dflash_synced &&
        session.dflash_deferred_rows == 0u &&
        session.dflash_deferred_pos0 == 0u &&
        !session.dflash_defer_inject;

cleanup:
    if (payload_fp) fclose(payload_fp);
    ds4_test_laguna_shape_scope_end(&shape_scope);
    return payload_cleared && snapshot_cleared;
}
#endif /* !DS4_NO_GPU */
#endif /* DS4_TEST_HOOKS */

void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens) {
    dump_tokens(&e->vocab, tokens);
}

int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    ds4_model model;
    ds4_vocab vocab;
    token_vec tokens = {0};

    if (!fp) fp = stdout;
    model_open(&model, model_path, false, false);
    config_validate_model(&model);
    vocab_load(&vocab, &model);
    tokenize_rendered_chat_vocab(&vocab, text ? text : "", &tokens);

    dump_tokens_fp(fp, &vocab, &tokens);
    token_vec_free(&tokens);
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}

/* Laguna evaluation owns its KV timeline and logits buffer.  Its argmax entry
 * point must use ds4_session_eval(), never the unsupported generic evaluator
 * below. */
static bool ds4_session_eval_argmax_uses_laguna_path(const ds4_session *s) {
    return ds4_session_is_laguna(s);
}

#ifdef DS4_TEST_HOOKS
typedef int (*ds4_test_argmax_eval_fn)(
        ds4_session *s, int token, char *err, size_t errlen);
static ds4_test_argmax_eval_fn g_ds4_test_argmax_eval_fn;
static uint32_t g_ds4_test_argmax_laguna_eval_calls;
#endif

/* Keep the public argmax boundary intact while letting model-independent
 * tests substitute only the selected session evaluator.  The production path
 * remains a direct call to ds4_session_eval(). */
static int ds4_session_eval_argmax_dispatch_eval(
        ds4_session *s,
        int           token,
        char         *err,
        size_t        errlen,
        bool          laguna_path) {
#ifdef DS4_TEST_HOOKS
    if (laguna_path) {
        g_ds4_test_argmax_laguna_eval_calls++;
        if (g_ds4_test_argmax_eval_fn) {
            return g_ds4_test_argmax_eval_fn(s, token, err, errlen);
        }
    }
#else
    (void)laguna_path;
#endif
    return ds4_session_eval(s, token, err, errlen);
}

int ds4_session_eval_argmax(ds4_session *s, int token, char *err, size_t errlen) {
    if (!s) return -1;
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (!laguna_metal_swa_gqa9_preflight(
            s->engine, "session argmax", err, errlen)) return -1;
#endif
    ds4_session_note_logits_dirty(s);
    const bool laguna_path = ds4_session_eval_argmax_uses_laguna_path(s);
    if (laguna_path) {
        if (ds4_session_eval_argmax_dispatch_eval(
                s, token, err, errlen, laguna_path) != 0) return -1;
        return ds4_session_argmax(s);
    }
#ifdef DS4_TEST_HOOKS
    g_ds4_test_route_generic_eval_calls++;
#endif
    snprintf(err, errlen, "generic graph argmax is unsupported");
    return -1;
}


/* Speculative decode state machine:
 * 1. commit the normal target token and use its logits to validate draft[0];
 * 2. let the DFlash support graph draft a tiny suffix from its own raw-cache frontier;
 * 3. verify the suffix with the target graph, committing only the accepted
 *    prefix and rolling back speculative Metal state on miss;
 * 4. fall back to ordinary one-token decode if the fast verifier cannot prove
 *    the target stream. */

int ds4_engine_generate_argmax(
        ds4_engine        *e,
        const ds4_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud) {
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (!laguna_metal_swa_gqa9_preflight(
            e, "generation", NULL, 0)) return 1;
    if (!laguna_metal_router_simd_topk_preflight(
            e, "generation", NULL, 0)) return 1;
#endif
#ifndef DS4_NO_GPU
    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal generation requested but the runtime is unavailable\n");
        return 1;
    }
    return generate_laguna_metal_argmax(model, vocab, weights, prompt,
                                        n_predict, ctx_size,
                                        emit, done, emit_ud,
                                        progress, progress_ud);
#else
    (void)e;
    (void)prompt;
    (void)n_predict;
    (void)ctx_size;
    (void)emit;
    (void)done;
    (void)emit_ud;
    (void)progress;
    (void)progress_ud;
    fprintf(stderr, "ds4: Metal generation is unavailable in this host-only build\n");
    return 1;
#endif
}


#ifdef DS4_TEST_HOOKS

int ds4_test_session_read_logits(ds4_session *s, float *out,
                                 uint64_t out_bytes) {
    if (!s || !out ||
        out_bytes < (uint64_t)DS4_N_VOCAB * sizeof(float)) {
        return 1;
    }
    return ds4_session_copy_logits(s, out, (int)DS4_N_VOCAB) ==
                   (int)DS4_N_VOCAB ? 0 : 1;
}

#endif /* DS4_TEST_HOOKS */

static int ds4_engine_open_internal(ds4_engine **out,
                                    const ds4_engine_options *opt);

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    return ds4_engine_open_internal(out, opt);
}

static int ds4_engine_open_internal(ds4_engine **out,
                                     const ds4_engine_options *opt) {
    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->dflash_model.fd = -1;
    e->quality = opt->quality;
    if (opt->dflash_draft_tokens < 0 ||
        opt->dflash_draft_tokens >= DS4_DFLASH_BLOCK_SIZE) {
        fprintf(stderr,
                "ds4: --dflash-draft must be between 1 and %u\n",
                DS4_DFLASH_BLOCK_SIZE - 1u);
        free(e);
        *out = NULL;
        return 1;
    }
    e->dflash_draft_tokens = opt->dflash_draft_tokens;
    if (e->dflash_draft_tokens <= 0) {
        e->dflash_draft_tokens = 3;
    }
    if (opt->dflash_p_min_set &&
        (!isfinite(opt->dflash_p_min) ||
         opt->dflash_p_min < 0.0f || opt->dflash_p_min > 1.0f)) {
        fprintf(stderr,
                "ds4: --dflash-p-min must be between 0 and 1\n");
        free(e);
        *out = NULL;
        return 1;
    }
    e->dflash_p_min =
        opt->dflash_p_min_set ? opt->dflash_p_min : 0.4f;
    if (opt->n_threads > 0) g_requested_threads = (uint32_t)opt->n_threads;
    ds4_acquire_instance_lock();

#ifdef DS4_TEST_HOOKS
    g_ds4_test_engine_model_open_calls++;
#endif
    model_open(&e->model, opt->model_path, true, false);
    /* Admit the model before weight warming or graph setup.
     * config_validate_model fails closed before legacy family selection can
     * mutate the global shape state. */
    config_validate_model(&e->model);
    if (opt->warm_weights) model_warm_weights(&e->model);
    weights_bind(&e->weights, &e->model);

    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA) {
        if (opt->inspect_only) {
            if (opt->dflash_path && opt->dflash_path[0]) {
                model_open(&e->dflash_model, opt->dflash_path, false, false);
                e->support_kind =
                    support_model_detect(&e->dflash_model,
                                         &e->support_stages);
                if (e->support_kind != DS4_SUPPORT_DFLASH) {
                    fprintf(stderr,
                            "ds4: unsupported --dflash support model %s; "
                            "expected architecture=dflash\n",
                            opt->dflash_path);
                    ds4_engine_close(e);
                    *out = NULL;
                    return 1;
                }
            }
            *out = e;
            return 0;
        }
        vocab_load(&e->vocab, &e->model);
    } else if (!opt->inspect_only) {
        if (opt->dflash_path && opt->dflash_path[0]) {
            fprintf(stderr,
                    "ds4: --dflash is supported only with Laguna S 2.1\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        vocab_load(&e->vocab, &e->model);
    }
    const char *support_path = opt->dflash_path;
    if (support_path && support_path[0]) {
        ds4_model *support_model = &e->dflash_model;
        model_open(support_model, support_path, true, false);
        e->support_kind =
            support_model_detect(support_model, &e->support_stages);
        if (e->support_kind != DS4_SUPPORT_DFLASH) {
            fprintf(stderr,
                    "ds4: unsupported --dflash support model %s; "
                    "expected architecture=dflash\n",
                    support_path);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        lgn_dflash_weights_bind(&e->dflash_weights, &e->dflash_model);
        if (e->dflash_weights.fc->type == LGN_TENSOR_BF16) {
            e->dflash_f16_map = lgn_dflash_prepare_f16_map(
                &e->dflash_model,
                ds4_lgn_dflash_parallel_for,
                NULL);
            if (!e->dflash_f16_map) {
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            e->dflash_f16_map_size = e->dflash_model.size;
        }
        e->dflash_ready = true;
        fprintf(stderr,
                "ds4: Laguna DFlash support loaded: %s "
                "(weights=%s, draft=%d, p-min=%.2f, block=%u, "
                "cache=%u)\n",
                support_path,
                lgn_model_tensor_type_name(e->dflash_weights.fc->type),
                e->dflash_draft_tokens,
                e->dflash_p_min,
                e->dflash_weights.block_size,
                DS4_DFLASH_CACHE_CAP);
    }

#ifndef DS4_NO_GPU
    /* The supported runtime is always the whole-model Laguna Metal graph. */
    e->metal_ready = ds4_gpu_init() != 0;
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal runtime unavailable; aborting startup\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    ds4_gpu_set_quality(e->quality);
    const uint64_t model_tensor_bytes =
        e->model.size > e->model.tensor_data_pos ?
            e->model.size - e->model.tensor_data_pos : 0;
    e->startup_model_span_bytes = model_tensor_bytes;
    const int model_map_ok = ds4_gpu_set_model_map_range(
        e->model.map,
        e->model.size,
        e->model.tensor_data_pos,
        model_tensor_bytes,
        e->model.max_tensor_bytes);
    if (!model_map_ok) {
        fprintf(stderr,
                "ds4: Metal failed to map model views; aborting startup. "
                "This is commonly caused by insufficient memory or accelerator VM budget.\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    const bool support_model_runtime_ready = e->dflash_ready;
    const ds4_model *support_model = &e->dflash_model;
    const void *support_model_map = lgn_dflash_weight_map(
        support_model, e->dflash_f16_map);
    const uint64_t support_model_map_size = lgn_dflash_weight_map_size(
        support_model, e->dflash_f16_map, e->dflash_f16_map_size);
    if (support_model_runtime_ready &&
        !ds4_gpu_set_model_map_range(support_model_map,
                                      support_model_map_size,
                                      support_model->tensor_data_pos,
                                      support_model_map_size >
                                              support_model->tensor_data_pos ?
                                          support_model_map_size -
                                              support_model->tensor_data_pos : 0,
                                      support_model->max_tensor_bytes)) {
        fprintf(stderr,
                "ds4: Metal failed to map support model views; aborting startup. "
                "This is commonly caused by insufficient memory or accelerator VM budget.\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    (void)ds4_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
    if (support_model_runtime_ready) {
        /* A BF16 DFlash support map is an anonymous F16 shadow, so it has no
         * backing descriptor; quantized support weights remain file-backed. */
        const int support_model_fd = lgn_dflash_weight_map_fd(
            support_model, e->dflash_f16_map);
        (void)ds4_gpu_set_model_fd_for_map(support_model_fd,
                                           support_model_map);
        (void)ds4_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
    }
    fprintf(stderr, "ds4: Metal runtime initialized for Laguna graph\n");
#else
    fprintf(stderr, "ds4: Metal runtime is unavailable in this host-only build\n");
    ds4_engine_close(e);
    *out = NULL;
    return 1;
#endif

    if (!opt->inspect_only) {
        ds4_engine_print_startup_memory(e, opt->context_size);
    }
    *out = e;
    return 0;
}

void ds4_engine_summary(ds4_engine *e) {
    model_summary(&e->model);
    if (e->dflash_model.map) {
        printf("\nsupport model");
        if (e->support_kind != DS4_SUPPORT_NONE) {
            printf(" (%s", support_kind_name(e->support_kind));
            if (e->support_stages) printf(", stages=%u", e->support_stages);
            printf(")");
        }
        printf(":\n");
        model_summary(&e->dflash_model);
    }
}

int ds4_engine_vocab_size(ds4_engine *e) {
    return e ? e->vocab.n_vocab : 0;
}

const char *ds4_engine_model_name(ds4_engine *e) {
    (void)e;
    return DS4_MODEL_SHAPE_NAME;
}

int ds4_engine_layer_count(ds4_engine *e) {
    (void)e;
    return (int)DS4_N_LAYER;
}

int ds4_engine_model_id(ds4_engine *e) {
    (void)e;
    return (int)DS4_MODEL_VARIANT;
}

bool ds4_engine_is_laguna(ds4_engine *e) {
    (void)e;
    return DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA;
}

const char *ds4_engine_default_system_prompt(ds4_engine *e) {
    (void)e;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA) {
        return "You are a helpful, conversationally-fluent assistant made by Poolside. "
               "You are here to be helpful to users through natural language conversations.";
    }
    return "You are a helpful assistant";
}

void ds4_engine_sampling_defaults(ds4_engine *e, float *temperature,
                                  int *top_k, float *top_p, float *min_p) {
    (void)e;
    if (!temperature || !top_k || !top_p || !min_p) return;
    *temperature = DS4_DEFAULT_TEMPERATURE;
    *top_k = 0;
    *top_p = DS4_DEFAULT_TOP_P;
    *min_p = DS4_DEFAULT_MIN_P;

    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA) {
        *temperature = 0.7f;
        *top_k = 20;
        *top_p = 0.95f;
        *min_p = 0.05f;
    }
}

static bool ds4_session_register_with_engine(ds4_session *s) {
    ds4_engine *e = s ? s->engine : NULL;
    if (!e || s->registered_with_engine) return false;
    if (e->closing) {
        fprintf(stderr,
                "ds4: cannot register a session while the engine is closing\n");
        return false;
    }
    if (e->live_sessions == SIZE_MAX) {
        fprintf(stderr, "ds4: engine live-session count overflow\n");
        return false;
    }
    e->live_sessions++;
    s->registered_with_engine = true;
    return true;
}

static void ds4_session_unregister_from_engine(ds4_session *s) {
    if (!s || !s->registered_with_engine) return;
    ds4_engine *e = s->engine;
    s->registered_with_engine = false;
    if (!e) {
        fprintf(stderr,
                "ds4: registered session lost its engine during teardown\n");
        return;
    }
    if (e->live_sessions == 0) {
        fprintf(stderr,
                "ds4: engine live-session count was already zero during session teardown\n");
        return;
    }
    e->live_sessions--;
}

static int ds4_session_publish(ds4_session **out, ds4_session *s) {
    if (!out || !s || !ds4_session_register_with_engine(s)) {
        ds4_session_free(s);
        return 1;
    }
    *out = s;
    return 0;
}

void ds4_engine_close(ds4_engine *e) {
    if (!e) return;
    if (e->closing) {
        fprintf(stderr, "ds4: engine close is already in progress\n");
        return;
    }
    if (e->live_sessions != 0) {
        fprintf(stderr,
                "ds4: refusing to close engine with %zu live session%s\n",
                e->live_sessions,
                e->live_sessions == 1 ? "" : "s");
        return;
    }
    e->closing = true;
    ds4_engine_close_note(e, DS4_ENGINE_CLOSE_BEGIN);

    /* Model tensors are exposed to graph backends as no-copy views.  Drain
     * submitted work and destroy every backend cache while the main, support,
     * and optional shadow mappings are all still valid. */
    ds4_engine_close_note(e, DS4_ENGINE_CLOSE_GPU_DRAIN_BEGIN);
#ifndef DS4_NO_GPU
    bool gpu_drained = false;
    if (e->metal_ready) {
        /* metal_ready proves initialization, so this guard prevents the
         * synchronize API from initializing a backend during teardown. */
        gpu_drained = ds4_gpu_synchronize() != 0;
        ds4_engine_close_note_drain_result(e, gpu_drained);
        if (!gpu_drained) {
            fprintf(stderr,
                    "ds4: warning: GPU drain failed during engine close; "
                    "the terminal wait completed and backend cleanup "
                    "will continue\n");
        }
    } else {
        ds4_engine_close_note_drain_result(e, true);
    }
#endif
    ds4_engine_close_note(e, DS4_ENGINE_CLOSE_GPU_DRAINED);
    ds4_engine_close_note(e, DS4_ENGINE_CLOSE_GPU_CLEANUP_BEGIN);
#ifndef DS4_NO_GPU
    ds4_gpu_cleanup();
    e->metal_ready = false;
#endif
    ds4_engine_close_note(e, DS4_ENGINE_CLOSE_GPU_CLEANED);

    /* No worker or host-side tensor alias may survive into model unmapping. */
    ds4_threads_shutdown();
    ds4_engine_close_note(e, DS4_ENGINE_CLOSE_CPU_WORKERS_STOPPED);
    weights_free(&e->weights);
    memset(&e->dflash_weights, 0, sizeof(e->dflash_weights));
    vocab_free(&e->vocab);
    ds4_engine_close_note(e, DS4_ENGINE_CLOSE_HOST_ALIASES_CLEARED);

    lgn_dflash_release_f16_map(e->dflash_f16_map,
                               e->dflash_f16_map_size);
    e->dflash_f16_map = NULL;
    e->dflash_f16_map_size = 0;
    ds4_engine_close_note(e, DS4_ENGINE_CLOSE_DFLASH_SHADOW_UNMAPPED);
    model_close(&e->dflash_model);
    model_close(&e->model);
    ds4_engine_close_note(e, DS4_ENGINE_CLOSE_MODEL_MAPS_CLOSED);
    ds4_release_instance_lock();
    ds4_engine_close_note(e, DS4_ENGINE_CLOSE_LOCK_RELEASED);
    ds4_engine_close_note(e, DS4_ENGINE_CLOSE_ALLOCATIONS_RELEASING);
    free(e);
}


int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
    if (e->closing) {
        fprintf(stderr, "ds4: cannot create a session while the engine is closing\n");
        return 1;
    }
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (!laguna_metal_swa_gqa9_preflight(
            e, "session create", NULL, 0)) return 1;
    if (!laguna_metal_router_simd_topk_preflight(
            e, "session create", NULL, 0)) return 1;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA &&
        !laguna_metal_router_decode_fused_preflight(
            &e->model, &e->weights)) return 1;
#endif
#ifdef DS4_NO_GPU
    (void)ctx_size;
    return 1;
#else
    if (!e->metal_ready) return 1;

    /* Reject an explicit dense-Q8 request before allocating the session graph
     * or any KV/scratch state.  Later entrypoints repeat this cheap preflight
     * so an environment change after session creation cannot silently switch
     * to stock fallback. */
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA &&
        !laguna_dense_q8_gate_up_swiglu_preflight(
            &e->model, &e->weights, NULL)) {
        return 1;
    }

    ds4_session *s = xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA) {
#ifdef __APPLE__
        if (!laguna_metal_decode_residual_norm_preflight()) {
            free(s);
            return 1;
        }
        /* The explicit Q/K SIMD32 request is a session-wide graph contract.
         * Validate its exact geometry and PSO before laguna_graph_alloc can
         * allocate scratch/KV or write shared graph state. */
        if (!laguna_metal_qk_norm_rope_simd32_preflight()) {
            free(s);
            return 1;
        }
#endif
        if (!laguna_graph_alloc(&s->laguna_graph, (uint32_t)ctx_size)) {
            free(s);
            return 1;
        }
        s->laguna_graph_ready = true;
        if (e->dflash_ready) {
            if (!dflash_graph_alloc(&s->dflash_graph)) {
                laguna_graph_free(&s->laguna_graph);
                free(s);
                return 1;
            }
            s->dflash_graph_ready = true;
        }
        s->prefill_cap = (uint32_t)ctx_size;
        s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        s->sample_probs =
            xmalloc((size_t)DS4_N_VOCAB * sizeof(s->sample_probs[0]));
        return ds4_session_publish(out, s);
    }
    /* Generic graph sessions are unreachable for the supported Laguna
     * model contract. */
    free(s);
    return 1;
#endif
}

void ds4_session_free(ds4_session *s) {
    if (!s) return;
#ifndef DS4_NO_GPU
    if (s->dflash_cycles != 0u) {
        fprintf(stderr,
                "ds4: DFlash: %" PRIu64 " cycles, %" PRIu64
                "/%" PRIu64 " draft tokens accepted (%.1f%%), "
                "%" PRIu64 " low-confidence skipped, "
                "pipeline %.2f ms/cycle\n",
                s->dflash_cycles,
                s->dflash_accepted,
                s->dflash_proposed,
                s->dflash_proposed ?
                    100.0 * (double)s->dflash_accepted /
                        (double)s->dflash_proposed : 0.0,
                s->dflash_pruned,
                (s->dflash_draft_ms + s->dflash_verify_ms) /
                    (double)s->dflash_cycles);
    }
    dflash_graph_free(&s->dflash_graph);
    laguna_graph_free(&s->laguna_graph);
#endif
    token_vec_free(&s->checkpoint);
    free(s->logits);
    free(s->sample_probs);
    free(s->sample_cands.v);
    ds4_session_unregister_from_engine(s);
    s->engine = NULL;
    free(s);
}

#ifdef DS4_TEST_HOOKS
bool ds4_test_engine_session_lifecycle(void) {
    bool ok = true;
    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->dflash_model.fd = -1;

    ds4_session *registered = xcalloc(1, sizeof(*registered));
    registered->engine = e;
    ok = ok && ds4_session_register_with_engine(registered);
    ok = ok && registered->registered_with_engine && e->live_sessions == 1;

    /* A partially-built/failed session must not consume another session's
     * registration or underflow the engine count when its cleanup runs. */
    ds4_session *unregistered = xcalloc(1, sizeof(*unregistered));
    unregistered->engine = e;
    ds4_session_free(unregistered);
    ok = ok && e->live_sessions == 1;

    /* Supported lifetime calls are serialized: close must refuse while the
     * registered session exists, then remain retryable after that session is
     * freed.  No raw engine pointer is concurrently dereferenced here. */
    ds4_engine_close(e);
    ok = ok && !e->closing && e->live_sessions == 1;

    ds4_session_free(registered);
    ok = ok && e->live_sessions == 0;

    ds4_session *out = (ds4_session *)(uintptr_t)0x1;
    ok = ok && ds4_session_create(&out, e, 0) != 0 &&
         out == (ds4_session *)(uintptr_t)0x1 && e->live_sessions == 0;

    e->closing = true;
    ok = ok && ds4_session_create(&out, e, 1) != 0 &&
         out == (ds4_session *)(uintptr_t)0x1 && e->live_sessions == 0;
    ds4_session *late = xcalloc(1, sizeof(*late));
    late->engine = e;
    ok = ok && ds4_session_publish(&out, late) != 0 &&
         out == (ds4_session *)(uintptr_t)0x1 && e->live_sessions == 0;
    e->closing = false;

    e->live_sessions = SIZE_MAX;
    ds4_session *overflow = xcalloc(1, sizeof(*overflow));
    overflow->engine = e;
    ok = ok && !ds4_session_register_with_engine(overflow) &&
         !overflow->registered_with_engine && e->live_sessions == SIZE_MAX;
    ds4_session_free(overflow);
    ok = ok && e->live_sessions == SIZE_MAX;
    e->live_sessions = 0;

    ds4_test_engine_close_trace retry_trace = {
        .engine = e,
        .maps_live_through_gpu = true,
    };
    ds4_test_engine_close_trace *previous = g_ds4_test_engine_close_trace;
    g_ds4_test_engine_close_trace = &retry_trace;
    ds4_engine_close(e);
    g_ds4_test_engine_close_trace = previous;
    ok = ok && retry_trace.phase_count != 0 &&
         retry_trace.phases[0] == DS4_ENGINE_CLOSE_BEGIN &&
         retry_trace.phases[retry_trace.phase_count - 1] ==
             DS4_ENGINE_CLOSE_ALLOCATIONS_RELEASING;
    return ok;
}

bool ds4_test_engine_close_order(void) {
    static const ds4_engine_close_phase expected[] = {
        DS4_ENGINE_CLOSE_BEGIN,
        DS4_ENGINE_CLOSE_GPU_DRAIN_BEGIN,
        DS4_ENGINE_CLOSE_GPU_DRAINED,
        DS4_ENGINE_CLOSE_GPU_CLEANUP_BEGIN,
        DS4_ENGINE_CLOSE_GPU_CLEANED,
        DS4_ENGINE_CLOSE_CPU_WORKERS_STOPPED,
        DS4_ENGINE_CLOSE_HOST_ALIASES_CLEARED,
        DS4_ENGINE_CLOSE_DFLASH_SHADOW_UNMAPPED,
        DS4_ENGINE_CLOSE_MODEL_MAPS_CLOSED,
        DS4_ENGINE_CLOSE_LOCK_RELEASED,
        DS4_ENGINE_CLOSE_ALLOCATIONS_RELEASING,
    };

    const long page_long = sysconf(_SC_PAGESIZE);
    if (page_long <= 0) return false;
    const size_t page = (size_t)page_long;
#if defined(MAP_ANONYMOUS)
    const int anonymous = MAP_ANONYMOUS;
#else
    const int anonymous = MAP_ANON;
#endif
    void *maps[3] = {0};
    for (size_t i = 0; i < sizeof(maps) / sizeof(maps[0]); i++) {
        maps[i] = mmap(NULL, page, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | anonymous, -1, 0);
        if (maps[i] == MAP_FAILED) {
            maps[i] = NULL;
            for (size_t j = 0; j < i; j++) munmap(maps[j], page);
            return false;
        }
    }

    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->dflash_model.fd = -1;
    e->model.map = maps[0];
    e->model.size = page;
    e->dflash_model.map = maps[1];
    e->dflash_model.size = page;
    e->dflash_f16_map = maps[2];
    e->dflash_f16_map_size = page;
    memset(&e->weights, 0xa5, sizeof(e->weights));
    memset(&e->dflash_weights, 0xa5, sizeof(e->dflash_weights));

    ds4_test_engine_close_trace trace = {
        .engine = e,
        .maps_live_through_gpu = true,
    };
    ds4_test_engine_close_trace *previous = g_ds4_test_engine_close_trace;
    g_ds4_test_engine_close_trace = &trace;
    ds4_engine_close(e);
    g_ds4_test_engine_close_trace = previous;

    const bool phase_count_ok =
        trace.phase_count == sizeof(expected) / sizeof(expected[0]);
    const bool phases_ok =
        phase_count_ok && memcmp(trace.phases, expected, sizeof(expected)) == 0;
    const bool ok = phases_ok && trace.maps_live_through_gpu &&
                    trace.aliases_cleared_before_unmap &&
                    trace.dflash_shadow_unmapped_first &&
                    trace.model_maps_unmapped;
    if (!ok) {
        fprintf(stderr,
                "ds4: engine close-order test failed: phases=%zu/%zu order=%d "
                "gpu_maps=%d aliases=%d shadow=%d models=%d\n",
                trace.phase_count,
                sizeof(expected) / sizeof(expected[0]),
                phases_ok,
                trace.maps_live_through_gpu,
                trace.aliases_cleared_before_unmap,
                trace.dflash_shadow_unmapped_first,
                trace.model_maps_unmapped);
    }
    return ok;
}

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
/* Preserve terminal GPU-drain failure coverage without resurrecting the
 * removed shared-prefill workspace owner.  This queues real Metal work,
 * injects the one-shot synchronize report after that work is drained, and
 * proves engine close still cleans tensors and mappings in ownership order. */
bool ds4_test_engine_close_drain_failure(void) {
    const long page_long = sysconf(_SC_PAGESIZE);
    if (page_long <= 0) return false;
    const size_t page = (size_t)page_long;
#if defined(MAP_ANONYMOUS)
    const int anonymous = MAP_ANONYMOUS;
#else
    const int anonymous = MAP_ANON;
#endif

    ds4_gpu_cleanup();
    if (!ds4_gpu_test_cleanup_state_is_clean() || !ds4_gpu_init()) {
        ds4_gpu_cleanup();
        return false;
    }

    void *maps[3] = {0};
    bool setup_ok = true;
    for (size_t i = 0; i < sizeof(maps) / sizeof(maps[0]); i++) {
        maps[i] = mmap(NULL, page, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | anonymous, -1, 0);
        if (maps[i] == MAP_FAILED) {
            maps[i] = NULL;
            setup_ok = false;
            break;
        }
        memset(maps[i], 0x5a, page);
    }

    ds4_gpu_tensor *src = NULL;
    ds4_gpu_tensor *dst = NULL;
    bool model_map_registered = false;
    uint64_t live_handles_before_close = 0;
    uint64_t live_bytes_before_close = 0;
    if (setup_ok) {
        model_map_registered =
            ds4_gpu_set_model_map_range(maps[0], page, 0, page, page) != 0;
        src = ds4_gpu_tensor_alloc(64);
        dst = ds4_gpu_tensor_alloc(64);
        setup_ok = model_map_registered && src != NULL && dst != NULL &&
                   ds4_gpu_begin_commands() != 0 &&
                   ds4_gpu_tensor_copy(dst, 0, src, 0, 64) != 0 &&
                   ds4_gpu_commands_active() != 0 &&
                   ds4_gpu_test_tensor_tracking_state(
                       &live_handles_before_close, &live_bytes_before_close) != 0 &&
                   live_handles_before_close >= 2u &&
                   live_bytes_before_close != 0;
    }

    ds4_engine *e = NULL;
    ds4_test_engine_close_trace trace = {0};
    if (setup_ok) {
        e = xcalloc(1, sizeof(*e));
        e->model.fd = -1;
        e->dflash_model.fd = -1;
        e->metal_ready = true;
        e->model.map = maps[0];
        e->model.size = page;
        e->dflash_model.map = maps[1];
        e->dflash_model.size = page;
        e->dflash_f16_map = maps[2];
        e->dflash_f16_map_size = page;
        trace.engine = e;
        trace.maps_live_through_gpu = true;
        ds4_test_engine_close_trace *previous = g_ds4_test_engine_close_trace;
        g_ds4_test_engine_close_trace = &trace;
        ds4_gpu_test_inject_synchronize_failure();
        ds4_engine_close(e);
        g_ds4_test_engine_close_trace = previous;
        e = NULL;
    }

    if (e) {
        if (ds4_gpu_commands_active()) (void)ds4_gpu_discard_commands();
        ds4_engine_close(e);
    } else if (!setup_ok) {
        if (ds4_gpu_commands_active()) (void)ds4_gpu_discard_commands();
        ds4_gpu_tensor_free(src);
        ds4_gpu_tensor_free(dst);
        for (size_t i = 0; i < sizeof(maps) / sizeof(maps[0]); i++) {
            if (maps[i]) munmap(maps[i], page);
        }
        ds4_gpu_cleanup();
    }

    return setup_ok && trace.gpu_drain_reported_failure &&
           trace.maps_live_through_gpu && trace.model_maps_unmapped &&
           ds4_gpu_test_cleanup_state_is_clean();
}
#endif

#endif /* DS4_TEST_HOOKS */

void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}

void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud) {
    if (!s) return;
    s->cancel = fn;
    s->cancel_ud = ud;
}

void ds4_session_set_speculative_enabled(ds4_session *s, bool enabled) {
    if (!s || s->speculative_enabled == enabled) return;
    s->speculative_enabled = enabled;
#ifndef DS4_NO_GPU
    /* The target checkpoint remains valid, but a disabled DFlash cache may
     * miss arbitrary intervening tokens. Rebuild its last-window state on the
     * next enabled sync instead of guessing whether it stayed current. */
    ds4_session_dflash_invalidate(s);
    s->dflash_baseline_ms = 0.0;
    s->dflash_baseline_tokens = 0;
    s->dflash_cycles_since_baseline = 0;
    s->dflash_window_ms = 0.0;
    s->dflash_window_tokens = 0;
    s->dflash_window_cycles = 0;
    s->dflash_slow_windows = 0;
    s->dflash_active_draft = 0;
    s->dflash_best_draft = 0;
    s->dflash_best_ms_per_token = 0.0;
    s->dflash_stage_full_accepts = 0;
    s->dflash_suspended = false;
    s->dflash_guard_decided = false;
#endif
}

static bool ds4_session_cancelled(ds4_session *s) {
    return s && s->cancel && s->cancel(s->cancel_ud);
}

void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total) {
    if (!s || !s->progress || !event) return;
    s->progress(s->progress_ud, event, current, total);
}


#ifndef DS4_NO_GPU
static bool ds4_session_dflash_enabled(const ds4_session *s) {
    return s && s->speculative_enabled && s->engine &&
           s->engine->dflash_ready &&
           s->dflash_graph_ready &&
           !s->dflash_suspended;
}

static ds4_laguna_feature_capture ds4_session_dflash_capture(
        ds4_session *s,
        uint32_t     src_row0,
        uint32_t     n_rows) {
    ds4_laguna_feature_capture capture;
    memset(&capture, 0, sizeof(capture));
    if (!ds4_session_dflash_enabled(s) || n_rows == 0u) return capture;
    capture.features = s->dflash_graph.features;
    capture.target_layers = s->engine->dflash_weights.target_layers;
    capture.n_aux = DS4_DFLASH_N_AUX;
    capture.src_row0 = src_row0;
    capture.dst_row0 = 0;
    capture.n_rows = n_rows;
    return capture;
}

/* A GPU-draft verifier borrows the caller's active batch.  The caller must
 * observe an active owner here: a missing batch means the helper violated its
 * ownership contract or a terminal boundary already ran. */
static bool ds4_session_dflash_discard_owned_commands(void) {
    return ds4_gpu_commands_active() && ds4_gpu_discard_commands() != 0;
}

static void ds4_session_dflash_quarantine(ds4_session *s) {
    if (!s) return;
    s->checkpoint_valid = false;
    s->dflash_synced = false;
#ifdef __APPLE__
    laguna_metal_target_evidence_discard(&s->laguna_graph);
#endif
}

/* A submitted target snapshot is not a usable rollback source until the
 * command buffer that wrote its backup has completed successfully.  Keep the
 * completion proof at the restore boundary so terminal failures cannot read a
 * stale or unwritten backup merely because an earlier flush succeeded. */
static bool ds4_session_dflash_restore_snapshot(
        ds4_session *s,
        bool         snapshot_completed,
        uint32_t     pos0,
        uint32_t     accepted_rows,
        uint32_t     n_rows) {
    if (!snapshot_completed) return true;
    if (!s) return false;
    if (!laguna_graph_spec_restore(&s->laguna_graph,
                                   pos0,
                                   accepted_rows,
                                   n_rows)) {
        return false;
    }
#ifdef __APPLE__
    /* laguna_graph_spec_restore submits the Apple rollback CB and leaves it
     * pending.  Do not let the caller append checkpoint tokens or mark
     * DFlash synchronized until this exact CB has completed successfully. */
    if (ds4_gpu_wait_submitted_commands() == 0) return false;
#endif
    return true;
}

static bool ds4_session_dflash_finish_capture(
        ds4_session *s,
        uint32_t     pos0,
        uint32_t     n_rows) {
    if (!ds4_session_dflash_enabled(s) || n_rows == 0u) return true;
    if (!dflash_graph_encode_inject(&s->dflash_graph,
                                    s->engine,
                                    pos0,
                                    n_rows)) {
        s->dflash_synced = false;
        /* end/wait may have executed part of the injection before reporting
         * failure; the host checkpoint must not continue to certify it. */
        s->checkpoint_valid = false;
        return false;
    }
#ifdef __APPLE__
    /* dflash_graph_encode_inject closes and waits the active target command
     * after appending its K-only support feature.  Only now may the deferred
     * target snapshot be used as path evidence. */
    if (!laguna_metal_qk_norm_rope_simd32_target_evidence_complete(
            &s->laguna_graph) ||
        !laguna_metal_rope_atlas_target_evidence_complete(
            &s->laguna_graph)) {
        s->dflash_synced = false;
        s->checkpoint_valid = false;
        return false;
    }
#endif
    return true;
}

static bool ds4_session_dflash_flush_deferred(ds4_session *s) {
    if (!s || s->dflash_deferred_rows == 0u) return true;
    if (!ds4_session_dflash_finish_capture(s,
                                           s->dflash_deferred_pos0,
                                           s->dflash_deferred_rows)) {
        return false;
    }
    s->dflash_deferred_rows = 0u;
    s->dflash_deferred_pos0 = 0u;
    return true;
}
#endif



static int ds4_session_eval_internal(ds4_session *s, int token,
                                     char *err, size_t errlen);


/* Bring the live backend state to exactly the supplied token prefix.
 *
 * ds4-server and the REPL are stateless at the text/API layer but stateful here:
 * they resend or rebuild the full transcript, and this function decides whether
 * the live checkpoint is a prefix.  A matching prefix is extended in one of two
 * ways:
 *
 *   - long suffix: batched layer-major prefill, aligned to absolute chunk
 *     boundaries so the Laguna target graph's KV frontiers finalize in the
 *     same order as a cold prompt;
 *   - short suffix: ordinary one-token decode, which is faster below the
 *     measured crossover and preserves exact autoregressive semantics.
 *
 * A non-matching prompt discards the checkpoint and prefills from token zero.
 */
static int ds4_session_sync_internal(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);

int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (s && !laguna_metal_swa_gqa9_preflight(
            s->engine, "session sync", err, errlen)) return 1;
    if (s && !laguna_metal_router_simd_topk_preflight(
            s->engine, "session sync", err, errlen)) return 1;
#endif
    ds4_session_note_logits_dirty(s);
    return ds4_session_sync_internal(s, prompt, err, errlen);
}

static int ds4_session_sync_internal(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt) {
        snprintf(err, errlen, "missing session or prompt");
        return 1;
    }
    if (prompt->len <= 0) {
        snprintf(err, errlen, "empty prompt");
        return 1;
    }
    if (prompt->len >= s->ctx_size) {
        snprintf(err, errlen,
                 "prompt length %d exceeds context %d (one token of generation room is required)",
                 prompt->len, s->ctx_size);
        return 1;
    }
    if (ds4_session_cancelled(s)) {
        snprintf(err, errlen, "interrupted");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }
#ifdef DS4_TEST_HOOKS
    if (ds4_session_is_laguna(s) && g_ds4_test_route_sync_fn) {
        g_ds4_test_route_sync_calls++;
        return g_ds4_test_route_sync_fn(s, prompt, err, errlen);
    }
#endif
#ifdef DS4_NO_GPU
    (void)s;
    (void)prompt;
    snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    const char *runtime_name = DS4_RUNTIME_NAME;
    (void)runtime_name; (void)e;
    if (ds4_session_is_laguna(s)) {
        if (!laguna_dense_q8_gate_up_swiglu_preflight(
                    &e->model, &e->weights, NULL)) {
            snprintf(err, errlen,
                     "%s Laguna dense Q8 gate/up+SwiGLU preflight failed",
                     runtime_name);
            return 1;
        }
#ifdef __APPLE__
        if (!laguna_metal_decode_residual_norm_preflight()) {
            snprintf(err, errlen,
                     "%s Laguna decode residual fusion preflight failed",
                     runtime_name);
            return 1;
        }
        if (!laguna_metal_qk_norm_rope_simd32_preflight()) {
            snprintf(err, errlen,
                     "%s Laguna Q/K norm/RoPE SIMD32 preflight failed",
                     runtime_name);
            return 1;
        }
#endif
        if (!s->laguna_graph_ready) {
            snprintf(err, errlen, "%s Laguna graph is not initialized",
                     runtime_name);
            return 1;
        }
        s->dflash_baseline_ms = 0.0;
        s->dflash_baseline_tokens = 0;
        s->dflash_cycles_since_baseline = 0;
        s->dflash_window_ms = 0.0;
        s->dflash_window_tokens = 0;
        s->dflash_window_cycles = 0;
        s->dflash_slow_windows = 0;
        s->dflash_active_draft = 0;
        s->dflash_best_draft = 0;
        s->dflash_best_ms_per_token = 0.0;
        s->dflash_stage_full_accepts = 0;
        s->dflash_suspended = false;
        s->dflash_guard_decided = false;
        s->dflash_defer_inject = false;
        if (!ds4_session_dflash_flush_deferred(s)) {
            snprintf(err, errlen,
                     "%s DFlash deferred injection failed before sync",
                     runtime_name);
            s->checkpoint_valid = false;
            s->dflash_synced = false;
            return 1;
        }
        int start = 0;
        if (s->checkpoint_valid &&
            prompt->len >= s->checkpoint.len &&
            ds4_tokens_starts_with(prompt, &s->checkpoint)) {
            start = s->checkpoint.len;
        } else {
            s->checkpoint.len = 0;
            s->checkpoint_valid = false;
            s->dflash_synced = false;
        }

        const bool dflash_enabled = ds4_session_dflash_enabled(s);
        /*
         * Replaying only the last DFlash window is not correct after the
         * support cache has fallen behind: Laguna's sliding-window KV ring
         * still contains newer rows, so the replay would attend to future
         * tokens. Rebuild the target and support caches together from token
         * zero. This is only paid when speculative decoding is re-enabled
         * after being disabled or auto-paused.
         */
        if (dflash_enabled && !s->dflash_synced && start > 0) {
            start = 0;
            s->checkpoint.len = 0;
            s->checkpoint_valid = false;
        }
        int capture_floor = prompt->len > (int)DS4_DFLASH_CACHE_CAP ?
            prompt->len - (int)DS4_DFLASH_CACHE_CAP : 0;
        if (s->dflash_synced && capture_floor < start) capture_floor = start;

        for (int i = start; i < prompt->len;) {
            if (ds4_session_cancelled(s)) {
                snprintf(err, errlen, "interrupted");
                s->checkpoint_valid = s->checkpoint.len != 0;
                return DS4_SESSION_SYNC_INTERRUPTED;
            }
            uint32_t n = (uint32_t)(prompt->len - i);
            if (n > s->laguna_graph.prefill_cap) {
                n = s->laguna_graph.prefill_cap;
            }
            const bool last = i + (int)n == prompt->len;
            int capture_begin = i;
            uint32_t capture_rows = 0;
            ds4_laguna_feature_capture capture;
            memset(&capture, 0, sizeof(capture));
            if (dflash_enabled) {
                if (capture_begin < capture_floor) capture_begin = capture_floor;
                const int capture_end = i + (int)n;
                if (capture_begin < capture_end) {
                    capture_rows = (uint32_t)(capture_end - capture_begin);
                    capture = ds4_session_dflash_capture(
                        s,
                        (uint32_t)(capture_begin - i),
                        capture_rows);
                }
            }
            const ds4_laguna_feature_capture *capture_ptr =
                capture_rows ? &capture : NULL;
            bool ok = n == 1u ?
                laguna_graph_forward_token(&s->laguna_graph,
                                           &e->model,
                                           &e->weights,
                                           prompt->v[i],
                                           (uint32_t)i,
                                           capture_ptr,
                                           last ? s->logits : NULL) :
                laguna_graph_forward_batch(&s->laguna_graph,
                                           &e->model,
                                           &e->weights,
                                           prompt->v + i,
                                           NULL,
                                           n,
                                           (uint32_t)i,
                                           last ? s->logits : NULL,
                                           NULL,
                                           capture_ptr,
                                           s->display_progress,
                                           s->display_progress_ud,
                                           prompt->len);
            if (!ok) {
                snprintf(err, errlen,
                         "%s Laguna prefill failed at token %d",
                         runtime_name,
                         i);
                s->checkpoint_valid = false;
                s->dflash_synced = false;
                return 1;
            }
            if (capture_rows &&
                !ds4_session_dflash_finish_capture(
                    s,
                    (uint32_t)capture_begin,
                    capture_rows)) {
                snprintf(err, errlen,
                         "%s DFlash cache injection failed at token %d",
                         runtime_name,
                         capture_begin);
                s->checkpoint_valid = false;
                return 1;
            }
            for (uint32_t j = 0; j < n; j++) {
                token_vec_push(&s->checkpoint, prompt->v[i + (int)j]);
            }
            i += (int)n;
            if (s->progress) {
                s->progress(s->progress_ud,
                            "prefill_chunk",
                            i,
                            prompt->len);
            }
        }
        if (dflash_enabled) s->dflash_synced = true;
        s->checkpoint_valid = true;
        return 0;
    }
    snprintf(err, errlen, "%s session route is unsupported",
             DS4_RUNTIME_NAME);
    return 1;
#endif
}

/* Return true when canonicalization would replace already-sampled tokens.
 *
 * A DS4 session checkpoint is more than a token vector: the Laguna target
 * graph also contains persistent KV rings and optional DFlash feature history.
 * Replacing any part of the live tail requires restoring that whole frontier
 * first.  Extending exactly at the live end is safe; rewriting behind it is
 * not an in-place operation. */
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}

/* Replace the live suffix after a shared prefix.
 *
 * This is used after parsing a generated tool call.  The model may have emitted
 * DSML in an order that is semantically valid but not byte-for-byte equal to the
 * canonical prompt we will see on the next request.  Rewriting only the token
 * checkpoint is not enough: the backend still contains target KV and optional
 * DFlash rows for the old suffix.  Until we have a real frontier snapshot at
 * the rewrite point, any replacement behind the live end reports that a rebuild
 * is needed without mutating the session.  The server may still find an older
 * disk KV checkpoint before falling back to a full replay. */
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen) {
    if (!s || !prompt) {
        snprintf(err, errlen, "missing session or prompt");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (prompt->len <= 0) {
        snprintf(err, errlen, "empty prompt");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (prompt->len >= s->ctx_size) {
        snprintf(err, errlen,
                 "prompt length %d exceeds context %d (one token of generation room is required)",
                 prompt->len, s->ctx_size);
        return DS4_SESSION_REWRITE_ERROR;
    }
    ds4_session_note_logits_dirty(s);
    if (!s->checkpoint_valid) {
        snprintf(err, errlen, "session has no valid checkpoint");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (common < 0 || common > s->checkpoint.len || common > prompt->len) {
        snprintf(err, errlen, "invalid rewrite prefix");
        return DS4_SESSION_REWRITE_ERROR;
    }
    for (int i = 0; i < common; i++) {
        if (s->checkpoint.v[i] != prompt->v[i]) {
            snprintf(err, errlen, "rewrite prefix does not match live checkpoint");
            return DS4_SESSION_REWRITE_ERROR;
        }
    }

    if (common == s->checkpoint.len) {
        return ds4_session_sync(s, prompt, err, errlen) == 0 ?
            DS4_SESSION_REWRITE_OK : DS4_SESSION_REWRITE_ERROR;
    }

    if (ds4_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        snprintf(err, errlen, "rewrite needs rebuild: common=%d live=%d canonical=%d",
                 common, s->checkpoint.len, prompt->len);
        return DS4_SESSION_REWRITE_REBUILD_NEEDED;
    }

    snprintf(err, errlen, "unexpected canonical rewrite state");
    return DS4_SESSION_REWRITE_ERROR;
}

int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt) {
    if (!s->checkpoint_valid) return 0;
    int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    int i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}

int ds4_session_argmax(ds4_session *s) {
    return sample_argmax(s->logits, DS4_N_VOCAB);
}

int ds4_session_argmax_excluding(ds4_session *s, int excluded_id) {
    if (!s || !s->logits) return -1;
    if (getenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == NULL) {
        return argmax_f32_excluding_unrolled8(
                s->logits, DS4_N_VOCAB, excluded_id);
    }
    int best = -1;
    float best_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if ((int)i == excluded_id) continue;
        const float v = s->logits[i];
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    return best;
}

int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!logits || n_vocab <= 0) return 0;
    float *scratch = xmalloc((size_t)n_vocab * sizeof(scratch[0]));
    sample_arena cands = {0};
    const int token = sample_top_p_min_p(logits, (uint32_t)n_vocab,
                                         temperature, top_k, top_p, min_p,
                                         rng, scratch, &cands);
    free(cands.v);
    free(scratch);
    return token;
}

int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!s->engine->dflash_ready || !s->speculative_enabled) {
        return sample_top_p_min_p(s->logits, DS4_N_VOCAB, temperature, top_k,
                                  top_p, min_p, rng, s->sample_probs,
                                  &s->sample_cands);
    }
    const double t0 = now_sec();
    const int token =
        sample_top_p_min_p(s->logits, DS4_N_VOCAB, temperature, top_k,
                           top_p, min_p, rng, s->sample_probs,
                           &s->sample_cands);
    s->last_sample_ms = (now_sec() - t0) * 1000.0;
    return token;
}

/* The full-vocab double-precision softmax normalizer, computed once per
 * logits state and shared by the logprob observers: a request typically asks
 * for the sampled token's logprob and the top-k alternatives together, and
 * the scalar double-exp pass over the vocab dwarfs everything else in those
 * calls.  The computation keeps the original precision and accumulation
 * order, so a cached result is bit-identical to recomputation.  The top-k
 * observer supplies the max from its ranking pass so this helper only scans
 * the vocabulary for the sum on a cold top-k request. */
static bool ds4_session_logsumexp(ds4_session *s, double *logsum_out,
                                  bool have_max, float known_max_logit) {
    if (s->logsumexp_valid && s->logsumexp_gen == s->logits_gen) {
#ifdef DS4_TEST_HOOKS
        g_ds4_test_logprob_stats.cache_hits++;
#endif
        *logsum_out = s->logsumexp;
        return s->logsumexp_ok;
    }
    float max_logit = known_max_logit;
    if (!have_max) {
#ifdef DS4_TEST_HOOKS
        g_ds4_test_logprob_stats.max_scans++;
#endif
        max_logit = DS4_NEG_INF;
        for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
            const float v = s->logits[i];
            if (isfinite(v) && v > max_logit) max_logit = v;
        }
    }
    double logsum = 0.0;
    bool ok = false;
    if (isfinite(max_logit)) {
#ifdef DS4_TEST_HOOKS
        g_ds4_test_logprob_stats.sum_scans++;
#endif
        double sum = 0.0;
        for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
            const float v = s->logits[i];
            if (isfinite(v)) sum += exp((double)v - (double)max_logit);
        }
        logsum = (double)max_logit + log(sum);
        ok = true;
    }
    s->logsumexp = logsum;
    s->logsumexp_ok = ok;
    s->logsumexp_gen = s->logits_gen;
    s->logsumexp_valid = true;
    *logsum_out = logsum;
    return ok;
}

int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k) {
    if (!s || !out || k <= 0) return 0;
    if (k > (int)DS4_N_VOCAB) k = (int)DS4_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = DS4_NEG_INF;
        out[i].logprob = DS4_NEG_INF;
    }

    float max_logit = DS4_NEG_INF;
#ifdef DS4_TEST_HOOKS
    g_ds4_test_logprob_stats.max_scans++;
#endif
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;

    double logsum = 0.0;
    if (!ds4_session_logsumexp(s, &logsum, true, max_logit)) return 0;
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ? (float)((double)out[i].logit - logsum) : DS4_NEG_INF;
    }
    return k;
}

int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out) {
    if (!s || !out || token < 0 || token >= (int)DS4_N_VOCAB) return 0;

    double logsum = 0.0;
    if (!ds4_session_logsumexp(s, &logsum, false, DS4_NEG_INF)) return 0;
    out->id = token;
    out->logit = s->logits[token];
    out->logprob = isfinite(out->logit) ? (float)((double)out->logit - logsum) : DS4_NEG_INF;
    return 1;
}

int ds4_session_copy_logits(ds4_session *s, float *out, int cap) {
    if (!s || !out || cap < (int)DS4_N_VOCAB) return 0;
    memcpy(out, s->logits, (size_t)DS4_N_VOCAB * sizeof(out[0]));
    return (int)DS4_N_VOCAB;
}

int ds4_session_set_logits(ds4_session *s, const float *logits, int n) {
    if (!s || !logits || n != (int)DS4_N_VOCAB) return 1;
    ds4_session_note_logits_dirty(s);
    memcpy(s->logits, logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    return 0;
}

#ifdef DS4_TEST_HOOKS
/* Exercise the cache without loading a model.  The probe deliberately uses
 * the public logits setter so a mutation must advance the same generation
 * counter as production callers. */
int ds4_test_logprob_cache_probe(void) {
    ds4_session s = {0};
    float *initial = xmalloc((size_t)DS4_N_VOCAB * sizeof(initial[0]));
    float *mutated = xmalloc((size_t)DS4_N_VOCAB * sizeof(mutated[0]));
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        initial[i] = -3.0f - (float)(i % 257u) * 0.001f;
    }
    initial[17] = 2.5f;
    const uint32_t neg_inf_bits = 0xff800000u;
    const uint32_t nan_bits = 0x7fc00000u;
    memcpy(&initial[23], &neg_inf_bits, sizeof(neg_inf_bits));
    memcpy(&initial[29], &nan_bits, sizeof(nan_bits));
    memcpy(mutated, initial,
           (size_t)DS4_N_VOCAB * sizeof(mutated[0]));
    s.logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s.logits[0]));
    memcpy(s.logits, initial,
           (size_t)DS4_N_VOCAB * sizeof(s.logits[0]));
    /* A synthetic session has no eval entry point to mark its first logits. */
    ds4_session_note_logits_dirty(&s);

    ds4_test_logprob_stats_reset();
    ds4_token_score top[4], top_again[4], token_before, token_after;
    bool ok = ds4_session_top_logprobs(&s, top, 4) == 4;
    ds4_test_logprob_stats stats = {0};
    ds4_test_logprob_stats_get(&stats);
    ok = ok && stats.max_scans == 1 && stats.sum_scans == 1 &&
         stats.cache_hits == 0 && top[0].id == 17;

    /* Repeating the observer still scans to select the top entries, but must
     * reuse the already accumulated normalizer. */
    ok = ok && ds4_session_top_logprobs(&s, top_again, 4) == 4;
    ds4_test_logprob_stats_get(&stats);
    ok = ok && stats.max_scans == 2 && stats.sum_scans == 1 &&
         stats.cache_hits == 1;
    ok = ok && ds4_session_token_logprob(&s, 17, &token_before) == 1;
    ds4_test_logprob_stats_get(&stats);
    ok = ok && stats.max_scans == 2 && stats.sum_scans == 1 &&
         stats.cache_hits == 2;

    /* Check the cached arithmetic against an independent copy that uses the
     * same max-then-sum order as the original implementation. */
    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (isfinite(initial[i]) && initial[i] > max_logit) {
            max_logit = initial[i];
        }
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (isfinite(initial[i])) {
            sum += exp((double)initial[i] - (double)max_logit);
        }
    }
    const double reference_logsum = (double)max_logit + log(sum);
    const float reference_logprob =
        (float)((double)initial[17] - reference_logsum);
    ok = ok && top[0].logprob == reference_logprob &&
         top_again[0].logprob == reference_logprob &&
         token_before.logprob == reference_logprob;

    mutated[17] = 3.75f;
    ok = ok && ds4_session_set_logits(&s, mutated, (int)DS4_N_VOCAB) == 0;
    ok = ok && ds4_session_token_logprob(&s, 17, &token_after) == 1;
    ds4_test_logprob_stats_get(&stats);
    ok = ok && stats.max_scans == 3 && stats.sum_scans == 2 &&
         stats.cache_hits == 2 && token_after.logprob != token_before.logprob &&
         token_after.logit == 3.75f;

    free(s.logits);
    free(initial);
    free(mutated);
    return ok ? 0 : 1;
}
#endif

/* Pay the one-time first-submission GPU cost (pipeline ramp plus model-heap
 * residency for the batched prefill kernels) outside any measured window. */
void ds4_session_gpu_warmup(ds4_session *s) {
    (void)s;
}

static int ds4_session_eval_internal(ds4_session *s, int token,
                                     char *err, size_t errlen) {
    if (!s) return 1;
#ifdef DS4_NO_GPU
    (void)s;
    (void)token;
    snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    if (ds4_session_is_laguna(s)) {
        if (!laguna_dense_q8_gate_up_swiglu_preflight(
                    &e->model, &e->weights, NULL)) {
            if (errlen) snprintf(err, errlen,
                                 "%s Laguna dense Q8 gate/up+SwiGLU "
                                 "preflight failed",
                                 DS4_RUNTIME_NAME);
            return 1;
        }
#ifdef __APPLE__
        if (!laguna_metal_decode_residual_norm_preflight()) {
            if (errlen) snprintf(err, errlen,
                                 "%s Laguna decode residual fusion preflight failed",
                                 DS4_RUNTIME_NAME);
            return 1;
        }
        if (!laguna_metal_qk_norm_rope_simd32_preflight()) {
            if (errlen) snprintf(err, errlen,
                                 "%s Laguna Q/K norm/RoPE SIMD32 preflight failed",
                                 DS4_RUNTIME_NAME);
            return 1;
        }
#endif
        if (!s->laguna_graph_ready) {
            if (errlen) snprintf(err, errlen,
                                 "%s Laguna graph is not initialized",
                                 DS4_RUNTIME_NAME);
            return 1;
        }
        if ((uint32_t)s->checkpoint.len >= s->laguna_graph.ctx_size) {
            if (errlen) snprintf(err, errlen,
                                 "Laguna Metal context reached (%u)",
                                 s->laguna_graph.ctx_size);
            return 1;
        }
        const bool dflash_enabled = ds4_session_dflash_enabled(s);
        const bool dflash_was_synced = s->dflash_synced;
        if (dflash_enabled && !s->dflash_defer_inject &&
            !ds4_session_dflash_flush_deferred(s)) {
            if (errlen) snprintf(err, errlen,
                                 "%s DFlash deferred injection failed",
                                 DS4_RUNTIME_NAME);
            s->checkpoint_valid = false;
            return 1;
        }
        ds4_laguna_feature_capture capture =
            ds4_session_dflash_capture(s, 0, 1);
        if (dflash_enabled && s->dflash_defer_inject) {
            const uint32_t pos = (uint32_t)s->checkpoint.len;
            if (s->dflash_deferred_rows == 0u) {
                s->dflash_deferred_pos0 = pos;
            } else if (pos != s->dflash_deferred_pos0 +
                              s->dflash_deferred_rows ||
                       s->dflash_deferred_rows >=
                           DS4_DFLASH_BLOCK_SIZE) {
                if (errlen) snprintf(err, errlen,
                                     "DFlash deferred capture is not contiguous");
                s->checkpoint_valid = false;
                s->dflash_synced = false;
                return 1;
            }
            capture.dst_row0 = s->dflash_deferred_rows;
        }
        if (!laguna_graph_forward_token(&s->laguna_graph,
                                        &e->model,
                                        &e->weights,
                                        token,
                                        (uint32_t)s->checkpoint.len,
                                        dflash_enabled ? &capture : NULL,
                                        s->logits)) {
            if (errlen) snprintf(err, errlen, "%s Laguna decode failed",
                                 DS4_RUNTIME_NAME);
            s->checkpoint_valid = false;
            s->dflash_synced = false;
            return 1;
        }
        if (dflash_enabled && s->dflash_defer_inject) {
            s->dflash_deferred_rows++;
        } else if (dflash_enabled &&
            !ds4_session_dflash_finish_capture(
                s, (uint32_t)s->checkpoint.len, 1)) {
            if (errlen) snprintf(err, errlen,
                                 "%s DFlash cache injection failed",
                                 DS4_RUNTIME_NAME);
            s->checkpoint_valid = false;
            return 1;
        }
        if (dflash_enabled) {
            s->dflash_synced =
                dflash_was_synced || s->checkpoint.len == 0;
        }
        token_vec_push(&s->checkpoint, token);
        s->checkpoint_valid = true;
        return 0;
    }
    #ifdef DS4_TEST_HOOKS
    g_ds4_test_route_generic_eval_calls++;
    if (g_ds4_test_route_generic_access_forbidden) {
        snprintf(err, errlen, "unsupported generic evaluator was reached");
    } else {
        snprintf(err, errlen, "generic graph session evaluation is unsupported");
    }
    #else
    snprintf(err, errlen, "generic graph session evaluation is unsupported");
    #endif
    return 1;
#endif
}

static int ds4_session_eval_probe(ds4_session *s, int token,
                                  char *err, size_t errlen) {
#ifdef DS4_TEST_HOOKS
    if (ds4_session_is_laguna(s) && g_ds4_test_route_eval_fn) {
        g_ds4_test_route_eval_calls++;
        return g_ds4_test_route_eval_fn(s, token, err, errlen);
    }
#endif
    return ds4_session_eval_internal(s, token, err, errlen);
}

int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (s && !laguna_metal_swa_gqa9_preflight(
            s->engine, "session eval", err, errlen)) return 1;
    if (s && !laguna_metal_router_simd_topk_preflight(
            s->engine, "session eval", err, errlen)) return 1;
#endif
    ds4_session_note_logits_dirty(s);
    return ds4_session_eval_probe(s, token, err, errlen);
}

int ds4_sessions_eval_batch(ds4_decode_item *items, int count,
                            char *err, size_t errlen) {
    if (!items || count <= 0) {
        if (err && errlen) snprintf(err, errlen, "empty decode batch");
        return 1;
    }
    if (count == 1) {
        return ds4_session_eval(items[0].session, items[0].token, err, errlen);
    }
    ds4_session *first = items[0].session;
    if (!first || !first->engine) {
        if (err && errlen) snprintf(err, errlen, "decode batch has no session");
        return 1;
    }
    ds4_engine *e = first->engine;
    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        if (!s || s->engine != e) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d belongs to a different engine", i);
            }
            return 1;
        }
        if (items[i].token < 0 || items[i].token >= (int)DS4_N_VOCAB) {
            if (err && errlen) {
                snprintf(err, errlen, "decode batch item %d has an invalid token", i);
            }
            return 1;
        }
        for (int j = 0; j < i; j++) {
            if (items[j].session == s) {
                if (err && errlen) {
                    snprintf(err, errlen,
                             "decode batch repeats session at items %d and %d",
                             j, i);
                }
                return 1;
            }
        }
        if (s->checkpoint.len >= s->ctx_size) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d reached its context limit", i);
            }
            return 1;
        }
    }

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (!laguna_metal_swa_gqa9_preflight(
            first->engine, "session batch eval", err, errlen)) return 1;
    if (!laguna_metal_router_simd_topk_preflight(
            e, "session batch eval", err, errlen)) return 1;
#endif
    for (int i = 0; i < count; i++) {
        ds4_session_note_logits_dirty(items[i].session);
    }

    /* Preserve logical all-or-nothing behavior even on the serialized path.
     * A failure can leave earlier members advanced, so force every member to
     * rebuild before it is used again. */
    for (int i = 0; i < count; i++) {
        if (ds4_session_eval(items[i].session, items[i].token,
                             err, errlen) != 0) {
            for (int j = 0; j < count; j++) {
                ds4_session_invalidate(items[j].session);
            }
            return 1;
        }
    }
    return 0;
}

int ds4_sessions_eval_batch_with_prefill(
        ds4_decode_item *items,
        int count,
        ds4_session *prefill_session,
        const ds4_tokens *prefill_prompt,
        char *err,
        size_t errlen) {
    if (!items || count <= 0 || !prefill_session || !prefill_prompt ||
        !prefill_session->engine) {
        if (err && errlen) snprintf(err, errlen, "invalid mixed model batch");
        return 1;
    }
    if (!prefill_session->checkpoint_valid ||
        prefill_prompt->len <= prefill_session->checkpoint.len ||
        prefill_prompt->len >= prefill_session->ctx_size ||
        !ds4_tokens_starts_with(prefill_prompt, &prefill_session->checkpoint)) {
        if (err && errlen) {
            snprintf(err, errlen,
                     "mixed prefill must extend a valid session checkpoint");
        }
        return 1;
    }
    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        if (!s || s == prefill_session ||
            s->engine != prefill_session->engine ||
            !s->checkpoint_valid || s->checkpoint.len >= s->ctx_size ||
            items[i].token < 0 || items[i].token >= (int)DS4_N_VOCAB) {
            if (err && errlen) {
                snprintf(err, errlen, "invalid mixed decode item %d", i);
            }
            return 1;
        }
        for (int j = 0; j < i; j++) {
            if (items[j].session == s) {
                if (err && errlen) {
                    snprintf(err, errlen,
                             "mixed decode repeats session at items %d and %d",
                             j, i);
                }
                return 1;
            }
        }
    }

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (!laguna_metal_swa_gqa9_preflight(
            prefill_session->engine, "session mixed prefill/eval",
            err, errlen)) return 1;
    if (!laguna_metal_router_simd_topk_preflight(
            prefill_session->engine, "session mixed prefill/eval", err, errlen)) {
        return 1;
    }
#endif

    ds4_session_note_logits_dirty(prefill_session);
    for (int i = 0; i < count; i++) {
        ds4_session_note_logits_dirty(items[i].session);
    }

    int rc = ds4_session_sync(prefill_session, prefill_prompt, err, errlen);
    if (rc != 0) return rc;
    rc = ds4_sessions_eval_batch(items, count, err, errlen);
    if (rc != 0) ds4_session_invalidate(prefill_session);
    return rc;
}


#ifndef DS4_NO_GPU
static int ds4_session_eval_dflash_speculative_argmax(
        ds4_session *s,
        int          first_token,
        int          max_tokens,
        int          eos_token,
        int         *accepted,
        int          accepted_cap,
        char        *err,
        size_t       errlen) {
    ds4_engine *e = s->engine;
#ifdef __APPLE__
    /* Do this before snapshotting target KV or touching the support graph so
     * an explicit SIMD32 request can never degrade into a split verifier. */
    if (!laguna_metal_qk_norm_rope_simd32_preflight()) {
        if (errlen) snprintf(err, errlen,
                             "Laguna Q/K norm/RoPE SIMD32 preflight failed");
        return -1;
    }
#endif
    if (!ds4_session_dflash_enabled(s) || !s->dflash_synced ||
        e->dflash_draft_tokens <= 0 || first_token == eos_token ||
        max_tokens <= 1 || accepted_cap <= 1) {
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    if (s->dflash_suspended || s->dflash_baseline_tokens < 3u) {
        const double decode_t0 = now_sec();
        const bool defer_inject = !s->dflash_suspended;
        s->dflash_defer_inject = defer_inject;
        const int eval_rc = ds4_session_eval(s, first_token, err, errlen);
        s->dflash_defer_inject = false;
        if (eval_rc != 0) return -1;
        const double decode_ms = (now_sec() - decode_t0) * 1000.0;
        if (!s->dflash_suspended) {
            s->dflash_baseline_tokens++;
            /* The first target token completes the support-model injection
             * left queued by prefill. Keep it out of the plain-decode
             * baseline, then inject all three calibration rows in one batch. */
            if (s->dflash_baseline_tokens > 1u) {
                s->dflash_baseline_ms += decode_ms + s->last_sample_ms;
            }
            if (s->dflash_baseline_tokens == 3u) {
                s->dflash_baseline_ms /= 2.0;
                if (!ds4_session_dflash_flush_deferred(s)) {
                    if (errlen) snprintf(err, errlen,
                                         "DFlash baseline injection failed");
                    return -1;
                }
            }
        }
        accepted[0] = first_token;
        return 1;
    }
    if (s->dflash_cycles_since_baseline >= 64u) {
        /* Long memory-bandwidth-bound runs can settle at a different clock
         * than their first few tokens. Periodically spend one ordinary target
         * token to refresh the break-even estimate under current conditions. */
        const double decode_t0 = now_sec();
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        const double current_ms =
            (now_sec() - decode_t0) * 1000.0 + s->last_sample_ms;
        s->dflash_baseline_ms =
            0.5 * s->dflash_baseline_ms + 0.5 * current_ms;
        s->dflash_cycles_since_baseline = 0;
        accepted[0] = first_token;
        return 1;
    }

    uint32_t requested_draft = (uint32_t)e->dflash_draft_tokens;
    if (requested_draft > DS4_DFLASH_BLOCK_SIZE - 1u) {
        requested_draft = DS4_DFLASH_BLOCK_SIZE - 1u;
    }
    if (s->dflash_active_draft == 0u) {
        s->dflash_active_draft = requested_draft < 3u ? requested_draft : 3u;
    }
    uint32_t n_draft = requested_draft;
    if (n_draft > s->dflash_active_draft) {
        n_draft = s->dflash_active_draft;
    }
    if (n_draft > (uint32_t)(max_tokens - 1)) {
        n_draft = (uint32_t)(max_tokens - 1);
    }
    if (n_draft > (uint32_t)(accepted_cap - 1)) {
        n_draft = (uint32_t)(accepted_cap - 1);
    }
    const int room = s->ctx_size - s->checkpoint.len;
    if (room <= 1) n_draft = 0;
    else if (n_draft > (uint32_t)(room - 1)) {
        n_draft = (uint32_t)(room - 1);
    }
    if (n_draft == 0u) {
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }

    const uint32_t pos0 = (uint32_t)s->checkpoint.len;
    uint32_t n_rows = n_draft + 1u;
    const uint32_t generated_draft = n_draft;
    int draft_top[DS4_DFLASH_BLOCK_SIZE] = {0};
    int target_top[DS4_DFLASH_BLOCK_SIZE] = {0};
    int verify_tokens[DS4_DFLASH_BLOCK_SIZE] = {0};
    const float draft_p_min = e->dflash_p_min;
    bool draft_read_ok = false;
    bool snapshot_submitted = false;
    bool snapshot_completed = false;

    const double cycle_t0 = now_sec();
    if (!laguna_graph_spec_snapshot(&s->laguna_graph, pos0, n_rows)) {
        if (errlen) snprintf(err, errlen, "Laguna verifier snapshot failed");
        return -1;
    }
#ifdef __APPLE__
    /* Commit the target KV backup before the speculative replacement batch is
     * allowed to be discarded.  The replacement batch is queue-ordered after
     * this nonblocking flush, so restore can safely read the committed backup
     * later without a CPU wait here. */
    if (ds4_gpu_flush_commands() == 0) {
        ds4_session_dflash_quarantine(s);
        if (errlen) snprintf(err, errlen,
                             "Laguna verifier snapshot commit failed");
        return -1;
    }
#endif
    snapshot_submitted = true;

    /* Rejected support-model rows are overwritten by the next draft or
     * target-feature injection before they can be consumed. The target graph
     * has different staged-attention semantics and still needs its rollback. */
    if (!dflash_graph_draft_block(&s->dflash_graph,
                                  e,
                                  first_token,
                                  pos0,
                                  n_draft)) {
        fprintf(stderr,
                "ds4: DFlash draft failed; falling back to Laguna decode\n");
        const bool draft_discard_ok =
            ds4_session_dflash_discard_owned_commands();
        /* A successful pre-submit discard proves the support transaction was
         * never promoted.  Preserve the valid host checkpoint while disabling
         * another speculative attempt, matching the existing fallback path. */
#ifdef __APPLE__
        laguna_metal_target_evidence_discard(&s->laguna_graph);
#endif
        s->dflash_synced = false;
        if (!draft_discard_ok) {
            /* A failed discard may have waited work that was already
             * submitted.  Do not let the old checkpoint certify it. */
            ds4_session_dflash_quarantine(s);
            if (errlen) snprintf(err, errlen,
                                 "DFlash draft rollback failed");
            return -1;
        }
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    const double submit_done = now_sec();
    bool target_preencoded = false;
#ifdef __APPLE__
    if (draft_p_min > 0.0f) {
        if (ds4_gpu_flush_commands() == 0) {
            /* The draft may already be in flight even when creation of the
             * replacement batch fails.  Treat this as terminal and poison
             * both sync and checkpoint state. */
            const bool rollback_ok = !ds4_gpu_commands_active() ||
                ds4_gpu_discard_commands() != 0;
            ds4_session_dflash_quarantine(s);
            if (errlen) snprintf(err, errlen, "%s",
                                 rollback_ok ?
                                 "DFlash draft batch flush failed" :
                                 "DFlash draft batch flush rollback failed");
            return -1;
        }
        /*
         * Confidence keeps the full draft on most cycles. Encode that common
         * verifier while the draft is executing, then either commit it after
         * reading confidence or discard it and encode the shorter verifier.
         */
        verify_tokens[0] = first_token;
        ds4_laguna_feature_capture speculative_capture =
            ds4_session_dflash_capture(s, 0, n_rows);
        target_preencoded = laguna_graph_forward_batch(
            &s->laguna_graph,
            &e->model,
            &e->weights,
            verify_tokens,
            s->dflash_graph.argmax,
            n_rows,
            pos0,
            NULL,
            target_top,
            &speculative_capture,
            NULL,
            NULL,
            0);
        if (!target_preencoded) {
            const bool target_discard_ok =
                ds4_session_dflash_discard_owned_commands();
            if (!target_discard_ok) {
                ds4_session_dflash_quarantine(s);
                if (errlen) snprintf(err, errlen,
                                     "DFlash target pre-encode rollback failed");
                return -1;
            }
            /* This terminal discard waits the snapshot CB(s) that were
             * submitted before the target attempt.  Do not later mistake an
             * empty wait for that proof; carry the successful boundary state
             * forward explicitly. */
            if (snapshot_submitted) snapshot_completed = true;
        } else if (!ds4_gpu_commands_active()) {
            ds4_session_dflash_quarantine(s);
            if (errlen) snprintf(err, errlen,
                                 "DFlash target pre-encode lost its batch");
            return -1;
        }
    }
#endif
    if (draft_p_min > 0.0f) {
        float draft_probabilities[DS4_DFLASH_BLOCK_SIZE] = {0};
#ifdef __APPLE__
        const bool snapshot_wait_ok =
            ds4_gpu_wait_submitted_commands() != 0;
        /* wait_submitted_commands covers the committed snapshot and the
         * speculative draft CB in queue order.  A reported failure means the
         * backup must remain untrusted, even if the device completed some
         * work before reporting that failure. */
        snapshot_completed = snapshot_wait_ok;
        draft_read_ok =
            snapshot_wait_ok &&
            ds4_gpu_tensor_read(
                s->dflash_graph.argmax,
                0,
                draft_top,
                (uint64_t)n_rows * sizeof(draft_top[0])) != 0 &&
            ds4_gpu_tensor_read(
                s->dflash_graph.probabilities,
                0,
                draft_probabilities,
                (uint64_t)n_rows *
                    sizeof(draft_probabilities[0])) != 0;
#else
        const bool snapshot_end_ok =
            ds4_gpu_end_commands() != 0;
        snapshot_completed = snapshot_end_ok;
        draft_read_ok =
            snapshot_end_ok &&
            ds4_gpu_tensor_read(
                s->dflash_graph.argmax,
                0,
                draft_top,
                (uint64_t)n_rows * sizeof(draft_top[0])) != 0 &&
            ds4_gpu_tensor_read(
                s->dflash_graph.probabilities,
                0,
                draft_probabilities,
                (uint64_t)n_rows *
                    sizeof(draft_probabilities[0])) != 0;
#endif
        for (uint32_t i = 0; draft_read_ok && i < n_draft; i++) {
            if (draft_probabilities[i + 1u] < draft_p_min) {
                n_draft = i;
                n_rows = n_draft + 1u;
                break;
            }
        }
        if (target_preencoded && n_draft != generated_draft) {
            draft_read_ok = ds4_session_dflash_discard_owned_commands();
#ifdef __APPLE__
            /* The speculative target was discarded before completion; never
             * let its pre-encode snapshot certify the replacement target. */
            laguna_metal_target_evidence_discard(&s->laguna_graph);
#endif
            target_preencoded = false;
        }
#ifdef DS4_TEST_HOOKS
        /* Fault-injection hook for the narrow confidence-read failure path:
         * the speculative target may already have been encoded/deferred, so
         * its evidence must be discarded before returning the error. */
        const char *confidence_fault =
            getenv("DS4_TEST_DFLASH_CONFIDENCE_READ_FAIL");
        if (confidence_fault && strcmp(confidence_fault, "1") == 0) {
            draft_read_ok = false;
        }
#endif
        if (!draft_read_ok ||
            (!target_preencoded && ds4_gpu_begin_commands() == 0)) {
            const bool cleanup_ok = !ds4_gpu_commands_active() ?
                true : ds4_session_dflash_discard_owned_commands();
            ds4_session_dflash_quarantine(s);
            if (!cleanup_ok) {
                if (errlen) snprintf(err, errlen,
                                     "DFlash confidence rollback failed");
                return -1;
            }
            if (errlen) snprintf(err, errlen,
                                 "DFlash confidence cutoff failed");
            return -1;
        }
    }

    verify_tokens[0] = first_token;
    ds4_laguna_feature_capture capture =
        ds4_session_dflash_capture(s, 0, n_rows);
    bool verify_ok = target_preencoded;
    if (!target_preencoded) {
        verify_ok = laguna_graph_forward_batch(
                &s->laguna_graph,
                &e->model,
                &e->weights,
                verify_tokens,
                s->dflash_graph.argmax,
                n_rows,
                pos0,
                NULL,
                target_top,
                &capture,
                NULL,
                NULL,
                0);
        if (!verify_ok) {
            const bool discard_ok =
                ds4_session_dflash_discard_owned_commands();
            ds4_session_dflash_quarantine(s);
            if (errlen) snprintf(err, errlen,
                                 discard_ok ?
                                 "DFlash target verifier failed" :
                                 "DFlash target verifier rollback failed");
            return -1;
        }
    }
    bool inject_ok = false;
    if (verify_ok) {
        inject_ok = ds4_session_dflash_finish_capture(s, pos0, n_rows);
        /* p_min == 0 intentionally keeps snapshot, verification, and
         * injection in one transaction.  finish_capture's successful
         * terminal boundary is the completion proof for that snapshot. */
        if (inject_ok) snapshot_completed = true;
    }
    bool target_read_ok = false;
    if (inject_ok) {
        target_read_ok = ds4_gpu_tensor_read(
            s->laguna_graph.spec_argmax,
            0,
            target_top,
            (uint64_t)n_rows * sizeof(target_top[0])) != 0;
        if (!draft_read_ok) {
            draft_read_ok = ds4_gpu_tensor_read(
                s->dflash_graph.argmax,
                0,
                draft_top,
                (uint64_t)n_rows * sizeof(draft_top[0])) != 0;
        }
    }
    if (!verify_ok || !inject_ok || !target_read_ok || !draft_read_ok) {
        ds4_session_dflash_quarantine(s);
        if (snapshot_completed) {
            const bool repair_ok = ds4_session_dflash_restore_snapshot(
                s, snapshot_completed, pos0, 0, n_rows);
            /* This is best-effort repair after a terminal verifier failure;
             * neither repair outcome may re-certify the host checkpoint. */
            if (!repair_ok) ds4_session_dflash_quarantine(s);
        }
        if (errlen) snprintf(err, errlen,
                             "%s DFlash target verification failed",
                             DS4_RUNTIME_NAME);
        return -1;
    }

    int n_accept = 1;
    accepted[0] = first_token;
    for (uint32_t i = 0; i < n_draft; i++) {
        const int proposal = draft_top[i + 1u];
        if (target_top[i] != proposal) break;
        accepted[n_accept++] = proposal;
        if (proposal == eos_token) break;
    }

    /* The accepted-prefix restore is the only rollback that can repair the
     * target cache after a successful verifier.  Refuse to use the backup if
     * its submitting command never acquired a successful completion proof. */
    if (!snapshot_submitted || !snapshot_completed) {
        ds4_session_dflash_quarantine(s);
        if (errlen) snprintf(err, errlen,
                             "DFlash verifier snapshot completion missing");
        return -1;
    }

    const bool logits_ok = laguna_graph_read_spec_logits(
        &s->laguna_graph,
        (uint32_t)(n_accept - 1),
        s->logits);
    const bool target_restore_ok = ds4_session_dflash_restore_snapshot(
        s, snapshot_completed, pos0, (uint32_t)n_accept, n_rows);
    if (!logits_ok || !target_restore_ok) {
        ds4_session_dflash_quarantine(s);
        if (errlen) snprintf(err, errlen,
                             "DFlash verifier rollback failed");
        return -1;
    }
    for (int i = 0; i < n_accept; i++) {
        token_vec_push(&s->checkpoint, accepted[i]);
    }
    s->checkpoint_valid = true;
    s->dflash_synced = true;

    const double verify_done = now_sec();
    /* The anonymous F16 support mapping is installed lazily. Its first draft
     * cycle faults roughly 2 GiB of weights and is intentionally a one-time
     * warmup, not evidence that steady-state speculation is unprofitable. */
    const bool dflash_warmup_cycle = s->dflash_cycles == 0u;
    s->dflash_cycles++;
    s->dflash_cycles_since_baseline++;
    s->dflash_proposed += generated_draft;
    s->dflash_pruned += generated_draft - n_draft;
    s->dflash_accepted += (uint64_t)(n_accept - 1);
    s->dflash_draft_ms += (submit_done - cycle_t0) * 1000.0;
    s->dflash_verify_ms += (verify_done - submit_done) * 1000.0;
    const double cycle_ms =
        (verify_done - cycle_t0) * 1000.0 + s->last_sample_ms;
    if (!dflash_warmup_cycle) {
        s->dflash_window_ms += cycle_ms;
        s->dflash_window_tokens += (uint64_t)n_accept;
        s->dflash_window_cycles++;
    }
    if (!dflash_warmup_cycle &&
        n_draft == generated_draft &&
        (uint32_t)(n_accept - 1) == generated_draft) {
        s->dflash_stage_full_accepts++;
    }
    const uint64_t cumulative_tokens =
        s->dflash_cycles + s->dflash_accepted;
    const double cumulative_ms_per_token =
        cumulative_tokens != 0u ?
        (s->dflash_draft_ms + s->dflash_verify_ms) /
            (double)cumulative_tokens : 0.0;
    if (!s->dflash_guard_decided &&
        s->dflash_window_cycles >= 5u &&
        s->dflash_window_tokens != 0u) {
        const double spec_ms_per_token =
            s->dflash_window_ms / (double)s->dflash_window_tokens;
        if (spec_ms_per_token < s->dflash_baseline_ms * 0.98) {
            if (s->dflash_best_draft == 0u ||
                spec_ms_per_token < s->dflash_best_ms_per_token) {
                s->dflash_best_draft = s->dflash_active_draft;
                s->dflash_best_ms_per_token = spec_ms_per_token;
            }
            if (s->dflash_active_draft < requested_draft) {
                const bool short_block_correlates =
                    s->dflash_active_draft > 3u ||
                    (s->dflash_stage_full_accepts == 5u &&
                     s->dflash_proposed != 0u &&
                     s->dflash_accepted * 10u >=
                         s->dflash_proposed * 9u);
                if (short_block_correlates) {
                    uint32_t next = s->dflash_active_draft * 2u + 1u;
                    if (next > requested_draft) next = requested_draft;
                    s->dflash_active_draft = next;
                    if (getenv("DS4_DFLASH_TIMING") != NULL) {
                        fprintf(stderr,
                                "ds4: DFlash testing draft depth %u "
                                "after %.2f ms/token at depth %u\n",
                                next,
                                spec_ms_per_token,
                                s->dflash_best_draft);
                    }
                } else {
                    if (s->dflash_cycles >= 10u) {
                        s->dflash_guard_decided = true;
                    }
                    if (getenv("DS4_DFLASH_TIMING") != NULL) {
                        fprintf(stderr,
                                "ds4: DFlash retaining draft depth %u; "
                                "only %u/5 calibration blocks were fully "
                                "accepted\n",
                                s->dflash_active_draft,
                                s->dflash_stage_full_accepts);
                    }
                }
            } else {
                s->dflash_guard_decided = true;
                if (s->dflash_best_draft != s->dflash_active_draft) {
                    const uint32_t rejected = s->dflash_active_draft;
                    s->dflash_active_draft = s->dflash_best_draft;
                    fprintf(stderr,
                            "ds4: DFlash using draft depth %u; depth %u "
                            "measured %.2f ms/token versus %.2f ms/token "
                            "at the best depth\n",
                            s->dflash_active_draft,
                            rejected,
                            spec_ms_per_token,
                            s->dflash_best_ms_per_token);
                }
            }
        } else if (s->dflash_best_draft != 0u &&
                   s->dflash_best_draft != s->dflash_active_draft) {
            const uint32_t rejected = s->dflash_active_draft;
            s->dflash_active_draft = s->dflash_best_draft;
            s->dflash_guard_decided = true;
            fprintf(stderr,
                    "ds4: DFlash using draft depth %u; depth %u measured "
                    "%.2f ms/token versus %.2f ms normal decode\n",
                    s->dflash_active_draft,
                    rejected,
                    spec_ms_per_token,
                    s->dflash_baseline_ms);
        } else if (s->dflash_best_draft == s->dflash_active_draft &&
                   cumulative_ms_per_token <
                       s->dflash_baseline_ms * 0.98) {
            s->dflash_guard_decided = true;
            if (getenv("DS4_DFLASH_TIMING") != NULL) {
                fprintf(stderr,
                        "ds4: DFlash retaining draft depth %u after local "
                        "slowdown; cumulative cost is %.2f ms/token\n",
                        s->dflash_active_draft,
                        cumulative_ms_per_token);
            }
        } else if (s->dflash_cycles < 10u &&
                   spec_ms_per_token < s->dflash_baseline_ms * 1.05) {
            if (getenv("DS4_DFLASH_TIMING") != NULL) {
                fprintf(stderr,
                        "ds4: DFlash calibration is within startup noise "
                        "(%.2f vs %.2f ms/token); measuring another window\n",
                        spec_ms_per_token,
                        s->dflash_baseline_ms);
            }
        } else {
            s->dflash_guard_decided = true;
            s->dflash_suspended = true;
            s->dflash_synced = false;
            fprintf(stderr,
                    "ds4: DFlash paused for this turn "
                    "(%.2f ms/token speculative vs %.2f ms normal decode)\n",
                    spec_ms_per_token,
                    s->dflash_baseline_ms);
        }
        s->dflash_window_ms = 0.0;
        s->dflash_window_tokens = 0;
        s->dflash_window_cycles = 0;
        s->dflash_stage_full_accepts = 0;
    } else if (s->dflash_guard_decided &&
               s->dflash_window_cycles >= 8u &&
               s->dflash_window_tokens != 0u) {
        const double spec_ms_per_token =
            s->dflash_window_ms / (double)s->dflash_window_tokens;
        if (spec_ms_per_token >= s->dflash_baseline_ms * 1.02) {
            s->dflash_slow_windows++;
        } else {
            s->dflash_slow_windows = 0;
        }
        if (s->dflash_slow_windows >= 2u &&
            cumulative_ms_per_token >= s->dflash_baseline_ms * 1.02) {
            s->dflash_suspended = true;
            s->dflash_synced = false;
            fprintf(stderr,
                    "ds4: DFlash paused for this turn after measured slowdown "
                    "(%.2f ms/token cumulative vs %.2f ms normal decode)\n",
                    cumulative_ms_per_token,
                    s->dflash_baseline_ms);
        }
        s->dflash_window_ms = 0.0;
        s->dflash_window_tokens = 0;
        s->dflash_window_cycles = 0;
    }
    if (getenv("DS4_DFLASH_TIMING") != NULL) {
        fprintf(stderr,
                "ds4: DFlash cycle drafted=%u verified=%u accepted=%d "
                "pipeline=%.3f ms\n",
                generated_draft,
                n_draft,
                n_accept - 1,
                (verify_done - cycle_t0) * 1000.0);
    }
    return n_accept;
}
#endif

int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    if (!s || !accepted || max_tokens <= 0 || accepted_cap <= 0) return 0;
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (!laguna_metal_swa_gqa9_preflight(
            s->engine, "speculative eval", err, errlen)) return -1;
    if (!laguna_metal_router_simd_topk_preflight(
            s->engine, "speculative eval", err, errlen)) return -1;
    const bool laguna_router_simd_topk_trace =
        ds4_session_is_laguna(s) &&
        laguna_metal_router_simd_topk_trace_enabled();
    if (laguna_router_simd_topk_trace) {
        /* DFlash may stage a support-model command stream before its target
         * batch. Reset at this public boundary; the graph batch itself also
         * resets defensively before an ordinary non-deferred invocation. */
        laguna_metal_router_simd_topk_trace_reset();
    }
#endif
    ds4_session_note_logits_dirty(s);
#ifdef DS4_NO_GPU
    (void)s; (void)first_token; (void)max_tokens; (void)eos_token;
    (void)accepted; (void)accepted_cap;
    snprintf(err, errlen, "GPU support is not compiled in");
    return -1;
#else
    ds4_engine *e = s->engine;
    if (ds4_session_is_laguna(s) &&
        e->support_kind == DS4_SUPPORT_DFLASH) {
        if (!laguna_dense_q8_gate_up_swiglu_preflight(
                    &e->model, &e->weights, NULL)) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "%s Laguna dense Q8 gate/up+SwiGLU preflight failed",
                         DS4_RUNTIME_NAME);
            }
            return -1;
        }
#ifdef __APPLE__
        /* DFlash's first action is a speculative graph snapshot.  Reject a
         * malformed/unsupported residual-fusion request before that snapshot
         * or any support/KV mutation, matching the ordinary session paths. */
        if (!laguna_metal_decode_residual_norm_preflight()) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "%s Laguna decode residual fusion preflight failed",
                         DS4_RUNTIME_NAME);
            }
            return -1;
        }
#endif
        const int rc = ds4_session_eval_dflash_speculative_argmax(
            s,
            first_token,
            max_tokens,
            eos_token,
            accepted,
            accepted_cap,
            err,
            errlen);
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
        if (laguna_router_simd_topk_trace) {
            /* The DFlash helper waits before reading/rolling back the target
             * graph, so this report is safe even when its first target batch
             * was encoded with deferred completion. */
            laguna_metal_router_simd_topk_trace_report(
                "Laguna speculative eval");
        }
#endif
        return rc;
    }
    /* Non-DFlash speculative requests use the ordinary target evaluator. */
    if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
    accepted[0] = first_token;
    return 1;
#endif
}

void ds4_session_invalidate(ds4_session *s) {
    if (!s) return;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
#ifndef DS4_NO_GPU
    ds4_session_dflash_invalidate(s);
#endif
}

void ds4_session_rewind(ds4_session *s, int pos) {
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
#ifndef DS4_NO_GPU
    ds4_session_dflash_invalidate(s);
#endif
}

int ds4_session_pos(ds4_session *s) {
    return s->checkpoint.len;
}

int ds4_session_ctx(ds4_session *s) {
    return s->ctx_size;
}

int ds4_session_prefill_cap(ds4_session *s) {
    return s ? (int)s->prefill_cap : 0;
}

#ifdef DS4_TEST_HOOKS
#ifndef DS4_NO_GPU
static int ds4_test_laguna_argmax_eval_stub(
        ds4_session *s, int token, char *err, size_t errlen) {
    if (!s || !s->logits || token != 23 || DS4_N_VOCAB <= 37u) {
        if (err && errlen) snprintf(err, errlen, "invalid Laguna test eval");
        return 1;
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) s->logits[i] = -2.0f;
    s->logits[37] = 9.0f;
    return 0;
}

static int ds4_test_laguna_route_sync_stub(
        ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0) {
        if (err && errlen) snprintf(err, errlen, "invalid Laguna sync seam");
        return 1;
    }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    return 0;
}

static int ds4_test_laguna_route_sync_fail_stub(
        ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    (void)s;
    (void)prompt;
    if (err && errlen) snprintf(err, errlen, "synthetic Laguna sync interruption");
    return DS4_SESSION_SYNC_INTERRUPTED;
}

static int ds4_test_laguna_route_eval_stub(
        ds4_session *s, int token, char *err, size_t errlen) {
    if (!s || token < 0 || token >= (int)DS4_N_VOCAB) {
        if (err && errlen) snprintf(err, errlen, "invalid Laguna eval seam");
        return 1;
    }
    token_vec_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    return 0;
}

static int ds4_test_laguna_route_eval_fail_second_stub(
        ds4_session *s, int token, char *err, size_t errlen) {
    if (g_ds4_test_route_eval_calls == 2u) {
        (void)s;
        (void)token;
        if (err && errlen) snprintf(err, errlen, "synthetic Laguna decode failure");
        return 1;
    }
    return ds4_test_laguna_route_eval_stub(s, token, err, errlen);
}

/* Exercise the public mixed-prefill and argmax boundaries with model-free
 * Laguna leaves.  The route hooks replace only sync/eval after validation;
 * the raw-evaluator guard makes any accidental generic dispatch observable. */
bool ds4_test_laguna_session_routes(void) {
    ds4_test_laguna_shape_scope shape_scope;
    ds4_test_laguna_shape_scope_begin(&shape_scope);

    const char *saved_swa_env = getenv("DS4_METAL_LAGUNA_SWA_GQA9");
    char *saved_swa_value = NULL;
    if (saved_swa_env) {
        const size_t n = strlen(saved_swa_env) + 1u;
        saved_swa_value = malloc(n);
        if (saved_swa_value) memcpy(saved_swa_value, saved_swa_env, n);
    }
    bool env_ready = true;
    if (!saved_swa_env || saved_swa_value) {
        env_ready = setenv("DS4_METAL_LAGUNA_SWA_GQA9", "0", 1) == 0;
    }

    ds4_engine engine;
    memset(&engine, 0, sizeof(engine));
    engine.support_kind = DS4_SUPPORT_NONE;
    engine.metal_ready = true;

    ds4_session prefill;
    ds4_session decoder;
    memset(&prefill, 0, sizeof(prefill));
    memset(&decoder, 0, sizeof(decoder));
    prefill.engine = &engine;
    prefill.ctx_size = 8;
    prefill.checkpoint_valid = true;
    decoder.engine = &engine;
    decoder.ctx_size = 8;
    decoder.checkpoint_valid = true;
    decoder.logits = calloc(DS4_N_VOCAB, sizeof(decoder.logits[0]));

    const int prompt_tokens[] = {11};
    const ds4_tokens prompt = {
        .v = (int *)(uintptr_t)prompt_tokens,
        .len = 1,
        .cap = 1,
    };
    ds4_decode_item item = { .session = &decoder, .token = 17 };
    const bool laguna_session = ds4_session_is_laguna(&prefill) &&
                                ds4_session_is_laguna(&decoder);
    g_ds4_test_argmax_eval_fn = ds4_test_laguna_argmax_eval_stub;
    g_ds4_test_argmax_laguna_eval_calls = 0;
    g_ds4_test_route_sync_fn = ds4_test_laguna_route_sync_stub;
    g_ds4_test_route_eval_fn = ds4_test_laguna_route_eval_stub;
    g_ds4_test_route_sync_calls = 0;
    g_ds4_test_route_eval_calls = 0;
    g_ds4_test_route_generic_eval_calls = 0;
    g_ds4_test_route_generic_access_forbidden = true;
    char err[128] = {0};
    const int mixed_rc = ds4_sessions_eval_batch_with_prefill(
        &item, 1, &prefill, &prompt, err, sizeof(err));
    const int argmax = ds4_session_eval_argmax(
        &decoder, 23, err, sizeof(err));
    const bool decoder_valid_before_null_spec = decoder.checkpoint_valid;
    const int decoder_len_before_null_spec = decoder.checkpoint.len;
    const int decoder_token_before_null_spec =
        decoder.checkpoint.len > 0 ? decoder.checkpoint.v[0] : 0;
    const uint32_t route_eval_calls_before_null_spec =
        g_ds4_test_route_eval_calls;
    const uint32_t generic_eval_calls_before_null_spec =
        g_ds4_test_route_generic_eval_calls;
    const uint64_t logits_gen_before_null_spec = decoder.logits_gen;
    const int null_spec_rc = ds4_session_eval_speculative_argmax(
        &decoder, 23, 1, -1, NULL, 1, err, sizeof(err));
    const bool route_success =
        env_ready && laguna_session && decoder.logits && mixed_rc == 0 &&
        prefill.checkpoint_valid && prefill.checkpoint.len == prompt.len &&
        decoder.checkpoint_valid && decoder.checkpoint.len == 1 &&
        argmax == 37 &&
        g_ds4_test_route_sync_calls == 1u &&
        g_ds4_test_route_eval_calls == 1u &&
        g_ds4_test_argmax_laguna_eval_calls == 1u &&
        g_ds4_test_route_generic_eval_calls == 0u &&
        null_spec_rc == 0 && decoder.logits_gen == logits_gen_before_null_spec &&
        decoder.checkpoint_valid == decoder_valid_before_null_spec &&
        decoder.checkpoint.len == decoder_len_before_null_spec &&
        (decoder.checkpoint.len == 0 ||
         decoder.checkpoint.v[0] == decoder_token_before_null_spec) &&
        g_ds4_test_route_eval_calls == route_eval_calls_before_null_spec &&
        g_ds4_test_route_generic_eval_calls == generic_eval_calls_before_null_spec &&
        err[0] == '\0';

    /* Invalid transactions must be rejected before any member's logits
     * generation counter is dirtied.  This covers both the serialized batch
     * validator and the mixed-prefill prefix/context validator. */
    ds4_session invalid_decoder;
    memset(&invalid_decoder, 0, sizeof(invalid_decoder));
    invalid_decoder.engine = &engine;
    invalid_decoder.ctx_size = 8;
    invalid_decoder.checkpoint_valid = true;
    ds4_decode_item invalid_items[2] = {
        { .session = &decoder, .token = 18 },
        { .session = &invalid_decoder, .token = -1 },
    };
    const uint64_t decoder_gen_before_invalid_batch = decoder.logits_gen;
    const uint64_t invalid_gen_before_invalid_batch = invalid_decoder.logits_gen;
    char invalid_err[128] = {0};
    const int invalid_batch_rc = ds4_sessions_eval_batch(
        invalid_items, 2, invalid_err, sizeof(invalid_err));
    const int bad_prompt_tokens[] = {99, 12};
    const ds4_tokens bad_prompt = {
        .v = (int *)(uintptr_t)bad_prompt_tokens,
        .len = 2,
        .cap = 2,
    };
    const uint64_t prefill_gen_before_invalid_mixed = prefill.logits_gen;
    const uint64_t decoder_gen_before_invalid_mixed = decoder.logits_gen;
    const int invalid_mixed_rc = ds4_sessions_eval_batch_with_prefill(
        &item, 1, &prefill, &bad_prompt, invalid_err, sizeof(invalid_err));
    const bool invalid_input_ok =
        invalid_batch_rc != 0 && invalid_mixed_rc != 0 &&
        decoder.logits_gen == decoder_gen_before_invalid_batch &&
        invalid_decoder.logits_gen == invalid_gen_before_invalid_batch &&
        prefill.logits_gen == prefill_gen_before_invalid_mixed &&
        decoder.logits_gen == decoder_gen_before_invalid_mixed;

    /* A real serialized decode failure after one member has advanced must
     * invalidate every member, so no caller can resume from a partial batch. */
    ds4_session batch_a;
    ds4_session batch_b;
    memset(&batch_a, 0, sizeof(batch_a));
    memset(&batch_b, 0, sizeof(batch_b));
    batch_a.engine = &engine;
    batch_b.engine = &engine;
    batch_a.ctx_size = 8;
    batch_b.ctx_size = 8;
    batch_a.checkpoint_valid = true;
    batch_b.checkpoint_valid = true;
    ds4_decode_item failing_items[2] = {
        { .session = &batch_a, .token = 18 },
        { .session = &batch_b, .token = 19 },
    };
    g_ds4_test_route_eval_fn = ds4_test_laguna_route_eval_fail_second_stub;
    g_ds4_test_route_eval_calls = 0;
    memset(invalid_err, 0, sizeof(invalid_err));
    const int decode_fail_rc = ds4_sessions_eval_batch(
        failing_items, 2, invalid_err, sizeof(invalid_err));
    const bool decode_failure_ok =
        decode_fail_rc != 0 &&
        g_ds4_test_route_eval_calls == 2u &&
        !batch_a.checkpoint_valid && batch_a.checkpoint.len == 0 &&
        !batch_b.checkpoint_valid && batch_b.checkpoint.len == 0;

    /* A sync interruption is a pre-decode failure: it returns unchanged
     * checkpoint validity/positions, invokes no decoder, and does not trigger
     * the all-members invalidation reserved for a decode failure. */
    const int sync_fail_prompt_tokens[] = {11, 12};
    const ds4_tokens sync_fail_prompt = {
        .v = (int *)(uintptr_t)sync_fail_prompt_tokens,
        .len = 2,
        .cap = 2,
    };
    const bool prefill_valid_before_sync_fail = prefill.checkpoint_valid;
    const int prefill_len_before_sync_fail = prefill.checkpoint.len;
    const int prefill_token_before_sync_fail =
        prefill.checkpoint.len > 0 ? prefill.checkpoint.v[0] : 0;
    const bool decoder_valid_before_sync_fail = decoder.checkpoint_valid;
    const int decoder_len_before_sync_fail = decoder.checkpoint.len;
    const int decoder_token_before_sync_fail =
        decoder.checkpoint.len > 0 ? decoder.checkpoint.v[0] : 0;
    g_ds4_test_route_sync_fn = ds4_test_laguna_route_sync_fail_stub;
    g_ds4_test_route_eval_fn = ds4_test_laguna_route_eval_stub;
    g_ds4_test_route_sync_calls = 0;
    g_ds4_test_route_eval_calls = 0;
    memset(invalid_err, 0, sizeof(invalid_err));
    const int sync_fail_rc = ds4_sessions_eval_batch_with_prefill(
        &item, 1, &prefill, &sync_fail_prompt,
        invalid_err, sizeof(invalid_err));
    const bool sync_failure_ok =
        sync_fail_rc == DS4_SESSION_SYNC_INTERRUPTED &&
        g_ds4_test_route_sync_calls == 1u &&
        g_ds4_test_route_eval_calls == 0u &&
        g_ds4_test_route_generic_eval_calls == 0u &&
        prefill.checkpoint_valid == prefill_valid_before_sync_fail &&
        prefill.checkpoint.len == prefill_len_before_sync_fail &&
        (prefill.checkpoint.len == 0 ||
         prefill.checkpoint.v[0] == prefill_token_before_sync_fail) &&
        decoder.checkpoint_valid == decoder_valid_before_sync_fail &&
        decoder.checkpoint.len == decoder_len_before_sync_fail &&
        (decoder.checkpoint.len == 0 ||
         decoder.checkpoint.v[0] == decoder_token_before_sync_fail);
    const bool route_ok = route_success && invalid_input_ok &&
                          decode_failure_ok && sync_failure_ok;

    g_ds4_test_argmax_eval_fn = NULL;
    g_ds4_test_argmax_laguna_eval_calls = 0;
    g_ds4_test_route_sync_fn = NULL;
    g_ds4_test_route_eval_fn = NULL;
    g_ds4_test_route_sync_calls = 0;
    g_ds4_test_route_eval_calls = 0;
    g_ds4_test_route_generic_eval_calls = 0;
    g_ds4_test_route_generic_access_forbidden = false;
    token_vec_free(&prefill.checkpoint);
    token_vec_free(&decoder.checkpoint);
    free(decoder.logits);
    if (saved_swa_value) {
        (void)setenv("DS4_METAL_LAGUNA_SWA_GQA9", saved_swa_value, 1);
    } else if (!saved_swa_env) {
        (void)unsetenv("DS4_METAL_LAGUNA_SWA_GQA9");
    }
    free(saved_swa_value);

    ds4_test_laguna_shape_scope_end(&shape_scope);
    return route_ok;
}

/* Exercise both sides of the storage boundary.  The direct lgn_* leg proves
 * that base storage teardown leaves extension/diagnostic state alone; the
 * wrapper leg proves the complete owner still gets an idempotent full free. */
bool ds4_test_laguna_graph_lifecycle(void) {
    _Static_assert(sizeof(lgn_gpu_graph) == 3248u,
                   "Laguna graph layout changed on Apple");
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = *lgn_model_shape();

    uint64_t handles_before = 0;
    uint64_t bytes_before = 0;
    if (!ds4_gpu_test_tensor_tracking_state(&handles_before,
                                            &bytes_before)) {
        g_ds4_shape = saved_shape;
        return false;
    }

    bool ok = true;
    lgn_gpu_graph direct = {0};
    ds4_shape wrong_family = g_ds4_shape;
    wrong_family.family = DS4_MODEL_FAMILY_DEEPSEEK4;
    direct.dense_q8_fusion_enabled = true;
    direct.dense_q8_pending.decode_mid_fused = 77u;
    int32_t shared_id = -1;
    float shared_weight = 0.0f;
    if (lgn_graph_alloc(&direct, 0u, &g_ds4_shape) ||
        lgn_graph_alloc(&direct, 1u, &wrong_family) ||
        !lgn_graph_alloc(&direct, 1u, &g_ds4_shape) ||
        !direct.tokens || !direct.cur || !direct.logits ||
        direct.cache_cap[0] != 1u || direct.cache_cap[1] != 1u ||
        direct.cache_cap[47] != 1u || !direct.dense_q8_fusion_enabled ||
        direct.dense_q8_pending.decode_mid_fused != 77u ||
        !ds4_gpu_tensor_read(direct.shared_selected, 0, &shared_id,
                             sizeof(shared_id)) ||
        !ds4_gpu_tensor_read(direct.shared_weight, 0, &shared_weight,
                             sizeof(shared_weight)) ||
        shared_id != 0 || shared_weight != 1.0f) {
        ok = false;
    }
    lgn_graph_free(&direct);
    ok = ok && direct.dense_q8_fusion_enabled &&
         direct.dense_q8_pending.decode_mid_fused == 77u;
    memset(&direct, 0, sizeof(direct));

    ds4_laguna_gpu_graph wrapped = {0};
    if (!laguna_graph_alloc(&wrapped, 1u) ||
        !wrapped.tokens || !wrapped.cur || !wrapped.logits ||
        wrapped.cache_cap[0] != 1u || wrapped.cache_cap[1] != 1u ||
        wrapped.cache_cap[47] != 1u ||
        !laguna_graph_enable_gpu_argmax(&wrapped) || !wrapped.argmax ||
        !wrapped.gpu_argmax_enabled) {
        ok = false;
    }
    laguna_graph_free(&wrapped);
    laguna_graph_free(&wrapped);
    for (size_t i = 0; i < sizeof(wrapped); i++) {
        if (((const unsigned char *)&wrapped)[i] != 0u) ok = false;
    }

    uint64_t handles_after = 0;
    uint64_t bytes_after = 0;
    ok = ok && ds4_gpu_test_tensor_tracking_state(&handles_after,
                                                    &bytes_after) &&
         handles_after == handles_before && bytes_after == bytes_before;
    g_ds4_shape = saved_shape;
    return ok;
}

/* Verify the DFlash module owns a complete storage-only owner while the
 * ds4.c wrapper still owns admission and failure diagnostics. */
bool ds4_test_laguna_dflash_graph_lifecycle(void) {
    _Static_assert(sizeof(lgn_dflash_graph) == 328u,
                   "DFlash graph layout changed on Apple");
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = *lgn_model_shape();
    const lgn_dflash_profile *profile = lgn_dflash_profile_get();

    uint64_t handles_before = 0;
    uint64_t bytes_before = 0;
    if (!profile ||
        !ds4_gpu_test_tensor_tracking_state(&handles_before, &bytes_before)) {
        g_ds4_shape = saved_shape;
        return false;
    }

    bool ok = true;
    lgn_dflash_graph partial = {0};
    partial.key_cache[0] = ds4_gpu_tensor_alloc(16u);
    if (!partial.key_cache[0]) ok = false;
    lgn_dflash_graph_free(&partial);
    lgn_dflash_graph_free(&partial);
    for (size_t i = 0; i < sizeof(partial); i++) {
        if (((const unsigned char *)&partial)[i] != 0u) ok = false;
    }

    lgn_dflash_graph direct = {0};
    if (!lgn_dflash_graph_alloc(&direct) ||
        direct.feature_cap != LGN_DFLASH_CACHE_CAP ||
        direct.block_cap != LGN_DFLASH_BLOCK_SIZE ||
        direct.cache_cap != LGN_DFLASH_CACHE_CAP) {
        ok = false;
    }
#define LGN_DFLASH_TEST_REQUIRED(name) \
    do { if (!direct.name) ok = false; } while (0)
    LGN_DFLASH_TEST_REQUIRED(features);
    LGN_DFLASH_TEST_REQUIRED(encoder);
    LGN_DFLASH_TEST_REQUIRED(encoder_norm);
    LGN_DFLASH_TEST_REQUIRED(norm);
    LGN_DFLASH_TEST_REQUIRED(tokens);
    LGN_DFLASH_TEST_REQUIRED(cur);
    LGN_DFLASH_TEST_REQUIRED(next);
    LGN_DFLASH_TEST_REQUIRED(q);
    LGN_DFLASH_TEST_REQUIRED(k);
    LGN_DFLASH_TEST_REQUIRED(v);
    LGN_DFLASH_TEST_REQUIRED(gate);
    LGN_DFLASH_TEST_REQUIRED(heads);
    LGN_DFLASH_TEST_REQUIRED(attn_out);
    LGN_DFLASH_TEST_REQUIRED(after_attn);
    LGN_DFLASH_TEST_REQUIRED(ffn_norm);
    LGN_DFLASH_TEST_REQUIRED(ffn_gate);
    LGN_DFLASH_TEST_REQUIRED(ffn_up);
    LGN_DFLASH_TEST_REQUIRED(ffn_mid);
    LGN_DFLASH_TEST_REQUIRED(ffn_out);
    LGN_DFLASH_TEST_REQUIRED(staged_key);
    LGN_DFLASH_TEST_REQUIRED(staged_value);
    LGN_DFLASH_TEST_REQUIRED(output_norm);
    LGN_DFLASH_TEST_REQUIRED(logits);
    LGN_DFLASH_TEST_REQUIRED(argmax);
    LGN_DFLASH_TEST_REQUIRED(probabilities);
#undef LGN_DFLASH_TEST_REQUIRED
    uint64_t scratch_sum = 0;
#define LGN_DFLASH_TEST_SUM(name) \
    do { scratch_sum += ds4_gpu_tensor_bytes(direct.name); } while (0)
    LGN_DFLASH_TEST_SUM(features);
    LGN_DFLASH_TEST_SUM(encoder);
    LGN_DFLASH_TEST_SUM(encoder_norm);
    LGN_DFLASH_TEST_SUM(norm);
    LGN_DFLASH_TEST_SUM(tokens);
    LGN_DFLASH_TEST_SUM(cur);
    LGN_DFLASH_TEST_SUM(next);
    LGN_DFLASH_TEST_SUM(q);
    LGN_DFLASH_TEST_SUM(k);
    LGN_DFLASH_TEST_SUM(v);
    LGN_DFLASH_TEST_SUM(gate);
    LGN_DFLASH_TEST_SUM(heads);
    LGN_DFLASH_TEST_SUM(attn_out);
    LGN_DFLASH_TEST_SUM(after_attn);
    LGN_DFLASH_TEST_SUM(ffn_norm);
    LGN_DFLASH_TEST_SUM(ffn_gate);
    LGN_DFLASH_TEST_SUM(ffn_up);
    LGN_DFLASH_TEST_SUM(ffn_mid);
    LGN_DFLASH_TEST_SUM(ffn_out);
    LGN_DFLASH_TEST_SUM(staged_key);
    LGN_DFLASH_TEST_SUM(staged_value);
    LGN_DFLASH_TEST_SUM(output_norm);
    LGN_DFLASH_TEST_SUM(logits);
    LGN_DFLASH_TEST_SUM(argmax);
    LGN_DFLASH_TEST_SUM(probabilities);
#undef LGN_DFLASH_TEST_SUM
    uint64_t kv_sum = 0;
    for (uint32_t il = 0; il < LGN_DFLASH_N_LAYER; il++) {
        if (!direct.key_cache[il] || !direct.value_cache[il]) ok = false;
        kv_sum += ds4_gpu_tensor_bytes(direct.key_cache[il]);
        kv_sum += ds4_gpu_tensor_bytes(direct.value_cache[il]);
    }
    if (direct.scratch_bytes != scratch_sum ||
        direct.kv_bytes != kv_sum ||
        direct.scratch_bytes == 0u || direct.kv_bytes == 0u ||
        direct.features == NULL || direct.key_cache[0] == NULL ||
        ds4_gpu_commands_active() != 0) {
        ok = false;
    }
    lgn_dflash_graph_free(&direct);
    lgn_dflash_graph_free(&direct);
    for (size_t i = 0; i < sizeof(direct); i++) {
        if (((const unsigned char *)&direct)[i] != 0u) ok = false;
    }

    ds4_dflash_gpu_graph wrapped = {0};
    ds4_shape wrong_family = g_ds4_shape;
    wrong_family.family = DS4_MODEL_FAMILY_DEEPSEEK4;
    g_ds4_shape = wrong_family;
    if (dflash_graph_alloc(&wrapped)) ok = false;
    for (size_t i = 0; i < sizeof(wrapped); i++) {
        if (((const unsigned char *)&wrapped)[i] != 0u) ok = false;
    }
    g_ds4_shape = *lgn_model_shape();
    if (!dflash_graph_alloc(&wrapped) ||
        wrapped.feature_cap != LGN_DFLASH_CACHE_CAP ||
        wrapped.block_cap != LGN_DFLASH_BLOCK_SIZE ||
        wrapped.cache_cap != LGN_DFLASH_CACHE_CAP ||
        !wrapped.features || !wrapped.logits || !wrapped.argmax ||
        !wrapped.probabilities || !wrapped.key_cache[0] ||
        !wrapped.value_cache[LGN_DFLASH_N_LAYER - 1u]) {
        ok = false;
    }
    dflash_graph_free(&wrapped);
    dflash_graph_free(&wrapped);
    for (size_t i = 0; i < sizeof(wrapped); i++) {
        if (((const unsigned char *)&wrapped)[i] != 0u) ok = false;
    }

    uint64_t handles_after = 0;
    uint64_t bytes_after = 0;
    ok = ok && ds4_gpu_test_tensor_tracking_state(&handles_after,
                                                    &bytes_after) &&
         handles_after == handles_before && bytes_after == bytes_before;
    g_ds4_shape = saved_shape;
    return ok;
}
#ifdef __APPLE__
/* Snapshot validity is a command-boundary contract, not a host pointer
 * convention.  Commit a synthetic target backup, discard a later active
 * mutation, and restore the exact committed row.  A second path deliberately
 * discards an uncommitted snapshot and verifies that it is never restored
 * from its stale backup. */
static bool dflash_graph_test_spec_snapshot_restore(void) {
    enum {
        TEST_CACHE_CAP = 2u,
    };
    const size_t test_row_bytes =
        (size_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t);
    ds4_laguna_gpu_graph target;
    memset(&target, 0, sizeof(target));
    ds4_gpu_tensor *mutation_key = NULL;
    ds4_gpu_tensor *mutation_value = NULL;
    uint8_t cache_host[TEST_CACHE_CAP * test_row_bytes];
    uint8_t value_host[TEST_CACHE_CAP * test_row_bytes];
    uint8_t current_key[test_row_bytes];
    uint8_t current_value[test_row_bytes];
    uint8_t mutated_key[test_row_bytes];
    uint8_t mutated_value[test_row_bytes];
    uint8_t readback[test_row_bytes];
    uint32_t chosen = UINT32_MAX;
    bool ok = false;

    for (size_t i = 0; i < test_row_bytes; i++) {
        cache_host[i] = 0x31u;
        cache_host[test_row_bytes + i] = 0x42u;
        value_host[i] = 0x51u;
        value_host[test_row_bytes + i] = 0x62u;
        current_key[i] = 0x42u;
        current_value[i] = 0x62u;
        mutated_key[i] = 0x91u;
        mutated_value[i] = 0xa2u;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!ds4_laguna_layer_is_swa(il)) continue;
        if (chosen == UINT32_MAX) chosen = il;
        target.cache_cap[il] = TEST_CACHE_CAP;
        target.key_cache[il] = ds4_gpu_tensor_alloc(sizeof(cache_host));
        target.value_cache[il] = ds4_gpu_tensor_alloc(sizeof(value_host));
        if (!target.key_cache[il] || !target.value_cache[il] ||
            !ds4_gpu_tensor_write(target.key_cache[il], 0,
                                   cache_host, sizeof(cache_host)) ||
            !ds4_gpu_tensor_write(target.value_cache[il], 0,
                                   value_host, sizeof(value_host))) {
            goto cleanup;
        }
    }
    if (chosen == UINT32_MAX || !laguna_graph_ensure_spec_scratch(&target)) {
        goto cleanup;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!ds4_laguna_layer_is_swa(il)) continue;
        if (!target.spec_key_backup[il] ||
            !target.spec_value_backup[il] ||
            !ds4_gpu_tensor_write(target.spec_key_backup[il],
                                  0, mutated_key, test_row_bytes) ||
            !ds4_gpu_tensor_write(target.spec_value_backup[il],
                                  0, mutated_value, test_row_bytes)) {
            goto cleanup;
        }
    }
    mutation_key = ds4_gpu_tensor_alloc(test_row_bytes);
    mutation_value = ds4_gpu_tensor_alloc(test_row_bytes);
    if (!mutation_key || !mutation_value ||
        !ds4_gpu_tensor_write(mutation_key, 0, mutated_key, test_row_bytes) ||
        !ds4_gpu_tensor_write(mutation_value, 0, mutated_value, test_row_bytes)) {
        goto cleanup;
    }

    if (!laguna_graph_spec_snapshot(&target, 1u, 1u) ||
        !ds4_gpu_commands_active() ||
        ds4_gpu_flush_commands() != 1 ||
        !ds4_gpu_commands_active() ||
        !ds4_gpu_tensor_copy(target.key_cache[chosen], test_row_bytes,
                             mutation_key, 0, test_row_bytes) ||
        !ds4_gpu_tensor_copy(target.value_cache[chosen], test_row_bytes,
                             mutation_value, 0, test_row_bytes) ||
        ds4_gpu_discard_commands() != 1 ||
        ds4_gpu_commands_active() != 0 ||
        !laguna_graph_spec_restore(&target, 1u, 0u, 1u) ||
        ds4_gpu_wait_submitted_commands() != 1 ||
        !ds4_gpu_tensor_read(target.key_cache[chosen], test_row_bytes,
                             readback, test_row_bytes) ||
        memcmp(readback, current_key, test_row_bytes) != 0 ||
        !ds4_gpu_tensor_read(target.value_cache[chosen], test_row_bytes,
                             readback, test_row_bytes) ||
        memcmp(readback, current_value, test_row_bytes) != 0 ||
        !ds4_gpu_tensor_read(target.key_cache[chosen], 0,
                             readback, test_row_bytes) ||
        memcmp(readback, cache_host, test_row_bytes) != 0 ||
        !ds4_gpu_tensor_read(target.value_cache[chosen], 0,
                             readback, test_row_bytes) ||
        memcmp(readback, value_host, test_row_bytes) != 0) {
        goto cleanup;
    }

    /* A discarded snapshot must not be treated as a valid backup. */
    for (size_t i = 0; i < test_row_bytes; i++) {
        cache_host[test_row_bytes + i] = 0x73u;
        value_host[test_row_bytes + i] = 0x83u;
    }
    if (!ds4_gpu_tensor_write(target.key_cache[chosen], 0,
                              cache_host, sizeof(cache_host)) ||
        !ds4_gpu_tensor_write(target.value_cache[chosen], 0,
                              value_host, sizeof(value_host)) ||
        !laguna_graph_spec_snapshot(&target, 1u, 1u) ||
        ds4_gpu_discard_commands() != 1 ||
        ds4_gpu_commands_active() != 0 ||
        !ds4_gpu_tensor_read(target.key_cache[chosen], test_row_bytes,
                             readback, test_row_bytes) ||
        memcmp(readback, cache_host + test_row_bytes, test_row_bytes) != 0 ||
        !ds4_gpu_tensor_read(target.value_cache[chosen], test_row_bytes,
                             readback, test_row_bytes) ||
        memcmp(readback, value_host + test_row_bytes, test_row_bytes) != 0) {
        goto cleanup;
    }

    /* A submitted snapshot is not complete merely because its CB was
     * submitted.  The hook below waits a real Metal CB to completion, then
     * reports a terminal wait failure and suppresses completion evidence.  A
     * later active write is committed so an accidental stale-backup restore
     * would visibly replace the distinct post-failure sentinel. */
    for (size_t i = 0; i < test_row_bytes; i++) {
        cache_host[test_row_bytes + i] = 0x73u;
        value_host[test_row_bytes + i] = 0x83u;
    }
    if (!ds4_gpu_tensor_write(target.key_cache[chosen], 0,
                              cache_host, sizeof(cache_host)) ||
        !ds4_gpu_tensor_write(target.value_cache[chosen], 0,
                              value_host, sizeof(value_host))) {
        goto cleanup;
    }
    const uint64_t wait_target_generated_before =
        ds4_gpu_laguna_rope_atlas_completed_generated_count();
    const uint64_t wait_target_consumed_before =
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
    const uint64_t wait_target_family0_before =
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u);
    const uint64_t wait_target_family1_before =
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u);
    const uint64_t wait_support_generated_before =
        ds4_gpu_laguna_rope_support_atlas_completed_generated_count();
    const uint64_t wait_support_consumed_before =
        ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count();
    if (!laguna_graph_spec_snapshot(&target, 1u, 1u) ||
        !ds4_gpu_commands_active() ||
        !ds4_gpu_laguna_rope_atlas_generate(1u, 1u) ||
        ds4_gpu_flush_commands() != 1 ||
        !ds4_gpu_commands_active()) {
        goto cleanup;
    }
    ds4_gpu_test_inject_wait_submitted_failure();
    const bool snapshot_wait_ok = ds4_gpu_wait_submitted_commands() != 0;
    ds4_session failed_wait_state;
    memset(&failed_wait_state, 0, sizeof(failed_wait_state));
    failed_wait_state.checkpoint_valid = true;
    failed_wait_state.dflash_synced = true;
    if (snapshot_wait_ok ||
        !ds4_gpu_commands_active() ||
        ds4_gpu_laguna_rope_atlas_completed_generated_count() !=
            wait_target_generated_before ||
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() !=
            wait_target_consumed_before ||
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u) !=
            wait_target_family0_before ||
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u) !=
            wait_target_family1_before ||
        ds4_gpu_laguna_rope_support_atlas_completed_generated_count() !=
            wait_support_generated_before ||
        ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count() !=
            wait_support_consumed_before) {
        goto cleanup;
    }
    ds4_session_dflash_quarantine(&failed_wait_state);
    if (failed_wait_state.checkpoint_valid || failed_wait_state.dflash_synced ||
        !ds4_gpu_tensor_copy(target.key_cache[chosen], test_row_bytes,
                             mutation_key, 0, test_row_bytes) ||
        !ds4_gpu_tensor_copy(target.value_cache[chosen], test_row_bytes,
                             mutation_value, 0, test_row_bytes) ||
        ds4_gpu_end_commands() != 1 ||
        ds4_gpu_commands_active() ||
        !ds4_session_dflash_restore_snapshot(
            &failed_wait_state, snapshot_wait_ok, 1u, 0u, 1u) ||
        !ds4_gpu_tensor_read(target.key_cache[chosen], test_row_bytes,
                             readback, test_row_bytes) ||
        memcmp(readback, mutated_key, test_row_bytes) != 0 ||
        !ds4_gpu_tensor_read(target.value_cache[chosen], test_row_bytes,
                             readback, test_row_bytes) ||
        memcmp(readback, mutated_value, test_row_bytes) != 0 ||
        !ds4_gpu_tensor_read(target.key_cache[chosen], 0,
                             readback, test_row_bytes) ||
        memcmp(readback, cache_host, test_row_bytes) != 0 ||
        !ds4_gpu_tensor_read(target.value_cache[chosen], 0,
                             readback, test_row_bytes) ||
        memcmp(readback, value_host, test_row_bytes) != 0) {
        goto cleanup;
    }

    /* The accepted-prefix product boundary must certify the restore CB
     * before it publishes checkpoint/DFlash state.  Put real atlas evidence
     * in an earlier pending CB, then inject a post-completion wait failure
     * while the accepted-prefix restore is submitted behind it. */
    if (!ds4_gpu_tensor_write(target.key_cache[chosen], 0,
                              cache_host, sizeof(cache_host)) ||
        !ds4_gpu_tensor_write(target.value_cache[chosen], 0,
                              value_host, sizeof(value_host)) ||
        !laguna_graph_spec_snapshot(&target, 1u, 1u) ||
        !ds4_gpu_commands_active() ||
        ds4_gpu_flush_commands() != 1 ||
        !ds4_gpu_commands_active() ||
        ds4_gpu_discard_commands() != 1 ||
        ds4_gpu_commands_active() ||
        ds4_gpu_wait_submitted_commands() != 1 ||
        !ds4_gpu_tensor_write(target.key_cache[chosen], test_row_bytes,
                              mutated_key, test_row_bytes) ||
        !ds4_gpu_tensor_write(target.value_cache[chosen], test_row_bytes,
                              mutated_value, test_row_bytes)) {
        goto cleanup;
    }
    const uint64_t restore_target_generated_before =
        ds4_gpu_laguna_rope_atlas_completed_generated_count();
    const uint64_t restore_target_consumed_before =
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
    const uint64_t restore_target_family0_before =
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u);
    const uint64_t restore_target_family1_before =
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u);
    const uint64_t restore_support_generated_before =
        ds4_gpu_laguna_rope_support_atlas_completed_generated_count();
    const uint64_t restore_support_consumed_before =
        ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count();
    if (ds4_gpu_begin_commands() != 1 ||
        !ds4_gpu_laguna_rope_atlas_generate(1u, 1u) ||
        ds4_gpu_submit_commands() != 1 ||
        ds4_gpu_commands_active()) {
        goto cleanup;
    }
    ds4_session accepted_wait_state;
    memset(&accepted_wait_state, 0, sizeof(accepted_wait_state));
    accepted_wait_state.laguna_graph = target;
    accepted_wait_state.checkpoint_valid = true;
    accepted_wait_state.dflash_synced = true;
    ds4_gpu_test_inject_wait_submitted_failure();
    const bool accepted_restore_ok =
        ds4_session_dflash_restore_snapshot(
            &accepted_wait_state, true, 1u, 0u, 1u);
    if (accepted_restore_ok) goto cleanup;
    ds4_session_dflash_quarantine(&accepted_wait_state);
    if (accepted_wait_state.checkpoint_valid ||
        accepted_wait_state.dflash_synced ||
        ds4_gpu_commands_active() ||
        ds4_gpu_wait_submitted_commands() != 1 ||
        ds4_gpu_laguna_rope_atlas_completed_generated_count() !=
            restore_target_generated_before ||
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() !=
            restore_target_consumed_before ||
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u) !=
            restore_target_family0_before ||
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u) !=
            restore_target_family1_before ||
        ds4_gpu_laguna_rope_support_atlas_completed_generated_count() !=
            restore_support_generated_before ||
        ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count() !=
            restore_support_consumed_before) {
        goto cleanup;
    }
    ok = true;

cleanup:
    if (ds4_gpu_commands_active()) (void)ds4_gpu_discard_commands();
    (void)ds4_gpu_wait_submitted_commands();
    ds4_gpu_tensor_free(mutation_value);
    ds4_gpu_tensor_free(mutation_key);
    laguna_graph_free(&target);
    return ok;
}

/* Model-independent command-ownership seam.  Each iteration queues real
 * DFlash K/V conversion, a support-K/RoPE kernel, and a target Q/K/RoPE
 * consumer in one Metal batch; fail_after injects the caller's post-layer
 * failure before the next layer.  This keeps rollback coverage on actual
 * queued writes without fabricating a model graph or moving scheduler policy
 * into the storage module. */
static bool dflash_graph_test_record_layers(
        ds4_dflash_gpu_graph *g,
        ds4_gpu_tensor       *source,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              k_weight_offset,
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        uint32_t              cache_cap,
        ds4_gpu_tensor       *target_q,
        ds4_gpu_tensor       *target_k,
        uint32_t              fail_after_layer) {
    const uint32_t n_tokens = 1u;
    const uint32_t n_head = 8u;
    const uint32_t target_n_head = 72u;
    const uint32_t head_dim = 128u;
    const uint64_t support_row_bytes =
        (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t target_q_bytes =
        (uint64_t)n_tokens * target_n_head * head_dim * sizeof(float);
    const uint64_t target_k_bytes = support_row_bytes;
    const uint64_t cache_bytes =
        (uint64_t)cache_cap * n_head * head_dim * sizeof(uint16_t);
    if (!g || !g->cur || !g->next || !source || !model_map ||
        !key_cache || !value_cache || !target_q || !target_k ||
        cache_cap == 0u ||
        !dflash_graph_commands_active() ||
        ds4_gpu_tensor_bytes(source) < support_row_bytes ||
        ds4_gpu_tensor_bytes(g->cur) < support_row_bytes ||
        ds4_gpu_tensor_bytes(g->next) < support_row_bytes ||
        ds4_gpu_tensor_bytes(key_cache) < cache_bytes ||
        ds4_gpu_tensor_bytes(value_cache) < cache_bytes ||
        ds4_gpu_tensor_bytes(target_q) < target_q_bytes ||
        ds4_gpu_tensor_bytes(target_k) < target_k_bytes) {
        return false;
    }
    ds4_gpu_tensor *saved_cur = g->cur;
    ds4_gpu_tensor *saved_next = g->next;
    const uint32_t rope_pos0 = 1u;
    for (uint32_t il = 0; il < 2u; il++) {
        if (il == fail_after_layer) {
            dflash_graph_restore_cursors(g, saved_cur, saved_next);
            return false;
        }
        const uint32_t pos0 = il + 1u;
        if (!ds4_gpu_tensor_copy(
                g->cur, 0, source, 0, support_row_bytes) ||
            !ds4_gpu_laguna_head_rms_norm_rope_support_tensor(
                g->cur,
                model_map,
                model_size,
                k_weight_offset,
                n_tokens,
                n_head,
                head_dim,
                head_dim,
                rope_pos0,
                262144u,
                500000.0f,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f,
                1e-6f) ||
            !ds4_gpu_laguna_qk_head_rms_norm_rope_tensor(
                target_q,
                target_k,
                model_map,
                model_size,
                0u,
                k_weight_offset,
                n_tokens,
                target_n_head,
                n_head,
                head_dim,
                head_dim,
                rope_pos0,
                262144u,
                /* The target atlas family is the code-defined SWA contract;
                 * DFlash support uses the separate 500000-family below. */
                10000.0f,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f,
                1e-6f) ||
            !ds4_gpu_dflash_commit_kv_tensor(
                key_cache,
                value_cache,
                g->cur,
                g->cur,
                pos0,
                n_tokens,
                cache_cap,
                n_head,
                head_dim)) {
            dflash_graph_restore_cursors(g, saved_cur, saved_next);
            return false;
        }
        ds4_gpu_tensor *tmp = g->cur;
        g->cur = g->next;
        g->next = tmp;
    }
    return true;
}

bool ds4_test_laguna_dflash_command_ownership(void) {
    const char *atlas_env_name = "DS4_METAL_LAGUNA_ROPE_ATLAS";
    const char *simd_env_name = "DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32";
    const char *saved_atlas_value = getenv(atlas_env_name);
    const char *saved_simd_value = getenv(simd_env_name);
    char *saved_atlas = NULL;
    char *saved_simd = NULL;
    if (saved_atlas_value) {
        saved_atlas = ds4_strdup(saved_atlas_value);
        if (!saved_atlas) return false;
    }
    if (saved_simd_value) {
        saved_simd = ds4_strdup(saved_simd_value);
        if (!saved_simd) {
            free(saved_atlas);
            return false;
        }
    }

    bool ok = false;
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = *lgn_model_shape();
    void *model_raw = NULL;
    ds4_gpu_tensor *source = NULL;
    ds4_gpu_tensor *cache = NULL;
    ds4_gpu_tensor *next = NULL;
    ds4_gpu_tensor *key_cache = NULL;
    ds4_gpu_tensor *value_cache = NULL;
    ds4_gpu_tensor *target_q = NULL;
    ds4_gpu_tensor *target_k = NULL;
    ds4_dflash_gpu_graph graph;
    memset(&graph, 0, sizeof(graph));
    enum {
        TEST_VALUES = 8 * 128,
        TARGET_Q_VALUES = 72 * 128,
        TEST_CACHE_CAP = 4,
        TEST_CACHE_VALUES = TEST_CACHE_CAP * TEST_VALUES,
    };
    float source_host[TEST_VALUES];
    float poison_host[TEST_VALUES];
    float readback[TEST_VALUES];
    float target_q_host[TARGET_Q_VALUES];
    uint16_t key_poison[TEST_CACHE_VALUES];
    uint16_t value_poison[TEST_CACHE_VALUES];
    uint16_t key_readback[TEST_CACHE_VALUES];
    uint16_t value_readback[TEST_CACHE_VALUES];
    for (size_t i = 0; i < TEST_VALUES; i++) {
        source_host[i] = 0.25f + (float)(i % 31u) / 37.0f;
    }
    memset(poison_host, 0xa5, sizeof(poison_host));
    for (size_t i = 0; i < TARGET_Q_VALUES; i++) {
        target_q_host[i] = -0.5f + (float)(i % 29u) / 41.0f;
    }
    for (size_t i = 0; i < TEST_CACHE_VALUES; i++) {
        key_poison[i] = (uint16_t)(0x3100u + (i % 251u));
        value_poison[i] = (uint16_t)(0x5200u + (i % 241u));
    }

    if (setenv(atlas_env_name, "1", 1) != 0 ||
        setenv(simd_env_name, "0", 1) != 0 ||
        ds4_gpu_laguna_rope_atlas_plan_reset_for_test() != 1 ||
        ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test() != 1 ||
        ds4_gpu_laguna_qk_head_norm_rope_simd32_preflight(
            48u, 8u, 128u, 64u) != 0 ||
        ds4_gpu_laguna_rope_atlas_preflight(
            72u, 8u, 128u, 128u) != 1) {
        goto cleanup;
    }

    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t q_weight_bytes =
        (uint64_t)TARGET_Q_VALUES * sizeof(float);
    const uint64_t k_weight_offset =
        (q_weight_bytes + page - 1u) / page * page;
    const uint64_t k_weight_bytes = (uint64_t)TEST_VALUES * sizeof(float);
    const uint64_t model_size = k_weight_offset + k_weight_bytes;
    if (posix_memalign(&model_raw, (size_t)page, (size_t)model_size) != 0 ||
        !model_raw) {
        goto cleanup;
    }
    memset(model_raw, 0, (size_t)model_size);
    float *q_weight = (float *)model_raw;
    float *k_weight = (float *)((uint8_t *)model_raw + k_weight_offset);
    for (size_t i = 0; i < TARGET_Q_VALUES; i++) q_weight[i] = 1.0f;
    for (size_t i = 0; i < TEST_VALUES; i++) k_weight[i] = 1.0f;
    if (!ds4_gpu_set_model_map(model_raw, model_size)) goto cleanup;

    const uint64_t tensor_bytes = sizeof(source_host);
    const uint64_t cache_bytes = sizeof(key_poison);
    const uint64_t target_q_bytes = sizeof(target_q_host);
    source = ds4_gpu_tensor_alloc(tensor_bytes);
    cache = ds4_gpu_tensor_alloc(tensor_bytes);
    next = ds4_gpu_tensor_alloc(tensor_bytes);
    key_cache = ds4_gpu_tensor_alloc(cache_bytes);
    value_cache = ds4_gpu_tensor_alloc(cache_bytes);
    target_q = ds4_gpu_tensor_alloc(target_q_bytes);
    target_k = ds4_gpu_tensor_alloc(tensor_bytes);
    if (!source || !cache || !next || !key_cache || !value_cache ||
        !target_q || !target_k ||
        !ds4_gpu_tensor_write(source, 0, source_host, tensor_bytes) ||
        !ds4_gpu_tensor_write(cache, 0, poison_host, tensor_bytes) ||
        !ds4_gpu_tensor_write(next, 0, poison_host, tensor_bytes) ||
        !ds4_gpu_tensor_write(key_cache, 0, key_poison, cache_bytes) ||
        !ds4_gpu_tensor_write(value_cache, 0, value_poison, cache_bytes) ||
        !ds4_gpu_tensor_write(target_q, 0, target_q_host, target_q_bytes) ||
        !ds4_gpu_tensor_write(target_k, 0, source_host, tensor_bytes)) {
        goto cleanup;
    }
    graph.cur = cache;
    graph.next = next;

    /* A draft recorder cannot manufacture its own transaction. */
    if (dflash_graph_commands_active() ||
        !dflash_graph_test_spec_snapshot_restore() ||
        dflash_graph_test_record_layers(
            &graph, source, model_raw, model_size, k_weight_offset,
            key_cache, value_cache, TEST_CACHE_CAP, target_q, target_k, 0u)) {
        goto cleanup;
    }

    const uint64_t target_generated_before =
        ds4_gpu_laguna_rope_atlas_completed_generated_count();
    const uint64_t target_consumed_before =
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count();
    const uint64_t target_family0_before =
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u);
    const uint64_t target_family1_before =
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u);
    const uint64_t support_generated_before =
        ds4_gpu_laguna_rope_support_atlas_completed_generated_count();
    const uint64_t support_consumed_before =
        ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count();
    if (ds4_gpu_begin_commands() != 1 ||
        dflash_graph_test_record_layers(
            &graph, source, model_raw, model_size, k_weight_offset,
            key_cache, value_cache, TEST_CACHE_CAP, target_q, target_k, 1u) ||
        graph.cur != cache || graph.next != next ||
        ds4_gpu_discard_commands() != 1 ||
        ds4_gpu_commands_active() != 0 ||
        ds4_gpu_tensor_read(cache, 0, readback, tensor_bytes) == 0 ||
        memcmp(readback, poison_host, tensor_bytes) != 0 ||
        ds4_gpu_tensor_read(next, 0, readback, tensor_bytes) == 0 ||
        memcmp(readback, poison_host, tensor_bytes) != 0 ||
        ds4_gpu_tensor_read(key_cache, 0, key_readback, cache_bytes) == 0 ||
        memcmp(key_readback, key_poison, cache_bytes) != 0 ||
        ds4_gpu_tensor_read(value_cache, 0, value_readback, cache_bytes) == 0 ||
        memcmp(value_readback, value_poison, cache_bytes) != 0 ||
        ds4_gpu_laguna_rope_atlas_completed_generated_count() !=
            target_generated_before ||
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() !=
            target_consumed_before ||
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u) !=
            target_family0_before ||
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u) !=
            target_family1_before ||
        ds4_gpu_laguna_rope_support_atlas_completed_generated_count() !=
            support_generated_before ||
        ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count() !=
            support_consumed_before) {
        goto cleanup;
    }

    if (!ds4_gpu_tensor_write(cache, 0, poison_host, tensor_bytes) ||
        !ds4_gpu_tensor_write(next, 0, poison_host, tensor_bytes) ||
        !ds4_gpu_tensor_write(key_cache, 0, key_poison, cache_bytes) ||
        !ds4_gpu_tensor_write(value_cache, 0, value_poison, cache_bytes) ||
        ds4_gpu_begin_commands() != 1 ||
        !dflash_graph_test_record_layers(
            &graph, source, model_raw, model_size, k_weight_offset,
            key_cache, value_cache, TEST_CACHE_CAP, target_q, target_k,
            UINT32_MAX) ||
        !dflash_graph_commands_active() ||
        ds4_gpu_flush_commands() != 1 ||
        !dflash_graph_commands_active() ||
        ds4_gpu_wait_submitted_commands() != 1 ||
        ds4_gpu_discard_commands() != 1 ||
        ds4_gpu_commands_active() != 0 ||
        graph.cur != cache || graph.next != next ||
        ds4_gpu_tensor_read(cache, 0, readback, tensor_bytes) == 0 ||
        memcmp(readback, poison_host, tensor_bytes) == 0) {
        goto cleanup;
    }
    float next_readback[TEST_VALUES];
    if (ds4_gpu_tensor_read(next, 0, next_readback, tensor_bytes) == 0 ||
        memcmp(next_readback, poison_host, tensor_bytes) == 0 ||
        ds4_gpu_tensor_read(key_cache, 0, key_readback, cache_bytes) == 0 ||
        ds4_gpu_tensor_read(value_cache, 0, value_readback, cache_bytes) == 0) {
        goto cleanup;
    }
    for (uint32_t row = 0; row < TEST_CACHE_CAP; row++) {
        const size_t row_offset = (size_t)row * TEST_VALUES;
        const bool intended = row == 1u || row == 2u;
        const bool key_changed = memcmp(
            key_readback + row_offset, key_poison + row_offset,
            TEST_VALUES * sizeof(uint16_t)) != 0;
        const bool value_changed = memcmp(
            value_readback + row_offset, value_poison + row_offset,
            TEST_VALUES * sizeof(uint16_t)) != 0;
        if (intended != key_changed || intended != value_changed) goto cleanup;
    }
    if (ds4_gpu_laguna_rope_atlas_completed_generated_count() !=
            target_generated_before + 1u ||
        ds4_gpu_laguna_rope_atlas_completed_consumed_dispatch_count() !=
            target_consumed_before + 2u ||
        ds4_gpu_laguna_rope_atlas_completed_family_count(0u) !=
            target_family0_before ||
        ds4_gpu_laguna_rope_atlas_completed_family_count(1u) !=
            target_family1_before + 2u ||
        ds4_gpu_laguna_rope_support_atlas_completed_generated_count() !=
            support_generated_before + 1u ||
        ds4_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count() !=
            support_consumed_before + 2u) {
        goto cleanup;
    }
    ok = true;

cleanup:
    if (ds4_gpu_commands_active()) (void)ds4_gpu_discard_commands();
    (void)ds4_gpu_wait_submitted_commands();
    (void)ds4_gpu_laguna_rope_atlas_plan_reset_for_test();
    (void)ds4_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test();
    ds4_gpu_tensor_free(target_k);
    ds4_gpu_tensor_free(target_q);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(next);
    ds4_gpu_tensor_free(cache);
    ds4_gpu_tensor_free(source);
    free(model_raw);
    if (saved_atlas) {
        (void)setenv(atlas_env_name, saved_atlas, 1);
        free(saved_atlas);
    } else {
        (void)unsetenv(atlas_env_name);
    }
    if (saved_simd) {
        (void)setenv(simd_env_name, saved_simd, 1);
        free(saved_simd);
    } else {
        (void)unsetenv(simd_env_name);
    }
    g_ds4_shape = saved_shape;
    return ok;
}
#endif
#endif
#endif
