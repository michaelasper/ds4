#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <inttypes.h>
#include <stddef.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <fcntl.h>
#include <limits.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sysctl.h>

#include "lgn2.h"
#include "lgn2_gpu.h"

/*
 * Objective-C Metal glue for the C engine.
 *
 * The C code owns model semantics and graph scheduling.  This file owns only
 * Metal objects: device/queue/library setup, mmap-backed weight views, command
 * batching, persistent tensors, scratch buffers, and thin wrappers around the
 * kernel files in the metal directory.  Keeping this boundary narrow makes the
 * inference path readable from C while still using Objective-C where Metal
 * requires it.
 */

enum {
    LGN2_METAL_TENSOR_Q4_0    = 2,
    LGN2_METAL_TENSOR_Q8_0    = 8,
    LGN2_METAL_TENSOR_Q2_K    = 10,
    LGN2_METAL_TENSOR_Q3_K    = 11,
    LGN2_METAL_TENSOR_Q4_K    = 12,
    LGN2_METAL_TENSOR_Q5_K    = 13,
    LGN2_METAL_TENSOR_Q6_K    = 14,
    LGN2_METAL_TENSOR_Q8_K    = 15,
    LGN2_METAL_TENSOR_IQ2_XXS = 16,
    LGN2_METAL_TENSOR_MXFP4   = 39,
};

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary> g_library;
static id<MTLCommandBuffer> g_batch_cb;
static id<MTLComputeCommandEncoder> g_batch_enc;
static BOOL g_batch_has_work;
/* Monotonic identity for the caller-owned command-batch session.  Flushes
 * may replace the underlying command buffer, but do not create a new public
 * batch; only a successful begin advances this epoch. */
static uint64_t g_command_batch_epoch;
/* The screen certificate is valid only for the library's actual compile
 * mode, not merely for the current environment (which callers can change
 * after lgn2_gpu_init has already compiled the Metal source). */
static int g_metal_math_safe;
static NSMutableArray<id<MTLCommandBuffer>> *g_pending_cbs;
typedef struct {
    uint64_t qk_simd32;
    uint64_t target_generated;
    uint32_t target_generated_tokens;
    uint32_t target_generated_pos0;
    uint64_t target_consumed;
    uint64_t target_family[2];
    uint64_t support_generated;
    uint32_t support_generated_tokens;
    uint32_t support_generated_pos0;
    uint64_t support_consumed;
} lgn2_gpu_laguna_atlas_cb_evidence;
static NSMutableArray<NSData *> *g_pending_laguna_atlas_evidence;
#ifdef LGN2_TEST_HOOKS
/* One grouped-MoE completion record per committed batch command buffer.  It
 * stays paired with g_pending_cbs until that exact buffer has been waited. */
static NSMutableArray<NSNumber *> *g_pending_glm_grouped_moe_evidence;
#endif
static lgn2_gpu_laguna_atlas_cb_evidence g_batch_laguna_atlas_evidence;
static lgn2_gpu_laguna_atlas_cb_evidence g_owned_laguna_atlas_evidence;
static id<MTLComputePipelineState> g_get_rows_f32_pipeline;
static id<MTLComputePipelineState> g_get_rows_i32_pipeline;
static id<MTLComputePipelineState> g_get_rows_q8_0_pipeline;
static id<MTLComputePipelineState> g_get_rows_q4_0_pipeline;
static id<MTLComputePipelineState> g_get_rows_q4_K_pipeline;
static id<MTLComputePipelineState> g_cpy_f32_f32_pipeline;
static id<MTLComputePipelineState> g_cpy_f32_f16_pipeline;
static id<MTLComputePipelineState> g_cpy_contig_f32_f16_pipeline;
static id<MTLComputePipelineState> g_swiglu_pipeline;
static id<MTLComputePipelineState> g_swiglu_flat_pipeline;
static id<MTLComputePipelineState> g_add_pipeline;
static id<MTLComputePipelineState> g_add2_pipeline;
static id<MTLComputePipelineState> g_add3_pipeline;
static id<MTLComputePipelineState> g_moe_sum8_pipeline;
static id<MTLComputePipelineState> g_moe_sum10_pipeline;
static id<MTLComputePipelineState> g_mul_pipeline;
static id<MTLComputePipelineState> g_rms_norm_pipeline;
static id<MTLComputePipelineState> g_add_rms_norm_pipeline;
/* Optional Laguna S2.1 decode fusion.  Keep this lazy so an older explicit
 * LGN2_METAL_NORM_SOURCE override remains usable until the opt-in path asks
 * for the new kernel. */
static id<MTLComputePipelineState> g_laguna_add3_rms_norm_pipeline;
static int g_laguna_add3_rms_norm_pipeline_checked;
static id<MTLComputePipelineState> g_unary_sigmoid_pipeline;
static id<MTLComputePipelineState> g_unary_silu_pipeline;
static id<MTLComputePipelineState> g_unary_softplus_pipeline;
static id<MTLComputePipelineState> g_unary_sqrt_pipeline;
static id<MTLComputePipelineState> g_unary_clamp_pipeline;
static id<MTLComputePipelineState> g_unary_scale_pipeline;
static id<MTLComputePipelineState> g_unary_fill_pipeline;
static id<MTLComputePipelineState> g_unary_fill_f16_pipeline;
static id<MTLComputePipelineState> g_bin_mul_scalar_pipeline;
static id<MTLComputePipelineState> g_bin_div_row_pipeline;
static id<MTLComputePipelineState> g_argsort_f32_i32_desc_pipeline;
static id<MTLComputePipelineState> g_argsort_merge_f32_i32_desc_pipeline;
static id<MTLComputePipelineState> g_sum_rows_f32_f32_pipeline;
/* Laguna's retained router selector still uses these source-level names. */
static id<MTLComputePipelineState> g_dsv4_softplus_sqrt_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_finalize_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_finalize_one_simd_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_finalize_weights_one_simd_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_transform_finalize_weights_one_simd_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_weights_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_weights_batch_pipeline;
static id<MTLComputePipelineState> g_glm_router_select_one_pipeline;
static id<MTLComputePipelineState> g_glm_router_select_one_simd_pipeline;
/* Optional fused Laguna decode router (logits + SIMD top-k in one
 * dispatch).  Lazy like the SIMD selector above it. */
static id<MTLComputePipelineState> g_laguna_router_decode_fused_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu2_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu4_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_pair_swiglu_r1_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q3_k_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q3_k_pair_swiglu_r1_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_down_r1_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q3_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q3_k_down_r1_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_down_r1_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_pair_swiglu_mapped_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q6_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_laguna_routed_shared_pair_pipeline;
static id<MTLComputePipelineState> g_laguna_routed_shared_q4_down_pipeline;
static id<MTLComputePipelineState> g_laguna_routed_shared_q6_down_pipeline;
static id<MTLComputePipelineState> g_laguna_head_norm_rope_pipeline;
static id<MTLComputePipelineState> g_laguna_qk_head_norm_rope_pipeline;
/* The 32-lane Laguna Q/K retile is deliberately lazy.  Keeping it out of
 * init-time required PSOs preserves env-off compatibility with older
 * LGN2_METAL_LAGUNA_SOURCE overrides; an env-on preflight fails clearly if the
 * override predates this contract. */
static id<MTLComputePipelineState> g_laguna_qk_head_norm_rope_simd32_pipeline;
static int g_laguna_qk_head_norm_rope_simd32_pipeline_checked;
static int g_laguna_qk_head_norm_rope_simd32_invalid_env_reported;
static uint64_t g_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count;
static uint64_t g_laguna_qk_head_norm_rope_simd32_completed_dispatch_count;
/* -2 means the graph has not selected a Q/K plan yet.  Once selected,
 * consumers never consult getenv for the selector or its trace flag. */
static int g_laguna_qk_head_norm_rope_simd32_plan_mode = -2;
static int g_laguna_qk_head_norm_rope_simd32_trace_mode = -1;
/* Strict Laguna RoPE angle atlas.  These PSOs stay lazy so an explicit source
 * override remains usable while the optional experiment is off. */
static id<MTLComputePipelineState> g_laguna_rope_atlas_pipeline;
static id<MTLComputePipelineState> g_laguna_rope_support_atlas_pipeline;
static id<MTLComputePipelineState> g_laguna_head_norm_rope_atlas_pipeline;
static id<MTLComputePipelineState> g_laguna_qk_head_norm_rope_atlas_pipeline;
static id<MTLComputePipelineState> g_laguna_qk_head_norm_rope_simd32_atlas_pipeline;
static int g_laguna_rope_atlas_pipeline_checked;
static int g_laguna_rope_atlas_invalid_env_reported;
/* Cached with the immutable atlas graph plan.  Consumers must not perform a
 * getenv per layer/token merely to decide whether optional tracing is on. */
static int g_laguna_rope_atlas_trace_mode = -1;
static id<MTLBuffer> g_laguna_rope_atlas_buffer;
static NSUInteger g_laguna_rope_atlas_buffer_bytes;
static id<MTLBuffer> g_laguna_rope_support_atlas_buffer;
static NSUInteger g_laguna_rope_support_atlas_buffer_bytes;
static uint64_t g_laguna_rope_atlas_valid_epoch;
static uint32_t g_laguna_rope_atlas_valid_tokens;
static uint32_t g_laguna_rope_atlas_valid_pos0;
static int g_laguna_rope_atlas_valid;
static int g_laguna_rope_atlas_valid_completed;
static uint64_t g_laguna_rope_support_atlas_valid_epoch;
static uint32_t g_laguna_rope_support_atlas_valid_tokens;
static uint32_t g_laguna_rope_support_atlas_valid_pos0;
static int g_laguna_rope_support_atlas_valid;
static int g_laguna_rope_support_atlas_valid_completed;
static uint64_t g_laguna_rope_atlas_encoded_dispatch_count;
static uint64_t g_laguna_rope_atlas_consumed_dispatch_count;
static uint64_t g_laguna_rope_atlas_consumed_family_count[2];
static uint64_t g_laguna_rope_support_atlas_encoded_dispatch_count;
static uint64_t g_laguna_rope_support_atlas_consumed_dispatch_count;
static uint64_t g_laguna_rope_atlas_completed_generated_count;
static uint64_t g_laguna_rope_atlas_completed_consumed_dispatch_count;
static uint64_t g_laguna_rope_atlas_completed_consumed_family_count[2];
static uint64_t g_laguna_rope_support_atlas_completed_generated_count;
static uint64_t g_laguna_rope_support_atlas_completed_consumed_dispatch_count;
/* -2 means no graph/preflight plan has been selected yet.  Once selected,
 * route kernels never consult getenv; this is also the disabled fast path. */
static int g_laguna_rope_atlas_plan_mode = -2;
static id<MTLComputePipelineState> g_laguna_store_kv_pipeline;
static id<MTLComputePipelineState> g_laguna_attention_pipeline;
static id<MTLComputePipelineState> g_laguna_stage_kv_pipeline;
static id<MTLComputePipelineState> g_laguna_prefill_attention_pipeline;
static id<MTLComputePipelineState> g_laguna_commit_kv_pipeline;
static id<MTLComputePipelineState> g_laguna_q6_k_matmul_pipeline;
static id<MTLComputePipelineState> g_laguna_argmax_f32_pipeline;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipeline_cache;

static NSMutableDictionary<NSString *, id<MTLBuffer>> *g_model_buffer_cache;
static id g_model_residency_set;

static id<MTLBuffer> g_flash_attn_zero_mask_buffer;
static id<MTLBuffer> g_flash_attn_pad_buffer;
static id<MTLBuffer> g_flash_attn_tmp_buffer;
static id<MTLBuffer> g_embed_rows_buffer;
static id<MTLBuffer> g_router_selection_buffer;
static id<MTLBuffer> g_router_weight_sum_buffer;
static id<MTLBuffer> g_glm_router_simd_topk_stats_buffer;
static int g_glm_router_simd_topk_reported;
static uint64_t g_glm_router_simd_topk_encoded_rows;
static uint64_t g_glm_router_simd_topk_encoded_dispatches;
static id<MTLBuffer> g_indexer_topk_buffer;
static id<MTLBuffer> g_moe_gate_scratch_buffer;
static id<MTLBuffer> g_moe_down_scratch_buffer;
static id<MTLBuffer> g_moe_id_map_buffer;
static const void *g_model_map_ptr;
static uint64_t g_model_map_size;
static uint64_t g_model_mapped_offset;
static uint64_t g_model_mapped_size;
static uint64_t g_model_mapped_max_tensor_bytes;
static uint64_t g_tensor_alloc_live_bytes;
static uint64_t g_tensor_alloc_peak_bytes;
static pthread_mutex_t g_tensor_mu = PTHREAD_MUTEX_INITIALIZER;
static uintptr_t *g_tensor_live_slots;
static size_t g_tensor_live_cap;
static size_t g_tensor_live_count;
static size_t g_tensor_live_tombs;
static uint64_t g_model_buffer_cache_bytes;
static uint64_t g_model_buffer_cache_evictions;
static int g_model_buffer_cache_over_limit;
static int g_model_residency_added_to_queue;
static int g_metal4_runtime_available;
static int g_metal4_family_supported;
static int g_metal4_queue_supported;
static int g_metal4_m5_neural_accelerators_hint;
static int g_metal4_tensor_api_enabled;
static int g_metal4_tensor_api_compile_supported;
static char g_metal_device_name[128];
static int lgn2_gpu_model_map_log_enabled(void);
static NSUInteger g_flash_attn_zero_mask_bytes;
static NSUInteger g_flash_attn_pad_bytes;
static NSUInteger g_flash_attn_tmp_bytes;
static NSUInteger g_embed_rows_bytes;
static NSUInteger g_router_selection_bytes;
static NSUInteger g_router_weight_sum_bytes;
static NSUInteger g_indexer_topk_bytes;
static NSUInteger g_moe_gate_scratch_bytes;
static NSUInteger g_moe_down_scratch_bytes;
static NSUInteger g_moe_id_map_bytes;
static int g_initialized;
#ifdef LGN2_TEST_HOOKS
/* -1 means no library has been checked in this lifecycle, 0 means the
 * current source lacked the exact ABI fingerprint marker, and 1 means the
 * marker was found.  This is test-only evidence; production dispatch never
 * consults it. */
static int g_metal_moe_abi_sentinel_status = -1;

int lgn2_gpu_test_moe_abi_sentinel_status(void) {
    return g_metal_moe_abi_sentinel_status;
}

typedef enum {
    LGN2_GPU_TEST_INIT_FAIL_NONE = 0,
    LGN2_GPU_TEST_INIT_FAIL_DEVICE,
    LGN2_GPU_TEST_INIT_FAIL_QUEUE,
    LGN2_GPU_TEST_INIT_FAIL_BOOKKEEPING,
    LGN2_GPU_TEST_INIT_FAIL_FIRST_PIPELINE,
} lgn2_gpu_test_init_failpoint;

static lgn2_gpu_test_init_failpoint g_test_init_failpoint;
static int g_test_synchronize_fail_once;
/* The wait-failure hook is consumed only after at least one submitted
 * command buffer has really completed.  This lets tests exercise the
 * reported-failure/quarantine path without fabricating a pre-submit error. */
static int g_test_wait_submitted_fail_once;

static int lgn2_gpu_test_init_should_fail(
        lgn2_gpu_test_init_failpoint failpoint) {
    if (g_test_init_failpoint != failpoint) return 0;
    /* One-shot by construction so the cleanup path itself and the retry are
     * never influenced by the injected failure. */
    g_test_init_failpoint = LGN2_GPU_TEST_INIT_FAIL_NONE;
    return 1;
}

static int lgn2_gpu_test_synchronize_result(int result) {
    if (!g_test_synchronize_fail_once) return result;
    g_test_synchronize_fail_once = 0;
    return 0;
}
#endif
static int g_quality_mode;
static int g_tensor_matmul_suppressed;
static int g_mpp_invalid_env_reported;
/* Runtime selectors are captured once for each Metal initialization
 * lifecycle.  Per-dispatch getenv reparsing would let a mid-graph env change
 * select arithmetic that no longer matches the stock route. */
static uint32_t g_q8_mv_ext_max_tokens = 16u;
static uint32_t g_glm_grouped_moe_min_tokens = 96u;
static int g_direct_kv_prefill_mode;
/* Laguna SWA selectors are lifecycle state.  Capturing them once keeps an
 * A/B environment edit from changing arithmetic halfway through a graph. */
static int g_laguna_swa_gqa9_mode;
static int g_laguna_swa_gqa3_mode;
static int g_laguna_staged_swa_mode;
static int g_laguna_swa_selectors_snapshot_valid;

/* Test-only route evidence.  Counters stay cold unless a focused test arms
 * them, so normal inference pays no atomic/locking cost. */
#ifdef LGN2_TEST_HOOKS
static int g_laguna_test_route_hooks;
static uint64_t g_laguna_test_direct_kv_count;
static uint64_t g_laguna_test_wrap_kv_count;
static uint64_t g_laguna_test_fused_q8_count;
static uint64_t g_laguna_test_stock_q8_count;
static uint64_t g_laguna_test_fused_q8_bco_false_count;
static uint64_t g_laguna_test_fused_q8_bco_true_count;
/* Completion-scoped evidence for the opt-in Laguna decode-router path.  The
 * counters are deliberately test-only: production inference has no extra
 * synchronization or reporting work. */
static uint64_t g_laguna_router_fused_encoded_dispatches;
/* Dispatches recorded in the current command batch, an owned one-off
 * command buffer, or already-committed pending command buffers are tracked
 * separately so discard and asynchronous submit cannot be mistaken for
 * completion. */
static uint64_t g_laguna_router_fused_batch_dispatches;
static uint64_t g_laguna_router_fused_owned_dispatches;
static uint64_t g_laguna_router_fused_pending_dispatches;
static uint64_t g_laguna_router_fused_completed_dispatches;
enum {
    LGN2_LAGUNA_TEST_DECODE_ORDINARY = 0,
    LGN2_LAGUNA_TEST_DECODE_GQA3 = 1,
    LGN2_LAGUNA_TEST_DECODE_GQA9 = 2,
    LGN2_LAGUNA_TEST_DECODE_STAGED = 3,
    LGN2_LAGUNA_TEST_DECODE_GLOBAL_GROUPED = 4,
    LGN2_LAGUNA_TEST_DECODE_ROUTE_COUNT = 5,
};
static uint64_t g_laguna_test_decode_route_counts[
    LGN2_LAGUNA_TEST_DECODE_ROUTE_COUNT];
static uint64_t g_laguna_test_decode_route_batch[
    LGN2_LAGUNA_TEST_DECODE_ROUTE_COUNT];
static uint64_t g_laguna_test_decode_route_inflight[
    LGN2_LAGUNA_TEST_DECODE_ROUTE_COUNT];
/* Completion-scoped grouped-MoE evidence.  The focused model-independent
 * fixture resets these counters before its row-loop oracle and then asserts
 * exactly one completed grouped command in the 32-token leg. */
static uint64_t g_test_glm_grouped_moe_encoded_dispatches;
static uint64_t g_test_glm_grouped_moe_batch_dispatches;
static uint64_t g_test_glm_grouped_moe_owned_dispatches;
static uint64_t g_test_glm_grouped_moe_completed_dispatches;
/* Completion-scoped evidence for the exact-Q4 verifier route.  This route
 * deliberately bypasses grouped dispatch, so it has its own counters. */
static uint64_t g_test_glm_exact_q4_encoded_dispatches;
static uint64_t g_test_glm_exact_q4_completed_dispatches;
#endif
static double lgn2_gpu_gib(uint64_t bytes);

static uint64_t lgn2_gpu_system_memory_bytes(void) {
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) != 0) return 0;
    return len == sizeof(bytes) ? bytes : 0;
}

static void lgn2_gpu_print_device_summary(void) {
    const char *name = g_device.name ? [g_device.name UTF8String] : "unknown Metal device";
    uint64_t mem = lgn2_gpu_system_memory_bytes();
    if (mem) {
        double gib = (double)mem / 1024.0 / 1024.0 / 1024.0;
        fprintf(stderr, "lgn2: Metal device %s, %.2f GiB RAM\n", name, gib);
    } else {
        fprintf(stderr, "lgn2: Metal device %s\n", name);
    }
}

#define LGN2_METAL_MAX_MODEL_VIEWS 4096
/* Compatibility fallback for callers that cannot provide a parsed GGUF tensor
 * span. The normal LGN2 engine passes the exact maximum tensor byte size. */
#define LGN2_METAL_FALLBACK_MAX_TENSOR_BYTES (4ull * 1024ull * 1024ull * 1024ull)

typedef struct {
    __strong id<MTLBuffer> buffer;
    const void *model_map;
    uint64_t model_size;
    uint64_t model_offset;
    uint64_t bytes;
} lgn2_gpu_model_view;

static lgn2_gpu_model_view g_model_views[LGN2_METAL_MAX_MODEL_VIEWS];
static uint32_t g_model_view_count;


@interface LGN2MetalTensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, assign) uint64_t offset;
@property(nonatomic, assign) uint64_t bytes;
@property(nonatomic, assign) uint8_t owner;
@end

@implementation LGN2MetalTensor
@end

static LGN2MetalTensor *lgn2_gpu_tensor_obj(lgn2_gpu_tensor *tensor) {
    return (__bridge LGN2MetalTensor *)tensor;
}

static const LGN2MetalTensor *lgn2_gpu_tensor_const_obj(const lgn2_gpu_tensor *tensor) {
    return (__bridge const LGN2MetalTensor *)tensor;
}

/* C code owns lgn2_gpu_tensor handles as retained Objective-C objects.  Freeing
 * the same opaque handle twice would make the second __bridge_transfer release
 * an already-deallocated object, which macOS reports as malloc corruption.  The
 * live table lets free validate a handle before touching Objective-C state; the
 * same mutex also serializes the diagnostic allocation counters. */
static uint64_t lgn2_gpu_tensor_ptr_hash(uintptr_t ptr) {
    uint64_t x = (uint64_t)(ptr >> 4);
    x ^= x >> 33;
    x *= UINT64_C(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x *= UINT64_C(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    return x;
}

static int lgn2_gpu_tensor_live_resize_locked(size_t min_cap) {
    size_t new_cap = 1024;
    while (new_cap < min_cap) new_cap <<= 1;

    uintptr_t *new_slots = calloc(new_cap, sizeof(new_slots[0]));
    if (!new_slots) return 0;

    for (size_t i = 0; i < g_tensor_live_cap; i++) {
        const uintptr_t key = g_tensor_live_slots[i];
        if (key == 0 || key == UINTPTR_MAX) continue;

        size_t idx = (size_t)lgn2_gpu_tensor_ptr_hash(key) & (new_cap - 1);
        while (new_slots[idx] != 0) idx = (idx + 1) & (new_cap - 1);
        new_slots[idx] = key;
    }

    free(g_tensor_live_slots);
    g_tensor_live_slots = new_slots;
    g_tensor_live_cap = new_cap;
    g_tensor_live_tombs = 0;
    return 1;
}

static int lgn2_gpu_tensor_live_insert_locked(const void *ptr) {
    if (!ptr || (uintptr_t)ptr == UINTPTR_MAX) return 0;
    if ((g_tensor_live_count + g_tensor_live_tombs + 1) * 10 >=
        g_tensor_live_cap * 7)
    {
        const size_t min_cap = g_tensor_live_cap ? g_tensor_live_cap * 2 : 1024;
        if (!lgn2_gpu_tensor_live_resize_locked(min_cap)) return 0;
    }

    const uintptr_t key = (uintptr_t)ptr;
    size_t idx = (size_t)lgn2_gpu_tensor_ptr_hash(key) & (g_tensor_live_cap - 1);
    size_t tomb = (size_t)-1;
    for (;;) {
        const uintptr_t cur = g_tensor_live_slots[idx];
        if (cur == key) return 0;
        if (cur == UINTPTR_MAX) {
            if (tomb == (size_t)-1) tomb = idx;
        } else if (cur == 0) {
            if (tomb != (size_t)-1) {
                idx = tomb;
                g_tensor_live_tombs--;
            }
            g_tensor_live_slots[idx] = key;
            g_tensor_live_count++;
            return 1;
        }
        idx = (idx + 1) & (g_tensor_live_cap - 1);
    }
}

static int lgn2_gpu_tensor_live_remove_locked(const void *ptr) {
    if (!ptr || g_tensor_live_cap == 0) return 0;

    const uintptr_t key = (uintptr_t)ptr;
    size_t idx = (size_t)lgn2_gpu_tensor_ptr_hash(key) & (g_tensor_live_cap - 1);
    for (;;) {
        const uintptr_t cur = g_tensor_live_slots[idx];
        if (cur == 0) return 0;
        if (cur == key) {
            g_tensor_live_slots[idx] = UINTPTR_MAX;
            g_tensor_live_count--;
            g_tensor_live_tombs++;
            return 1;
        }
        idx = (idx + 1) & (g_tensor_live_cap - 1);
    }
}

static int lgn2_gpu_tensor_track_alloc_locked(
        const void *ptr,
        uint64_t bytes,
        uint64_t *live_snap,
        uint64_t *peak_snap)
{
    if (!lgn2_gpu_tensor_live_insert_locked(ptr)) return 0;

    g_tensor_alloc_live_bytes += bytes;
    if (g_tensor_alloc_live_bytes > g_tensor_alloc_peak_bytes) {
        g_tensor_alloc_peak_bytes = g_tensor_alloc_live_bytes;
    }
    if (live_snap) *live_snap = g_tensor_alloc_live_bytes;
    if (peak_snap) *peak_snap = g_tensor_alloc_peak_bytes;
    return 1;
}

static int lgn2_gpu_tensor_track_view_locked(const void *ptr) {
    return lgn2_gpu_tensor_live_insert_locked(ptr);
}

static int lgn2_gpu_tensor_prepare_free(
        lgn2_gpu_tensor *tensor,
        uint8_t *owner,
        uint64_t *bytes,
        uint64_t *live_snap,
        uint64_t *peak_snap)
{
    pthread_mutex_lock(&g_tensor_mu);
    if (!lgn2_gpu_tensor_live_remove_locked(tensor)) {
        pthread_mutex_unlock(&g_tensor_mu);
        fprintf(stderr,
                "lgn2: Metal tensor free ignored for unknown handle %p\n",
                (void *)tensor);
        return 0;
    }

    LGN2MetalTensor *obj = lgn2_gpu_tensor_obj(tensor);
    const uint8_t obj_owner = obj.owner;
    const uint64_t obj_bytes = obj.bytes;
    if (obj_owner) {
        if (obj_bytes <= g_tensor_alloc_live_bytes) {
            g_tensor_alloc_live_bytes -= obj_bytes;
        } else {
            g_tensor_alloc_live_bytes = 0;
        }
    }
    if (owner) *owner = obj_owner;
    if (bytes) *bytes = obj_bytes;
    if (live_snap) *live_snap = g_tensor_alloc_live_bytes;
    if (peak_snap) *peak_snap = g_tensor_alloc_peak_bytes;
    pthread_mutex_unlock(&g_tensor_mu);
    return 1;
}

static void lgn2_gpu_tensor_tracking_reset(void) {
    pthread_mutex_lock(&g_tensor_mu);
    if (g_tensor_live_count != 0) {
        fprintf(stderr,
                "lgn2: Metal cleanup discarded %zu live tensor handles\n",
                g_tensor_live_count);
    }
    free(g_tensor_live_slots);
    g_tensor_live_slots = NULL;
    g_tensor_live_cap = 0;
    g_tensor_live_count = 0;
    g_tensor_live_tombs = 0;
    g_tensor_alloc_live_bytes = 0;
    g_tensor_alloc_peak_bytes = 0;
    pthread_mutex_unlock(&g_tensor_mu);
}

static id<MTLBuffer> lgn2_gpu_tensor_buffer(const lgn2_gpu_tensor *tensor) {
    if (!tensor) return nil;
    const LGN2MetalTensor *obj = lgn2_gpu_tensor_const_obj(tensor);
    return obj.buffer;
}

static NSUInteger lgn2_gpu_tensor_offset(const lgn2_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const LGN2MetalTensor *obj = lgn2_gpu_tensor_const_obj(tensor);
    return (NSUInteger)obj.offset;
}

static id<MTLCommandBuffer> lgn2_gpu_new_command_buffer(void);
static lgn2_gpu_laguna_atlas_cb_evidence *
lgn2_gpu_laguna_atlas_current_cb_evidence(void) {
    return g_batch_cb ? &g_batch_laguna_atlas_evidence :
                        &g_owned_laguna_atlas_evidence;
}

static void lgn2_gpu_laguna_atlas_evidence_zero(
        lgn2_gpu_laguna_atlas_cb_evidence *e) {
    if (e) memset(e, 0, sizeof(*e));
}

static void lgn2_gpu_laguna_atlas_publish(
        const lgn2_gpu_laguna_atlas_cb_evidence *e) {
    if (!e) return;
    g_laguna_qk_head_norm_rope_simd32_completed_dispatch_count += e->qk_simd32;
    g_laguna_rope_atlas_completed_generated_count += e->target_generated;
    g_laguna_rope_atlas_completed_consumed_dispatch_count += e->target_consumed;
    g_laguna_rope_atlas_completed_consumed_family_count[0] += e->target_family[0];
    g_laguna_rope_atlas_completed_consumed_family_count[1] += e->target_family[1];
    g_laguna_rope_support_atlas_completed_generated_count += e->support_generated;
    g_laguna_rope_support_atlas_completed_consumed_dispatch_count += e->support_consumed;
    if (e->target_generated != 0 &&
        g_laguna_rope_atlas_valid &&
        g_laguna_rope_atlas_valid_tokens == e->target_generated_tokens &&
        g_laguna_rope_atlas_valid_pos0 == e->target_generated_pos0) {
        g_laguna_rope_atlas_valid_completed = 1;
    }
    if (e->support_generated != 0 &&
        g_laguna_rope_support_atlas_valid &&
        g_laguna_rope_support_atlas_valid_tokens ==
            e->support_generated_tokens &&
        g_laguna_rope_support_atlas_valid_pos0 ==
            e->support_generated_pos0) {
        g_laguna_rope_support_atlas_valid_completed = 1;
    }
}

static void lgn2_gpu_laguna_atlas_register_pending(
        id<MTLCommandBuffer> cb,
        const lgn2_gpu_laguna_atlas_cb_evidence *e) {
    if (!g_pending_laguna_atlas_evidence) return;
    [g_pending_laguna_atlas_evidence addObject:
        [NSData dataWithBytes:e length:sizeof(*e)]];
    [g_pending_cbs addObject:cb];
#ifdef LGN2_TEST_HOOKS
    [g_pending_glm_grouped_moe_evidence addObject:
        @(g_test_glm_grouped_moe_batch_dispatches)];
    g_test_glm_grouped_moe_batch_dispatches = 0;
#endif
}

static id<MTLCommandBuffer> lgn2_gpu_command_buffer(int *owned) {
    if (g_batch_cb) {
        *owned = 0;
        return g_batch_cb;
    }
    *owned = 1;
    id<MTLCommandBuffer> cb = lgn2_gpu_new_command_buffer();
    if (cb) {
        lgn2_gpu_laguna_atlas_evidence_zero(&g_owned_laguna_atlas_evidence);
    }
    return cb;
}

static id<MTLComputeCommandEncoder> lgn2_gpu_compute_encoder(id<MTLCommandBuffer> cb) {
    if (g_batch_cb && cb == g_batch_cb) {
        g_batch_has_work = YES;
        if (!g_batch_enc) {
            g_batch_enc = [cb computeCommandEncoder];
        }
        return g_batch_enc;
    }
    return [cb computeCommandEncoder];
}

static void lgn2_gpu_end_compute_encoder(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    if (!enc) return;
    if (g_batch_cb && cb == g_batch_cb && enc == g_batch_enc) return;
    [enc endEncoding];
}

static void lgn2_gpu_close_batch_encoder(void) {
    if (!g_batch_enc) return;
    [g_batch_enc endEncoding];
    g_batch_enc = nil;
}

static double g_gpu_busy_accum;
static uint64_t g_gpu_busy_cbs;

/* A failed command buffer can leave a cross-threadgroup arrival counter at an
 * arbitrary partial value.  Drop cached ownership instead of CPU-resetting
 * buffers that another in-flight command buffer might still reference; bound
 * resources remain retained by their command buffers/transient list. */
static void lgn2_gpu_invalidate_completion_counters(void) {
}

static int lgn2_gpu_wait_command_buffer(id<MTLCommandBuffer> cb, const char *label) {
    [cb waitUntilCompleted];
    if (getenv("LGN2_METAL_GPU_BUSY_PROFILE")) {
        const double busy = cb.GPUEndTime - cb.GPUStartTime;
        if (busy > 0) g_gpu_busy_accum += busy;
        if ((++g_gpu_busy_cbs % 64u) == 0u) {
            fprintf(stderr, "lgn2: gpu busy accum %.1f ms over %llu cbs\n",
                    g_gpu_busy_accum * 1000.0,
                    (unsigned long long)g_gpu_busy_cbs);
        }
    }
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "lgn2: Metal %s failed: %s\n",
                label, [[cb.error localizedDescription] UTF8String]);
        lgn2_gpu_invalidate_completion_counters();
        return 0;
    }
    return 1;
}

static id<MTLCommandBuffer> lgn2_gpu_new_command_buffer(void) {
    static int initialized;
    static int use_unretained;
    if (!initialized) {
        use_unretained = getenv("LGN2_METAL_UNRETAINED_COMMAND_BUFFERS") != NULL;
        initialized = 1;
    }
    if (use_unretained) {
        return [g_queue commandBufferWithUnretainedReferences];
    }
    return [g_queue commandBuffer];
}

static uint64_t lgn2_gpu_exact_view_cache_limit_bytes(void) {
    static int initialized;
    static uint64_t limit_bytes;
    if (initialized) return limit_bytes;

    const uint64_t mib = 1024ull * 1024ull;
    const uint64_t gib = 1024ull * mib;
    limit_bytes = 64ull * gib;

    const char *gib_env = getenv("LGN2_METAL_EXACT_VIEW_CACHE_GIB");
    if (gib_env && gib_env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(gib_env, &end, 10);
        if (end != gib_env && *end == '\0') {
            limit_bytes = v > UINT64_MAX / gib ? UINT64_MAX : (uint64_t)v * gib;
        }
    }

    const char *mib_env = getenv("LGN2_METAL_EXACT_VIEW_CACHE_MIB");
    if (mib_env && mib_env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(mib_env, &end, 10);
        if (end != mib_env && *end == '\0') {
            limit_bytes = v > UINT64_MAX / mib ? UINT64_MAX : (uint64_t)v * mib;
        }
    }

    initialized = 1;
    return limit_bytes;
}

static void lgn2_gpu_model_buffer_cache_note_insert(uint64_t bytes) {
    if (g_model_buffer_cache_bytes > UINT64_MAX - bytes) {
        g_model_buffer_cache_bytes = UINT64_MAX;
    } else {
        g_model_buffer_cache_bytes += bytes;
    }

    const uint64_t limit = lgn2_gpu_exact_view_cache_limit_bytes();
    if (limit != 0 && g_model_buffer_cache_bytes > limit) {
        g_model_buffer_cache_over_limit = 1;
    }
}

static void lgn2_gpu_model_buffer_cache_clear(const char *reason) {
    if (!g_model_buffer_cache) {
        g_model_buffer_cache_bytes = 0;
        g_model_buffer_cache_over_limit = 0;
        return;
    }

    const NSUInteger entries = [g_model_buffer_cache count];
    if (entries != 0) {
        if (getenv("LGN2_METAL_EXACT_VIEW_CACHE_PROFILE") != NULL) {
            fprintf(stderr,
                    "lgn2: Metal exact model view cache evict reason=%s entries=%lu bytes=%.2f GiB limit=%.2f GiB\n",
                    reason ? reason : "unknown",
                    (unsigned long)entries,
                    lgn2_gpu_gib(g_model_buffer_cache_bytes),
                    lgn2_gpu_gib(lgn2_gpu_exact_view_cache_limit_bytes()));
        }
        [g_model_buffer_cache removeAllObjects];
        g_model_buffer_cache_evictions++;
    }
    g_model_buffer_cache_bytes = 0;
    g_model_buffer_cache_over_limit = 0;
}

static void lgn2_gpu_model_buffer_cache_maybe_evict(const char *reason) {
    if (g_model_buffer_cache_over_limit) {
        lgn2_gpu_model_buffer_cache_clear(reason);
    }
}

#ifdef LGN2_TEST_HOOKS
static void lgn2_gpu_laguna_test_decode_route_batch_completed(int ok);
#endif

static int lgn2_gpu_wait_pending_command_buffers(const char *label) {
    int ok = 1;
#ifdef LGN2_TEST_HOOKS
    int injected_failure = 0;
    uint64_t grouped_completed = 0;
#endif
    const NSUInteger count = [g_pending_cbs count];
    for (NSUInteger i = 0; i < count; i++) {
        id<MTLCommandBuffer> pending = [g_pending_cbs objectAtIndex:i];
        const BOOL completed = lgn2_gpu_wait_command_buffer(pending, label);
        if (!completed) {
            ok = 0;
        }
#ifdef LGN2_TEST_HOOKS
        /* Consume this fault only after the device has completed real work.
         * Report failure for the boundary and suppress every evidence record
         * from this wait, so completion cannot promote a speculative path. */
        if (completed && !injected_failure &&
            g_test_wait_submitted_fail_once) {
            g_test_wait_submitted_fail_once = 0;
            injected_failure = 1;
            ok = 0;
        }
#endif
#ifdef LGN2_TEST_HOOKS
        const uint64_t grouped_evidence =
            i < [g_pending_glm_grouped_moe_evidence count]
                ? [[g_pending_glm_grouped_moe_evidence objectAtIndex:i]
                    unsignedLongLongValue]
                : 0;
        if (completed && !injected_failure) {
            grouped_completed += grouped_evidence;
        }
#endif
        if (completed
#ifdef LGN2_TEST_HOOKS
            && !injected_failure
#endif
            && i < [g_pending_laguna_atlas_evidence count]) {
            NSData *data = [g_pending_laguna_atlas_evidence objectAtIndex:i];
            if ([data length] == sizeof(lgn2_gpu_laguna_atlas_cb_evidence)) {
                lgn2_gpu_laguna_atlas_cb_evidence evidence;
                [data getBytes:&evidence length:sizeof(evidence)];
                lgn2_gpu_laguna_atlas_publish(&evidence);
            }
        }
    }
    [g_pending_cbs removeAllObjects];
    [g_pending_laguna_atlas_evidence removeAllObjects];
#ifdef LGN2_TEST_HOOKS
    [g_pending_glm_grouped_moe_evidence removeAllObjects];
#endif
#ifdef LGN2_TEST_HOOKS
    /* These command buffers were already committed by flush/submit.  Count
     * them only after their completion wait, and
     * drop the evidence on an error or timeout. */
    if (ok) {
        g_laguna_router_fused_completed_dispatches +=
            g_laguna_router_fused_pending_dispatches;
        g_test_glm_grouped_moe_completed_dispatches += grouped_completed;
    }
    g_laguna_router_fused_pending_dispatches = 0;
    lgn2_gpu_laguna_test_decode_route_batch_completed(ok);
#endif
    if (!ok) {
        g_laguna_rope_atlas_valid = 0;
        g_laguna_rope_atlas_valid_completed = 0;
        g_laguna_rope_support_atlas_valid = 0;
        g_laguna_rope_support_atlas_valid_completed = 0;
    }
    return ok;
}

static int lgn2_gpu_finish_command_buffer(id<MTLCommandBuffer> cb, int owned, const char *label) {
    if (!owned) return 1;

    [cb commit];
    int ok = lgn2_gpu_wait_pending_command_buffers(label);
    if (!lgn2_gpu_wait_command_buffer(cb, label)) {
        ok = 0;
        g_laguna_rope_atlas_valid = 0;
        g_laguna_rope_atlas_valid_completed = 0;
        g_laguna_rope_support_atlas_valid = 0;
        g_laguna_rope_support_atlas_valid_completed = 0;
        lgn2_gpu_laguna_atlas_evidence_zero(&g_owned_laguna_atlas_evidence);
    } else {
        lgn2_gpu_laguna_atlas_publish(&g_owned_laguna_atlas_evidence);
        lgn2_gpu_laguna_atlas_evidence_zero(&g_owned_laguna_atlas_evidence);
    }
    lgn2_gpu_model_buffer_cache_maybe_evict(label);
#ifdef LGN2_TEST_HOOKS
    /* A dispatch is evidence only after its owning command buffer has
     * completed.  Failed command buffers never report a successful fused
     * graph execution. */
    if (ok) g_laguna_router_fused_completed_dispatches +=
        g_laguna_router_fused_batch_dispatches +
        g_laguna_router_fused_owned_dispatches;
    g_laguna_router_fused_batch_dispatches = 0;
    g_laguna_router_fused_owned_dispatches = 0;
    if (ok) g_test_glm_grouped_moe_completed_dispatches +=
        g_test_glm_grouped_moe_batch_dispatches +
        g_test_glm_grouped_moe_owned_dispatches;
    g_test_glm_grouped_moe_batch_dispatches = 0;
    g_test_glm_grouped_moe_owned_dispatches = 0;
#endif
    return ok;
}

static int lgn2_gpu_device_name_contains(const char *needle);

static int lgn2_gpu_use_m5_private_scratch(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = lgn2_gpu_device_name_contains("M5");
        initialized = 1;
    }
    return enabled;
}

static int lgn2_gpu_scratch_needs_cpu_access(const char *label) {
    if (!label) return 0;
    return strstr(label, "mask") != NULL ||
           strcmp(label, "lgn2_attention_output_group_ids") == 0;
}

static MTLResourceOptions lgn2_gpu_model_resource_options(void) {
    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (getenv("LGN2_METAL_MODEL_UNTRACKED") != NULL) {
        options |= MTLResourceHazardTrackingModeUntracked;
    }
    return options;
}

static int lgn2_gpu_ensure_scratch_buffer(
        id<MTLBuffer> __strong *buffer,
        NSUInteger    *capacity,
        NSUInteger     bytes,
        const char    *label) {
    if (*buffer && *capacity >= bytes) return 1;
    if (bytes == 0) bytes = 1;
    if (bytes > NSUIntegerMax) return 0;

    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (lgn2_gpu_use_m5_private_scratch() &&
        !lgn2_gpu_scratch_needs_cpu_access(label)) {
        /*
         * M5 scratch buffers that only flow between Metal kernels do not need
         * CPU-visible shared storage. This reduces shared-memory traffic and
         * residency pressure for the long prefill scratch pools without
         * changing the public buffer lifetime model. Keep default hazard
         * tracking because the graph reuses these buffers across dependent
         * compute encoders.
         */
        options = MTLResourceStorageModePrivate;
    }

    *buffer = [g_device newBufferWithLength:bytes options:options];
    if (!*buffer && options != MTLResourceStorageModeShared) {
        *buffer = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    if (!*buffer) {
        fprintf(stderr, "lgn2: failed to allocate Metal scratch buffer %s (%llu bytes)\n",
                label, (unsigned long long)bytes);
        *capacity = 0;
        return 0;
    }
    (*buffer).label = [NSString stringWithUTF8String:label];
    *capacity = bytes;
    return 1;
}

static int lgn2_gpu_ensure_zero_attention_mask(NSUInteger bytes) {
    if (g_flash_attn_zero_mask_buffer &&
        g_flash_attn_zero_mask_bytes >= bytes) {
        return 1;
    }

    /* Grow in 8192-key chunks. Laguna contexts can exceed the old 8192-key
     * fixed allocation, while exact-size growth would reallocate every decode
     * token once that boundary is crossed. */
    const NSUInteger quantum = 8192u * sizeof(uint16_t);
    if (bytes > NSUIntegerMax - (quantum - 1u)) return 0;
    const NSUInteger capacity =
        ((bytes + quantum - 1u) / quantum) * quantum;
    if (!lgn2_gpu_ensure_scratch_buffer(&g_flash_attn_zero_mask_buffer,
                                         &g_flash_attn_zero_mask_bytes,
                                         capacity,
                                         "lgn2_flash_attn_zero_mask")) {
        return 0;
    }
    void *contents = [g_flash_attn_zero_mask_buffer contents];
    if (!contents) return 0;
    memset(contents, 0, g_flash_attn_zero_mask_bytes);
    return 1;
}

static uint64_t round_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1) & ~(align - 1);
}

static uint64_t lgn2_gpu_effective_model_max_tensor_bytes(uint64_t map_size, uint64_t max_tensor_bytes) {
    if (max_tensor_bytes != 0) return max_tensor_bytes;
    return map_size < LGN2_METAL_FALLBACK_MAX_TENSOR_BYTES ?
           map_size : LGN2_METAL_FALLBACK_MAX_TENSOR_BYTES;
}

static id<MTLComputePipelineState> lgn2_gpu_get_pipeline(const char *function_name);
static int lgn2_gpu_warm_model_views(void);
static double lgn2_gpu_gib(uint64_t bytes);

static double lgn2_gpu_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}


static int lgn2_gpu_progress_enabled(void) {
    return lgn2_log_is_tty(stderr);
}

static void lgn2_gpu_progress_begin(const char *what) {
    if (!lgn2_gpu_progress_enabled()) return;
    fprintf(stderr, "lgn2: %s...", what);
    fflush(stderr);
}

static void lgn2_gpu_progress_done(void) {
    if (!lgn2_gpu_progress_enabled()) return;
    fputs(" done\n", stderr);
    fflush(stderr);
}

static void lgn2_gpu_progress_failed(void) {
    if (!lgn2_gpu_progress_enabled()) return;
    fputs(" failed\n", stderr);
    fflush(stderr);
}

static void lgn2_gpu_model_views_clear(void) {
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        g_model_views[i].buffer = nil;
        g_model_views[i].model_map = NULL;
        g_model_views[i].model_size = 0;
        g_model_views[i].model_offset = 0;
        g_model_views[i].bytes = 0;
    }
    g_model_view_count = 0;
}

static void lgn2_gpu_model_residency_clear(void) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (g_model_residency_set) {
            if (g_model_residency_added_to_queue &&
                g_queue &&
                [g_queue respondsToSelector:@selector(removeResidencySet:)]) {
                [g_queue removeResidencySet:g_model_residency_set];
            }
            [g_model_residency_set endResidency];
            [g_model_residency_set removeAllAllocations];
            g_model_residency_set = nil;
        }
    }
#endif
    g_model_residency_added_to_queue = 0;
}

static int lgn2_gpu_model_residency_request_views(void) {
    if (g_model_view_count == 0 ||
        getenv("LGN2_METAL_NO_RESIDENCY") != NULL) {
        return 1;
    }

#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        /*
         * Register all model views as one residency set before inference. This
         * is a GPU residency/budgeting hint, not a request to fault the whole
         * 80+ GB file into memory. Its purpose is to make the driver see the
         * complete set of large shared allocations during setup instead of
         * discovering them lazily from the first measured graph command, where
         * VM validation and residency accounting would look like model compute.
         */
        MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"lgn2_model";
        desc.initialCapacity = g_model_view_count;

        NSError *error = nil;
        g_model_residency_set = [g_device newResidencySetWithDescriptor:desc error:&error];
        if (!g_model_residency_set) {
            fprintf(stderr, "lgn2: Metal model residency set creation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }

        for (uint32_t i = 0; i < g_model_view_count; i++) {
            [g_model_residency_set addAllocation:g_model_views[i].buffer];
        }
        [g_model_residency_set commit];
        [g_model_residency_set requestResidency];
        if (getenv("LGN2_METAL_DISABLE_QUEUE_RESIDENCY_SET") == NULL &&
            g_queue &&
            [g_queue respondsToSelector:@selector(addResidencySet:)]) {
            [g_queue addResidencySet:g_model_residency_set];
            g_model_residency_added_to_queue = 1;
        }
    }
#endif

    return 1;
}

static int lgn2_gpu_add_model_view_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    map_offset,
        uint64_t    map_size,
        uint64_t    max_tensor_bytes,
        bool        use_default_view_cap,
        uint64_t   *mapped_model_size_out) {
    const uint64_t page = (uint64_t)getpagesize();
    const uintptr_t model_addr = (uintptr_t)model_map;

    if ((model_addr & (uintptr_t)(page - 1)) != 0) {
        fprintf(stderr, "lgn2: Metal model mmap base is not page aligned\n");
        return 0;
    }
    if (map_offset > model_size || map_size > model_size - map_offset) {
        fprintf(stderr, "lgn2: Metal model mapped range is outside the GGUF mapping\n");
        return 0;
    }
    const uint64_t page_model_offset = map_offset & ~(page - 1);
    const uint64_t leading = map_offset - page_model_offset;
    if (map_size > UINT64_MAX - leading ||
        leading + map_size > UINT64_MAX - (page - 1))
    {
        fprintf(stderr, "lgn2: Metal model mapped range overflows page alignment\n");
        return 0;
    }
    const uint64_t mapped_model_size = round_up_u64(leading + map_size, page);
    uint64_t max_buffer = (uint64_t)[g_device maxBufferLength];
    max_buffer &= ~(page - 1);

    /*
     * Wrap only the tensor-data part of the GGUF file. Metadata is parsed by the
     * CPU and is never dereferenced by kernels, so exposing it to Metal only
     * grows the residency set and the VM range the driver must validate.
     *
     * Metal buffers have a device-specific maximum length, and this model is
     * larger than that maximum on the target machines. Creating one no-copy
     * buffer per tensor would avoid the length limit, but it would also move a
     * lot of VM-object creation and residency bookkeeping into graph setup. The
     * stable shape here is a tiny number of page-aligned views created once.
     *
     * Adjacent views intentionally overlap by more than the largest tensor, plus
     * one page for alignment. That invariant guarantees every tensor lies wholly
     * inside at least one view, so hot paths pass one buffer and one inner byte
     * offset. We never split a weight tensor across command encoders.
     */
    if (max_tensor_bytes > map_size) {
        fprintf(stderr, "lgn2: Metal model max tensor span is larger than a mapped tensor span\n");
        return 0;
    }
    if (max_tensor_bytes > UINT64_MAX - (page - 1)) {
        fprintf(stderr, "lgn2: Metal model max tensor span overflows page alignment\n");
        return 0;
    }
    const uint64_t max_tensor_rounded = round_up_u64(max_tensor_bytes, page);
    if (max_tensor_rounded > UINT64_MAX - page) {
        fprintf(stderr, "lgn2: Metal model view overlap overflows page slack\n");
        return 0;
    }
    const uint64_t overlap = max_tensor_rounded + page;
    if (max_buffer == 0 || max_buffer <= overlap) {
        fprintf(stderr,
                "lgn2: Metal maxBufferLength is too small for LGN2 model views "
                "(max tensor %.2f GiB, max buffer %.2f GiB)\n",
                lgn2_gpu_gib(max_tensor_bytes),
                lgn2_gpu_gib(max_buffer));
        return 0;
    }

    uint64_t view_limit = max_buffer;
    const char *view_limit_env = getenv("LGN2_METAL_MODEL_VIEW_MAX_GIB");
    if (view_limit_env && view_limit_env[0]) {
        char *end = NULL;
        unsigned long long gib = strtoull(view_limit_env, &end, 10);
        if (end != view_limit_env && gib > 0) {
            uint64_t env_limit = gib * 1024ull * 1024ull * 1024ull;
            env_limit &= ~(page - 1);
            if (env_limit > 0) view_limit = env_limit;
        }
    } else if (use_default_view_cap && mapped_model_size > max_buffer) {
        /*
         * Very large no-copy buffers can make Metal's VM validation dominate
         * startup or the first graph command on multi-hundred-GiB slices. Keep
         * ordinary contiguous model mappings unchanged, but let distributed
         * span maps use smaller overlapping views when a range already has to
         * be split.
         */
        const uint64_t default_limit = 128ull * 1024ull * 1024ull * 1024ull;
        if (view_limit > default_limit) view_limit = default_limit;
    }
    if (view_limit > max_buffer) view_limit = max_buffer;
    view_limit &= ~(page - 1);
    if (view_limit == 0 || view_limit <= overlap) {
        fprintf(stderr,
                "lgn2: Metal model view cap is too small for LGN2 model views "
                "(cap %.2f GiB, max tensor %.2f GiB)\n",
                lgn2_gpu_gib(view_limit),
                lgn2_gpu_gib(max_tensor_bytes));
        return 0;
    }

    const uint64_t step = view_limit - overlap;
    uint64_t off = 0;
    while (off < mapped_model_size) {
        if (g_model_view_count == LGN2_METAL_MAX_MODEL_VIEWS) {
            fprintf(stderr, "lgn2: Metal model needs more mapped views than expected\n");
            return 0;
        }

        uint64_t view_bytes = mapped_model_size - off;
        if (view_bytes > view_limit) view_bytes = view_limit;

        id<MTLBuffer> buffer = [g_device newBufferWithBytesNoCopy:(void *)(model_addr + page_model_offset + off)
                                                           length:(NSUInteger)view_bytes
                                                          options:lgn2_gpu_model_resource_options()
                                                      deallocator:nil];
        if (!buffer) {
            fprintf(stderr,
                    "lgn2: Metal could not wrap mmaped model view at %.2f GiB, size %.2f GiB\n",
                    (double)(page_model_offset + off) / (1024.0 * 1024.0 * 1024.0),
                    (double)view_bytes / (1024.0 * 1024.0 * 1024.0));
            return 0;
        }
        buffer.label = [NSString stringWithFormat:@"lgn2_model_view_%u", g_model_view_count];

        g_model_views[g_model_view_count].buffer = buffer;
        g_model_views[g_model_view_count].model_map = model_map;
        g_model_views[g_model_view_count].model_size = model_size;
        g_model_views[g_model_view_count].model_offset = page_model_offset + off;
        g_model_views[g_model_view_count].bytes = view_bytes;
        g_model_view_count++;

        if (off + view_bytes >= mapped_model_size) break;
        off += step;
    }

    if (mapped_model_size_out) *mapped_model_size_out += mapped_model_size;
    return 1;
}

static int lgn2_gpu_finish_model_views(
        double t0,
        uint64_t mapped_model_size,
        uint64_t display_offset) {
    const double t_mapped = lgn2_gpu_now_ms();
    const int request_residency = getenv("LGN2_METAL_NO_RESIDENCY") == NULL;
    if (request_residency) lgn2_gpu_progress_begin("requesting Metal residency (may take tens of seconds)");
    if (!lgn2_gpu_model_residency_request_views()) {
        if (request_residency) lgn2_gpu_progress_failed();
        return 0;
    }
    if (request_residency) lgn2_gpu_progress_done();
    const double t_resident = lgn2_gpu_now_ms();
    int warmed = 1;
    const double t_warm0 = lgn2_gpu_now_ms();
    const int warm_model_views = getenv("LGN2_METAL_NO_RESIDENCY") == NULL &&
                                 getenv("LGN2_METAL_NO_MODEL_WARMUP") == NULL;
    if (warm_model_views) {
        /*
         * The first GPU command touching no-copy mmap storage can pay command
         * queue setup, page-table validation, and shared-allocation residency
         * costs. Sample each model view here so timed graph execution starts
         * after that one-time work. The stride is intentionally coarse: this is
         * a validation touch over the VM ranges, not a full model prefetch. A
         * dense prefetch would create exactly the kind of memory pressure and
         * startup stalls this path is designed to avoid.
         */
        lgn2_gpu_progress_begin("warming Metal model views");
        warmed = lgn2_gpu_warm_model_views();
        if (warmed) lgn2_gpu_progress_done();
        else lgn2_gpu_progress_failed();
    }
    const double t_warm = lgn2_gpu_now_ms();
    if (lgn2_gpu_model_map_log_enabled()) {
        fprintf(stderr,
                "lgn2: Metal model views created in %.3f ms, residency requested in %.3f ms, warmup %.3f ms (mapped %.2f MiB from offset %.2f MiB)\n",
                t_mapped - t0,
                t_resident - t_mapped,
                t_warm - t_warm0,
                mapped_model_size / 1024.0 / 1024.0,
                display_offset / 1024.0 / 1024.0);
    }
    if (!warmed) return 0;
    return 1;
}

static int lgn2_gpu_map_model_views(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    map_offset,
        uint64_t    map_size,
        uint64_t    max_tensor_bytes) {
    const double t0 = lgn2_gpu_now_ms();
    uint64_t mapped_model_size = 0;
    if (!lgn2_gpu_add_model_view_range(model_map,
                                      model_size,
                                      map_offset,
                                      map_size,
                                      max_tensor_bytes,
                                      false,
                                      &mapped_model_size)) {
        return 0;
    }
    return lgn2_gpu_finish_model_views(t0, mapped_model_size, map_offset);
}

static id<MTLComputePipelineState> lgn2_gpu_get_mul_mm_pipeline(
        const char *function_name,
        bool        bc_inp,
        bool        bc_out) {
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d_bco=%d",
                     function_name, bc_inp ? 1 : 0, bc_out ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [constants setConstantValue:&bc_out type:MTLDataTypeBool atIndex:701];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "lgn2: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "lgn2: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> lgn2_gpu_get_mul_mm_id_pipeline(
        const char *function_name,
        bool        bc_inp) {
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d",
                     function_name, bc_inp ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "lgn2: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "lgn2: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> lgn2_gpu_get_pipeline(
        const char *function_name) {
    NSString *key = [NSString stringWithFormat:@"%s", function_name];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "lgn2: Metal %s function not found\n", function_name);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "lgn2: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState>
lgn2_gpu_laguna_add3_rms_norm_pipeline(void) {
    if (!g_laguna_add3_rms_norm_pipeline_checked) {
        g_laguna_add3_rms_norm_pipeline_checked = 1;
        g_laguna_add3_rms_norm_pipeline =
            lgn2_gpu_get_pipeline("kernel_add3_rms_norm_mul_f32_4");
    }
    return g_laguna_add3_rms_norm_pipeline;
}

static id<MTLComputePipelineState>
lgn2_gpu_laguna_qk_head_norm_rope_simd32_pipeline(void) {
    if (!g_laguna_qk_head_norm_rope_simd32_pipeline_checked) {
        g_laguna_qk_head_norm_rope_simd32_pipeline_checked = 1;
        g_laguna_qk_head_norm_rope_simd32_pipeline =
            lgn2_gpu_get_pipeline(
                "kernel_laguna_qk_head_rms_norm_rope_neox_simd32");
    }
    return g_laguna_qk_head_norm_rope_simd32_pipeline;
}

typedef struct {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_families;
    uint32_t atlas_stride;
    uint32_t n_rot[2];
    uint32_t n_ctx_orig[2];
    float    freq_base[2];
    float    freq_scale[2];
    float    ext_factor[2];
    float    attn_factor[2];
    float    beta_fast[2];
    float    beta_slow[2];
} lgn2_gpu_laguna_rope_atlas_args;

static int lgn2_gpu_laguna_qk_head_norm_rope_simd32_env_mode_impl(void);

static id<MTLComputePipelineState>
lgn2_gpu_laguna_rope_atlas_pipeline(void) {
    if (!g_laguna_rope_atlas_pipeline_checked) {
        g_laguna_rope_atlas_pipeline_checked = 1;
        g_laguna_rope_atlas_pipeline = lgn2_gpu_get_pipeline(
            "kernel_laguna_rope_atlas");
        g_laguna_rope_support_atlas_pipeline = lgn2_gpu_get_pipeline(
            "kernel_laguna_rope_support_atlas");
        g_laguna_head_norm_rope_atlas_pipeline = lgn2_gpu_get_pipeline(
            "kernel_laguna_head_rms_norm_rope_neox_atlas");
        g_laguna_qk_head_norm_rope_atlas_pipeline = lgn2_gpu_get_pipeline(
            "kernel_laguna_qk_head_rms_norm_rope_neox_atlas");
        g_laguna_qk_head_norm_rope_simd32_atlas_pipeline =
            lgn2_gpu_get_pipeline(
                "kernel_laguna_qk_head_rms_norm_rope_neox_simd32_atlas");
    }
    return g_laguna_rope_atlas_pipeline;
}

static int lgn2_gpu_laguna_rope_atlas_env_mode_impl(void) {
    const char *v = getenv("LGN2_METAL_LAGUNA_ROPE_ATLAS");
    if (!v || !v[0] || strcmp(v, "0") == 0) return 0;
    if (strcmp(v, "1") == 0) return 1;
    if (!g_laguna_rope_atlas_invalid_env_reported) {
        fprintf(stderr,
                "lgn2: invalid LGN2_METAL_LAGUNA_ROPE_ATLAS=%s; "
                "requested configuration is fatal (use unset, empty, 0, or 1)\n",
                v);
        g_laguna_rope_atlas_invalid_env_reported = 1;
    }
    return -1;
}

static int lgn2_gpu_laguna_rope_atlas_plan_mode(void) {
    return g_laguna_rope_atlas_plan_mode;
}

static int lgn2_gpu_laguna_rope_atlas_single_geometry_ok(
        uint32_t n_head, uint32_t head_dim, uint32_t n_rot) {
    return (n_head == 8u || n_head == 48u || n_head == 72u) &&
           head_dim == 128u && (n_rot == 64u || n_rot == 128u);
}

static int lgn2_gpu_laguna_rope_atlas_geometry_ok(
        uint32_t n_q_head, uint32_t n_k_head,
        uint32_t head_dim, uint32_t n_rot) {
    if (n_k_head == 0u) {
        return lgn2_gpu_laguna_rope_atlas_single_geometry_ok(
            n_q_head, head_dim, n_rot);
    }
    return (n_q_head == 48u || n_q_head == 72u) &&
           n_k_head == 8u && head_dim == 128u &&
           (n_rot == 64u || n_rot == 128u);
}

/* Return the fixed atlas plane for the exact Laguna model contract.  The
 * atlas is not a generic RoPE cache: accepting nearby parameters would make
 * a caller silently consume coefficients from the wrong family. */
static int lgn2_gpu_laguna_rope_atlas_family_for_args(
        uint32_t n_rot, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow) {
    if (n_rot == 64u && n_ctx_orig == 8192u &&
        freq_base == 500000.0f && freq_scale == (1.0f / 32.0f) &&
        ext_factor == 1.0f && attn_factor == 1.0f &&
        beta_fast == 32.0f && beta_slow == 1.0f) {
        return 0;
    }
    if (n_rot == 128u && n_ctx_orig == 262144u &&
        freq_base == 10000.0f && freq_scale == 1.0f &&
        ext_factor == 0.0f && attn_factor == 1.0f &&
        beta_fast == 0.0f && beta_slow == 0.0f) {
        return 1;
    }
    /* DFlash support has a deliberately independent one-family atlas.  Its
     * context/frequency contract is not interchangeable with target SWA. */
    if (n_rot == 128u && n_ctx_orig == 262144u &&
        freq_base == 500000.0f && freq_scale == 1.0f &&
        ext_factor == 0.0f && attn_factor == 1.0f &&
        beta_fast == 0.0f && beta_slow == 0.0f) {
        return 2;
    }
    return -1;
}

static int lgn2_gpu_laguna_rope_atlas_pipeline_usable(
        id<MTLComputePipelineState> pipeline,
        NSUInteger min_threads,
        NSUInteger required_tew,
        const char *label) {
    if (!pipeline) {
        fprintf(stderr,
                "lgn2: Laguna RoPE atlas requested but %s pipeline is unavailable; "
                "refusing legacy fallback\n",
                label);
        return 0;
    }
    if (required_tew != 0u && pipeline.threadExecutionWidth != required_tew) {
        fprintf(stderr,
                "lgn2: Laguna RoPE atlas %s pipeline has execution width %lu, "
                "requires %lu; requested configuration fails\n",
                label,
                (unsigned long)pipeline.threadExecutionWidth,
                (unsigned long)required_tew);
        return 0;
    }
    if (pipeline.maxTotalThreadsPerThreadgroup < min_threads) {
        fprintf(stderr,
                "lgn2: Laguna RoPE atlas %s pipeline supports %lu threads, "
                "needs at least %lu; requested configuration fails\n",
                label,
                (unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
                (unsigned long)min_threads);
        return 0;
    }
    return 1;
}

static int lgn2_gpu_laguna_rope_atlas_buffer_ensure(void);
static int lgn2_gpu_laguna_rope_support_atlas_buffer_ensure(void);
int lgn2_gpu_laguna_rope_atlas_trace_enabled(void);

int lgn2_gpu_laguna_rope_atlas_env_mode(void) {
    return lgn2_gpu_laguna_rope_atlas_env_mode_impl();
}

int lgn2_gpu_laguna_rope_atlas_plan_mode_cached(void) {
    return lgn2_gpu_laguna_rope_atlas_plan_mode();
}

static int lgn2_gpu_laguna_rope_atlas_target_reuse_ready_impl(
        uint32_t n_tokens, uint32_t pos0) {
    if (g_laguna_rope_atlas_plan_mode != 1 ||
        n_tokens == 0u || n_tokens > 16384u ||
        pos0 > UINT32_MAX - (n_tokens - 1u)) {
        return 0;
    }
    return g_laguna_rope_atlas_valid &&
           g_laguna_rope_atlas_valid_completed &&
           g_laguna_rope_atlas_valid_tokens == n_tokens &&
           g_laguna_rope_atlas_valid_pos0 == pos0;
}

int lgn2_gpu_laguna_rope_atlas_target_reuse_ready(
        uint32_t n_tokens, uint32_t pos0) {
    /* This query is intentionally read-only: no environment parsing, buffer
     * allocation, command encoding, or validity promotion occurs here. */
    return lgn2_gpu_laguna_rope_atlas_target_reuse_ready_impl(
        n_tokens, pos0);
}

int lgn2_gpu_laguna_rope_atlas_preflight(
        uint32_t n_q_head, uint32_t n_k_head,
        uint32_t head_dim, uint32_t n_rot) {
    /* The first successful/failed preflight freezes the graph plan.  A caller
     * changing the environment mid-session cannot silently change layer
     * routing; focused tests use the explicit reset hook below. */
    if (g_laguna_rope_atlas_plan_mode == -2) {
        g_laguna_rope_atlas_plan_mode = lgn2_gpu_laguna_rope_atlas_env_mode_impl();
        const char *trace = getenv("LGN2_METAL_LAGUNA_ROPE_ATLAS_TRACE");
        g_laguna_rope_atlas_trace_mode =
            trace && strcmp(trace, "1") == 0 ? 1 : 0;
    }
    const int mode = lgn2_gpu_laguna_rope_atlas_plan_mode();
    if (mode <= 0) return mode;
    int simd32_mode =
        lgn2_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached();
    if (simd32_mode == -2) {
        /* Atlas preflight is also a graph boundary for direct atlas users:
         * select the Q/K plan once, then keep all later atlas checks on that
         * frozen value rather than reparsing the selector per route. */
        /* Atlas can be selected by a single K-only support route.  That
         * route has no paired Q/K geometry; freeze SIMD32 against the
         * canonical Laguna paired contract instead of feeding 8/0 into the
         * paired selector. */
        if (lgn2_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                    48u, 8u, 128u, 64u) < 0) {
            g_laguna_rope_atlas_plan_mode = -1;
            return -1;
        }
        simd32_mode =
            lgn2_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached();
    }
    if (simd32_mode < 0) {
        g_laguna_rope_atlas_plan_mode = -1;
        return -1;
    }
    if (!lgn2_gpu_laguna_rope_atlas_geometry_ok(
            n_q_head, n_k_head, head_dim, n_rot)) {
        fprintf(stderr,
                "lgn2: Laguna RoPE atlas requested but geometry is unsupported "
                "(q_heads=%u k_heads=%u head_dim=%u n_rot=%u); refusing legacy fallback\n",
                n_q_head, n_k_head, head_dim, n_rot);
        g_laguna_rope_atlas_plan_mode = -1;
        return -1;
    }
    if (!g_initialized && !lgn2_gpu_init()) {
        fprintf(stderr,
                "lgn2: Laguna RoPE atlas requested but Metal initialization failed\n");
        g_laguna_rope_atlas_plan_mode = -1;
        return -1;
    }
    (void)lgn2_gpu_laguna_rope_atlas_pipeline();
    if (!lgn2_gpu_laguna_rope_atlas_pipeline_usable(
            g_laguna_rope_atlas_pipeline, 64u, 0u, "generation") ||
        !lgn2_gpu_laguna_rope_atlas_pipeline_usable(
            g_laguna_rope_support_atlas_pipeline, 64u, 0u,
            "support generation") ||
        !lgn2_gpu_laguna_rope_atlas_pipeline_usable(
            g_laguna_head_norm_rope_atlas_pipeline, 128u, 0u, "single") ||
        !lgn2_gpu_laguna_rope_atlas_pipeline_usable(
            g_laguna_qk_head_norm_rope_atlas_pipeline, 128u, 0u, "paired")) {
        g_laguna_rope_atlas_plan_mode = -1;
        return -1;
    }
    /* The built-in atlas source always includes this PSO.  Requiring it at
     * preflight makes a source override fail closed even when the current
     * graph happens to select stock Q/K. */
    if (!lgn2_gpu_laguna_rope_atlas_pipeline_usable(
            g_laguna_qk_head_norm_rope_simd32_atlas_pipeline,
            32u, 32u, "SIMD32 atlas paired")) {
        g_laguna_rope_atlas_plan_mode = -1;
        return -1;
    }
    (void)simd32_mode;
    if (!lgn2_gpu_laguna_rope_atlas_buffer_ensure() ||
        !lgn2_gpu_laguna_rope_support_atlas_buffer_ensure()) {
        fprintf(stderr,
                "lgn2: Laguna RoPE atlas requested but target/support storage "
                "could not be allocated before graph planning\n");
        g_laguna_rope_atlas_plan_mode = -1;
        return -1;
    }
    return 1;
}

static int lgn2_gpu_laguna_rope_atlas_buffer_ensure(void) {
    const uint64_t max_tokens = 16384u;
    const uint64_t coeffs_per_token = 2u * 64u;
    const uint64_t bytes64 = max_tokens * coeffs_per_token * sizeof(float) * 2u;
    if (bytes64 > (uint64_t)NSUIntegerMax) return 0;
    if (g_laguna_rope_atlas_buffer &&
        g_laguna_rope_atlas_buffer_bytes >= (NSUInteger)bytes64) return 1;
    id<MTLBuffer> buffer = [g_device newBufferWithLength:(NSUInteger)bytes64
                                                 options:MTLResourceStorageModePrivate];
    if (!buffer) {
        fprintf(stderr,
                "lgn2: Laguna RoPE atlas allocation failed (%.2f MiB)\n",
                (double)bytes64 / 1048576.0);
        return 0;
    }
    buffer.label = @"lgn2_laguna_rope_atlas";
    g_laguna_rope_atlas_buffer = buffer;
    g_laguna_rope_atlas_buffer_bytes = (NSUInteger)bytes64;
    g_laguna_rope_atlas_valid = 0;
    g_laguna_rope_atlas_valid_completed = 0;
    return 1;
}

static int lgn2_gpu_laguna_rope_support_atlas_buffer_ensure(void) {
    const uint64_t max_tokens = 512u;
    const uint64_t bytes64 = max_tokens * 64u * sizeof(float) * 2u;
    if (bytes64 > (uint64_t)NSUIntegerMax) return 0;
    if (g_laguna_rope_support_atlas_buffer &&
        g_laguna_rope_support_atlas_buffer_bytes >= (NSUInteger)bytes64) {
        return 1;
    }
    id<MTLBuffer> buffer = [g_device newBufferWithLength:(NSUInteger)bytes64
                                                 options:MTLResourceStorageModePrivate];
    if (!buffer) {
        fprintf(stderr,
                "lgn2: Laguna DFlash support RoPE atlas allocation failed "
                "(%.2f KiB)\n", (double)bytes64 / 1024.0);
        return 0;
    }
    buffer.label = @"lgn2_laguna_rope_support_atlas";
    g_laguna_rope_support_atlas_buffer = buffer;
    g_laguna_rope_support_atlas_buffer_bytes = (NSUInteger)bytes64;
    g_laguna_rope_support_atlas_valid = 0;
    g_laguna_rope_support_atlas_valid_completed = 0;
    return 1;
}

static int lgn2_gpu_laguna_rope_atlas_prepare(
        uint32_t n_tokens, uint32_t pos0) {
    if (n_tokens == 0u || n_tokens > 16384u) return 0;
    if (pos0 > UINT32_MAX - (n_tokens - 1u)) return 0;
    if (!lgn2_gpu_laguna_rope_atlas_buffer_ensure()) return 0;

    if ((lgn2_gpu_laguna_rope_atlas_target_reuse_ready_impl(n_tokens, pos0) ||
         (g_laguna_rope_atlas_valid &&
          !g_laguna_rope_atlas_valid_completed &&
          g_laguna_rope_atlas_valid_epoch == g_command_batch_epoch &&
          g_laguna_rope_atlas_valid_tokens == n_tokens &&
          g_laguna_rope_atlas_valid_pos0 == pos0))) {
        return 1;
    }

    lgn2_gpu_laguna_rope_atlas_args args = {
        .n_tokens = n_tokens,
        .pos0 = pos0,
        .n_families = 2u,
        .atlas_stride = 2u,
        .n_rot = {64u, 128u},
        .n_ctx_orig = {8192u, 262144u},
        .freq_base = {500000.0f, 10000.0f},
        .freq_scale = {1.0f / 32.0f, 1.0f},
        .ext_factor = {1.0f, 0.0f},
        .attn_factor = {1.0f, 1.0f},
        .beta_fast = {32.0f, 0.0f},
        .beta_slow = {1.0f, 0.0f},
    };
    int owned = 0;
    id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    if (!enc || !g_laguna_rope_atlas_pipeline ||
        !g_laguna_rope_atlas_buffer) {
        if (enc) lgn2_gpu_end_compute_encoder(cb, enc);
        if (owned) {
            (void)lgn2_gpu_finish_command_buffer(
                cb, owned, "Laguna RoPE atlas generation unavailable");
        }
        return 0;
    }
    [enc setComputePipelineState:g_laguna_rope_atlas_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:g_laguna_rope_atlas_buffer offset:0 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(1, 2, n_tokens)
         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    g_laguna_rope_atlas_encoded_dispatch_count++;
    g_laguna_rope_atlas_valid_completed = 0;
    lgn2_gpu_laguna_atlas_current_cb_evidence()->target_generated++;
    lgn2_gpu_laguna_atlas_current_cb_evidence()->target_generated_tokens =
        n_tokens;
    lgn2_gpu_laguna_atlas_current_cb_evidence()->target_generated_pos0 = pos0;
    if (lgn2_gpu_laguna_rope_atlas_trace_enabled()) {
        fprintf(stderr,
                "lgn2: Laguna RoPE atlas generation encoded tokens=%u pos0=%u "
                "families=global64+swa128\n",
                n_tokens, pos0);
    }
    lgn2_gpu_end_compute_encoder(cb, enc);
    g_laguna_rope_atlas_valid = 1;
    g_laguna_rope_atlas_valid_epoch = g_command_batch_epoch;
    g_laguna_rope_atlas_valid_tokens = n_tokens;
    g_laguna_rope_atlas_valid_pos0 = pos0;
    if (owned && !lgn2_gpu_finish_command_buffer(
            cb, owned, "Laguna RoPE atlas generation")) {
        g_laguna_rope_atlas_valid = 0;
        g_laguna_rope_atlas_valid_completed = 0;
        return 0;
    }
    return 1;
}

static int lgn2_gpu_laguna_rope_support_atlas_prepare(
        uint32_t n_tokens, uint32_t pos0) {
    if (n_tokens == 0u || n_tokens > 512u) return 0;
    if (pos0 > UINT32_MAX - (n_tokens - 1u)) return 0;
    if (!lgn2_gpu_laguna_rope_support_atlas_buffer_ensure()) return 0;
    if (g_laguna_rope_support_atlas_valid &&
        (g_laguna_rope_support_atlas_valid_completed ||
         g_laguna_rope_support_atlas_valid_epoch == g_command_batch_epoch) &&
        g_laguna_rope_support_atlas_valid_tokens == n_tokens &&
        g_laguna_rope_support_atlas_valid_pos0 == pos0) {
        return 1;
    }

    lgn2_gpu_laguna_rope_atlas_args args = {
        .n_tokens = n_tokens,
        .pos0 = pos0,
        .n_families = 1u,
        .atlas_stride = 1u,
        .n_rot = {128u, 0u},
        .n_ctx_orig = {262144u, 0u},
        .freq_base = {500000.0f, 0.0f},
        .freq_scale = {1.0f, 0.0f},
        .ext_factor = {0.0f, 0.0f},
        .attn_factor = {1.0f, 0.0f},
        .beta_fast = {0.0f, 0.0f},
        .beta_slow = {0.0f, 0.0f},
    };
    int owned = 0;
    id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    if (!enc || !g_laguna_rope_support_atlas_pipeline ||
        !g_laguna_rope_support_atlas_buffer) {
        if (enc) lgn2_gpu_end_compute_encoder(cb, enc);
        if (owned) {
            (void)lgn2_gpu_finish_command_buffer(
                cb, owned, "Laguna DFlash support RoPE atlas unavailable");
        }
        return 0;
    }
    [enc setComputePipelineState:g_laguna_rope_support_atlas_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:g_laguna_rope_support_atlas_buffer offset:0 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(1, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    g_laguna_rope_support_atlas_encoded_dispatch_count++;
    g_laguna_rope_support_atlas_valid_completed = 0;
    lgn2_gpu_laguna_atlas_current_cb_evidence()->support_generated++;
    lgn2_gpu_laguna_atlas_current_cb_evidence()->support_generated_tokens =
        n_tokens;
    lgn2_gpu_laguna_atlas_current_cb_evidence()->support_generated_pos0 = pos0;
    lgn2_gpu_end_compute_encoder(cb, enc);
    g_laguna_rope_support_atlas_valid = 1;
    g_laguna_rope_support_atlas_valid_epoch = g_command_batch_epoch;
    g_laguna_rope_support_atlas_valid_tokens = n_tokens;
    g_laguna_rope_support_atlas_valid_pos0 = pos0;
    if (owned && !lgn2_gpu_finish_command_buffer(
            cb, owned, "Laguna DFlash support RoPE atlas generation")) {
        g_laguna_rope_support_atlas_valid = 0;
        g_laguna_rope_support_atlas_valid_completed = 0;
        return 0;
    }
    return 1;
}

int lgn2_gpu_laguna_rope_atlas_trace_enabled(void) {
    return g_laguna_rope_atlas_trace_mode == 1;
}

static void lgn2_gpu_laguna_rope_atlas_trace_consume(
        const char *route, uint32_t family, uint32_t n_tokens,
        uint32_t pos0) {
    if (!lgn2_gpu_laguna_rope_atlas_trace_enabled()) return;
    fprintf(stderr,
            "lgn2: Laguna RoPE atlas consumer encoded route=%s family=%s tokens=%u "
            "pos0=%u\n",
            route ? route : "unknown",
            family == 0u ? "global64" :
            (family == 1u ? "target-swa128" : "dflash-support128"),
            n_tokens, pos0);
}

int lgn2_gpu_laguna_rope_atlas_generate(uint32_t n_tokens, uint32_t pos0) {
    if (lgn2_gpu_laguna_rope_atlas_preflight(72u, 8u, 128u, 64u) != 1) {
        return 0;
    }
    if (lgn2_gpu_laguna_rope_atlas_plan_mode() != 1) return 0;
    return lgn2_gpu_laguna_rope_atlas_prepare(n_tokens, pos0);
}

#ifdef LGN2_TEST_HOOKS
int lgn2_gpu_laguna_rope_atlas_plan_reset_for_test(void) {
    if (g_batch_cb || (g_pending_cbs && [g_pending_cbs count] != 0)) return 0;
    g_laguna_rope_atlas_plan_mode = -2;
    g_laguna_rope_atlas_trace_mode = -1;
    g_laguna_rope_atlas_invalid_env_reported = 0;
    g_laguna_rope_atlas_valid = 0;
    g_laguna_rope_support_atlas_valid = 0;
    g_laguna_rope_atlas_valid_completed = 0;
    g_laguna_rope_support_atlas_valid_completed = 0;
    return 1;
}
#endif

uint64_t lgn2_gpu_laguna_rope_atlas_encoded_dispatch_count(void) {
    return g_laguna_rope_atlas_encoded_dispatch_count;
}

uint64_t lgn2_gpu_laguna_rope_atlas_consumed_dispatch_count(void) {
    return g_laguna_rope_atlas_consumed_dispatch_count;
}

uint64_t lgn2_gpu_laguna_rope_atlas_consumed_family_count(uint32_t family) {
    return family < 2u ? g_laguna_rope_atlas_consumed_family_count[family] : 0u;
}

uint64_t lgn2_gpu_laguna_rope_atlas_completed_generated_count(void) {
    return g_laguna_rope_atlas_completed_generated_count;
}

uint64_t lgn2_gpu_laguna_rope_atlas_completed_consumed_dispatch_count(void) {
    return g_laguna_rope_atlas_completed_consumed_dispatch_count;
}

uint64_t lgn2_gpu_laguna_rope_atlas_completed_family_count(uint32_t family) {
    return family < 2u ? g_laguna_rope_atlas_completed_consumed_family_count[family] : 0u;
}

uint64_t lgn2_gpu_laguna_rope_support_atlas_encoded_dispatch_count(void) {
    return g_laguna_rope_support_atlas_encoded_dispatch_count;
}

uint64_t lgn2_gpu_laguna_rope_support_atlas_consumed_dispatch_count(void) {
    return g_laguna_rope_support_atlas_consumed_dispatch_count;
}

uint64_t lgn2_gpu_laguna_rope_support_atlas_completed_generated_count(void) {
    return g_laguna_rope_support_atlas_completed_generated_count;
}

uint64_t lgn2_gpu_laguna_rope_support_atlas_completed_consumed_dispatch_count(void) {
    return g_laguna_rope_support_atlas_completed_consumed_dispatch_count;
}

#ifdef LGN2_TEST_HOOKS
int lgn2_gpu_laguna_rope_atlas_valid_completed_for_test(void) {
    return g_laguna_rope_atlas_valid_completed ? 1 : 0;
}

int lgn2_gpu_laguna_rope_atlas_test_poison(uint32_t support_atlas) {
    if (lgn2_gpu_laguna_rope_atlas_plan_mode() != 1 || !g_batch_cb) return 0;
    id<MTLBuffer> buffer = support_atlas
        ? g_laguna_rope_support_atlas_buffer : g_laguna_rope_atlas_buffer;
    NSUInteger bytes = support_atlas
        ? g_laguna_rope_support_atlas_buffer_bytes : g_laguna_rope_atlas_buffer_bytes;
    if (!buffer || bytes == 0) return 0;
    lgn2_gpu_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [g_batch_cb blitCommandEncoder];
    if (!blit) return 0;
    [blit fillBuffer:buffer range:NSMakeRange(0, bytes) value:0];
    [blit endEncoding];
    g_batch_has_work = YES;
    return 1;
}
#endif

static int lgn2_gpu_laguna_qk_head_norm_rope_simd32_usable(
        id<MTLComputePipelineState> pipeline) {
    if (!pipeline) return 0;
    if (pipeline.threadExecutionWidth != 32u) {
        fprintf(stderr,
                "lgn2: Laguna Q/K norm/RoPE SIMD32 pipeline has execution "
                "width %lu, requires 32; requested configuration fails\n",
                (unsigned long)pipeline.threadExecutionWidth);
        return 0;
    }
    if (pipeline.maxTotalThreadsPerThreadgroup < 32u) {
        fprintf(stderr,
                "lgn2: Laguna Q/K norm/RoPE SIMD32 pipeline supports %lu "
                "threads, needs at least 32; requested configuration fails\n",
                (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);
        return 0;
    }
    return 1;
}

static int lgn2_gpu_disable_hot_pipeline_statics(void) {
    static int initialized;
    static int disabled;
    if (!initialized) {
        disabled = getenv("LGN2_METAL_DISABLE_HOT_PIPELINE_STATICS") != NULL;
        initialized = 1;
    }
    return disabled;
}

static id<MTLComputePipelineState> lgn2_gpu_hot_pipeline(
        id<MTLComputePipelineState> pipeline,
        const char *fallback_name) {
    if (!lgn2_gpu_disable_hot_pipeline_statics()) return pipeline;
    return lgn2_gpu_get_pipeline(fallback_name);
}


static int lgn2_gpu_device_name_contains(const char *needle);

static int lgn2_gpu_env_value_eq(const char *v, size_t n, const char *literal) {
    size_t m = strlen(literal);
    if (n != m) return 0;
    for (size_t i = 0; i < n; i++) {
        if (tolower((unsigned char)v[i]) != tolower((unsigned char)literal[i])) return 0;
    }
    return 1;
}

static int lgn2_gpu_env_bool(const char *name) {
    const char *v = getenv(name);
    if (!v) return -1;

    while (isspace((unsigned char)*v)) v++;
    size_t n = strlen(v);
    while (n > 0 && isspace((unsigned char)v[n - 1])) n--;
    if (n == 0) return 1;

    if (lgn2_gpu_env_value_eq(v, n, "1") ||
        lgn2_gpu_env_value_eq(v, n, "true") ||
        lgn2_gpu_env_value_eq(v, n, "yes") ||
        lgn2_gpu_env_value_eq(v, n, "on")) {
        return 1;
    }
    if (lgn2_gpu_env_value_eq(v, n, "0") ||
        lgn2_gpu_env_value_eq(v, n, "false") ||
        lgn2_gpu_env_value_eq(v, n, "no") ||
        lgn2_gpu_env_value_eq(v, n, "off")) {
        return 0;
    }

    if (!g_mpp_invalid_env_reported) {
        fprintf(stderr,
                "lgn2: invalid Metal boolean environment value %s=%.*s; treating presence as enabled\n",
                name, (int)n, v);
        g_mpp_invalid_env_reported = 1;
    }
    return 1;
}

/* This experiment is intentionally stricter than the legacy presence-based
 * toggles: a typo must fail before graph work, never silently change the
 * reduction topology or select the ordinary 128-thread kernel. */
static int lgn2_gpu_laguna_qk_head_norm_rope_simd32_env_mode_impl(void) {
    const char *v = getenv("LGN2_METAL_LAGUNA_QK_NORM_ROPE_SIMD32");
    if (!v || !v[0] || strcmp(v, "0") == 0) return 0;
    if (strcmp(v, "1") == 0) return 1;
    if (!g_laguna_qk_head_norm_rope_simd32_invalid_env_reported) {
        fprintf(stderr,
                "lgn2: invalid LGN2_METAL_LAGUNA_QK_NORM_ROPE_SIMD32=%s; "
                "requested configuration is fatal (use unset, empty, 0, or 1)\n",
                v);
        g_laguna_qk_head_norm_rope_simd32_invalid_env_reported = 1;
    }
    return -1;
}

static int lgn2_gpu_laguna_qk_head_norm_rope_simd32_geometry_ok(
        uint32_t n_q_head,
        uint32_t n_k_head,
        uint32_t head_dim,
        uint32_t n_rot) {
    return (n_q_head == 48u || n_q_head == 72u) &&
           n_k_head == 8u && head_dim == 128u &&
           (n_rot == 64u || n_rot == 128u);
}

int lgn2_gpu_laguna_qk_head_norm_rope_simd32_env_mode(void) {
    return lgn2_gpu_laguna_qk_head_norm_rope_simd32_env_mode_impl();
}

int lgn2_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached(void) {
    return g_laguna_qk_head_norm_rope_simd32_plan_mode;
}

int lgn2_gpu_laguna_qk_head_norm_rope_simd32_trace_enabled(void) {
    return g_laguna_qk_head_norm_rope_simd32_trace_mode == 1;
}

#ifdef LGN2_TEST_HOOKS
int lgn2_gpu_laguna_qk_head_norm_rope_simd32_plan_reset_for_test(void) {
    if (g_batch_cb || (g_pending_cbs && [g_pending_cbs count] != 0)) return 0;
    g_laguna_qk_head_norm_rope_simd32_plan_mode = -2;
    g_laguna_qk_head_norm_rope_simd32_trace_mode = -1;
    g_laguna_qk_head_norm_rope_simd32_invalid_env_reported = 0;
    return 1;
}
#endif

int lgn2_gpu_laguna_qk_head_norm_rope_simd32_preflight(
        uint32_t n_q_head,
        uint32_t n_k_head,
        uint32_t head_dim,
        uint32_t n_rot) {
    if (g_laguna_qk_head_norm_rope_simd32_plan_mode == -2) {
        g_laguna_qk_head_norm_rope_simd32_plan_mode =
            lgn2_gpu_laguna_qk_head_norm_rope_simd32_env_mode_impl();
        const char *trace =
            getenv("LGN2_METAL_LAGUNA_QK_NORM_ROPE_SIMD32_TRACE");
        g_laguna_qk_head_norm_rope_simd32_trace_mode =
            trace && strcmp(trace, "1") == 0 ? 1 : 0;
    }
    const int mode = g_laguna_qk_head_norm_rope_simd32_plan_mode;
    if (mode <= 0) return mode;
    if (!lgn2_gpu_laguna_qk_head_norm_rope_simd32_geometry_ok(
                n_q_head, n_k_head, head_dim, n_rot)) {
        fprintf(stderr,
                "lgn2: Laguna Q/K norm/RoPE SIMD32 requested but geometry "
                "is unsupported (q_heads=%u k_heads=%u head_dim=%u n_rot=%u); "
                "refusing stock fallback\n",
                n_q_head, n_k_head, head_dim, n_rot);
        g_laguna_qk_head_norm_rope_simd32_plan_mode = -1;
        return -1;
    }
    if (!g_initialized && !lgn2_gpu_init()) {
        fprintf(stderr,
                "lgn2: Laguna Q/K norm/RoPE SIMD32 requested but Metal "
                "initialization failed\n");
        g_laguna_qk_head_norm_rope_simd32_plan_mode = -1;
        return -1;
    }
    id<MTLComputePipelineState> pipeline =
        lgn2_gpu_laguna_qk_head_norm_rope_simd32_pipeline();
    if (!lgn2_gpu_laguna_qk_head_norm_rope_simd32_usable(pipeline)) {
        if (!pipeline) {
            fprintf(stderr,
                    "lgn2: Laguna Q/K norm/RoPE SIMD32 requested but its "
                    "pipeline is unavailable; refusing stock fallback\n");
        }
        g_laguna_qk_head_norm_rope_simd32_plan_mode = -1;
        return -1;
    }
    return 1;
}

/* Q8 decode dispatch configuration is a process-lifecycle snapshot.  Keep
 * this parser next to the Metal environment helpers so graph admission and
 * every decode descriptor cannot accidentally grow separate getenv caches.
 * Empty/unset/0 select the established defaults; explicit numeric values are
 * validated instead of silently clamped or falling back. */
static lgn2_gpu_q8_decode_config g_q8_decode_config;
static int g_q8_decode_config_initialized;

static int lgn2_gpu_q8_parse_decimal_selector(
        const char *name,
        int         default_value,
        int         min_value,
        int         max_value,
        int         zero_is_default,
        int        *value_out) {
    const char *v = getenv(name);
    if (!v || v[0] == '\0') {
        *value_out = default_value;
        return 1;
    }
    if (zero_is_default && v[0] == '0' && v[1] == '\0') {
        *value_out = default_value;
        return 1;
    }

    errno = 0;
    char *end = NULL;
    unsigned long parsed = strtoul(v, &end, 10);
    if (end == v || errno == ERANGE || *end != '\0' ||
        parsed < (unsigned long)min_value ||
        parsed > (unsigned long)max_value) {
        fprintf(stderr,
                "lgn2: fatal: invalid %s='%s'; expected %s\n",
                name,
                v,
                zero_is_default ?
                    "unset, empty, 0, or a bounded decimal value" :
                    "unset, empty, or a bounded decimal value");
        *value_out = -1;
        return 0;
    }
    *value_out = (int)parsed;
    return 1;
}

int lgn2_gpu_q8_decode_config_snapshot(lgn2_gpu_q8_decode_config *out) {
    if (!g_q8_decode_config_initialized) {
        g_q8_decode_config.q8_mv_nsg_override = 0;
        g_q8_decode_config.q8_mv_rows = 2;
        const int nsg_ok = lgn2_gpu_q8_parse_decimal_selector(
            "LGN2_METAL_Q8_MV_NSG", 0, 1, 8, 1,
            &g_q8_decode_config.q8_mv_nsg_override);
        int rows_ok = lgn2_gpu_q8_parse_decimal_selector(
            "LGN2_METAL_Q8_MV_ROWS", 2, 2, 4, 1,
            &g_q8_decode_config.q8_mv_rows);
        if (rows_ok && g_q8_decode_config.q8_mv_rows != 2 &&
            g_q8_decode_config.q8_mv_rows != 4) {
            fprintf(stderr,
                    "lgn2: fatal: invalid LGN2_METAL_Q8_MV_ROWS='%s'; "
                    "expected unset, empty, 0, 2, or 4\n",
                    getenv("LGN2_METAL_Q8_MV_ROWS"));
            g_q8_decode_config.q8_mv_rows = -1;
            rows_ok = 0;
        }
        g_q8_decode_config_initialized = 1;
        (void)nsg_ok;
        (void)rows_ok;
    }
    if (out) *out = g_q8_decode_config;
    return g_q8_decode_config.q8_mv_nsg_override < 0 ||
                   g_q8_decode_config.q8_mv_rows < 0 ? -1 : 1;
}

#define LGN2_METAL_LAGUNA_SWA_GQA9 "LGN2_METAL_LAGUNA_SWA_GQA9"
#define LGN2_METAL_LAGUNA_SWA_GQA3 "LGN2_METAL_LAGUNA_SWA_GQA3"
#define LGN2_METAL_LAGUNA_STAGED_SWA "LGN2_METAL_LAGUNA_STAGED_SWA"

/* GQA9 is an opt-in route whose source/PSO ABI must be unambiguous.  Do not
 * trim or accept the broad boolean vocabulary here: unset, empty, and the
 * literal string 0 are off; only the literal string 1 is on. */
static int lgn2_gpu_laguna_swa_gqa9_parse(const char *value) {
    if (!value || value[0] == '\0' || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") == 0) return 1;
    return -1;
}

static int lgn2_gpu_laguna_swa_gqa9_env_mode(void) {
    if (g_initialized && g_laguna_swa_selectors_snapshot_valid) {
        return g_laguna_swa_gqa9_mode;
    }
    return lgn2_gpu_laguna_swa_gqa9_parse(
        getenv(LGN2_METAL_LAGUNA_SWA_GQA9));
}

static int lgn2_gpu_laguna_swa_gqa3_enabled(void) {
    if (g_initialized && g_laguna_swa_selectors_snapshot_valid) {
        return g_laguna_swa_gqa3_mode;
    }
    return lgn2_gpu_env_bool(LGN2_METAL_LAGUNA_SWA_GQA3) > 0;
}

static int lgn2_gpu_laguna_staged_swa_enabled(void) {
    if (g_initialized && g_laguna_swa_selectors_snapshot_valid) {
        return g_laguna_staged_swa_mode;
    }
    return lgn2_gpu_env_bool(LGN2_METAL_LAGUNA_STAGED_SWA) > 0;
}

static int lgn2_gpu_snapshot_lifecycle_selectors(void) {
    g_laguna_swa_gqa9_mode = lgn2_gpu_laguna_swa_gqa9_parse(
        getenv(LGN2_METAL_LAGUNA_SWA_GQA9));
    if (g_laguna_swa_gqa9_mode < 0) {
        fprintf(stderr,
                "lgn2: invalid %s='%s'; expected unset, empty, 0, or literal 1\n",
                LGN2_METAL_LAGUNA_SWA_GQA9,
                getenv(LGN2_METAL_LAGUNA_SWA_GQA9) ?
                    getenv(LGN2_METAL_LAGUNA_SWA_GQA9) : "");
        g_laguna_swa_selectors_snapshot_valid = 0;
        return 0;
    }
    g_laguna_swa_gqa3_mode =
        lgn2_gpu_env_bool(LGN2_METAL_LAGUNA_SWA_GQA3) > 0;
    g_laguna_staged_swa_mode =
        lgn2_gpu_env_bool(LGN2_METAL_LAGUNA_STAGED_SWA) > 0;
    g_laguna_swa_selectors_snapshot_valid = 1;
    g_q8_mv_ext_max_tokens =
        lgn2_gpu_q8_mv_ext_max_tokens_parse(
            getenv("LGN2_METAL_Q8_MV_EXT_MAX_TOKENS"));
    g_glm_grouped_moe_min_tokens =
        lgn2_gpu_laguna_moe_min_tokens_parse(
            getenv("LGN2_METAL_LAGUNA_GROUPED_MOE_MIN_TOKENS"));
    g_direct_kv_prefill_mode =
        lgn2_gpu_laguna_direct_kv_prefill_env_mode(
            getenv("LGN2_METAL_LAGUNA_DIRECT_KV_PREFILL"));
    return 1;
}

uint32_t lgn2_gpu_laguna_q8_mv_ext_max_tokens(void) {
    return g_q8_mv_ext_max_tokens;
}

#ifdef LGN2_TEST_HOOKS
void lgn2_gpu_test_laguna_route_counters_reset(void) {
    g_laguna_test_direct_kv_count = 0;
    g_laguna_test_wrap_kv_count = 0;
    g_laguna_test_fused_q8_count = 0;
    g_laguna_test_stock_q8_count = 0;
    g_laguna_test_fused_q8_bco_false_count = 0;
    g_laguna_test_fused_q8_bco_true_count = 0;
    memset(g_laguna_test_decode_route_counts, 0,
           sizeof(g_laguna_test_decode_route_counts));
    memset(g_laguna_test_decode_route_batch, 0,
           sizeof(g_laguna_test_decode_route_batch));
    memset(g_laguna_test_decode_route_inflight, 0,
           sizeof(g_laguna_test_decode_route_inflight));
    g_laguna_test_route_hooks = 1;
}

int lgn2_gpu_test_laguna_route_counters(uint64_t *direct_kv,
                                       uint64_t *wrap_kv,
                                       uint64_t *fused_q8,
                                       uint64_t *stock_q8) {
    if (!g_initialized || !g_laguna_test_route_hooks) return 0;
    if (direct_kv) *direct_kv = g_laguna_test_direct_kv_count;
    if (wrap_kv) *wrap_kv = g_laguna_test_wrap_kv_count;
    if (fused_q8) *fused_q8 = g_laguna_test_fused_q8_count;
    if (stock_q8) *stock_q8 = g_laguna_test_stock_q8_count;
    return 1;
}

int lgn2_gpu_test_laguna_q8_bco_counters(uint64_t *bco_false,
                                        uint64_t *bco_true) {
    if (!g_initialized || !g_laguna_test_route_hooks) return 0;
    if (bco_false) *bco_false = g_laguna_test_fused_q8_bco_false_count;
    if (bco_true) *bco_true = g_laguna_test_fused_q8_bco_true_count;
    return 1;
}

int lgn2_gpu_test_laguna_decode_route_counters(
        uint64_t *ordinary,
        uint64_t *gqa3,
        uint64_t *gqa9,
        uint64_t *staged,
        uint64_t *global_grouped) {
    if (!g_initialized || !g_laguna_test_route_hooks) return 0;
    if (ordinary) *ordinary =
        g_laguna_test_decode_route_counts[LGN2_LAGUNA_TEST_DECODE_ORDINARY];
    if (gqa3) *gqa3 =
        g_laguna_test_decode_route_counts[LGN2_LAGUNA_TEST_DECODE_GQA3];
    if (gqa9) *gqa9 =
        g_laguna_test_decode_route_counts[LGN2_LAGUNA_TEST_DECODE_GQA9];
    if (staged) *staged =
        g_laguna_test_decode_route_counts[LGN2_LAGUNA_TEST_DECODE_STAGED];
    if (global_grouped) *global_grouped =
        g_laguna_test_decode_route_counts[
            LGN2_LAGUNA_TEST_DECODE_GLOBAL_GROUPED];
    return 1;
}

void lgn2_gpu_test_glm_grouped_moe_counters_reset(void) {
    g_test_glm_grouped_moe_encoded_dispatches = 0;
    g_test_glm_grouped_moe_batch_dispatches = 0;
    g_test_glm_grouped_moe_owned_dispatches = 0;
    g_test_glm_grouped_moe_completed_dispatches = 0;
}

int lgn2_gpu_test_glm_grouped_moe_counters(
        uint64_t *encoded_dispatches,
        uint64_t *completed_dispatches) {
    if (encoded_dispatches) {
        *encoded_dispatches = g_test_glm_grouped_moe_encoded_dispatches;
    }
    if (completed_dispatches) {
        *completed_dispatches = g_test_glm_grouped_moe_completed_dispatches;
    }
    return 1;
}

void lgn2_gpu_test_glm_exact_q4_counters_reset(void) {
    g_test_glm_exact_q4_encoded_dispatches = 0;
    g_test_glm_exact_q4_completed_dispatches = 0;
}

int lgn2_gpu_test_glm_exact_q4_counters(
        uint64_t *encoded_dispatches,
        uint64_t *completed_dispatches) {
    if (encoded_dispatches) {
        *encoded_dispatches = g_test_glm_exact_q4_encoded_dispatches;
    }
    if (completed_dispatches) {
        *completed_dispatches = g_test_glm_exact_q4_completed_dispatches;
    }
    return 1;
}

void lgn2_gpu_test_laguna_set_direct_kv_mode(int mode) {
    g_direct_kv_prefill_mode = mode;
}

static void lgn2_gpu_laguna_test_decode_route_note(
        int route,
        int owned) {
    if (!g_laguna_test_route_hooks ||
        route < 0 || route >= LGN2_LAGUNA_TEST_DECODE_ROUTE_COUNT) return;
    if (owned) {
        g_laguna_test_decode_route_counts[route]++;
    } else {
        g_laguna_test_decode_route_batch[route]++;
    }
}

static void lgn2_gpu_laguna_test_decode_route_batch_committed(void) {
    if (!g_laguna_test_route_hooks) return;
    for (int i = 0; i < LGN2_LAGUNA_TEST_DECODE_ROUTE_COUNT; i++) {
        g_laguna_test_decode_route_inflight[i] +=
            g_laguna_test_decode_route_batch[i];
        g_laguna_test_decode_route_batch[i] = 0;
    }
}

static void lgn2_gpu_laguna_test_decode_route_batch_completed(int ok) {
    if (!g_laguna_test_route_hooks) return;
    if (ok) {
        for (int i = 0; i < LGN2_LAGUNA_TEST_DECODE_ROUTE_COUNT; i++) {
            g_laguna_test_decode_route_counts[i] +=
                g_laguna_test_decode_route_inflight[i];
        }
    }
    memset(g_laguna_test_decode_route_inflight, 0,
           sizeof(g_laguna_test_decode_route_inflight));
}

static int lgn2_gpu_laguna_test_decode_route_kind(
        uint32_t cache_cap,
        uint32_t key_start,
        uint32_t key_count,
        uint32_t n_head,
        uint32_t n_head_kv,
        int      gqa9_selected,
        int      gqa3_selected,
        int      staged_swa_active) {
    const bool exact_swa =
        cache_cap == 512u && key_count == 512u &&
        n_head == 72u && n_head_kv == 8u;
    /* The single-token decode entry point suppresses the grouped kernels when
     * staged SWA is selected, but it does not itself execute the staged
     * multi-row kernel.  Keep this completion counter honest: a staged label
     * here means the staged selector won precedence over a grouped selector,
     * while the actual staged kernel is counted at its own completion site
     * below. */
    if (exact_swa && staged_swa_active &&
        (gqa9_selected || gqa3_selected)) {
        return LGN2_LAGUNA_TEST_DECODE_STAGED;
    }
    if (exact_swa && gqa9_selected && !staged_swa_active) {
        return LGN2_LAGUNA_TEST_DECODE_GQA9;
    }
    if (exact_swa && gqa3_selected && !gqa9_selected &&
        !staged_swa_active) {
        return LGN2_LAGUNA_TEST_DECODE_GQA3;
    }
    if (cache_cap > 512u && key_start == 0u &&
        n_head % 3u == 0u &&
        ((n_head / n_head_kv) % 3u) == 0u &&
        ((key_count >= 512u && key_count < cache_cap) ||
         (key_count >= 1024u && key_count % 32u == 0u))) {
        return LGN2_LAGUNA_TEST_DECODE_GLOBAL_GROUPED;
    }
    return LGN2_LAGUNA_TEST_DECODE_ORDINARY;
}
#endif

static int lgn2_gpu_mpp_available(void) {
    return g_metal4_tensor_api_enabled && !g_quality_mode &&
           !g_tensor_matmul_suppressed;
}

/*
 * Retained Metal4 defaults live here instead of behind user-visible options.
 * The public runtime has one automatic accelerated path plus the global
 * LGN2_METAL_DISABLE_METAL4 comparison switch.  Benchmark-only alternatives that
 * lost during M5 work are removed or kept out of the dispatch path so future
 * changes do not accidentally turn old experiments into new modes.
 */
static void lgn2_gpu_warn_mpp_fallback(void) {
    static int warned;
    if (!warned) {
        fprintf(stderr, "lgn2: accelerated Metal prefill matmul unavailable; falling back to legacy kernel\n");
        warned = 1;
    }
}

static int lgn2_gpu_device_name_contains(const char *needle) {
    return g_metal_device_name[0] != '\0' && strstr(g_metal_device_name, needle) != NULL;
}

int lgn2_gpu_device_is_pre_m5_apple_silicon(void) {
    return strncmp(g_metal_device_name, "Apple M", 7) == 0 &&
           g_metal_device_name[7] >= '1' &&
           g_metal_device_name[7] <= '4' &&
           (g_metal_device_name[8] == '\0' ||
            g_metal_device_name[8] == ' ');
}

int lgn2_gpu_device_is_m5_apple_silicon(void) {
    return strncmp(g_metal_device_name, "Apple M5", 8) == 0 &&
           (g_metal_device_name[8] == '\0' ||
            g_metal_device_name[8] == ' ');
}

static int lgn2_gpu_compile_tensor_probe(void) {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (!g_device) return 0;
    if (@available(macOS 26.0, *)) {
        const char *src =
            "#include <metal_stdlib>\n"
            "#include <metal_tensor>\n"
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            "using namespace metal;\n"
            "using namespace mpp::tensor_ops;\n"
            "kernel void ds4_tensor_probe(\n"
            "        tensor<device half,  dextents<int32_t, 2>> A [[buffer(0)]],\n"
            "        tensor<device half,  dextents<int32_t, 2>> B [[buffer(1)]],\n"
            "        device float *C [[buffer(2)]],\n"
            "        uint2 tgid [[threadgroup_position_in_grid]]) {\n"
            "    auto tA = A.slice(0, (int)tgid.y);\n"
            "    auto tB = B.slice((int)tgid.x, 0);\n"
            "    matmul2d<matmul2d_descriptor(16, 16, dynamic_extent), execution_simdgroups<4>> mm;\n"
            "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();\n"
            "    auto sA = tA.slice(0, 0);\n"
            "    auto sB = tB.slice(0, 0);\n"
            "    mm.run(sB, sA, cT);\n"
            "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16));\n"
            "    cT.store(tC);\n"
            "}\n";

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:src];
        id<MTLLibrary> probe_library = [g_device newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
        if (!probe_library) {
            fprintf(stderr, "lgn2: Metal 4 tensor API probe compile failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        id<MTLFunction> fn = [probe_library newFunctionWithName:@"ds4_tensor_probe"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal 4 tensor API probe function missing\n");
            return 0;
        }
        error = nil;
        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            fprintf(stderr, "lgn2: Metal 4 tensor API probe pipeline failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        return 1;
    }
#endif
    return 0;
}

static void lgn2_gpu_detect_metal4_features(void) {
    g_metal4_runtime_available = 0;
    g_metal4_family_supported = 0;
    g_metal4_queue_supported = 0;
    g_metal4_m5_neural_accelerators_hint = 0;
    g_metal4_tensor_api_enabled = 0;
    g_metal4_tensor_api_compile_supported = 0;
    g_metal_device_name[0] = '\0';

    if (!g_device) return;

    const char *name = [[g_device name] UTF8String];
    if (name) {
        snprintf(g_metal_device_name, sizeof(g_metal_device_name), "%s", name);
    }

    const int metal4_disabled = lgn2_gpu_env_bool("LGN2_METAL_DISABLE_METAL4") > 0;

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
        g_metal4_runtime_available = 1;
        g_metal4_family_supported =
            !metal4_disabled && [g_device supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
        g_metal4_queue_supported = [g_device respondsToSelector:@selector(newMTL4CommandQueue)] ? 1 : 0;

        /*
         * Apple does not currently expose a separate "Neural Accelerator" bit
         * through Metal. On public M5 systems the hardware signal is the device
         * generation plus Metal 4 support, so keep this as a conservative hint.
         */
        if (g_metal4_family_supported && lgn2_gpu_device_name_contains("M5")) {
            g_metal4_m5_neural_accelerators_hint = 1;
        }

        if (g_metal4_family_supported) {
            const int default_enable =
                lgn2_gpu_device_name_contains("M5") ||
                lgn2_gpu_device_name_contains("M6") ||
                lgn2_gpu_device_name_contains("A19") ||
                lgn2_gpu_device_name_contains("A20");

            /*
             * Metal 4 TensorOps are portable in source, but on pre-M5 hardware
             * they can map to ordinary shader fallbacks.  Keep the automatic
             * fast path restricted to hardware generations where the Neural
             * Accelerator/TensorOps path is expected to pay off; older Metal
             * machines continue to use the established kernels unless a future
             * device is explicitly added here.
             */
            if (default_enable) {
                g_metal4_tensor_api_compile_supported = lgn2_gpu_compile_tensor_probe();
                g_metal4_tensor_api_enabled = g_metal4_tensor_api_compile_supported;
                if (!g_metal4_tensor_api_enabled) {
                    fprintf(stderr, "lgn2: Metal 4 tensor API probe failed; using legacy Metal kernels\n");
                }
            } else {
                fprintf(stderr, "lgn2: Metal 4 tensor API disabled for pre-M5/pre-A19 devices\n");
            }
        }
    }
#endif
}

static int lgn2_gpu_warm_model_views(void) {
    if (g_model_view_count == 0) return 1;

    id<MTLComputePipelineState> pipeline = lgn2_gpu_get_pipeline("kernel_touch_u8_stride");
    if (!pipeline) return 0;

    uint64_t stride = 1024ull * 1024ull;
    const char *stride_env = getenv("LGN2_METAL_MODEL_WARMUP_STRIDE_MB");
    if (stride_env && stride_env[0]) {
        char *end = NULL;
        unsigned long long mb = strtoull(stride_env, &end, 10);
        if (end != stride_env && mb > 0 && mb <= 1024) {
            stride = mb * 1024ull * 1024ull;
        }
    }
    const char *stride_kb_env = getenv("LGN2_METAL_MODEL_WARMUP_STRIDE_KB");
    if (stride_kb_env && stride_kb_env[0]) {
        char *end = NULL;
        unsigned long long kb = strtoull(stride_kb_env, &end, 10);
        if (end != stride_kb_env && kb > 0 && kb <= 1024ull * 1024ull) {
            stride = kb * 1024ull;
            const uint64_t page = (uint64_t)getpagesize();
            if (stride < page) stride = page;
        }
    }

    uint64_t total_touches = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        total_touches += (g_model_views[i].bytes + stride - 1) / stride;
    }
    if (total_touches == 0 || total_touches > (uint64_t)NSUIntegerMax) return 0;

    const NSUInteger out_bytes = (NSUInteger)total_touches;
    id<MTLBuffer> out = [g_device newBufferWithLength:out_bytes
                                             options:MTLResourceStorageModeShared];
    if (!out) {
        fprintf(stderr, "lgn2: Metal model warmup scratch allocation failed\n");
        return 0;
    }
    out.label = @"lgn2_model_warmup";

    id<MTLCommandBuffer> cb = lgn2_gpu_new_command_buffer();
    if (!cb) {
        fprintf(stderr, "lgn2: Metal model warmup command buffer allocation failed\n");
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    uint64_t dst_offset = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        const uint64_t bytes = g_model_views[i].bytes;
        const uint64_t n = (bytes + stride - 1) / stride;
        [enc setBuffer:g_model_views[i].buffer offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc setBytes:&stride length:sizeof(stride) atIndex:2];
        [enc setBytes:&bytes length:sizeof(bytes) atIndex:3];
        [enc setBytes:&dst_offset length:sizeof(dst_offset) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((n + 255) / 256), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        dst_offset += n;
    }
    lgn2_gpu_end_compute_encoder(cb, enc);

    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "lgn2: Metal model warmup failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    return 1;
}

static const char *lgn2_gpu_mul_mm_id_map0_name(uint32_t ne20) {
    switch (ne20) {
        case 1:  return "kernel_mul_mm_id_map0_ne20_1";
        case 2:  return "kernel_mul_mm_id_map0_ne20_2";
        case 4:  return "kernel_mul_mm_id_map0_ne20_4";
        case 5:  return "kernel_mul_mm_id_map0_ne20_5";
        case 6:  return "kernel_mul_mm_id_map0_ne20_6";
        case 8:  return "kernel_mul_mm_id_map0_ne20_8";
        case 10: return "kernel_mul_mm_id_map0_ne20_10";
        case 16: return "kernel_mul_mm_id_map0_ne20_16";
        case 22: return "kernel_mul_mm_id_map0_ne20_22";
        default: return NULL;
    }
}






static id<MTLComputePipelineState> lgn2_gpu_get_mul_mv_pipeline(
        const char *function_name,
        int16_t     nsg) {
    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d", function_name, (int)nsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "lgn2: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "lgn2: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> lgn2_gpu_get_mul_mv_ext_pipeline(
        const char *function_name,
        int16_t     nsg,
        int16_t     nxpsg) {
    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d_nxpsg=%d",
                     function_name, (int)nsg, (int)nxpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg   type:MTLDataTypeShort atIndex:600];
    [constants setConstantValue:&nxpsg type:MTLDataTypeShort atIndex:601];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "lgn2: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "lgn2: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> lgn2_gpu_get_flash_attn_pad_pipeline(
        bool    has_mask,
        int32_t ncpsg) {
    /*
     * Decode calls this once per layer with identical arguments, so memoize
     * the last hit and skip the NSString key + dictionary lookup on the hot
     * path.  The generic cache below remains the fallback for new variants.
     * The rollback switch restores the dictionary path for same-binary A/B.
     */
    static struct {
        bool m;
        int32_t nc;
        id<MTLComputePipelineState> pipeline;
    } memo;
    const bool memo_disabled =
        getenv("LGN2_METAL_DISABLE_PRE_M5_FLASH_ATTN_PAD_BLK_MEMO") != NULL;
    if (!memo_disabled && memo.pipeline && memo.m == has_mask && memo.nc == ncpsg) {
        return memo.pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_pad_mask=%d_ncpsg=%d",
                     has_mask ? 1 : 0, (int)ncpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        if (!memo_disabled) {
            memo = (typeof(memo)){ has_mask, ncpsg, cached };
        }
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask type:MTLDataTypeBool atIndex:100];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:125];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_pad"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "lgn2: Metal kernel_flash_attn_ext_pad function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "lgn2: Metal kernel_flash_attn_ext_pad pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    if (!memo_disabled) {
        memo = (typeof(memo)){ has_mask, ncpsg, pipeline };
    }
    return pipeline;
}

static id<MTLComputePipelineState> lgn2_gpu_get_flash_attn_vec_pipeline(
        const char *function_name,
        bool        has_mask,
        bool        has_sinks,
        bool        has_bias,
        bool        has_scap,
        bool        has_kvpad,
        bool        shared_kvpad,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg,
        int32_t     nwg) {
    /*
     * Decode calls this once per layer with identical arguments, so memoize
     * the last hit and skip the NSString key + dictionary lookup on the hot
     * path.  The generic cache below remains the fallback for new variants.
     */
    static struct {
        const char *fn;
        bool m, s, b, c, k, sp;
        int32_t n10, n20, sg, wg;
        id<MTLComputePipelineState> pipeline;
    } memo;
    if (memo.pipeline && memo.fn != NULL && strcmp(memo.fn, function_name) == 0 &&
        memo.m == has_mask && memo.s == has_sinks && memo.b == has_bias &&
        memo.c == has_scap && memo.k == has_kvpad && memo.sp == shared_kvpad &&
        memo.n10 == ns10 && memo.n20 == ns20 && memo.sg == nsg && memo.wg == nwg) {
        return memo.pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_sharedpad=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
                     function_name,
                     has_mask ? 1 : 0,
                     has_sinks ? 1 : 0,
                     has_bias ? 1 : 0,
                     has_scap ? 1 : 0,
                     has_kvpad ? 1 : 0,
                     shared_kvpad ? 1 : 0,
                     (int)ns10,
                     (int)ns20,
                     (int)nsg,
                     (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                               has_scap, has_kvpad, shared_kvpad, ns10, ns20,
                               nsg, nwg, cached };
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:400];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:401];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:402];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:403];
    [constants setConstantValue:&has_kvpad type:MTLDataTypeBool atIndex:404];
    [constants setConstantValue:&shared_kvpad type:MTLDataTypeBool atIndex:405];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:420];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:421];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:422];
    [constants setConstantValue:&nwg  type:MTLDataTypeInt atIndex:423];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "lgn2: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "lgn2: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                           has_scap, has_kvpad, shared_kvpad, ns10, ns20,
                           nsg, nwg, pipeline };
    return pipeline;
}

/* The staged Laguna SWA verifier uses a separate qF32/F16 Flash-vector
 * specialization with an extended argument struct.  Keep this pipeline out
 * of the mandatory init set: older devices/libraries can use the row loop. */
static id<MTLComputePipelineState>
lgn2_gpu_get_laguna_staged_flash_attn_vec_pipeline(
        const char *function_name,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg,
        int32_t     nwg) {
    NSString *key = [NSString stringWithFormat:
        @"%s_virtual=1_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
        function_name, (int)ns10, (int)ns20, (int)nsg, (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    const bool has_mask = false;
    const bool has_sinks = false;
    const bool has_bias = false;
    const bool has_scap = false;
    const bool has_kvpad = false;
    const bool shared_kvpad = false;
    MTLFunctionConstantValues *constants =
        [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask
                           type:MTLDataTypeBool atIndex:400];
    [constants setConstantValue:&has_sinks
                           type:MTLDataTypeBool atIndex:401];
    [constants setConstantValue:&has_bias
                           type:MTLDataTypeBool atIndex:402];
    [constants setConstantValue:&has_scap
                           type:MTLDataTypeBool atIndex:403];
    [constants setConstantValue:&has_kvpad
                           type:MTLDataTypeBool atIndex:404];
    [constants setConstantValue:&shared_kvpad
                           type:MTLDataTypeBool atIndex:405];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:420];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:421];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:422];
    [constants setConstantValue:&nwg  type:MTLDataTypeInt atIndex:423];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "lgn2: Metal %s staged function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }
    id<MTLComputePipelineState> pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "lgn2: Metal %s staged pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }
    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState>
lgn2_gpu_get_laguna_flash_attn_reduce_gate_pipeline(int32_t dv, int32_t nwg) {
    static int32_t memo_dv, memo_nwg;
    static id<MTLComputePipelineState> memo_pipeline;
    if (memo_pipeline && memo_dv == dv && memo_nwg == nwg) {
        return memo_pipeline;
    }

    NSString *key = [NSString stringWithFormat:
        @"kernel_laguna_flash_attn_reduce_gate_f32_dv=%d_nwg=%d",
        (int)dv, (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        memo_dv = dv;
        memo_nwg = nwg;
        memo_pipeline = cached;
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&dv type:MTLDataTypeInt atIndex:500];
    [constants setConstantValue:&nwg type:MTLDataTypeInt atIndex:501];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library
        newFunctionWithName:@"kernel_laguna_flash_attn_reduce_gate_f32"
        constantValues:constants
        error:&error];
    if (!fn) {
        fprintf(stderr,
                "lgn2: Metal Laguna attention reduce/gate function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }
    id<MTLComputePipelineState> pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr,
                "lgn2: Metal Laguna attention reduce/gate pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    memo_dv = dv;
    memo_nwg = nwg;
    memo_pipeline = pipeline;
    return pipeline;
}

static uint32_t lgn2_gpu_flash_attn_vec_nsg(uint32_t n_keys, uint32_t nwg, uint32_t ncpsg) {
    uint32_t nsg = 1;
    while (2u * nwg * nsg * ncpsg < n_keys && nsg < 4u) {
        nsg *= 2u;
    }
    return nsg;
}

static int lgn2_gpu_trace_allocs(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = getenv("LGN2_METAL_TRACE_ALLOCS") != NULL;
        initialized = 1;
    }
    return enabled;
}

static double lgn2_gpu_mib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

static double lgn2_gpu_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

void lgn2_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
}

void lgn2_gpu_set_tensor_matmul_suppressed(bool suppressed) {
    g_tensor_matmul_suppressed = suppressed ? 1 : 0;
}

static int lgn2_gpu_model_map_log_enabled(void) {
    return 1;
}

static id<MTLBuffer> lgn2_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset);

static id<MTLBuffer> lgn2_gpu_wrap_model_exact_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset);

static const char *lgn2_gpu_source =
"#include <metal_stdlib>\n"
"#ifdef LGN2_METAL_HAS_TENSOR\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"#endif\n"
"using namespace metal;\n"
"#ifdef LGN2_METAL_HAS_TENSOR\n"
"using namespace mpp::tensor_ops;\n"
"#endif\n"
"\n"
"#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
"#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
"#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
"#define QK8_0 32\n"
"#ifndef QK_K\n"
"#define QK_K 256\n"
"#endif\n"
"#define N_SIMDWIDTH 32\n"
"#define N_R0_Q8_0 2\n"
"#define N_SG_Q8_0 4\n"
"#define FC_MUL_MV 600\n"
"#define FC_MUL_MM 700\n"
"#define FC_BIN 1300\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"#define M_PI_F 3.14159265358979323846f\n"
"\n"
"// Reads one byte per stride to warm model-backed pages without copying the\n"
"// model. This is outside inference and exists only to reduce first-use stalls.\n"
"kernel void kernel_touch_u8_stride(\n"
"        device const uchar    *src        [[buffer(0)]],\n"
"        device uchar          *dst        [[buffer(1)]],\n"
"        constant ulong        &stride     [[buffer(2)]],\n"
"        constant ulong        &bytes      [[buffer(3)]],\n"
"        constant ulong        &dst_offset [[buffer(4)]],\n"
"        uint gid [[thread_position_in_grid]]) {\n"
"    ulong off = (ulong)gid * stride;\n"
"    if (off >= bytes) return;\n"
"    dst[dst_offset + (ulong)gid] = src[off];\n"
"}\n"
"\n"
"enum lgn2_sort_order {\n"
"    LGN2_SORT_ORDER_ASC,\n"
"    LGN2_SORT_ORDER_DESC,\n"
"};\n"
"\n"
"struct block_q8_0 {\n"
"    half d;\n"
"    int8_t qs[QK8_0];\n"
"};\n"
"\n"
"struct block_q8_K {\n"
"    float d;\n"
"    int8_t qs[QK_K];\n"
"    int16_t bsums[QK_K / 16];\n"
"};\n"
"\n"
"\n";

static NSString *lgn2_gpu_full_source(void) {
    NSString *base = [NSString stringWithUTF8String:lgn2_gpu_source];
    NSFileManager *fm = [NSFileManager defaultManager];
    /*
     * Kernels are kept as separate files for review, then concatenated into one
     * Metal library.  A non-empty source override is authoritative for that
     * component: an unreadable path fails library construction instead of
     * silently compiling a different built-in file.  Empty/unset overrides
     * retain the normal built-in search path.
     */
    NSArray<NSArray<NSString *> *> *required_sources = @[
        @[@"LGN2_METAL_FLASH_ATTN_SOURCE", @"metal/flash_attn.metal"],
        @[@"LGN2_METAL_DENSE_SOURCE",      @"metal/dense.metal"],
        @[@"LGN2_METAL_MOE_SOURCE",        @"metal/moe.metal"],
        @[@"LGN2_METAL_UNARY_SOURCE",      @"metal/unary.metal"],
        @[@"LGN2_METAL_DSV4_MISC_SOURCE",  @"metal/dsv4_misc.metal"],
        @[@"LGN2_METAL_LAGUNA_SOURCE",     @"metal/laguna.metal"],
        @[@"LGN2_METAL_DFLASH_SOURCE",     @"metal/dflash.metal"],
        @[@"LGN2_METAL_ARGSORT_SOURCE",    @"metal/argsort.metal"],
        @[@"LGN2_METAL_CPY_SOURCE",        @"metal/cpy.metal"],
        @[@"LGN2_METAL_GET_ROWS_SOURCE",   @"metal/get_rows.metal"],
        @[@"LGN2_METAL_GLU_SOURCE",        @"metal/glu.metal"],
        @[@"LGN2_METAL_NORM_SOURCE",       @"metal/norm.metal"],
        @[@"LGN2_METAL_BIN_SOURCE",        @"metal/bin.metal"],
    ];

    NSMutableString *source = [NSMutableString stringWithString:base];
    for (NSArray<NSString *> *spec in required_sources) {
        const char *override_path = getenv([spec[0] UTF8String]);
        NSMutableArray<NSString *> *paths = [NSMutableArray array];
        const BOOL exclusive_override =
            override_path && override_path[0];
        if (override_path && override_path[0]) {
            /* A non-empty override is a source-selection contract, not a
             * preferred search path.  Falling through after a typo would
             * silently run a different kernel implementation. */
            [paths addObject:[NSString stringWithUTF8String:override_path]];
        }
        if (!exclusive_override) {
            [paths addObject:spec[1]];
            [paths addObject:[@"./" stringByAppendingString:spec[1]]];
        }

        NSString *loaded = nil;
        NSString *loaded_path = nil;
        for (NSString *path in paths) {
            if (![fm fileExistsAtPath:path]) continue;

            NSError *error = nil;
            loaded = [NSString stringWithContentsOfFile:path
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
            if (!loaded) {
                fprintf(stderr, "lgn2: failed to read Metal source %s: %s\n",
                        [path UTF8String], [[error localizedDescription] UTF8String]);
                return nil;
            }
            loaded_path = path;
            break;
        }

        if (!loaded) {
            if (exclusive_override) {
                fprintf(stderr,
                        "lgn2: Metal source override %s for %s is unreadable; "
                        "override is exclusive and built-in fallback is disabled\n",
                        override_path, [spec[0] UTF8String]);
            } else {
                fprintf(stderr,
                        "lgn2: Metal source %s not found (set %s to override)\n",
                        [spec[1] UTF8String], [spec[0] UTF8String]);
            }
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", loaded_path, loaded];
    }
    return source;
}

typedef struct {
    int32_t  ne00t;
    int32_t  ne00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} lgn2_gpu_get_rows_args;

typedef struct {
    int32_t  n_embd;
    int32_t  n_vocab;
    int32_t  n_tokens;
    uint64_t src_row_bytes;
    uint64_t dst_row_bytes;
    uint64_t token_stride;
} lgn2_gpu_get_rows_q8_0_args;

typedef struct {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} lgn2_gpu_cpy_args;

static lgn2_gpu_cpy_args lgn2_gpu_make_cpy_1d_args(
        uint32_t n,
        uint64_t src_elem,
        uint64_t dst_elem) {
    return (lgn2_gpu_cpy_args) {
        .nk0 = (int64_t)n,
        .ne00 = (int64_t)n,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = src_elem,
        .nb01 = (uint64_t)n * src_elem,
        .nb02 = (uint64_t)n * src_elem,
        .nb03 = (uint64_t)n * src_elem,
        .ne0 = (int64_t)n,
        .ne1 = 1,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = dst_elem,
        .nb1 = (uint64_t)n * dst_elem,
        .nb2 = (uint64_t)n * dst_elem,
        .nb3 = (uint64_t)n * dst_elem,
    };
}

static NSUInteger lgn2_gpu_cpy_threads(uint32_t n, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth = 32u;
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)n && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)n) nth = (NSUInteger)n;
    return nth ? nth : 1u;
}

static int lgn2_gpu_encode_cpy_f32_f32_3d_src_strided(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_col_stride,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride);

static int lgn2_gpu_encode_cpy_f32_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);

typedef struct {
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
} lgn2_gpu_glu_args;

typedef struct {
    uint32_t n;
} lgn2_gpu_add_flat_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    uint64_t o1[8];
} lgn2_gpu_bin_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    slope;
    float    scale;
    float    bias;
    float    val;
    float    min;
    float    max;
} lgn2_gpu_unary_args;



static int lgn2_gpu_encode_bin_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const lgn2_gpu_bin_args   *args,
        id<MTLBuffer>               a,
        NSUInteger                  a_off,
        id<MTLBuffer>               b,
        NSUInteger                  b_off,
        id<MTLBuffer>               out,
        NSUInteger                  out_off);

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} lgn2_gpu_q8_0_matvec_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} lgn2_gpu_mul_mm_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} lgn2_gpu_mul_mv_ext_args;

typedef lgn2_gpu_q8_0_matvec_args lgn2_gpu_f16_matvec_args;

static lgn2_gpu_q8_0_matvec_args lgn2_gpu_make_q8_0_mv_args(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row_bytes = (in_dim / 32u) * 34u;
    return (lgn2_gpu_q8_0_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = 34,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * sizeof(float),
        .nb13 = in_dim * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

static lgn2_gpu_f16_matvec_args lgn2_gpu_make_f16_mv_args(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row_bytes = in_dim * sizeof(uint16_t);
    return (lgn2_gpu_f16_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = sizeof(uint16_t),
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * sizeof(float),
        .nb13 = in_dim * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

static lgn2_gpu_q8_0_matvec_args lgn2_gpu_make_f32_mv_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_vec) {
    const uint64_t row_bytes = in_dim * sizeof(float);
    return (lgn2_gpu_q8_0_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_vec,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_vec * sizeof(float),
        .nb13 = in_dim * n_vec * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_vec,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

typedef struct {
    const char *function_name;
    int16_t     nsg;
    int32_t     nr0;
    NSUInteger  smem;
} lgn2_gpu_mv_dispatch;

static lgn2_gpu_mv_dispatch lgn2_gpu_make_q8_0_mv_dispatch(void) {
    /* The supported world-1 route uses the established NR4 geometry unless
     * the lifecycle-frozen NSG selector requests a supported override.  The
     * selector is parsed once at the Metal lifecycle boundary so changing
     * getenv() after one descriptor's first use cannot produce a mixed
     * dispatch configuration. */
    lgn2_gpu_q8_decode_config config;
    const int valid = lgn2_gpu_q8_decode_config_snapshot(&config);
    const int16_t nsg = valid < 0 ? 0 : (int16_t)(
        config.q8_mv_nsg_override > 0 ?
            config.q8_mv_nsg_override : 4);
    return (lgn2_gpu_mv_dispatch) {
        .function_name = "kernel_mul_mv_q8_0_f32",
        .nsg = nsg,
        .nr0 = 2,
        .smem = 32u * 2u * sizeof(float),
    };
}

static lgn2_gpu_mv_dispatch lgn2_gpu_make_plain_mv_dispatch(
        uint64_t in_dim,
        int      f32_weights) {
    if (in_dim < 32) {
        return (lgn2_gpu_mv_dispatch) {
            .function_name = f32_weights ? "kernel_mul_mv_f32_f32_short" : "kernel_mul_mv_f16_f32_short",
            .nsg = 1,
            .nr0 = 32,
            .smem = 0,
        };
    }

    const int16_t nsg = (int16_t)((in_dim + 127u) / 128u > 8u ? 8u : (in_dim + 127u) / 128u);
    const int use_4 = (in_dim % 4u) == 0;
    return (lgn2_gpu_mv_dispatch) {
        .function_name = f32_weights
            ? (use_4 ? "kernel_mul_mv_f32_f32_4" : "kernel_mul_mv_f32_f32")
            : (use_4 ? "kernel_mul_mv_f16_f32_4" : "kernel_mul_mv_f16_f32"),
        .nsg = nsg,
        .nr0 = 2,
        .smem = 32u * 2u * sizeof(float),
    };
}

#ifdef LGN2_TEST_HOOKS
int lgn2_gpu_test_q8_decode_nsg(void) {
    return lgn2_gpu_make_q8_0_mv_dispatch().nsg;
}
#endif

static lgn2_gpu_mul_mm_args lgn2_gpu_make_mm_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t row_bytes) {
    return (lgn2_gpu_mul_mm_args) {
        .ne00 = (int32_t)in_dim,
        .ne02 = 1,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static lgn2_gpu_mul_mv_ext_args lgn2_gpu_make_mv_ext_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t elem_bytes,
        uint64_t row_bytes) {
    return (lgn2_gpu_mul_mv_ext_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = elem_bytes,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_tok,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static int16_t lgn2_gpu_mv_ext_nxpsg(uint64_t in_dim, uint64_t n_tok) {
    if ((in_dim % 256u) == 0 && n_tok < 3) return 16;
    if ((in_dim % 128u) == 0) return 8;
    return 4;
}

static int16_t lgn2_gpu_mv_ext_r1ptg(uint64_t n_tok) {
    switch (n_tok) {
    case 2: return 2;
    case 3:
    case 6: return 3;
    case 4:
    case 7:
    case 8: return 4;
    case 5: return 5;
    default: return n_tok > 8 ? 4 : 0;
    }
}

static const char *lgn2_gpu_mv_ext_name(int q8, int16_t r1ptg) {
    if (q8) {
        switch (r1ptg) {
        case 2: return "kernel_mul_mv_ext_q8_0_f32_r1_2";
        case 3: return "kernel_mul_mv_ext_q8_0_f32_r1_3";
        case 4: return "kernel_mul_mv_ext_q8_0_f32_r1_4";
        case 5: return "kernel_mul_mv_ext_q8_0_f32_r1_5";
        default: return NULL;
        }
    }

    switch (r1ptg) {
    case 2: return "kernel_mul_mv_ext_f16_f32_r1_2";
    case 3: return "kernel_mul_mv_ext_f16_f32_r1_3";
    case 4: return "kernel_mul_mv_ext_f16_f32_r1_4";
    case 5: return "kernel_mul_mv_ext_f16_f32_r1_5";
    default: return NULL;
    }
}

static const char *lgn2_gpu_mv_ext_f32_name(int16_t r1ptg) {
    switch (r1ptg) {
    case 2: return "kernel_mul_mv_ext_f32_f32_r1_2";
    case 3: return "kernel_mul_mv_ext_f32_f32_r1_3";
    case 4: return "kernel_mul_mv_ext_f32_f32_r1_4";
    case 5: return "kernel_mul_mv_ext_f32_f32_r1_5";
    default: return NULL;
    }
}

static const char *lgn2_gpu_mv_ext_q8_pair_swiglu_name(int16_t r1ptg) {
    switch (r1ptg) {
    case 2: return "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_2";
    case 3: return "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_3";
    case 4: return "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_4";
    case 5: return "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_5";
    default: return NULL;
    }
}

typedef struct {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
} lgn2_gpu_rms_norm_args;

static lgn2_gpu_rms_norm_args lgn2_gpu_make_rms_norm_args(uint32_t n, uint32_t rows, float eps) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    return (lgn2_gpu_rms_norm_args) {
        .ne00 = (int32_t)n,
        .ne00_t = (int32_t)(n / 4u),
        .nb1 = row_bytes,
        .nb2 = row_bytes * rows,
        .nb3 = row_bytes * rows,
        .eps = eps,
        .nef1 = { (int32_t)rows, 1, 1 },
        .nef2 = { 1, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { row_bytes * rows, row_bytes, row_bytes },
        .nbf3 = { row_bytes * rows, row_bytes, row_bytes },
    };
}

static NSUInteger lgn2_gpu_rms_norm_threads(uint32_t n) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < 1024u) nth *= 2u;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

typedef struct {
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21;
    int32_t  ne20;
    uint64_t nb21;
} lgn2_gpu_mul_mm_id_map_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
    uint32_t reserved[3];
} lgn2_gpu_mul_mm_id_args;

/* Keep the host mirror byte-sized and position-compatible with the pre-world-1
 * source override.  The retired TP fields are now zeroed reserved storage. */
_Static_assert(offsetof(lgn2_gpu_mul_mm_id_args, reserved) == 92u,
               "routed mul_mm_id reserved tail moved");
_Static_assert(sizeof(lgn2_gpu_mul_mm_id_args) == 104u,
               "routed mul_mm_id ABI layout drifted");

static lgn2_gpu_mul_mm_id_map_args lgn2_gpu_make_mul_mm_id_map_args(
        uint32_t src0_cols,
        uint32_t src0_experts,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens);

static lgn2_gpu_mul_mm_id_args lgn2_gpu_make_mul_mm_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens);
static lgn2_gpu_mul_mm_id_args lgn2_gpu_make_mul_mm_id_args_src1_size(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t src1_elem_size);

static int lgn2_gpu_encode_mul_mm_id_map(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        const lgn2_gpu_mul_mm_id_map_args *map_args,
        const lgn2_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off);

static int lgn2_gpu_encode_mul_mm_id_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const lgn2_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes);
typedef struct {
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} lgn2_gpu_flash_attn_pad_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} lgn2_gpu_flash_attn_blk_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
    uint32_t laguna_stage_pos_mod;
    uint32_t laguna_stage_n_tokens;
    uint32_t laguna_stage_cache_cap;
    uint32_t laguna_stage_pad0;
    uint64_t laguna_stage_nb11;
    uint64_t laguna_stage_nb12;
} lgn2_gpu_flash_attn_vec_args;

/* Keep the ordinary host argument size aligned with the default Metal source;
 * older LGN2_METAL_FLASH_ATTN_SOURCE overrides may use the shorter historical
 * struct and tolerate these trailing bytes.  The virtual specialization uses
 * a distinct Metal argument type so the staged path never changes the generic
 * function-constant ABI. */
typedef struct {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
    uint32_t laguna_stage_pos_mod;
    uint32_t laguna_stage_n_tokens;
    uint32_t laguna_stage_cache_cap;
    uint32_t laguna_stage_pad0;
    uint64_t laguna_stage_nb11;
    uint64_t laguna_stage_nb12;
} lgn2_gpu_flash_attn_vec_virtual_args;

typedef struct {
    int32_t nrows;
} lgn2_gpu_flash_attn_reduce_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
} lgn2_gpu_kargs_argsort;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
} lgn2_gpu_kargs_argsort_merge;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} lgn2_gpu_kargs_sum_rows;

typedef struct {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
} lgn2_gpu_dsv4_router_select_one_args;

typedef struct {
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t stats_enabled;
} lgn2_gpu_glm_router_select_one_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float    eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t rope_atlas_family;
    uint32_t rope_atlas_stride;
} lgn2_gpu_laguna_norm_rope_args;

typedef struct {
    uint32_t cache_cap;
    uint32_t cache_row;
    uint32_t n_head_kv;
    uint32_t head_dim;
} lgn2_gpu_laguna_kv_store_args;

typedef struct {
    uint32_t in_dim;
    uint32_t q_dim;
    uint32_t kv_dim;
    uint32_t gate_dim;
} lgn2_gpu_laguna_qkvg_args;

typedef struct {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t cache_cap;
    uint32_t key_start;
    uint32_t key_count;
    float    scale;
    uint32_t pad0;
} lgn2_gpu_laguna_attention_args;

typedef struct {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t key_count0;
    uint32_t n_tokens;
    uint32_t nsg;
    uint32_t nwg;
    float    scale;
} lgn2_gpu_laguna_gqa3_decode_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    float    scale;
    uint32_t pad0;
} lgn2_gpu_laguna_prefill_attention_args;

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_tokens;
    uint32_t pad0;
    uint64_t row_bytes;
} lgn2_gpu_laguna_q6_matmul_args;

typedef struct {
    uint32_t in_dim;
    uint32_t mid_dim;
    uint32_t out_dim;
    uint32_t n_total_expert;
    uint32_t n_expert_used;
    uint32_t n_tokens;
    uint32_t mid_token_stride;
    uint32_t down_type;
    uint32_t reserved[4];
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t up_expert_bytes;
    uint64_t up_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
} lgn2_gpu_glm_routed_moe_args;

_Static_assert(offsetof(lgn2_gpu_glm_routed_moe_args, reserved) == 32u,
               "routed Laguna MoE reserved prefix moved");
_Static_assert(offsetof(lgn2_gpu_glm_routed_moe_args, gate_expert_bytes) == 48u,
               "routed Laguna MoE stride moved");
_Static_assert(sizeof(lgn2_gpu_glm_routed_moe_args) == 96u,
               "routed Laguna MoE ABI layout drifted");

typedef struct {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float    clamp_value;
} lgn2_gpu_dsv4_moe_swiglu_weight_args;

typedef struct {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
} lgn2_gpu_dsv4_moe_sum_args;

/* Compile the single in-repo Metal source and create the pipelines that every
 * session uses. Shape-dependent kernels with function constants are built
 * lazily by the small lgn2_gpu_get_* caches, so startup stays predictable
 * while long-context prefill and decode can still pick specialized variants.
 *
 * Keep this as an implementation function: the public wrapper below owns the
 * one failure exit and therefore unwinds every one of the many shader/pipeline
 * construction failures through the same idempotent cleanup path. */
static int lgn2_gpu_init_impl(void) {
    if (g_initialized) return 1;

#ifdef LGN2_TEST_HOOKS
    g_metal_moe_abi_sentinel_status = -1;
#endif

    /* Freeze the Q8 decode selector before any Metal graph or command buffer
     * can be admitted.  This is deliberately a process boundary:
     * lgn2_gpu_cleanup() releases Metal resources but does not reopen the
     * environment snapshot for a later in-process engine. */
    if (lgn2_gpu_q8_decode_config_snapshot(NULL) < 0) return 0;

    @autoreleasepool {
        /* A failed init must not leave a stale lifecycle snapshot visible to
         * a later raw preflight after the caller changes the environment. */
        g_laguna_swa_selectors_snapshot_valid = 0;
        if (!lgn2_gpu_snapshot_lifecycle_selectors()) return 0;
        if (g_direct_kv_prefill_mode < 0) {
            fprintf(stderr,
                    "lgn2: invalid LGN2_METAL_LAGUNA_DIRECT_KV_PREFILL; "
                    "expected unset, empty, 0, or literal 1\n");
            return 0;
        }
#ifdef LGN2_TEST_HOOKS
        g_laguna_test_route_hooks = 0;
        g_laguna_test_direct_kv_count = 0;
        g_laguna_test_wrap_kv_count = 0;
        g_laguna_test_fused_q8_count = 0;
        g_laguna_test_stock_q8_count = 0;
        g_laguna_test_fused_q8_bco_false_count = 0;
        g_laguna_test_fused_q8_bco_true_count = 0;
        memset(g_laguna_test_decode_route_counts, 0,
               sizeof(g_laguna_test_decode_route_counts));
        memset(g_laguna_test_decode_route_batch, 0,
               sizeof(g_laguna_test_decode_route_batch));
        memset(g_laguna_test_decode_route_inflight, 0,
               sizeof(g_laguna_test_decode_route_inflight));
        g_test_glm_grouped_moe_encoded_dispatches = 0;
        g_test_glm_grouped_moe_batch_dispatches = 0;
        g_test_glm_grouped_moe_owned_dispatches = 0;
        g_test_glm_grouped_moe_completed_dispatches = 0;
        g_test_glm_exact_q4_encoded_dispatches = 0;
        g_test_glm_exact_q4_completed_dispatches = 0;
#endif
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "lgn2: Metal device not available\n");
            return 0;
        }
#ifdef LGN2_TEST_HOOKS
        if (lgn2_gpu_test_init_should_fail(LGN2_GPU_TEST_INIT_FAIL_DEVICE)) {
            return 0;
        }
#endif
        lgn2_gpu_print_device_summary();
        lgn2_gpu_detect_metal4_features();

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "lgn2: failed to create Metal command queue\n");
            g_device = nil;
            return 0;
        }
#ifdef LGN2_TEST_HOOKS
        if (lgn2_gpu_test_init_should_fail(LGN2_GPU_TEST_INIT_FAIL_QUEUE)) {
            return 0;
        }
#endif
        g_model_buffer_cache = [NSMutableDictionary dictionary];
        g_model_buffer_cache_bytes = 0;
        g_model_buffer_cache_evictions = 0;
        g_model_buffer_cache_over_limit = 0;
        g_pipeline_cache = [NSMutableDictionary dictionary];
        g_pending_cbs = [NSMutableArray array];
        g_pending_laguna_atlas_evidence = [NSMutableArray array];
#ifdef LGN2_TEST_HOOKS
        g_pending_glm_grouped_moe_evidence = [NSMutableArray array];
#endif
        if (!g_model_buffer_cache || !g_pipeline_cache ||
            !g_pending_cbs ||
            !g_pending_laguna_atlas_evidence
#ifdef LGN2_TEST_HOOKS
            || !g_pending_glm_grouped_moe_evidence
#endif
        ) {
            fprintf(stderr, "lgn2: Metal bookkeeping allocation failed\n");
            g_pending_cbs = nil;
            g_pending_laguna_atlas_evidence = nil;
#ifdef LGN2_TEST_HOOKS
            g_pending_glm_grouped_moe_evidence = nil;
#endif
            g_pipeline_cache = nil;
            g_model_buffer_cache = nil;
            g_queue = nil;
            g_device = nil;
            return 0;
        }
#ifdef LGN2_TEST_HOOKS
        if (lgn2_gpu_test_init_should_fail(
                LGN2_GPU_TEST_INIT_FAIL_BOOKKEEPING)) {
            return 0;
        }
#endif

        NSError *error = nil;
        NSString *source = lgn2_gpu_full_source();
        if (!source) {
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        NSMutableDictionary *macros = [NSMutableDictionary new];
        if (g_metal4_tensor_api_enabled) {
            macros[@"LGN2_METAL_HAS_TENSOR"] = @"1";
            fprintf(stderr, "lgn2: Metal 4 tensor API enabled for Tensor kernels\n");
        }

        const int drift_norm_unify       = lgn2_gpu_env_bool("LGN2_METAL_NORM_RSQRT_DISABLE") != 0; // default ON
        const int drift_rope_exp2_log2   = lgn2_gpu_env_bool("LGN2_METAL_ROPE_EXP2_LOG2")     >  0; // default OFF
        const int drift_math_safe        = lgn2_gpu_env_bool("LGN2_METAL_MATH_SAFE")          >  0; // default OFF

        if (drift_math_safe) {
            // MTLCompileOptions.fastMathEnabled defaults to YES and Apple's
            // headers explicitly say this "may violate the IEEE 754 standard".
            // Different fast-math optimizations get applied across the
            // matmul2d cooperative-tensor path and the legacy
            // simdgroup_multiply_accumulate path on M5, amplifying the
            // mismatch. MTLMathModeSafe pins the entire library to strict
            // IEEE-754 semantics. Diagnostic-only: useful to localize drift
            // sources but not to ship as a default.
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
                fprintf(stderr, "lgn2: Metal shader library math mode = safe (strict IEEE-754) by LGN2_METAL_MATH_SAFE\n");
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
                fprintf(stderr, "lgn2: Metal shader library fast-math disabled by LGN2_METAL_MATH_SAFE (pre-macOS 15)\n");
            }
        }

        if (drift_norm_unify)     macros[@"LGN2_METAL_NORM_RSQRT_DISABLE"] = @"1";
        if (drift_rope_exp2_log2) macros[@"LGN2_METAL_ROPE_EXP2_LOG2"]     = @"1";
        fprintf(stderr,
                "lgn2: drift-patch flags norm_unify=%s rope_exp2_log2=%s math_safe=%s tensor_matmul=%s\n",
                drift_norm_unify     ? "on"  : "off",
                drift_rope_exp2_log2 ? "on"  : "off",
                drift_math_safe      ? "on"  : "off",
                g_metal4_tensor_api_enabled ? "on" : "off");
        options.preprocessorMacros = macros;
        id<MTLLibrary> library = [g_device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            fprintf(stderr, "lgn2: Metal shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_library = library;
#ifdef LGN2_TEST_HOOKS
        g_metal_moe_abi_sentinel_status =
            [library newFunctionWithName:
                @"kernel_laguna_moe_abi_v2_mulmmid104_routed96_stride48"] ? 1 : 0;
#else
        id<MTLFunction> moe_abi_sentinel =
            [library newFunctionWithName:
                @"kernel_laguna_moe_abi_v2_mulmmid104_routed96_stride48"];
#endif
        if (
#ifdef LGN2_TEST_HOOKS
            g_metal_moe_abi_sentinel_status == 0
#else
            !moe_abi_sentinel
#endif
        ) {
            fprintf(stderr,
                    "lgn2: Metal MoE source ABI sentinel "
                    "kernel_laguna_moe_abi_v2_mulmmid104_routed96_stride48 missing; "
                    "LGN2_METAL_MOE_SOURCE is stale or incompatible\n");
            g_library = nil;
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_metal_math_safe = drift_math_safe;

        id<MTLFunction> fn = [library newFunctionWithName:@"kernel_get_rows_f32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_get_rows_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_f32_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_get_rows_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
#ifdef LGN2_TEST_HOOKS
        if (lgn2_gpu_test_init_should_fail(
                LGN2_GPU_TEST_INIT_FAIL_FIRST_PIPELINE)) {
            return 0;
        }
#endif

        fn = [library newFunctionWithName:@"kernel_get_rows_i32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_get_rows_i32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_i32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_i32_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_get_rows_i32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_q8_0_f32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_get_rows_q8_0_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_q8_0_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_q8_0_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_get_rows_q8_0_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_q4_0_f32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_get_rows_q4_0_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_q4_0_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_q4_0_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_get_rows_q4_0_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_q4_K_f32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_get_rows_q4_K_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_q4_K_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_q4_K_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_get_rows_q4_K_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f32_f32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_cpy_f32_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f32_f32_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_cpy_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f32_f16"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_cpy_f32_f16 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f32_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f32_f16_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_cpy_f32_f16 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_contig_f32_f16_4"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_cpy_contig_f32_f16_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_contig_f32_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_contig_f32_f16_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_cpy_contig_f32_f16_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_swiglu_f32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_swiglu_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_swiglu_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_swiglu_flat_f32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_swiglu_flat_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        g_swiglu_flat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_swiglu_flat_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_swiglu_flat_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_moe_sum8_f32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_dsv4_moe_sum8_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_moe_sum8_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_sum8_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_dsv4_moe_sum8_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_moe_sum10_f32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_dsv4_moe_sum10_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_moe_sum10_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_sum10_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_dsv4_moe_sum10_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_op = 0;
        int16_t bin_f = 1;
        bool bin_rb = false;
        bool bin_cb = false;
        [bin_constants setConstantValue:&bin_op type:MTLDataTypeShort atIndex:1300];
        [bin_constants setConstantValue:&bin_f  type:MTLDataTypeShort atIndex:1301];
        [bin_constants setConstantValue:&bin_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_constants setConstantValue:&bin_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_bin_fuse_f32_f32_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_add_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_add_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_bin_fuse_f32_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_add2_f32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_add2_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        g_add2_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_add2_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_add2_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_add3_f32"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_add3_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        g_add3_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_add3_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_add3_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_mul_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_mul_plain_op = 2;
        int16_t bin_mul_plain_f = 1;
        bool bin_mul_plain_rb = false;
        bool bin_mul_plain_cb = false;
        [bin_mul_constants setConstantValue:&bin_mul_plain_op type:MTLDataTypeShort atIndex:1300];
        [bin_mul_constants setConstantValue:&bin_mul_plain_f  type:MTLDataTypeShort atIndex:1301];
        [bin_mul_constants setConstantValue:&bin_mul_plain_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_mul_constants setConstantValue:&bin_mul_plain_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_mul_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_bin_fuse_f32_f32_f32 mul function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_mul_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_mul_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_bin_fuse_f32_f32_f32 mul pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_mul_scalar_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_mul_op = 2;
        int16_t bin_mul_f = 1;
        bool bin_mul_rb = false;
        bool bin_mul_cb = true;
        [bin_mul_scalar_constants setConstantValue:&bin_mul_op type:MTLDataTypeShort atIndex:1300];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_f  type:MTLDataTypeShort atIndex:1301];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_mul_scalar_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_bin_mul_scalar_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_bin_mul_scalar_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_div_row_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_div_op = 3;
        int16_t bin_div_f = 1;
        bool bin_div_rb = false;
        bool bin_div_cb = true;
        [bin_div_row_constants setConstantValue:&bin_div_op type:MTLDataTypeShort atIndex:1300];
        [bin_div_row_constants setConstantValue:&bin_div_f  type:MTLDataTypeShort atIndex:1301];
        [bin_div_row_constants setConstantValue:&bin_div_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_div_row_constants setConstantValue:&bin_div_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_div_row_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_bin_fuse_f32_f32_f32 div-row function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_bin_div_row_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_bin_div_row_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_bin_fuse_f32_f32_f32 div-row pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_rms_norm_mul_f32_4"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_rms_norm_mul_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rms_norm_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_rms_norm_mul_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_add_rms_norm_mul_f32_4"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_add_rms_norm_mul_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_add_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_add_rms_norm_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_add_rms_norm_mul_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_argsort_f32_i32_desc"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_argsort_f32_i32_desc function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_argsort_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_argsort_f32_i32_desc_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_argsort_f32_i32_desc pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_argsort_merge_f32_i32_desc"];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_argsort_merge_f32_i32_desc function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_argsort_merge_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_argsort_merge_f32_i32_desc_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_argsort_merge_f32_i32_desc pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *sum_rows_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t sum_rows_op = 10;
        [sum_rows_constants setConstantValue:&sum_rows_op type:MTLDataTypeShort atIndex:1400];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_sum_rows_f32_f32"
                           constantValues:sum_rows_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_sum_rows_f32_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_sum_rows_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_sum_rows_f32_f32_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_sum_rows_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_sigmoid_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_sigmoid_op = 102;
        bool unary_cnt = false;
        [unary_sigmoid_constants setConstantValue:&unary_sigmoid_op type:MTLDataTypeShort atIndex:1200];
        [unary_sigmoid_constants setConstantValue:&unary_cnt        type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_sigmoid_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 sigmoid function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_sigmoid_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_sigmoid_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 sigmoid pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_silu_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_silu_op = 106;
        [unary_silu_constants setConstantValue:&unary_silu_op type:MTLDataTypeShort atIndex:1200];
        [unary_silu_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_silu_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 silu function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_silu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_silu_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 silu pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_softplus_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_softplus_op = 115;
        [unary_softplus_constants setConstantValue:&unary_softplus_op type:MTLDataTypeShort atIndex:1200];
        [unary_softplus_constants setConstantValue:&unary_cnt         type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_softplus_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 softplus function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_softplus_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_softplus_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 softplus pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_sqrt_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_sqrt_op = 14;
        [unary_sqrt_constants setConstantValue:&unary_sqrt_op type:MTLDataTypeShort atIndex:1200];
        [unary_sqrt_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_sqrt_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 sqrt function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_sqrt_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_sqrt_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 sqrt pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_clamp_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_clamp_op = 12;
        [unary_clamp_constants setConstantValue:&unary_clamp_op type:MTLDataTypeShort atIndex:1200];
        [unary_clamp_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32"
                           constantValues:unary_clamp_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32 clamp function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_clamp_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_clamp_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32 clamp pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_scale_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_scale_op = 10;
        [unary_scale_constants setConstantValue:&unary_scale_op type:MTLDataTypeShort atIndex:1200];
        [unary_scale_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_scale_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 scale function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_scale_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_scale_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 scale pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_fill_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_fill_op = 11;
        [unary_fill_constants setConstantValue:&unary_fill_op type:MTLDataTypeShort atIndex:1200];
        [unary_fill_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_fill_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 fill function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_fill_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_fill_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f32_f32_4 fill pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f16_f16"
                           constantValues:unary_fill_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f16_f16 fill function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_fill_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_fill_f16_pipeline) {
            fprintf(stderr, "lgn2: Metal kernel_unary_f16_f16 fill pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_dsv4_softplus_sqrt_pipeline =
            lgn2_gpu_get_pipeline("kernel_dsv4_softplus_sqrt_f32_4");
        g_dsv4_router_finalize_one_pipeline =
            lgn2_gpu_get_pipeline("kernel_dsv4_router_finalize_one");
        g_dsv4_router_finalize_one_simd_pipeline =
            lgn2_gpu_get_pipeline("kernel_dsv4_router_finalize_one_simd");
        g_dsv4_router_finalize_weights_one_simd_pipeline =
            lgn2_gpu_get_pipeline("kernel_dsv4_router_finalize_weights_one_simd");
        g_dsv4_router_transform_finalize_weights_one_simd_pipeline =
            lgn2_gpu_get_pipeline(
                "kernel_dsv4_router_transform_finalize_weights_one_simd");
        g_dsv4_router_weights_one_pipeline =
            lgn2_gpu_get_pipeline("kernel_dsv4_router_weights_one");
        g_glm_router_select_one_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_router_select_one");
        g_glm_q4_k_pair_swiglu_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu_f32");
        g_glm_q4_k_pair_swiglu2_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu2_f32");
        g_glm_q4_k_pair_swiglu4_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu4_f32");
        g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu2_mapped_f32");
        g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu2_mapped_row_f32");
        g_glm_q2_k_pair_swiglu_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q2_K_pair_swiglu_f32");
        g_glm_q2_k_pair_swiglu_r1_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q2_K_pair_swiglu_r1_f32");
        g_glm_q3_k_pair_swiglu_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q3_K_pair_swiglu_f32");
        g_glm_q3_k_pair_swiglu_r1_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q3_K_pair_swiglu_r1_f32");
        g_glm_q2_k_down_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q2_K_down_f32");
        g_glm_q2_k_down_r1_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q2_K_down_r1_f32");
        g_glm_q3_k_down_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q3_K_down_f32");
        g_glm_q3_k_down_r1_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q3_K_down_r1_f32");
        g_glm_q4_k_down_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q4_K_down_simd_f32");
        g_glm_q4_k_down_r1_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q4_K_down_r1_simd_f32");
        g_glm_q5_k_pair_swiglu_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q5_K_pair_swiglu_f32");
        g_glm_q5_k_pair_swiglu_mapped_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q5_K_pair_swiglu_mapped_f32");
        g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q5_K_pair_swiglu_mapped_row_f32");
        g_glm_q5_k_down_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q5_K_down_f32");
        g_glm_q6_k_down_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_q6_K_down_f32");
        g_laguna_routed_shared_pair_pipeline =
            lgn2_gpu_get_pipeline(
                "kernel_laguna_q4_K_routed_shared_pair_swiglu_f32");
        g_laguna_routed_shared_q4_down_pipeline =
            lgn2_gpu_get_pipeline(
                "kernel_laguna_q4_K_routed_shared_down_f32");
        g_laguna_routed_shared_q6_down_pipeline =
            lgn2_gpu_get_pipeline(
                "kernel_laguna_q6_K_routed_shared_down_f32");
        g_laguna_head_norm_rope_pipeline =
            lgn2_gpu_get_pipeline("kernel_laguna_head_rms_norm_rope_neox");
        g_laguna_qk_head_norm_rope_pipeline =
            lgn2_gpu_get_pipeline("kernel_laguna_qk_head_rms_norm_rope_neox");
        g_laguna_store_kv_pipeline =
            lgn2_gpu_get_pipeline("kernel_laguna_store_kv_f16");
        g_laguna_attention_pipeline =
            lgn2_gpu_get_pipeline("kernel_laguna_attention_decode_gqa_f16");
        g_laguna_stage_kv_pipeline =
            lgn2_gpu_get_pipeline("kernel_laguna_stage_kv_f16");
        g_laguna_prefill_attention_pipeline =
            lgn2_gpu_get_pipeline("kernel_laguna_attention_prefill_gqa_f16");
        g_laguna_commit_kv_pipeline =
            lgn2_gpu_get_pipeline("kernel_laguna_commit_kv_f16");
        g_laguna_q6_k_matmul_pipeline =
            lgn2_gpu_get_pipeline("kernel_laguna_q6_K_matmul_f32");
        g_laguna_argmax_f32_pipeline =
            lgn2_gpu_get_pipeline("kernel_laguna_argmax_f32");
        g_dsv4_router_weights_batch_pipeline =
            lgn2_gpu_get_pipeline("kernel_dsv4_router_weights_batch");
        if (!g_dsv4_softplus_sqrt_pipeline ||
            !g_dsv4_router_finalize_one_pipeline ||
            !g_dsv4_router_weights_one_pipeline ||
            !g_glm_router_select_one_pipeline ||
            !g_glm_q4_k_pair_swiglu_f32_pipeline ||
            !g_glm_q4_k_pair_swiglu2_f32_pipeline ||
            !g_glm_q4_k_pair_swiglu4_f32_pipeline ||
            !g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline ||
            !g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline ||
            !g_glm_q2_k_pair_swiglu_f32_pipeline ||
            !g_glm_q2_k_pair_swiglu_r1_f32_pipeline ||
            !g_glm_q3_k_pair_swiglu_f32_pipeline ||
            !g_glm_q3_k_pair_swiglu_r1_f32_pipeline ||
            !g_glm_q2_k_down_f32_pipeline ||
            !g_glm_q2_k_down_r1_f32_pipeline ||
            !g_glm_q3_k_down_f32_pipeline ||
            !g_glm_q3_k_down_r1_f32_pipeline ||
            !g_glm_q4_k_down_f32_pipeline ||
            !g_glm_q4_k_down_r1_f32_pipeline ||
            !g_glm_q5_k_pair_swiglu_f32_pipeline ||
            !g_glm_q5_k_pair_swiglu_mapped_f32_pipeline ||
            !g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline ||
            !g_glm_q5_k_down_f32_pipeline ||
            !g_glm_q6_k_down_f32_pipeline ||
            !g_laguna_routed_shared_pair_pipeline ||
            !g_laguna_routed_shared_q4_down_pipeline ||
            !g_laguna_routed_shared_q6_down_pipeline ||
            !g_laguna_head_norm_rope_pipeline ||
            !g_laguna_qk_head_norm_rope_pipeline ||
            !g_laguna_store_kv_pipeline ||
            !g_laguna_attention_pipeline ||
            !g_laguna_stage_kv_pipeline ||
            !g_laguna_prefill_attention_pipeline ||
            !g_laguna_commit_kv_pipeline ||
            !g_laguna_q6_k_matmul_pipeline) {
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_initialized = 1;
    }

    return 1;
}

int lgn2_gpu_init(void) {
    if (g_initialized) return 1;

    const int ok = lgn2_gpu_init_impl();
    if (!ok) {
        /* This also clears selector snapshots captured before the first Metal
         * allocation.  Callers can retry without inheriting any device,
         * queue, library, pipeline, view, cache, or selector state. */
        lgn2_gpu_cleanup();
    }
    return ok;
}



lgn2_gpu_tensor *lgn2_gpu_tensor_alloc(uint64_t bytes) {
    if (!g_initialized && !lgn2_gpu_init()) return NULL;
    if (bytes == 0 || bytes > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        LGN2MetalTensor *tensor = [LGN2MetalTensor new];
        tensor.buffer = [g_device newBufferWithLength:(NSUInteger)bytes
                                              options:MTLResourceStorageModeShared];
        if (!tensor.buffer) {
            return NULL;
        }
        tensor.offset = 0;
        tensor.bytes = bytes;
        tensor.owner = 1;
        uint64_t live_snap = 0;
        uint64_t peak_snap = 0;
        pthread_mutex_lock(&g_tensor_mu);
        const int tracked = lgn2_gpu_tensor_track_alloc_locked(
                (__bridge const void *)tensor,
                bytes,
                &live_snap,
                &peak_snap);
        pthread_mutex_unlock(&g_tensor_mu);
        if (!tracked) {
            fprintf(stderr, "lgn2: failed to track Metal tensor allocation\n");
            tensor.buffer = nil;
            return NULL;
        }
        if (lgn2_gpu_trace_allocs()) {
            fprintf(stderr,
                    "lgn2: Metal tensor alloc %.3f MiB live %.3f MiB peak %.3f MiB\n",
                    (double)bytes / (1024.0 * 1024.0),
                    (double)live_snap / (1024.0 * 1024.0),
                    (double)peak_snap / (1024.0 * 1024.0));
        }
        return (__bridge_retained lgn2_gpu_tensor *)tensor;
    }
}

lgn2_gpu_tensor *lgn2_gpu_tensor_view(const lgn2_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base) return NULL;
    const LGN2MetalTensor *base_obj = lgn2_gpu_tensor_const_obj(base);
    if (offset > base_obj.bytes || bytes > base_obj.bytes - offset) return NULL;
    if (base_obj.offset > UINT64_MAX - offset) return NULL;
    const uint64_t absolute_offset = base_obj.offset + offset;
    if (absolute_offset > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        LGN2MetalTensor *view = [LGN2MetalTensor new];
        view.buffer = base_obj.buffer;
        view.offset = absolute_offset;
        view.bytes = bytes;
        view.owner = 0;
        pthread_mutex_lock(&g_tensor_mu);
        const int tracked = lgn2_gpu_tensor_track_view_locked((__bridge const void *)view);
        pthread_mutex_unlock(&g_tensor_mu);
        if (!tracked) {
            fprintf(stderr, "lgn2: failed to track Metal tensor view\n");
            view.buffer = nil;
            return NULL;
        }
        return (__bridge_retained lgn2_gpu_tensor *)view;
    }
}

void lgn2_gpu_tensor_free(lgn2_gpu_tensor *tensor) {
    if (!tensor) return;
    @autoreleasepool {
        uint8_t owner = 0;
        uint64_t bytes = 0;
        uint64_t live_snap = 0;
        uint64_t peak_snap = 0;
        if (!lgn2_gpu_tensor_prepare_free(tensor,
                                         &owner,
                                         &bytes,
                                         &live_snap,
                                         &peak_snap)) {
            return;
        }
        LGN2MetalTensor *obj = (__bridge_transfer LGN2MetalTensor *)tensor;
        if (owner) {
            if (lgn2_gpu_trace_allocs()) {
                fprintf(stderr,
                        "lgn2: Metal tensor free %.3f MiB live %.3f MiB peak %.3f MiB\n",
                        (double)bytes / (1024.0 * 1024.0),
                        (double)live_snap / (1024.0 * 1024.0),
                        (double)peak_snap / (1024.0 * 1024.0));
            }
        }
        obj.buffer = nil;
        obj.offset = 0;
        obj.bytes = 0;
        obj.owner = 0;
    }
}

uint64_t lgn2_gpu_tensor_bytes(const lgn2_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const LGN2MetalTensor *obj = lgn2_gpu_tensor_const_obj(tensor);
    return obj.bytes;
}

void *lgn2_gpu_tensor_contents(lgn2_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    LGN2MetalTensor *obj = lgn2_gpu_tensor_obj(tensor);
    return (uint8_t *)[obj.buffer contents] + obj.offset;
}

int lgn2_gpu_tensor_fill_f32(lgn2_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > lgn2_gpu_tensor_bytes(tensor) / sizeof(float)) return 0;
    float *p = lgn2_gpu_tensor_contents(tensor);
    if (!p && count != 0) return 0;
    for (uint64_t i = 0; i < count; i++) p[i] = value;
    return 1;
}

int lgn2_gpu_tensor_write(lgn2_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    LGN2MetalTensor *obj = lgn2_gpu_tensor_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    if (bytes != 0) {
        memcpy((uint8_t *)[obj.buffer contents] + obj.offset + offset, data, (size_t)bytes);
    }
    return 1;
}

int lgn2_gpu_tensor_read(const lgn2_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    const LGN2MetalTensor *obj = lgn2_gpu_tensor_const_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    if (bytes != 0) {
        memcpy(data, (const uint8_t *)[obj.buffer contents] + obj.offset + offset, (size_t)bytes);
    }
    return 1;
}

int lgn2_gpu_tensor_copy(lgn2_gpu_tensor *dst, uint64_t dst_offset,
                          const lgn2_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes) {
    if (!dst || !src) return 0;
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    LGN2MetalTensor *d = lgn2_gpu_tensor_obj(dst);
    const LGN2MetalTensor *s = lgn2_gpu_tensor_const_obj(src);
    if (dst_offset > d.bytes || bytes > d.bytes - dst_offset) return 0;
    if (src_offset > s.bytes || bytes > s.bytes - src_offset) return 0;
    if (bytes == 0) return 1;
    if (!g_batch_cb) return 0;

    lgn2_gpu_close_batch_encoder();
    g_batch_has_work = YES;
    id<MTLBlitCommandEncoder> blit = [g_batch_cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:s.buffer
            sourceOffset:(NSUInteger)(s.offset + src_offset)
                toBuffer:d.buffer
       destinationOffset:(NSUInteger)(d.offset + dst_offset)
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    return 1;
}

int lgn2_gpu_tensor_copy_f32_to_f16(lgn2_gpu_tensor *dst, uint64_t dst_offset,
                                   const lgn2_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count) {
    if (!dst || !src) return 0;
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    LGN2MetalTensor *d = lgn2_gpu_tensor_obj(dst);
    const LGN2MetalTensor *s = lgn2_gpu_tensor_const_obj(src);
    if (count == 0) return 1;
    if (count > UINT64_MAX / sizeof(float) ||
        count > UINT64_MAX / sizeof(uint16_t)) {
        return 0;
    }
    const uint64_t src_bytes = count * sizeof(float);
    const uint64_t dst_bytes = count * sizeof(uint16_t);
    if (src_offset > s.bytes || src_bytes > s.bytes - src_offset ||
        dst_offset > d.bytes || dst_bytes > d.bytes - dst_offset) {
        return 0;
    }

    @autoreleasepool {
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        uint64_t done = 0;
        int ok = 1;
        while (done < count && ok) {
            uint64_t chunk64 = count - done;
            if (chunk64 > UINT32_MAX) chunk64 = UINT32_MAX;
            const uint32_t chunk = (uint32_t)chunk64;
            ok = lgn2_gpu_encode_cpy_f32_f16_1d(
                    cb,
                    s.buffer,
                    (NSUInteger)(s.offset + src_offset + done * sizeof(float)),
                    d.buffer,
                    (NSUInteger)(d.offset + dst_offset + done * sizeof(uint16_t)),
                    chunk);
            done += chunk;
        }
        if (ok) ok = lgn2_gpu_finish_command_buffer(cb, owned, "tensor f32 to f16 copy");
        return ok;
    }
}

int lgn2_gpu_pack_slot_rows_f32_tensor(
        lgn2_gpu_tensor       *out,
        const lgn2_gpu_tensor *slots,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !slots || n_rows == 0 || width == 0 || n_slots == 0 ||
        slot_cap == 0 || n_rows > slot_cap) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> slotsbuf = lgn2_gpu_tensor_buffer(slots);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        uint64_t row_bytes = 0;
        uint64_t slot_plane_bytes = 0;
        uint64_t slots_bytes = 0;
        uint64_t out_rows = 0;
        uint64_t out_bytes = 0;
        if ((uint64_t)width > UINT64_MAX / sizeof(float)) return 0;
        row_bytes = (uint64_t)width * sizeof(float);
        if ((uint64_t)slot_cap > UINT64_MAX / row_bytes) return 0;
        slot_plane_bytes = (uint64_t)slot_cap * row_bytes;
        if ((uint64_t)n_slots > UINT64_MAX / slot_plane_bytes) return 0;
        slots_bytes = (uint64_t)n_slots * slot_plane_bytes;
        if ((uint64_t)n_rows > UINT64_MAX / n_slots) return 0;
        out_rows = (uint64_t)n_rows * n_slots;
        if (out_rows > UINT64_MAX / row_bytes) return 0;
        out_bytes = out_rows * row_bytes;
        if (!slotsbuf || !outbuf ||
            lgn2_gpu_tensor_bytes(slots) < slots_bytes ||
            lgn2_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "lgn2: Metal slot-row pack received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        if (!lgn2_gpu_encode_cpy_f32_f32_3d_src_strided(cb,
                                                        slotsbuf,
                                                        lgn2_gpu_tensor_offset(slots),
                                                        outbuf,
                                                        lgn2_gpu_tensor_offset(out),
                                                        width,
                                                        n_slots,
                                                        n_rows,
                                                        sizeof(float),
                                                        slot_plane_bytes,
                                                        row_bytes,
                                                        row_bytes,
                                                        (uint64_t)n_slots * row_bytes)) {
            return 0;
        }
        if (!lgn2_gpu_finish_command_buffer(cb, owned, "slot-row pack")) return 0;
    }

    return 1;
}

int lgn2_gpu_begin_commands(void) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (g_batch_cb) return 0;
    g_batch_cb = lgn2_gpu_new_command_buffer();
    g_batch_has_work = NO;
    lgn2_gpu_laguna_atlas_evidence_zero(&g_batch_laguna_atlas_evidence);
    if (g_batch_cb) {
        if (g_command_batch_epoch != UINT64_MAX) g_command_batch_epoch++;
    }
    return g_batch_cb != nil;
}

int lgn2_gpu_flush_encoder(void) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!g_batch_cb) return 0;
    lgn2_gpu_close_batch_encoder();
    return 1;
}

int lgn2_gpu_flush_commands(void) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!g_batch_cb) return 0;

    lgn2_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    g_batch_has_work = NO;
#ifdef LGN2_TEST_HOOKS
    g_laguna_router_fused_pending_dispatches +=
        g_laguna_router_fused_batch_dispatches;
    g_laguna_router_fused_batch_dispatches = 0;
    lgn2_gpu_laguna_test_decode_route_batch_committed();
#endif
    [cb commit];
    lgn2_gpu_laguna_atlas_register_pending(cb, &g_batch_laguna_atlas_evidence);
    lgn2_gpu_laguna_atlas_evidence_zero(&g_batch_laguna_atlas_evidence);

    g_batch_cb = lgn2_gpu_new_command_buffer();
    g_batch_has_work = NO;
    lgn2_gpu_laguna_atlas_evidence_zero(&g_batch_laguna_atlas_evidence);
    if (!g_batch_cb) {
        (void)lgn2_gpu_wait_pending_command_buffers("command batch");
        return 0;
    }
    return 1;
}

int lgn2_gpu_submit_commands(void) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!g_batch_cb) return 0;

    lgn2_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    g_batch_has_work = NO;
#ifdef LGN2_TEST_HOOKS
    g_laguna_router_fused_pending_dispatches +=
        g_laguna_router_fused_batch_dispatches;
    g_laguna_router_fused_batch_dispatches = 0;
    lgn2_gpu_laguna_test_decode_route_batch_committed();
#endif
    [cb commit];
    lgn2_gpu_laguna_atlas_register_pending(cb, &g_batch_laguna_atlas_evidence);
    lgn2_gpu_laguna_atlas_evidence_zero(&g_batch_laguna_atlas_evidence);
    return 1;
}

int lgn2_gpu_wait_submitted_commands(void) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if ([g_pending_cbs count] == 0) return 1;
    return lgn2_gpu_wait_pending_command_buffers("submitted command batch");
}

int lgn2_gpu_discard_commands(void) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!g_batch_cb) return 0;
    lgn2_gpu_close_batch_encoder();
    g_batch_cb = nil;
    g_batch_has_work = NO;
#ifdef LGN2_TEST_HOOKS
    /* The current batch is discarded; only previously committed pending
     * batches, if any, remain eligible for completion accounting. */
    g_laguna_router_fused_batch_dispatches = 0;
    g_test_glm_grouped_moe_batch_dispatches = 0;
    g_test_glm_grouped_moe_owned_dispatches = 0;
    memset(g_laguna_test_decode_route_batch, 0,
           sizeof(g_laguna_test_decode_route_batch));
#endif
    g_laguna_rope_atlas_valid = 0;
    g_laguna_rope_atlas_valid_completed = 0;
    g_laguna_rope_support_atlas_valid = 0;
    g_laguna_rope_support_atlas_valid_completed = 0;
    lgn2_gpu_laguna_atlas_evidence_zero(&g_batch_laguna_atlas_evidence);
    if ([g_pending_cbs count] != 0 &&
        lgn2_gpu_wait_pending_command_buffers("discarded command batch") == 0) {
        return 0;
    }
    return 1;
}

int lgn2_gpu_commands_active(void) {
    return g_batch_cb != nil;
}

#ifdef LGN2_TEST_HOOKS
/* Diagnostics/tests only.  Read the existing pending-command array directly;
 * do not add hot-path bookkeeping merely to support this getter. */
uint32_t lgn2_gpu_diagnostic_pending_command_buffer_count(void) {
    if (!g_initialized || !g_pending_cbs) return 0;
    const NSUInteger count = [g_pending_cbs count];
    return count > (NSUInteger)UINT32_MAX ? UINT32_MAX : (uint32_t)count;
}
#endif

int lgn2_gpu_end_commands(void) {
    if (!g_batch_cb) return 0;
    lgn2_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    g_batch_has_work = NO;
    g_owned_laguna_atlas_evidence = g_batch_laguna_atlas_evidence;
    lgn2_gpu_laguna_atlas_evidence_zero(&g_batch_laguna_atlas_evidence);
    const int ok = lgn2_gpu_finish_command_buffer(cb, 1, "command batch");
#ifdef LGN2_TEST_HOOKS
    if (ok) {
        for (int i = 0; i < LGN2_LAGUNA_TEST_DECODE_ROUTE_COUNT; i++) {
            g_laguna_test_decode_route_counts[i] +=
                g_laguna_test_decode_route_batch[i];
        }
    }
    memset(g_laguna_test_decode_route_batch, 0,
           sizeof(g_laguna_test_decode_route_batch));
#endif
    return ok;
}

int lgn2_gpu_synchronize(void) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (g_batch_cb) {
        const int result = lgn2_gpu_end_commands();
#ifdef LGN2_TEST_HOOKS
        return lgn2_gpu_test_synchronize_result(result);
#else
        return result;
#endif
    }
    if ([g_pending_cbs count] != 0) {
        int ok = lgn2_gpu_wait_pending_command_buffers("synchronize");
        lgn2_gpu_model_buffer_cache_maybe_evict("synchronize");
#ifdef LGN2_TEST_HOOKS
        return lgn2_gpu_test_synchronize_result(ok);
#else
        return ok;
#endif
    }

    id<MTLCommandBuffer> cb = lgn2_gpu_new_command_buffer();
    if (!cb) {
#ifdef LGN2_TEST_HOOKS
        return lgn2_gpu_test_synchronize_result(0);
#else
        return 0;
#endif
    }
    const int result = lgn2_gpu_finish_command_buffer(cb, 1, "synchronize");
#ifdef LGN2_TEST_HOOKS
    return lgn2_gpu_test_synchronize_result(result);
#else
    return result;
#endif
}

void lgn2_gpu_cleanup(void) {
    @autoreleasepool {
        /* Partial initialization owns the same globals as a complete
         * lifecycle.  Objective-C nil messaging and the idempotent helpers
         * below make this one unwind safe before, during, and after init. */
        if (g_batch_cb) {
            lgn2_gpu_close_batch_encoder();
            [g_batch_cb commit];
            [g_batch_cb waitUntilCompleted];
            g_batch_cb = nil;
        }
        g_batch_cb = nil;
        g_batch_enc = nil;
        g_batch_has_work = NO;
        lgn2_gpu_laguna_atlas_evidence_zero(&g_batch_laguna_atlas_evidence);
        lgn2_gpu_laguna_atlas_evidence_zero(&g_owned_laguna_atlas_evidence);
        g_command_batch_epoch = 0;
        (void)lgn2_gpu_wait_pending_command_buffers("cleanup");
        g_get_rows_f32_pipeline = nil;
        g_get_rows_i32_pipeline = nil;
        g_get_rows_q8_0_pipeline = nil;
        g_get_rows_q4_0_pipeline = nil;
        g_get_rows_q4_K_pipeline = nil;
        g_cpy_f32_f32_pipeline = nil;
        g_cpy_f32_f16_pipeline = nil;
        g_cpy_contig_f32_f16_pipeline = nil;
        g_swiglu_pipeline = nil;
        g_swiglu_flat_pipeline = nil;
        g_add_pipeline = nil;
        g_add2_pipeline = nil;
        g_add3_pipeline = nil;
        g_moe_sum8_pipeline = nil;
        g_moe_sum10_pipeline = nil;
        g_mul_pipeline = nil;
        g_bin_mul_scalar_pipeline = nil;
        g_bin_div_row_pipeline = nil;
        g_unary_sigmoid_pipeline = nil;
        g_unary_silu_pipeline = nil;
        g_unary_softplus_pipeline = nil;
        g_unary_sqrt_pipeline = nil;
        g_unary_clamp_pipeline = nil;
        g_unary_scale_pipeline = nil;
        g_unary_fill_pipeline = nil;
        g_unary_fill_f16_pipeline = nil;
        g_rms_norm_pipeline = nil;
        g_add_rms_norm_pipeline = nil;
        g_laguna_add3_rms_norm_pipeline = nil;
        g_laguna_add3_rms_norm_pipeline_checked = 0;
        g_laguna_qk_head_norm_rope_simd32_pipeline = nil;
        g_laguna_qk_head_norm_rope_simd32_pipeline_checked = 0;
        g_laguna_qk_head_norm_rope_simd32_invalid_env_reported = 0;
        g_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count = 0;
        g_laguna_qk_head_norm_rope_simd32_completed_dispatch_count = 0;
        g_laguna_qk_head_norm_rope_simd32_plan_mode = -2;
        g_laguna_qk_head_norm_rope_simd32_trace_mode = -1;
        g_laguna_rope_atlas_pipeline = nil;
        g_laguna_rope_support_atlas_pipeline = nil;
        g_laguna_head_norm_rope_atlas_pipeline = nil;
        g_laguna_qk_head_norm_rope_atlas_pipeline = nil;
        g_laguna_qk_head_norm_rope_simd32_atlas_pipeline = nil;
        g_laguna_rope_atlas_pipeline_checked = 0;
        g_laguna_rope_atlas_invalid_env_reported = 0;
        g_laguna_rope_atlas_trace_mode = -1;
        g_laguna_rope_atlas_buffer = nil;
        g_laguna_rope_atlas_buffer_bytes = 0;
        g_laguna_rope_support_atlas_buffer = nil;
        g_laguna_rope_support_atlas_buffer_bytes = 0;
        g_laguna_rope_atlas_valid = 0;
        g_laguna_rope_atlas_valid_completed = 0;
        g_laguna_rope_atlas_valid_epoch = 0;
        g_laguna_rope_atlas_valid_tokens = 0;
        g_laguna_rope_atlas_valid_pos0 = 0;
        g_laguna_rope_support_atlas_valid = 0;
        g_laguna_rope_support_atlas_valid_completed = 0;
        g_laguna_rope_support_atlas_valid_epoch = 0;
        g_laguna_rope_support_atlas_valid_tokens = 0;
        g_laguna_rope_support_atlas_valid_pos0 = 0;
        g_laguna_rope_atlas_encoded_dispatch_count = 0;
        g_laguna_rope_atlas_consumed_dispatch_count = 0;
        g_laguna_rope_atlas_consumed_family_count[0] = 0;
        g_laguna_rope_atlas_consumed_family_count[1] = 0;
        g_laguna_rope_support_atlas_encoded_dispatch_count = 0;
        g_laguna_rope_support_atlas_consumed_dispatch_count = 0;
        g_laguna_rope_atlas_completed_generated_count = 0;
        g_laguna_rope_atlas_completed_consumed_dispatch_count = 0;
        g_laguna_rope_atlas_completed_consumed_family_count[0] = 0;
        g_laguna_rope_atlas_completed_consumed_family_count[1] = 0;
        g_laguna_rope_support_atlas_completed_generated_count = 0;
        g_laguna_rope_support_atlas_completed_consumed_dispatch_count = 0;
        g_laguna_rope_atlas_plan_mode = -2;
        g_argsort_f32_i32_desc_pipeline = nil;
        g_argsort_merge_f32_i32_desc_pipeline = nil;
        g_sum_rows_f32_f32_pipeline = nil;
        g_dsv4_softplus_sqrt_pipeline = nil;
        g_dsv4_router_finalize_one_pipeline = nil;
        g_dsv4_router_finalize_one_simd_pipeline = nil;
        g_dsv4_router_finalize_weights_one_simd_pipeline = nil;
        g_dsv4_router_transform_finalize_weights_one_simd_pipeline = nil;
        g_dsv4_router_weights_one_pipeline = nil;
        g_glm_router_select_one_pipeline = nil;
        g_glm_router_select_one_simd_pipeline = nil;
        g_laguna_router_decode_fused_pipeline = nil;
        g_glm_q4_k_pair_swiglu_f32_pipeline = nil;
        g_glm_q4_k_pair_swiglu2_f32_pipeline = nil;
        g_glm_q4_k_pair_swiglu4_f32_pipeline = nil;
        g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline = nil;
        g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline = nil;
        g_glm_q2_k_pair_swiglu_f32_pipeline = nil;
        g_glm_q2_k_pair_swiglu_r1_f32_pipeline = nil;
        g_glm_q3_k_pair_swiglu_f32_pipeline = nil;
        g_glm_q3_k_pair_swiglu_r1_f32_pipeline = nil;
        g_glm_q2_k_down_f32_pipeline = nil;
        g_glm_q2_k_down_r1_f32_pipeline = nil;
        g_glm_q3_k_down_f32_pipeline = nil;
        g_glm_q3_k_down_r1_f32_pipeline = nil;
        g_glm_q4_k_down_f32_pipeline = nil;
        g_glm_q4_k_down_r1_f32_pipeline = nil;
        g_glm_q5_k_pair_swiglu_f32_pipeline = nil;
        g_glm_q5_k_pair_swiglu_mapped_f32_pipeline = nil;
        g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline = nil;
        g_glm_q5_k_down_f32_pipeline = nil;
        g_glm_q6_k_down_f32_pipeline = nil;
        g_laguna_routed_shared_pair_pipeline = nil;
        g_laguna_routed_shared_q4_down_pipeline = nil;
        g_laguna_routed_shared_q6_down_pipeline = nil;
        g_laguna_head_norm_rope_pipeline = nil;
        g_laguna_qk_head_norm_rope_pipeline = nil;
        g_laguna_qk_head_norm_rope_simd32_pipeline = nil;
        g_laguna_qk_head_norm_rope_simd32_pipeline_checked = 0;
        g_laguna_qk_head_norm_rope_simd32_invalid_env_reported = 0;
        g_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count = 0;
        g_laguna_qk_head_norm_rope_simd32_completed_dispatch_count = 0;
        g_laguna_qk_head_norm_rope_simd32_plan_mode = -2;
        g_laguna_qk_head_norm_rope_simd32_trace_mode = -1;
        g_laguna_rope_atlas_pipeline = nil;
        g_laguna_rope_support_atlas_pipeline = nil;
        g_laguna_head_norm_rope_atlas_pipeline = nil;
        g_laguna_qk_head_norm_rope_atlas_pipeline = nil;
        g_laguna_qk_head_norm_rope_simd32_atlas_pipeline = nil;
        g_laguna_rope_atlas_pipeline_checked = 0;
        g_laguna_rope_atlas_invalid_env_reported = 0;
        g_laguna_rope_atlas_trace_mode = -1;
        g_laguna_rope_atlas_buffer = nil;
        g_laguna_rope_atlas_buffer_bytes = 0;
        g_laguna_rope_support_atlas_buffer = nil;
        g_laguna_rope_support_atlas_buffer_bytes = 0;
        g_laguna_rope_atlas_valid = 0;
        g_laguna_rope_atlas_valid_completed = 0;
        g_laguna_rope_atlas_valid_epoch = 0;
        g_laguna_rope_atlas_valid_tokens = 0;
        g_laguna_rope_atlas_valid_pos0 = 0;
        g_laguna_rope_support_atlas_valid = 0;
        g_laguna_rope_support_atlas_valid_completed = 0;
        g_laguna_rope_support_atlas_valid_epoch = 0;
        g_laguna_rope_support_atlas_valid_tokens = 0;
        g_laguna_rope_support_atlas_valid_pos0 = 0;
        g_laguna_rope_atlas_encoded_dispatch_count = 0;
        g_laguna_rope_atlas_consumed_dispatch_count = 0;
        g_laguna_rope_atlas_consumed_family_count[0] = 0;
        g_laguna_rope_atlas_consumed_family_count[1] = 0;
        g_laguna_rope_support_atlas_encoded_dispatch_count = 0;
        g_laguna_rope_support_atlas_consumed_dispatch_count = 0;
        g_laguna_rope_atlas_completed_generated_count = 0;
        g_laguna_rope_atlas_completed_consumed_dispatch_count = 0;
        g_laguna_rope_atlas_completed_consumed_family_count[0] = 0;
        g_laguna_rope_atlas_completed_consumed_family_count[1] = 0;
        g_laguna_rope_support_atlas_completed_generated_count = 0;
        g_laguna_rope_support_atlas_completed_consumed_dispatch_count = 0;
        g_laguna_rope_atlas_plan_mode = -2;
        g_laguna_store_kv_pipeline = nil;
        g_laguna_attention_pipeline = nil;
        g_laguna_stage_kv_pipeline = nil;
        g_laguna_prefill_attention_pipeline = nil;
        g_laguna_commit_kv_pipeline = nil;
        g_laguna_q6_k_matmul_pipeline = nil;
        g_laguna_argmax_f32_pipeline = nil;
        g_dsv4_router_weights_batch_pipeline = nil;
        g_flash_attn_zero_mask_buffer = nil;
        g_flash_attn_pad_buffer = nil;
        g_flash_attn_tmp_buffer = nil;
        g_embed_rows_buffer = nil;
        g_router_selection_buffer = nil;
        g_router_weight_sum_buffer = nil;
        g_glm_router_simd_topk_stats_buffer = nil;
        g_indexer_topk_buffer = nil;
        g_moe_gate_scratch_buffer = nil;
        g_moe_down_scratch_buffer = nil;
        g_moe_id_map_buffer = nil;
        g_model_map_ptr = NULL;
        g_model_map_size = 0;
        g_model_mapped_offset = 0;
        g_model_mapped_size = 0;
        g_model_mapped_max_tensor_bytes = 0;
        lgn2_gpu_tensor_tracking_reset();
        g_flash_attn_zero_mask_bytes = 0;
        g_flash_attn_pad_bytes = 0;
        g_flash_attn_tmp_bytes = 0;
        g_embed_rows_bytes = 0;
        g_router_selection_bytes = 0;
        g_router_weight_sum_bytes = 0;
        g_indexer_topk_bytes = 0;
        g_moe_gate_scratch_bytes = 0;
        g_moe_down_scratch_bytes = 0;
        g_moe_id_map_bytes = 0;
        g_model_buffer_cache_bytes = 0;
        g_model_buffer_cache_evictions = 0;
        g_model_buffer_cache_over_limit = 0;
        lgn2_gpu_model_residency_clear();
        lgn2_gpu_model_views_clear();
        [g_pipeline_cache removeAllObjects];
        g_pipeline_cache = nil;
        [g_model_buffer_cache removeAllObjects];
        g_model_buffer_cache = nil;
        g_pending_cbs = nil;
        g_pending_laguna_atlas_evidence = nil;
#ifdef LGN2_TEST_HOOKS
        g_pending_glm_grouped_moe_evidence = nil;
#endif
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        g_metal4_runtime_available = 0;
        g_metal4_family_supported = 0;
        g_metal4_queue_supported = 0;
        g_metal4_m5_neural_accelerators_hint = 0;
        g_metal4_tensor_api_enabled = 0;
        g_metal4_tensor_api_compile_supported = 0;
        g_metal_device_name[0] = '\0';
        g_metal_math_safe = 0;
        g_q8_mv_ext_max_tokens = 16u;
        g_glm_grouped_moe_min_tokens = 96u;
        g_direct_kv_prefill_mode = 0;
        g_laguna_swa_gqa9_mode = 0;
        g_laguna_swa_gqa3_mode = 0;
        g_laguna_staged_swa_mode = 0;
        g_laguna_swa_selectors_snapshot_valid = 0;
#ifdef LGN2_TEST_HOOKS
        g_test_synchronize_fail_once = 0;
        g_test_wait_submitted_fail_once = 0;
        g_laguna_test_route_hooks = 0;
        g_laguna_test_direct_kv_count = 0;
        g_laguna_test_wrap_kv_count = 0;
        g_laguna_test_fused_q8_count = 0;
        g_laguna_test_stock_q8_count = 0;
        g_laguna_test_fused_q8_bco_false_count = 0;
        g_laguna_test_fused_q8_bco_true_count = 0;
        g_laguna_router_fused_encoded_dispatches = 0;
        g_laguna_router_fused_batch_dispatches = 0;
        g_laguna_router_fused_owned_dispatches = 0;
        g_laguna_router_fused_pending_dispatches = 0;
        g_laguna_router_fused_completed_dispatches = 0;
        g_test_glm_grouped_moe_encoded_dispatches = 0;
        g_test_glm_grouped_moe_batch_dispatches = 0;
        g_test_glm_grouped_moe_owned_dispatches = 0;
        g_test_glm_grouped_moe_completed_dispatches = 0;
        g_test_glm_exact_q4_encoded_dispatches = 0;
        g_test_glm_exact_q4_completed_dispatches = 0;
        memset(g_laguna_test_decode_route_counts, 0,
               sizeof(g_laguna_test_decode_route_counts));
        memset(g_laguna_test_decode_route_batch, 0,
               sizeof(g_laguna_test_decode_route_batch));
        memset(g_laguna_test_decode_route_inflight, 0,
               sizeof(g_laguna_test_decode_route_inflight));
#endif
        g_initialized = 0;
    }
}

static uint64_t lgn2_gpu_q8_0_row_bytes(uint32_t n_embd) {
    return (((uint64_t)n_embd + 31u) / 32u) * 34u;
}

static int lgn2_gpu_quant_row_bytes(
        uint32_t  type,
        uint32_t  n_embd,
        uint64_t *row_bytes_out) {
    if (!row_bytes_out || n_embd == 0) return 0;
    switch (type) {
    case LGN2_METAL_TENSOR_Q8_0:
        *row_bytes_out = (((uint64_t)n_embd + 31u) / 32u) * 34u;
        return 1;
    case LGN2_METAL_TENSOR_Q4_0:
        *row_bytes_out = (((uint64_t)n_embd + 31u) / 32u) * 18u;
        return 1;
    case LGN2_METAL_TENSOR_Q4_K:
        if ((n_embd % 256u) != 0) return 0;
        *row_bytes_out = ((uint64_t)n_embd / 256u) * 144u;
        return 1;
    default:
        return 0;
    }
}

static int lgn2_gpu_q8_0_table_bytes(
        uint32_t  n_vocab,
        uint32_t  n_embd,
        uint64_t *bytes_out) {
    if (!bytes_out || n_vocab == 0 || n_embd == 0) return 0;
    const uint64_t row_bytes = lgn2_gpu_q8_0_row_bytes(n_embd);
    if (row_bytes != 0 && (uint64_t)n_vocab > UINT64_MAX / row_bytes) return 0;
    *bytes_out = (uint64_t)n_vocab * row_bytes;
    return 1;
}

static int lgn2_gpu_quant_table_bytes(
        uint32_t  type,
        uint32_t  n_vocab,
        uint32_t  n_embd,
        uint64_t *bytes_out) {
    if (!bytes_out || n_vocab == 0 || n_embd == 0) return 0;
    uint64_t row_bytes = 0;
    if (!lgn2_gpu_quant_row_bytes(type, n_embd, &row_bytes)) return 0;
    if (row_bytes != 0 && (uint64_t)n_vocab > UINT64_MAX / row_bytes) return 0;
    *bytes_out = (uint64_t)n_vocab * row_bytes;
    return 1;
}

static int lgn2_gpu_encode_get_rows_q8_0(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        weight,
        NSUInteger           weight_offset,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_offset,
        const int32_t       *single_token,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_vocab,
        uint32_t             n_tokens,
        uint32_t             n_embd) {
    if (!cb || !weight || !out || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }
    if (!tokens && (!single_token || n_tokens != 1)) {
        return 0;
    }

    lgn2_gpu_get_rows_q8_0_args args = {
        .n_embd = (int32_t)n_embd,
        .n_vocab = (int32_t)n_vocab,
        .n_tokens = (int32_t)n_tokens,
        .src_row_bytes = lgn2_gpu_q8_0_row_bytes(n_embd),
        .dst_row_bytes = (uint64_t)n_embd * sizeof(float),
        .token_stride = sizeof(int32_t),
    };

    NSUInteger nth = 32u;
    const NSUInteger max_threads = g_get_rows_q8_0_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;
    const NSUInteger nblocks = ((NSUInteger)n_embd + 31u) / 32u;

    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_q8_0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weight offset:weight_offset atIndex:1];
    if (tokens) {
        [enc setBuffer:tokens offset:tokens_offset atIndex:2];
    } else {
        [enc setBytes:single_token length:sizeof(*single_token) atIndex:2];
    }
    [enc setBuffer:out offset:out_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nblocks, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int lgn2_gpu_encode_get_rows_quant(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        weight,
        NSUInteger           weight_offset,
        uint32_t             weight_type,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_offset,
        const int32_t       *single_token,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_vocab,
        uint32_t             n_tokens,
        uint32_t             n_embd) {
    if (weight_type == LGN2_METAL_TENSOR_Q8_0) {
        return lgn2_gpu_encode_get_rows_q8_0(cb,
                                            weight,
                                            weight_offset,
                                            tokens,
                                            tokens_offset,
                                            single_token,
                                            out,
                                            out_offset,
                                            n_vocab,
                                            n_tokens,
                                            n_embd);
    }
    if (!cb || !weight || !out || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }
    if (!tokens && (!single_token || n_tokens != 1)) {
        return 0;
    }

    uint64_t src_row_bytes = 0;
    if (!lgn2_gpu_quant_row_bytes(weight_type, n_embd, &src_row_bytes)) return 0;
    lgn2_gpu_get_rows_q8_0_args args = {
        .n_embd = (int32_t)n_embd,
        .n_vocab = (int32_t)n_vocab,
        .n_tokens = (int32_t)n_tokens,
        .src_row_bytes = src_row_bytes,
        .dst_row_bytes = (uint64_t)n_embd * sizeof(float),
        .token_stride = sizeof(int32_t),
    };

    id<MTLComputePipelineState> pipeline = nil;
    NSUInteger block_width = 0;
    if (weight_type == LGN2_METAL_TENSOR_Q4_0) {
        pipeline = g_get_rows_q4_0_pipeline;
        block_width = 32u;
    } else if (weight_type == LGN2_METAL_TENSOR_Q4_K) {
        pipeline = g_get_rows_q4_K_pipeline;
        block_width = 256u;
    }
    if (!pipeline || block_width == 0) return 0;

    NSUInteger nth = block_width;
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;
    const NSUInteger nblocks = ((NSUInteger)n_embd + block_width - 1u) / block_width;

    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weight offset:weight_offset atIndex:1];
    if (tokens) {
        [enc setBuffer:tokens offset:tokens_offset atIndex:2];
    } else {
        [enc setBytes:single_token length:sizeof(*single_token) atIndex:2];
    }
    [enc setBuffer:out offset:out_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nblocks, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int lgn2_gpu_embed_token_q8_0_tensor(
        lgn2_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        const uint64_t out_bytes = (uint64_t)n_embd * sizeof(float);
        if (!outbuf || lgn2_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "lgn2: Metal Q8_0 embedding received undersized output buffer\n");
            return 0;
        }

        uint64_t weight_bytes = 0;
        if (!lgn2_gpu_q8_0_table_bytes(n_vocab, n_embd, &weight_bytes) ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "lgn2: Metal Q8_0 embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        uint32_t token_for_kernel = token;
        id<MTLBuffer> wbuf = nil;
        const bool exact_token_row =
            getenv("LGN2_METAL_DISABLE_TOKEN_EMBED_EXACT_VIEW") == NULL;
        if (exact_token_row) {
            const uint64_t row_bytes = lgn2_gpu_q8_0_row_bytes(n_embd);
            const uint64_t token_rel = (uint64_t)token * row_bytes;
            if (token_rel > weight_bytes || row_bytes > weight_bytes - token_rel) {
                fprintf(stderr, "lgn2: Metal Q8_0 embedding token row is outside the mapped table\n");
                return 0;
            }
            wbuf = lgn2_gpu_wrap_model_exact_range(model_map,
                                                  model_size,
                                                  weight_offset + token_rel,
                                                  row_bytes,
                                                  &inner_offset);
            token_for_kernel = 0;
        } else {
            wbuf = lgn2_gpu_wrap_model_range(model_map,
                                            model_size,
                                            weight_offset,
                                            weight_bytes,
                                            &inner_offset);
        }
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t token_i32 = (int32_t)token_for_kernel;
        if (!lgn2_gpu_encode_get_rows_q8_0(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           nil,
                                           0,
                                           &token_i32,
                                           outbuf,
                                           lgn2_gpu_tensor_offset(out),
                                           n_vocab,
                                           1,
                                           n_embd)) {
            return 0;
        }

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "q8_0 embed token")) return 0;
    }

    return 1;
}

int lgn2_gpu_embed_tokens_q8_0_tensor(
        lgn2_gpu_tensor       *out,
        const lgn2_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        id<MTLBuffer> tokbuf = lgn2_gpu_tensor_buffer(tokens);
        const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * sizeof(float);
        const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
        if (!outbuf || !tokbuf ||
            lgn2_gpu_tensor_bytes(out) < out_bytes ||
            lgn2_gpu_tensor_bytes(tokens) < token_bytes) {
            fprintf(stderr, "lgn2: Metal Q8_0 batched embedding received undersized buffers\n");
            return 0;
        }

        uint64_t weight_bytes = 0;
        if (!lgn2_gpu_q8_0_table_bytes(n_vocab, n_embd, &weight_bytes) ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "lgn2: Metal Q8_0 batched embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            lgn2_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!lgn2_gpu_encode_get_rows_q8_0(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           tokbuf,
                                           lgn2_gpu_tensor_offset(tokens),
                                           NULL,
                                           outbuf,
                                           lgn2_gpu_tensor_offset(out),
                                           n_vocab,
                                           n_tokens,
                                           n_embd)) {
            return 0;
        }

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "q8_0 embed tokens")) return 0;
    }

    return 1;
}

int lgn2_gpu_embed_token_quant_tensor(
        lgn2_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          weight_type,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd) {
    if (weight_type == LGN2_METAL_TENSOR_Q8_0) {
        return lgn2_gpu_embed_token_q8_0_tensor(out,
                                               model_map,
                                               model_size,
                                               weight_offset,
                                               n_vocab,
                                               token,
                                               n_embd);
    }
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        const uint64_t out_bytes = (uint64_t)n_embd * sizeof(float);
        if (!outbuf || lgn2_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "lgn2: Metal quant embedding received undersized output buffer\n");
            return 0;
        }

        uint64_t weight_bytes = 0;
        uint64_t row_bytes = 0;
        if (!lgn2_gpu_quant_table_bytes(weight_type, n_vocab, n_embd, &weight_bytes) ||
            !lgn2_gpu_quant_row_bytes(weight_type, n_embd, &row_bytes) ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "lgn2: Metal quant embedding range is outside the mapped model\n");
            return 0;
        }

        const uint64_t token_rel = (uint64_t)token * row_bytes;
        if (token_rel > weight_bytes || row_bytes > weight_bytes - token_rel) {
            fprintf(stderr, "lgn2: Metal quant embedding token row is outside the mapped table\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            lgn2_gpu_wrap_model_exact_range(model_map,
                                           model_size,
                                           weight_offset + token_rel,
                                           row_bytes,
                                           &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t token_i32 = 0;
        if (!lgn2_gpu_encode_get_rows_quant(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           weight_type,
                                           nil,
                                           0,
                                           &token_i32,
                                           outbuf,
                                           lgn2_gpu_tensor_offset(out),
                                           n_vocab,
                                           1,
                                           n_embd)) {
            return 0;
        }

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "quant embed token")) return 0;
    }

    return 1;
}

int lgn2_gpu_embed_tokens_quant_tensor(
        lgn2_gpu_tensor       *out,
        const lgn2_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd) {
    if (weight_type == LGN2_METAL_TENSOR_Q8_0) {
        return lgn2_gpu_embed_tokens_q8_0_tensor(out,
                                                tokens,
                                                model_map,
                                                model_size,
                                                weight_offset,
                                                n_vocab,
                                                n_tokens,
                                                n_embd);
    }
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        id<MTLBuffer> tokbuf = lgn2_gpu_tensor_buffer(tokens);
        const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * sizeof(float);
        const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
        if (!outbuf || !tokbuf ||
            lgn2_gpu_tensor_bytes(out) < out_bytes ||
            lgn2_gpu_tensor_bytes(tokens) < token_bytes) {
            fprintf(stderr, "lgn2: Metal quant batched embedding received undersized buffers\n");
            return 0;
        }

        uint64_t weight_bytes = 0;
        if (!lgn2_gpu_quant_table_bytes(weight_type, n_vocab, n_embd, &weight_bytes) ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "lgn2: Metal quant batched embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            lgn2_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!lgn2_gpu_encode_get_rows_quant(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           weight_type,
                                           tokbuf,
                                           lgn2_gpu_tensor_offset(tokens),
                                           NULL,
                                           outbuf,
                                           lgn2_gpu_tensor_offset(out),
                                           n_vocab,
                                           n_tokens,
                                           n_embd)) {
            return 0;
        }

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "quant embed tokens")) return 0;
    }

    return 1;
}

int lgn2_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!model_map || model_size == 0) return 0;
    if (map_offset > model_size || map_size == 0 || map_size > model_size - map_offset) return 0;
    max_tensor_bytes = lgn2_gpu_effective_model_max_tensor_bytes(map_size, max_tensor_bytes);

    @autoreleasepool {
        if (g_model_map_ptr == model_map &&
            g_model_map_size == model_size &&
            g_model_mapped_offset == map_offset &&
            g_model_mapped_size == map_size &&
            g_model_mapped_max_tensor_bytes == max_tensor_bytes) {
            return 1;
        }

        for (uint32_t i = 0; i < g_model_view_count; i++) {
            if (g_model_views[i].model_map == model_map &&
                g_model_views[i].model_size == model_size &&
                map_offset >= g_model_views[i].model_offset &&
                map_offset + map_size <= g_model_views[i].model_offset + g_model_views[i].bytes) {
                return 1;
            }
        }

        lgn2_gpu_model_residency_clear();
        if (!lgn2_gpu_map_model_views(model_map, model_size, map_offset, map_size, max_tensor_bytes)) {
            lgn2_gpu_model_residency_clear();
            return 0;
        }
        g_model_map_ptr = model_map;
        g_model_map_size = model_size;
        g_model_mapped_offset = map_offset;
        g_model_mapped_size = map_size;
        g_model_mapped_max_tensor_bytes = max_tensor_bytes;
        if (lgn2_gpu_model_map_log_enabled()) {
            fprintf(stderr,
                    "lgn2: Metal mapped mmaped model as %u overlapping shared buffers\n",
                    g_model_view_count);
        }
        return 1;
    }
}

int lgn2_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    return lgn2_gpu_set_model_map_range(model_map, model_size, 0, model_size, 0);
}

int lgn2_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    (void)fd;
    (void)model_map;
    return 1;
}

static id<MTLBuffer> lgn2_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    (void)model_map;
    if (model_size == 0 || offset > model_size || len > model_size - offset) {
        fprintf(stderr, "lgn2: Metal model range is outside the mapped model\n");
        return nil;
    }

    const uint64_t end = offset + len;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        if (g_model_views[i].model_map != model_map ||
            g_model_views[i].model_size != model_size) {
            continue;
        }
        const uint64_t view_start = g_model_views[i].model_offset;
        const uint64_t view_end = view_start + g_model_views[i].bytes;
        if (offset >= view_start && end <= view_end) {
            *inner_offset = offset - view_start;
            return g_model_views[i].buffer;
        }
    }

    fprintf(stderr,
            "lgn2: Metal model range %.2f..%.2f GiB is not covered by mapped model views\n",
            lgn2_gpu_gib(offset),
            lgn2_gpu_gib(end));
    return nil;
}

static id<MTLBuffer> lgn2_gpu_wrap_model_exact_range_impl(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    if (!model_map || !g_device ||
        !g_model_buffer_cache ||
        model_size == 0 || offset > model_size || len > model_size - offset) {
        fprintf(stderr, "lgn2: Metal exact model range is outside the mapped model\n");
        return nil;
    }

    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t page_offset = offset & ~(page - 1);
    const uint64_t leading = offset - page_offset;
    if (len > UINT64_MAX - leading ||
        leading + len > UINT64_MAX - (page - 1)) {
        fprintf(stderr, "lgn2: Metal exact model range overflows page alignment\n");
        return nil;
    }
    uint64_t view_bytes = round_up_u64(leading + len, page);
    if (view_bytes > model_size - page_offset) view_bytes = model_size - page_offset;
    if (leading + len > view_bytes) {
        fprintf(stderr, "lgn2: Metal exact model range alignment exceeds mapped model\n");
        return nil;
    }
    if (view_bytes > (uint64_t)[g_device maxBufferLength]) {
        fprintf(stderr,
                "lgn2: Metal exact model range %.2f GiB exceeds maxBufferLength %.2f GiB\n",
                lgn2_gpu_gib(view_bytes),
                lgn2_gpu_gib((uint64_t)[g_device maxBufferLength]));
        return nil;
    }

    NSString *key = nil;
    id<MTLBuffer> buffer = nil;
    key = [NSString stringWithFormat:@"%p:%llu:%llu:%llu",
           model_map,
           (unsigned long long)model_size,
           (unsigned long long)page_offset,
           (unsigned long long)view_bytes];
    buffer = [g_model_buffer_cache objectForKey:key];
    if (!buffer) {
        const uintptr_t base = (uintptr_t)model_map;
        buffer = [g_device newBufferWithBytesNoCopy:(void *)(base + page_offset)
                                             length:(NSUInteger)view_bytes
                                            options:lgn2_gpu_model_resource_options()
                                        deallocator:nil];
        if (!buffer) {
            fprintf(stderr,
                    "lgn2: Metal could not wrap exact mmaped model range at %.2f GiB, size %.2f MiB\n",
                    lgn2_gpu_gib(page_offset),
                    lgn2_gpu_mib(view_bytes));
            return nil;
        }
        buffer.label = @"lgn2_model_exact_view";
        [g_model_buffer_cache setObject:buffer forKey:key];
        lgn2_gpu_model_buffer_cache_note_insert(view_bytes);
    }

    if (inner_offset) *inner_offset = leading;
    return buffer;
}

static id<MTLBuffer> lgn2_gpu_wrap_model_exact_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    return lgn2_gpu_wrap_model_exact_range_impl(model_map,
                                               model_size,
                                               offset,
                                               len,
                                               inner_offset);
}

#ifdef LGN2_TEST_HOOKS
void lgn2_gpu_test_inject_synchronize_failure(void) {
    g_test_synchronize_fail_once = 1;
}

void lgn2_gpu_test_inject_wait_submitted_failure(void) {
    g_test_wait_submitted_fail_once = 1;
}

int lgn2_gpu_test_tensor_tracking_state(uint64_t *live_handles,
                                       uint64_t *live_bytes) {
    pthread_mutex_lock(&g_tensor_mu);
    if (live_handles) *live_handles = (uint64_t)g_tensor_live_count;
    if (live_bytes) *live_bytes = g_tensor_alloc_live_bytes;
    pthread_mutex_unlock(&g_tensor_mu);
    return 1;
}

int lgn2_gpu_test_cleanup_state_is_clean(void) {
    pthread_mutex_lock(&g_tensor_mu);
    const int tensors_clean =
        g_tensor_live_slots == NULL && g_tensor_live_cap == 0 &&
        g_tensor_live_count == 0 && g_tensor_live_tombs == 0 &&
        g_tensor_alloc_live_bytes == 0 && g_tensor_alloc_peak_bytes == 0;
    pthread_mutex_unlock(&g_tensor_mu);

    return !g_initialized &&
           !g_device && !g_queue && !g_library &&
           !g_metal4_runtime_available && !g_metal4_family_supported &&
           !g_metal4_queue_supported &&
           !g_metal4_m5_neural_accelerators_hint &&
           !g_metal4_tensor_api_enabled &&
           !g_metal4_tensor_api_compile_supported &&
           g_metal_device_name[0] == '\0' &&
           !g_batch_cb && !g_batch_enc &&
           !g_batch_has_work && g_command_batch_epoch == 0 &&
           !g_pending_cbs && !g_pending_laguna_atlas_evidence &&
#ifdef LGN2_TEST_HOOKS
           !g_pending_glm_grouped_moe_evidence &&
#endif
           !g_model_buffer_cache &&
           !g_pipeline_cache &&
           g_model_buffer_cache_bytes == 0 &&
           g_model_buffer_cache_evictions == 0 &&
           g_model_buffer_cache_over_limit == 0 &&
           g_model_view_count == 0 && !g_model_residency_set &&
           g_model_residency_added_to_queue == 0 &&
           g_model_map_ptr == NULL &&
           g_model_map_size == 0 && g_model_mapped_offset == 0 &&
           g_model_mapped_size == 0 &&
           g_model_mapped_max_tensor_bytes == 0 &&
           !g_get_rows_f32_pipeline &&
           !g_glm_q2_k_pair_swiglu_r1_f32_pipeline &&
           !g_glm_q3_k_pair_swiglu_r1_f32_pipeline &&
           !g_glm_q2_k_down_r1_f32_pipeline &&
           !g_glm_q3_k_down_r1_f32_pipeline &&
           !g_glm_q4_k_down_r1_f32_pipeline &&
           g_laguna_swa_gqa9_mode == 0 &&
           g_laguna_swa_gqa3_mode == 0 &&
           g_laguna_staged_swa_mode == 0 &&
           g_direct_kv_prefill_mode == 0 &&
           !g_laguna_swa_selectors_snapshot_valid &&
           g_laguna_qk_head_norm_rope_simd32_plan_mode == -2 &&
           g_laguna_qk_head_norm_rope_simd32_trace_mode == -1 &&
           g_laguna_rope_atlas_plan_mode == -2 &&
           g_laguna_rope_atlas_trace_mode == -1 &&
           g_test_init_failpoint == LGN2_GPU_TEST_INIT_FAIL_NONE &&
           g_test_synchronize_fail_once == 0 &&
           tensors_clean;
}

int lgn2_gpu_test_lifecycle_cleanup(void) {
    static const lgn2_gpu_test_init_failpoint failpoints[] = {
        LGN2_GPU_TEST_INIT_FAIL_DEVICE,
        LGN2_GPU_TEST_INIT_FAIL_QUEUE,
        LGN2_GPU_TEST_INIT_FAIL_BOOKKEEPING,
        LGN2_GPU_TEST_INIT_FAIL_FIRST_PIPELINE,
    };
    int ok = 1;

    lgn2_gpu_cleanup();
    if (!lgn2_gpu_test_cleanup_state_is_clean()) {
        fprintf(stderr, "lgn2: Metal lifecycle test did not start clean\n");
        return 0;
    }

    for (size_t i = 0; i < sizeof(failpoints) / sizeof(failpoints[0]); i++) {
        g_test_init_failpoint = failpoints[i];
        if (lgn2_gpu_init() != 0 || !lgn2_gpu_test_cleanup_state_is_clean()) {
            fprintf(stderr,
                    "lgn2: Metal partial-init cleanup failed at failpoint %d\n",
                    (int)failpoints[i]);
            lgn2_gpu_cleanup();
            return 0;
        }
    }

    if (!lgn2_gpu_init()) {
        fprintf(stderr, "lgn2: Metal lifecycle retry did not initialize\n");
        return 0;
    }

    const size_t page = (size_t)getpagesize();
#if defined(MAP_ANONYMOUS)
    const int anonymous = MAP_ANONYMOUS;
#else
    const int anonymous = MAP_ANON;
#endif
    void *model_map = mmap(NULL, page, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | anonymous, -1, 0);
    if (model_map == MAP_FAILED) {
        model_map = NULL;
        ok = 0;
    }

    lgn2_gpu_tensor *workspace_src = NULL;
    lgn2_gpu_tensor *workspace_dst = NULL;
    if (ok) {
        memset(model_map, 0x5a, page);
        ok = lgn2_gpu_set_model_map_range(model_map,
                                         (uint64_t)page,
                                         0,
                                         (uint64_t)page,
                                         (uint64_t)page) != 0;
    }
    if (ok) {
        uint64_t inner_offset = UINT64_MAX;
        id<MTLBuffer> cached = lgn2_gpu_wrap_model_exact_range(
            model_map, (uint64_t)page, 0, 64, &inner_offset);
        ok = cached != nil && inner_offset == 0 &&
             g_model_view_count != 0 &&
             [g_model_buffer_cache count] != 0;
        cached = nil;
    }
    if (ok) {
        workspace_src = lgn2_gpu_tensor_alloc(64);
        workspace_dst = lgn2_gpu_tensor_alloc(64);
        ok = workspace_src != NULL && workspace_dst != NULL &&
             lgn2_gpu_begin_commands() != 0 &&
             lgn2_gpu_tensor_copy(workspace_dst, 0,
                                 workspace_src, 0, 64) != 0;
    }
    if (ok) {
        /* This is the engine-close sequence in miniature: drain actual Metal
         * work, retire workspace handles while tracking is alive, then tear
         * down the view/cache backend while the mmap remains valid. */
        ok = lgn2_gpu_synchronize() != 0;
    }

    if (workspace_dst) lgn2_gpu_tensor_free(workspace_dst);
    if (workspace_src) lgn2_gpu_tensor_free(workspace_src);
    lgn2_gpu_cleanup();
    if (model_map) munmap(model_map, page);

    ok = ok && lgn2_gpu_test_cleanup_state_is_clean();
    /* A second cleanup proves the fully-clean state is also a safe input. */
    lgn2_gpu_cleanup();
    ok = ok && lgn2_gpu_test_cleanup_state_is_clean();
    if (!ok) {
        fprintf(stderr,
                "lgn2: Metal drain/workspace/view-cache lifecycle test failed\n");
    }
    return ok;
}
#endif

int lgn2_gpu_indexer_topk_tensor(
        lgn2_gpu_tensor       *selected,
        const lgn2_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 || top_k > n_comp) return 0;

    @autoreleasepool {
        const uint64_t score_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)top_k * n_tokens * sizeof(uint32_t);
        id<MTLBuffer> scorebuf = lgn2_gpu_tensor_buffer(scores);
        id<MTLBuffer> selbuf = lgn2_gpu_tensor_buffer(selected);
        if (!scorebuf || !selbuf ||
            lgn2_gpu_tensor_bytes(scores) < score_bytes ||
            lgn2_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "lgn2: Metal graph indexer top-k received undersized buffers\n");
            return 0;
        }
        NSUInteger max_threads = g_argsort_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup;
        if (max_threads == 0) max_threads = 256;
        int32_t nth = 1;
        while ((uint32_t)nth < n_comp && (uint64_t)2u * (uint64_t)nth <= (uint64_t)max_threads) {
            nth *= 2;
        }
        const int32_t npr = (int32_t)((n_comp + (uint32_t)nth - 1u) / (uint32_t)nth);
        const int32_t block_top_k = (int32_t)(top_k < (uint32_t)nth ? top_k : (uint32_t)nth);
        int32_t work_width = (int32_t)top_k;
        if (npr > 1) {
            const int32_t last_block = (int32_t)n_comp - (npr - 1) * nth;
            work_width = (npr - 1) * block_top_k + (last_block < block_top_k ? last_block : block_top_k);
        }
        const uint64_t scratch_row_bytes = (uint64_t)work_width * sizeof(uint32_t);
        const bool one_pass = npr <= 1;
        const uint64_t scratch_bytes = one_pass ? scratch_row_bytes * n_tokens :
            2u * scratch_row_bytes * n_tokens;
        if (!lgn2_gpu_ensure_scratch_buffer(&g_indexer_topk_buffer,
                                             &g_indexer_topk_bytes,
                                             (NSUInteger)scratch_bytes,
                                             "lgn2_indexer_topk")) {
            return 0;
        }

        lgn2_gpu_kargs_argsort args = {
            .ne00 = (int32_t)n_comp,
            .ne01 = (int32_t)n_tokens,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)n_comp * sizeof(float),
            .nb02 = (uint64_t)n_comp * n_tokens * sizeof(float),
            .nb03 = (uint64_t)n_comp * n_tokens * sizeof(float),
            .ne0 = work_width,
            .ne1 = (int32_t)n_tokens,
            .ne2 = 1,
            .ne3 = 1,
            .top_k = block_top_k,
        };
        // kernel_argsort_f32_i32_desc stages the block's scores behind the
        // index array: nth int32 indices + nth float scores.
        const NSUInteger smem = (((NSUInteger)nth * (sizeof(int32_t) + sizeof(float))) + 15u) & ~(NSUInteger)15u;

        NSUInteger cur_off = 0;
        NSUInteger next_off = (NSUInteger)scratch_row_bytes * n_tokens;
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_argsort_f32_i32_desc_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:scorebuf offset:lgn2_gpu_tensor_offset(scores) atIndex:1];
        [enc setBuffer:one_pass ? selbuf : g_indexer_topk_buffer
              offset:one_pass ? lgn2_gpu_tensor_offset(selected) : cur_off
             atIndex:2];
        [enc setThreadgroupMemoryLength:smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)npr * n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake((NSUInteger)nth, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        int32_t len = block_top_k;
        while (len < work_width) {
            const int32_t nm = (work_width + 2 * len - 1) / (2 * len);
            const bool final_merge = nm == 1;
            NSUInteger merge_threads = g_argsort_merge_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup;
            if (merge_threads == 0 || merge_threads > 512u) merge_threads = 512u;
            if (merge_threads > (NSUInteger)len) merge_threads = (NSUInteger)len;
            if (merge_threads == 0) merge_threads = 1;

            lgn2_gpu_kargs_argsort_merge merge_args = {
                .ne00 = (int64_t)n_comp,
                .ne01 = (int64_t)n_tokens,
                .ne02 = 1,
                .ne03 = 1,
                .nb00 = sizeof(float),
                .nb01 = (uint64_t)n_comp * sizeof(float),
                .nb02 = (uint64_t)n_comp * n_tokens * sizeof(float),
                .nb03 = (uint64_t)n_comp * n_tokens * sizeof(float),
                .ne0 = work_width,
                .ne1 = (int32_t)n_tokens,
                .ne2 = 1,
                .ne3 = 1,
                .top_k = nm == 1 ? (int32_t)top_k : work_width,
                .len = len,
            };

            enc = lgn2_gpu_compute_encoder(cb);
            [enc setComputePipelineState:g_argsort_merge_f32_i32_desc_pipeline];
            [enc setBytes:&merge_args length:sizeof(merge_args) atIndex:0];
            [enc setBuffer:scorebuf offset:lgn2_gpu_tensor_offset(scores) atIndex:1];
            [enc setBuffer:g_indexer_topk_buffer offset:cur_off atIndex:2];
            [enc setBuffer:final_merge ? selbuf : g_indexer_topk_buffer
                  offset:final_merge ? lgn2_gpu_tensor_offset(selected) : next_off
                 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nm * n_tokens, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(merge_threads, 1, 1)];
            lgn2_gpu_end_compute_encoder(cb, enc);

            const NSUInteger tmp = cur_off;
            cur_off = next_off;
            next_off = tmp;
            len <<= 1;
        }

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "indexer top-k")) return 0;
    }

    return 1;
}

int lgn2_gpu_laguna_argmax_available(void) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    return g_laguna_argmax_f32_pipeline != nil;
}

static int lgn2_gpu_laguna_add3_rms_norm_pipeline_usable(
        id<MTLComputePipelineState> pipeline,
        uint32_t                    n) {
    if (!pipeline) return 0;
    /* The kernel's reduction stages one float per SIMD lane in a fixed
     * 32-entry threadgroup buffer.  A different execution width would make
     * the existing simdgroup indexing/second reduction topology invalid. */
    const NSUInteger execution_width = pipeline.threadExecutionWidth;
    if (execution_width != 32u) {
        fprintf(stderr,
                "lgn2: Laguna add3+RMS norm pipeline has execution width %lu, "
                "requires 32\n",
                (unsigned long)execution_width);
        return 0;
    }
    const NSUInteger required = lgn2_gpu_rms_norm_threads(n);
    if (required > execution_width && required % execution_width != 0u) {
        fprintf(stderr,
                "lgn2: Laguna add3+RMS norm needs %lu threads, which "
                "leaves a partial SIMD group at width %lu\n",
                (unsigned long)required,
                (unsigned long)execution_width);
        return 0;
    }
    if (pipeline.maxTotalThreadsPerThreadgroup < required) {
        fprintf(stderr,
                "lgn2: Laguna add3+RMS norm pipeline supports %lu "
                "threads, needs at least %lu for n=%u\n",
                (unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
                (unsigned long)required,
                n);
        return 0;
    }
    return 1;
}

int lgn2_gpu_laguna_decode_residual_norm_available(void) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_laguna_add3_rms_norm_pipeline();
        if (!pipeline) return 0;
        return lgn2_gpu_laguna_add3_rms_norm_pipeline_usable(pipeline, 3072u);
    }
}

static int lgn2_gpu_metal_ranges_overlap(
        const lgn2_gpu_tensor *a,
        uint64_t              a_bytes,
        const lgn2_gpu_tensor *b,
        uint64_t              b_bytes) {
    id<MTLBuffer> abuf = lgn2_gpu_tensor_buffer(a);
    id<MTLBuffer> bbuf = lgn2_gpu_tensor_buffer(b);
    if (!abuf || !bbuf || abuf != bbuf || a_bytes == 0 || b_bytes == 0) {
        return 0;
    }
    const uint64_t a0 = (uint64_t)lgn2_gpu_tensor_offset(a);
    const uint64_t b0 = (uint64_t)lgn2_gpu_tensor_offset(b);
    if (a0 > UINT64_MAX - a_bytes || b0 > UINT64_MAX - b_bytes) {
        return 1;
    }
    const uint64_t a1 = a0 + a_bytes;
    const uint64_t b1 = b0 + b_bytes;
    return a0 < b1 && b0 < a1;
}

int lgn2_gpu_add3_rms_norm_weight_rows_tensor(
        lgn2_gpu_tensor       *norm_out,
        lgn2_gpu_tensor       *sum_out,
        const lgn2_gpu_tensor *a,
        const lgn2_gpu_tensor *b,
        const lgn2_gpu_tensor *c,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n,
        uint32_t              rows,
        float                 eps) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!norm_out || !sum_out || !a || !b || !c || !model_map ||
        model_size == 0 || n == 0 || rows == 0 || (n & 3u) != 0 ||
        n > (uint32_t)INT32_MAX || rows > (uint32_t)INT32_MAX) {
        fprintf(stderr,
                "lgn2: Laguna add3+RMS norm received invalid arguments\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_laguna_add3_rms_norm_pipeline();
        if (!pipeline) {
            fprintf(stderr,
                    "lgn2: Laguna add3+RMS norm pipeline is unavailable\n");
            return 0;
        }
        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        if ((uint64_t)rows > UINT64_MAX / row_bytes) {
            fprintf(stderr,
                    "lgn2: Laguna add3+RMS norm row count overflows activation size\n");
            return 0;
        }
        const uint64_t bytes = row_bytes * rows;
        if (!lgn2_gpu_laguna_add3_rms_norm_pipeline_usable(pipeline, n)) {
            return 0;
        }
        id<MTLBuffer> abuf = lgn2_gpu_tensor_buffer(a);
        id<MTLBuffer> bbuf = lgn2_gpu_tensor_buffer(b);
        id<MTLBuffer> cbuf = lgn2_gpu_tensor_buffer(c);
        id<MTLBuffer> sumbuf = lgn2_gpu_tensor_buffer(sum_out);
        id<MTLBuffer> normbuf = lgn2_gpu_tensor_buffer(norm_out);
        if (!abuf || !bbuf || !cbuf || !sumbuf || !normbuf ||
            lgn2_gpu_tensor_bytes(a) < bytes ||
            lgn2_gpu_tensor_bytes(b) < bytes ||
            lgn2_gpu_tensor_bytes(c) < bytes ||
            lgn2_gpu_tensor_bytes(sum_out) < bytes ||
            lgn2_gpu_tensor_bytes(norm_out) < bytes ||
            lgn2_gpu_metal_ranges_overlap(sum_out, bytes, norm_out, bytes) ||
            lgn2_gpu_metal_ranges_overlap(sum_out, bytes, a, bytes) ||
            lgn2_gpu_metal_ranges_overlap(sum_out, bytes, b, bytes) ||
            lgn2_gpu_metal_ranges_overlap(sum_out, bytes, c, bytes) ||
            lgn2_gpu_metal_ranges_overlap(norm_out, bytes, a, bytes) ||
            lgn2_gpu_metal_ranges_overlap(norm_out, bytes, b, bytes) ||
            lgn2_gpu_metal_ranges_overlap(norm_out, bytes, c, bytes)) {
            fprintf(stderr,
                    "lgn2: Laguna add3+RMS norm received undersized or aliased activation buffers\n");
            return 0;
        }
        if (weight_offset > model_size ||
            row_bytes > model_size - weight_offset) {
            fprintf(stderr,
                    "lgn2: Laguna add3+RMS norm range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = lgn2_gpu_wrap_model_range(model_map,
                                                       model_size,
                                                       weight_offset,
                                                       row_bytes,
                                                       &inner_offset);
        if (!wbuf) return 0;

        lgn2_gpu_rms_norm_args args =
            lgn2_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        if (!enc) {
            if (owned) {
                (void)lgn2_gpu_finish_command_buffer(
                    cb, owned, "Laguna add3+RMS norm setup");
            }
            return 0;
        }

        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:abuf offset:lgn2_gpu_tensor_offset(a) atIndex:1];
        [enc setBuffer:bbuf offset:lgn2_gpu_tensor_offset(b) atIndex:2];
        [enc setBuffer:cbuf offset:lgn2_gpu_tensor_offset(c) atIndex:3];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:4];
        [enc setBuffer:sumbuf offset:lgn2_gpu_tensor_offset(sum_out) atIndex:5];
        [enc setBuffer:normbuf offset:lgn2_gpu_tensor_offset(norm_out) atIndex:6];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(lgn2_gpu_rms_norm_threads(n), 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(
                cb, owned, "Laguna add3+RMS norm")) return 0;
    }
    return 1;
}

/* Encode the Laguna argmax into an already-open compute encoder.  The Q8
 * lm-head screen uses this between its coarse and exact dispatches; keeping
 * it in the same command buffer preserves producer/consumer ordering for
 * both standalone calls and graph batches. */
static int lgn2_gpu_encode_laguna_argmax_encoder(
        id<MTLComputeCommandEncoder> enc,
        lgn2_gpu_tensor       *out_idx,
        const lgn2_gpu_tensor *logits,
        uint32_t              n_vocab) {
    enum { LGN2_LAGUNA_ARGMAX_THREADS = 256 };
    if (!enc || !g_laguna_argmax_f32_pipeline || !out_idx || !logits ||
        n_vocab == 0 ||
        lgn2_gpu_tensor_bytes(out_idx) < sizeof(int32_t) ||
        lgn2_gpu_tensor_bytes(logits) < (uint64_t)n_vocab * sizeof(float)) {
        return 0;
    }

    id<MTLBuffer> logitsbuf = lgn2_gpu_tensor_buffer(logits);
    id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out_idx);
    if (!logitsbuf || !outbuf) return 0;
    [enc setComputePipelineState:g_laguna_argmax_f32_pipeline];
    [enc setBuffer:logitsbuf
            offset:lgn2_gpu_tensor_offset(logits)
           atIndex:0];
    [enc setBuffer:outbuf
            offset:lgn2_gpu_tensor_offset(out_idx)
           atIndex:1];
    [enc setBytes:&n_vocab length:sizeof(n_vocab) atIndex:2];
    [enc setThreadgroupMemoryLength:
              (NSUInteger)LGN2_LAGUNA_ARGMAX_THREADS * sizeof(float)
                          atIndex:0];
    [enc setThreadgroupMemoryLength:
              (NSUInteger)LGN2_LAGUNA_ARGMAX_THREADS * sizeof(uint32_t)
                          atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(LGN2_LAGUNA_ARGMAX_THREADS, 1, 1)];
    return 1;
}

int lgn2_gpu_laguna_argmax_tensor(
        lgn2_gpu_tensor       *out_idx,
        const lgn2_gpu_tensor *logits,
        uint32_t              n_vocab) {
    if (!lgn2_gpu_laguna_argmax_available() ||
        !out_idx || !logits || n_vocab == 0 ||
        lgn2_gpu_tensor_bytes(out_idx) < sizeof(int32_t) ||
        lgn2_gpu_tensor_bytes(logits) < (uint64_t)n_vocab * sizeof(float)) {
        fprintf(stderr, "lgn2: Laguna GPU argmax received invalid buffers\n");
        return 0;
    }

    @autoreleasepool {
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        if (!enc) {
            if (owned) (void)lgn2_gpu_finish_command_buffer(
                cb, owned, "Laguna F32 argmax setup");
            return 0;
        }
        if (!lgn2_gpu_encode_laguna_argmax_encoder(enc, out_idx, logits,
                                                   n_vocab)) {
            lgn2_gpu_end_compute_encoder(cb, enc);
            if (owned) (void)lgn2_gpu_finish_command_buffer(
                cb, owned, "Laguna F32 argmax");
            return 0;
        }
        lgn2_gpu_end_compute_encoder(cb, enc);
        return lgn2_gpu_finish_command_buffer(cb, owned, "Laguna F32 argmax");
    }
}

typedef struct {
    uint32_t n_blocks;
    uint32_t packed_block_bytes;
} lgn2_gpu_laguna_q8_lmhead_pack_args;

typedef struct {
    uint32_t n_blocks;
    uint32_t n_rows;
    uint32_t packed_block_bytes;
    uint32_t pad0;
} lgn2_gpu_laguna_q8_lmhead_coarse_args;

typedef struct {
    uint32_t candidate_rows;
    uint32_t candidate_row_blocks;
    uint32_t coarse_nonfinite;
    uint32_t exact_row_blocks;
    float    winner_value;
    int32_t  winner_index;
    uint32_t pad1;
    /* Keep the complete v1 prefix ABI stable for parent-source overrides. */
    uint32_t compact_pair_count;
    uint32_t exact_dispatch_groups;
} lgn2_gpu_laguna_q8_lmhead_screen_stats_host;

typedef struct {
    uint32_t n_rows;
    uint32_t n_blocks;
    uint32_t pair_capacity;
} lgn2_gpu_laguna_q8_lmhead_compact_args;

struct lgn2_gpu_laguna_q8_lmhead_screen {
    const void *model_map;
    uint64_t model_size;
    uint64_t weight_offset;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t n_blocks;
    uint64_t packed_block_bytes;
    uint64_t dispatch_count;
    uint64_t last_stats_epoch;
    uint32_t collect_stats;
    uint64_t packed_bytes;
    double sidecopy_init_ms;
    uint32_t use_v2;
    uint32_t pair_capacity;
    lgn2_gpu_tensor *packed;
    lgn2_gpu_tensor *coarse;
    lgn2_gpu_tensor *delta;
    lgn2_gpu_tensor *candidate;
    lgn2_gpu_tensor *exact_values;
    lgn2_gpu_tensor *coarse_index;
    lgn2_gpu_tensor *pair_ids;
    lgn2_gpu_tensor *dispatch_args;
    lgn2_gpu_tensor *stats;
};

static void lgn2_gpu_laguna_q8_lmhead_screen_free_partial(
        lgn2_gpu_laguna_q8_lmhead_screen *screen) {
    if (!screen) return;
    lgn2_gpu_tensor_free(screen->packed);
    lgn2_gpu_tensor_free(screen->coarse);
    lgn2_gpu_tensor_free(screen->delta);
    lgn2_gpu_tensor_free(screen->candidate);
    lgn2_gpu_tensor_free(screen->exact_values);
    lgn2_gpu_tensor_free(screen->coarse_index);
    lgn2_gpu_tensor_free(screen->pair_ids);
    lgn2_gpu_tensor_free(screen->dispatch_args);
    lgn2_gpu_tensor_free(screen->stats);
    free(screen);
}

static int lgn2_gpu_laguna_q8_lmhead_screen_pipelines_ready(void) {
    return g_laguna_argmax_f32_pipeline &&
           lgn2_gpu_get_pipeline("kernel_laguna_q8_lmhead_pack") &&
           lgn2_gpu_get_mul_mv_pipeline(
               "kernel_laguna_q8_lmhead_coarse", 8) &&
           lgn2_gpu_get_mul_mv_pipeline(
               "kernel_laguna_q8_lmhead_seed_exact", 8) &&
           lgn2_gpu_get_pipeline("kernel_laguna_q8_lmhead_candidates") &&
           lgn2_gpu_get_mul_mv_pipeline(
               "kernel_laguna_q8_lmhead_exact_candidates", 8) &&
           lgn2_gpu_get_pipeline(
               "kernel_laguna_q8_lmhead_argmax_candidates");
}

static int lgn2_gpu_laguna_q8_lmhead_screen_v2_env_mode(void) {
    const char *value = getenv("LGN2_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2");
    if (!value || value[0] == '\0' || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") == 0) return 1;
    fprintf(stderr,
            "lgn2: invalid LGN2_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2='%s'; "
            "expected unset, empty, 0, or literal 1\n",
            value);
    return -1;
}

/* Fallback is opt-in separately from the strict v2 selector.  A v2 request
 * otherwise has require semantics: silently dispatching the old full-grid
 * exact stage would make a benchmark arm misrepresent what it tested. */
static int lgn2_gpu_laguna_q8_lmhead_screen_v2_fallback_requested(void) {
    const char *value = getenv(
        "LGN2_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2_FALLBACK");
    if (!value || value[0] == '\0' || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") == 0) return 1;
    fprintf(stderr,
            "lgn2: invalid LGN2_METAL_LAGUNA_Q8_LMHEAD_SCREEN_V2_FALLBACK='%s'; "
            "expected unset, empty, 0, or literal 1\n",
            value);
    return -1;
}

static int lgn2_gpu_laguna_q8_lmhead_indirect_dispatch_available(void) {
    /* Keep a deterministic test hook for the unavailable-device contract;
     * the Laguna runner deliberately targets Apple-Silicon GPU families that
     * guarantee indirect dispatch rather than inferring support from an old
     * macOS selector exposed by legacy Mac GPUs. */
    const char *disabled = getenv(
        "LGN2_METAL_LAGUNA_Q8_LMHEAD_V2_DISABLE_INDIRECT");
    if (disabled && strcmp(disabled, "1") == 0) return 0;
    if (@available(macOS 11.0, *)) {
        return [g_device supportsFamily:MTLGPUFamilyApple7] ? 1 : 0;
    }
    return 0;
}

static int lgn2_gpu_laguna_q8_lmhead_screen_v2_pipelines_ready(void) {
    return lgn2_gpu_laguna_q8_lmhead_screen_pipelines_ready() &&
           lgn2_gpu_get_pipeline(
               "kernel_laguna_q8_lmhead_compact_pairs") &&
           lgn2_gpu_get_mul_mv_pipeline(
               "kernel_laguna_q8_lmhead_exact_compacted", 8) &&
           lgn2_gpu_laguna_q8_lmhead_indirect_dispatch_available();
}

lgn2_gpu_laguna_q8_lmhead_screen *
lgn2_gpu_laguna_q8_lmhead_screen_create(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    weight_offset,
        uint64_t    in_dim,
        uint64_t    out_dim) {
    if (!g_initialized && !lgn2_gpu_init()) return NULL;
    if (lgn2_gpu_commands_active()) {
        fprintf(stderr,
                "lgn2: Laguna Q8 lm-head screen cannot be prepared during an active command batch\n");
        return NULL;
    }
    const int requested_v2 =
        lgn2_gpu_laguna_q8_lmhead_screen_v2_env_mode();
    const int allow_v2_fallback =
        lgn2_gpu_laguna_q8_lmhead_screen_v2_fallback_requested();
    if (requested_v2 < 0 || allow_v2_fallback < 0) return NULL;
    int use_v2 = requested_v2 > 0;
    lgn2_gpu_q8_decode_config q8_config;
    const int q8_config_valid =
        lgn2_gpu_q8_decode_config_snapshot(&q8_config);
    const int q8_rows_mode = q8_config_valid < 0 ? -1 : q8_config.q8_mv_rows;
    const char *q8_rows_value = q8_config_valid < 0 ?
        "invalid snapshot" : (q8_rows_mode == 2 ? "2" : "4");
    const int math_safe = g_metal_math_safe &&
        lgn2_gpu_env_bool("LGN2_METAL_MATH_SAFE") > 0;
    if (!model_map || model_size == 0 ||
        in_dim != 3072u || out_dim != 100352u ||
        (in_dim & 31u) != 0u ||
        !math_safe ||
        getenv("LGN2_METAL_Q8_DECODE_MPP") != NULL ||
        q8_rows_mode != 2 ||
        getenv("LGN2_METAL_ENABLE_OUTPUT_Q8_NR4") != NULL) {
        if (!g_metal_math_safe ||
            lgn2_gpu_env_bool("LGN2_METAL_MATH_SAFE") <= 0) {
            fprintf(stderr,
                    "lgn2: Laguna Q8 lm-head screen requires LGN2_METAL_MATH_SAFE=1 "
                    "at Metal library compile time and screen creation\n");
        }
        fprintf(stderr,
                "lgn2: Laguna Q8 lm-head screen requested with incompatible shape or dispatch environment\n");
        if (q8_rows_mode != 2) {
            fprintf(stderr,
                    "lgn2: Laguna Q8 lm-head screen requires "
                    "LGN2_METAL_Q8_MV_ROWS unset/empty or literal 2 "
                    "(got '%s')\n",
                    q8_rows_value);
        }
        return NULL;
    }

    const uint64_t n_blocks = in_dim / 32u;
    const uint64_t raw_row_bytes = n_blocks * 34u;
    const uint64_t packed_block_bytes = 18u;
    if (out_dim > UINT64_MAX / raw_row_bytes ||
        out_dim > UINT64_MAX / n_blocks ||
        out_dim * n_blocks > UINT64_MAX / packed_block_bytes) {
        fprintf(stderr, "lgn2: Laguna Q8 lm-head screen size overflow\n");
        return NULL;
    }
    const uint64_t weight_bytes = out_dim * raw_row_bytes;
    const uint64_t packed_bytes = out_dim * n_blocks * packed_block_bytes;
    if (weight_offset > model_size ||
        weight_bytes > model_size - weight_offset ||
        packed_bytes > (uint64_t)NSUIntegerMax) {
        fprintf(stderr, "lgn2: Laguna Q8 lm-head screen weight range is invalid\n");
        return NULL;
    }
    if (use_v2 && !lgn2_gpu_laguna_q8_lmhead_screen_v2_pipelines_ready()) {
        if (allow_v2_fallback > 0 &&
            lgn2_gpu_laguna_q8_lmhead_screen_pipelines_ready()) {
            fprintf(stderr,
                    "lgn2: Laguna Q8 lm-head screen v2 unavailable; "
                    "using explicit v1 fallback\n");
            use_v2 = 0;
        } else {
            fprintf(stderr,
                    "lgn2: Laguna Q8 lm-head screen v2 requires GPU "
                    "indirect dispatch and compact pipelines\n");
            return NULL;
        }
    }
    if (!lgn2_gpu_laguna_q8_lmhead_screen_pipelines_ready()) {
        fprintf(stderr,
                "lgn2: Laguna Q8 lm-head screen requested but a Metal pipeline is unavailable\n");
        return NULL;
    }

    uint64_t inner_offset = 0;
    id<MTLBuffer> weightbuf = lgn2_gpu_wrap_model_range(
        model_map, model_size, weight_offset, weight_bytes, &inner_offset);
    if (!weightbuf) return NULL;

    lgn2_gpu_laguna_q8_lmhead_screen *screen =
        calloc(1, sizeof(*screen));
    if (!screen) return NULL;
    screen->model_map = model_map;
    screen->model_size = model_size;
    screen->weight_offset = weight_offset;
    screen->in_dim = in_dim;
    screen->out_dim = out_dim;
    screen->n_blocks = n_blocks;
    screen->packed_block_bytes = packed_block_bytes;
    screen->collect_stats =
        (lgn2_gpu_env_bool("LGN2_METAL_LAGUNA_Q8_LMHEAD_TRACE") > 0 ||
         lgn2_gpu_env_bool("LGN2_TEST_LAGUNA_Q8_LMHEAD_SCREEN") > 0) ? 1u : 0u;
    const double init_t0 = lgn2_gpu_now_ms();
    screen->packed_bytes = packed_bytes;
    screen->use_v2 = use_v2 ? 1u : 0u;
    screen->pair_capacity =
        (uint32_t)((out_dim + 1u) / 2u);
    screen->packed = lgn2_gpu_tensor_alloc(packed_bytes);
    screen->coarse = lgn2_gpu_tensor_alloc(out_dim * sizeof(float));
    screen->delta = lgn2_gpu_tensor_alloc(out_dim * sizeof(float));
    screen->candidate = lgn2_gpu_tensor_alloc(out_dim * sizeof(uint8_t));
    screen->exact_values = lgn2_gpu_tensor_alloc(out_dim * sizeof(float));
    screen->coarse_index = lgn2_gpu_tensor_alloc(sizeof(int32_t));
    if (screen->use_v2) {
        screen->pair_ids = lgn2_gpu_tensor_alloc(
            (uint64_t)screen->pair_capacity * sizeof(uint32_t));
        screen->dispatch_args = lgn2_gpu_tensor_alloc(3u * sizeof(uint32_t));
    }
    screen->stats = lgn2_gpu_tensor_alloc(
        sizeof(lgn2_gpu_laguna_q8_lmhead_screen_stats_host));
    if (!screen->packed || !screen->coarse || !screen->delta ||
        !screen->candidate || !screen->exact_values ||
        !screen->coarse_index || (screen->use_v2 &&
        (!screen->pair_ids || !screen->dispatch_args)) || !screen->stats) {
        lgn2_gpu_laguna_q8_lmhead_screen_free_partial(screen);
        return NULL;
    }

    lgn2_gpu_laguna_q8_lmhead_pack_args args = {
        .n_blocks = (uint32_t)(out_dim * n_blocks),
        .packed_block_bytes = (uint32_t)packed_block_bytes,
    };
    int owned = 0;
    id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
    if (!cb) {
        lgn2_gpu_laguna_q8_lmhead_screen_free_partial(screen);
        return NULL;
    }
    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    id<MTLComputePipelineState> pipeline =
        lgn2_gpu_get_pipeline("kernel_laguna_q8_lmhead_pack");
    if (!enc || !pipeline) {
        lgn2_gpu_end_compute_encoder(cb, enc);
        if (owned) (void)lgn2_gpu_finish_command_buffer(
            cb, owned, "Laguna Q8 lm-head sidecopy setup");
        lgn2_gpu_laguna_q8_lmhead_screen_free_partial(screen);
        return NULL;
    }
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weightbuf offset:(NSUInteger)inner_offset atIndex:1];
    [enc setBuffer:lgn2_gpu_tensor_buffer(screen->packed)
            offset:lgn2_gpu_tensor_offset(screen->packed) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(
                                  ((NSUInteger)args.n_blocks + 255u) / 256u,
                                  1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    if (!lgn2_gpu_finish_command_buffer(cb, owned,
                                       "Laguna Q8 lm-head sidecopy")) {
        lgn2_gpu_laguna_q8_lmhead_screen_free_partial(screen);
        return NULL;
    }
    screen->sidecopy_init_ms = lgn2_gpu_now_ms() - init_t0;

    fprintf(stderr,
            "lgn2: Laguna Q8 lm-head screen prepared (packed %.2f MiB, "
            "%llu bytes, init %.3f ms, mode=%s)\n",
            (double)packed_bytes / (1024.0 * 1024.0),
            (unsigned long long)packed_bytes,
            screen->sidecopy_init_ms,
            screen->use_v2 ? "v2-indirect" : "v1-full-grid");
    return screen;
}

void lgn2_gpu_laguna_q8_lmhead_screen_destroy(
        lgn2_gpu_laguna_q8_lmhead_screen *screen) {
    lgn2_gpu_laguna_q8_lmhead_screen_free_partial(screen);
}

int lgn2_gpu_laguna_q8_lmhead_screen_v2_enabled(
        const lgn2_gpu_laguna_q8_lmhead_screen *screen) {
    return screen && screen->use_v2 != 0;
}

int lgn2_gpu_laguna_q8_lmhead_screen_tensor(
        lgn2_gpu_laguna_q8_lmhead_screen *screen,
        lgn2_gpu_tensor       *out_idx,
        lgn2_gpu_tensor       *out_value,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        const lgn2_gpu_tensor *x) {
    if (!screen || !out_idx || !model_map || !x ||
        model_map != screen->model_map ||
        model_size != screen->model_size ||
        weight_offset != screen->weight_offset ||
        lgn2_gpu_tensor_bytes(out_idx) < sizeof(int32_t) ||
        (out_value && lgn2_gpu_tensor_bytes(out_value) < sizeof(float)) ||
        lgn2_gpu_tensor_bytes(x) < screen->in_dim * sizeof(float)) {
        fprintf(stderr, "lgn2: Laguna Q8 lm-head screen received invalid buffers\n");
        return 0;
    }

    const uint64_t raw_row_bytes = screen->n_blocks * 34u;
    const uint64_t weight_bytes = screen->out_dim * raw_row_bytes;
    uint64_t inner_offset = 0;
    id<MTLBuffer> weightbuf = lgn2_gpu_wrap_model_range(
        model_map, model_size, weight_offset, weight_bytes, &inner_offset);
    id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
    id<MTLBuffer> outidxbuf = lgn2_gpu_tensor_buffer(out_idx);
    id<MTLBuffer> outvaluebuf = out_value ?
        lgn2_gpu_tensor_buffer(out_value) : nil;
    id<MTLBuffer> packedbuf = lgn2_gpu_tensor_buffer(screen->packed);
    id<MTLBuffer> coarsebuf = lgn2_gpu_tensor_buffer(screen->coarse);
    id<MTLBuffer> deltabuf = lgn2_gpu_tensor_buffer(screen->delta);
    id<MTLBuffer> candidatebuf = lgn2_gpu_tensor_buffer(screen->candidate);
    id<MTLBuffer> valuesbuf = lgn2_gpu_tensor_buffer(screen->exact_values);
    id<MTLBuffer> coarseidxbuf = lgn2_gpu_tensor_buffer(screen->coarse_index);
    id<MTLBuffer> pairidsbuf = screen->use_v2 ?
        lgn2_gpu_tensor_buffer(screen->pair_ids) : nil;
    id<MTLBuffer> dispatchbuf = screen->use_v2 ?
        lgn2_gpu_tensor_buffer(screen->dispatch_args) : nil;
    id<MTLBuffer> statsbuf = lgn2_gpu_tensor_buffer(screen->stats);
    if (!weightbuf || !xbuf || !outidxbuf || !packedbuf || !coarsebuf ||
        !deltabuf || !candidatebuf || !valuesbuf || !coarseidxbuf ||
        (screen->use_v2 && (!pairidsbuf || !dispatchbuf)) || !statsbuf) {
        return 0;
    }

    if (screen->collect_stats && g_batch_cb &&
        g_command_batch_epoch != 0 &&
        screen->last_stats_epoch == g_command_batch_epoch) {
        fprintf(stderr,
                "lgn2: Laguna Q8 lm-head screen trace allows only one call "
                "per active command batch\n");
        return 0;
    }

    if (screen->collect_stats) {
        lgn2_gpu_laguna_q8_lmhead_screen_stats_host zero_stats = {0};
        zero_stats.winner_index = -1;
        /* seed_exact always evaluates one complete NR0=2 pair.  Count it
         * once here; exact_candidates skips that pair and only counts the
         * additional admitted pairs below. */
        zero_stats.exact_row_blocks =
            2u * (uint32_t)screen->n_blocks;
        if (!lgn2_gpu_tensor_write(screen->stats, 0,
                                  &zero_stats, sizeof(zero_stats))) {
            return 0;
        }
    }
    if (screen->dispatch_count != UINT64_MAX) screen->dispatch_count++;

    id<MTLComputePipelineState> coarse_pipeline =
        lgn2_gpu_get_mul_mv_pipeline("kernel_laguna_q8_lmhead_coarse", 8);
    id<MTLComputePipelineState> seed_pipeline =
        lgn2_gpu_get_mul_mv_pipeline(
            "kernel_laguna_q8_lmhead_seed_exact", 8);
    id<MTLComputePipelineState> candidate_pipeline =
        lgn2_gpu_get_pipeline("kernel_laguna_q8_lmhead_candidates");
    id<MTLComputePipelineState> exact_pipeline =
        lgn2_gpu_get_mul_mv_pipeline(
            "kernel_laguna_q8_lmhead_exact_candidates", 8);
    id<MTLComputePipelineState> compact_pipeline = screen->use_v2 ?
        lgn2_gpu_get_pipeline("kernel_laguna_q8_lmhead_compact_pairs") : nil;
    id<MTLComputePipelineState> exact_compact_pipeline = screen->use_v2 ?
        lgn2_gpu_get_mul_mv_pipeline(
            "kernel_laguna_q8_lmhead_exact_compacted", 8) : nil;
    id<MTLComputePipelineState> reduce_pipeline =
        lgn2_gpu_get_pipeline("kernel_laguna_q8_lmhead_argmax_candidates");
    if (!coarse_pipeline || !seed_pipeline || !candidate_pipeline ||
        !exact_pipeline || !reduce_pipeline ||
        (screen->use_v2 && (!compact_pipeline || !exact_compact_pipeline ||
                            !lgn2_gpu_laguna_q8_lmhead_indirect_dispatch_available()))) {
        fprintf(stderr,
                "lgn2: Laguna Q8 lm-head screen pipeline became unavailable\n");
        return 0;
    }

    int owned = 0;
    id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    if (!enc) {
        if (owned) (void)lgn2_gpu_finish_command_buffer(
            cb, owned, "Laguna Q8 lm-head screen setup");
        return 0;
    }
    if (screen->collect_stats && g_batch_cb &&
        g_command_batch_epoch != 0) {
        screen->last_stats_epoch = g_command_batch_epoch;
    }

    /* This is an ordinary serial compute encoder: its dispatches execute in
     * encoded order, which supplies the coarse -> seed -> candidate -> exact
     * -> reduce dependencies.  A future concurrent encoder would need
     * explicit buffer barriers at each producer/consumer boundary. */

    lgn2_gpu_laguna_q8_lmhead_coarse_args coarse_args = {
        .n_blocks = (uint32_t)screen->n_blocks,
        .n_rows = (uint32_t)screen->out_dim,
        .packed_block_bytes = (uint32_t)screen->packed_block_bytes,
    };
    [enc setComputePipelineState:coarse_pipeline];
    [enc setBytes:&coarse_args length:sizeof(coarse_args) atIndex:0];
    [enc setBuffer:packedbuf offset:lgn2_gpu_tensor_offset(screen->packed)
           atIndex:1];
    [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
    [enc setBuffer:coarsebuf offset:lgn2_gpu_tensor_offset(screen->coarse)
           atIndex:3];
    [enc setBuffer:deltabuf offset:lgn2_gpu_tensor_offset(screen->delta)
           atIndex:4];
    [enc setThreadgroupMemoryLength:4u * 32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(
                                  ((NSUInteger)screen->out_dim + 1u) / 2u,
                                  1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];

    if (!lgn2_gpu_encode_laguna_argmax_encoder(
            enc, screen->coarse_index, screen->coarse,
            (uint32_t)screen->out_dim)) {
        lgn2_gpu_end_compute_encoder(cb, enc);
        if (owned) (void)lgn2_gpu_finish_command_buffer(
            cb, owned, "Laguna Q8 coarse argmax");
        return 0;
    }

    lgn2_gpu_q8_0_matvec_args exact_args = lgn2_gpu_make_q8_0_mv_args(
        screen->in_dim, screen->out_dim);
    [enc setComputePipelineState:seed_pipeline];
    [enc setBytes:&exact_args length:sizeof(exact_args) atIndex:0];
    [enc setBuffer:weightbuf offset:(NSUInteger)inner_offset atIndex:1];
    [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
    [enc setBuffer:valuesbuf offset:lgn2_gpu_tensor_offset(screen->exact_values)
           atIndex:3];
    [enc setBuffer:coarseidxbuf
           offset:lgn2_gpu_tensor_offset(screen->coarse_index) atIndex:4];
    [enc setThreadgroupMemoryLength:2u * 32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];

    const uint32_t n_rows = (uint32_t)screen->out_dim;
    const uint32_t n_blocks = (uint32_t)screen->n_blocks;
    [enc setComputePipelineState:candidate_pipeline];
    [enc setBytes:&n_rows length:sizeof(n_rows) atIndex:0];
    [enc setBytes:&n_blocks length:sizeof(n_blocks) atIndex:1];
    [enc setBuffer:coarsebuf offset:lgn2_gpu_tensor_offset(screen->coarse)
           atIndex:2];
    [enc setBuffer:deltabuf offset:lgn2_gpu_tensor_offset(screen->delta)
           atIndex:3];
    [enc setBuffer:valuesbuf offset:lgn2_gpu_tensor_offset(screen->exact_values)
           atIndex:4];
    [enc setBuffer:coarseidxbuf
           offset:lgn2_gpu_tensor_offset(screen->coarse_index) atIndex:5];
    [enc setBuffer:candidatebuf offset:lgn2_gpu_tensor_offset(screen->candidate)
           atIndex:6];
    [enc setBuffer:statsbuf offset:lgn2_gpu_tensor_offset(screen->stats)
           atIndex:7];
    [enc setBytes:&screen->collect_stats length:sizeof(screen->collect_stats)
           atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(
                                  ((NSUInteger)n_rows + 255u) / 256u,
                                  1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    if (screen->use_v2) {
        lgn2_gpu_laguna_q8_lmhead_compact_args compact_args = {
            .n_rows = n_rows,
            .n_blocks = n_blocks,
            .pair_capacity = screen->pair_capacity,
        };
        [enc setComputePipelineState:compact_pipeline];
        [enc setBytes:&compact_args length:sizeof(compact_args) atIndex:0];
        [enc setBuffer:candidatebuf
               offset:lgn2_gpu_tensor_offset(screen->candidate) atIndex:1];
        [enc setBuffer:pairidsbuf
               offset:lgn2_gpu_tensor_offset(screen->pair_ids) atIndex:2];
        [enc setBuffer:dispatchbuf
               offset:lgn2_gpu_tensor_offset(screen->dispatch_args) atIndex:3];
        [enc setBuffer:statsbuf offset:lgn2_gpu_tensor_offset(screen->stats)
               atIndex:4];
        [enc setBytes:&screen->collect_stats length:sizeof(screen->collect_stats)
               atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

        [enc setComputePipelineState:exact_compact_pipeline];
        [enc setBytes:&exact_args length:sizeof(exact_args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:valuesbuf
               offset:lgn2_gpu_tensor_offset(screen->exact_values) atIndex:3];
        [enc setBuffer:pairidsbuf
               offset:lgn2_gpu_tensor_offset(screen->pair_ids) atIndex:4];
        [enc setBuffer:coarseidxbuf
               offset:lgn2_gpu_tensor_offset(screen->coarse_index) atIndex:5];
        [enc setBuffer:statsbuf offset:lgn2_gpu_tensor_offset(screen->stats)
               atIndex:6];
        [enc setBytes:&screen->collect_stats length:sizeof(screen->collect_stats)
               atIndex:7];
        [enc setThreadgroupMemoryLength:2u * 32u * sizeof(float) atIndex:0];
        /* compact_pairs wrote exactly three u32s in this same serial
         * encoder.  No host readback or CPU-derived group count is involved. */
        [enc dispatchThreadgroupsWithIndirectBuffer:dispatchbuf
                                indirectBufferOffset:
                                    lgn2_gpu_tensor_offset(screen->dispatch_args)
                                threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    } else {
        [enc setComputePipelineState:exact_pipeline];
        [enc setBytes:&exact_args length:sizeof(exact_args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:valuesbuf offset:lgn2_gpu_tensor_offset(screen->exact_values)
               atIndex:3];
        [enc setBuffer:candidatebuf offset:lgn2_gpu_tensor_offset(screen->candidate)
               atIndex:4];
        [enc setBuffer:coarseidxbuf
               offset:lgn2_gpu_tensor_offset(screen->coarse_index) atIndex:5];
        [enc setBuffer:statsbuf offset:lgn2_gpu_tensor_offset(screen->stats)
               atIndex:6];
        [enc setBytes:&screen->collect_stats length:sizeof(screen->collect_stats)
               atIndex:7];
        [enc setThreadgroupMemoryLength:2u * 32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(
                                      ((NSUInteger)n_rows + 1u) / 2u,
                                      1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    }

    [enc setComputePipelineState:reduce_pipeline];
    [enc setBuffer:valuesbuf offset:lgn2_gpu_tensor_offset(screen->exact_values)
           atIndex:0];
    [enc setBuffer:candidatebuf offset:lgn2_gpu_tensor_offset(screen->candidate)
           atIndex:1];
    [enc setBuffer:outidxbuf offset:lgn2_gpu_tensor_offset(out_idx) atIndex:2];
    if (outvaluebuf) {
        [enc setBuffer:outvaluebuf offset:lgn2_gpu_tensor_offset(out_value)
               atIndex:3];
    } else {
        [enc setBuffer:statsbuf
               offset:lgn2_gpu_tensor_offset(screen->stats) +
                      offsetof(lgn2_gpu_laguna_q8_lmhead_screen_stats_host,
                               winner_value)
               atIndex:3];
    }
    [enc setBuffer:statsbuf offset:lgn2_gpu_tensor_offset(screen->stats)
           atIndex:4];
    [enc setBytes:&n_rows length:sizeof(n_rows) atIndex:5];
    [enc setBytes:&screen->collect_stats length:sizeof(screen->collect_stats)
           atIndex:6];
    [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
    [enc setThreadgroupMemoryLength:256u * sizeof(uint32_t) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return lgn2_gpu_finish_command_buffer(cb, owned,
                                         "Laguna Q8 lm-head screen");
}

int lgn2_gpu_laguna_q8_lmhead_screen_stats(
        const lgn2_gpu_laguna_q8_lmhead_screen *screen,
        uint32_t *candidate_rows,
        uint32_t *candidate_row_blocks,
        uint32_t *coarse_nonfinite,
        uint64_t *dispatch_count,
        uint64_t *packed_bytes,
        double   *sidecopy_init_ms,
        uint32_t *exact_row_blocks,
        int32_t  *winner_index,
        float    *winner_value) {
    if (!screen || !screen->collect_stats || !screen->stats) return 0;
    lgn2_gpu_laguna_q8_lmhead_screen_stats_host stats;
    if (!lgn2_gpu_tensor_read(screen->stats, 0, &stats, sizeof(stats))) {
        return 0;
    }
    if (candidate_rows) *candidate_rows = stats.candidate_rows;
    if (candidate_row_blocks) {
        *candidate_row_blocks = stats.candidate_row_blocks;
    }
    if (coarse_nonfinite) *coarse_nonfinite = stats.coarse_nonfinite;
    if (dispatch_count) *dispatch_count = screen->dispatch_count;
    if (packed_bytes) *packed_bytes = screen->packed_bytes;
    if (sidecopy_init_ms) *sidecopy_init_ms = screen->sidecopy_init_ms;
    if (exact_row_blocks) *exact_row_blocks = stats.exact_row_blocks;
    if (winner_index) *winner_index = stats.winner_index;
    if (winner_value) *winner_value = stats.winner_value;
    return 1;
}

int lgn2_gpu_laguna_q8_lmhead_screen_stats_v2(
        const lgn2_gpu_laguna_q8_lmhead_screen *screen,
        uint32_t *compact_pair_count,
        uint32_t *exact_dispatch_groups) {
    if (!screen || !screen->use_v2 || !screen->collect_stats ||
        !screen->stats) return 0;
    lgn2_gpu_laguna_q8_lmhead_screen_stats_host stats;
    if (!lgn2_gpu_tensor_read(screen->stats, 0, &stats, sizeof(stats))) {
        return 0;
    }
    if (compact_pair_count) *compact_pair_count = stats.compact_pair_count;
    if (exact_dispatch_groups) {
        *exact_dispatch_groups = stats.exact_dispatch_groups;
    }
    return 1;
}

static int lgn2_gpu_matmul_q8_0_legacy_tensor(
        lgn2_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        uint64_t                n_tok,
        int16_t                 short_row_nxpsg) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "lgn2: Metal Q8_0 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        const uint64_t row_bytes = blocks * 34;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "lgn2: Metal Q8_0 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = lgn2_gpu_wrap_model_range(model_map,
                                                      model_size,
                                                      weight_offset,
                                                      weight_bytes,
                                                      &inner_offset);
        if (!wbuf) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            lgn2_gpu_q8_0_matvec_args mv_args = lgn2_gpu_make_q8_0_mv_args(in_dim, out_dim);
            lgn2_gpu_mv_dispatch mv_dispatch = lgn2_gpu_make_q8_0_mv_dispatch();
            if (out_dim > 65536u) mv_dispatch.nsg = 8;
            mv_args.nr0 = mv_dispatch.nr0;
            id<MTLComputePipelineState> pipeline =
                lgn2_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
            if (!pipeline) return 0;

            id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            lgn2_gpu_end_compute_encoder(cb, enc);

            if (!lgn2_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matvec")) {
                return 0;
            }
            return 1;
        }

        const uint64_t mv_ext_max_tokens = g_q8_mv_ext_max_tokens;
        if (n_tok <= mv_ext_max_tokens && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = short_row_nxpsg != 0 ?
                short_row_nxpsg : lgn2_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = lgn2_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = lgn2_gpu_mv_ext_name(1, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? lgn2_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            lgn2_gpu_mul_mv_ext_args args =
                lgn2_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, 34, row_bytes);

            id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) / (NSUInteger)r0ptg,
                                                  ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) / (NSUInteger)r1ptg,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
            lgn2_gpu_end_compute_encoder(cb, enc);

            if (!lgn2_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor mul_mv_ext")) {
                return 0;
            }
            return 1;
        }

        /*
         * Dense Q8_0 prefill is the cleanest LGN2 TensorOps shape: M/N/K are
         * aligned and the RHS activation matrix is already dense.  The retained
         * kernel dequantizes each 64x32 weight tile to half in threadgroup
         * memory, then uses direct-RHS MPP for the activation tile.  This avoids
         * staging RHS into threadgroup memory and was the direct replacement for
         * the slower generic MPP prototype.
         */
        /*
         * An unaligned token count used to send the entire projection through
         * the generic kernel.  Run its aligned prefix through TensorOps and
         * leave only the final partial tile to the boundary-safe kernel.
         * Tiny prompts do not amortize the second dispatch, while --quality
         * deliberately retains the single-kernel arithmetic schedule.
         */
        const bool split_nax_prefix =
            !g_quality_mode && n_tok >= 192u && (n_tok % 32u) != 0u;
        const uint64_t nax_rows =
            (n_tok % 32u) == 0u ? n_tok :
            (split_nax_prefix ? n_tok - (n_tok % 32u) : 0u);
        uint64_t generic_row0 = 0u;
        uint64_t generic_rows = n_tok;
        if (lgn2_gpu_mpp_available() &&
            nax_rows >= 32u &&
            (in_dim % 64u) == 0 &&
            (out_dim % 64u) == 0) {
            uint64_t nax_tile_n = 32u;
            if ((nax_rows % 128u) == 0) {
                nax_tile_n = 128u;
            } else if ((nax_rows % 64u) == 0) {
                nax_tile_n = 64u;
            }
            const char *nax_fn = nax_tile_n == 128u
                ? "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128"
                : (nax_tile_n == 64u
                    ? "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64"
                    : "kernel_mul_mm_q8_0_f32_nax_direct_rhs");
            id<MTLComputePipelineState> pipeline =
                lgn2_gpu_get_mul_mm_pipeline(nax_fn, false, false);
            if (pipeline) {
                lgn2_gpu_mul_mm_args args =
                    lgn2_gpu_make_mm_args(in_dim, out_dim, nax_rows, row_bytes);

                id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:2u * 64u * 32u * sizeof(uint16_t) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(nax_rows / nax_tile_n),
                                                      (NSUInteger)out_dim / 64u,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                lgn2_gpu_end_compute_encoder(cb, enc);

                if (nax_rows == n_tok) {
                    if (!lgn2_gpu_finish_command_buffer(cb, owned, "Q8_0 NAX tensor matmul")) {
                        return 0;
                    }
                    return 1;
                }
                generic_row0 = nax_rows;
                generic_rows = n_tok - nax_rows;
            }
            if (!pipeline) lgn2_gpu_warn_mpp_fallback();
        }

        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out =
            (out_dim % 64u) != 0 || (generic_rows % 32u) != 0;
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_mul_mm_pipeline("kernel_mul_mm_q8_0_f32", bc_inp, bc_out);
        if (!pipeline) return 0;

        lgn2_gpu_mul_mm_args args =
            lgn2_gpu_make_mm_args(in_dim, out_dim, generic_rows, row_bytes);

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf
                offset:lgn2_gpu_tensor_offset(x) +
                       (NSUInteger)(generic_row0 * in_dim * sizeof(float))
               atIndex:2];
        [enc setBuffer:outbuf
                offset:lgn2_gpu_tensor_offset(out) +
                       (NSUInteger)(generic_row0 * out_dim * sizeof(float))
               atIndex:3];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)generic_rows + 31u) / 32u,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matmul")) {
            return 0;
        }
    }

    return 1;
}

int lgn2_gpu_matmul_q8_0_tensor(
        lgn2_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    const int profile_requested =
        n_tok > 8u && lgn2_gpu_env_bool("LGN2_METAL_Q8_PREFILL_PROFILE") > 0;
    int profile_prefill = 0;
    int split_batch_for_profile = 0;
    const char *profile_label = NULL;
    char profile_label_buf[128];
    char profile_fallback[128];
    if (profile_requested) {
        snprintf(profile_fallback, sizeof(profile_fallback),
                 "q8 weight_off=%llu in=%llu out=%llu tok=%llu",
                 (unsigned long long)weight_offset,
                 (unsigned long long)in_dim,
                 (unsigned long long)out_dim,
                 (unsigned long long)n_tok);
        snprintf(profile_label_buf, sizeof(profile_label_buf), "%s", profile_fallback);
        profile_label = profile_label_buf;
        const char *profile_filter = getenv("LGN2_METAL_Q8_PREFILL_PROFILE_FILTER");
        profile_prefill =
            profile_requested &&
            (!profile_filter || !profile_filter[0] ||
             strstr(profile_label, profile_filter) != NULL);
    }
    if (profile_prefill) {
        if (g_batch_cb) {
            if (lgn2_gpu_end_commands() == 0 || lgn2_gpu_begin_commands() == 0) {
                return 0;
            }
            split_batch_for_profile = 1;
        }
    }

    const double profile_t0 = profile_prefill ? lgn2_gpu_now_ms() : 0.0;
    int ok = lgn2_gpu_matmul_q8_0_legacy_tensor(out, model_map, model_size,
                                               weight_offset, in_dim, out_dim,
                                               x, n_tok, 0);
#ifdef LGN2_TEST_HOOKS
    if (ok && g_laguna_test_route_hooks) g_laguna_test_stock_q8_count++;
#endif
    if (profile_prefill) {
        if (split_batch_for_profile && lgn2_gpu_end_commands() == 0) {
            ok = 0;
        }
        const double elapsed_ms = lgn2_gpu_now_ms() - profile_t0;
        fprintf(stderr,
                "lgn2: Metal Q8_0 prefill profile %s in=%llu out=%llu tok=%llu %.3f ms\n",
                profile_label ? profile_label : profile_fallback,
                (unsigned long long)in_dim,
                (unsigned long long)out_dim,
                (unsigned long long)n_tok,
                elapsed_ms);
        if (split_batch_for_profile && lgn2_gpu_begin_commands() == 0) {
            ok = 0;
        }
    }
    return ok;
}

int lgn2_gpu_matmul_q8_0_dflash_tensor(
        lgn2_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        uint64_t                n_tok) {
    /*
     * DFlash always evaluates a very short row block. On Apple GPUs, assigning
     * 16 output lanes to each SIMD group keeps these small Q8 projections
     * better occupied than the generic shape heuristic.
     */
    return lgn2_gpu_matmul_q8_0_legacy_tensor(out, model_map, model_size,
                                             weight_offset, in_dim, out_dim,
                                             x, n_tok, 16);
}

int lgn2_gpu_matmul_q8_0_decode_rows_exact_tensor(
        lgn2_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const lgn2_gpu_tensor *x,
        uint32_t              n_rows) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !x || !model_map || n_rows == 0 ||
        n_rows > INT32_MAX || in_dim == 0 || out_dim == 0 ||
        (in_dim & 31u) != 0 || in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX ||
        in_dim > UINT64_MAX / n_rows / sizeof(float) ||
        out_dim > UINT64_MAX / n_rows / sizeof(float) ||
        lgn2_gpu_tensor_bytes(x) <
            (uint64_t)n_rows * in_dim * sizeof(float) ||
        lgn2_gpu_tensor_bytes(out) <
            (uint64_t)n_rows * out_dim * sizeof(float)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        const uint64_t blocks = in_dim / 32u;
        const uint64_t row_bytes = blocks * 34u;
        if (!xbuf || !outbuf ||
            out_dim > UINT64_MAX / row_bytes) {
            return 0;
        }
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = lgn2_gpu_wrap_model_range(
                model_map, model_size, weight_offset, weight_bytes,
                &inner_offset);
        if (!wbuf) return 0;

        lgn2_gpu_mv_dispatch dispatch = lgn2_gpu_make_q8_0_mv_dispatch();
        if (out_dim > 65536u) dispatch.nsg = 8;
        lgn2_gpu_q8_0_matvec_args args =
            lgn2_gpu_make_q8_0_mv_args(in_dim, out_dim);
        args.ne11 = (int32_t)n_rows;
        args.nb12 = (uint64_t)n_rows * in_dim * sizeof(float);
        args.nb13 = args.nb12;
        args.ne1 = (int32_t)n_rows;
        args.nr0 = dispatch.nr0;

        const uint32_t rows_per_group =
            n_rows >= 4u ? 4u : n_rows == 3u ? 3u :
            n_rows == 2u ? 2u : 1u;
        const char *function_name =
            rows_per_group == 4u ?
                "kernel_mul_mv_q8_0_f32_rows4_exact" :
            rows_per_group == 3u ?
                "kernel_mul_mv_q8_0_f32_rows3_exact" :
            rows_per_group == 2u ?
                "kernel_mul_mv_q8_0_f32_rows2_exact" :
                dispatch.function_name;
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_mul_mv_pipeline(function_name, dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:
            rows_per_group * dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:
                MTLSizeMake(((NSUInteger)out_dim +
                             (NSUInteger)dispatch.nr0 - 1u) /
                                (NSUInteger)dispatch.nr0,
                            ((NSUInteger)n_rows + rows_per_group - 1u) /
                                rows_per_group,
                            1)
             threadsPerThreadgroup:
                MTLSizeMake(32, (NSUInteger)dispatch.nsg, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        return lgn2_gpu_finish_command_buffer(
                cb, owned, "Q8_0 exact decode-row matvec");
    }
}

static const char *lgn2_gpu_q4_mv_ext_name(uint32_t weight_type, int16_t r1ptg) {
    const char *prefix = NULL;
    if (weight_type == LGN2_METAL_TENSOR_Q4_K) {
        prefix = "kernel_mul_mv_ext_q4_K_f32_r1_";
    } else if (weight_type == LGN2_METAL_TENSOR_Q4_0) {
        prefix = "kernel_mul_mv_ext_q4_0_f32_r1_";
    } else {
        return NULL;
    }
    switch (r1ptg) {
    case 1: return weight_type == LGN2_METAL_TENSOR_Q4_K ?
        "kernel_mul_mv_ext_q4_K_f32_r1_1" : "kernel_mul_mv_ext_q4_0_f32_r1_1";
    case 2: return weight_type == LGN2_METAL_TENSOR_Q4_K ?
        "kernel_mul_mv_ext_q4_K_f32_r1_2" : "kernel_mul_mv_ext_q4_0_f32_r1_2";
    case 3: return weight_type == LGN2_METAL_TENSOR_Q4_K ?
        "kernel_mul_mv_ext_q4_K_f32_r1_3" : "kernel_mul_mv_ext_q4_0_f32_r1_3";
    case 4: return weight_type == LGN2_METAL_TENSOR_Q4_K ?
        "kernel_mul_mv_ext_q4_K_f32_r1_4" : "kernel_mul_mv_ext_q4_0_f32_r1_4";
    case 5: return weight_type == LGN2_METAL_TENSOR_Q4_K ?
        "kernel_mul_mv_ext_q4_K_f32_r1_5" : "kernel_mul_mv_ext_q4_0_f32_r1_5";
    default:
        (void)prefix;
        return NULL;
    }
}

static const char *lgn2_gpu_q4_mm_name(uint32_t weight_type) {
    switch (weight_type) {
    case LGN2_METAL_TENSOR_Q4_0: return "kernel_mul_mm_q4_0_f32";
    case LGN2_METAL_TENSOR_Q4_K: return "kernel_mul_mm_q4_K_f32";
    default: return NULL;
    }
}

static const char *lgn2_gpu_q4_nax_name(uint32_t weight_type, uint64_t tile_n) {
    if (weight_type == LGN2_METAL_TENSOR_Q4_0) {
        return tile_n == 128u ? "kernel_mul_mm_q4_0_f32_nax_direct_rhs_n128" :
               tile_n == 64u  ? "kernel_mul_mm_q4_0_f32_nax_direct_rhs_n64" :
                                 "kernel_mul_mm_q4_0_f32_nax_direct_rhs";
    }
    if (weight_type == LGN2_METAL_TENSOR_Q4_K) {
        return tile_n == 128u ? "kernel_mul_mm_q4_K_f32_nax_direct_rhs_n128" :
               tile_n == 64u  ? "kernel_mul_mm_q4_K_f32_nax_direct_rhs_n64" :
                                 "kernel_mul_mm_q4_K_f32_nax_direct_rhs";
    }
    return NULL;
}

static int lgn2_gpu_matmul_quant_impl_tensor(
        lgn2_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        uint64_t                n_tok) {
    if (weight_type == LGN2_METAL_TENSOR_Q8_0) {
        return lgn2_gpu_matmul_q8_0_legacy_tensor(out,
                                                 model_map,
                                                 model_size,
                                                 weight_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x,
                                                 n_tok,
                                                 0);
    }
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !x || !model_map ||
        in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    uint64_t row_bytes = 0;
    if (!lgn2_gpu_quant_row_bytes(weight_type, (uint32_t)in_dim, &row_bytes)) {
        fprintf(stderr, "lgn2: Metal quant matmul received unsupported type/dim (%u, in=%llu)\n",
                weight_type,
                (unsigned long long)in_dim);
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "lgn2: Metal quant tensor matmul received undersized activation buffers\n");
            return 0;
        }

        if (out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "lgn2: Metal quant tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = lgn2_gpu_wrap_model_range(model_map,
                                                      model_size,
                                                      weight_offset,
                                                      weight_bytes,
                                                      &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        /*
         * Small-batch Q4_K goes to the classic (llama.cpp-style) matvec:
         * the mul_mv_ext family tops out around 220 GB/s on M5 for the GLM
         * DenseQ4 decode shapes while this impl streams 530-650 GB/s
         * (misc/q4mv_bench.m). Falls through to ext when unavailable.
         */
        if (weight_type == LGN2_METAL_TENSOR_Q4_K &&
            n_tok <= 8u &&
            (in_dim % 256u) == 0 &&
            getenv("LGN2_METAL_DISABLE_Q4_MV_CLASSIC") == NULL) {
            const int16_t nsg = 2;
            id<MTLComputePipelineState> pipeline =
                lgn2_gpu_get_mul_mv_ext_pipeline("kernel_mul_mv_q4_K_dense_f32", nsg, 8);
            if (pipeline) {
                lgn2_gpu_q8_0_matvec_args args = {
                    .ne00 = (int32_t)in_dim,
                    .ne01 = (int32_t)out_dim,
                    .ne02 = 1,
                    .nb00 = 1,
                    .nb01 = row_bytes,
                    .nb02 = row_bytes * out_dim,
                    .nb03 = row_bytes * out_dim,
                    .ne10 = (int32_t)in_dim,
                    .ne11 = (int32_t)n_tok,
                    .ne12 = 1,
                    .nb10 = sizeof(float),
                    .nb11 = in_dim * sizeof(float),
                    .nb12 = in_dim * n_tok * sizeof(float),
                    .nb13 = in_dim * n_tok * sizeof(float),
                    .ne0 = (int32_t)out_dim,
                    .ne1 = (int32_t)n_tok,
                    .nr0 = 2,
                    .r2 = 1,
                    .r3 = 1,
                };
                const uint64_t rows_ptg = (uint64_t)nsg * 2u;

                id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:32 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + rows_ptg - 1u) / rows_ptg,
                                                      (NSUInteger)n_tok,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
                lgn2_gpu_end_compute_encoder(cb, enc);

                if (!lgn2_gpu_finish_command_buffer(cb, owned, "Q4_K classic mul_mv")) return 0;
                return 1;
            }
        }

        if (n_tok <= 8u && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = lgn2_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = (n_tok == 1u) ? 1 : lgn2_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = lgn2_gpu_q4_mv_ext_name(weight_type, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? lgn2_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (pipeline) {
                const int16_t nypsg = 32 / nxpsg;
                const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
                lgn2_gpu_mul_mv_ext_args args =
                    lgn2_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, row_bytes, row_bytes);

                id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) / (NSUInteger)r0ptg,
                                                      ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) / (NSUInteger)r1ptg,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
                lgn2_gpu_end_compute_encoder(cb, enc);

                if (!lgn2_gpu_finish_command_buffer(cb, owned, "Q4 tensor mul_mv_ext")) return 0;
                return 1;
            }
        }

        if (lgn2_gpu_mpp_available() &&
            n_tok >= 32u &&
            (in_dim % 64u) == 0 &&
            (out_dim % 64u) == 0 &&
            (n_tok % 32u) == 0) {
            uint64_t nax_tile_n = 32u;
            if ((n_tok % 128u) == 0) {
                nax_tile_n = 128u;
            } else if ((n_tok % 64u) == 0) {
                nax_tile_n = 64u;
            }
            const char *nax_fn = lgn2_gpu_q4_nax_name(weight_type, nax_tile_n);
            id<MTLComputePipelineState> pipeline =
                nax_fn ? lgn2_gpu_get_mul_mm_pipeline(nax_fn, false, false) : nil;
            if (pipeline) {
                lgn2_gpu_mul_mm_args args = lgn2_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

                id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:64u * 32u * sizeof(uint16_t) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(n_tok / nax_tile_n),
                                                      (NSUInteger)out_dim / 64u,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                lgn2_gpu_end_compute_encoder(cb, enc);

                if (!lgn2_gpu_finish_command_buffer(cb, owned, "Q4 NAX tensor matmul")) return 0;
                return 1;
            }
            lgn2_gpu_warn_mpp_fallback();
        }

        const char *mm_fn = lgn2_gpu_q4_mm_name(weight_type);
        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;
        id<MTLComputePipelineState> pipeline =
            mm_fn ? lgn2_gpu_get_mul_mm_pipeline(mm_fn, bc_inp, bc_out) : nil;
        if (!pipeline) return 0;

        lgn2_gpu_mul_mm_args args = lgn2_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "Q4 tensor matmul")) return 0;
    }

    return 1;
}

int lgn2_gpu_matmul_quant_tensor(
        lgn2_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        uint64_t                n_tok) {
    return lgn2_gpu_matmul_quant_impl_tensor(out,
                                            model_map,
                                            model_size,
                                            weight_offset,
                                            weight_type,
                                            in_dim,
                                            out_dim,
                                            x,
                                            n_tok);
}

int lgn2_gpu_matmul_q6_K_tensor(
        lgn2_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const lgn2_gpu_tensor *x,
        uint64_t              n_tok) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !model_map || !x || n_tok == 0 ||
        in_dim == 0 || (in_dim % 256u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    const uint64_t row_bytes = (in_dim / 256u) * 210u;
    if (out_dim > UINT64_MAX / row_bytes ||
        n_tok > UINT64_MAX / in_dim ||
        n_tok > UINT64_MAX / out_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * row_bytes;
    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    if (weight_offset > model_size ||
        weight_bytes > model_size - weight_offset ||
        lgn2_gpu_tensor_bytes(x) < x_bytes ||
        lgn2_gpu_tensor_bytes(out) < out_bytes) {
        fprintf(stderr, "lgn2: Metal Q6_K matmul received an invalid model range or activation buffer\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = lgn2_gpu_wrap_model_range(model_map,
                                                       model_size,
                                                       weight_offset,
                                                       weight_bytes,
                                                       &inner_offset);
        id<MTLComputePipelineState> pipeline = lgn2_gpu_hot_pipeline(
            g_laguna_q6_k_matmul_pipeline,
            "kernel_laguna_q6_K_matmul_f32");
        if (!xbuf || !outbuf || !wbuf || !pipeline) return 0;

        lgn2_gpu_laguna_q6_matmul_args args = {
            .in_dim = (uint32_t)in_dim,
            .out_dim = (uint32_t)out_dim,
            .n_tokens = (uint32_t)n_tok,
            .row_bytes = row_bytes,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + 3u) / 4u,
                                              (NSUInteger)n_tok,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        if (!lgn2_gpu_finish_command_buffer(cb, owned, "Laguna Q6_K matmul")) return 0;
    }
    return 1;
}

int lgn2_gpu_matmul_q8_0_pair_tensor(
        lgn2_gpu_tensor       *out0,
        lgn2_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const lgn2_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out0 || !out1 || !model_map || !x || n_tok != 1 ||
        out0_dim == 0 || out1_dim == 0 || (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out0_dim > UINT32_MAX || out1_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> out0buf = lgn2_gpu_tensor_buffer(out0);
        id<MTLBuffer> out1buf = lgn2_gpu_tensor_buffer(out1);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out0_bytes = out0_dim * sizeof(float);
        const uint64_t out1_bytes = out1_dim * sizeof(float);
        if (!xbuf || !out0buf || !out1buf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(out0) < out0_bytes ||
            lgn2_gpu_tensor_bytes(out1) < out1_bytes) {
            fprintf(stderr, "lgn2: Metal paired Q8_0 matvec received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t weight0_bytes = out0_dim * row_bytes;
        const uint64_t weight1_bytes = out1_dim * row_bytes;
        if (weight0_offset > model_size || weight0_bytes > model_size - weight0_offset ||
            weight1_offset > model_size || weight1_bytes > model_size - weight1_offset) {
            fprintf(stderr, "lgn2: Metal paired Q8_0 matvec range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner0 = 0;
        uint64_t inner1 = 0;
        id<MTLBuffer> weight0buf =
            lgn2_gpu_wrap_model_range(model_map, model_size,
                                     weight0_offset, weight0_bytes, &inner0);
        id<MTLBuffer> weight1buf =
            lgn2_gpu_wrap_model_range(model_map, model_size,
                                     weight1_offset, weight1_bytes, &inner1);
        if (!weight0buf || !weight1buf) return 0;

        lgn2_gpu_mv_dispatch dispatch0 = lgn2_gpu_make_q8_0_mv_dispatch();
        lgn2_gpu_mv_dispatch dispatch1 = lgn2_gpu_make_q8_0_mv_dispatch();
        if (out0_dim > 65536u) dispatch0.nsg = 8;
        if (out1_dim > 65536u) dispatch1.nsg = 8;
        /* A common threadgroup shape is required to retain each standalone
         * reduction tree. Mixed 4/8-simdgroup extents use the existing fallback. */
        if (dispatch0.nsg != dispatch1.nsg) return 0;

        lgn2_gpu_q8_0_matvec_args args0 = lgn2_gpu_make_q8_0_mv_args(in_dim, out0_dim);
        lgn2_gpu_q8_0_matvec_args args1 = lgn2_gpu_make_q8_0_mv_args(in_dim, out1_dim);
        args0.nr0 = dispatch0.nr0;
        args1.nr0 = dispatch1.nr0;
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32_pair", dispatch0.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args0 length:sizeof(args0) atIndex:0];
        [enc setBytes:&args1 length:sizeof(args1) atIndex:1];
        [enc setBuffer:weight0buf offset:(NSUInteger)inner0 atIndex:2];
        [enc setBuffer:weight1buf offset:(NSUInteger)inner1 atIndex:3];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:4];
        [enc setBuffer:out0buf offset:lgn2_gpu_tensor_offset(out0) atIndex:5];
        [enc setBuffer:out1buf offset:lgn2_gpu_tensor_offset(out1) atIndex:6];
        [enc setThreadgroupMemoryLength:2u * dispatch0.smem atIndex:0];
        const uint64_t max_out_dim = out0_dim > out1_dim ? out0_dim : out1_dim;
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)max_out_dim +
                                               (NSUInteger)dispatch0.nr0 - 1u) /
                                              (NSUInteger)dispatch0.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)dispatch0.nsg, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "paired Q8_0 matvec")) return 0;
    }

    return 1;
}

int lgn2_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        lgn2_gpu_tensor       *out0,
        lgn2_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out0_dim,
        uint64_t              out1_dim,
        const lgn2_gpu_tensor *x,
        uint32_t              n_rows) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out0 || !out1 || !model_map || !x || n_rows == 0u ||
        out0_dim == 0u || out1_dim == 0u || (in_dim & 31u) != 0u ||
        in_dim > UINT32_MAX || out0_dim > UINT32_MAX ||
        out1_dim > UINT32_MAX ||
        in_dim > UINT64_MAX / n_rows / sizeof(float) ||
        out0_dim > UINT64_MAX / n_rows / sizeof(float) ||
        out1_dim > UINT64_MAX / n_rows / sizeof(float) ||
        lgn2_gpu_tensor_bytes(x) <
            (uint64_t)n_rows * in_dim * sizeof(float) ||
        lgn2_gpu_tensor_bytes(out0) <
            (uint64_t)n_rows * out0_dim * sizeof(float) ||
        lgn2_gpu_tensor_bytes(out1) <
            (uint64_t)n_rows * out1_dim * sizeof(float)) {
        return 0;
    }
    @autoreleasepool {
        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        if (out0_dim > UINT64_MAX / row_bytes ||
            out1_dim > UINT64_MAX / row_bytes) {
            return 0;
        }
        const uint64_t weight0_bytes = out0_dim * row_bytes;
        const uint64_t weight1_bytes = out1_dim * row_bytes;
        if (weight0_offset > model_size ||
            weight0_bytes > model_size - weight0_offset ||
            weight1_offset > model_size ||
            weight1_bytes > model_size - weight1_offset) {
            return 0;
        }

        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> out0buf = lgn2_gpu_tensor_buffer(out0);
        id<MTLBuffer> out1buf = lgn2_gpu_tensor_buffer(out1);
        uint64_t inner0 = 0;
        uint64_t inner1 = 0;
        id<MTLBuffer> weight0buf =
            lgn2_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight0_offset,
                                     weight0_bytes,
                                     &inner0);
        id<MTLBuffer> weight1buf =
            lgn2_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight1_offset,
                                     weight1_bytes,
                                     &inner1);
        if (!xbuf || !out0buf || !out1buf ||
            !weight0buf || !weight1buf) {
            return 0;
        }

        lgn2_gpu_mv_dispatch dispatch0 = lgn2_gpu_make_q8_0_mv_dispatch();
        lgn2_gpu_mv_dispatch dispatch1 = lgn2_gpu_make_q8_0_mv_dispatch();
        if (out0_dim > 65536u) dispatch0.nsg = 8;
        if (out1_dim > 65536u) dispatch1.nsg = 8;
        if (dispatch0.nsg != dispatch1.nsg) return 0;

        lgn2_gpu_q8_0_matvec_args args0 =
            lgn2_gpu_make_q8_0_mv_args(in_dim, out0_dim);
        lgn2_gpu_q8_0_matvec_args args1 =
            lgn2_gpu_make_q8_0_mv_args(in_dim, out1_dim);
        args0.ne11 = (int32_t)n_rows;
        args0.ne1 = (int32_t)n_rows;
        args0.nr0 = dispatch0.nr0;
        args1.ne11 = (int32_t)n_rows;
        args1.ne1 = (int32_t)n_rows;
        args1.nr0 = dispatch1.nr0;
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_mul_mv_pipeline(
                "kernel_mul_mv_q8_0_f32_pair", dispatch0.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args0 length:sizeof(args0) atIndex:0];
        [enc setBytes:&args1 length:sizeof(args1) atIndex:1];
        [enc setBuffer:weight0buf offset:(NSUInteger)inner0 atIndex:2];
        [enc setBuffer:weight1buf offset:(NSUInteger)inner1 atIndex:3];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:4];
        [enc setBuffer:out0buf offset:lgn2_gpu_tensor_offset(out0) atIndex:5];
        [enc setBuffer:out1buf offset:lgn2_gpu_tensor_offset(out1) atIndex:6];
        [enc setThreadgroupMemoryLength:2u * dispatch0.smem atIndex:0];
        const uint64_t max_out_dim =
            out0_dim > out1_dim ? out0_dim : out1_dim;
        [enc dispatchThreadgroups:
                MTLSizeMake(((NSUInteger)max_out_dim +
                             (NSUInteger)dispatch0.nr0 - 1u) /
                                (NSUInteger)dispatch0.nr0,
                            n_rows,
                            1)
             threadsPerThreadgroup:
                MTLSizeMake(32, (NSUInteger)dispatch0.nsg, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        return lgn2_gpu_finish_command_buffer(
                cb, owned, "paired Q8_0 exact decode-row matvec");
    }
}

static int lgn2_gpu_shared_gate_up_swiglu_q8_0_impl(
        lgn2_gpu_tensor       *gate,
        lgn2_gpu_tensor       *up,
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        float                   clamp,
        int                     store_gate_up) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!mid || !x || !model_map ||
        (store_gate_up && (!gate || !up)) ||
        in_dim == 0 || out_dim == 0 ||
        (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        (out_dim & 1u) != 0 ||
        in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float) ||
        !isfinite(clamp) || clamp < 0.0f) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = lgn2_gpu_tensor_buffer(mid);
        id<MTLBuffer> gatebuf = store_gate_up ?
            lgn2_gpu_tensor_buffer(gate) : midbuf;
        id<MTLBuffer> upbuf = store_gate_up ?
            lgn2_gpu_tensor_buffer(up) : midbuf;
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            (store_gate_up && lgn2_gpu_tensor_bytes(gate) < out_bytes) ||
            (store_gate_up && lgn2_gpu_tensor_bytes(up) < out_bytes) ||
            lgn2_gpu_tensor_bytes(mid) < out_bytes) {
            fprintf(stderr, "lgn2: Metal shared expert fused gate/up received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        if (blocks > UINT64_MAX / 34u) return 0;
        const uint64_t row_bytes = blocks * 34;
        if (row_bytes != 0 && out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
            up_offset > model_size || weight_bytes > model_size - up_offset) {
            fprintf(stderr, "lgn2: Metal shared expert fused gate/up range is outside the mapped model\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        id<MTLBuffer> gate_wbuf = lgn2_gpu_wrap_model_range(model_map,
                                                           model_size,
                                                           gate_offset,
                                                           weight_bytes,
                                                           &gate_inner);
        id<MTLBuffer> up_wbuf = lgn2_gpu_wrap_model_range(model_map,
                                                         model_size,
                                                         up_offset,
                                                         weight_bytes,
                                                         &up_inner);
        if (!gate_wbuf || !up_wbuf) return 0;

        lgn2_gpu_q8_0_matvec_args args = lgn2_gpu_make_q8_0_mv_args(in_dim, out_dim);
        lgn2_gpu_mv_dispatch mv_dispatch = lgn2_gpu_make_q8_0_mv_dispatch();
        args.nr0 = mv_dispatch.nr0;
        const char *fn_name = store_gate_up ?
            "kernel_dsv4_shared_gate_up_swiglu_q8_0" :
            "kernel_dsv4_shared_mid_swiglu_q8_0";
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_mul_mv_pipeline(fn_name, mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_wbuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_wbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:gatebuf offset:(store_gate_up ?
                                       lgn2_gpu_tensor_offset(gate) :
                                       lgn2_gpu_tensor_offset(mid)) atIndex:4];
        [enc setBuffer:upbuf offset:(store_gate_up ?
                                     lgn2_gpu_tensor_offset(up) :
                                     lgn2_gpu_tensor_offset(mid)) atIndex:5];
        [enc setBuffer:midbuf offset:lgn2_gpu_tensor_offset(mid) atIndex:6];
        [enc setBytes:&clamp length:sizeof(clamp) atIndex:7];
        [enc setThreadgroupMemoryLength:2u * mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                                  (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb,
                                           owned,
                                           store_gate_up ?
                                           "shared expert fused gate/up" :
                                           "shared expert fused mid")) {
            return 0;
        }
    }

    return 1;
}

/* Decode-only fusion of the router logits matvec with the shared-expert
 * gate/up SwiGLU: one dispatch instead of two on the same normalized FFN
 * input.  Bit-exact by construction (see metal/dense.metal).  Returns 1 on
 * success, 0 when the shape is unsupported (caller falls back), -1 on
 * error. */
static int lgn2_gpu_shared_q8_swiglu_pipeline_ready(
        const char *function_name,
        int16_t     nsg,
        NSUInteger  shared_bytes) {
    if (!function_name || nsg <= 0 || !g_device ||
        shared_bytes > NSUIntegerMax / 2u) {
        return 0;
    }
    id<MTLComputePipelineState> pipeline =
        lgn2_gpu_get_mul_mv_pipeline(function_name, nsg);
    if (!pipeline || pipeline.threadExecutionWidth != 32u) return 0;
    const NSUInteger threads = 32u * (NSUInteger)nsg;
    if (threads > pipeline.maxTotalThreadsPerThreadgroup ||
        (NSUInteger)(2u * shared_bytes) > [g_device maxThreadgroupMemoryLength]) {
        return 0;
    }
    return 1;
}

int lgn2_gpu_shared_mid_swiglu_q8_0_available(void) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    @autoreleasepool {
        const lgn2_gpu_mv_dispatch dispatch =
            lgn2_gpu_make_q8_0_mv_dispatch();
        /* The dense Q8 SwiGLU library currently exposes only the NR2
         * topology.  Keep this capability probe honest if the generic
         * dispatch policy changes later; there is no dense NR4 PSO today. */
        if (dispatch.nr0 != 2) return 0;
        return lgn2_gpu_shared_q8_swiglu_pipeline_ready(
                   "kernel_dsv4_shared_mid_swiglu_q8_0",
                   dispatch.nsg,
                   dispatch.smem);
    }
}

int lgn2_gpu_shared_gate_up_swiglu_q8_0_tensor(
        lgn2_gpu_tensor       *gate,
        lgn2_gpu_tensor       *up,
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        float                   clamp) {
    return lgn2_gpu_shared_gate_up_swiglu_q8_0_impl(gate,
                                                   up,
                                                   mid,
                                                   model_map,
                                                   model_size,
                                                   gate_offset,
                                                   up_offset,
                                                   in_dim,
                                                   out_dim,
                                                   x,
                                                   clamp,
                                                   1);
}

int lgn2_gpu_shared_mid_swiglu_q8_0_tensor(
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        float                   clamp) {
    return lgn2_gpu_shared_gate_up_swiglu_q8_0_impl(NULL,
                                                   NULL,
                                                   mid,
                                                   model_map,
                                                   model_size,
                                                   gate_offset,
                                                   up_offset,
                                                   in_dim,
                                                   out_dim,
                                                   x,
                                                   clamp,
                                                   0);
}

/* Batched sibling of the decode-only fused dense Q8 gate/up+SwiGLU: one
 * tiled pass computes both projections for all rows and applies SwiGLU in
 * the tile epilogue, so the normed rows are read once and the gate/up rows
 * never reach device memory.  The kernel mirrors kernel_mul_mm_q8_0_f32
 * tile-for-tile, which is the kernel ordinary prefill uses while TensorOps
 * is suppressed. */
static int lgn2_gpu_laguna_dense_q8_batch_pipeline_valid(
        id<MTLComputePipelineState> pipeline) {
    return pipeline && pipeline.threadExecutionWidth == 32u &&
        pipeline.maxTotalThreadsPerThreadgroup >= 128u &&
        (NSUInteger)(4u * 64u * 32u * sizeof(uint16_t)) <=
            [g_device maxThreadgroupMemoryLength];
}

int lgn2_gpu_laguna_dense_q8_gate_up_swiglu_batch_available(void) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    @autoreleasepool {
        /* bco is a real function-constant specialization: aligned production
         * shapes (out_dim % 64 == 0 and n_tok % 32 == 0) use false, while
         * ragged tails use true. Warm and validate both before graph capture
         * so a missing/old source cannot fail after attention or KV mutation. */
        for (int bco = 0; bco <= 1; bco++) {
            id<MTLComputePipelineState> pipeline =
                lgn2_gpu_get_mul_mm_pipeline(
                    "kernel_mul_mm_q8_0_f32_pair_swiglu", false, bco != 0);
            if (!lgn2_gpu_laguna_dense_q8_batch_pipeline_valid(pipeline)) {
                return 0;
            }
        }
        return 1;
    }
}

int lgn2_gpu_laguna_dense_q8_gate_up_swiglu_batch_tensor(
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!mid || !x || !model_map ||
        n_tok == 0 || n_tok > UINT32_MAX ||
        n_tok > (uint64_t)INT32_MAX ||
        in_dim == 0 || (in_dim & 31u) != 0 ||
        in_dim > (uint64_t)INT32_MAX ||
        out_dim == 0 || out_dim > (uint64_t)INT32_MAX) {
        return 0;
    }
    if (n_tok > UINT64_MAX / in_dim || n_tok > UINT64_MAX / out_dim ||
        n_tok * in_dim > UINT64_MAX / sizeof(float) ||
        n_tok * out_dim > UINT64_MAX / sizeof(float)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = lgn2_gpu_tensor_buffer(mid);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t mid_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !midbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(mid) < mid_bytes) {
            fprintf(stderr, "lgn2: Metal batched fused Q8_0 gate/up SwiGLU received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        if (blocks > UINT64_MAX / 34u) return 0;
        const uint64_t row_bytes = blocks * 34;
        if (row_bytes != 0 && out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (gate_offset > model_size ||
            weight_bytes > model_size - gate_offset ||
            up_offset > model_size ||
            weight_bytes > model_size - up_offset) {
            fprintf(stderr, "lgn2: Metal batched fused Q8_0 gate/up SwiGLU range is outside the mapped model\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        id<MTLBuffer> gate_wbuf =
            lgn2_gpu_wrap_model_range(model_map, model_size, gate_offset, weight_bytes, &gate_inner);
        id<MTLBuffer> up_wbuf =
            lgn2_gpu_wrap_model_range(model_map, model_size, up_offset, weight_bytes, &up_inner);
        if (!gate_wbuf || !up_wbuf) return 0;

        const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;
        id<MTLComputePipelineState> pipeline = lgn2_gpu_get_mul_mm_pipeline(
            "kernel_mul_mm_q8_0_f32_pair_swiglu", false, bc_out);
        if (!pipeline) return 0;

        lgn2_gpu_mul_mm_args args =
            lgn2_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_wbuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_wbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:midbuf offset:lgn2_gpu_tensor_offset(mid) atIndex:4];
        [enc setThreadgroupMemoryLength:4u * 64u * 32u * sizeof(uint16_t)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned,
                                           "Laguna batched fused Q8_0 gate/up SwiGLU")) {
            return 0;
        }
#ifdef LGN2_TEST_HOOKS
        if (g_laguna_test_route_hooks) {
            g_laguna_test_fused_q8_count++;
            if (bc_out) g_laguna_test_fused_q8_bco_true_count++;
            else g_laguna_test_fused_q8_bco_false_count++;
        }
#endif
    }

    return 1;
}

int lgn2_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        lgn2_gpu_tensor       *gate,
        lgn2_gpu_tensor       *up,
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        float                   clamp) {
    return lgn2_gpu_shared_gate_up_swiglu_q8_0_impl(gate,
                                                   up,
                                                   mid,
                                                   model_map,
                                                   model_size,
                                                   gate_offset,
                                                   up_offset,
                                                   in_dim,
                                                   out_dim,
                                                   x,
                                                   clamp,
                                                   1);
}

int lgn2_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        lgn2_gpu_tensor       *gate,
        lgn2_gpu_tensor       *up,
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    if (n_tok == 1) {
        return lgn2_gpu_shared_gate_up_swiglu_q8_0_tensor(gate,
                                                         up,
                                                         mid,
                                                         model_map,
                                                         model_size,
                                                         gate_offset,
                                                         up_offset,
                                                         in_dim,
                                                         out_dim,
                                                         x,
                                                         clamp);
    }
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!gate || !up || !mid || !x || !model_map ||
        n_tok == 0 || in_dim == 0 || out_dim == 0 ||
        (in_dim & 31u) != 0 || (in_dim % 128u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        (out_dim & 1u) != 0 ||
        in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float) ||
        !isfinite(clamp) || clamp < 0.0f) {
        return 0;
    }

    const uint64_t mv_ext_max_tokens = g_q8_mv_ext_max_tokens;
    if (n_tok > mv_ext_max_tokens) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = lgn2_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = lgn2_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = lgn2_gpu_tensor_buffer(mid);
        const uint64_t x_row_bytes = in_dim * sizeof(float);
        const uint64_t out_row_bytes = out_dim * sizeof(float);
        if ((x_row_bytes != 0 && n_tok > UINT64_MAX / x_row_bytes) ||
            (out_row_bytes != 0 && n_tok > UINT64_MAX / out_row_bytes)) {
            return 0;
        }
        const uint64_t x_bytes = n_tok * x_row_bytes;
        const uint64_t out_bytes = n_tok * out_row_bytes;
        if (!xbuf || !gatebuf || !upbuf || !midbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(gate) < out_bytes ||
            lgn2_gpu_tensor_bytes(up) < out_bytes ||
            lgn2_gpu_tensor_bytes(mid) < out_bytes) {
            fprintf(stderr, "lgn2: Metal fused Q8_0 gate/up rows received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        if (blocks > UINT64_MAX / 34u) return 0;
        const uint64_t row_bytes = blocks * 34;
        if (row_bytes != 0 && out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
            up_offset > model_size || weight_bytes > model_size - up_offset) {
            fprintf(stderr, "lgn2: Metal fused Q8_0 gate/up rows range is outside the mapped model\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        id<MTLBuffer> gate_wbuf =
            lgn2_gpu_wrap_model_range(model_map, model_size, gate_offset, weight_bytes, &gate_inner);
        id<MTLBuffer> up_wbuf =
            lgn2_gpu_wrap_model_range(model_map, model_size, up_offset, weight_bytes, &up_inner);
        if (!gate_wbuf || !up_wbuf) return 0;

        const int16_t nsg = 2;
        const int16_t nxpsg = lgn2_gpu_mv_ext_nxpsg(in_dim, n_tok);
        const int16_t r1ptg = lgn2_gpu_mv_ext_r1ptg(n_tok);
        const char *fn_name = lgn2_gpu_mv_ext_q8_pair_swiglu_name(r1ptg);
        id<MTLComputePipelineState> pipeline =
            fn_name ? lgn2_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
        if (!pipeline) return 0;

        const int16_t nypsg = 32 / nxpsg;
        const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
        lgn2_gpu_mul_mv_ext_args args =
            lgn2_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, 34, row_bytes);

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_wbuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_wbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:gatebuf offset:lgn2_gpu_tensor_offset(gate) atIndex:4];
        [enc setBuffer:upbuf offset:lgn2_gpu_tensor_offset(up) atIndex:5];
        [enc setBuffer:midbuf offset:lgn2_gpu_tensor_offset(mid) atIndex:6];
        [enc setBytes:&clamp length:sizeof(clamp) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) /
                                                  (NSUInteger)r0ptg,
                                              ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) /
                                                  (NSUInteger)r1ptg,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "fused Q8_0 gate/up rows")) {
            return 0;
        }
    }

    return 1;
}

int lgn2_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        lgn2_gpu_tensor       *gate,
        lgn2_gpu_tensor       *up,
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    if (n_tok == 1) {
        return lgn2_gpu_shared_gate_up_swiglu_q8_0_tensor(gate,
                                                         up,
                                                         mid,
                                                         model_map,
                                                         model_size,
                                                         gate_offset,
                                                         up_offset,
                                                         in_dim,
                                                         out_dim,
                                                         x,
                                                         clamp);
    }
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!gate || !up || !mid || !x || !model_map ||
        n_tok == 0 || in_dim == 0 || out_dim == 0 ||
        (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        (out_dim & 1u) != 0 ||
        in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float) ||
        !isfinite(clamp) || clamp < 0.0f) {
        return 0;
    }

    const uint64_t x_row_bytes = in_dim * sizeof(float);
    const uint64_t out_row_bytes = out_dim * sizeof(float);
    if ((x_row_bytes != 0 && n_tok > UINT64_MAX / x_row_bytes) ||
        (out_row_bytes != 0 && n_tok > UINT64_MAX / out_row_bytes)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = lgn2_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = lgn2_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = lgn2_gpu_tensor_buffer(mid);
        const uint64_t x_bytes = n_tok * x_row_bytes;
        const uint64_t out_bytes = n_tok * out_row_bytes;
        if (!xbuf || !gatebuf || !upbuf || !midbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(gate) < out_bytes ||
            lgn2_gpu_tensor_bytes(up) < out_bytes ||
            lgn2_gpu_tensor_bytes(mid) < out_bytes) {
            fprintf(stderr, "lgn2: Metal shared expert scalar-row fused gate/up received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        if (blocks > UINT64_MAX / 34u) return 0;
        const uint64_t row_bytes = blocks * 34u;
        if (out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
            up_offset > model_size || weight_bytes > model_size - up_offset) {
            fprintf(stderr, "lgn2: Metal shared expert scalar-row fused gate/up range is outside the mapped model\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        id<MTLBuffer> gate_wbuf =
            lgn2_gpu_wrap_model_range(model_map, model_size, gate_offset, weight_bytes, &gate_inner);
        id<MTLBuffer> up_wbuf =
            lgn2_gpu_wrap_model_range(model_map, model_size, up_offset, weight_bytes, &up_inner);
        if (!gate_wbuf || !up_wbuf) return 0;

        lgn2_gpu_q8_0_matvec_args args = lgn2_gpu_make_q8_0_mv_args(in_dim, out_dim);
        lgn2_gpu_mv_dispatch mv_dispatch = lgn2_gpu_make_q8_0_mv_dispatch();
        args.nr0 = mv_dispatch.nr0;
        args.ne11 = (int32_t)n_tok;
        args.nb12 = n_tok * x_row_bytes;
        args.nb13 = args.nb12;
        args.ne1 = (int32_t)n_tok;
        const char *fn_name = "kernel_dsv4_shared_gate_up_swiglu_q8_0";
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_mul_mv_pipeline(fn_name, mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_wbuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_wbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:gatebuf offset:lgn2_gpu_tensor_offset(gate) atIndex:4];
        [enc setBuffer:upbuf offset:lgn2_gpu_tensor_offset(up) atIndex:5];
        [enc setBuffer:midbuf offset:lgn2_gpu_tensor_offset(mid) atIndex:6];
        [enc setBytes:&clamp length:sizeof(clamp) atIndex:7];
        [enc setThreadgroupMemoryLength:2u * mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(
                ((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                    (NSUInteger)mv_dispatch.nr0,
                (NSUInteger)n_tok,
                1)
             threadsPerThreadgroup:
                MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "shared expert scalar-row fused gate/up")) {
            return 0;
        }
    }

    return 1;
}

int lgn2_gpu_matmul_f16_tensor(
        lgn2_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "lgn2: Metal F16 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(uint16_t);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "lgn2: Metal F16 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            lgn2_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            lgn2_gpu_f16_matvec_args mv_args = lgn2_gpu_make_f16_mv_args(in_dim, out_dim);
            lgn2_gpu_mv_dispatch mv_dispatch =
                lgn2_gpu_make_plain_mv_dispatch(in_dim, 0);
            if (!g_quality_mode && (out_dim == 512u || out_dim == 1024u) && in_dim >= 4096u) {
                mv_dispatch.nr0 = 4;
                mv_dispatch.smem = 32u * 4u * sizeof(float);
            }
            mv_args.nr0 = mv_dispatch.nr0;
            id<MTLComputePipelineState> pipeline =
                lgn2_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
            if (!pipeline) return 0;

            id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
            if (mv_dispatch.smem) {
                [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            lgn2_gpu_end_compute_encoder(cb, enc);

            if (!lgn2_gpu_finish_command_buffer(cb, owned, "F16 tensor matvec")) return 0;
            return 1;
        }

        if (n_tok <= 8 && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = lgn2_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = lgn2_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = lgn2_gpu_mv_ext_name(0, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? lgn2_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            lgn2_gpu_mul_mv_ext_args args =
                lgn2_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, sizeof(uint16_t), row_bytes);

            id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) / (NSUInteger)r0ptg,
                                                  ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) / (NSUInteger)r1ptg,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
            lgn2_gpu_end_compute_encoder(cb, enc);

            if (!lgn2_gpu_finish_command_buffer(cb, owned, "F16 tensor mul_mv_ext")) return 0;
            return 1;
        }

        /*
         * Same direct-RHS TensorOps structure as Q8_0, but for F16 model
         * matrices.  The 128-token RHS tile is kept when the batch alignment
         * allows it because the later tile_n=64 retest was neutral/slower.
         */
        if (lgn2_gpu_mpp_available() &&
            n_tok >= 32u &&
            (in_dim % 32u) == 0 &&
            (out_dim % 64u) == 0 &&
            (n_tok % 32u) == 0) {
            uint64_t nax_tile_n = 32u;
            if ((n_tok % 128u) == 0) {
                nax_tile_n = 128u;
            } else if ((n_tok % 64u) == 0) {
                nax_tile_n = 64u;
            }
            const char *nax_fn = nax_tile_n == 128u
                ? "kernel_mul_mm_f16_f32_mpp_direct_rhs_n128"
                : (nax_tile_n == 64u
                    ? "kernel_mul_mm_f16_f32_mpp_direct_rhs_n64"
                    : "kernel_mul_mm_f16_f32_mpp_direct_rhs");
            id<MTLComputePipelineState> pipeline =
                lgn2_gpu_get_mul_mm_pipeline(nax_fn, false, false);
            if (pipeline) {
                lgn2_gpu_mul_mm_args args = lgn2_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

                id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:2u * 64u * 32u * sizeof(uint16_t) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(n_tok / nax_tile_n),
                                                      (NSUInteger)out_dim / 64u,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                lgn2_gpu_end_compute_encoder(cb, enc);

                if (!lgn2_gpu_finish_command_buffer(cb, owned, "F16 NAX tensor matmul")) {
                    return 0;
                }
                return 1;
            }
            lgn2_gpu_warn_mpp_fallback();
        }

        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_mul_mm_pipeline("kernel_mul_mm_f16_f32", bc_inp, bc_out);
        if (!pipeline) return 0;

        lgn2_gpu_mul_mm_args args = lgn2_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "F16 tensor matmul")) return 0;
    }

    return 1;
}


int lgn2_gpu_matmul_f32_tensor(
        lgn2_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const lgn2_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok == 0 || n_tok > UINT32_MAX) return 0;
    if (in_dim > UINT64_MAX / n_tok / sizeof(float) ||
        out_dim > UINT64_MAX / n_tok / sizeof(float)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "lgn2: Metal F32 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(float);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "lgn2: Metal F32 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            lgn2_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            lgn2_gpu_q8_0_matvec_args mv_args = lgn2_gpu_make_f32_mv_args(in_dim, out_dim, 1);
            lgn2_gpu_mv_dispatch mv_dispatch = lgn2_gpu_make_plain_mv_dispatch(in_dim, 1);
            mv_args.nr0 = mv_dispatch.nr0;
            id<MTLComputePipelineState> pipeline =
                lgn2_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
            if (!pipeline) return 0;

            id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
            if (mv_dispatch.smem) {
                [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            lgn2_gpu_end_compute_encoder(cb, enc);

            if (!lgn2_gpu_finish_command_buffer(cb, owned, "F32 tensor matvec")) return 0;
            return 1;
        }

        if (n_tok <= 8 && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = lgn2_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = lgn2_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = lgn2_gpu_mv_ext_f32_name(r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? lgn2_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            lgn2_gpu_mul_mv_ext_args args =
                lgn2_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, sizeof(float), row_bytes);

            id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) / (NSUInteger)r0ptg,
                                                  ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) / (NSUInteger)r1ptg,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
            lgn2_gpu_end_compute_encoder(cb, enc);

            if (!lgn2_gpu_finish_command_buffer(cb, owned, "F32 tensor mul_mv_ext")) return 0;
            return 1;
        }

        /* Generic multi-row path (GLM prefill shapes: one grid row per
         * token through the plain matvec pipeline). */
        {
            lgn2_gpu_q8_0_matvec_args mv_args = lgn2_gpu_make_f32_mv_args(in_dim, out_dim, n_tok);
            lgn2_gpu_mv_dispatch mv_dispatch = lgn2_gpu_make_plain_mv_dispatch(in_dim, 1);
            mv_args.nr0 = mv_dispatch.nr0;
            id<MTLComputePipelineState> pipeline =
                lgn2_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
            if (!pipeline) return 0;

            id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
            if (mv_dispatch.smem) {
                [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  (NSUInteger)n_tok,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            lgn2_gpu_end_compute_encoder(cb, enc);

            if (!lgn2_gpu_finish_command_buffer(cb, owned, "F32 tensor matmul")) return 0;
        }
    }

    return 1;
}

int lgn2_gpu_matmul_f32_decode_rows_exact_tensor(
        lgn2_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const lgn2_gpu_tensor *x,
        uint32_t              n_rows) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !x || !model_map || n_rows == 0u ||
        in_dim == 0u || out_dim == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        in_dim > UINT64_MAX / n_rows / sizeof(float) ||
        out_dim > UINT64_MAX / n_rows / sizeof(float) ||
        lgn2_gpu_tensor_bytes(x) <
            (uint64_t)n_rows * in_dim * sizeof(float) ||
        lgn2_gpu_tensor_bytes(out) <
            (uint64_t)n_rows * out_dim * sizeof(float)) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t row_bytes = in_dim * sizeof(float);
        if (out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            return 0;
        }

        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = lgn2_gpu_wrap_model_range(model_map,
                                                       model_size,
                                                       weight_offset,
                                                       weight_bytes,
                                                       &inner_offset);
        if (!xbuf || !outbuf || !wbuf) return 0;

        lgn2_gpu_q8_0_matvec_args args =
            lgn2_gpu_make_f32_mv_args(in_dim, out_dim, n_rows);
        lgn2_gpu_mv_dispatch dispatch =
            lgn2_gpu_make_plain_mv_dispatch(in_dim, 1);
        args.nr0 = dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_mul_mv_pipeline(dispatch.function_name, dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
        if (dispatch.smem) {
            [enc setThreadgroupMemoryLength:dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:
                MTLSizeMake(((NSUInteger)out_dim +
                             (NSUInteger)dispatch.nr0 - 1u) /
                                (NSUInteger)dispatch.nr0,
                            n_rows,
                            1)
             threadsPerThreadgroup:
                MTLSizeMake(32, (NSUInteger)dispatch.nsg, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        return lgn2_gpu_finish_command_buffer(
                cb, owned, "F32 exact decode-row matvec");
    }
}

int lgn2_gpu_rms_norm_weight_tensor(
        lgn2_gpu_tensor       *out,
        const lgn2_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    return lgn2_gpu_rms_norm_weight_rows_tensor(out, x, model_map, model_size, weight_offset, n, 1, eps);
}

int lgn2_gpu_rms_norm_weight_rows_tensor(
        lgn2_gpu_tensor       *out,
        const lgn2_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        const uint64_t bytes = row_bytes * rows;
        if (!xbuf || !outbuf ||
            lgn2_gpu_tensor_bytes(x) < bytes ||
            lgn2_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "lgn2: Metal weighted RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (weight_offset > model_size || row_bytes > model_size - weight_offset) {
            fprintf(stderr, "lgn2: Metal weighted RMS norm range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = lgn2_gpu_wrap_model_range(model_map,
                                                       model_size,
                                                       weight_offset,
                                                       row_bytes,
                                                       &inner_offset);
        if (!wbuf) return 0;

        lgn2_gpu_rms_norm_args args = lgn2_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(lgn2_gpu_rms_norm_threads(n), 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "weighted RMS norm")) return 0;
    }

    return 1;
}

int lgn2_gpu_add_rms_norm_weight_rows_tensor(
        lgn2_gpu_tensor       *norm_out,
        lgn2_gpu_tensor       *sum_out,
        const lgn2_gpu_tensor *a,
        const lgn2_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!norm_out || !sum_out || !a || !b || n == 0 || rows == 0 ||
        (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> abuf = lgn2_gpu_tensor_buffer(a);
        id<MTLBuffer> bbuf = lgn2_gpu_tensor_buffer(b);
        id<MTLBuffer> sumbuf = lgn2_gpu_tensor_buffer(sum_out);
        id<MTLBuffer> normbuf = lgn2_gpu_tensor_buffer(norm_out);
        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        if ((uint64_t)rows > UINT64_MAX / row_bytes) {
            fprintf(stderr, "lgn2: Metal add+RMS norm row count overflows activation size\n");
            return 0;
        }
        const uint64_t bytes = row_bytes * rows;
        if (!abuf || !bbuf || !sumbuf || !normbuf ||
            lgn2_gpu_tensor_bytes(a) < bytes ||
            lgn2_gpu_tensor_bytes(b) < bytes ||
            lgn2_gpu_tensor_bytes(sum_out) < bytes ||
            lgn2_gpu_tensor_bytes(norm_out) < bytes) {
            fprintf(stderr, "lgn2: Metal add+RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (weight_offset > model_size || row_bytes > model_size - weight_offset) {
            fprintf(stderr, "lgn2: Metal add+RMS norm range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = lgn2_gpu_wrap_model_range(model_map,
                                                       model_size,
                                                       weight_offset,
                                                       row_bytes,
                                                       &inner_offset);
        if (!wbuf) return 0;

        lgn2_gpu_rms_norm_args args = lgn2_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_add_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:abuf offset:lgn2_gpu_tensor_offset(a) atIndex:1];
        [enc setBuffer:bbuf offset:lgn2_gpu_tensor_offset(b) atIndex:2];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:3];
        [enc setBuffer:sumbuf offset:lgn2_gpu_tensor_offset(sum_out) atIndex:4];
        [enc setBuffer:normbuf offset:lgn2_gpu_tensor_offset(norm_out) atIndex:5];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(lgn2_gpu_rms_norm_threads(n), 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "add+RMS norm")) return 0;
    }

    return 1;
}

static NSUInteger lgn2_gpu_align_up_ns(NSUInteger value, NSUInteger align) {
    return (value + align - 1u) & ~(align - 1u);
}

static int lgn2_gpu_encode_cpy_f32_f32_3d_src_strided(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_col_stride,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    lgn2_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = src_col_stride,
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = lgn2_gpu_cpy_threads(cols, g_cpy_f32_f32_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int lgn2_gpu_encode_cpy_f32_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    const int use_contiguous =
        lgn2_gpu_env_bool("LGN2_METAL_DISABLE_CONTIG_F32_F16_COPY") <= 0;
    id<MTLComputePipelineState> pipeline = use_contiguous
        ? g_cpy_contig_f32_f16_pipeline
        : g_cpy_f32_f16_pipeline;
    const NSUInteger work_items = use_contiguous
        ? ((NSUInteger)n + 3u) / 4u
        : (NSUInteger)n;
    const NSUInteger nth = lgn2_gpu_cpy_threads((uint32_t)work_items, pipeline);
    const NSUInteger groups = (work_items + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    if (use_contiguous) {
        [enc setBytes:&n length:sizeof(n) atIndex:0];
    } else {
        lgn2_gpu_cpy_args args =
            lgn2_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(uint16_t));
        [enc setBytes:&args length:sizeof(args) atIndex:0];
    }
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int lgn2_gpu_swiglu_tensor(
        lgn2_gpu_tensor       *out,
        const lgn2_gpu_tensor *gate,
        const lgn2_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !gate || !up || n == 0) return 0;
    if (!isfinite(clamp) || clamp < 0.0f || !isfinite(weight)) return 0;

    @autoreleasepool {
        id<MTLBuffer> gatebuf = lgn2_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = lgn2_gpu_tensor_buffer(up);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * sizeof(float);
        if (!gatebuf || !upbuf || !outbuf ||
            lgn2_gpu_tensor_bytes(gate) < bytes ||
            lgn2_gpu_tensor_bytes(up) < bytes ||
            lgn2_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "lgn2: Metal SwiGLU received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        lgn2_gpu_glu_args args = {
            .ne00 = (int32_t)n,
            .nb01 = (uint64_t)n * sizeof(float),
            .ne10 = (int32_t)n,
            .nb11 = (uint64_t)n * sizeof(float),
            .ne0 = (int32_t)n,
            .nb1 = (uint64_t)n * sizeof(float),
            .i00 = 0,
            .i10 = 0,
            .alpha = weight,
            .limit = clamp,
        };
        NSUInteger nth = g_swiglu_flat_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        if (nth > (NSUInteger)n) nth = (NSUInteger)n;
        if (nth == 0u) nth = 1u;
        const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_swiglu_flat_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gatebuf offset:lgn2_gpu_tensor_offset(gate) atIndex:1];
        [enc setBuffer:upbuf offset:lgn2_gpu_tensor_offset(up) atIndex:2];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "SwiGLU")) return 0;
    }

    return 1;
}

int lgn2_gpu_add_tensor(
        lgn2_gpu_tensor       *out,
        const lgn2_gpu_tensor *a,
        const lgn2_gpu_tensor *b,
        uint32_t                n) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !a || !b || n == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> abuf = lgn2_gpu_tensor_buffer(a);
        id<MTLBuffer> bbuf = lgn2_gpu_tensor_buffer(b);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * sizeof(float);
        if (!abuf || !bbuf || !outbuf ||
            lgn2_gpu_tensor_bytes(a) < bytes ||
            lgn2_gpu_tensor_bytes(b) < bytes ||
            lgn2_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "lgn2: Metal tensor add received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        lgn2_gpu_add_flat_args args = { .n = n };
        NSUInteger nth = g_add2_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        if (nth > (NSUInteger)n) nth = (NSUInteger)n;
        if (nth == 0u) nth = 1u;
        const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_add2_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:abuf offset:lgn2_gpu_tensor_offset(a) atIndex:1];
        [enc setBuffer:bbuf offset:lgn2_gpu_tensor_offset(b) atIndex:2];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "tensor add")) return 0;
    }

    return 1;
}

int lgn2_gpu_add3_tensor(
        lgn2_gpu_tensor       *out,
        const lgn2_gpu_tensor *a,
        const lgn2_gpu_tensor *b,
        const lgn2_gpu_tensor *c,
        uint32_t                n) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !a || !b || !c || n == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> abuf = lgn2_gpu_tensor_buffer(a);
        id<MTLBuffer> bbuf = lgn2_gpu_tensor_buffer(b);
        id<MTLBuffer> cbuf = lgn2_gpu_tensor_buffer(c);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * sizeof(float);
        if (!abuf || !bbuf || !cbuf || !outbuf ||
            lgn2_gpu_tensor_bytes(a) < bytes ||
            lgn2_gpu_tensor_bytes(b) < bytes ||
            lgn2_gpu_tensor_bytes(c) < bytes ||
            lgn2_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "lgn2: Metal tensor add3 received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        lgn2_gpu_add_flat_args args = { .n = n };
        NSUInteger nth = g_add3_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        if (nth > (NSUInteger)n) nth = (NSUInteger)n;
        if (nth == 0u) nth = 1u;
        const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_add3_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:abuf offset:lgn2_gpu_tensor_offset(a) atIndex:1];
        [enc setBuffer:bbuf offset:lgn2_gpu_tensor_offset(b) atIndex:2];
        [enc setBuffer:cbuf offset:lgn2_gpu_tensor_offset(c) atIndex:3];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "tensor add3")) return 0;
    }

    return 1;
}

static NSUInteger lgn2_gpu_bin_threads(uint32_t width, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth_max = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = 1u;
    while (2u * nth < (NSUInteger)width && nth < nth_max) nth *= 2u;
    return nth ? nth : 1u;
}


static int lgn2_gpu_encode_bin_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const lgn2_gpu_bin_args   *args,
        id<MTLBuffer>               a,
        NSUInteger                  a_off,
        id<MTLBuffer>               b,
        NSUInteger                  b_off,
        id<MTLBuffer>               out,
        NSUInteger                  out_off) {
    if (!cb || !pipeline || !args || !a || !b || !out || args->ne0 <= 0 || args->ne1 <= 0) {
        return 0;
    }

    const NSUInteger nth = lgn2_gpu_bin_threads((uint32_t)args->ne0, pipeline);
    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:a offset:a_off atIndex:1];
    [enc setBuffer:b offset:b_off atIndex:2];
    [enc setBuffer:out offset:out_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)args->ne1,
                                          (NSUInteger)args->ne2,
                                          (NSUInteger)args->ne3)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return 1;
}


static lgn2_gpu_mul_mm_id_map_args lgn2_gpu_make_mul_mm_id_map_args(
        uint32_t src0_cols,
        uint32_t src0_experts,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * sizeof(float);
    return (lgn2_gpu_mul_mm_id_map_args) {
        .ne02 = (int32_t)src0_experts,
        .ne10 = (int32_t)src0_cols,
        .ne11 = (int32_t)src1_expert_rows,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne21 = (int32_t)n_tokens,
        .ne20 = (int32_t)selected_experts,
        .nb21 = (uint64_t)selected_experts * sizeof(int32_t),
    };
}

static lgn2_gpu_mul_mm_id_args lgn2_gpu_make_mul_mm_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    return lgn2_gpu_make_mul_mm_id_args_src1_size(src0_cols,
                                                   src0_rows,
                                                   src0_experts,
                                                   src0_row_bytes,
                                                   src0_expert_bytes,
                                                   src1_expert_rows,
                                                   selected_experts,
                                                   n_tokens,
                                                   sizeof(float));
}

static lgn2_gpu_mul_mm_id_args lgn2_gpu_make_mul_mm_id_args_src1_size(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t src1_elem_size) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * src1_elem_size;
    return (lgn2_gpu_mul_mm_id_args) {
        .ne00 = (int32_t)src0_cols,
        .ne02 = (int32_t)src0_experts,
        .nb01 = src0_row_bytes,
        .nb02 = src0_expert_bytes,
        .nb03 = (uint64_t)src0_experts * src0_expert_bytes,
        .ne11 = (int32_t)src1_expert_rows,
        .nb10 = src1_elem_size,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .nb13 = (uint64_t)n_tokens * (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne20 = (int32_t)selected_experts,
        .ne21 = (int32_t)n_tokens,
        .ne0 = (int32_t)src0_rows,
        .ne1 = (int32_t)selected_experts,
        .r2 = 1,
        .r3 = 1,
        .reserved = {0u, 0u, 0u},
    };
}

static const char *lgn2_gpu_metal_tensor_type_name(uint32_t type) {
    switch (type) {
    case LGN2_METAL_TENSOR_IQ2_XXS: return "iq2_xxs";
    case LGN2_METAL_TENSOR_Q2_K:    return "q2_k";
    case LGN2_METAL_TENSOR_Q3_K:    return "q3_k";
    case LGN2_METAL_TENSOR_Q4_K:    return "q4_k";
    case LGN2_METAL_TENSOR_Q5_K:    return "q5_k";
    case LGN2_METAL_TENSOR_Q6_K:    return "q6_k";
    case LGN2_METAL_TENSOR_MXFP4:   return "mxfp4";
    default:                       return "unknown";
    }
}

static const char *lgn2_gpu_trim_env_value(const char *env, size_t *len_out) {
    if (len_out) *len_out = 0;
    if (!env) return NULL;

    while (isspace((unsigned char)*env)) env++;
    size_t n = strlen(env);
    while (n > 0 && isspace((unsigned char)env[n - 1])) n--;
    if (len_out) *len_out = n;
    return env;
}

static bool lgn2_gpu_profile_layer_value_match(const char *env, uint32_t layer_index) {
    size_t env_len = 0;
    env = lgn2_gpu_trim_env_value(env, &env_len);
    if (!env || env_len == 0) return true;
    if (lgn2_gpu_env_value_eq(env, env_len, "all")) return true;

    const char *p = env;
    const char *end_env = env + env_len;
    while (p < end_env) {
        while (p < end_env && (*p == ' ' || *p == '\t' || *p == ',')) p++;
        if (p >= end_env) break;

        char *end = NULL;
        const unsigned long first = strtoul(p, &end, 10);
        if (end == p || end > end_env || first > UINT32_MAX) return false;

        unsigned long last = first;
        p = end;
        if (p < end_env && *p == '-') {
            p++;
            last = strtoul(p, &end, 10);
            if (end == p || end > end_env || last > UINT32_MAX) return false;
            p = end;
        }

        if (first <= layer_index && layer_index <= last) return true;
        while (p < end_env && (*p == ' ' || *p == '\t')) p++;
        if (p < end_env && *p != ',') return false;
    }
    return false;
}

static bool lgn2_gpu_stage_profile_enabled_for_layer(const char *flag_env_name,
                                                    const char *layer_env_name,
                                                    uint32_t    layer_index) {
    size_t flag_len = 0;
    const char *flag = lgn2_gpu_trim_env_value(getenv(flag_env_name), &flag_len);
    if (!flag) return false;

    size_t layer_len = 0;
    const char *layer = lgn2_gpu_trim_env_value(getenv(layer_env_name), &layer_len);
    const bool has_layer_filter = layer && layer_len != 0;

    if (flag_len != 0) {
        if (lgn2_gpu_env_value_eq(flag, flag_len, "0") ||
            lgn2_gpu_env_value_eq(flag, flag_len, "false") ||
            lgn2_gpu_env_value_eq(flag, flag_len, "no") ||
            lgn2_gpu_env_value_eq(flag, flag_len, "off")) {
            return false;
        }
        if (!has_layer_filter &&
            !lgn2_gpu_env_value_eq(flag, flag_len, "1") &&
            !lgn2_gpu_env_value_eq(flag, flag_len, "true") &&
            !lgn2_gpu_env_value_eq(flag, flag_len, "yes") &&
            !lgn2_gpu_env_value_eq(flag, flag_len, "on") &&
            !lgn2_gpu_env_value_eq(flag, flag_len, "all")) {
            return lgn2_gpu_profile_layer_value_match(flag, layer_index);
        }
    }

    return lgn2_gpu_profile_layer_value_match(layer, layer_index);
}

static id<MTLComputePipelineState> lgn2_gpu_routed_mm_pipeline(uint32_t type) {
    switch (type) {
    case LGN2_METAL_TENSOR_Q8_0:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f32", false);
    case LGN2_METAL_TENSOR_Q8_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_K_f32", false);
    case LGN2_METAL_TENSOR_IQ2_XXS:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f32", false);
    case LGN2_METAL_TENSOR_Q2_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f32", false);
    case LGN2_METAL_TENSOR_Q3_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q3_K_f32", false);
    case LGN2_METAL_TENSOR_Q4_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f32", false);
    case LGN2_METAL_TENSOR_Q5_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q5_K_f32", false);
    case LGN2_METAL_TENSOR_Q6_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q6_K_f32", false);
    case LGN2_METAL_TENSOR_MXFP4:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_mxfp4_f32", false);
    default:
        return nil;
    }
}

static id<MTLComputePipelineState> lgn2_gpu_routed_mm_f16_rhs_pipeline(uint32_t type) {
    switch (type) {
    case LGN2_METAL_TENSOR_Q8_0:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f16", false);
    case LGN2_METAL_TENSOR_Q8_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_K_f16", false);
    case LGN2_METAL_TENSOR_IQ2_XXS:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f16", false);
    case LGN2_METAL_TENSOR_Q2_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f16", false);
    case LGN2_METAL_TENSOR_Q3_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q3_K_f16", false);
    case LGN2_METAL_TENSOR_Q4_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f16", false);
    case LGN2_METAL_TENSOR_Q5_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q5_K_f16", false);
    case LGN2_METAL_TENSOR_Q6_K:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q6_K_f16", false);
    case LGN2_METAL_TENSOR_MXFP4:
        return lgn2_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_mxfp4_f16", false);
    default:
        return nil;
    }
}
























static int lgn2_gpu_encode_mul_mm_id_map(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        const lgn2_gpu_mul_mm_id_map_args *map_args,
        const lgn2_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off) {
    if (!cb || !map_pipeline || !map_args || !mm_args || !ids ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes =
        (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes) return 0;
    const NSUInteger work_offset = (tpe_bytes + hids_bytes + 7u) & ~7u;
    const uint64_t pair_rows =
        (uint64_t)(uint32_t)mm_args->ne20 * (uint32_t)mm_args->ne21;
    const uint64_t work_cap =
        (pair_rows + 31u * (uint32_t)mm_args->ne02 + 31u) / 32u;
    const NSUInteger work_item_bytes = 2u * sizeof(uint32_t);
    if (work_cap > (NSUIntegerMax - 8u) / work_item_bytes ||
        work_offset > NSUIntegerMax - 8u -
                          (NSUInteger)work_cap * work_item_bytes) {
        return 0;
    }
    const NSUInteger total_bytes =
        work_offset + 8u + (NSUInteger)work_cap * work_item_bytes;
    if (!lgn2_gpu_ensure_scratch_buffer(&g_moe_id_map_buffer,
                                         &g_moe_id_map_bytes,
                                         total_bytes,
                                         "lgn2_moe_id_map")) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:map_pipeline];
    [enc setBytes:map_args length:sizeof(*map_args) atIndex:0];
    [enc setBuffer:ids offset:ids_off atIndex:1];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:3];
    [enc setBuffer:g_moe_id_map_buffer offset:work_offset atIndex:4];
    [enc setThreadgroupMemoryLength:(NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne20 * sizeof(uint16_t) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake((NSUInteger)mm_args->ne02, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int lgn2_gpu_encode_mul_mm_id_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const lgn2_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes) {
    if (!cb || !mm_pipeline || !mm_args || !src0 || !src1 || !dst ||
        !g_moe_id_map_buffer ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }
    /*
     * The routed MoE grouped matmul uses the legacy 32-token expert-major tile.
     * The removed TensorOps variant was not semantically stable on evals, so keep
     * this encoder tied to the tested simdgroup kernel shape.
     */
    const bool use_resource_hints =
        getenv("LGN2_METAL_MOE_MM_ID_USE_RESOURCES") != NULL &&
        getenv("LGN2_METAL_DISABLE_MOE_MM_ID_USE_RESOURCES") == NULL;

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes = (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes) {
        return 0;
    }
    const NSUInteger work_offset = (tpe_bytes + hids_bytes + 7u) & ~7u;
    const uint64_t pair_rows =
        (uint64_t)(uint32_t)mm_args->ne20 * (uint32_t)mm_args->ne21;
    const uint64_t work_cap =
        (pair_rows + 31u * (uint32_t)mm_args->ne02 + 31u) / 32u;
    const NSUInteger work_item_bytes = 2u * sizeof(uint32_t);
    if (work_cap > NSUIntegerMax ||
        work_offset > NSUIntegerMax - 8u ||
        (NSUInteger)work_cap >
            (NSUIntegerMax - work_offset - 8u) / work_item_bytes ||
        g_moe_id_map_bytes <
            work_offset + 8u + (NSUInteger)work_cap * work_item_bytes) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:mm_pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:3];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:4];
    [enc setBuffer:dst offset:dst_off atIndex:5];
    [enc setBuffer:g_moe_id_map_buffer offset:work_offset atIndex:6];
    if (use_resource_hints) {
        [enc useResource:src0 usage:MTLResourceUsageRead];
    }
    [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)work_cap,
                                          ((NSUInteger)mm_args->ne0 + 63u) / 64u,
                                          1)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return 1;
}





static int lgn2_gpu_encode_moe_swiglu_weight(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate,
        NSUInteger           gate_off,
        id<MTLBuffer>        up,
        NSUInteger           up_off,
        id<MTLBuffer>        mid,
        NSUInteger           mid_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             width,
        uint32_t             rows,
        float                clamp_value,
        bool                 mid_f16) {
    if (!cb || !gate || !up || !mid || !weights || width == 0 || rows == 0) return 0;

    id<MTLComputePipelineState> pipeline =
        lgn2_gpu_get_pipeline(mid_f16 ? "kernel_dsv4_moe_swiglu_weight_f16" :
                                         "kernel_dsv4_moe_swiglu_weight");
    if (!pipeline) return 0;

    lgn2_gpu_dsv4_moe_swiglu_weight_args args = {
        .width = width,
        .rows = rows,
        .gate_row_stride = (uint64_t)width * sizeof(float),
        .up_row_stride = (uint64_t)width * sizeof(float),
        .mid_row_stride = (uint64_t)width * (mid_f16 ? sizeof(uint16_t) : sizeof(float)),
        .weight_stride = sizeof(float),
        .write_clamped = getenv("LGN2_METAL_MOE_WRITE_CLAMPED_ACT") != NULL ? 1u : 0u,
        .clamp_value = clamp_value,
    };

    NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > width) nth = width;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate    offset:gate_off    atIndex:1];
    [enc setBuffer:up      offset:up_off      atIndex:2];
    [enc setBuffer:mid     offset:mid_off     atIndex:3];
    [enc setBuffer:weights offset:weights_off atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int lgn2_gpu_encode_moe_sum8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_tokens == 0) return 0;

    if (!g_moe_sum8_pipeline) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    lgn2_gpu_dsv4_moe_sum_args args = {
        .width = out_dim,
        .tokens = n_tokens,
        .src_token_stride = 8u * out_row_bytes,
        .dst_token_stride = out_row_bytes,
    };

    NSUInteger nth = g_moe_sum8_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > out_dim) nth = out_dim;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_sum8_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:experts offset:experts_off atIndex:1];
    [enc setBuffer:out     offset:out_off     atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int lgn2_gpu_encode_moe_sum10(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_tokens == 0) return 0;
    if (!g_moe_sum10_pipeline) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    lgn2_gpu_dsv4_moe_sum_args args = {
        .width = out_dim,
        .tokens = n_tokens,
        .src_token_stride = 10u * out_row_bytes,
        .dst_token_stride = out_row_bytes,
    };

    NSUInteger nth = g_moe_sum10_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > out_dim) nth = out_dim;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_sum10_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:experts offset:experts_off atIndex:1];
    [enc setBuffer:out offset:out_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static lgn2_gpu_bin_args lgn2_gpu_make_moe_add_args(
        uint32_t out_dim,
        uint32_t n_tokens,
        uint64_t src0_token_stride,
        uint64_t src1_token_stride,
        uint64_t dst_token_stride) {
    return (lgn2_gpu_bin_args) {
        .ne00 = (int32_t)out_dim,
        .ne01 = (int32_t)n_tokens,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src0_token_stride,
        .nb02 = (uint64_t)n_tokens * src0_token_stride,
        .nb03 = (uint64_t)n_tokens * src0_token_stride,
        .ne10 = (int32_t)out_dim,
        .ne11 = (int32_t)n_tokens,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_token_stride,
        .nb12 = (uint64_t)n_tokens * src1_token_stride,
        .nb13 = (uint64_t)n_tokens * src1_token_stride,
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tokens,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_token_stride,
        .nb2 = (uint64_t)n_tokens * dst_token_stride,
        .nb3 = (uint64_t)n_tokens * dst_token_stride,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int lgn2_gpu_encode_moe_sum_experts(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_expert,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_expert < 2 || n_tokens == 0) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t expert_token_stride = (uint64_t)n_expert * out_row_bytes;

    if (n_expert == 8 &&
        lgn2_gpu_encode_moe_sum8(cb,
                                  experts,
                                  experts_off,
                                  out,
                                  out_off,
                                  out_dim,
                                  n_tokens)) {
        return 1;
    }

    if (n_expert == 10 &&
        lgn2_gpu_encode_moe_sum10(cb,
                                  experts,
                                  experts_off,
                                  out,
                                  out_off,
                                  out_dim,
                                  n_tokens)) {
        return 1;
    }

    lgn2_gpu_bin_args first =
        lgn2_gpu_make_moe_add_args(out_dim, n_tokens, expert_token_stride, expert_token_stride, out_row_bytes);
    if (!lgn2_gpu_encode_bin_f32_rows(cb,
                                       g_add_pipeline,
                                       &first,
                                       experts,
                                       experts_off,
                                       experts,
                                       experts_off + (NSUInteger)out_row_bytes,
                                       out,
                                       out_off)) {
        return 0;
    }

    lgn2_gpu_bin_args accum =
        lgn2_gpu_make_moe_add_args(out_dim, n_tokens, out_row_bytes, expert_token_stride, out_row_bytes);
    for (uint32_t slot = 2; slot < n_expert; slot++) {
        if (!lgn2_gpu_encode_bin_f32_rows(cb,
                                           g_add_pipeline,
                                           &accum,
                                           out,
                                           out_off,
                                           experts,
                                           experts_off + (NSUInteger)((uint64_t)slot * out_row_bytes),
                                           out,
                                           out_off)) {
            return 0;
        }
    }
    return 1;
}





int lgn2_gpu_laguna_qkvg_f16_tensor(
        lgn2_gpu_tensor       *q,
        lgn2_gpu_tensor       *k,
        lgn2_gpu_tensor       *v,
        lgn2_gpu_tensor       *gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint64_t              k_weight_offset,
        uint64_t              v_weight_offset,
        uint64_t              gate_weight_offset,
        uint32_t              in_dim,
        uint32_t              q_dim,
        uint32_t              kv_dim,
        uint32_t              gate_dim,
        const lgn2_gpu_tensor *x) {
    int qk_plan_mode =
        lgn2_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached();
    if (qk_plan_mode == -2) {
        if (lgn2_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                    48u, 8u, 128u, 64u) < 0) return 0;
        qk_plan_mode =
            lgn2_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached();
    }
    if (qk_plan_mode < 0) return 0;
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!q || !k || !v || !gate || !model_map || !x || in_dim == 0u ||
        q_dim == 0u || kv_dim == 0u || gate_dim == 0u ||
        (in_dim % 32u) != 0u || (q_dim & 1u) != 0u ||
        (kv_dim & 1u) != 0u || (gate_dim & 1u) != 0u ||
        kv_dim > UINT32_MAX / 2u ||
        q_dim > UINT32_MAX - 2u * kv_dim ||
        q_dim + 2u * kv_dim > UINT32_MAX - gate_dim) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> qbuf = lgn2_gpu_tensor_buffer(q);
        id<MTLBuffer> kbuf = lgn2_gpu_tensor_buffer(k);
        id<MTLBuffer> vbuf = lgn2_gpu_tensor_buffer(v);
        id<MTLBuffer> gatebuf = lgn2_gpu_tensor_buffer(gate);
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
        const uint64_t q_out_bytes = (uint64_t)q_dim * sizeof(float);
        const uint64_t kv_out_bytes = (uint64_t)kv_dim * sizeof(float);
        const uint64_t gate_out_bytes = (uint64_t)gate_dim * sizeof(float);
        if (!qbuf || !kbuf || !vbuf || !gatebuf || !xbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(q) < q_out_bytes ||
            lgn2_gpu_tensor_bytes(k) < kv_out_bytes ||
            lgn2_gpu_tensor_bytes(v) < kv_out_bytes ||
            lgn2_gpu_tensor_bytes(gate) < gate_out_bytes) {
            return 0;
        }

        const uint64_t row_bytes = (uint64_t)in_dim * sizeof(uint16_t);
        if (q_dim > UINT64_MAX / row_bytes ||
            kv_dim > UINT64_MAX / row_bytes ||
            gate_dim > UINT64_MAX / row_bytes) {
            return 0;
        }
        const uint64_t q_weight_bytes = (uint64_t)q_dim * row_bytes;
        const uint64_t kv_weight_bytes = (uint64_t)kv_dim * row_bytes;
        const uint64_t gate_weight_bytes = (uint64_t)gate_dim * row_bytes;
#define LGN2_LAGUNA_WEIGHT_RANGE_OK(offset_, bytes_) \
        ((offset_) <= model_size && (bytes_) <= model_size - (offset_))
        if (!LGN2_LAGUNA_WEIGHT_RANGE_OK(q_weight_offset, q_weight_bytes) ||
            !LGN2_LAGUNA_WEIGHT_RANGE_OK(k_weight_offset, kv_weight_bytes) ||
            !LGN2_LAGUNA_WEIGHT_RANGE_OK(v_weight_offset, kv_weight_bytes) ||
            !LGN2_LAGUNA_WEIGHT_RANGE_OK(gate_weight_offset, gate_weight_bytes)) {
            return 0;
        }
#undef LGN2_LAGUNA_WEIGHT_RANGE_OK

        uint64_t q_inner = 0, k_inner = 0, v_inner = 0, gate_inner = 0;
        id<MTLBuffer> qwbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, q_weight_offset, q_weight_bytes, &q_inner);
        id<MTLBuffer> kwbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, k_weight_offset, kv_weight_bytes, &k_inner);
        id<MTLBuffer> vwbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, v_weight_offset, kv_weight_bytes, &v_inner);
        id<MTLBuffer> gwbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, gate_weight_offset, gate_weight_bytes,
            &gate_inner);
        id<MTLComputePipelineState> pipeline = lgn2_gpu_get_mul_mv_pipeline(
            "kernel_laguna_qkvg_f16_f32", 8);
        if (!qwbuf || !kwbuf || !vwbuf || !gwbuf || !pipeline) return 0;

        const lgn2_gpu_laguna_qkvg_args args = {
            .in_dim = in_dim,
            .q_dim = q_dim,
            .kv_dim = kv_dim,
            .gate_dim = gate_dim,
        };
        const NSUInteger rows =
            (NSUInteger)q_dim + 2u * kv_dim + gate_dim;
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qwbuf offset:(NSUInteger)q_inner atIndex:1];
        [enc setBuffer:kwbuf offset:(NSUInteger)k_inner atIndex:2];
        [enc setBuffer:vwbuf offset:(NSUInteger)v_inner atIndex:3];
        [enc setBuffer:gwbuf offset:(NSUInteger)gate_inner atIndex:4];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:5];
        [enc setBuffer:qbuf offset:lgn2_gpu_tensor_offset(q) atIndex:6];
        [enc setBuffer:kbuf offset:lgn2_gpu_tensor_offset(k) atIndex:7];
        [enc setBuffer:vbuf offset:lgn2_gpu_tensor_offset(v) atIndex:8];
        [enc setBuffer:gatebuf offset:lgn2_gpu_tensor_offset(gate) atIndex:9];
        [enc setThreadgroupMemoryLength:32u * 2u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((rows + 1u) / 2u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        if (!lgn2_gpu_finish_command_buffer(cb, owned,
                                            "Laguna Q/K/V/gate projection")) {
            return 0;
        }
    }
    return 1;
}

int lgn2_gpu_laguna_attn_output_residual_f16_tensor(
        lgn2_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const lgn2_gpu_tensor *x,
        const lgn2_gpu_tensor *residual) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !model_map || !x || !residual || in_dim == 0u ||
        out_dim == 0u || (in_dim % 32u) != 0u || (out_dim & 1u) != 0u) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> residualbuf = lgn2_gpu_tensor_buffer(residual);
        const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!outbuf || !xbuf || !residualbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(residual) < out_bytes ||
            lgn2_gpu_tensor_bytes(out) < out_bytes) {
            return 0;
        }

        const uint64_t row_bytes = (uint64_t)in_dim * sizeof(uint16_t);
        if (out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
        if (weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            return 0;
        }

        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, weight_offset, weight_bytes, &weight_inner);
        lgn2_gpu_mv_dispatch dispatch =
            lgn2_gpu_make_plain_mv_dispatch(in_dim, 0);
        id<MTLComputePipelineState> pipeline = lgn2_gpu_get_mul_mv_pipeline(
            "kernel_laguna_attn_output_residual_f16_f32", dispatch.nsg);
        if (!weightbuf || !pipeline) return 0;

        lgn2_gpu_f16_matvec_args args =
            lgn2_gpu_make_f16_mv_args(in_dim, out_dim);
        args.nr0 = 2;
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:residualbuf offset:lgn2_gpu_tensor_offset(residual)
                atIndex:3];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * 2u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + 1u) / 2u,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)dispatch.nsg, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        if (!lgn2_gpu_finish_command_buffer(
                cb, owned, "Laguna attention output/residual")) {
            return 0;
        }
    }
    return 1;
}

/* Internal single-tensor implementation.  Target callers are rejected while
 * the explicit paired selector is on; only the named DFlash support wrapper
 * below may pass allow_selector=true. */
static int lgn2_gpu_laguna_head_rms_norm_rope_tensor_impl(
        lgn2_gpu_tensor *x,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        n_rot,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow,
        float           eps,
        bool            allow_selector) {
    int selector_mode =
        lgn2_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached();
    if (selector_mode == -2) {
        if (lgn2_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                    48u, 8u, 128u, 64u) < 0) return 0;
        selector_mode =
            lgn2_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached();
    }
    /* Atlas selection is frozen by graph preflight.  The unplanned state is
     * treated as disabled for legacy/direct callers; production Laguna plans
     * always run the preflight before reaching this layer loop. */
    const int atlas_plan_mode = lgn2_gpu_laguna_rope_atlas_plan_mode();
    const int atlas_mode = atlas_plan_mode == -1 ? -1 :
                           (atlas_plan_mode > 0 ? 1 : 0);
    if (selector_mode < 0 || (selector_mode > 0 && !allow_selector)) {
        return 0;
    }
    if (atlas_mode < 0) return 0;
    if (!x || !model_map || n_tokens == 0 || n_head == 0 ||
        head_dim == 0 || head_dim > 128u || n_rot == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0u ||
        pos0 > UINT32_MAX - (n_tokens - 1u) ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow) ||
        !isfinite(eps) || eps <= 0.0f) {
        return 0;
    }
    const int atlas_family = atlas_mode > 0 ?
        lgn2_gpu_laguna_rope_atlas_family_for_args(
            n_rot, n_ctx_orig, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow) : -1;
    if (atlas_mode > 0 &&
        (atlas_family < 0 || (atlas_family == 2 && !allow_selector) ||
         (atlas_family != 2 && allow_selector) ||
         !lgn2_gpu_laguna_rope_atlas_geometry_ok(n_head, 0u, head_dim, n_rot))) {
        return 0;
    }
    if (!g_initialized && !lgn2_gpu_init()) return 0;

    const uint64_t values = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t bytes = values * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    if (lgn2_gpu_tensor_bytes(x) < bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        fprintf(stderr, "lgn2: Metal Laguna head norm/RoPE received an invalid buffer or weight range\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf = lgn2_gpu_wrap_model_range(model_map,
                                                            model_size,
                                                            weight_offset,
                                                            weight_bytes,
                                                            &weight_inner);
        id<MTLComputePipelineState> pipeline = nil;
        if (atlas_mode > 0) {
            if (atlas_family == 2) {
                if (!lgn2_gpu_laguna_rope_support_atlas_prepare(n_tokens, pos0)) {
                    return 0;
                }
            } else if (!lgn2_gpu_laguna_rope_atlas_prepare(n_tokens, pos0)) {
                return 0;
            }
            pipeline = g_laguna_head_norm_rope_atlas_pipeline;
        } else {
            pipeline = lgn2_gpu_hot_pipeline(
                g_laguna_head_norm_rope_pipeline,
                "kernel_laguna_head_rms_norm_rope_neox");
        }
        if (!xbuf || !weightbuf || !pipeline) return 0;

        lgn2_gpu_laguna_norm_rope_args args = {
            .n_tokens = n_tokens,
            .n_head = n_head,
            .head_dim = head_dim,
            .n_rot = n_rot,
            .pos0 = pos0,
            .n_ctx_orig = n_ctx_orig,
            .eps = eps,
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .rope_atlas_family = atlas_family >= 0 ?
                (uint32_t)(atlas_family == 2 ? 0 : atlas_family) : 0u,
            .rope_atlas_stride = atlas_family == 2 ? 1u : 2u,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        if (!enc) {
            if (owned) {
                (void)lgn2_gpu_finish_command_buffer(
                    cb, owned, "Laguna head norm/RoPE encoder unavailable");
            }
            return 0;
        }
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:2];
        if (atlas_mode > 0) {
            [enc setBuffer:atlas_family == 2
                         ? g_laguna_rope_support_atlas_buffer
                         : g_laguna_rope_atlas_buffer
                    offset:0 atIndex:3];
        }
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_head, n_tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        if (atlas_mode > 0) {
            if (atlas_family == 2) {
                g_laguna_rope_support_atlas_consumed_dispatch_count++;
                lgn2_gpu_laguna_atlas_current_cb_evidence()->support_consumed++;
                lgn2_gpu_laguna_rope_atlas_trace_consume(
                    "support-single", 2u, n_tokens, pos0);
            } else {
                g_laguna_rope_atlas_consumed_dispatch_count++;
                g_laguna_rope_atlas_consumed_family_count[atlas_family]++;
                lgn2_gpu_laguna_atlas_current_cb_evidence()->target_consumed++;
                lgn2_gpu_laguna_atlas_current_cb_evidence()->target_family[atlas_family]++;
                lgn2_gpu_laguna_rope_atlas_trace_consume(
                    "single", (uint32_t)atlas_family, n_tokens, pos0);
            }
        }
        if (!lgn2_gpu_finish_command_buffer(cb, owned, "Laguna head norm/RoPE")) return 0;
    }
    return 1;
}

int lgn2_gpu_laguna_head_rms_norm_rope_tensor(
        lgn2_gpu_tensor *x,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        n_rot,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow,
        float           eps) {
    return lgn2_gpu_laguna_head_rms_norm_rope_tensor_impl(
        x, model_map, model_size, weight_offset, n_tokens, n_head,
        head_dim, n_rot, pos0, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, eps, false);
}

/* Explicit DFlash support route.  This is intentionally the stock single-K
 * PSO even when LGN2_METAL_LAGUNA_QK_NORM_ROPE_SIMD32=1: the selector is a
 * contract for target Laguna Q/K calls, while support K staging has no Q
 * operand and must not increment the paired-target evidence counter. */
int lgn2_gpu_laguna_head_rms_norm_rope_support_tensor(
        lgn2_gpu_tensor *x,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        n_rot,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow,
        float           eps) {
    return lgn2_gpu_laguna_head_rms_norm_rope_tensor_impl(
        x, model_map, model_size, weight_offset, n_tokens, n_head,
        head_dim, n_rot, pos0, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, eps, true);
}

int lgn2_gpu_laguna_qk_head_rms_norm_rope_tensor(
        lgn2_gpu_tensor *q,
        lgn2_gpu_tensor *k,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        q_weight_offset,
        uint64_t        k_weight_offset,
        uint32_t        n_tokens,
        uint32_t        n_q_head,
        uint32_t        n_k_head,
        uint32_t        head_dim,
        uint32_t        n_rot,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow,
        float           eps) {
    int simd32_mode =
        lgn2_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached();
    if (simd32_mode == -2) {
        if (lgn2_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                    n_q_head, n_k_head, head_dim, n_rot) < 0) return 0;
        simd32_mode =
            lgn2_gpu_laguna_qk_head_norm_rope_simd32_plan_mode_cached();
    }
    const int atlas_plan = lgn2_gpu_laguna_rope_atlas_plan_mode();
    const int atlas_mode = atlas_plan == -1 ? -1 : (atlas_plan > 0 ? 1 : 0);
    if (simd32_mode < 0) return 0;
    if (atlas_mode < 0) return 0;
    if (simd32_mode > 0 &&
        lgn2_gpu_laguna_qk_head_norm_rope_simd32_preflight(
                n_q_head, n_k_head, head_dim, n_rot) != 1) {
        return 0;
    }
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!q || !k || !model_map || n_tokens == 0 || n_q_head == 0 ||
        n_k_head == 0 || n_q_head > UINT32_MAX - n_k_head ||
        head_dim == 0 || head_dim > 128u || n_rot == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0u ||
        pos0 > UINT32_MAX - (n_tokens - 1u) ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow) ||
        !isfinite(eps) || eps <= 0.0f) {
        return 0;
    }
    const int atlas_family = atlas_mode > 0 ?
        lgn2_gpu_laguna_rope_atlas_family_for_args(
            n_rot, n_ctx_orig, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow) : -1;
    if (atlas_mode > 0 && (atlas_family < 0 ||
                           (atlas_family == 2 && n_q_head != 72u) ||
                           !lgn2_gpu_laguna_rope_atlas_geometry_ok(
                               n_q_head, n_k_head, head_dim, n_rot))) {
        return 0;
    }

    const uint64_t q_values = (uint64_t)n_tokens * n_q_head * head_dim;
    const uint64_t k_values = (uint64_t)n_tokens * n_k_head * head_dim;
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    if (lgn2_gpu_tensor_bytes(q) < q_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(k) < k_values * sizeof(float) ||
        q_weight_offset > model_size ||
        weight_bytes > model_size - q_weight_offset ||
        k_weight_offset > model_size ||
        weight_bytes > model_size - k_weight_offset) {
        fprintf(stderr, "lgn2: Metal Laguna Q/K norm/RoPE received an invalid buffer or weight range\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> qbuf = lgn2_gpu_tensor_buffer(q);
        id<MTLBuffer> kbuf = lgn2_gpu_tensor_buffer(k);
        uint64_t q_weight_inner = 0;
        uint64_t k_weight_inner = 0;
        id<MTLBuffer> q_weightbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, q_weight_offset, weight_bytes,
            &q_weight_inner);
        id<MTLBuffer> k_weightbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, k_weight_offset, weight_bytes,
            &k_weight_inner);
        const bool use_simd32 = simd32_mode > 0;
        const bool use_atlas = atlas_mode > 0;
        id<MTLComputePipelineState> pipeline = nil;
        const char *pipeline_name =
            "kernel_laguna_qk_head_rms_norm_rope_neox";
        if (use_atlas) {
            if (atlas_family == 2) {
                if (!lgn2_gpu_laguna_rope_support_atlas_prepare(n_tokens, pos0)) {
                    return 0;
                }
            } else if (!lgn2_gpu_laguna_rope_atlas_prepare(n_tokens, pos0)) {
                return 0;
            }
            if (use_simd32) {
                pipeline = g_laguna_qk_head_norm_rope_simd32_atlas_pipeline;
                pipeline_name =
                    "kernel_laguna_qk_head_rms_norm_rope_neox_simd32_atlas";
            } else {
                pipeline = g_laguna_qk_head_norm_rope_atlas_pipeline;
                pipeline_name =
                    "kernel_laguna_qk_head_rms_norm_rope_neox_atlas";
            }
        } else if (use_simd32) {
            id<MTLComputePipelineState> simd32_pipeline =
                lgn2_gpu_laguna_qk_head_norm_rope_simd32_pipeline();
            pipeline = simd32_pipeline;
            pipeline_name =
                "kernel_laguna_qk_head_rms_norm_rope_neox_simd32";
        } else {
            pipeline = lgn2_gpu_hot_pipeline(
                g_laguna_qk_head_norm_rope_pipeline,
                "kernel_laguna_qk_head_rms_norm_rope_neox");
        }
        if (!qbuf || !kbuf || !q_weightbuf || !k_weightbuf || !pipeline) {
            return 0;
        }

        lgn2_gpu_laguna_norm_rope_args args = {
            .n_tokens = n_tokens,
            .n_head = n_q_head + n_k_head,
            .head_dim = head_dim,
            .n_rot = n_rot,
            .pos0 = pos0,
            .n_ctx_orig = n_ctx_orig,
            .eps = eps,
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .rope_atlas_family = atlas_family >= 0 ?
                (uint32_t)(atlas_family == 2 ? 0 : atlas_family) : 0u,
            .rope_atlas_stride = atlas_family == 2 ? 1u : 2u,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        if (!enc) {
            if (owned) {
                (void)lgn2_gpu_finish_command_buffer(
                    cb, owned, "Laguna Q/K head norm/RoPE encoder unavailable");
            }
            return 0;
        }
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:lgn2_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:kbuf offset:lgn2_gpu_tensor_offset(k) atIndex:2];
        [enc setBuffer:q_weightbuf offset:(NSUInteger)q_weight_inner atIndex:3];
        [enc setBuffer:k_weightbuf offset:(NSUInteger)k_weight_inner atIndex:4];
        [enc setBytes:&n_q_head length:sizeof(n_q_head) atIndex:5];
        if (use_atlas) {
            [enc setBuffer:atlas_family == 2
                         ? g_laguna_rope_support_atlas_buffer
                         : g_laguna_rope_atlas_buffer
                    offset:0 atIndex:6];
        }
        [enc setThreadgroupMemoryLength:128u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_q_head + n_k_head, n_tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(use_simd32 ? 32u : 128u,
                                               1, 1)];
        if (use_simd32) {
            /* This is encoded-dispatch evidence, not completion accounting:
             * graph callers may keep the command batch open. */
            g_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count++;
            lgn2_gpu_laguna_atlas_current_cb_evidence()->qk_simd32++;
        }
        if (use_atlas) {
            if (atlas_family == 2) {
                g_laguna_rope_support_atlas_consumed_dispatch_count++;
                lgn2_gpu_laguna_atlas_current_cb_evidence()->support_consumed++;
                lgn2_gpu_laguna_rope_atlas_trace_consume(
                    use_simd32 ? "support-paired-simd32" : "support-paired",
                    2u, n_tokens, pos0);
            } else {
                g_laguna_rope_atlas_consumed_dispatch_count++;
                g_laguna_rope_atlas_consumed_family_count[atlas_family]++;
                lgn2_gpu_laguna_atlas_current_cb_evidence()->target_consumed++;
                lgn2_gpu_laguna_atlas_current_cb_evidence()->target_family[atlas_family]++;
                lgn2_gpu_laguna_rope_atlas_trace_consume(
                    use_simd32 ? "paired-simd32" : "paired",
                    (uint32_t)atlas_family, n_tokens, pos0);
            }
        }
        lgn2_gpu_end_compute_encoder(cb, enc);
        if (!lgn2_gpu_finish_command_buffer(
                cb, owned, use_simd32
                    ? "Laguna Q/K head norm/RoPE SIMD32"
                    : "Laguna Q/K head norm/RoPE")) {
            return 0;
        }
        if (use_simd32) {
            if (lgn2_gpu_laguna_qk_head_norm_rope_simd32_trace_enabled()) {
                fprintf(stderr,
                        "lgn2: Laguna Q/K norm/RoPE SIMD32 encoded kernel=%s "
                        "tew=%lu max_threads=%lu tokens=%u q_heads=%u "
                        "k_heads=%u\n",
                        pipeline_name,
                        (unsigned long)pipeline.threadExecutionWidth,
                        (unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
                        n_tokens, n_q_head, n_k_head);
            }
        }
    }
    return 1;
}

uint64_t lgn2_gpu_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count(void) {
    return g_laguna_qk_head_norm_rope_simd32_encoded_dispatch_count;
}

uint64_t lgn2_gpu_laguna_qk_head_norm_rope_simd32_completed_dispatch_count(void) {
    return g_laguna_qk_head_norm_rope_simd32_completed_dispatch_count;
}

static int lgn2_gpu_encode_laguna_flash_attention_decode(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        headsbuf,
        NSUInteger           heads_offset,
        id<MTLBuffer>        qbuf,
        NSUInteger           q_offset,
        id<MTLBuffer>        gatebuf,
        NSUInteger           gate_offset,
        id<MTLBuffer>        keybuf,
        NSUInteger           key_offset,
        id<MTLBuffer>        valuebuf,
        NSUInteger           value_offset,
        uint32_t              cache_cap,
        uint32_t              key_count,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale,
        int                   gqa9_selected,
        int                   gqa3_selected,
        int                   staged_swa_active) {
    if (!cb || !headsbuf || !qbuf || !gatebuf || !keybuf || !valuebuf ||
        key_count == 0u || key_count > cache_cap || n_head == 0u ||
        n_head_kv == 0u || n_head % n_head_kv != 0u || head_dim != 128u) {
        return 0;
    }

    const uint32_t ncpsg = 32u;
    const uint32_t nwg = 32u;
    const uint32_t nsg = lgn2_gpu_flash_attn_vec_nsg(key_count, nwg, ncpsg);
    const bool has_pad = (key_count % ncpsg) != 0u;
    const bool swa_gqa9_requested =
        gqa9_selected != 0 &&
        cache_cap == 512u && key_count == 512u &&
        n_head == 72u && n_head_kv == 8u && head_dim == 128u;
    const bool swa_gqa3_requested =
        gqa3_selected != 0 &&
        cache_cap == 512u && key_count == 512u &&
        n_head == 72u && n_head_kv == 8u && head_dim == 128u;
    /* Precedence for the SWA full ring: staged SWA > GQA9 > GQA3.  Staged SWA
     * reduces virtual-ring rows with the ordinary Flash arithmetic, so it must
     * not be silently mixed with a grouped opt-in.  GQA9 groups all 9 heads of
     * the production 72/8 ratio and strictly dominates GQA3 for that shape, so
     * GQA3 is dropped when both are exported.  Each lower-precedence knob is
     * ignored with a one-time stderr notice (benchmarked separately). */
    static int staged_swa_gqa_conflict_reported;
    if (staged_swa_active &&
        (swa_gqa9_requested || swa_gqa3_requested) &&
        !staged_swa_gqa_conflict_reported) {
        fprintf(stderr,
                "lgn2: Metal Laguna SWA grouped decode (GQA9/GQA3) ignored "
                "while staged SWA is enabled; staged SWA takes precedence "
                "(benchmark these flags separately)\n");
        staged_swa_gqa_conflict_reported = 1;
    }
    static int gqa9_gqa3_conflict_reported;
    if (swa_gqa9_requested && swa_gqa3_requested && !staged_swa_active &&
        !gqa9_gqa3_conflict_reported) {
        fprintf(stderr,
                "lgn2: Metal Laguna SWA GQA3 ignored while SWA GQA9 is enabled; "
                "GQA9 takes precedence for the 72/8 ring (benchmark these flags "
                "separately)\n");
        gqa9_gqa3_conflict_reported = 1;
    }
    const bool use_gqa9 =
        swa_gqa9_requested && !staged_swa_active &&
        (n_head % 9u) == 0u &&
        ((n_head / n_head_kv) % 9u) == 0u;
    const bool use_gqa3 =
        (cache_cap > 512u ||
         (swa_gqa3_requested && !staged_swa_active && !use_gqa9)) &&
        (n_head % 3u) == 0u &&
        ((n_head / n_head_kv) % 3u) == 0u;
    const NSUInteger head_bytes = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger cache_row_bytes =
        (NSUInteger)n_head_kv * head_bytes;
    const NSUInteger mask_bytes = (NSUInteger)key_count * sizeof(uint16_t);
    const NSUInteger pad_bytes = has_pad ?
        2u * (NSUInteger)ncpsg * cache_row_bytes * n_head_kv +
        (NSUInteger)ncpsg * sizeof(uint16_t) : 1u;
    const NSUInteger nrows = (NSUInteger)n_head;
    const NSUInteger tmp_bytes =
        nrows * head_dim * nwg * sizeof(float) +
        nrows * 2u * nwg * sizeof(float);

    if (!lgn2_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                       &g_flash_attn_tmp_bytes,
                                       tmp_bytes,
                                       "lgn2_laguna_flash_attn_tmp")) {
        return 0;
    }
    const bool use_grouped = use_gqa3 || use_gqa9;
    if (!use_grouped &&
        (!lgn2_gpu_ensure_zero_attention_mask(mask_bytes) ||
         !lgn2_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "lgn2_laguna_flash_attn_pad"))) {
        return 0;
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_pad && !use_grouped) {
        pad_pipeline = lgn2_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        use_gqa9 ? lgn2_gpu_get_pipeline(
                       "kernel_laguna_attention_decode_gqa9_split_f16") :
        use_gqa3 ? lgn2_gpu_get_pipeline(
                       "kernel_laguna_attention_decode_gqa3_split_f16") :
        lgn2_gpu_get_flash_attn_vec_pipeline(
            "kernel_flash_attn_ext_vec_qf32_f16_dk128_dv128",
            true, false, false, false, has_pad, false,
            (int32_t)head_dim, (int32_t)head_dim,
            (int32_t)nsg, (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        lgn2_gpu_get_laguna_flash_attn_reduce_gate_pipeline(
            (int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    id<MTLComputeCommandEncoder> enc = nil;
    if (has_pad && !use_grouped) {
        lgn2_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)key_count,
            .ne_12_2 = (int32_t)n_head_kv,
            .ne_12_3 = 1,
            .nb11 = cache_row_bytes,
            .nb12 = head_bytes,
            .nb13 = (uint64_t)cache_cap * cache_row_bytes,
            .nb21 = cache_row_bytes,
            .nb22 = head_bytes,
            .nb23 = (uint64_t)cache_cap * cache_row_bytes,
            .ne31 = 1,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = mask_bytes,
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };
        enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:keybuf offset:key_offset atIndex:1];
        [enc setBuffer:valuebuf offset:value_offset atIndex:2];
        [enc setBuffer:g_flash_attn_zero_mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, n_head_kv, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
    }

    const NSUInteger q_row_bytes =
        (NSUInteger)n_head * head_dim * sizeof(float);
    lgn2_gpu_flash_attn_vec_args vec_args = {
        .ne01 = 1,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = q_row_bytes,
        .nb02 = (uint64_t)head_dim * sizeof(float),
        .nb03 = q_row_bytes,
        .ne11 = (int32_t)key_count,
        .ne_12_2 = (int32_t)n_head_kv,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = cache_row_bytes,
        .nb12 = head_bytes,
        .nb13 = (uint64_t)cache_cap * cache_row_bytes,
        .ns20 = (int32_t)head_dim,
        .nb21 = cache_row_bytes,
        .nb22 = head_bytes,
        .nb23 = (uint64_t)cache_cap * cache_row_bytes,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = mask_bytes,
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = 1,
        .ne3 = 1,
        .scale = scale,
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };
    const NSUInteger shared_elems =
        (2u * lgn2_gpu_align_up_ns(head_dim, 128u) + 4u * ncpsg +
         2u * lgn2_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes =
        lgn2_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBuffer:qbuf offset:q_offset atIndex:1];
    [enc setBuffer:keybuf offset:key_offset atIndex:2];
    [enc setBuffer:valuebuf offset:value_offset atIndex:3];
    if (use_gqa3 || use_gqa9) {
        const uint32_t group = use_gqa9 ? 9u : 3u;
        const lgn2_gpu_laguna_gqa3_decode_args grouped_args = {
            .n_head = n_head,
            .n_head_kv = n_head_kv,
            .head_dim = head_dim,
            .key_count0 = key_count,
            .n_tokens = 1u,
            .nsg = nsg,
            .nwg = nwg,
            .scale = scale,
        };
        const NSUInteger grouped_shared_bytes =
            (NSUInteger)group * nsg * (2u + head_dim) * sizeof(float);
        [enc setBytes:&grouped_args length:sizeof(grouped_args) atIndex:0];
        [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:grouped_shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_head / group, 1, nwg)
             threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    } else {
        [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
        [enc setBuffer:g_flash_attn_zero_mask_buffer offset:0 atIndex:4];
        [enc setBuffer:gatebuf offset:gate_offset atIndex:5];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
        [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
        [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, n_head, nwg)
             threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    }
    lgn2_gpu_end_compute_encoder(cb, enc);

    lgn2_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:heads_offset atIndex:2];
    [enc setBuffer:gatebuf offset:gate_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int lgn2_gpu_laguna_store_attention_tensor(
        lgn2_gpu_tensor       *heads,
        lgn2_gpu_tensor       *key_cache,
        lgn2_gpu_tensor       *value_cache,
        const lgn2_gpu_tensor *q,
        const lgn2_gpu_tensor *k,
        const lgn2_gpu_tensor *v,
        const lgn2_gpu_tensor *gate,
        uint32_t              pos,
        uint32_t              cache_cap,
        uint32_t              key_start,
        uint32_t              key_count,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale) {
    /* Parse and probe the optional grouped route before Metal initialization
     * can open a command batch.  In particular, an old source that lacks the
     * GQA9 symbol must fail before the KV store below is recorded into a
     * caller-owned batch. */
    const int gqa9_mode = lgn2_gpu_laguna_swa_gqa9_env_mode();
    if (gqa9_mode < 0) return 0;
    /* GQA9 is shape-gated to the SWA full ring.  A global 48/8 layer (or a
     * short/partial ring) must keep using its ordinary/GQA3 route even when
     * the lifecycle opt-in is exported; only an eligible GQA9 dispatch needs
     * the optional source/PSO preflight. */
    const bool gqa9_shape =
        cache_cap == 512u && key_count == 512u &&
        n_head == 72u && n_head_kv == 8u && head_dim == 128u;
    const int gqa9_preflight = gqa9_mode > 0 && gqa9_shape ?
        lgn2_gpu_laguna_swa_gqa9_preflight(
            cache_cap, key_count, n_head, n_head_kv, head_dim) : 0;
    if (gqa9_preflight < 0) return 0;
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    const int gqa9_selected =
        lgn2_gpu_laguna_swa_gqa9_env_mode() > 0;
    const int gqa3_selected = lgn2_gpu_laguna_swa_gqa3_enabled();
    const int staged_swa_active =
        lgn2_gpu_laguna_staged_swa_enabled() > 0;
#ifdef LGN2_TEST_HOOKS
    int decode_route_kind = LGN2_LAGUNA_TEST_DECODE_ORDINARY;
#endif
    if (!heads || !key_cache || !value_cache || !q || !k || !v || !gate ||
        cache_cap == 0 || key_count == 0 || key_count > cache_cap ||
        n_head == 0 || n_head_kv == 0 || n_head % n_head_kv != 0 ||
        head_dim != 128u || !isfinite(scale) || scale <= 0.0f) {
        return 0;
    }
    const uint64_t q_values = (uint64_t)n_head * head_dim;
    const uint64_t kv_values = (uint64_t)n_head_kv * head_dim;
    const uint64_t cache_values = (uint64_t)cache_cap * kv_values;
    if (lgn2_gpu_tensor_bytes(q) < q_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(k) < kv_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(v) < kv_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(gate) < (uint64_t)n_head * sizeof(float) ||
        lgn2_gpu_tensor_bytes(heads) < q_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(key_cache) < cache_values * sizeof(uint16_t) ||
        lgn2_gpu_tensor_bytes(value_cache) < cache_values * sizeof(uint16_t)) {
        fprintf(stderr, "lgn2: Metal Laguna attention received undersized buffers\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = lgn2_gpu_tensor_buffer(heads);
        id<MTLBuffer> keybuf = lgn2_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valuebuf = lgn2_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> qbuf = lgn2_gpu_tensor_buffer(q);
        id<MTLBuffer> kbuf = lgn2_gpu_tensor_buffer(k);
        id<MTLBuffer> vbuf = lgn2_gpu_tensor_buffer(v);
        id<MTLBuffer> gatebuf = lgn2_gpu_tensor_buffer(gate);
        id<MTLComputePipelineState> store_pipeline = lgn2_gpu_hot_pipeline(
            g_laguna_store_kv_pipeline, "kernel_laguna_store_kv_f16");
        id<MTLComputePipelineState> attention_pipeline = lgn2_gpu_hot_pipeline(
            g_laguna_attention_pipeline,
            "kernel_laguna_attention_decode_gqa_f16");
        if (!headsbuf || !keybuf || !valuebuf || !qbuf || !kbuf || !vbuf ||
            !gatebuf || !store_pipeline || !attention_pipeline) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        lgn2_gpu_laguna_kv_store_args store_args = {
            .cache_cap = cache_cap,
            .cache_row = pos % cache_cap,
            .n_head_kv = n_head_kv,
            .head_dim = head_dim,
        };
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:store_pipeline];
        [enc setBytes:&store_args length:sizeof(store_args) atIndex:0];
        [enc setBuffer:kbuf offset:lgn2_gpu_tensor_offset(k) atIndex:1];
        [enc setBuffer:vbuf offset:lgn2_gpu_tensor_offset(v) atIndex:2];
        [enc setBuffer:keybuf offset:lgn2_gpu_tensor_offset(key_cache) atIndex:3];
        [enc setBuffer:valuebuf offset:lgn2_gpu_tensor_offset(value_cache) atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_values, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        /* The global-attention cache is contiguous until it reaches capacity.
         * A full sliding ring also contains exactly the active key set; its
         * physical rotation does not affect an unmasked softmax reduction.
         * An unaligned full global cache uses the old kernel so tail padding
         * never reads beyond the interleaved KV allocation.
         *
         * Grouped-eligible shapes (the production 72/8 and 48/8 ratios, both
         * GQA3-eligible) use a split-K kernel that strides keys and needs no
         * tail padding, so they may enter the flash path below 1024 keys once
         * the history is long enough to amortize the three-dispatch split-K
         * cost.  Below this floor the ungrouped single/8-SIMD kernel remains
         * cheaper.  This is a default-path change: the same grouped family is
         * already the default at >=1024 keys, so extending it down to the 512
         * boundary only removes the 3x redundant KV traffic the ungrouped
         * kernel paid in the 513-1023 range while keeping the padding-clamped
         * ungrouped gate intact for non-grouped-eligible shapes. */
        const bool grouped_eligible =
            (n_head % 3u) == 0u &&
            ((n_head / n_head_kv) % 3u) == 0u;
        const bool full_sliding_ring =
            cache_cap == 512u && key_count == cache_cap;
        const bool use_flash_decode =
            full_sliding_ring ||
            (cache_cap > 512u && key_start == 0u &&
             ((grouped_eligible && key_count >= 512u &&
               key_count < cache_cap) ||
              (key_count >= 1024u &&
               (key_count < cache_cap || key_count % 32u == 0u))));
        if (use_flash_decode) {
            if (!lgn2_gpu_encode_laguna_flash_attention_decode(
                    cb,
                    headsbuf,
                    lgn2_gpu_tensor_offset(heads),
                    qbuf,
                    lgn2_gpu_tensor_offset(q),
                    gatebuf,
                    lgn2_gpu_tensor_offset(gate),
                    keybuf,
                    lgn2_gpu_tensor_offset(key_cache),
                    valuebuf,
                    lgn2_gpu_tensor_offset(value_cache),
                    cache_cap,
                    key_count,
                    n_head,
                    n_head_kv,
                    head_dim,
                    scale,
                    gqa9_selected,
                    gqa3_selected,
                    staged_swa_active)) {
                return 0;
            }
#ifdef LGN2_TEST_HOOKS
            decode_route_kind = lgn2_gpu_laguna_test_decode_route_kind(
                cache_cap,
                key_start,
                key_count,
                n_head,
                n_head_kv,
                gqa9_selected,
                gqa3_selected,
                staged_swa_active);
#endif
        } else {
            lgn2_gpu_laguna_attention_args attention_args = {
                .n_head = n_head,
                .n_head_kv = n_head_kv,
                .head_dim = head_dim,
                .cache_cap = cache_cap,
                .key_start = key_start,
                .key_count = key_count,
                .scale = scale,
            };
            enc = lgn2_gpu_compute_encoder(cb);
            [enc setComputePipelineState:attention_pipeline];
            [enc setBytes:&attention_args length:sizeof(attention_args) atIndex:0];
            [enc setBuffer:qbuf offset:lgn2_gpu_tensor_offset(q) atIndex:1];
            [enc setBuffer:gatebuf offset:lgn2_gpu_tensor_offset(gate) atIndex:2];
            [enc setBuffer:keybuf offset:lgn2_gpu_tensor_offset(key_cache) atIndex:3];
            [enc setBuffer:valuebuf offset:lgn2_gpu_tensor_offset(value_cache) atIndex:4];
            [enc setBuffer:headsbuf offset:lgn2_gpu_tensor_offset(heads) atIndex:5];
            [enc setThreadgroupMemoryLength:(8u + 8u + 8u * 128u) * sizeof(float)
                                    atIndex:0];
            const NSUInteger attention_threads = key_count > 256u ? 256u : 32u;
            [enc dispatchThreadgroups:MTLSizeMake(n_head, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(attention_threads, 1, 1)];
            lgn2_gpu_end_compute_encoder(cb, enc);
        }
        if (!lgn2_gpu_finish_command_buffer(cb, owned, "Laguna KV store + attention")) return 0;
#ifdef LGN2_TEST_HOOKS
        lgn2_gpu_laguna_test_decode_route_note(decode_route_kind, owned);
#endif
    }
    return 1;
}

static int lgn2_gpu_encode_laguna_flash_attention_decode_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        headsbuf,
        NSUInteger           heads_offset,
        id<MTLBuffer>        qbuf,
        NSUInteger           q_offset,
        id<MTLBuffer>        gatebuf,
        NSUInteger           gate_offset,
        id<MTLBuffer>        keybuf,
        NSUInteger           key_offset,
        id<MTLBuffer>        valuebuf,
        NSUInteger           value_offset,
        uint32_t              cache_cap,
        uint32_t              key_count0,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale) {
    if (!cb || !headsbuf || !qbuf || !gatebuf || !keybuf || !valuebuf ||
        cache_cap <= 512u || key_count0 == 0u || n_tokens < 2u ||
        n_tokens > 16u || key_count0 > cache_cap - n_tokens + 1u ||
        n_head == 0u || n_head_kv == 0u ||
        (n_head % 3u) != 0u || n_head % n_head_kv != 0u ||
        ((n_head / n_head_kv) % 3u) != 0u || head_dim != 128u) {
        return 0;
    }

    const uint32_t nwg = 32u;
    const uint32_t ncpsg = 32u;
    const uint32_t key_count_max = key_count0 + n_tokens - 1u;
    const uint32_t nsg =
        lgn2_gpu_flash_attn_vec_nsg(key_count_max, nwg, ncpsg);
    if (lgn2_gpu_flash_attn_vec_nsg(key_count0, nwg, ncpsg) != nsg) {
        return 0;
    }
    const NSUInteger nrows = (NSUInteger)n_tokens * n_head;
    const NSUInteger tmp_bytes =
        nrows * head_dim * nwg * sizeof(float) +
        nrows * 2u * nwg * sizeof(float);
    if (!lgn2_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                       &g_flash_attn_tmp_bytes,
                                       tmp_bytes,
                                       "lgn2_laguna_flash_attn_tmp")) {
        return 0;
    }

    id<MTLComputePipelineState> attention_pipeline =
        lgn2_gpu_get_pipeline("kernel_laguna_attention_decode_gqa3_split_f16");
    id<MTLComputePipelineState> reduce_pipeline =
        lgn2_gpu_get_laguna_flash_attn_reduce_gate_pipeline(
            (int32_t)head_dim, (int32_t)nwg);
    if (!attention_pipeline || !reduce_pipeline) return 0;

    const lgn2_gpu_laguna_gqa3_decode_args args = {
        .n_head = n_head,
        .n_head_kv = n_head_kv,
        .head_dim = head_dim,
        .key_count0 = key_count0,
        .n_tokens = n_tokens,
        .nsg = nsg,
        .nwg = nwg,
        .scale = scale,
    };
    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attention_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:q_offset atIndex:1];
    [enc setBuffer:keybuf offset:key_offset atIndex:2];
    [enc setBuffer:valuebuf offset:value_offset atIndex:3];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:
        3u * nsg * (2u + head_dim) * sizeof(float)
                            atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_head / 3u, n_tokens, nwg)
         threadsPerThreadgroup:MTLSizeMake(32u, nsg, 1u)];
    lgn2_gpu_end_compute_encoder(cb, enc);

    const lgn2_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = lgn2_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:heads_offset atIndex:2];
    [enc setBuffer:gatebuf offset:gate_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1u, 1u)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1u, 1u)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return 1;
}

/* Stage a short post-wrap SWA verifier block, then run one ordinary qF32/F16
 * Flash-vector dispatch over all physical ring slots.  The virtual-cache
 * function constant makes buffer 4/5 the staged K/V rows; the reducer remains
 * the established Laguna reducer/gate path and uses row*n_head+head layout. */
static int lgn2_gpu_encode_laguna_flash_attention_staged_swa_rows(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> stage_pipeline,
        id<MTLBuffer>        headsbuf,
        NSUInteger           heads_offset,
        id<MTLBuffer>        qbuf,
        NSUInteger           q_offset,
        id<MTLBuffer>        kbuf,
        NSUInteger           k_offset,
        id<MTLBuffer>        vbuf,
        NSUInteger           v_offset,
        id<MTLBuffer>        gatebuf,
        NSUInteger           gate_offset,
        id<MTLBuffer>        keybuf,
        NSUInteger           key_offset,
        id<MTLBuffer>        valuebuf,
        NSUInteger           value_offset,
        id<MTLBuffer>        stagedkeybuf,
        NSUInteger           stagedkey_offset,
        id<MTLBuffer>        stagedvaluebuf,
        NSUInteger           stagedvalue_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale) {
    if (!cb || !stage_pipeline || !headsbuf || !qbuf || !kbuf || !vbuf ||
        !gatebuf || !keybuf || !valuebuf ||
        !stagedkeybuf || !stagedvaluebuf || n_tokens < 2u ||
        n_tokens > 16u || cache_cap != 512u || pos0 < cache_cap ||
        n_head == 0u || n_head_kv == 0u || n_head % n_head_kv != 0u ||
        head_dim != 128u || !isfinite(scale) || scale <= 0.0f) {
        return 0;
    }

    const uint32_t nwg = 32u;
    const uint32_t ncpsg = 32u;
    const uint32_t nsg =
        lgn2_gpu_flash_attn_vec_nsg(cache_cap, nwg, ncpsg);
    const NSUInteger nrows = (NSUInteger)n_tokens * n_head;
    const NSUInteger head_bytes = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger cache_row_bytes =
        (NSUInteger)n_head_kv * head_bytes;
    const NSUInteger q_row_bytes =
        (NSUInteger)n_head * head_dim * sizeof(float);
    const NSUInteger kv_values =
        (NSUInteger)n_tokens * n_head_kv * head_dim;
    const NSUInteger tmp_bytes =
        nrows * head_dim * nwg * sizeof(float) +
        nrows * 2u * nwg * sizeof(float);
    if (!lgn2_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                       &g_flash_attn_tmp_bytes,
                                       tmp_bytes,
                                       "lgn2_laguna_staged_flash_attn_tmp")) {
        return 0;
    }

    id<MTLComputePipelineState> vec_pipeline =
        lgn2_gpu_get_laguna_staged_flash_attn_vec_pipeline(
            "kernel_flash_attn_ext_vec_qf32_f16_dk128_dv128_virtual",
            (int32_t)head_dim,
            (int32_t)head_dim,
            (int32_t)nsg,
            (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        lgn2_gpu_get_laguna_flash_attn_reduce_gate_pipeline(
            (int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    /* Keep the persistent ring untouched until the staged Flash reduction has
     * completed.  This is the same F16 conversion used by ordinary prefill. */
    const lgn2_gpu_laguna_prefill_attention_args stage_args = {
        .n_tokens = n_tokens,
        .pos0 = pos0,
        .cache_cap = cache_cap,
        .n_head = n_head,
        .n_head_kv = n_head_kv,
        .head_dim = head_dim,
        .scale = scale,
        .pad0 = 0u,
    };
    id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:stage_pipeline];
    [enc setBytes:&stage_args length:sizeof(stage_args) atIndex:0];
    [enc setBuffer:kbuf offset:k_offset atIndex:1];
    [enc setBuffer:vbuf offset:v_offset atIndex:2];
    [enc setBuffer:stagedkeybuf offset:stagedkey_offset atIndex:3];
    [enc setBuffer:stagedvaluebuf offset:stagedvalue_offset atIndex:4];
    [enc dispatchThreads:MTLSizeMake(kv_values, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    lgn2_gpu_end_compute_encoder(cb, enc);

    lgn2_gpu_flash_attn_vec_virtual_args args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = q_row_bytes,
        .nb02 = (uint64_t)head_dim * sizeof(float),
        .nb03 = q_row_bytes,
        .ne11 = (int32_t)cache_cap,
        .ne_12_2 = (int32_t)n_head_kv,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = cache_row_bytes,
        .nb12 = head_bytes,
        .nb13 = (uint64_t)cache_cap * cache_row_bytes,
        .ns20 = (int32_t)head_dim,
        .nb21 = cache_row_bytes,
        .nb22 = head_bytes,
        .nb23 = (uint64_t)cache_cap * cache_row_bytes,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = 0,
        .nb32 = 0,
        .nb33 = 0,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = scale,
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
        .laguna_stage_pos_mod = pos0 % cache_cap,
        .laguna_stage_n_tokens = n_tokens,
        .laguna_stage_cache_cap = cache_cap,
        .laguna_stage_pad0 = 0,
        .laguna_stage_nb11 = cache_row_bytes,
        .laguna_stage_nb12 = head_bytes,
    };
    enc = lgn2_gpu_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:q_offset atIndex:1];
    [enc setBuffer:keybuf offset:key_offset atIndex:2];
    [enc setBuffer:valuebuf offset:value_offset atIndex:3];
    [enc setBuffer:stagedkeybuf offset:stagedkey_offset atIndex:4];
    [enc setBuffer:stagedvaluebuf offset:stagedvalue_offset atIndex:5];
    /* The virtual specialization does not read buffer 6, but keep the
     * established vector argument slots fully bound for validation. */
    [enc setBuffer:keybuf offset:key_offset atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    const NSUInteger shared_elems =
        (2u * lgn2_gpu_align_up_ns(head_dim, 128u) + 4u * ncpsg +
         2u * lgn2_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes =
        lgn2_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32u, nsg, 1u)];
    lgn2_gpu_end_compute_encoder(cb, enc);

    const lgn2_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = lgn2_gpu_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:heads_offset atIndex:2];
    [enc setBuffer:gatebuf offset:gate_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1u, 1u)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1u, 1u)];
    lgn2_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int lgn2_gpu_laguna_attention_prefill_tensor(
        lgn2_gpu_tensor       *heads,
        lgn2_gpu_tensor       *key_cache,
        lgn2_gpu_tensor       *value_cache,
        lgn2_gpu_tensor       *staged_key,
        lgn2_gpu_tensor       *staged_value,
        const lgn2_gpu_tensor *q,
        const lgn2_gpu_tensor *k,
        const lgn2_gpu_tensor *v,
        const lgn2_gpu_tensor *gate,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        float                 scale,
        int                   split_decode_rows) {
    /* Probe the optional GQA9 source/shape contract before the first staged
     * or direct KV command.  Passing the full ring as key_count keeps an
     * explicit request fail-closed even when this prefill begins before the
     * first complete window. */
    const int gqa9_mode = lgn2_gpu_laguna_swa_gqa9_env_mode();
    if (gqa9_mode < 0) return 0;
    const bool gqa9_shape =
        cache_cap == 512u && n_head == 72u &&
        n_head_kv == 8u && head_dim == 128u;
    const int gqa9_preflight = gqa9_mode > 0 && gqa9_shape ?
        lgn2_gpu_laguna_swa_gqa9_preflight(
            cache_cap, cache_cap, n_head, n_head_kv, head_dim) : 0;
    if (gqa9_preflight < 0) return 0;
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    const int gqa9_selected =
        lgn2_gpu_laguna_swa_gqa9_env_mode() > 0;
    const int gqa3_selected = lgn2_gpu_laguna_swa_gqa3_enabled();
    const int staged_swa_active =
        lgn2_gpu_laguna_staged_swa_enabled() > 0;
    /* The selector is lifecycle-snapshotted and rejected during init. Keep
     * this guard at the low-level entry as well so a malformed opt-in can
     * never reach either the split-row or ordinary command encoder. */
    if (g_direct_kv_prefill_mode < 0) {
        fprintf(stderr,
                "lgn2: invalid LGN2_METAL_LAGUNA_DIRECT_KV_PREFILL; "
                "expected unset, empty, 0, or literal 1\n");
        return 0;
    }
    if (!heads || !key_cache || !value_cache || !staged_key ||
        !staged_value || !q || !k || !v || !gate || n_tokens == 0 ||
        pos0 > UINT32_MAX - n_tokens || cache_cap == 0 || n_head == 0 ||
        n_head_kv == 0 || n_head % n_head_kv != 0 || head_dim != 128u ||
        !isfinite(scale) || scale <= 0.0f) {
        return 0;
    }

    const uint64_t q_values = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_values =
        (uint64_t)n_tokens * n_head_kv * head_dim;
    const uint64_t cache_values =
        (uint64_t)cache_cap * n_head_kv * head_dim;
    if (lgn2_gpu_tensor_bytes(q) < q_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(k) < kv_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(v) < kv_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(gate) <
            (uint64_t)n_tokens * n_head * sizeof(float) ||
        lgn2_gpu_tensor_bytes(heads) < q_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(staged_key) < kv_values * sizeof(uint16_t) ||
        lgn2_gpu_tensor_bytes(staged_value) < kv_values * sizeof(uint16_t) ||
        lgn2_gpu_tensor_bytes(key_cache) < cache_values * sizeof(uint16_t) ||
        lgn2_gpu_tensor_bytes(value_cache) < cache_values * sizeof(uint16_t)) {
        fprintf(stderr, "lgn2: Metal Laguna prefill attention received undersized buffers\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = lgn2_gpu_tensor_buffer(heads);
        id<MTLBuffer> keybuf = lgn2_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valuebuf = lgn2_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> stagedkeybuf = lgn2_gpu_tensor_buffer(staged_key);
        id<MTLBuffer> stagedvaluebuf = lgn2_gpu_tensor_buffer(staged_value);
        id<MTLBuffer> qbuf = lgn2_gpu_tensor_buffer(q);
        id<MTLBuffer> kbuf = lgn2_gpu_tensor_buffer(k);
        id<MTLBuffer> vbuf = lgn2_gpu_tensor_buffer(v);
        id<MTLBuffer> gatebuf = lgn2_gpu_tensor_buffer(gate);
        id<MTLComputePipelineState> store_pipeline = lgn2_gpu_hot_pipeline(
            g_laguna_store_kv_pipeline, "kernel_laguna_store_kv_f16");
        id<MTLComputePipelineState> stage_pipeline = lgn2_gpu_hot_pipeline(
            g_laguna_stage_kv_pipeline, "kernel_laguna_stage_kv_f16");
        const uint32_t heads_per_kv = n_head / n_head_kv;
        const bool use_gqa6 =
            (n_head % 6u) == 0u && (heads_per_kv % 6u) == 0u;
        const bool use_gqa3 =
            !use_gqa6 && (n_head % 3u) == 0u && (heads_per_kv % 3u) == 0u;
        id<MTLComputePipelineState> attention_pipeline = use_gqa6 ?
            lgn2_gpu_get_pipeline("kernel_laguna_attention_prefill_gqa6_f16") :
            use_gqa3 ?
                lgn2_gpu_get_pipeline("kernel_laguna_attention_prefill_gqa3_f16") :
                lgn2_gpu_hot_pipeline(g_laguna_prefill_attention_pipeline,
                                     "kernel_laguna_attention_prefill_gqa_f16");
        id<MTLComputePipelineState> decode_attention_pipeline =
            lgn2_gpu_hot_pipeline(g_laguna_attention_pipeline,
                                 "kernel_laguna_attention_decode_gqa_f16");
        id<MTLComputePipelineState> commit_pipeline = lgn2_gpu_hot_pipeline(
            g_laguna_commit_kv_pipeline, "kernel_laguna_commit_kv_f16");
        if (!headsbuf || !keybuf || !valuebuf || !stagedkeybuf ||
            !stagedvaluebuf || !qbuf || !kbuf || !vbuf || !gatebuf ||
            !store_pipeline || !stage_pipeline || !attention_pipeline ||
            !decode_attention_pipeline || !commit_pipeline) {
            return 0;
        }

        /*
         * A speculative verifier has only a handful of query rows.  At long
         * context, assigning one SIMD group to each query makes that group
         * walk thousands of keys serially.  Encode the established split-key
         * decode attention once per row instead.  The rows still share one
         * command buffer, and storing them in order preserves causal and SWA
         * ring semantics exactly.
         */
        const bool use_split_decode_rows =
            split_decode_rows != 0 && n_tokens <= 16u;
        if (use_split_decode_rows) {
            const NSUInteger q_row_bytes =
                (NSUInteger)n_head * head_dim * sizeof(float);
            const NSUInteger kv_row_bytes =
                (NSUInteger)n_head_kv * head_dim * sizeof(float);
            const NSUInteger gate_row_bytes =
                (NSUInteger)n_head * sizeof(float);
            int owned = 0;
            id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
            if (!cb) return 0;

            const bool use_staged_swa_rows =
                split_decode_rows != 0 &&
                lgn2_gpu_laguna_staged_swa_enabled() > 0 &&
                n_tokens >= 2u && n_tokens <= 16u &&
                cache_cap == 512u && pos0 >= cache_cap;
            const bool require_staged_swa_rows =
                lgn2_gpu_env_bool("LGN2_METAL_LAGUNA_REQUIRE_STAGED_SWA") > 0;
            int staged_swa_rows = 0;
            if (use_staged_swa_rows) {
                staged_swa_rows =
                    lgn2_gpu_encode_laguna_flash_attention_staged_swa_rows(
                    cb,
                    stage_pipeline,
                    headsbuf,
                    lgn2_gpu_tensor_offset(heads),
                    qbuf,
                    lgn2_gpu_tensor_offset(q),
                    kbuf,
                    lgn2_gpu_tensor_offset(k),
                    vbuf,
                    lgn2_gpu_tensor_offset(v),
                    gatebuf,
                    lgn2_gpu_tensor_offset(gate),
                    keybuf,
                    lgn2_gpu_tensor_offset(key_cache),
                    valuebuf,
                    lgn2_gpu_tensor_offset(value_cache),
                    stagedkeybuf,
                    lgn2_gpu_tensor_offset(staged_key),
                    stagedvaluebuf,
                    lgn2_gpu_tensor_offset(staged_value),
                    pos0,
                    n_tokens,
                    cache_cap,
                    n_head,
                    n_head_kv,
                    head_dim,
                    scale);
            }
            if (staged_swa_rows) {
                id<MTLComputeCommandEncoder> enc =
                    lgn2_gpu_compute_encoder(cb);
                if (!enc) return 0;
                const lgn2_gpu_laguna_prefill_attention_args args = {
                    .n_tokens = n_tokens,
                    .pos0 = pos0,
                    .cache_cap = cache_cap,
                    .n_head = n_head,
                    .n_head_kv = n_head_kv,
                    .head_dim = head_dim,
                    .scale = scale,
                    .pad0 = 0u,
                };
                [enc setComputePipelineState:commit_pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:stagedkeybuf
                        offset:lgn2_gpu_tensor_offset(staged_key)
                       atIndex:1];
                [enc setBuffer:stagedvaluebuf
                        offset:lgn2_gpu_tensor_offset(staged_value)
                       atIndex:2];
                [enc setBuffer:keybuf
                        offset:lgn2_gpu_tensor_offset(key_cache)
                       atIndex:3];
                [enc setBuffer:valuebuf
                        offset:lgn2_gpu_tensor_offset(value_cache)
                       atIndex:4];
                [enc dispatchThreads:
                        MTLSizeMake((NSUInteger)kv_values, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                lgn2_gpu_end_compute_encoder(cb, enc);
                if (!lgn2_gpu_finish_command_buffer(
                        cb, owned, "Laguna staged SWA attention")) {
                    return 0;
                }
#ifdef LGN2_TEST_HOOKS
                /* This is the staged kernel's completion-scoped evidence;
                 * unlike selector diagnostics it is recorded only after the
                 * command buffer has finished successfully. */
                lgn2_gpu_laguna_test_decode_route_note(
                    LGN2_LAGUNA_TEST_DECODE_STAGED, owned);
#endif
                return 1;
            }
            if (use_staged_swa_rows && require_staged_swa_rows) {
                fprintf(stderr,
                        "lgn2: required Metal Laguna staged SWA pipeline unavailable; refusing row-loop fallback\n");
                return 0;
            }

            const uint32_t first_key_count =
                MIN(pos0 + 1u, cache_cap);
            const bool use_batched_global_rows =
                n_tokens >= 2u && n_tokens <= 16u &&
                cache_cap > 512u &&
                pos0 + n_tokens <= cache_cap &&
                (n_head % 3u) == 0u &&
                (heads_per_kv % 3u) == 0u &&
                lgn2_gpu_flash_attn_vec_nsg(first_key_count, 32u, 32u) ==
                lgn2_gpu_flash_attn_vec_nsg(
                    first_key_count + n_tokens - 1u, 32u, 32u);
            if (use_batched_global_rows) {
                for (uint32_t row = 0; row < n_tokens; row++) {
                    const uint32_t pos = pos0 + row;
                    const lgn2_gpu_laguna_kv_store_args store_args = {
                        .cache_cap = cache_cap,
                        .cache_row = pos,
                        .n_head_kv = n_head_kv,
                        .head_dim = head_dim,
                    };
                    id<MTLComputeCommandEncoder> enc =
                        lgn2_gpu_compute_encoder(cb);
                    [enc setComputePipelineState:store_pipeline];
                    [enc setBytes:&store_args
                           length:sizeof(store_args)
                          atIndex:0];
                    [enc setBuffer:kbuf
                            offset:lgn2_gpu_tensor_offset(k) +
                                   (NSUInteger)row * kv_row_bytes
                           atIndex:1];
                    [enc setBuffer:vbuf
                            offset:lgn2_gpu_tensor_offset(v) +
                                   (NSUInteger)row * kv_row_bytes
                           atIndex:2];
                    [enc setBuffer:keybuf
                            offset:lgn2_gpu_tensor_offset(key_cache)
                           atIndex:3];
                    [enc setBuffer:valuebuf
                            offset:lgn2_gpu_tensor_offset(value_cache)
                           atIndex:4];
                    [enc dispatchThreads:
                            MTLSizeMake((NSUInteger)n_head_kv * head_dim,
                                        1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    lgn2_gpu_end_compute_encoder(cb, enc);
                }
                if (!lgn2_gpu_encode_laguna_flash_attention_decode_rows(
                        cb,
                        headsbuf,
                        lgn2_gpu_tensor_offset(heads),
                        qbuf,
                        lgn2_gpu_tensor_offset(q),
                        gatebuf,
                        lgn2_gpu_tensor_offset(gate),
                        keybuf,
                        lgn2_gpu_tensor_offset(key_cache),
                        valuebuf,
                        lgn2_gpu_tensor_offset(value_cache),
                        cache_cap,
                        first_key_count,
                        n_tokens,
                        n_head,
                        n_head_kv,
                        head_dim,
                        scale)) {
                    return 0;
                }
                if (!lgn2_gpu_finish_command_buffer(
                        cb, owned, "Laguna batched-row split attention")) {
                    return 0;
                }
                return 1;
            }

            const bool use_batched_swa_rows =
                n_tokens >= 2u && n_tokens <= 16u &&
                cache_cap == 512u &&
                pos0 + n_tokens <= cache_cap;
            if (use_batched_swa_rows) {
                id<MTLComputePipelineState> rows_store_pipeline =
                    lgn2_gpu_get_pipeline(
                        "kernel_laguna_store_kv_rows_f16");
                id<MTLComputePipelineState> rows_attention_pipeline =
                    lgn2_gpu_get_pipeline(
                        "kernel_laguna_attention_decode_rows_gqa_f16");
                if (!rows_store_pipeline || !rows_attention_pipeline) {
                    return 0;
                }
                const lgn2_gpu_laguna_prefill_attention_args rows_args = {
                    .n_tokens = n_tokens,
                    .pos0 = pos0,
                    .cache_cap = cache_cap,
                    .n_head = n_head,
                    .n_head_kv = n_head_kv,
                    .head_dim = head_dim,
                    .scale = scale,
                    .pad0 = 0u,
                };
                id<MTLComputeCommandEncoder> enc =
                    lgn2_gpu_compute_encoder(cb);
                [enc setComputePipelineState:rows_store_pipeline];
                [enc setBytes:&rows_args
                       length:sizeof(rows_args)
                      atIndex:0];
                [enc setBuffer:kbuf
                        offset:lgn2_gpu_tensor_offset(k)
                       atIndex:1];
                [enc setBuffer:vbuf
                        offset:lgn2_gpu_tensor_offset(v)
                       atIndex:2];
                [enc setBuffer:keybuf
                        offset:lgn2_gpu_tensor_offset(key_cache)
                       atIndex:3];
                [enc setBuffer:valuebuf
                        offset:lgn2_gpu_tensor_offset(value_cache)
                       atIndex:4];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_values, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                lgn2_gpu_end_compute_encoder(cb, enc);

                enc = lgn2_gpu_compute_encoder(cb);
                [enc setComputePipelineState:rows_attention_pipeline];
                [enc setBytes:&rows_args
                       length:sizeof(rows_args)
                      atIndex:0];
                [enc setBuffer:qbuf
                        offset:lgn2_gpu_tensor_offset(q)
                       atIndex:1];
                [enc setBuffer:gatebuf
                        offset:lgn2_gpu_tensor_offset(gate)
                       atIndex:2];
                [enc setBuffer:keybuf
                        offset:lgn2_gpu_tensor_offset(key_cache)
                       atIndex:3];
                [enc setBuffer:valuebuf
                        offset:lgn2_gpu_tensor_offset(value_cache)
                       atIndex:4];
                [enc setBuffer:headsbuf
                        offset:lgn2_gpu_tensor_offset(heads)
                       atIndex:5];
                [enc setThreadgroupMemoryLength:
                        (8u + 8u + 8u * 128u) * sizeof(float)
                                        atIndex:0];
                const uint32_t last_key_count = pos0 + n_tokens;
                const NSUInteger threads =
                    last_key_count > 256u ? 256u : 32u;
                [enc dispatchThreadgroups:
                        MTLSizeMake(n_head, n_tokens, 1)
                     threadsPerThreadgroup:
                        MTLSizeMake(threads, 1, 1)];
                lgn2_gpu_end_compute_encoder(cb, enc);

                if (!lgn2_gpu_finish_command_buffer(
                        cb, owned,
                        "Laguna batched sliding-window attention")) {
                    return 0;
                }
                return 1;
            }

            for (uint32_t row = 0; row < n_tokens; row++) {
                const uint32_t pos = pos0 + row;
                const uint32_t key_count = MIN(pos + 1u, cache_cap);
                const uint32_t key_start = pos + 1u - key_count;
                const lgn2_gpu_laguna_kv_store_args store_args = {
                    .cache_cap = cache_cap,
                    .cache_row = pos % cache_cap,
                    .n_head_kv = n_head_kv,
                    .head_dim = head_dim,
                };
                id<MTLComputeCommandEncoder> enc =
                    lgn2_gpu_compute_encoder(cb);
                [enc setComputePipelineState:store_pipeline];
                [enc setBytes:&store_args
                       length:sizeof(store_args)
                      atIndex:0];
                [enc setBuffer:kbuf
                        offset:lgn2_gpu_tensor_offset(k) +
                               (NSUInteger)row * kv_row_bytes
                       atIndex:1];
                [enc setBuffer:vbuf
                        offset:lgn2_gpu_tensor_offset(v) +
                               (NSUInteger)row * kv_row_bytes
                       atIndex:2];
                [enc setBuffer:keybuf
                        offset:lgn2_gpu_tensor_offset(key_cache)
                       atIndex:3];
                [enc setBuffer:valuebuf
                        offset:lgn2_gpu_tensor_offset(value_cache)
                       atIndex:4];
                [enc dispatchThreads:
                        MTLSizeMake((NSUInteger)n_head_kv * head_dim, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                lgn2_gpu_end_compute_encoder(cb, enc);

                const bool full_sliding_ring =
                    cache_cap == 512u && key_count == cache_cap;
                const bool use_flash_decode =
                    full_sliding_ring ||
                    (cache_cap > 512u && key_start == 0u &&
                     key_count >= 1024u &&
                     (key_count < cache_cap || key_count % 32u == 0u));
                if (use_flash_decode) {
                    if (!lgn2_gpu_encode_laguna_flash_attention_decode(
                            cb,
                            headsbuf,
                            lgn2_gpu_tensor_offset(heads) +
                                (NSUInteger)row * q_row_bytes,
                            qbuf,
                            lgn2_gpu_tensor_offset(q) +
                                (NSUInteger)row * q_row_bytes,
                            gatebuf,
                            lgn2_gpu_tensor_offset(gate) +
                                (NSUInteger)row * gate_row_bytes,
                            keybuf,
                            lgn2_gpu_tensor_offset(key_cache),
                            valuebuf,
                            lgn2_gpu_tensor_offset(value_cache),
                            cache_cap,
                            key_count,
                            n_head,
                            n_head_kv,
                            head_dim,
                            scale,
                            gqa9_selected,
                            gqa3_selected,
                            staged_swa_active)) {
                        return 0;
                    }
                } else {
                    const lgn2_gpu_laguna_attention_args attention_args = {
                        .n_head = n_head,
                        .n_head_kv = n_head_kv,
                        .head_dim = head_dim,
                        .cache_cap = cache_cap,
                        .key_start = key_start,
                        .key_count = key_count,
                        .scale = scale,
                    };
                    enc = lgn2_gpu_compute_encoder(cb);
                    [enc setComputePipelineState:decode_attention_pipeline];
                    [enc setBytes:&attention_args
                           length:sizeof(attention_args)
                          atIndex:0];
                    [enc setBuffer:qbuf
                            offset:lgn2_gpu_tensor_offset(q) +
                                   (NSUInteger)row * q_row_bytes
                           atIndex:1];
                    [enc setBuffer:gatebuf
                            offset:lgn2_gpu_tensor_offset(gate) +
                                   (NSUInteger)row * gate_row_bytes
                           atIndex:2];
                    [enc setBuffer:keybuf
                            offset:lgn2_gpu_tensor_offset(key_cache)
                           atIndex:3];
                    [enc setBuffer:valuebuf
                            offset:lgn2_gpu_tensor_offset(value_cache)
                           atIndex:4];
                    [enc setBuffer:headsbuf
                            offset:lgn2_gpu_tensor_offset(heads) +
                                   (NSUInteger)row * q_row_bytes
                           atIndex:5];
                    [enc setThreadgroupMemoryLength:
                            (8u + 8u + 8u * 128u) * sizeof(float)
                                            atIndex:0];
                    const NSUInteger threads =
                        key_count > 256u ? 256u : 32u;
                    [enc dispatchThreadgroups:MTLSizeMake(n_head, 1, 1)
                         threadsPerThreadgroup:
                            MTLSizeMake(threads, 1, 1)];
                    lgn2_gpu_end_compute_encoder(cb, enc);
                }
            }
            if (!lgn2_gpu_finish_command_buffer(
                    cb, owned, "Laguna split-row attention")) {
                return 0;
            }
            return 1;
        }

        const lgn2_gpu_laguna_prefill_attention_args args = {
            .n_tokens = n_tokens,
            .pos0 = pos0,
            .cache_cap = cache_cap,
            .n_head = n_head,
            .n_head_kv = n_head_kv,
            .head_dim = head_dim,
            .scale = scale,
            .pad0 = 0,
        };
        /*
         * Opt-in direct KV store for global layers.  When the whole chunk
         * fits past pos0 the ring cannot wrap inside the chunk, so the f16
         * conversion may land directly in the cache slot and the commit pass
         * is folded away; attention then reads the chunk rows from the cache
         * through the staged-slot view (staged row t == cache row pos0 + t).
         * SWA layers keep the staged path: their ring can overwrite rows
         * that early queries in the chunk still read.
         */
        const int direct_kv_mode = g_direct_kv_prefill_mode;
        const bool direct_kv =
            direct_kv_mode > 0 && cache_cap > 512u &&
            (uint64_t)pos0 + (uint64_t)n_tokens <= (uint64_t)cache_cap;

        /* The direct route deliberately reuses the already initialized
         * linear f32->f16 staging PSO.  It must not probe a per-call rows PSO
         * or execute a token division/modulo store kernel. */
        const uint64_t kv_row_bytes =
            (uint64_t)n_head_kv * head_dim * sizeof(uint16_t);
        if (direct_kv &&
            (uint64_t)pos0 > UINT64_MAX / kv_row_bytes) {
            return 0;
        }
        const uint64_t cache_offset_bytes =
            (uint64_t)pos0 * kv_row_bytes;
        if (direct_kv &&
            (cache_offset_bytes > (uint64_t)NSUIntegerMax ||
             lgn2_gpu_tensor_offset(key_cache) >
                 (NSUIntegerMax - (NSUInteger)cache_offset_bytes) ||
             lgn2_gpu_tensor_offset(value_cache) >
                 (NSUIntegerMax - (NSUInteger)cache_offset_bytes))) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:stage_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kbuf offset:lgn2_gpu_tensor_offset(k) atIndex:1];
        [enc setBuffer:vbuf offset:lgn2_gpu_tensor_offset(v) atIndex:2];
        [enc setBuffer:direct_kv ? keybuf : stagedkeybuf
                offset:(direct_kv ?
                            lgn2_gpu_tensor_offset(key_cache) +
                                (NSUInteger)cache_offset_bytes :
                            lgn2_gpu_tensor_offset(staged_key))
                atIndex:3];
        [enc setBuffer:direct_kv ? valuebuf : stagedvaluebuf
                offset:(direct_kv ?
                            lgn2_gpu_tensor_offset(value_cache) +
                                (NSUInteger)cache_offset_bytes :
                            lgn2_gpu_tensor_offset(staged_value))
                atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_values, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:attention_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:lgn2_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:gatebuf offset:lgn2_gpu_tensor_offset(gate) atIndex:2];
        [enc setBuffer:keybuf offset:lgn2_gpu_tensor_offset(key_cache) atIndex:3];
        [enc setBuffer:valuebuf offset:lgn2_gpu_tensor_offset(value_cache) atIndex:4];
        [enc setBuffer:direct_kv ? keybuf : stagedkeybuf
                offset:(direct_kv ?
                            lgn2_gpu_tensor_offset(key_cache) +
                                (NSUInteger)cache_offset_bytes :
                            lgn2_gpu_tensor_offset(staged_key))
                atIndex:5];
        [enc setBuffer:direct_kv ? valuebuf : stagedvaluebuf
                offset:(direct_kv ?
                            lgn2_gpu_tensor_offset(value_cache) +
                                (NSUInteger)cache_offset_bytes :
                            lgn2_gpu_tensor_offset(staged_value))
                atIndex:6];
        [enc setBuffer:headsbuf offset:lgn2_gpu_tensor_offset(heads) atIndex:7];
        const uint32_t attention_groups = use_gqa6 ?
            n_head / 6u : use_gqa3 ? n_head / 3u : n_head;
        [enc dispatchThreadgroups:MTLSizeMake(attention_groups,
                                              n_tokens,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!direct_kv) {
            enc = lgn2_gpu_compute_encoder(cb);
            [enc setComputePipelineState:commit_pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:stagedkeybuf offset:lgn2_gpu_tensor_offset(staged_key)
                    atIndex:1];
            [enc setBuffer:stagedvaluebuf offset:lgn2_gpu_tensor_offset(staged_value)
                    atIndex:2];
            [enc setBuffer:keybuf offset:lgn2_gpu_tensor_offset(key_cache) atIndex:3];
            [enc setBuffer:valuebuf offset:lgn2_gpu_tensor_offset(value_cache) atIndex:4];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_values, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            lgn2_gpu_end_compute_encoder(cb, enc);
        }

        if (!lgn2_gpu_finish_command_buffer(cb, owned,
                                            "Laguna prefill attention")) {
            return 0;
        }
        /* Count only a successfully submitted route.  This keeps the
         * test-only evidence from turning a preflight failure into a false
         * direct/wrap dispatch proof. */
#ifdef LGN2_TEST_HOOKS
        if (g_laguna_test_route_hooks) {
            if (direct_kv) g_laguna_test_direct_kv_count++;
            else g_laguna_test_wrap_kv_count++;
        }
#endif
    }
    return 1;
}

typedef struct {
    uint32_t n_rows;
    uint32_t n_embd;
    uint32_t n_aux;
    uint32_t aux_index;
    uint32_t src_row0;
    uint32_t dst_row0;
} lgn2_gpu_dflash_capture_args;

int lgn2_gpu_dflash_capture_rows_tensor(
        lgn2_gpu_tensor       *features,
        const lgn2_gpu_tensor *src,
        uint32_t              src_row0,
        uint32_t              dst_row0,
        uint32_t              n_rows,
        uint32_t              n_embd,
        uint32_t              n_aux,
        uint32_t              aux_index) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!features || !src || n_rows == 0u || n_embd == 0u ||
        n_aux == 0u || aux_index >= n_aux) {
        return 0;
    }
    const uint64_t src_rows = (uint64_t)src_row0 + n_rows;
    const uint64_t dst_rows = (uint64_t)dst_row0 + n_rows;
    const uint64_t src_values = src_rows * n_embd;
    const uint64_t dst_values = dst_rows * n_aux * n_embd;
    if (src_values > UINT64_MAX / sizeof(float) ||
        dst_values > UINT64_MAX / sizeof(float) ||
        lgn2_gpu_tensor_bytes(src) < src_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(features) < dst_values * sizeof(float)) {
        fprintf(stderr, "lgn2: Metal DFlash capture received undersized buffers\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> srcbuf = lgn2_gpu_tensor_buffer(src);
        id<MTLBuffer> dstbuf = lgn2_gpu_tensor_buffer(features);
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_pipeline("kernel_dflash_capture_rows");
        if (!srcbuf || !dstbuf || !pipeline) return 0;

        const lgn2_gpu_dflash_capture_args args = {
            .n_rows = n_rows,
            .n_embd = n_embd,
            .n_aux = n_aux,
            .aux_index = aux_index,
            .src_row0 = src_row0,
            .dst_row0 = dst_row0,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:srcbuf offset:lgn2_gpu_tensor_offset(src) atIndex:1];
        [enc setBuffer:dstbuf offset:lgn2_gpu_tensor_offset(features) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n_rows * n_embd, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        if (!lgn2_gpu_finish_command_buffer(cb, owned,
                                            "DFlash feature capture")) {
            return 0;
        }
    }
    return 1;
}

typedef struct {
    uint32_t n_rows;
    uint32_t n_embd;
    uint32_t n_aux;
    float eps;
} lgn2_gpu_dflash_aux_norm_args;

int lgn2_gpu_dflash_aux_norm_tensor(
        lgn2_gpu_tensor *features,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        n_rows,
        uint32_t        n_embd,
        uint32_t        n_aux,
        float           eps) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!features || !model_map || n_rows == 0u || n_embd == 0u ||
        n_aux == 0u || !isfinite(eps) || eps < 0.0f) {
        return 0;
    }
    const uint64_t feature_values = (uint64_t)n_rows * n_aux * n_embd;
    const uint64_t weight_values = (uint64_t)n_aux * n_embd;
    if (feature_values > UINT64_MAX / sizeof(float) ||
        weight_values > UINT64_MAX / sizeof(float) ||
        lgn2_gpu_tensor_bytes(features) <
            feature_values * sizeof(float) ||
        weight_offset > model_size ||
        weight_values * sizeof(float) > model_size - weight_offset) {
        fprintf(stderr, "lgn2: Metal DFlash auxiliary norm received invalid ranges\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> featurebuf = lgn2_gpu_tensor_buffer(features);
        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf = lgn2_gpu_wrap_model_range(
            model_map,
            model_size,
            weight_offset,
            weight_values * sizeof(float),
            &weight_inner);
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_pipeline("kernel_dflash_aux_rms_norm");
        if (!featurebuf || !weightbuf || !pipeline) return 0;

        const lgn2_gpu_dflash_aux_norm_args args = {
            .n_rows = n_rows,
            .n_embd = n_embd,
            .n_aux = n_aux,
            .eps = eps,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:featurebuf
                offset:lgn2_gpu_tensor_offset(features)
               atIndex:1];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:2];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_rows, n_aux, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        if (!lgn2_gpu_finish_command_buffer(cb, owned,
                                            "DFlash auxiliary norm")) {
            return 0;
        }
    }
    return 1;
}

typedef struct {
    uint32_t n_rows;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t width;
} lgn2_gpu_dflash_commit_kv_args;

int lgn2_gpu_dflash_commit_kv_tensor(
        lgn2_gpu_tensor       *key_cache,
        lgn2_gpu_tensor       *value_cache,
        const lgn2_gpu_tensor *k,
        const lgn2_gpu_tensor *v,
        uint32_t              pos0,
        uint32_t              n_rows,
        uint32_t              cache_cap,
        uint32_t              n_head_kv,
        uint32_t              head_dim) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!key_cache || !value_cache || !k || !v || n_rows == 0u ||
        cache_cap == 0u || n_head_kv == 0u || head_dim == 0u) {
        return 0;
    }
    const uint64_t width = (uint64_t)n_head_kv * head_dim;
    const uint64_t src_values = (uint64_t)n_rows * width;
    const uint64_t cache_values = (uint64_t)cache_cap * width;
    if (width > UINT32_MAX ||
        src_values > UINT64_MAX / sizeof(float) ||
        cache_values > UINT64_MAX / sizeof(uint16_t) ||
        lgn2_gpu_tensor_bytes(k) < src_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(v) < src_values * sizeof(float) ||
        lgn2_gpu_tensor_bytes(key_cache) <
            cache_values * sizeof(uint16_t) ||
        lgn2_gpu_tensor_bytes(value_cache) <
            cache_values * sizeof(uint16_t)) {
        fprintf(stderr, "lgn2: Metal DFlash KV commit received undersized buffers\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> keybuf = lgn2_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valuebuf = lgn2_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> kbuf = lgn2_gpu_tensor_buffer(k);
        id<MTLBuffer> vbuf = lgn2_gpu_tensor_buffer(v);
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_pipeline("kernel_dflash_commit_kv_f16");
        if (!keybuf || !valuebuf || !kbuf || !vbuf || !pipeline) return 0;

        const lgn2_gpu_dflash_commit_kv_args args = {
            .n_rows = n_rows,
            .pos0 = pos0,
            .cache_cap = cache_cap,
            .width = (uint32_t)width,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kbuf offset:lgn2_gpu_tensor_offset(k) atIndex:1];
        [enc setBuffer:vbuf offset:lgn2_gpu_tensor_offset(v) atIndex:2];
        [enc setBuffer:keybuf
                offset:lgn2_gpu_tensor_offset(key_cache)
               atIndex:3];
        [enc setBuffer:valuebuf
                offset:lgn2_gpu_tensor_offset(value_cache)
               atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)src_values, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        if (!lgn2_gpu_finish_command_buffer(cb, owned,
                                            "DFlash KV commit")) {
            return 0;
        }
    }
    return 1;
}

typedef struct {
    uint32_t n_rows;
    uint32_t n_vocab;
} lgn2_gpu_dflash_probabilities_args;

int lgn2_gpu_dflash_probabilities_tensor(
        lgn2_gpu_tensor       *probabilities,
        const lgn2_gpu_tensor *logits,
        const lgn2_gpu_tensor *argmax,
        uint32_t              n_rows,
        uint32_t              n_vocab) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!probabilities || !logits || !argmax ||
        n_rows == 0u || n_vocab == 0u ||
        lgn2_gpu_tensor_bytes(probabilities) <
            (uint64_t)n_rows * sizeof(float) ||
        lgn2_gpu_tensor_bytes(logits) <
            (uint64_t)n_rows * n_vocab * sizeof(float) ||
        lgn2_gpu_tensor_bytes(argmax) <
            (uint64_t)n_rows * sizeof(int32_t)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> probabilitiesbuf =
            lgn2_gpu_tensor_buffer(probabilities);
        id<MTLBuffer> logitsbuf = lgn2_gpu_tensor_buffer(logits);
        id<MTLBuffer> argmaxbuf = lgn2_gpu_tensor_buffer(argmax);
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_get_pipeline("kernel_dflash_probabilities");
        if (!probabilitiesbuf || !logitsbuf || !argmaxbuf || !pipeline) {
            return 0;
        }

        const lgn2_gpu_dflash_probabilities_args args = {
            .n_rows = n_rows,
            .n_vocab = n_vocab,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:logitsbuf
                offset:lgn2_gpu_tensor_offset(logits)
               atIndex:1];
        [enc setBuffer:argmaxbuf
                offset:lgn2_gpu_tensor_offset(argmax)
               atIndex:2];
        [enc setBuffer:probabilitiesbuf
                offset:lgn2_gpu_tensor_offset(probabilities)
               atIndex:3];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        if (!lgn2_gpu_finish_command_buffer(
                cb, owned, "DFlash draft probabilities")) {
            return 0;
        }
    }
    return 1;
}

static int lgn2_gpu_laguna_router_simd_topk_env_mode(void) {
    const char *value = getenv("LGN2_METAL_LAGUNA_ROUTER_SIMD_TOPK");
    if (!value || value[0] == '\0' || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") == 0) return 1;
    fprintf(stderr,
            "lgn2: fatal: LGN2_METAL_LAGUNA_ROUTER_SIMD_TOPK must be unset, "
            "0, or literal 1 (got %s)\n",
            value);
    return -1;
}

static id<MTLBuffer> lgn2_gpu_laguna_router_simd_topk_stats_buffer(void) {
    if (!g_glm_router_simd_topk_stats_buffer && g_device) {
        g_glm_router_simd_topk_stats_buffer =
            [g_device newBufferWithLength:2u * sizeof(uint32_t)
                                  options:MTLResourceStorageModeShared];
    }
    return g_glm_router_simd_topk_stats_buffer;
}

static bool lgn2_gpu_laguna_router_simd_topk_trace_enabled(void) {
    return getenv("LGN2_METAL_LAGUNA_ROUTER_SIMD_TOPK_TRACE") != NULL &&
           strcmp(getenv("LGN2_METAL_LAGUNA_ROUTER_SIMD_TOPK_TRACE"), "1") == 0;
}

/* Returns 1 only when the opt-in path is fully available, 0 for ordinary
 * env-off operation, and -1 for an explicit opt-in contract violation. */
static int lgn2_gpu_laguna_router_simd_topk_validate(
        uint32_t n_expert,
        uint32_t n_expert_used,
        float    expert_weight_scale) {
    const int mode = lgn2_gpu_laguna_router_simd_topk_env_mode();
    if (mode <= 0) return mode;
    if (n_expert != 256u || n_expert_used != 10u) {
        fprintf(stderr,
                "lgn2: fatal: Laguna router SIMD top-k requires n_expert=256 "
                "and n_expert_used=10 (got %u/%u)\n",
                n_expert, n_expert_used);
        return -1;
    }
    if (expert_weight_scale != 2.5f) {
        fprintf(stderr,
                "lgn2: fatal: Laguna router SIMD top-k requires "
                "expert_weight_scale=2.5 (got %.9g)\n",
                expert_weight_scale);
        return -1;
    }

    /* Keep env-off completely quiet and side-effect free.  An explicit
     * request, however, is an availability probe and must initialize Metal
     * before asking the library for the optional PSO; source overrides are
     * otherwise indistinguishable from an uninitialized device. */
    if (!g_initialized && !lgn2_gpu_init()) {
        fprintf(stderr,
                "lgn2: fatal: Laguna router SIMD top-k Metal initialization failed\n");
        return -1;
    }

    /* The source may be overridden for diagnostics.  Keep this optional
     * pipeline lazy so env-off old-source runs remain quiet and stock. */
    if (!g_glm_router_select_one_simd_pipeline) {
        g_glm_router_select_one_simd_pipeline =
            lgn2_gpu_get_pipeline("kernel_glm_router_select_one_simd");
    }
    if (!g_glm_router_select_one_simd_pipeline) {
        fprintf(stderr,
                "lgn2: fatal: Laguna router SIMD top-k kernel/PSO unavailable\n");
        return -1;
    }
    if (g_glm_router_select_one_simd_pipeline.threadExecutionWidth != 32u ||
        g_glm_router_select_one_simd_pipeline.maxTotalThreadsPerThreadgroup < 256u) {
        fprintf(stderr,
                "lgn2: fatal: Laguna router SIMD top-k requires TEW=32 and "
                "maxThreads>=256 (got TEW=%lu maxThreads=%lu)\n",
                (unsigned long)g_glm_router_select_one_simd_pipeline.threadExecutionWidth,
                (unsigned long)g_glm_router_select_one_simd_pipeline.maxTotalThreadsPerThreadgroup);
        return -1;
    }
    if (!lgn2_gpu_laguna_router_simd_topk_stats_buffer()) {
        fprintf(stderr,
                "lgn2: fatal: Laguna router SIMD top-k stats buffer unavailable\n");
        return -1;
    }
    return 1;
}

int lgn2_gpu_laguna_router_simd_topk_preflight(
        uint32_t n_expert,
        uint32_t n_expert_used,
        float    expert_weight_scale) {
    return lgn2_gpu_laguna_router_simd_topk_validate(
        n_expert, n_expert_used, expert_weight_scale);
}

/* A source override can compile the rest of the Laguna graph while predating
 * the optional GQA9 function, so resolving the PSO lazily from the attention
 * encoder is unsafe: that encoder may already have recorded the KV store into
 * the caller-owned batch.  Probe every resource the grouped route needs before
 * a command batch, graph, or cache can be touched. */
static int lgn2_gpu_laguna_swa_gqa9_shape_valid(
        uint32_t cache_cap,
        uint32_t key_count,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t head_dim) {
    return cache_cap == 512u && key_count == 512u &&
        n_head == 72u && n_head_kv == 8u && head_dim == 128u;
}

int lgn2_gpu_laguna_swa_gqa9_preflight(
        uint32_t cache_cap,
        uint32_t key_count,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t head_dim) {
    int mode = lgn2_gpu_laguna_swa_gqa9_env_mode();
    if (mode < 0) return -1;
    if (mode == 0) return 0;

    /* Staged SWA owns the full ring at higher precedence.  Parse GQA9 first
     * so a malformed explicit GQA9 value never hides behind staged routing,
     * but do not impose GQA9's production shape or source symbol on the
     * winning staged route. */
    if (lgn2_gpu_laguna_staged_swa_enabled() > 0) return 0;

    if (!lgn2_gpu_laguna_swa_gqa9_shape_valid(
            cache_cap, key_count, n_head, n_head_kv, head_dim)) {
        fprintf(stderr,
                "lgn2: %s requested but requires cache/key=512/512, "
                "heads=72/8, head_dim=128 (got cache/key=%u/%u "
                "heads=%u/%u dim=%u); refusing fallback\n",
                LGN2_METAL_LAGUNA_SWA_GQA9,
                cache_cap,
                key_count,
                n_head,
                n_head_kv,
                head_dim);
        return -1;
    }

    /* Env-off must remain quiet and must not initialize Metal.  The explicit
     * request is an availability probe; init is still before any command
     * batch and does not allocate a Laguna graph or mutate KV state. */
    if (!g_initialized && !lgn2_gpu_init()) {
        fprintf(stderr,
                "lgn2: %s requested but Metal initialization failed\n",
                LGN2_METAL_LAGUNA_SWA_GQA9);
        return -1;
    }
    mode = lgn2_gpu_laguna_swa_gqa9_env_mode();
    if (mode <= 0) return mode;

    @autoreleasepool {
        const uint32_t nwg = 32u;
        const uint32_t ncpsg = 32u;
        const uint32_t nsg = lgn2_gpu_flash_attn_vec_nsg(
            key_count, nwg, ncpsg);
        id<MTLComputePipelineState> grouped_pipeline =
            lgn2_gpu_get_pipeline(
                "kernel_laguna_attention_decode_gqa9_split_f16");
        id<MTLComputePipelineState> reduce_pipeline =
            lgn2_gpu_get_laguna_flash_attn_reduce_gate_pipeline(
                (int32_t)head_dim, (int32_t)nwg);
        const NSUInteger grouped_threads = 32u * (NSUInteger)nsg;
        const NSUInteger grouped_shared =
            9u * (NSUInteger)nsg * (2u + (NSUInteger)head_dim) *
            sizeof(float);
        const NSUInteger reduce_threads = 32u * (NSUInteger)nwg;
        const NSUInteger max_threadgroup_memory =
            g_device ? [g_device maxThreadgroupMemoryLength] : 0u;
        const bool grouped_ready =
            grouped_pipeline && reduce_pipeline &&
            grouped_pipeline.threadExecutionWidth == 32u &&
            grouped_pipeline.maxTotalThreadsPerThreadgroup >= grouped_threads &&
            reduce_pipeline.threadExecutionWidth == 32u &&
            reduce_pipeline.maxTotalThreadsPerThreadgroup >= reduce_threads &&
            max_threadgroup_memory >= grouped_shared;
        if (!grouped_ready) {
            fprintf(stderr,
                    "lgn2: %s requested but GQA9 PSO/TEW/threadgroup "
                    "resources are unavailable (gqa9=%s reduce=%s "
                    "gqa9_tew=%lu gqa9_max=%lu reduce_tew=%lu "
                    "reduce_max=%lu need_threads=%lu/%lu "
                    "shared=%lu max_shared=%lu); refusing fallback\n",
                    LGN2_METAL_LAGUNA_SWA_GQA9,
                    grouped_pipeline ? "yes" : "no",
                    reduce_pipeline ? "yes" : "no",
                    grouped_pipeline ?
                        (unsigned long)grouped_pipeline.threadExecutionWidth : 0ul,
                    grouped_pipeline ?
                        (unsigned long)grouped_pipeline.maxTotalThreadsPerThreadgroup : 0ul,
                    reduce_pipeline ?
                        (unsigned long)reduce_pipeline.threadExecutionWidth : 0ul,
                    reduce_pipeline ?
                        (unsigned long)reduce_pipeline.maxTotalThreadsPerThreadgroup : 0ul,
                    (unsigned long)grouped_threads,
                    (unsigned long)reduce_threads,
                    (unsigned long)grouped_shared,
                    (unsigned long)max_threadgroup_memory);
            return -1;
        }
    }
    return 1;
}

static void lgn2_gpu_laguna_router_simd_topk_report_once(void) {
    if (g_glm_router_simd_topk_reported) return;
    fprintf(stderr,
            "lgn2: Laguna router SIMD top-k selector enabled "
            "(n_expert=256 n_used=10 tew=32; nonfinite rows use stock fallback)\n");
    g_glm_router_simd_topk_reported = 1;
}

int lgn2_gpu_glm_router_select_tensor(
        lgn2_gpu_tensor       *selected,
        lgn2_gpu_tensor       *weights,
        lgn2_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const lgn2_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_expert == 0 || n_expert > 256u ||
        n_expert_used == 0 || n_expert_used > n_expert) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = lgn2_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = lgn2_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = lgn2_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = lgn2_gpu_tensor_buffer(probs);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            lgn2_gpu_tensor_bytes(logits) < (uint64_t)n_expert * sizeof(float) ||
            lgn2_gpu_tensor_bytes(selected) < (uint64_t)n_expert_used * sizeof(int32_t) ||
            lgn2_gpu_tensor_bytes(weights) < (uint64_t)n_expert_used * sizeof(float) ||
            lgn2_gpu_tensor_bytes(probs) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "lgn2: Metal Laguna router received undersized buffers\n");
            return 0;
        }

        const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
        if (bias_offset > model_size || bias_bytes > model_size - bias_offset) {
            fprintf(stderr, "lgn2: Metal Laguna router bias range is outside the mapped model\n");
            return 0;
        }
        const bool exact_bias_view =
            bias_bytes <= (1ull << 20) &&
            getenv("LGN2_METAL_DISABLE_DECODE_ROUTER_BIAS_EXACT_VIEWS") == NULL;
        uint64_t bias_inner = 0;
        id<MTLBuffer> biasbuf = exact_bias_view ?
            lgn2_gpu_wrap_model_exact_range(model_map,
                                           model_size,
                                           bias_offset,
                                           bias_bytes,
                                           &bias_inner) :
            lgn2_gpu_wrap_model_range(model_map,
                                     model_size,
                                     bias_offset,
                                     bias_bytes,
                                     &bias_inner);
        if (!biasbuf) return 0;

        const int simd_topk_mode =
            lgn2_gpu_laguna_router_simd_topk_validate(
                n_expert, n_expert_used, expert_weight_scale);
        if (simd_topk_mode < 0) return 0;
        const bool use_simd_topk = simd_topk_mode > 0;
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_hot_pipeline(
                use_simd_topk ? g_glm_router_select_one_simd_pipeline
                              : g_glm_router_select_one_pipeline,
                use_simd_topk ? "kernel_glm_router_select_one_simd"
                              : "kernel_glm_router_select_one");
        if (!pipeline) return 0;
        if (use_simd_topk) {
            lgn2_gpu_laguna_router_simd_topk_report_once();
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        lgn2_gpu_glm_router_select_one_args args = {
            .n_expert = n_expert,
            .n_expert_used = n_expert_used,
            .expert_weight_scale = expert_weight_scale,
            .stats_enabled = use_simd_topk &&
                lgn2_gpu_laguna_router_simd_topk_trace_enabled() ? 1u : 0u,
        };

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:logitsbuf offset:lgn2_gpu_tensor_offset(logits) atIndex:1];
        [enc setBuffer:biasbuf offset:(NSUInteger)bias_inner atIndex:2];
        [enc setBuffer:selectedbuf offset:lgn2_gpu_tensor_offset(selected) atIndex:3];
        [enc setBuffer:weightsbuf offset:lgn2_gpu_tensor_offset(weights) atIndex:4];
        [enc setBuffer:probsbuf offset:lgn2_gpu_tensor_offset(probs) atIndex:5];
        if (use_simd_topk) {
            [enc setBuffer:lgn2_gpu_laguna_router_simd_topk_stats_buffer()
                   offset:0 atIndex:6];
        }
        [enc setThreadgroupMemoryLength:use_simd_topk
                ? 512u * sizeof(float) + 256u * sizeof(uint32_t)
                : 256u * sizeof(float) + 256u * sizeof(int32_t)
                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (use_simd_topk) {
            g_glm_router_simd_topk_encoded_rows++;
            g_glm_router_simd_topk_encoded_dispatches++;
        }

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "Laguna router select")) return 0;
    }

    return 1;
}

typedef struct {
    uint32_t in_dim;
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t stats_enabled;
} lgn2_gpu_laguna_router_fused_args;

int lgn2_gpu_laguna_router_decode_fused_preflight(
        uint32_t in_dim,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float    expert_weight_scale) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    /* The production shader's shape fields are signed at the public
     * boundary even though this wrapper accepts uint32_t.  Keep oversized
     * dimensions fail-closed before the NSG arithmetic or argument cast. */
    if (in_dim == 0u || in_dim > (uint32_t)INT32_MAX ||
        n_expert > (uint32_t)INT32_MAX ||
        n_expert_used > (uint32_t)INT32_MAX ||
        (in_dim % 32u) != 0u ||
        in_dim > UINT32_MAX - 127u || n_expert != 256u ||
        n_expert_used != 10u || !isfinite(expert_weight_scale) ||
        expert_weight_scale != 2.5f) {
        fprintf(stderr,
                "lgn2: Metal fused Laguna router preflight rejected shape "
                "in=%u experts=%u/%u scale=%g\n",
                in_dim, n_expert, n_expert_used, expert_weight_scale);
        return 0;
    }
    const uint32_t nsg_raw = (in_dim + 127u) / 128u;
    const uint32_t nsg = nsg_raw > 8u ? 8u : nsg_raw;
    if (nsg != 8u) {
        fprintf(stderr,
                "lgn2: Metal fused Laguna router preflight requires NSG=8 "
                "(in=%u gives NSG=%u)\n", in_dim, nsg);
        return 0;
    }

    /* This is a single 256-threadgroup serial router by design.  Its shape
     * and source certificate are correctness gates; performance remains a
     * benchmark risk until the M5 measurements justify the opt-in. */
    if (!g_laguna_router_decode_fused_pipeline) {
        g_laguna_router_decode_fused_pipeline =
            lgn2_gpu_get_pipeline("kernel_laguna_router_decode_fused");
    }
    id<MTLComputePipelineState> pipeline = lgn2_gpu_hot_pipeline(
        g_laguna_router_decode_fused_pipeline,
        "kernel_laguna_router_decode_fused");
    if (!pipeline || pipeline.threadExecutionWidth != 32u ||
        pipeline.maxTotalThreadsPerThreadgroup < 256u) {
        fprintf(stderr,
                "lgn2: Metal fused Laguna router preflight requires the "
                "current source PSO with TEW=32 and maxThreads>=256\n");
        return 0;
    }
    /* The fused kernel always binds its completion/statistics buffer, even
     * with atomic reporting disabled.  Allocate it during admission so a
     * later command path cannot fail after it has changed graph state. */
    return lgn2_gpu_laguna_router_simd_topk_stats_buffer() != nil;
}

/* Opt-in fused Laguna decode router (LGN2_METAL_LAGUNA_ROUTER_DECODE_FUSED):
 * the F32 router matvec and the SIMD top-k selection share one dispatch with
 * the logits staged in threadgroup memory.  The matvec stage replicates the
 * stock decode-row matvec reduction tree, so selected/weights/probs/logits
 * are bit-identical to the two-dispatch path (see metal/dsv4_misc.metal).
 * Fails closed outside the replicated shape class.  The one-TG serial
 * geometry is a benchmark risk, not a reason to weaken these correctness
 * gates before M5 measurements exist. */
int lgn2_gpu_laguna_router_decode_fused_tensor(
        lgn2_gpu_tensor       *selected,
        lgn2_gpu_tensor       *weights,
        lgn2_gpu_tensor       *probs,
        lgn2_gpu_tensor       *logits,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        const lgn2_gpu_tensor *x,
        uint32_t              in_dim,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        float                 expert_weight_scale) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !x || !model_map) {
        return 0;
    }

    const uint32_t nsg_raw =
        in_dim <= UINT32_MAX - 127u ? (in_dim + 127u) / 128u : 0u;
    const uint32_t nsg = nsg_raw > 8u ? 8u : nsg_raw;
    if (in_dim == 0u || in_dim > (uint32_t)INT32_MAX ||
        n_expert > (uint32_t)INT32_MAX ||
        n_expert_used > (uint32_t)INT32_MAX ||
        (in_dim % 32u) != 0u || nsg != 8u ||
        n_expert != 256u || n_expert_used != 10u ||
        !isfinite(expert_weight_scale) || expert_weight_scale != 2.5f) {
        fprintf(stderr,
                "lgn2: Metal fused Laguna router received unsupported shape "
                "in=%u experts=%u/%u scale=%g\n",
                in_dim, n_expert, n_expert_used, expert_weight_scale);
        return 0;
    }
    if (!lgn2_gpu_laguna_router_decode_fused_preflight(
            in_dim, n_expert, n_expert_used, expert_weight_scale)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> logitsbuf = lgn2_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = lgn2_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = lgn2_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = lgn2_gpu_tensor_buffer(probs);
        const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
        const uint64_t expert_bytes = (uint64_t)n_expert * sizeof(float);
        const uint64_t selected_bytes =
            (uint64_t)n_expert_used * sizeof(int32_t);
        const uint64_t weights_bytes =
            (uint64_t)n_expert_used * sizeof(float);
        if (!xbuf || !logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(logits) < expert_bytes ||
            lgn2_gpu_tensor_bytes(probs) < expert_bytes ||
            lgn2_gpu_tensor_bytes(selected) < selected_bytes ||
            lgn2_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "lgn2: Metal fused Laguna router received undersized buffers\n");
            return 0;
        }

        if ((uint64_t)n_expert > UINT64_MAX / x_bytes) return 0;
        const uint64_t weight_bytes = (uint64_t)n_expert * x_bytes;
        if (weight_offset > model_size ||
            weight_bytes > model_size - weight_offset ||
            bias_offset > model_size ||
            expert_bytes > model_size - bias_offset) {
            fprintf(stderr, "lgn2: Metal fused Laguna router range is outside the mapped model\n");
            return 0;
        }
        uint64_t weight_inner = 0;
        id<MTLBuffer> wbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, weight_offset, weight_bytes, &weight_inner);
        uint64_t bias_inner = 0;
        id<MTLBuffer> biasbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, bias_offset, expert_bytes, &bias_inner);
        if (!wbuf || !biasbuf) return 0;

        if (!g_laguna_router_decode_fused_pipeline) {
            g_laguna_router_decode_fused_pipeline =
                lgn2_gpu_get_pipeline("kernel_laguna_router_decode_fused");
        }
        id<MTLComputePipelineState> pipeline = lgn2_gpu_hot_pipeline(
            g_laguna_router_decode_fused_pipeline,
            "kernel_laguna_router_decode_fused");
        if (!pipeline) return 0;
        if (pipeline.threadExecutionWidth != 32u ||
            pipeline.maxTotalThreadsPerThreadgroup < 256u) {
            fprintf(stderr,
                    "lgn2: Metal fused Laguna router requires TEW=32 and "
                    "maxThreads>=256 (got TEW=%lu maxThreads=%lu)\n",
                    (unsigned long)pipeline.threadExecutionWidth,
                    (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);
            return 0;
        }
        id<MTLBuffer> statsbuf = lgn2_gpu_laguna_router_simd_topk_stats_buffer();
        if (!statsbuf) return 0;

        lgn2_gpu_laguna_router_fused_args args = {
            .in_dim = in_dim,
            .n_expert = n_expert,
            .n_expert_used = n_expert_used,
            .expert_weight_scale = expert_weight_scale,
            .stats_enabled = 0u,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:biasbuf offset:(NSUInteger)bias_inner atIndex:3];
        [enc setBuffer:logitsbuf offset:lgn2_gpu_tensor_offset(logits) atIndex:4];
        [enc setBuffer:selectedbuf offset:lgn2_gpu_tensor_offset(selected) atIndex:5];
        [enc setBuffer:weightsbuf offset:lgn2_gpu_tensor_offset(weights) atIndex:6];
        [enc setBuffer:probsbuf offset:lgn2_gpu_tensor_offset(probs) atIndex:7];
        [enc setBuffer:statsbuf offset:0 atIndex:8];
        [enc setThreadgroupMemoryLength:
                (8u * 256u + 256u + 512u) * sizeof(float) +
                256u * sizeof(uint32_t)
               atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

#ifdef LGN2_TEST_HOOKS
        g_laguna_router_fused_encoded_dispatches++;
        if (owned) g_laguna_router_fused_owned_dispatches++;
        else g_laguna_router_fused_batch_dispatches++;
#endif
        if (!lgn2_gpu_finish_command_buffer(cb, owned, "fused Laguna router decode")) return 0;
    }

    return 1;
}

int lgn2_gpu_glm_router_select_batch_tensor(
        lgn2_gpu_tensor       *selected,
        lgn2_gpu_tensor       *weights,
        lgn2_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const lgn2_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_tokens == 0 ||
        n_expert == 0 || n_expert > 256u ||
        n_expert_used == 0 || n_expert_used > n_expert) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = lgn2_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = lgn2_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = lgn2_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = lgn2_gpu_tensor_buffer(probs);
        const uint64_t logits_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)n_tokens * n_expert_used * sizeof(int32_t);
        const uint64_t weights_bytes = (uint64_t)n_tokens * n_expert_used * sizeof(float);
        const uint64_t probs_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            lgn2_gpu_tensor_bytes(logits) < logits_bytes ||
            lgn2_gpu_tensor_bytes(selected) < selected_bytes ||
            lgn2_gpu_tensor_bytes(weights) < weights_bytes ||
            lgn2_gpu_tensor_bytes(probs) < probs_bytes) {
            fprintf(stderr, "lgn2: Metal Laguna batch router received undersized buffers\n");
            return 0;
        }

        const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
        if (bias_offset > model_size || bias_bytes > model_size - bias_offset) {
            fprintf(stderr, "lgn2: Metal Laguna batch router bias range is outside the mapped model\n");
            return 0;
        }
        uint64_t bias_inner = 0;
        id<MTLBuffer> biasbuf = lgn2_gpu_wrap_model_range(model_map, model_size,
                                                         bias_offset, bias_bytes,
                                                         &bias_inner);
        if (!biasbuf) return 0;

        const int simd_topk_mode =
            lgn2_gpu_laguna_router_simd_topk_validate(
                n_expert, n_expert_used, expert_weight_scale);
        if (simd_topk_mode < 0) return 0;
        const bool use_simd_topk = simd_topk_mode > 0;
        id<MTLComputePipelineState> pipeline =
            lgn2_gpu_hot_pipeline(
                use_simd_topk ? g_glm_router_select_one_simd_pipeline
                              : g_glm_router_select_one_pipeline,
                use_simd_topk ? "kernel_glm_router_select_one_simd"
                              : "kernel_glm_router_select_one");
        if (!pipeline) return 0;
        if (use_simd_topk) {
            lgn2_gpu_laguna_router_simd_topk_report_once();
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        lgn2_gpu_glm_router_select_one_args args = {
            .n_expert = n_expert,
            .n_expert_used = n_expert_used,
            .expert_weight_scale = expert_weight_scale,
            .stats_enabled = use_simd_topk &&
                lgn2_gpu_laguna_router_simd_topk_trace_enabled() ? 1u : 0u,
        };

        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:logitsbuf offset:lgn2_gpu_tensor_offset(logits) atIndex:1];
        [enc setBuffer:biasbuf offset:(NSUInteger)bias_inner atIndex:2];
        [enc setBuffer:selectedbuf offset:lgn2_gpu_tensor_offset(selected) atIndex:3];
        [enc setBuffer:weightsbuf offset:lgn2_gpu_tensor_offset(weights) atIndex:4];
        [enc setBuffer:probsbuf offset:lgn2_gpu_tensor_offset(probs) atIndex:5];
        if (use_simd_topk) {
            [enc setBuffer:lgn2_gpu_laguna_router_simd_topk_stats_buffer()
                   offset:0 atIndex:6];
        }
        [enc setThreadgroupMemoryLength:use_simd_topk
                ? 512u * sizeof(float) + 256u * sizeof(uint32_t)
                : 256u * sizeof(float) + 256u * sizeof(int32_t)
                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (use_simd_topk) {
            g_glm_router_simd_topk_encoded_rows += n_tokens;
            g_glm_router_simd_topk_encoded_dispatches++;
        }

        if (!lgn2_gpu_finish_command_buffer(cb, owned, "Laguna batch router select")) return 0;
    }

    return 1;
}

void lgn2_gpu_laguna_router_simd_topk_stats_reset(void) {
    g_glm_router_simd_topk_encoded_rows = 0;
    g_glm_router_simd_topk_encoded_dispatches = 0;
    id<MTLBuffer> stats = lgn2_gpu_laguna_router_simd_topk_stats_buffer();
    if (!stats) return;
    uint32_t *values = (uint32_t *)[stats contents];
    values[0] = 0u;
    values[1] = 0u;
}

int lgn2_gpu_laguna_router_simd_topk_stats(
        uint32_t *optimized_rows,
        uint32_t *fallback_rows,
        uint64_t *encoded_rows,
        uint64_t *encoded_dispatches) {
    if (!optimized_rows || !fallback_rows ||
        !encoded_rows || !encoded_dispatches) return 0;
    id<MTLBuffer> stats = lgn2_gpu_laguna_router_simd_topk_stats_buffer();
    if (!stats) return 0;
    const uint32_t *values = (const uint32_t *)[stats contents];
    *optimized_rows = values[0];
    *fallback_rows = values[1];
    *encoded_rows = g_glm_router_simd_topk_encoded_rows;
    *encoded_dispatches = g_glm_router_simd_topk_encoded_dispatches;
    return 1;
}

#ifdef LGN2_TEST_HOOKS
void lgn2_gpu_laguna_router_decode_fused_stats_reset(void) {
    g_laguna_router_fused_encoded_dispatches = 0;
    g_laguna_router_fused_batch_dispatches = 0;
    g_laguna_router_fused_owned_dispatches = 0;
    g_laguna_router_fused_pending_dispatches = 0;
    g_laguna_router_fused_completed_dispatches = 0;
}

int lgn2_gpu_laguna_router_decode_fused_stats(
        uint64_t *encoded_dispatches,
        uint64_t *completed_dispatches) {
    if (!encoded_dispatches || !completed_dispatches) return 0;
    *encoded_dispatches = g_laguna_router_fused_encoded_dispatches;
    *completed_dispatches = g_laguna_router_fused_completed_dispatches;
    return 1;
}
#endif

int lgn2_gpu_laguna_router_simd_topk_stats_after_wait(
        uint32_t *optimized_rows,
        uint32_t *fallback_rows,
        uint64_t *encoded_rows,
        uint64_t *encoded_dispatches) {
    if (!lgn2_gpu_synchronize()) return 0;
    return lgn2_gpu_laguna_router_simd_topk_stats(
        optimized_rows, fallback_rows, encoded_rows, encoded_dispatches);
}

static bool lgn2_gpu_glm_gate_pair_type_supported(
        uint32_t gate_type,
        uint32_t up_type) {
    return gate_type == up_type &&
           (gate_type == LGN2_METAL_TENSOR_Q2_K ||
            gate_type == LGN2_METAL_TENSOR_Q3_K ||
            gate_type == LGN2_METAL_TENSOR_Q4_K ||
            gate_type == LGN2_METAL_TENSOR_Q5_K);
}

static bool lgn2_gpu_glm_down_type_supported(uint32_t down_type) {
    return down_type == LGN2_METAL_TENSOR_Q2_K ||
           down_type == LGN2_METAL_TENSOR_Q3_K ||
           down_type == LGN2_METAL_TENSOR_Q4_K ||
           down_type == LGN2_METAL_TENSOR_Q5_K ||
           down_type == LGN2_METAL_TENSOR_Q6_K;
}

int lgn2_gpu_glm_routed_moe_one_tensor(
        lgn2_gpu_tensor       *out,
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const lgn2_gpu_tensor *selected,
        const lgn2_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const lgn2_gpu_tensor *x,
        bool                    force_resident) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    (void)force_resident;

    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_total_expert == 0 || n_expert == 0 || n_expert > 256u ||
        n_expert > n_total_expert ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        gate_expert_bytes == 0 || gate_row_bytes == 0 ||
        up_expert_bytes == 0 || up_row_bytes == 0 ||
        down_expert_bytes == 0 || down_row_bytes == 0 ||
        (expert_in_dim % 256u) != 0 ||
        (expert_mid_dim % 256u) != 0 ||
        !lgn2_gpu_glm_gate_pair_type_supported(gate_type, up_type) ||
        !lgn2_gpu_glm_down_type_supported(down_type)) {
        return 0;
    }

    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / gate_row_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / up_row_bytes ||
        (uint64_t)out_dim > UINT64_MAX / down_row_bytes) {
        fprintf(stderr, "lgn2: Metal Laguna routed MoE tensor byte size overflow\n");
        return 0;
    }

    const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes = (uint64_t)n_total_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        up_expert_bytes != (uint64_t)expert_mid_dim * up_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        fprintf(stderr, "lgn2: Metal Laguna routed MoE received inconsistent expert strides\n");
        return 0;
    }
    if (gate_offset > model_size || gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size || up_tensor_bytes > model_size - up_offset ||
        down_offset > model_size || down_tensor_bytes > model_size - down_offset) {
        fprintf(stderr, "lgn2: Metal Laguna routed MoE tensor range is outside the mapped model\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = lgn2_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = lgn2_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = lgn2_gpu_tensor_buffer(weights);
        const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(mid) < mid_bytes ||
            lgn2_gpu_tensor_bytes(out) < out_bytes ||
            lgn2_gpu_tensor_bytes(selected) < (uint64_t)n_expert * sizeof(int32_t) ||
            lgn2_gpu_tensor_bytes(weights) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "lgn2: Metal Laguna routed MoE received undersized activation buffers\n");
            return 0;
        }

        const BOOL gate_pair_q2 = gate_type == LGN2_METAL_TENSOR_Q2_K;
        const BOOL gate_pair_q3 = gate_type == LGN2_METAL_TENSOR_Q3_K;
        const BOOL gate_pair_q5 = gate_type == LGN2_METAL_TENSOR_Q5_K;
        const BOOL down_scalar_q2 = down_type == LGN2_METAL_TENSOR_Q2_K;
        const BOOL down_simd_q3 = down_type == LGN2_METAL_TENSOR_Q3_K;
        const BOOL down_scalar_q4 = down_type == LGN2_METAL_TENSOR_Q4_K;
        const BOOL down_simd_q4 = down_scalar_q4;
        const BOOL down_simd_q5 = down_type == LGN2_METAL_TENSOR_Q5_K;
        const BOOL down_simd_q6 = down_type == LGN2_METAL_TENSOR_Q6_K;
        const BOOL down_simd =
            down_simd_q3 || down_simd_q4 || down_simd_q5 || down_simd_q6;
        const BOOL glm_qmv_r1 =
            lgn2_gpu_env_bool("LGN2_METAL_LAGUNA_QMV_R1") > 0;

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gatebuf = nil;
        id<MTLBuffer> upbuf = nil;
        id<MTLBuffer> downbuf = nil;

        gatebuf = lgn2_gpu_wrap_model_range(model_map, model_size,
                                           gate_offset, gate_tensor_bytes,
                                           &gate_inner);
        upbuf = lgn2_gpu_wrap_model_range(model_map, model_size,
                                         up_offset, up_tensor_bytes,
                                         &up_inner);
        downbuf = lgn2_gpu_wrap_model_range(model_map, model_size,
                                           down_offset, down_tensor_bytes,
                                           &down_inner);
        if (!gatebuf || !upbuf || !downbuf) return 0;


        id<MTLComputePipelineState> pair_pipeline =
            gate_pair_q2 ?
            lgn2_gpu_hot_pipeline(
                glm_qmv_r1 ? g_glm_q2_k_pair_swiglu_r1_f32_pipeline :
                              g_glm_q2_k_pair_swiglu_f32_pipeline,
                glm_qmv_r1 ? "kernel_glm_q2_K_pair_swiglu_r1_f32" :
                              "kernel_glm_q2_K_pair_swiglu_f32") :
            gate_pair_q3 ?
            lgn2_gpu_hot_pipeline(
                glm_qmv_r1 ? g_glm_q3_k_pair_swiglu_r1_f32_pipeline :
                              g_glm_q3_k_pair_swiglu_f32_pipeline,
                glm_qmv_r1 ? "kernel_glm_q3_K_pair_swiglu_r1_f32" :
                              "kernel_glm_q3_K_pair_swiglu_f32") :
            gate_pair_q5 ?
            lgn2_gpu_hot_pipeline(g_glm_q5_k_pair_swiglu_f32_pipeline,
                                 "kernel_glm_q5_K_pair_swiglu_f32") :
            lgn2_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu2_f32_pipeline,
                                 "kernel_glm_q4_K_pair_swiglu2_f32");

        id<MTLComputePipelineState> down_pipeline =
            down_scalar_q2 ?
            lgn2_gpu_hot_pipeline(
                glm_qmv_r1 ? g_glm_q2_k_down_r1_f32_pipeline :
                              g_glm_q2_k_down_f32_pipeline,
                glm_qmv_r1 ? "kernel_glm_q2_K_down_r1_f32" :
                              "kernel_glm_q2_K_down_f32") :
            down_simd_q3 ?
            lgn2_gpu_hot_pipeline(
                glm_qmv_r1 ? g_glm_q3_k_down_r1_f32_pipeline :
                              g_glm_q3_k_down_f32_pipeline,
                glm_qmv_r1 ? "kernel_glm_q3_K_down_r1_f32" :
                              "kernel_glm_q3_K_down_f32") :
            down_scalar_q4 ?
            lgn2_gpu_hot_pipeline(
                glm_qmv_r1 ? g_glm_q4_k_down_r1_f32_pipeline :
                              g_glm_q4_k_down_f32_pipeline,
                glm_qmv_r1 ? "kernel_glm_q4_K_down_r1_simd_f32" :
                              "kernel_glm_q4_K_down_simd_f32") :
            down_simd_q5 ?
            lgn2_gpu_hot_pipeline(g_glm_q5_k_down_f32_pipeline,
                                 "kernel_glm_q5_K_down_f32") :
            lgn2_gpu_hot_pipeline(g_glm_q6_k_down_f32_pipeline,
                                 "kernel_glm_q6_K_down_f32");
        if (!pair_pipeline || !down_pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const bool glm_moe_stage_profile =
            g_batch_cb != nil &&
            lgn2_gpu_stage_profile_enabled_for_layer(
                    "LGN2_METAL_LAGUNA_MOE_ONE_STAGE_PROFILE",
                    "LGN2_METAL_LAGUNA_MOE_ONE_STAGE_PROFILE_LAYER",
                    layer_index);
        const char *glm_moe_stage_filter =
            getenv("LGN2_METAL_LAGUNA_MOE_STAGE_PROFILE_FILTER");
        const char *glm_pair_path =
            gate_pair_q2 ? (glm_qmv_r1 ? "q2_r1_simd_swiglu" :
                                        "q2_scalar_swiglu") :
            gate_pair_q3 ? (glm_qmv_r1 ? "q3_r1_simd_swiglu" :
                                        "q3_pair_simd_swiglu") :
            gate_pair_q5 ? "q5_pair_simd_swiglu" : "q4_pair2_simd_swiglu";
        const char *glm_down_path =
            down_scalar_q2 ? (glm_qmv_r1 ? "q2_r1_down_simd" :
                                         "q2_down_simd") :
            down_simd_q3 ? (glm_qmv_r1 ? "q3_r1_down_simd" :
                                         "q3_down_simd") :
            down_scalar_q4 ? (glm_qmv_r1 ? "q4_r1_down_simd" :
                                         "q4_down_simd") :
            down_simd_q5 ? "q5_down_simd" : "q6_down_simd";
        double glm_moe_stage_t0 = 0.0;
        if (glm_moe_stage_profile) {
            if (lgn2_gpu_end_commands() == 0 || lgn2_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = lgn2_gpu_command_buffer(&owned);
            if (!cb) return 0;
            glm_moe_stage_t0 = lgn2_gpu_now_ms();
        }
        int ok = 1;
#define LGN2_METAL_PROFILE_LAGUNA_MOE_ONE_STAGE(name) do { \
            if (ok && glm_moe_stage_profile) { \
                if (lgn2_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = lgn2_gpu_now_ms(); \
                    const int print_stage = \
                        !glm_moe_stage_filter || !glm_moe_stage_filter[0] || \
                        strstr(stage_name, glm_moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "lgn2: Metal Laguna routed MoE one stage layer=%u tokens=1 experts=%u " \
                                "gate=%s down=%s pair=%s down_path=%s %s=%.3f ms\n", \
                                layer_index, n_expert, \
                                lgn2_gpu_metal_tensor_type_name(gate_type), \
                                lgn2_gpu_metal_tensor_type_name(down_type), \
                                glm_pair_path, glm_down_path, \
                                stage_name, now_ms - glm_moe_stage_t0); \
                    } \
                    glm_moe_stage_t0 = now_ms; \
                    if (lgn2_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = lgn2_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        lgn2_gpu_glm_routed_moe_args args = {
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = n_total_expert,
            .n_expert_used = n_expert,
            .n_tokens = 1,
            .mid_token_stride = n_expert * expert_mid_dim,
            .down_type = down_type,
            .reserved = {0u, 0u, 0u, 0u},
            .gate_expert_bytes = gate_expert_bytes,
            .gate_row_bytes = gate_row_bytes,
            .up_expert_bytes = up_expert_bytes,
            .up_row_bytes = up_row_bytes,
            .down_expert_bytes = down_expert_bytes,
            .down_row_bytes = down_row_bytes,
        };
        const NSUInteger pair_x_groups =
            gate_pair_q2 ? (glm_qmv_r1 ?
                            (NSUInteger)((expert_mid_dim + 1u) / 2u) :
                            (NSUInteger)((expert_mid_dim + 7u) / 8u)) :
            gate_pair_q3 ? (glm_qmv_r1 ?
                            (NSUInteger)((expert_mid_dim + 1u) / 2u) :
                            (NSUInteger)((expert_mid_dim + 3u) / 4u)) :
            gate_pair_q5 ? (NSUInteger)((expert_mid_dim + 7u) / 8u) :
            (NSUInteger)((expert_mid_dim + 1u) / 2u);
        const NSUInteger pair_threadgroup_bytes = 0u;
        const NSUInteger pair_threads = 64u;
        const NSUInteger down_x_groups =
            down_scalar_q2 ? (glm_qmv_r1 ?
                              (NSUInteger)((out_dim + 1u) / 2u) :
                              (NSUInteger)((out_dim + 7u) / 8u)) :
            down_simd_q3 ? (glm_qmv_r1 ?
                            (NSUInteger)((out_dim + 1u) / 2u) :
                            (NSUInteger)((out_dim + 3u) / 4u)) :
            down_simd_q4 ? (glm_qmv_r1 ?
                            (NSUInteger)((out_dim + 1u) / 2u) :
                            (NSUInteger)((out_dim + 3u) / 4u)) :
            down_simd_q5 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q6 ? (NSUInteger)((out_dim + 3u) / 4u) :
            (NSUInteger)out_dim;
        const NSUInteger down_threadgroup_bytes =
            (down_scalar_q2 || down_simd) ? 0u : 256u * sizeof(float);
        const NSUInteger down_threads =
            (down_scalar_q2 || down_simd) ? 64u : 256u;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pair_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gatebuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:upbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:selectedbuf offset:lgn2_gpu_tensor_offset(selected) atIndex:4];
        [enc setBuffer:weightsbuf offset:lgn2_gpu_tensor_offset(weights) atIndex:5];
        [enc setBuffer:midbuf offset:lgn2_gpu_tensor_offset(mid) atIndex:6];
        if (pair_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:pair_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(pair_x_groups,
                                              (NSUInteger)n_expert,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(pair_threads, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        LGN2_METAL_PROFILE_LAGUNA_MOE_ONE_STAGE("pair");

        if (!ok) return 0;

        enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:downbuf offset:(NSUInteger)down_inner atIndex:1];
        [enc setBuffer:selectedbuf offset:lgn2_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:midbuf offset:lgn2_gpu_tensor_offset(mid) atIndex:3];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:4];
        if (down_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:down_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(down_x_groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        LGN2_METAL_PROFILE_LAGUNA_MOE_ONE_STAGE("down");

        if (!ok) return 0;
        if (!lgn2_gpu_finish_command_buffer(cb, owned, "Laguna routed MoE")) return 0;
#undef LGN2_METAL_PROFILE_LAGUNA_MOE_ONE_STAGE
    }

    return 1;
}

int lgn2_gpu_laguna_routed_shared_moe_one_tensor(
        lgn2_gpu_tensor                *routed_out,
        lgn2_gpu_tensor                *routed_mid,
        lgn2_gpu_tensor                *shared_out,
        lgn2_gpu_tensor                *shared_mid,
        const void                    *model_map,
        uint64_t                       model_size,
        const lgn2_gpu_laguna_moe_desc *routed,
        const lgn2_gpu_laguna_moe_desc *shared,
        uint32_t                       expert_in_dim,
        uint32_t                       expert_mid_dim,
        uint32_t                       out_dim,
        const lgn2_gpu_tensor          *selected,
        const lgn2_gpu_tensor          *weights,
        uint32_t                       n_total_expert,
        uint32_t                       n_expert,
        const lgn2_gpu_tensor          *shared_selected,
        const lgn2_gpu_tensor          *shared_weight,
        const lgn2_gpu_tensor          *x) {
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!routed_out || !routed_mid || !shared_out || !shared_mid ||
        !model_map || !routed || !shared || !selected || !weights ||
        !shared_selected || !shared_weight || !x || n_total_expert == 0u ||
        n_expert == 0u || n_expert > n_total_expert ||
        expert_in_dim == 0u || expert_mid_dim == 0u || out_dim == 0u ||
        routed->gate_row_bytes == 0u || routed->up_row_bytes == 0u ||
        routed->down_row_bytes == 0u || shared->gate_row_bytes == 0u ||
        shared->up_row_bytes == 0u || shared->down_row_bytes == 0u ||
        n_expert > UINT32_MAX / expert_mid_dim ||
        (expert_in_dim % 256u) != 0u || (expert_mid_dim % 256u) != 0u ||
        routed->gate_type != LGN2_METAL_TENSOR_Q4_K ||
        routed->up_type != LGN2_METAL_TENSOR_Q4_K ||
        shared->gate_type != LGN2_METAL_TENSOR_Q4_K ||
        shared->up_type != LGN2_METAL_TENSOR_Q4_K ||
        routed->down_type != shared->down_type ||
        (routed->down_type != LGN2_METAL_TENSOR_Q4_K &&
         routed->down_type != LGN2_METAL_TENSOR_Q6_K)) {
        return 0;
    }

    if (routed->gate_row_bytes > UINT64_MAX / expert_mid_dim ||
        routed->up_row_bytes > UINT64_MAX / expert_mid_dim ||
        routed->down_row_bytes > UINT64_MAX / out_dim ||
        shared->gate_row_bytes > UINT64_MAX / expert_mid_dim ||
        shared->up_row_bytes > UINT64_MAX / expert_mid_dim ||
        shared->down_row_bytes > UINT64_MAX / out_dim ||
        routed->gate_expert_bytes !=
            (uint64_t)expert_mid_dim * routed->gate_row_bytes ||
        routed->up_expert_bytes !=
            (uint64_t)expert_mid_dim * routed->up_row_bytes ||
        routed->down_expert_bytes !=
            (uint64_t)out_dim * routed->down_row_bytes ||
        shared->gate_expert_bytes !=
            (uint64_t)expert_mid_dim * shared->gate_row_bytes ||
        shared->up_expert_bytes !=
            (uint64_t)expert_mid_dim * shared->up_row_bytes ||
        shared->down_expert_bytes !=
            (uint64_t)out_dim * shared->down_row_bytes ||
        n_total_expert > UINT64_MAX / routed->gate_expert_bytes ||
        n_total_expert > UINT64_MAX / routed->up_expert_bytes ||
        n_total_expert > UINT64_MAX / routed->down_expert_bytes) {
        return 0;
    }

    const uint64_t routed_gate_bytes =
        (uint64_t)n_total_expert * routed->gate_expert_bytes;
    const uint64_t routed_up_bytes =
        (uint64_t)n_total_expert * routed->up_expert_bytes;
    const uint64_t routed_down_bytes =
        (uint64_t)n_total_expert * routed->down_expert_bytes;
    if (routed->gate_offset > model_size ||
        routed_gate_bytes > model_size - routed->gate_offset ||
        routed->up_offset > model_size ||
        routed_up_bytes > model_size - routed->up_offset ||
        routed->down_offset > model_size ||
        routed_down_bytes > model_size - routed->down_offset ||
        shared->gate_offset > model_size ||
        shared->gate_expert_bytes > model_size - shared->gate_offset ||
        shared->up_offset > model_size ||
        shared->up_expert_bytes > model_size - shared->up_offset ||
        shared->down_offset > model_size ||
        shared->down_expert_bytes > model_size - shared->down_offset) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> routedoutbuf = lgn2_gpu_tensor_buffer(routed_out);
        id<MTLBuffer> routedmidbuf = lgn2_gpu_tensor_buffer(routed_mid);
        id<MTLBuffer> sharedoutbuf = lgn2_gpu_tensor_buffer(shared_out);
        id<MTLBuffer> sharedmidbuf = lgn2_gpu_tensor_buffer(shared_mid);
        id<MTLBuffer> selectedbuf = lgn2_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = lgn2_gpu_tensor_buffer(weights);
        id<MTLBuffer> sharedselectedbuf = lgn2_gpu_tensor_buffer(shared_selected);
        id<MTLBuffer> sharedweightbuf = lgn2_gpu_tensor_buffer(shared_weight);
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        const uint64_t routed_mid_bytes =
            (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t shared_mid_bytes =
            (uint64_t)expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!routedoutbuf || !routedmidbuf || !sharedoutbuf || !sharedmidbuf ||
            !selectedbuf || !weightsbuf || !sharedselectedbuf ||
            !sharedweightbuf || !xbuf ||
            lgn2_gpu_tensor_bytes(routed_out) < out_bytes ||
            lgn2_gpu_tensor_bytes(shared_out) < out_bytes ||
            lgn2_gpu_tensor_bytes(routed_mid) < routed_mid_bytes ||
            lgn2_gpu_tensor_bytes(shared_mid) < shared_mid_bytes ||
            lgn2_gpu_tensor_bytes(selected) <
                (uint64_t)n_expert * sizeof(int32_t) ||
            lgn2_gpu_tensor_bytes(weights) <
                (uint64_t)n_expert * sizeof(float) ||
            lgn2_gpu_tensor_bytes(shared_selected) < sizeof(int32_t) ||
            lgn2_gpu_tensor_bytes(shared_weight) < sizeof(float) ||
            lgn2_gpu_tensor_bytes(x) <
                (uint64_t)expert_in_dim * sizeof(float)) {
            return 0;
        }

        uint64_t routed_gate_inner = 0, routed_up_inner = 0;
        uint64_t routed_down_inner = 0, shared_gate_inner = 0;
        uint64_t shared_up_inner = 0, shared_down_inner = 0;
        id<MTLBuffer> routedgatebuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, routed->gate_offset, routed_gate_bytes,
            &routed_gate_inner);
        id<MTLBuffer> routedupbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, routed->up_offset, routed_up_bytes,
            &routed_up_inner);
        id<MTLBuffer> routeddownbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, routed->down_offset, routed_down_bytes,
            &routed_down_inner);
        id<MTLBuffer> sharedgatebuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, shared->gate_offset,
            shared->gate_expert_bytes, &shared_gate_inner);
        id<MTLBuffer> sharedupbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, shared->up_offset,
            shared->up_expert_bytes, &shared_up_inner);
        id<MTLBuffer> shareddownbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, shared->down_offset,
            shared->down_expert_bytes, &shared_down_inner);
        id<MTLComputePipelineState> pair_pipeline = lgn2_gpu_hot_pipeline(
            g_laguna_routed_shared_pair_pipeline,
            "kernel_laguna_q4_K_routed_shared_pair_swiglu_f32");
        id<MTLComputePipelineState> down_pipeline =
            routed->down_type == LGN2_METAL_TENSOR_Q4_K ?
            lgn2_gpu_hot_pipeline(
                g_laguna_routed_shared_q4_down_pipeline,
                "kernel_laguna_q4_K_routed_shared_down_f32") :
            lgn2_gpu_hot_pipeline(
                g_laguna_routed_shared_q6_down_pipeline,
                "kernel_laguna_q6_K_routed_shared_down_f32");
        if (!routedgatebuf || !routedupbuf || !routeddownbuf ||
            !sharedgatebuf || !sharedupbuf || !shareddownbuf ||
            !pair_pipeline || !down_pipeline) {
            return 0;
        }

        const lgn2_gpu_glm_routed_moe_args routed_args = {
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = n_total_expert,
            .n_expert_used = n_expert,
            .n_tokens = 1,
            .mid_token_stride = n_expert * expert_mid_dim,
            .down_type = routed->down_type,
            .reserved = {0u, 0u, 0u, 0u},
            .gate_expert_bytes = routed->gate_expert_bytes,
            .gate_row_bytes = routed->gate_row_bytes,
            .up_expert_bytes = routed->up_expert_bytes,
            .up_row_bytes = routed->up_row_bytes,
            .down_expert_bytes = routed->down_expert_bytes,
            .down_row_bytes = routed->down_row_bytes,
        };
        const lgn2_gpu_glm_routed_moe_args shared_args = {
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = 1,
            .n_expert_used = 1,
            .n_tokens = 1,
            .mid_token_stride = expert_mid_dim,
            .down_type = shared->down_type,
            .reserved = {0u, 0u, 0u, 0u},
            .gate_expert_bytes = shared->gate_expert_bytes,
            .gate_row_bytes = shared->gate_row_bytes,
            .up_expert_bytes = shared->up_expert_bytes,
            .up_row_bytes = shared->up_row_bytes,
            .down_expert_bytes = shared->down_expert_bytes,
            .down_row_bytes = shared->down_row_bytes,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pair_pipeline];
        [enc setBytes:&routed_args length:sizeof(routed_args) atIndex:0];
        [enc setBytes:&shared_args length:sizeof(shared_args) atIndex:1];
        [enc setBuffer:routedgatebuf offset:(NSUInteger)routed_gate_inner atIndex:2];
        [enc setBuffer:routedupbuf offset:(NSUInteger)routed_up_inner atIndex:3];
        [enc setBuffer:sharedgatebuf offset:(NSUInteger)shared_gate_inner atIndex:4];
        [enc setBuffer:sharedupbuf offset:(NSUInteger)shared_up_inner atIndex:5];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:6];
        [enc setBuffer:selectedbuf offset:lgn2_gpu_tensor_offset(selected) atIndex:7];
        [enc setBuffer:weightsbuf offset:lgn2_gpu_tensor_offset(weights) atIndex:8];
        [enc setBuffer:sharedselectedbuf
                offset:lgn2_gpu_tensor_offset(shared_selected) atIndex:9];
        [enc setBuffer:sharedweightbuf
                offset:lgn2_gpu_tensor_offset(shared_weight) atIndex:10];
        [enc setBuffer:routedmidbuf offset:lgn2_gpu_tensor_offset(routed_mid)
                atIndex:11];
        [enc setBuffer:sharedmidbuf offset:lgn2_gpu_tensor_offset(shared_mid)
                atIndex:12];
        [enc dispatchThreadgroups:MTLSizeMake(
                ((NSUInteger)expert_mid_dim + 1u) / 2u,
                (NSUInteger)n_expert + 1u,
                1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&routed_args length:sizeof(routed_args) atIndex:0];
        [enc setBytes:&shared_args length:sizeof(shared_args) atIndex:1];
        [enc setBuffer:routeddownbuf offset:(NSUInteger)routed_down_inner atIndex:2];
        [enc setBuffer:shareddownbuf offset:(NSUInteger)shared_down_inner atIndex:3];
        [enc setBuffer:selectedbuf offset:lgn2_gpu_tensor_offset(selected) atIndex:4];
        [enc setBuffer:sharedselectedbuf
                offset:lgn2_gpu_tensor_offset(shared_selected) atIndex:5];
        [enc setBuffer:routedmidbuf offset:lgn2_gpu_tensor_offset(routed_mid)
                atIndex:6];
        [enc setBuffer:sharedmidbuf offset:lgn2_gpu_tensor_offset(shared_mid)
                atIndex:7];
        [enc setBuffer:routedoutbuf offset:lgn2_gpu_tensor_offset(routed_out)
                atIndex:8];
        [enc setBuffer:sharedoutbuf offset:lgn2_gpu_tensor_offset(shared_out)
                atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(
                ((NSUInteger)out_dim + 3u) / 4u, 2u, 1u)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);

        if (!lgn2_gpu_finish_command_buffer(
                cb, owned, "Laguna routed/shared MoE")) {
            return 0;
        }
    }
    return 1;
}

static bool lgn2_gpu_glm_grouped_moe_fast_default(void) {
    return !g_quality_mode;
}

static bool lgn2_gpu_glm_routed_moe_batch_grouped_available(
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t down_type,
        uint32_t n_expert,
        uint32_t n_tokens) {
    if (n_tokens < 32u ||
        !lgn2_gpu_glm_gate_pair_type_supported(gate_type, up_type) ||
        !lgn2_gpu_glm_down_type_supported(down_type) ||
        lgn2_gpu_mul_mm_id_map0_name(n_expert) == NULL) {
        return false;
    }
    const bool fast_default = lgn2_gpu_glm_grouped_moe_fast_default();
    if (!fast_default) {
        return false;
    }
    /* Diagnostic threshold: the grouped batch pays off well before 96 tokens
     * on some parts, so let the benchmark machine explore down to the
     * 32-token structural floor of the routed mul_mm_id kernels. */
    const uint64_t min_tokens = g_glm_grouped_moe_min_tokens;
    if (n_tokens < min_tokens) return false;

    return lgn2_gpu_get_pipeline(lgn2_gpu_mul_mm_id_map0_name(n_expert)) != nil &&
           lgn2_gpu_routed_mm_pipeline(gate_type) != nil &&
           lgn2_gpu_routed_mm_pipeline(up_type) != nil &&
           lgn2_gpu_routed_mm_f16_rhs_pipeline(down_type) != nil;
}

static bool lgn2_gpu_glm_grouped_moe_layer_enabled(uint32_t layer_index) {
    (void)layer_index;
    return true;
}

static int lgn2_gpu_glm_routed_moe_batch_grouped_tensor(
        lgn2_gpu_tensor       *out,
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const lgn2_gpu_tensor *selected,
        const lgn2_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const lgn2_gpu_tensor *x,
        uint32_t                n_tokens) {
    if (!lgn2_gpu_glm_routed_moe_batch_grouped_available(gate_type,
                                                         up_type,
                                                         down_type,
                                                         n_expert,
                                                         n_tokens)) {
        return 0;
    }
    if (n_expert > UINT32_MAX / n_tokens ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        return 0;
    }

    const uint32_t pair_rows = n_tokens * n_expert;
    if ((uint64_t)pair_rows > UINT64_MAX / expert_mid_dim ||
        (uint64_t)pair_rows > UINT64_MAX / out_dim ||
        (uint64_t)n_tokens > UINT64_MAX / expert_in_dim ||
        (uint64_t)n_tokens > UINT64_MAX / out_dim) {
        return 0;
    }

    const bool mid_f16 = true;
    const NSUInteger mm_id_threadgroup_bytes = 8192u;
    const uint64_t compact_mid_values = (uint64_t)pair_rows * expert_mid_dim;
    const uint64_t down_values = (uint64_t)pair_rows * out_dim;
    const uint64_t x_values = (uint64_t)n_tokens * expert_in_dim;
    const uint64_t out_values = (uint64_t)n_tokens * out_dim;
    if (compact_mid_values > UINT64_MAX / sizeof(float) ||
        compact_mid_values > UINT64_MAX / (mid_f16 ? sizeof(uint16_t) : sizeof(float)) ||
        down_values > UINT64_MAX / sizeof(float) ||
        x_values > UINT64_MAX / sizeof(float) ||
        out_values > UINT64_MAX / sizeof(float)) {
        return 0;
    }

    const uint64_t gate_scratch_bytes = compact_mid_values * sizeof(float);
    const uint64_t mid_bytes = compact_mid_values * (mid_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t down_scratch_bytes = down_values * sizeof(float);
    const uint64_t x_bytes = x_values * sizeof(float);
    const uint64_t out_bytes = out_values * sizeof(float);
    const uint64_t selected_values = (uint64_t)n_tokens * n_expert;
    const uint64_t selected_bytes = selected_values * sizeof(int32_t);
    const uint64_t weights_bytes = selected_values * sizeof(float);
    if (gate_scratch_bytes > UINT64_MAX - gate_scratch_bytes ||
        gate_scratch_bytes > NSUIntegerMax ||
        gate_scratch_bytes * 2ull > NSUIntegerMax ||
        down_scratch_bytes > NSUIntegerMax) {
        return 0;
    }

    const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes = (uint64_t)n_total_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = lgn2_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = lgn2_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = lgn2_gpu_tensor_buffer(weights);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(mid) < mid_bytes ||
            lgn2_gpu_tensor_bytes(out) < out_bytes ||
            lgn2_gpu_tensor_bytes(selected) < selected_bytes ||
            lgn2_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "lgn2: Metal Laguna grouped routed MoE received undersized activation buffers\n");
            return 0;
        }
        if (!lgn2_gpu_ensure_scratch_buffer(&g_moe_gate_scratch_buffer,
                                           &g_moe_gate_scratch_bytes,
                                           (NSUInteger)(gate_scratch_bytes * 2ull),
                                           "lgn2_glm_moe_gate_up_scratch")) {
            return 0;
        }
        if (n_expert > 1 &&
            !lgn2_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                           &g_moe_down_scratch_bytes,
                                           (NSUInteger)down_scratch_bytes,
                                           "lgn2_glm_moe_down_scratch")) {
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gatebuf = lgn2_gpu_wrap_model_range(model_map, model_size,
                                                         gate_offset, gate_tensor_bytes,
                                                         &gate_inner);
        id<MTLBuffer> upbuf = lgn2_gpu_wrap_model_range(model_map, model_size,
                                                       up_offset, up_tensor_bytes,
                                                       &up_inner);
        id<MTLBuffer> downbuf = lgn2_gpu_wrap_model_range(model_map, model_size,
                                                         down_offset, down_tensor_bytes,
                                                         &down_inner);
        if (!gatebuf || !upbuf || !downbuf) return 0;

        id<MTLComputePipelineState> map_pipeline =
            lgn2_gpu_get_pipeline(lgn2_gpu_mul_mm_id_map0_name(n_expert));
        id<MTLComputePipelineState> gate_pipeline = lgn2_gpu_routed_mm_pipeline(gate_type);
        id<MTLComputePipelineState> up_pipeline = lgn2_gpu_routed_mm_pipeline(up_type);
        id<MTLComputePipelineState> down_pipeline =
            lgn2_gpu_routed_mm_f16_rhs_pipeline(down_type);
        if (!map_pipeline || !gate_pipeline || !up_pipeline || !down_pipeline) {
            return 0;
        }

        lgn2_gpu_mul_mm_id_map_args map_args =
            lgn2_gpu_make_mul_mm_id_map_args(expert_in_dim, n_total_expert, 1, n_expert, n_tokens);
        lgn2_gpu_mul_mm_id_args gate_args =
            lgn2_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens);
        lgn2_gpu_mul_mm_id_args up_args =
            lgn2_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          up_row_bytes, up_expert_bytes,
                                          1, n_expert, n_tokens);
        lgn2_gpu_mul_mm_id_args down_args =
            lgn2_gpu_make_mul_mm_id_args_src1_size(expert_mid_dim, out_dim, n_total_expert,
                                                    down_row_bytes, down_expert_bytes,
                                                    n_expert, n_expert, n_tokens,
                                                    mid_f16 ? sizeof(uint16_t) : sizeof(float));
        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const bool glm_moe_stage_profile = false;
        const char *glm_moe_stage_filter = NULL;
        double glm_moe_stage_t0 = 0.0;
        if (glm_moe_stage_profile) {
            if (lgn2_gpu_end_commands() == 0 || lgn2_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = lgn2_gpu_command_buffer(&owned);
            if (!cb) return 0;
            glm_moe_stage_t0 = lgn2_gpu_now_ms();
        }

        int ok = 1;
#define LGN2_METAL_PROFILE_LAGUNA_GROUPED_MOE_STAGE(name) do { \
            if (ok && glm_moe_stage_profile) { \
                if (lgn2_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = lgn2_gpu_now_ms(); \
                    const int print_stage = \
                        !glm_moe_stage_filter || !glm_moe_stage_filter[0] || \
                        strstr(stage_name, glm_moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "lgn2: Metal Laguna grouped routed MoE stage layer=%u tokens=%u pairs=%u experts=%u " \
                                "gate=%s down=%s mid=%s %s=%.3f ms\n", \
                                layer_index, n_tokens, pair_rows, n_expert, \
                                lgn2_gpu_metal_tensor_type_name(gate_type), \
                                lgn2_gpu_metal_tensor_type_name(down_type), \
                                mid_f16 ? "f16" : "f32", \
                                stage_name, now_ms - glm_moe_stage_t0); \
                    } \
                    glm_moe_stage_t0 = now_ms; \
                    if (lgn2_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = lgn2_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        ok = lgn2_gpu_encode_mul_mm_id_map(cb,
                                           map_pipeline,
                                           &map_args,
                                           &gate_args,
                                           selectedbuf,
                                           lgn2_gpu_tensor_offset(selected));
        LGN2_METAL_PROFILE_LAGUNA_GROUPED_MOE_STAGE("map");
        if (ok) {
            ok = lgn2_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       gate_pipeline,
                                                       &gate_args,
                                                       gatebuf,
                                                       (NSUInteger)gate_inner,
                                                       xbuf,
                                                       lgn2_gpu_tensor_offset(x),
                                                       g_moe_gate_scratch_buffer,
                                                       0,
                                                       mm_id_threadgroup_bytes);
        }
        LGN2_METAL_PROFILE_LAGUNA_GROUPED_MOE_STAGE("gate");
        if (ok) {
            ok = lgn2_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       up_pipeline,
                                                       &up_args,
                                                       upbuf,
                                                       (NSUInteger)up_inner,
                                                       xbuf,
                                                       lgn2_gpu_tensor_offset(x),
                                                       g_moe_gate_scratch_buffer,
                                                       (NSUInteger)gate_scratch_bytes,
                                                       mm_id_threadgroup_bytes);
        }
        LGN2_METAL_PROFILE_LAGUNA_GROUPED_MOE_STAGE("up");
        if (ok) {
            ok = lgn2_gpu_encode_moe_swiglu_weight(cb,
                                                   g_moe_gate_scratch_buffer,
                                                   0,
                                                   g_moe_gate_scratch_buffer,
                                                   (NSUInteger)gate_scratch_bytes,
                                                   midbuf,
                                                   lgn2_gpu_tensor_offset(mid),
                                                   weightsbuf,
                                                   lgn2_gpu_tensor_offset(weights),
                                                   expert_mid_dim,
                                                   pair_rows,
                                                   0.0f,
                                                   mid_f16);
        }
        LGN2_METAL_PROFILE_LAGUNA_GROUPED_MOE_STAGE("activation_weight");

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : g_moe_down_scratch_buffer;
        NSUInteger down_dst_off = n_expert == 1 ? lgn2_gpu_tensor_offset(out) : 0;
        if (ok) {
            ok = lgn2_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       down_pipeline,
                                                       &down_args,
                                                       downbuf,
                                                       (NSUInteger)down_inner,
                                                       midbuf,
                                                       lgn2_gpu_tensor_offset(mid),
                                                       down_dst,
                                                       down_dst_off,
                                                       mm_id_threadgroup_bytes);
        }
        LGN2_METAL_PROFILE_LAGUNA_GROUPED_MOE_STAGE("down");
        if (ok && n_expert > 1) {
            ok = lgn2_gpu_encode_moe_sum_experts(cb,
                                                 down_dst,
                                                 down_dst_off,
                                                 outbuf,
                                                 lgn2_gpu_tensor_offset(out),
                                                 out_dim,
                                                 n_expert,
                                                 n_tokens);
        }
        LGN2_METAL_PROFILE_LAGUNA_GROUPED_MOE_STAGE("sum");
        if (!ok) return 0;

#ifdef LGN2_TEST_HOOKS
        g_test_glm_grouped_moe_encoded_dispatches++;
        if (owned) {
            g_test_glm_grouped_moe_owned_dispatches++;
        } else {
            g_test_glm_grouped_moe_batch_dispatches++;
        }
#endif
        if (!lgn2_gpu_finish_command_buffer(cb, owned, "Laguna grouped routed batch MoE")) {
            return 0;
        }
#undef LGN2_METAL_PROFILE_LAGUNA_GROUPED_MOE_STAGE
    }

    return 1;
}

static int lgn2_gpu_glm_routed_moe_batch_tensor_impl(
        lgn2_gpu_tensor       *out,
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const lgn2_gpu_tensor *selected,
        const lgn2_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const lgn2_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride,
        bool                    allow_grouped,
        bool                    force_scalar_q4_pair,
        bool                    exact_q4_evidence) {
#ifndef LGN2_TEST_HOOKS
    (void)exact_q4_evidence;
#endif
    if (exact_q4_evidence &&
        (gate_type != LGN2_METAL_TENSOR_Q4_K ||
         up_type != LGN2_METAL_TENSOR_Q4_K ||
         down_type != LGN2_METAL_TENSOR_Q4_K)) {
        return 0;
    }
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_tokens == 0 ||
        n_total_expert == 0 || n_expert == 0 || n_expert > 256u ||
        n_expert > n_total_expert ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        gate_expert_bytes == 0 || gate_row_bytes == 0 ||
        up_expert_bytes == 0 || up_row_bytes == 0 ||
        down_expert_bytes == 0 || down_row_bytes == 0 ||
        (expert_in_dim % 256u) != 0 ||
        (expert_mid_dim % 256u) != 0 ||
        !lgn2_gpu_glm_gate_pair_type_supported(gate_type, up_type) ||
        !lgn2_gpu_glm_down_type_supported(down_type)) {
        return 0;
    }

    const uint64_t per_token_mid = (uint64_t)n_expert * expert_mid_dim;
    if ((uint64_t)mid_token_stride < per_token_mid ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / gate_row_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / up_row_bytes ||
        (uint64_t)out_dim > UINT64_MAX / down_row_bytes ||
        (uint64_t)n_tokens > UINT64_MAX / expert_in_dim ||
        (uint64_t)n_tokens > UINT64_MAX / out_dim) {
        fprintf(stderr, "lgn2: Metal Laguna routed batch MoE tensor byte size overflow\n");
        return 0;
    }

    const uint64_t full_gate_tensor_bytes =
        (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t full_up_tensor_bytes =
        (uint64_t)n_total_expert * up_expert_bytes;
    const uint64_t full_down_tensor_bytes =
        (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        up_expert_bytes != (uint64_t)expert_mid_dim * up_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        fprintf(stderr, "lgn2: Metal Laguna routed batch MoE received inconsistent expert strides\n");
        return 0;
    }
    if (gate_offset > model_size || full_gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size || full_up_tensor_bytes > model_size - up_offset ||
        down_offset > model_size || full_down_tensor_bytes > model_size - down_offset) {
        fprintf(stderr, "lgn2: Metal Laguna routed batch MoE tensor range is outside the mapped model\n");
        return 0;
    }

    if (allow_grouped &&
        lgn2_gpu_glm_grouped_moe_layer_enabled(layer_index) &&
        lgn2_gpu_glm_routed_moe_batch_grouped_available(gate_type,
                                                       up_type,
                                                       down_type,
                                                       n_expert,
                                                       n_tokens)) {
        return lgn2_gpu_glm_routed_moe_batch_grouped_tensor(out,
                                                           mid,
                                                           model_map,
                                                           model_size,
                                                           gate_offset,
                                                           up_offset,
                                                           down_offset,
                                                           gate_type,
                                                           up_type,
                                                           down_type,
                                                           gate_expert_bytes,
                                                           gate_row_bytes,
                                                           up_expert_bytes,
                                                           up_row_bytes,
                                                           down_expert_bytes,
                                                           down_row_bytes,
                                                           expert_in_dim,
                                                           expert_mid_dim,
                                                           out_dim,
                                                           selected,
                                                           weights,
                                                           n_total_expert,
                                                           n_expert,
                                                           layer_index,
                                                           x,
                                                           n_tokens);
    }

    const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes = (uint64_t)n_total_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = lgn2_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = lgn2_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = lgn2_gpu_tensor_buffer(weights);
        const uint64_t x_values = (uint64_t)n_tokens * expert_in_dim;
        const uint64_t out_values = (uint64_t)n_tokens * out_dim;
        const uint64_t mid_values =
            (uint64_t)(n_tokens - 1u) * mid_token_stride + per_token_mid;
        const uint64_t selected_values = (uint64_t)n_tokens * n_expert;
        if (x_values > UINT64_MAX / sizeof(float) ||
            out_values > UINT64_MAX / sizeof(float) ||
            mid_values > UINT64_MAX / sizeof(float) ||
            selected_values > UINT64_MAX / sizeof(int32_t)) {
            fprintf(stderr, "lgn2: Metal Laguna routed batch MoE activation byte size overflow\n");
            return 0;
        }
        const uint64_t x_bytes = x_values * sizeof(float);
        const uint64_t mid_bytes = mid_values * sizeof(float);
        const uint64_t out_bytes = out_values * sizeof(float);
        const uint64_t selected_bytes = selected_values * sizeof(int32_t);
        const uint64_t weights_bytes = selected_values * sizeof(float);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            lgn2_gpu_tensor_bytes(x) < x_bytes ||
            lgn2_gpu_tensor_bytes(mid) < mid_bytes ||
            lgn2_gpu_tensor_bytes(out) < out_bytes ||
            lgn2_gpu_tensor_bytes(selected) < selected_bytes ||
            lgn2_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "lgn2: Metal Laguna routed batch MoE received undersized activation buffers\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gatebuf = nil;
        id<MTLBuffer> upbuf = nil;
        id<MTLBuffer> downbuf = nil;
        const BOOL gate_pair_q2 = gate_type == LGN2_METAL_TENSOR_Q2_K;
        const BOOL gate_pair_q3 = gate_type == LGN2_METAL_TENSOR_Q3_K;
        const BOOL gate_pair_q5 = gate_type == LGN2_METAL_TENSOR_Q5_K;
        const BOOL down_scalar_q2 = down_type == LGN2_METAL_TENSOR_Q2_K;
        const BOOL down_simd_q3 = down_type == LGN2_METAL_TENSOR_Q3_K;
        const BOOL down_scalar_q4 = down_type == LGN2_METAL_TENSOR_Q4_K;
        const BOOL down_simd_q4 = down_scalar_q4;
        const BOOL down_simd_q5 = down_type == LGN2_METAL_TENSOR_Q5_K;
        const BOOL down_simd_q6 = down_type == LGN2_METAL_TENSOR_Q6_K;
        const BOOL down_simd =
            down_simd_q3 || down_simd_q4 || down_simd_q5 || down_simd_q6;
        const BOOL enable_q4_pair4 = true;
        const BOOL q4_scalar_pair = false;
        const BOOL q4_pair2 =
            !gate_pair_q3 && !gate_pair_q5 && !q4_scalar_pair &&
            (force_scalar_q4_pair || !enable_q4_pair4);
        id<MTLComputePipelineState> pair_pipeline =
            gate_pair_q2 ?
             lgn2_gpu_hot_pipeline(g_glm_q2_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q2_K_pair_swiglu_f32") :
            gate_pair_q3 ?
             lgn2_gpu_hot_pipeline(g_glm_q3_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q3_K_pair_swiglu_f32") :
            gate_pair_q5 ?
             lgn2_gpu_hot_pipeline(g_glm_q5_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q5_K_pair_swiglu_f32") :
            (q4_scalar_pair ?
             lgn2_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q4_K_pair_swiglu_f32") :
             q4_pair2 ?
              lgn2_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu2_f32_pipeline,
                                   "kernel_glm_q4_K_pair_swiglu2_f32") :
              lgn2_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu4_f32_pipeline,
                                   "kernel_glm_q4_K_pair_swiglu4_f32"));
        id<MTLComputePipelineState> down_pipeline =
            down_scalar_q2 ?
            lgn2_gpu_hot_pipeline(g_glm_q2_k_down_f32_pipeline,
                                 "kernel_glm_q2_K_down_f32") :
            down_simd_q3 ?
            lgn2_gpu_hot_pipeline(g_glm_q3_k_down_f32_pipeline,
                                 "kernel_glm_q3_K_down_f32") :
            down_scalar_q4 ?
            lgn2_gpu_hot_pipeline(g_glm_q4_k_down_f32_pipeline,
                                 "kernel_glm_q4_K_down_f32") :
            down_simd_q5 ?
            lgn2_gpu_hot_pipeline(g_glm_q5_k_down_f32_pipeline,
                                 "kernel_glm_q5_K_down_f32") :
            lgn2_gpu_hot_pipeline(g_glm_q6_k_down_f32_pipeline,
                                 "kernel_glm_q6_K_down_f32");
        if (!pair_pipeline || !down_pipeline) return 0;

        gatebuf = lgn2_gpu_wrap_model_range(model_map, model_size,
                                           gate_offset, gate_tensor_bytes,
                                           &gate_inner);
        upbuf = lgn2_gpu_wrap_model_range(model_map, model_size,
                                         up_offset, up_tensor_bytes,
                                         &up_inner);
        downbuf = lgn2_gpu_wrap_model_range(model_map, model_size,
                                           down_offset, down_tensor_bytes,
                                           &down_inner);
        if (!gatebuf || !upbuf || !downbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const bool glm_moe_stage_profile = false;
        const char *glm_moe_stage_filter = NULL;
        const char *glm_pair_path =
                                    gate_pair_q2 ? "q2_scalar_swiglu" :
                                    gate_pair_q3 ? "q3_pair_simd_swiglu" :
                                    gate_pair_q5 ? "q5_pair_simd_swiglu" :
                                    (q4_scalar_pair ? "q4_scalar_swiglu" :
                                    (q4_pair2 ? "q4_pair2_simd_swiglu" :
                                                "q4_pair4_simd_swiglu"));
        const char *glm_down_path =
            down_scalar_q2 ? "q2_down_scalar" :
            down_simd_q3 ? "q3_down_simd" :
            down_scalar_q4 ? "q4_down_simd" :
            down_simd_q5 ? "q5_down_simd" : "q6_down_simd";
        double glm_moe_stage_t0 = 0.0;
        if (glm_moe_stage_profile) {
            if (lgn2_gpu_end_commands() == 0 || lgn2_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = lgn2_gpu_command_buffer(&owned);
            if (!cb) return 0;
            glm_moe_stage_t0 = lgn2_gpu_now_ms();
        }
        int ok = 1;
#define LGN2_METAL_PROFILE_LAGUNA_MOE_BATCH_STAGE(name) do { \
            if (ok && glm_moe_stage_profile) { \
                if (lgn2_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = lgn2_gpu_now_ms(); \
                    const int print_stage = \
                        !glm_moe_stage_filter || !glm_moe_stage_filter[0] || \
                        strstr(stage_name, glm_moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "lgn2: Metal Laguna routed MoE batch stage layer=%u tokens=%u experts=%u " \
                                "gate=%s down=%s pair=%s down_path=%s %s=%.3f ms\n", \
                                layer_index, n_tokens, n_expert, \
                                lgn2_gpu_metal_tensor_type_name(gate_type), \
                                lgn2_gpu_metal_tensor_type_name(down_type), \
                                glm_pair_path, glm_down_path, \
                                stage_name, now_ms - glm_moe_stage_t0); \
                    } \
                    glm_moe_stage_t0 = now_ms; \
                    if (lgn2_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = lgn2_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        lgn2_gpu_glm_routed_moe_args args = {
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = n_total_expert,
            .n_expert_used = n_expert,
            .n_tokens = n_tokens,
            .mid_token_stride = mid_token_stride,
            .down_type = down_type,
            .reserved = {0u, 0u, 0u, 0u},
            .gate_expert_bytes = gate_expert_bytes,
            .gate_row_bytes = gate_row_bytes,
            .up_expert_bytes = up_expert_bytes,
            .up_row_bytes = up_row_bytes,
            .down_expert_bytes = down_expert_bytes,
            .down_row_bytes = down_row_bytes,
        };
        const NSUInteger pair_x_groups =
            gate_pair_q2 ? (NSUInteger)((expert_mid_dim + 7u) / 8u) :
            gate_pair_q3 ? (NSUInteger)((expert_mid_dim + 3u) / 4u) :
            gate_pair_q5 ? (NSUInteger)((expert_mid_dim + 7u) / 8u) :
            q4_scalar_pair ? (NSUInteger)expert_mid_dim :
            q4_pair2 ? (NSUInteger)((expert_mid_dim + 1u) / 2u) :
            (NSUInteger)((expert_mid_dim + 7u) / 8u);
        const NSUInteger pair_threadgroup_bytes =
            q4_scalar_pair ? 512u * sizeof(float) : 0u;
        const NSUInteger pair_threads =
            q4_scalar_pair ? 256u : 64u;
        const NSUInteger down_x_groups =
            down_scalar_q2 ? (NSUInteger)((out_dim + 7u) / 8u) :
            down_simd_q3 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q4 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q5 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q6 ? (NSUInteger)((out_dim + 3u) / 4u) :
            (NSUInteger)out_dim;
        const NSUInteger down_threadgroup_bytes =
            down_simd ? 0u : 256u * sizeof(float);
        const NSUInteger down_threads =
            down_simd ? 64u : 256u;
        id<MTLComputeCommandEncoder> enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pair_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gatebuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:upbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:lgn2_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:selectedbuf offset:lgn2_gpu_tensor_offset(selected) atIndex:4];
        [enc setBuffer:weightsbuf offset:lgn2_gpu_tensor_offset(weights) atIndex:5];
        [enc setBuffer:midbuf offset:lgn2_gpu_tensor_offset(mid) atIndex:6];
        if (pair_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:pair_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(pair_x_groups,
                                              (NSUInteger)n_expert,
                                              (NSUInteger)n_tokens)
             threadsPerThreadgroup:MTLSizeMake(pair_threads, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        LGN2_METAL_PROFILE_LAGUNA_MOE_BATCH_STAGE("pair");

        enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:downbuf offset:(NSUInteger)down_inner atIndex:1];
        [enc setBuffer:selectedbuf offset:lgn2_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:midbuf offset:lgn2_gpu_tensor_offset(mid) atIndex:3];
        [enc setBuffer:outbuf offset:lgn2_gpu_tensor_offset(out) atIndex:4];
        if (down_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:down_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(down_x_groups,
                                              (NSUInteger)n_tokens,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1, 1)];
        lgn2_gpu_end_compute_encoder(cb, enc);
        LGN2_METAL_PROFILE_LAGUNA_MOE_BATCH_STAGE("down");

        if (!ok) return 0;
#ifdef LGN2_TEST_HOOKS
        if (exact_q4_evidence) {
            g_test_glm_exact_q4_encoded_dispatches++;
        }
#endif
        if (!lgn2_gpu_finish_command_buffer(cb, owned, "Laguna routed batch MoE")) return 0;
#ifdef LGN2_TEST_HOOKS
        if (exact_q4_evidence && owned) {
            g_test_glm_exact_q4_completed_dispatches++;
        }
#endif
#undef LGN2_METAL_PROFILE_LAGUNA_MOE_BATCH_STAGE
    }

    return 1;
}

int lgn2_gpu_glm_routed_moe_batch_tensor(
        lgn2_gpu_tensor       *out,
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const lgn2_gpu_tensor *selected,
        const lgn2_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const lgn2_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride,
        bool                    force_resident) {
    (void)force_resident;
    return lgn2_gpu_glm_routed_moe_batch_tensor_impl(out,
                                                    mid,
                                                    model_map,
                                                    model_size,
                                                    gate_offset,
                                                    up_offset,
                                                    down_offset,
                                                    gate_type,
                                                    up_type,
                                                    down_type,
                                                    gate_expert_bytes,
                                                    gate_row_bytes,
                                                    up_expert_bytes,
                                                    up_row_bytes,
                                                    down_expert_bytes,
                                                    down_row_bytes,
                                                    expert_in_dim,
                                                    expert_mid_dim,
                                                    out_dim,
                                                    selected,
                                                    weights,
                                                    n_total_expert,
                                                    n_expert,
                                                    layer_index,
                                                    x,
                                                    n_tokens,
                                                    mid_token_stride,
                                                    true,
                                                    false,
                                                    false);
}

int lgn2_gpu_glm_routed_moe_batch_decode_exact_q2_q3_tensor(
        lgn2_gpu_tensor       *out,
        lgn2_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              up_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              up_expert_bytes,
        uint64_t              up_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        const lgn2_gpu_tensor *selected,
        const lgn2_gpu_tensor *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              layer_index,
        const lgn2_gpu_tensor *x,
        uint32_t              n_tokens,
        uint32_t              mid_token_stride) {
    (void)layer_index;
    if (!g_initialized && !lgn2_gpu_init()) return 0;
    const bool q2 = gate_type == LGN2_METAL_TENSOR_Q2_K;
    const bool q3 = gate_type == LGN2_METAL_TENSOR_Q3_K;
    if (!out || !mid || !model_map || !selected || !weights || !x ||
        (!q2 && !q3) || up_type != gate_type || down_type != gate_type ||
        n_tokens == 0u || n_tokens > 16u ||
        n_total_expert == 0u || n_expert == 0u ||
        n_expert > 256u ||
        n_expert > n_total_expert ||
        expert_in_dim == 0u || expert_mid_dim == 0u || out_dim == 0u ||
        gate_expert_bytes == 0u || gate_row_bytes == 0u ||
        up_expert_bytes == 0u || up_row_bytes == 0u ||
        down_expert_bytes == 0u || down_row_bytes == 0u ||
        (expert_in_dim % 256u) != 0u ||
        (expert_mid_dim % 256u) != 0u) {
        return 0;
    }

    const uint64_t per_token_mid =
        (uint64_t)n_expert * expert_mid_dim;
    if ((uint64_t)mid_token_stride < per_token_mid ||
        per_token_mid > UINT32_MAX ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        gate_expert_bytes !=
            (uint64_t)expert_mid_dim * gate_row_bytes ||
        up_expert_bytes !=
            (uint64_t)expert_mid_dim * up_row_bytes ||
        down_expert_bytes !=
            (uint64_t)out_dim * down_row_bytes) {
        return 0;
    }

    const uint64_t gate_tensor_bytes =
        (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes =
        (uint64_t)n_total_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes =
        (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t x_values = (uint64_t)n_tokens * expert_in_dim;
    const uint64_t out_values = (uint64_t)n_tokens * out_dim;
    const uint64_t mid_values =
        (uint64_t)(n_tokens - 1u) * mid_token_stride + per_token_mid;
    const uint64_t selected_values =
        (uint64_t)n_tokens * n_expert;
    if (gate_offset > model_size ||
        gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size ||
        up_tensor_bytes > model_size - up_offset ||
        down_offset > model_size ||
        down_tensor_bytes > model_size - down_offset ||
        x_values > UINT64_MAX / sizeof(float) ||
        out_values > UINT64_MAX / sizeof(float) ||
        mid_values > UINT64_MAX / sizeof(float) ||
        selected_values > UINT64_MAX / sizeof(int32_t)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = lgn2_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = lgn2_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = lgn2_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf =
            lgn2_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf =
            lgn2_gpu_tensor_buffer(weights);
        if (!xbuf || !midbuf || !outbuf ||
            !selectedbuf || !weightsbuf ||
            lgn2_gpu_tensor_bytes(x) <
                x_values * sizeof(float) ||
            lgn2_gpu_tensor_bytes(mid) <
                mid_values * sizeof(float) ||
            lgn2_gpu_tensor_bytes(out) <
                out_values * sizeof(float) ||
            lgn2_gpu_tensor_bytes(selected) <
                selected_values * sizeof(int32_t) ||
            lgn2_gpu_tensor_bytes(weights) <
                selected_values * sizeof(float)) {
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gatebuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, gate_offset,
            gate_tensor_bytes, &gate_inner);
        id<MTLBuffer> upbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, up_offset,
            up_tensor_bytes, &up_inner);
        id<MTLBuffer> downbuf = lgn2_gpu_wrap_model_range(
            model_map, model_size, down_offset,
            down_tensor_bytes, &down_inner);
        id<MTLComputePipelineState> pair_pipeline =
            q2 ? lgn2_gpu_hot_pipeline(
                     g_glm_q2_k_pair_swiglu_f32_pipeline,
                     "kernel_glm_q2_K_pair_swiglu_f32") :
                 lgn2_gpu_hot_pipeline(
                     g_glm_q3_k_pair_swiglu_f32_pipeline,
                     "kernel_glm_q3_K_pair_swiglu_f32");
        id<MTLComputePipelineState> down_pipeline =
            q2 ? lgn2_gpu_hot_pipeline(
                     g_glm_q2_k_down_f32_pipeline,
                     "kernel_glm_q2_K_down_f32") :
                 lgn2_gpu_hot_pipeline(
                     g_glm_q3_k_down_f32_pipeline,
                     "kernel_glm_q3_K_down_f32");
        if (!gatebuf || !upbuf || !downbuf ||
            !pair_pipeline || !down_pipeline) {
            return 0;
        }

        /* The row-loop form remains available as a diagnostic A/B oracle.
         * The default path uses the token dimension already implemented by
         * the Q2_K/Q3_K kernels, so verifier rows share each pair/down
         * encoder without changing any per-token reduction. */
        const bool legacy_rows =
            getenv("LGN2_METAL_DISABLE_Q23_EXACT_MULTIROW") != NULL;
        lgn2_gpu_glm_routed_moe_args args = {
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = n_total_expert,
            .n_expert_used = n_expert,
            .n_tokens = legacy_rows ? 1u : n_tokens,
            .mid_token_stride = legacy_rows ?
                (uint32_t)per_token_mid : mid_token_stride,
            .down_type = down_type,
            .reserved = {0u, 0u, 0u, 0u},
            .gate_expert_bytes = gate_expert_bytes,
            .gate_row_bytes = gate_row_bytes,
            .up_expert_bytes = up_expert_bytes,
            .up_row_bytes = up_row_bytes,
            .down_expert_bytes = down_expert_bytes,
            .down_row_bytes = down_row_bytes,
        };
        const NSUInteger pair_groups = q2 ?
            ((NSUInteger)expert_mid_dim + 7u) / 8u :
            ((NSUInteger)expert_mid_dim + 3u) / 4u;
        const NSUInteger down_groups = q2 ?
            ((NSUInteger)out_dim + 7u) / 8u :
            ((NSUInteger)out_dim + 3u) / 4u;
        const uint64_t x_row_bytes =
            (uint64_t)expert_in_dim * sizeof(float);
        const uint64_t mid_row_bytes =
            (uint64_t)mid_token_stride * sizeof(float);
        const uint64_t out_row_bytes =
            (uint64_t)out_dim * sizeof(float);
        const uint64_t selected_row_bytes =
            (uint64_t)n_expert * sizeof(int32_t);
        const uint64_t weights_row_bytes =
            (uint64_t)n_expert * sizeof(float);

        int owned = 0;
        id<MTLCommandBuffer> cb =
            lgn2_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc =
            lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pair_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gatebuf
                offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:upbuf
                offset:(NSUInteger)up_inner atIndex:2];
        if (legacy_rows) {
            for (uint32_t row = 0; row < n_tokens; row++) {
                [enc setBuffer:xbuf
                        offset:lgn2_gpu_tensor_offset(x) +
                               (NSUInteger)(row * x_row_bytes)
                       atIndex:3];
                [enc setBuffer:selectedbuf
                        offset:lgn2_gpu_tensor_offset(selected) +
                               (NSUInteger)(row * selected_row_bytes)
                       atIndex:4];
                [enc setBuffer:weightsbuf
                        offset:lgn2_gpu_tensor_offset(weights) +
                               (NSUInteger)(row * weights_row_bytes)
                       atIndex:5];
                [enc setBuffer:midbuf
                        offset:lgn2_gpu_tensor_offset(mid) +
                               (NSUInteger)(row * mid_row_bytes)
                       atIndex:6];
                [enc dispatchThreadgroups:
                        MTLSizeMake(pair_groups,
                                    (NSUInteger)n_expert, 1u)
                     threadsPerThreadgroup:
                        MTLSizeMake(64u, 1u, 1u)];
            }
        } else {
            [enc setBuffer:xbuf
                    offset:lgn2_gpu_tensor_offset(x) atIndex:3];
            [enc setBuffer:selectedbuf
                    offset:lgn2_gpu_tensor_offset(selected) atIndex:4];
            [enc setBuffer:weightsbuf
                    offset:lgn2_gpu_tensor_offset(weights) atIndex:5];
            [enc setBuffer:midbuf
                    offset:lgn2_gpu_tensor_offset(mid) atIndex:6];
            [enc dispatchThreadgroups:
                    MTLSizeMake(pair_groups,
                                (NSUInteger)n_expert,
                                (NSUInteger)n_tokens)
                 threadsPerThreadgroup:
                    MTLSizeMake(64u, 1u, 1u)];
        }
        lgn2_gpu_end_compute_encoder(cb, enc);

        enc = lgn2_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:downbuf
                offset:(NSUInteger)down_inner atIndex:1];
        if (legacy_rows) {
            for (uint32_t row = 0; row < n_tokens; row++) {
                [enc setBuffer:selectedbuf
                        offset:lgn2_gpu_tensor_offset(selected) +
                               (NSUInteger)(row * selected_row_bytes)
                       atIndex:2];
                [enc setBuffer:midbuf
                        offset:lgn2_gpu_tensor_offset(mid) +
                               (NSUInteger)(row * mid_row_bytes)
                       atIndex:3];
                [enc setBuffer:outbuf
                        offset:lgn2_gpu_tensor_offset(out) +
                               (NSUInteger)(row * out_row_bytes)
                       atIndex:4];
                [enc dispatchThreadgroups:
                        MTLSizeMake(down_groups, 1u, 1u)
                     threadsPerThreadgroup:
                        MTLSizeMake(64u, 1u, 1u)];
            }
        } else {
            [enc setBuffer:selectedbuf
                    offset:lgn2_gpu_tensor_offset(selected) atIndex:2];
            [enc setBuffer:midbuf
                    offset:lgn2_gpu_tensor_offset(mid) atIndex:3];
            [enc setBuffer:outbuf
                    offset:lgn2_gpu_tensor_offset(out) atIndex:4];
            [enc dispatchThreadgroups:
                    MTLSizeMake(down_groups, (NSUInteger)n_tokens, 1u)
                 threadsPerThreadgroup:
                    MTLSizeMake(64u, 1u, 1u)];
        }
        lgn2_gpu_end_compute_encoder(cb, enc);

        return lgn2_gpu_finish_command_buffer(
            cb, owned, "Laguna exact Q2/Q3 verifier MoE");
    }
}

int lgn2_gpu_glm_routed_moe_batch_decode_exact_q4_tensor(
        lgn2_gpu_tensor       *out,
        lgn2_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const lgn2_gpu_tensor *selected,
        const lgn2_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const lgn2_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride) {
    if (gate_type != LGN2_METAL_TENSOR_Q4_K ||
        up_type != LGN2_METAL_TENSOR_Q4_K ||
        down_type != LGN2_METAL_TENSOR_Q4_K) {
        return 0;
    }
    return lgn2_gpu_glm_routed_moe_batch_tensor_impl(out,
                                                    mid,
                                                    model_map,
                                                    model_size,
                                                    gate_offset,
                                                    up_offset,
                                                    down_offset,
                                                    gate_type,
                                                    up_type,
                                                    down_type,
                                                    gate_expert_bytes,
                                                    gate_row_bytes,
                                                    up_expert_bytes,
                                                    up_row_bytes,
                                                    down_expert_bytes,
                                                    down_row_bytes,
                                                    expert_in_dim,
                                                    expert_mid_dim,
                                                    out_dim,
                                                    selected,
                                                    weights,
                                                    n_total_expert,
                                                    n_expert,
                                                    layer_index,
                                                    x,
                                                    n_tokens,
                                                    mid_token_stride,
                                                    false,
                                                    true,
                                                    true);
}
