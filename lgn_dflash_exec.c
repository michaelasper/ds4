/*
 * lgn_dflash_exec.c - private DFlash support execution recording.
 *
 * This module records support-model work into a transaction owned by lgn2_engine.c.
 * It does not begin, end, submit, discard, or wait command batches and never
 * stores engine maps or execution context.
 */

#include "lgn_dflash_exec.h"

#ifndef LGN2_NO_GPU

#include "lgn_model.h"

bool lgn_dflash_exec_context_valid(
        const lgn_dflash_exec_context *ctx) {
    if (!ctx || !ctx->support_map || ctx->support_map_size == 0u ||
        !ctx->support_weights ||
        ctx->target_context_length != LGN_DFLASH_TARGET_CONTEXT_LENGTH) {
        return false;
    }
    if ((ctx->f16_map == NULL) != (ctx->f16_map_size == 0u)) {
        return false;
    }
    return true;
}

static const void *lgn_dflash_exec_weight_map(
        const lgn_dflash_exec_context *ctx) {
    return ctx->f16_map ? ctx->f16_map : ctx->support_map;
}

static uint64_t lgn_dflash_exec_weight_map_size(
        const lgn_dflash_exec_context *ctx) {
    return ctx->f16_map ? ctx->f16_map_size : ctx->support_map_size;
}

bool lgn_dflash_exec_matmul(
        lgn2_gpu_tensor                *out,
        const lgn_dflash_exec_context *ctx,
        const lgn2_tensor              *weight,
        const lgn2_gpu_tensor          *x,
        uint32_t                       n_rows) {
    if (!out || !lgn_dflash_exec_context_valid(ctx) || !weight ||
        !x || weight->ndim < 2) {
        return false;
    }
    if (weight->type == LGN_TENSOR_BF16) {
        if (!ctx->f16_map || ctx->f16_map_size == 0u) return false;
        return lgn2_gpu_matmul_f16_tensor(
                   out,
                   ctx->f16_map,
                   ctx->f16_map_size,
                   weight->abs_offset,
                   weight->dim[0],
                   weight->dim[1],
                   x,
                   n_rows) != 0;
    }
    if (weight->type == LGN_TENSOR_Q8_0) {
#ifdef __APPLE__
        return lgn2_gpu_matmul_q8_0_dflash_tensor(
                   out,
                   ctx->support_map,
                   ctx->support_map_size,
                   weight->abs_offset,
                   weight->dim[0],
                   weight->dim[1],
                   x,
                   n_rows) != 0;
#else
        return lgn2_gpu_matmul_q8_0_tensor(
                   out,
                   ctx->support_map,
                   ctx->support_map_size,
                   weight->abs_offset,
                   weight->dim[0],
                   weight->dim[1],
                   x,
                   n_rows) != 0;
#endif
    }
    if (lgn_model_tensor_type_is_dense_quant(weight->type)) {
        return lgn2_gpu_matmul_quant_tensor(
                   out,
                   ctx->support_map,
                   ctx->support_map_size,
                   weight->abs_offset,
                   weight->type,
                   weight->dim[0],
                   weight->dim[1],
                   x,
                   n_rows) != 0;
    }
    return false;
}

bool lgn_dflash_exec_encode_record(
        lgn_dflash_graph              *g,
        const lgn_dflash_exec_context *ctx,
        uint32_t                       pos0,
        uint32_t                       n_rows) {
    const lgn_dflash_profile *profile = lgn_dflash_profile_get();
    if (!g || !lgn_dflash_exec_context_valid(ctx) || !profile ||
        !ctx->support_weights || n_rows == 0u ||
        n_rows > g->feature_cap || !lgn2_gpu_commands_active()) {
        return false;
    }

    const lgn_dflash_weights *w = ctx->support_weights;
    const uint32_t embd = profile->n_embd;
    const uint32_t n_head_kv = profile->n_head_kv;
    const uint32_t head_dim = profile->n_head_dim;
    const void *weight_map = lgn_dflash_exec_weight_map(ctx);
    const uint64_t weight_map_size = lgn_dflash_exec_weight_map_size(ctx);

    bool ok = lgn2_gpu_dflash_aux_norm_tensor(
                  g->features,
                  weight_map,
                  weight_map_size,
                  w->aux_norm->abs_offset,
                  n_rows,
                  embd,
                  profile->n_aux,
                  profile->rms_eps) != 0;
    if (ok) {
        ok = lgn_dflash_exec_matmul(g->encoder, ctx, w->fc,
                                    g->features, n_rows);
    }
    if (ok) {
        ok = lgn2_gpu_rms_norm_weight_rows_tensor(
                  g->encoder_norm,
                  g->encoder,
                  weight_map,
                  weight_map_size,
                  w->encoder_output_norm->abs_offset,
                  embd,
                  n_rows,
                  profile->rms_eps) != 0;
    }
    for (uint32_t il = 0; ok && il < profile->n_layer; il++) {
        const lgn_dflash_layer_weights *l = &w->layer[il];
        ok = lgn2_gpu_rms_norm_weight_rows_tensor(
                  g->norm,
                  g->encoder_norm,
                  weight_map,
                  weight_map_size,
                  l->attn_norm->abs_offset,
                  embd,
                  n_rows,
                  profile->rms_eps) != 0;
        if (ok) {
            ok = lgn_dflash_exec_matmul(g->k, ctx, l->attn_k,
                                        g->norm, n_rows) &&
                 lgn_dflash_exec_matmul(g->v, ctx, l->attn_v,
                                        g->norm, n_rows);
        }
        if (ok) {
#ifdef __APPLE__
            ok = lgn2_gpu_laguna_head_rms_norm_rope_support_tensor(
#else
            ok = lgn2_gpu_laguna_head_rms_norm_rope_tensor(
#endif
                      g->k,
                      weight_map,
                      weight_map_size,
                      l->attn_k_norm->abs_offset,
                      n_rows,
                      n_head_kv,
                      head_dim,
                      profile->n_rot,
                      pos0,
                      ctx->target_context_length,
                      profile->rope_freq_base,
                      1.0f,
                      0.0f,
                      1.0f,
                      0.0f,
                      0.0f,
                      profile->rms_eps) != 0;
        }
        if (ok) {
            ok = lgn2_gpu_dflash_commit_kv_tensor(
                      g->key_cache[il],
                      g->value_cache[il],
                      g->k,
                      g->v,
                      pos0,
                      n_rows,
                      g->cache_cap,
                      n_head_kv,
                      head_dim) != 0;
        }
    }
    return ok;
}

#endif /* !LGN2_NO_GPU */
