#ifndef LGN2_H
#define LGN2_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Public engine boundary.
 *
 * The CLI and server should treat lgn2_engine as the loaded model and
 * lgn2_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * lgn2_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    LGN2_THINK_NONE,
    LGN2_THINK_HIGH,
    LGN2_THINK_MAX,
} lgn2_think_mode;

typedef enum {
    LGN2_LOG_DEFAULT,
    LGN2_LOG_PREFILL,
    LGN2_LOG_GENERATION,
    LGN2_LOG_KVCACHE,
    LGN2_LOG_TOOL,
    LGN2_LOG_WARNING,
    LGN2_LOG_TIMING,
    LGN2_LOG_OK,
    LGN2_LOG_ERROR,
} lgn2_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} lgn2_tokens;

/* Minimal growable byte buffer for engine helpers that append into a
 * frontend-owned running buffer. */
typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} lgn2_buf;

typedef struct {
    int id;
    float logit;
    float logprob;
} lgn2_token_score;

#define LGN2_DEFAULT_TEMPERATURE 1.0f
#define LGN2_DEFAULT_TOP_P 1.0f
#define LGN2_DEFAULT_MIN_P 0.05f

typedef struct lgn2_engine lgn2_engine;
typedef struct lgn2_session lgn2_session;

typedef void (*lgn2_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*lgn2_session_cancel_fn)(void *ud);

#define LGN2_SESSION_SYNC_INTERRUPTED 2

typedef struct {
    const char *model_path;
    const char *dflash_path;
    int n_threads;
    int context_size;
    int dflash_draft_tokens;
    float dflash_p_min;
    bool warm_weights;
    bool quality;
    bool dflash_p_min_set;
    bool inspect_only;
} lgn2_engine_options;

typedef void (*lgn2_token_emit_fn)(void *ud, int token);
typedef void (*lgn2_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} lgn2_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} lgn2_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} lgn2_session_payload_file;

/* Engine/session lifetime contract: callers must externally serialize
 * lgn2_session_create(), lgn2_session_free(), and lgn2_engine_close() for the
 * same engine.  The engine must outlive every session created from it.
 * lgn2_engine_close() refuses to destroy an engine while registered sessions
 * remain; because the API is void, refusal leaves the engine alive so callers
 * can free those sessions and retry the close.  These rules are an ownership
 * contract, not internal cross-thread synchronization. */
int lgn2_engine_open(lgn2_engine **out, const lgn2_engine_options *opt);
void lgn2_engine_close(lgn2_engine *e);
void lgn2_engine_summary(lgn2_engine *e);
int lgn2_engine_vocab_size(lgn2_engine *e);
const char *lgn2_engine_model_name(lgn2_engine *e);
int lgn2_engine_layer_count(lgn2_engine *e);
/* Stable id for cache compatibility.  Laguna S2.1 deliberately keeps the
 * explicit private identity value 3; KVC/KV headers continue to serialize
 * this byte in place without changing their wire layout. */
int lgn2_engine_model_id(lgn2_engine *e);
bool lgn2_engine_is_laguna(lgn2_engine *e);
const char *lgn2_engine_default_system_prompt(lgn2_engine *e);
void lgn2_engine_sampling_defaults(lgn2_engine *e, float *temperature,
                                  int *top_k, float *top_p, float *min_p);
bool lgn2_think_mode_enabled(lgn2_think_mode mode);
const char *lgn2_think_mode_name(lgn2_think_mode mode);
const char *lgn2_think_max_prefix(void);
uint32_t lgn2_think_max_min_context(void);
lgn2_think_mode lgn2_think_mode_for_context(lgn2_think_mode mode, int ctx_size);
/* Estimate the Laguna target graph's base KV and scratch storage.  The
 * optional DFlash/speculative/session allocations are not included.  A
 * non-positive context or one beyond Laguna's trained context length returns
 * a zeroed estimate, matching lgn_graph_alloc()'s admission policy. */
lgn2_context_memory lgn2_context_memory_estimate(int ctx_size);
bool lgn2_log_is_tty(FILE *fp);
void lgn2_log(FILE *fp, lgn2_log_type type, const char *fmt, ...);
int lgn2_engine_generate_argmax(lgn2_engine *e, const lgn2_tokens *prompt,
                               int n_predict, int ctx_size,
                               lgn2_token_emit_fn emit,
                               lgn2_generation_done_fn done,
                               void *emit_ud,
                               lgn2_session_progress_fn progress,
                               void *progress_ud);
void lgn2_engine_dump_tokens(lgn2_engine *e, const lgn2_tokens *tokens);
int lgn2_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);

void lgn2_tokens_push(lgn2_tokens *tv, int token);
void lgn2_tokens_free(lgn2_tokens *tv);
void lgn2_tokens_copy(lgn2_tokens *dst, const lgn2_tokens *src);
bool lgn2_tokens_starts_with(const lgn2_tokens *tokens, const lgn2_tokens *prefix);

void lgn2_tokenize_text(lgn2_engine *e, const char *text, lgn2_tokens *out);
void lgn2_tokenize_rendered_chat(lgn2_engine *e, const char *text, lgn2_tokens *out);
void lgn2_chat_begin(lgn2_engine *e, lgn2_tokens *tokens);
void lgn2_encode_chat_prompt(
        lgn2_engine *e,
        const char *system,
        const char *prompt,
        lgn2_think_mode think_mode,
        lgn2_tokens *out);
void lgn2_chat_append_max_effort_prefix(lgn2_engine *e, lgn2_tokens *tokens);
void lgn2_chat_append_message(lgn2_engine *e, lgn2_tokens *tokens, const char *role, const char *content);
void lgn2_chat_append_assistant_prefix(lgn2_engine *e, lgn2_tokens *tokens, lgn2_think_mode think_mode);
void lgn2_chat_append_assistant_end(lgn2_engine *e, lgn2_tokens *tokens);

char *lgn2_token_text(lgn2_engine *e, int token, size_t *len);
/* Append-decoding twin of lgn2_token_text for per-token streaming loops: the
 * decoded bytes land directly in the running buffer, so no per-token heap
 * allocation or copy is needed. */
void lgn2_token_text_into(lgn2_engine *e, int token, lgn2_buf *b);
int lgn2_token_eos(lgn2_engine *e);
bool lgn2_token_is_stop(lgn2_engine *e, int token);
bool lgn2_token_is_thinking_control(lgn2_engine *e, int token);
bool lgn2_token_is_stop_for_think_mode(lgn2_engine *e,
                                      int token,
                                      lgn2_think_mode mode);
int lgn2_token_user(lgn2_engine *e);
int lgn2_token_assistant(lgn2_engine *e);

int lgn2_session_create(lgn2_session **out, lgn2_engine *e, int ctx_size);
void lgn2_session_free(lgn2_session *s);
void lgn2_session_set_progress(lgn2_session *s, lgn2_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void lgn2_session_set_display_progress(lgn2_session *s, lgn2_session_progress_fn fn, void *ud);
/* Optional cooperative cancellation.  lgn2_session_sync() checks it only at
 * safe boundaries where the live checkpoint is either unchanged or represents a
 * valid token prefix, and returns LGN2_SESSION_SYNC_INTERRUPTED when it stops. */
void lgn2_session_set_cancel(lgn2_session *s, lgn2_session_cancel_fn fn, void *ud);
/* Enable optional speculative support-model state for the next sync/decode
 * sequence. Frontends disable it for sampling modes that cannot use a greedy
 * verifier, avoiding support-model work on every ordinary decode token. */
void lgn2_session_set_speculative_enabled(lgn2_session *s, bool enabled);
void lgn2_session_report_progress(lgn2_session *s, const char *event, int current, int total);
typedef enum {
    LGN2_SESSION_REWRITE_ERROR = -1,
    LGN2_SESSION_REWRITE_OK = 0,
    /* The live Metal graph state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    LGN2_SESSION_REWRITE_REBUILD_NEEDED = 1,
} lgn2_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the Metal graph
 * state is refilled from scratch. */
int lgn2_session_sync(lgn2_session *s, const lgn2_tokens *prompt, char *err, size_t errlen);
bool lgn2_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
lgn2_session_rewrite_result lgn2_session_rewrite_from_common(
        lgn2_session *s, const lgn2_tokens *prompt, int common,
        char *err, size_t errlen);
int lgn2_session_common_prefix(lgn2_session *s, const lgn2_tokens *prompt);
int lgn2_session_argmax(lgn2_session *s);
int lgn2_session_argmax_excluding(lgn2_session *s, int excluded_id);
int lgn2_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int lgn2_session_sample(lgn2_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
#ifdef LGN2_TEST_HOOKS
struct lgn2_model;
int lgn2_test_sample_logits(const float *logits, uint32_t n_vocab,
                           float temperature, int top_k,
                           float top_p, float min_p, uint64_t *rng,
                           float *prob_scratch);
int lgn2_test_argmax_excluding_logits(const float *logits, uint32_t n_vocab,
                                     int excluded_id);
uint64_t lgn2_test_mixed_native_count(void);
typedef struct {
    uint64_t max_scans;
    uint64_t sum_scans;
    uint64_t cache_hits;
} lgn2_test_logprob_stats;
void lgn2_test_logprob_stats_reset(void);
void lgn2_test_logprob_stats_get(lgn2_test_logprob_stats *out);
int lgn2_test_logprob_cache_probe(void);
int lgn2_test_sample_arena_lifecycle(void);
bool lgn2_test_engine_session_lifecycle(void);
bool lgn2_test_engine_close_order(void);
bool lgn2_test_laguna_context_memory_estimator(void);
bool lgn2_test_model_summary(const struct lgn2_model *model, FILE *out);
/* Model-free tokenizer/chat contract coverage using a synthetic Laguna BPE
 * vocabulary.  This keeps chat behavior tests independent of a multi-GiB
 * production GGUF while still exercising the public token APIs. */
bool lgn2_test_laguna_chat(void);
#if defined(__APPLE__) && !defined(LGN2_NO_GPU)
bool lgn2_test_engine_close_drain_failure(void);
#endif
#if defined(__APPLE__) && !defined(LGN2_NO_GPU)
bool lgn2_test_laguna_graph_env_flag(void);
#endif
#ifndef LGN2_NO_GPU
/* Model-independent session-route contract: Laguna argmax uses the Laguna
 * evaluator and mixed raw-graph prefill is rejected before graph access. */
bool lgn2_test_laguna_session_routes(void);
#endif
#endif
int lgn2_session_top_logprobs(lgn2_session *s, lgn2_token_score *out, int k);
int lgn2_session_token_logprob(lgn2_session *s, int token, lgn2_token_score *out);
int lgn2_session_copy_logits(lgn2_session *s, float *out, int cap);
int lgn2_session_set_logits(lgn2_session *s, const float *logits, int n);
/* Pay the one-time first-submission GPU cost outside any measured window. */
void lgn2_session_gpu_warmup(lgn2_session *s);
int lgn2_session_eval(lgn2_session *s, int token, char *err, size_t errlen);

typedef struct {
    lgn2_session *session;
    int token;
} lgn2_decode_item;

/* Advance independent sessions by one token each. Batch size one is exactly
 * lgn2_session_eval(); retained host scheduling remains serialized when needed. */
int lgn2_sessions_eval_batch(lgn2_decode_item *items, int count,
                            char *err, size_t errlen);
/* Advance one resumed prefill suffix and an independent decode batch as one
 * scheduling step. Unsupported combinations use the ordinary serialized
 * session operations. */
int lgn2_sessions_eval_batch_with_prefill(
        lgn2_decode_item *items, int count,
        lgn2_session *prefill_session, const lgn2_tokens *prefill_prompt,
        char *err, size_t errlen);
int lgn2_session_eval_speculative_argmax(lgn2_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void lgn2_session_invalidate(lgn2_session *s);
void lgn2_session_rewind(lgn2_session *s, int pos);
int lgn2_session_pos(lgn2_session *s);
int lgn2_session_ctx(lgn2_session *s);
int lgn2_session_prefill_cap(lgn2_session *s);
int lgn2_engine_routed_quant_bits(lgn2_engine *e);
bool lgn2_engine_has_output_head(lgn2_engine *e);
bool lgn2_engine_has_dflash(lgn2_engine *e);
int lgn2_engine_dflash_draft_tokens(lgn2_engine *e);
const lgn2_tokens *lgn2_session_tokens(lgn2_session *s);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the LGN2-specific serialized graph state. */
#define LGN2_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define LGN2_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define LGN2_SESSION_PAYLOAD_U32_FIELDS 13u
#define LGN2_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define LGN2_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
#define LGN2_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t lgn2_session_payload_bytes(lgn2_session *s);
int lgn2_session_stage_payload(lgn2_session *s, lgn2_session_payload_file *out,
                              char *err, size_t errlen);
int lgn2_session_write_staged_payload(const lgn2_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void lgn2_session_payload_file_free(lgn2_session_payload_file *payload);
int lgn2_session_save_payload(lgn2_session *s, FILE *fp, char *err, size_t errlen);
int lgn2_session_load_payload(lgn2_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int lgn2_session_save_snapshot(lgn2_session *s, lgn2_session_snapshot *snap, char *err, size_t errlen);
int lgn2_session_load_snapshot(lgn2_session *s, const lgn2_session_snapshot *snap, char *err, size_t errlen);
void lgn2_session_snapshot_free(lgn2_session_snapshot *snap);

uint64_t lgn2_session_layer_payload_bytes(lgn2_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int lgn2_session_save_layer_payload(lgn2_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int lgn2_session_load_layer_payload(lgn2_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

#endif
