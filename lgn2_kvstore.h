#ifndef LGN2_KVSTORE_H
#define LGN2_KVSTORE_H

#include "lgn2.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define LGN2_KVSTORE_FIXED_HEADER 48u
#define LGN2_KVSTORE_DEFAULT_MB 4096
#define LGN2_KVSTORE_HIT_HALF_LIFE_SECONDS (6ull * 60ull * 60ull)

#define LGN2_KVSTORE_EXT_TOOL_MAP          (1u << 0)
#define LGN2_KVSTORE_EXT_RESPONSES_VISIBLE (1u << 1)
#define LGN2_KVSTORE_EXT_THINKING_VISIBLE  (1u << 2)
#define LGN2_KVSTORE_EXT_SESSION_TITLE     (1u << 3)

typedef enum {
    LGN2_KVSTORE_REASON_UNKNOWN   = 0,
    LGN2_KVSTORE_REASON_COLD      = 1,
    LGN2_KVSTORE_REASON_CONTINUED = 2,
    LGN2_KVSTORE_REASON_EVICT     = 3,
    LGN2_KVSTORE_REASON_SHUTDOWN  = 4,
    LGN2_KVSTORE_REASON_AGENT_SYSTEM  = 5,
    LGN2_KVSTORE_REASON_AGENT_SESSION = 6,
} lgn2_kvstore_reason;

typedef enum {
    LGN2_KVSTORE_LOG_DEFAULT,
    LGN2_KVSTORE_LOG_KVCACHE,
    LGN2_KVSTORE_LOG_WARNING,
} lgn2_kvstore_log_type;

typedef struct {
    /* The file name is the rendered byte prefix, not the token sequence. The
     * payload still carries the exact tokens and graph state; the hash only
     * answers "does this checkpoint represent the bytes at the front of the
     * incoming prompt?" */
    char sha[41];
    char *path;
    uint8_t quant_bits;
    /* Stored in header byte 7.  Flash is 0 for backward compatibility with
     * older cache files where this reserved byte was always written as zero. */
    uint8_t model_id;
    uint8_t reason;
    uint32_t tokens;
    uint32_t hits;
    uint32_t ctx_size;
    uint8_t ext_flags;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t payload_bytes;
    uint64_t text_bytes;
    uint64_t file_size;
} lgn2_kvstore_entry;

typedef struct {
    int min_tokens;
    int cold_max_tokens;
    int continued_interval_tokens;
    int boundary_trim_tokens;
    int boundary_align_tokens;
} lgn2_kvstore_options;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    bool reject_different_quant;
    lgn2_kvstore_options opt;
    int continued_last_store_tokens;
    lgn2_kvstore_entry *entry;
    int len;
    int cap;
    const char *log_name;
    void *log_ud;
    void (*log)(void *ud, lgn2_kvstore_log_type type, const char *msg);
} lgn2_kvstore;

typedef struct {
    const char *text;
    size_t text_len;
    uint8_t model_id;
    uint8_t quant_bits;
    uint32_t ctx_size;
    bool reject_different_quant;
} lgn2_kvstore_eviction_context;

typedef struct {
    void *ud;
    uint8_t ext_flag;
    bool (*serialized_size)(void *ud, const char *text, uint64_t *bytes_out);
    bool (*write)(void *ud, FILE *fp, const char *text, uint64_t *written_bytes);
    int (*load)(void *ud, FILE *fp, const void *wanted);
    const void *load_wanted;
} lgn2_kvstore_trailer_hooks;

typedef struct {
    int tokens;
    uint32_t text_bytes;
    uint8_t quant_bits;
    uint8_t ext_flags;
    double load_ms;
    char *path;
} lgn2_kvstore_load_result;

lgn2_kvstore_options lgn2_kvstore_default_options(void);
uint8_t lgn2_kvstore_reason_code(const char *reason);
const char *lgn2_kvstore_key_kind(uint8_t ext_flags);

bool lgn2_kvstore_open(lgn2_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, lgn2_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, lgn2_kvstore_log_type type, const char *msg),
                      void *log_ud);
void lgn2_kvstore_close(lgn2_kvstore *kc);
void lgn2_kvstore_clear(lgn2_kvstore *kc);
void lgn2_kvstore_entry_free(lgn2_kvstore_entry *e);

char *lgn2_kvstore_render_tokens_text(lgn2_engine *engine,
                                     const lgn2_tokens *tokens,
                                     size_t *out_len);
bool lgn2_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
void lgn2_kvstore_tokens_copy_prefix(lgn2_tokens *dst, const lgn2_tokens *src, int n);
void lgn2_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        lgn2_engine *engine,
        const lgn2_tokens *exact_prefix,
        const char *suffix_text,
        lgn2_tokens *out);

int lgn2_kvstore_store_len(const lgn2_kvstore *kc, int tokens);
int lgn2_kvstore_chat_anchor_pos(const lgn2_kvstore *kc,
                                const lgn2_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id);
int lgn2_kvstore_continued_store_target(const lgn2_kvstore *kc, int live_tokens);
void lgn2_kvstore_note_store(lgn2_kvstore *kc, int tokens);
int lgn2_kvstore_suppress_continued_store(lgn2_kvstore *kc, int tokens);
void lgn2_kvstore_restore_suppressed_continued(lgn2_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens);

bool lgn2_kvstore_file_size_fits(const lgn2_kvstore *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out);
double lgn2_kvstore_entry_eviction_score(const lgn2_kvstore_entry *e,
                                        const lgn2_tokens *live,
                                        uint64_t now,
                                        const lgn2_kvstore_eviction_context *incoming);
void lgn2_kvstore_evict(lgn2_kvstore *kc, const lgn2_tokens *live,
                       uint64_t extra_bytes,
                       const lgn2_kvstore_eviction_context *incoming);
int lgn2_kvstore_find_text_prefix(lgn2_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size);

bool lgn2_kvstore_store_live_prefix_text(lgn2_kvstore *kc,
                                        lgn2_engine *engine,
                                        lgn2_session *session,
                                        const lgn2_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const lgn2_kvstore_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len);
bool lgn2_kvstore_store_live_prefix(lgn2_kvstore *kc,
                                   lgn2_engine *engine,
                                   lgn2_session *session,
                                   const lgn2_tokens *tokens,
                                   int store_len,
                                   const char *reason,
                                   const lgn2_kvstore_trailer_hooks *hooks,
                                   char *err,
                                   size_t err_len);
bool lgn2_kvstore_maybe_store_continued(lgn2_kvstore *kc,
                                       lgn2_engine *engine,
                                       lgn2_session *session,
                                       const lgn2_kvstore_trailer_hooks *hooks,
                                       char *err,
                                       size_t err_len);
int lgn2_kvstore_try_load_text(lgn2_kvstore *kc,
                              lgn2_engine *engine,
                              lgn2_session *session,
                              const char *prompt_text,
                              lgn2_tokens *effective_prompt,
                              lgn2_kvstore_load_result *result,
                              const lgn2_kvstore_trailer_hooks *hooks,
                              bool responses_protocol);
void lgn2_kvstore_load_result_free(lgn2_kvstore_load_result *result);

bool lgn2_kvstore_read_header(FILE *fp, lgn2_kvstore_entry *e,
                             uint32_t *text_bytes);
bool lgn2_kvstore_read_entry_file(const char *path, const char sha[41],
                                 lgn2_kvstore_entry *out);
void lgn2_kvstore_fill_header(uint8_t h[LGN2_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes);
bool lgn2_kvstore_touch_file(const char *path, uint32_t hits);
bool lgn2_kvstore_sha_hex_name(const char *name, char sha[41]);
void lgn2_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
char *lgn2_kvstore_path_join(const char *dir, const char *name);
char *lgn2_kvstore_path_for_sha(lgn2_kvstore *kc, const char sha[41]);
void lgn2_kvstore_le_put32(uint8_t *p, uint32_t v);
uint32_t lgn2_kvstore_le_get32(const uint8_t *p);

#endif
