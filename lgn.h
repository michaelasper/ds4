#ifndef LGN_H
#define LGN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The first product fork is deliberately for one exact Laguna S 2.1
 * topology.  Keep these constants independent of lgn2_engine.c so the policy can be
 * tested and used by the future lgn_* modules without pulling in the engine
 * implementation. */
enum {
    LGN_LAYER_COUNT       = 48u,
    LGN_GLOBAL_HEAD_COUNT = 48u,
    LGN_SWA_HEAD_COUNT    = 72u,
};

/* The fork admits one GGUF architecture spelling.  The input is an explicit
 * byte span because GGUF strings are not NUL-terminated. */
bool lgn_architecture_is_supported(const char *value, size_t value_len);

/* Callback used by the Laguna Unicode-aware BPE pre-tokenizer. Each callback span
 * is a non-empty borrowed slice of the input, delivered in source order.  The
 * slice remains valid only until the callback returns.  Returning false stops
 * tokenization and makes lgn_bpe_pretokenize return false. */
typedef bool (*lgn_bpe_piece_fn)(const char *piece,
                                 size_t      piece_len,
                                 void       *userdata);

/* Split a NUL-terminated input into the exact byte spans consumed by
 * Laguna's BPE implementation.  This function performs no vocabulary or BPE
 * merge work; callers own those concerns in their callback. */
bool lgn_bpe_pretokenize(const char       *text,
                         lgn_bpe_piece_fn  emit,
                         void             *userdata);

/* Parse the optional Laguna decode command-buffer ladder.  An unset/empty
 * value is disabled and succeeds for every model layer count.  A nonempty
 * value is a strict decimal comma list, increasing and bounded by the model
 * layer count; uint64_t masks support at most 64 requested layers. */
bool lgn_decode_ladder_parse(const char *value,
                             uint32_t    layer_count,
                             uint64_t   *mask_out);

/* Format the canonical comma-separated representation of a ladder mask. */
bool lgn_decode_ladder_format(uint64_t mask,
                              uint32_t layer_count,
                              char    *out,
                              size_t   out_size);

/* Fixed Laguna S 2.1 layer topology.  Every fourth layer is the global
 * 48-head layer; the remaining layers use 72-head sliding-window attention.
 * Return zero for an out-of-range index so callers can retain their existing
 * fatal/error policy at the engine boundary. */
static inline uint32_t lgn_layer_head_count(uint32_t il) {
    if (il >= LGN_LAYER_COUNT) return 0;
    return (il % 4u) == 0 ? LGN_GLOBAL_HEAD_COUNT : LGN_SWA_HEAD_COUNT;
}

static inline bool lgn_layer_is_swa(uint32_t il) {
    return il < LGN_LAYER_COUNT && (il % 4u) != 0;
}

#endif /* LGN_H */
