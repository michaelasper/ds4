#ifndef DS4_LAGUNA_LADDER_H
#define DS4_LAGUNA_LADDER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Parse the optional Laguna decode command-buffer ladder.  An unset/empty
 * value is disabled and succeeds for every model layer count.  A nonempty
 * value is a strict decimal comma list, increasing and bounded by the model
 * layer count; uint64_t masks support at most 64 requested layers. */
static inline bool ds4_laguna_decode_ladder_parse(
        const char *value,
        uint32_t    layer_count,
        uint64_t   *mask_out) {
    if (!mask_out) return false;
    *mask_out = 0;
    if (!value || value[0] == '\0') return true;
    if (layer_count > 64u) return false;

    const char *p = value;
    uint64_t previous = 0;
    uint64_t mask = 0;
    bool have_previous = false;
    for (;;) {
        if (*p < '0' || *p > '9') return false;

        uint64_t layer = 0;
        while (*p >= '0' && *p <= '9') {
            const uint64_t digit = (uint64_t)(*p - '0');
            if (layer > (UINT64_MAX - digit) / 10u) return false;
            layer = layer * 10u + digit;
            p++;
        }
        if (layer >= layer_count ||
            (have_previous && layer <= previous)) {
            return false;
        }

        mask |= UINT64_C(1) << layer;
        previous = layer;
        have_previous = true;

        if (*p == '\0') {
            *mask_out = mask;
            return true;
        }
        if (*p != ',') return false;
        p++;
        if (*p == '\0') return false;
    }
}

#ifdef __APPLE__
#include <stdio.h>

static inline bool ds4_laguna_decode_ladder_format(
        uint64_t mask,
        uint32_t layer_count,
        char    *out,
        size_t    out_size) {
    if (!out || out_size == 0 || layer_count > 64u) return false;

    size_t used = 0;
    out[0] = '\0';
    for (uint32_t layer = 0; layer < layer_count; layer++) {
        if ((mask & (UINT64_C(1) << layer)) == 0) continue;
        const int n = snprintf(out + used,
                               out_size - used,
                               used == 0 ? "%u" : ",%u",
                               layer);
        if (n < 0 || (size_t)n >= out_size - used) return false;
        used += (size_t)n;
    }
    return true;
}
#endif

#endif /* DS4_LAGUNA_LADDER_H */
