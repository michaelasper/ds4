#include "lgn.h"

#include <stdio.h>
#include <string.h>

static int lgn_utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

static uint64_t lgn_next_utf8_char(const char *s,
                                   uint64_t    len,
                                   uint64_t    pos) {
    int n = lgn_utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static bool lgn_ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool lgn_ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool lgn_ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool lgn_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static uint32_t lgn_utf8_peek_one(const char *s,
                                  uint64_t    len,
                                  uint64_t    pos,
                                  uint64_t   *next) {
    const uint8_t c0 = (uint8_t)s[pos];
    int n = lgn_utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;

    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}

typedef struct {
    uint32_t cp;
    uint64_t next;
    bool valid;
    bool is_letter;
    bool is_number;
    bool is_whitespace;
} lgn_bpe_char_info;

static bool lgn_unicode_whitespace(uint32_t cp) {
    if (cp < 128) return lgn_ascii_space((uint8_t)cp);
    return cp == 0x0085 ||
           cp == 0x00a0 ||
           cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) ||
           cp == 0x2028 ||
           cp == 0x2029 ||
           cp == 0x202f ||
           cp == 0x205f ||
           cp == 0x3000;
}

static bool lgn_unicode_number(uint32_t cp) {
    if (cp < 128) return lgn_ascii_digit((uint8_t)cp);
    return (cp >= 0x0660 && cp <= 0x0669) ||
           (cp >= 0x06f0 && cp <= 0x06f9) ||
           (cp >= 0x07c0 && cp <= 0x07c9) ||
           (cp >= 0x0966 && cp <= 0x096f) ||
           (cp >= 0x09e6 && cp <= 0x09ef) ||
           (cp >= 0x0a66 && cp <= 0x0a6f) ||
           (cp >= 0x0ae6 && cp <= 0x0aef) ||
           (cp >= 0x0b66 && cp <= 0x0b6f) ||
           (cp >= 0x0be6 && cp <= 0x0bef) ||
           (cp >= 0x0c66 && cp <= 0x0c6f) ||
           (cp >= 0x0ce6 && cp <= 0x0cef) ||
           (cp >= 0x0d66 && cp <= 0x0d6f) ||
           (cp >= 0x0de6 && cp <= 0x0def) ||
           (cp >= 0x0e50 && cp <= 0x0e59) ||
           (cp >= 0x0ed0 && cp <= 0x0ed9) ||
           (cp >= 0x0f20 && cp <= 0x0f29) ||
           (cp >= 0x1040 && cp <= 0x1049) ||
           (cp >= 0x1090 && cp <= 0x1099) ||
           (cp >= 0x17e0 && cp <= 0x17e9) ||
           (cp >= 0x1810 && cp <= 0x1819) ||
           (cp >= 0xff10 && cp <= 0xff19);
}

static bool lgn_unicode_punct_symbol(uint32_t cp) {
    if (cp < 128) return lgn_ascii_punct_symbol((uint8_t)cp);
    return (cp >= 0x00a1 && cp <= 0x00a9) ||
           (cp >= 0x00ab && cp <= 0x00ac) ||
           (cp >= 0x00ae && cp <= 0x00b1) ||
           cp == 0x00b4 ||
           (cp >= 0x00b6 && cp <= 0x00b8) ||
           cp == 0x00bb ||
           cp == 0x00bf ||
           cp == 0x00d7 ||
           cp == 0x00f7 ||
           (cp >= 0x02c2 && cp <= 0x02df) ||
           (cp >= 0x02e5 && cp <= 0x02eb) ||
           (cp >= 0x02ed && cp <= 0x02ff) ||
           (cp >= 0x0375 && cp <= 0x037e) ||
           (cp >= 0x0384 && cp <= 0x0385) ||
           cp == 0x0387 ||
           (cp >= 0x055a && cp <= 0x055f) ||
           (cp >= 0x0589 && cp <= 0x058a) ||
           (cp >= 0x05be && cp <= 0x05c0) ||
           cp == 0x05c3 ||
           (cp >= 0x05c6 && cp <= 0x05c7) ||
           (cp >= 0x0609 && cp <= 0x060a) ||
           (cp >= 0x060c && cp <= 0x060d) ||
           cp == 0x061b ||
           (cp >= 0x061e && cp <= 0x061f) ||
           cp == 0x066a ||
           cp == 0x066d ||
           cp == 0x06d4 ||
           (cp >= 0x2000 && cp <= 0x206f) ||
           (cp >= 0x20a0 && cp <= 0x20cf) ||
           (cp >= 0x2100 && cp <= 0x214f) ||
           (cp >= 0x2190 && cp <= 0x23ff) ||
           (cp >= 0x2460 && cp <= 0x24ff) ||
           (cp >= 0x2500 && cp <= 0x2775) ||
           (cp >= 0x2794 && cp <= 0x2bff) ||
           (cp >= 0x2e00 && cp <= 0x2e7f) ||
           (cp >= 0x3000 && cp <= 0x303f) ||
           (cp >= 0xfd3e && cp <= 0xfd3f) ||
           (cp >= 0xfe10 && cp <= 0xfe6f) ||
           (cp >= 0xff01 && cp <= 0xff0f) ||
           (cp >= 0xff1a && cp <= 0xff20) ||
           (cp >= 0xff3b && cp <= 0xff40) ||
           (cp >= 0xff5b && cp <= 0xff65) ||
           (cp >= 0x1f000 && cp <= 0x1faff);
}

static lgn_bpe_char_info lgn_bpe_char_at(const char *s,
                                         uint64_t    len,
                                         uint64_t    pos) {
    lgn_bpe_char_info info;
    memset(&info, 0, sizeof(info));
    if (pos >= len) return info;

    info.valid = true;
    info.cp = lgn_utf8_peek_one(s, len, pos, &info.next);
    info.is_whitespace = lgn_unicode_whitespace(info.cp);
    info.is_number = lgn_unicode_number(info.cp);
    if (info.cp < 128) {
        info.is_letter = lgn_ascii_alpha((uint8_t)info.cp);
    } else {
        info.is_letter =
            !info.is_whitespace &&
            !info.is_number &&
            !lgn_unicode_punct_symbol(info.cp);
    }
    return info;
}

static uint32_t lgn_ascii_tolower_cp(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + ('a' - 'A');
    return cp;
}

static bool lgn_emit_span(const char       *text,
                          uint64_t          start,
                          uint64_t          end,
                          lgn_bpe_piece_fn emit,
                          void             *userdata) {
    return emit(text + start, (size_t)(end - start), userdata);
}

/* Laguna uses max_digits=1 and applies this segmentation policy independently
 * to each non-LF/LF span in lgn_bpe_pretokenize(). */
static bool lgn_bpe_tokenize_text_segment(const char       *text,
                                          uint64_t          len,
                                          int               max_digits,
                                          lgn_bpe_piece_fn  emit,
                                          void             *userdata) {
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        lgn_bpe_char_info cur = lgn_bpe_char_at(text, len, pos);

        if (!cur.valid) break;

        if (cur.cp == '\'' && cur.next < len) {
            lgn_bpe_char_info next = lgn_bpe_char_at(text, len, cur.next);
            uint32_t n1 = lgn_ascii_tolower_cp(next.cp);
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                pos = next.next;
                if (!lgn_emit_span(text, start, pos, emit, userdata)) return false;
                continue;
            }
            if (next.valid && next.next < len) {
                lgn_bpe_char_info next2 = lgn_bpe_char_at(text, len, next.next);
                uint32_t n2 = lgn_ascii_tolower_cp(next2.cp);
                if ((n1 == 'r' && n2 == 'e') ||
                    (n1 == 'v' && n2 == 'e') ||
                    (n1 == 'l' && n2 == 'l')) {
                    pos = next2.next;
                    if (!lgn_emit_span(text, start, pos, emit, userdata)) return false;
                    continue;
                }
            }
        }

        if (!(cur.cp == '\r' || cur.cp == '\n' || cur.is_number)) {
            lgn_bpe_char_info next = lgn_bpe_char_at(text, len, cur.next);
            if (cur.is_letter || next.is_letter) {
                pos = cur.next;
                while (pos < len) {
                    lgn_bpe_char_info scan = lgn_bpe_char_at(text, len, pos);
                    if (!scan.valid || !scan.is_letter) break;
                    pos = scan.next;
                }
                if (!lgn_emit_span(text, start, pos, emit, userdata)) return false;
                continue;
            }
        }

        if (cur.is_number) {
            int ndigits = 0;
            while (pos < len && ndigits < max_digits) {
                lgn_bpe_char_info scan = lgn_bpe_char_at(text, len, pos);
                if (!scan.valid || !scan.is_number) break;
                pos = scan.next;
                ndigits++;
            }
            if (!lgn_emit_span(text, start, pos, emit, userdata)) return false;
            continue;
        }

        lgn_bpe_char_info punct = cur;
        uint64_t punct_pos = pos;
        if (cur.cp == ' ') {
            punct_pos = cur.next;
            punct = lgn_bpe_char_at(text, len, punct_pos);
        }
        if (punct.valid &&
            !punct.is_whitespace &&
            !punct.is_letter &&
            !punct.is_number) {
            pos = punct_pos;
            while (pos < len) {
                lgn_bpe_char_info scan = lgn_bpe_char_at(text, len, pos);
                if (!scan.valid ||
                    scan.is_whitespace ||
                    scan.is_letter ||
                    scan.is_number) {
                    break;
                }
                pos = scan.next;
            }
            while (pos < len) {
                lgn_bpe_char_info scan = lgn_bpe_char_at(text, len, pos);
                if (!scan.valid || !(scan.cp == '\r' || scan.cp == '\n')) break;
                pos = scan.next;
            }
            if (!lgn_emit_span(text, start, pos, emit, userdata)) return false;
            continue;
        }

        if (cur.is_whitespace) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            uint64_t last_ws_start = pos;
            int nspace = 0;
            while (p < len) {
                lgn_bpe_char_info scan = lgn_bpe_char_at(text, len, p);
                if (!scan.valid || !scan.is_whitespace) break;
                last_ws_start = p;
                if (scan.cp == '\r' || scan.cp == '\n') last_newline_end = scan.next;
                p = scan.next;
                if (nspace < 2) nspace++;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (nspace > 1 && p < len) {
                pos = last_ws_start;
            } else {
                pos = p;
            }
            if (!lgn_emit_span(text, start, pos, emit, userdata)) return false;
            continue;
        }

        pos = cur.next;
        if (pos == start) pos = lgn_next_utf8_char(text, len, pos);
        if (!lgn_emit_span(text, start, pos, emit, userdata)) return false;
    }

    return true;
}

static bool lgn_bpe_tokenize_text_laguna(const char       *text,
                                         lgn_bpe_piece_fn  emit,
                                         void             *userdata) {
    const uint64_t len = strlen(text);
    uint64_t pos = 0;

    /* Laguna's first pre-tokenizer regex separates non-newline spans from
     * runs of LF bytes before applying the GPT-2-style expression. This is
     * observable for CRLF: CR remains in the preceding span and LF starts a
     * new one, so they must not merge into one BPE piece. */
    while (pos < len) {
        const uint64_t start = pos;
        if (text[pos] == '\n') {
            while (pos < len && text[pos] == '\n') pos++;
        } else {
            while (pos < len && text[pos] != '\n') pos++;
        }
        if (!lgn_bpe_tokenize_text_segment(text + start,
                                           pos - start,
                                           1,
                                           emit,
                                           userdata)) {
            return false;
        }
    }
    return true;
}

bool lgn_bpe_pretokenize(const char       *text,
                         lgn_bpe_piece_fn  emit,
                         void             *userdata) {
    if (!text || !emit) return false;
    return lgn_bpe_tokenize_text_laguna(text, emit, userdata);
}

bool lgn_architecture_is_supported(const char *value, size_t value_len) {
    static const char supported[] = "laguna";
    return value && value_len == sizeof(supported) - 1u &&
           memcmp(value, supported, sizeof(supported) - 1u) == 0;
}

bool lgn_decode_ladder_parse(const char *value,
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

bool lgn_decode_ladder_format(uint64_t mask,
                              uint32_t layer_count,
                              char    *out,
                              size_t   out_size) {
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
