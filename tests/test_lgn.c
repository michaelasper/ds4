#include "../lgn.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(condition, message) do {                                      \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL: %s\n", message);                            \
        failures++;                                                          \
    }                                                                         \
} while (0)

static void test_architecture(void) {
    CHECK(lgn_architecture_is_supported("laguna", 6),
          "literal Laguna architecture is accepted");
    CHECK(!lgn_architecture_is_supported(NULL, 0),
          "missing architecture is rejected");
    CHECK(!lgn_architecture_is_supported("", 0),
          "empty architecture is rejected");
    CHECK(!lgn_architecture_is_supported("Laguna", 6),
          "architecture matching is case-sensitive");
    CHECK(!lgn_architecture_is_supported("laguna ", 7),
          "architecture whitespace is rejected");
    CHECK(!lgn_architecture_is_supported("deepseek4", 9),
          "DeepSeek architecture is rejected");
    CHECK(!lgn_architecture_is_supported("glm-dsa", 7),
          "GLM architecture is rejected");
}

static void test_ladder_parser(void) {
    const uint64_t all_expected =
        (UINT64_C(1) << 0) |
        (UINT64_C(1) << 1) |
        (UINT64_C(1) << 7) |
        (UINT64_C(1) << 15) |
        (UINT64_C(1) << 23) |
        (UINT64_C(1) << 31) |
        (UINT64_C(1) << 39) |
        (UINT64_C(1) << 47);
    uint64_t mask = UINT64_MAX;

    CHECK(lgn_decode_ladder_parse(NULL, 48, &mask), "NULL ladder parses");
    CHECK(mask == 0, "NULL ladder clears mask");
    mask = UINT64_MAX;
    CHECK(lgn_decode_ladder_parse("", 48, &mask), "empty ladder parses");
    CHECK(mask == 0, "empty ladder clears mask");
    CHECK(lgn_decode_ladder_parse("7,15,23,31,39,47", 48, &mask),
          "valid sparse ladder parses");
    CHECK(mask == (all_expected & ~UINT64_C(0x3)),
          "valid sparse ladder mask");
    CHECK(lgn_decode_ladder_parse("0,1,7,15,23,31,39,47", 48, &mask),
          "valid full ladder parses");
    CHECK(mask == all_expected, "valid full ladder mask");
    CHECK(lgn_decode_ladder_parse("1,7,15,23,31,39", 48, &mask),
          "valid partial ladder parses");
    CHECK(mask == (all_expected & ~UINT64_C(0x800000000001)),
          "valid partial ladder mask");
    CHECK(lgn_decode_ladder_parse("47", 48, &mask), "last layer parses");
    CHECK(mask == (UINT64_C(1) << 47), "last layer mask");
    CHECK(lgn_decode_ladder_parse("0,63", 64, &mask),
          "64-layer boundary parses");
    CHECK(mask == ((UINT64_C(1) << 0) | (UINT64_C(1) << 63)),
          "64-layer boundary mask");

    mask = UINT64_MAX;
    CHECK(lgn_decode_ladder_parse(NULL, 65, &mask),
          "NULL ladder ignores layer-count limit");
    CHECK(mask == 0, "NULL ladder clears mask above limit");
    mask = UINT64_MAX;
    CHECK(lgn_decode_ladder_parse("", 79, &mask),
          "empty ladder ignores layer-count limit");
    CHECK(mask == 0, "empty ladder clears mask above limit");

    static const char *invalid[] = {
        "7,7", "15,7", "48", "-1", "1, 7", "1,,7", "1,7,",
        "1,x", "1,7\n", "18446744073709551616",
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        mask = UINT64_MAX;
        CHECK(!lgn_decode_ladder_parse(invalid[i], 48, &mask),
              "invalid ladder is rejected");
        CHECK(mask == 0, "invalid ladder clears mask");
    }
    mask = UINT64_MAX;
    CHECK(!lgn_decode_ladder_parse("0", 0, &mask),
          "zero-layer nonempty ladder is rejected");
    CHECK(mask == 0, "zero-layer rejection clears mask");
    mask = UINT64_MAX;
    CHECK(!lgn_decode_ladder_parse("0", 65, &mask),
          "nonempty ladder above mask limit is rejected");
    CHECK(mask == 0, "above-limit rejection clears mask");
    CHECK(!lgn_decode_ladder_parse("0", 48, NULL),
          "NULL output is rejected");
}

static void test_ladder_formatter(void) {
    const uint64_t mask =
        (UINT64_C(1) << 0) |
        (UINT64_C(1) << 1) |
        (UINT64_C(1) << 7) |
        (UINT64_C(1) << 15) |
        (UINT64_C(1) << 23) |
        (UINT64_C(1) << 31) |
        (UINT64_C(1) << 39) |
        (UINT64_C(1) << 47);
    char out[128];
    CHECK(lgn_decode_ladder_format(mask, 48, out, sizeof(out)),
          "valid ladder formats");
    CHECK(strcmp(out, "0,1,7,15,23,31,39,47") == 0,
          "ladder formatter is canonical");
    CHECK(lgn_decode_ladder_format(0, 48, out, sizeof(out)),
          "empty ladder formats");
    CHECK(strcmp(out, "") == 0, "empty ladder formatter output");
    char short_out[4];
    CHECK(!lgn_decode_ladder_format(mask, 48, short_out, sizeof(short_out)),
          "short ladder output is rejected");
    CHECK(!lgn_decode_ladder_format(mask, 65, out, sizeof(out)),
          "formatter rejects more than 64 layers");
    CHECK(!lgn_decode_ladder_format(mask, 48, NULL, sizeof(out)),
          "formatter rejects NULL output");
    CHECK(!lgn_decode_ladder_format(mask, 48, out, 0),
          "formatter rejects zero output size");
}

static void test_s21_topology(void) {
    CHECK(LGN_LAYER_COUNT == 48u, "S2.1 has 48 layers");
    CHECK(LGN_GLOBAL_HEAD_COUNT == 48u, "S2.1 global layers have 48 heads");
    CHECK(LGN_SWA_HEAD_COUNT == 72u, "S2.1 SWA layers have 72 heads");
    for (uint32_t il = 0; il < LGN_LAYER_COUNT; il++) {
        const bool global = (il % 4u) == 0;
        CHECK(lgn_layer_head_count(il) == (global ? 48u : 72u),
              "S2.1 layer head count");
        CHECK(lgn_layer_is_swa(il) == !global, "S2.1 SWA topology");
    }
    CHECK(lgn_layer_head_count(LGN_LAYER_COUNT) == 0,
          "out-of-range topology returns zero");
    CHECK(!lgn_layer_is_swa(LGN_LAYER_COUNT),
          "out-of-range topology is not SWA");
}

typedef struct {
    size_t count;
    size_t length[64];
    unsigned char bytes[64][128];
    size_t fail_after;
} span_capture;

typedef struct {
    const unsigned char *bytes;
    size_t length;
} expected_span;

static bool capture_span(const char *piece, size_t piece_len, void *userdata) {
    span_capture *capture = userdata;
    if (capture->count >= capture->fail_after) return false;
    if (capture->count >= sizeof(capture->length) / sizeof(capture->length[0]) ||
        piece_len > sizeof(capture->bytes[0])) {
        return false;
    }
    capture->length[capture->count] = piece_len;
    memcpy(capture->bytes[capture->count], piece, piece_len);
    capture->count++;
    return true;
}

static void check_pretokenized(const char          *name,
                               const char          *text,
                               const expected_span *expected,
                               size_t               expected_count) {
    span_capture capture;
    memset(&capture, 0, sizeof(capture));
    capture.fail_after = (size_t)-1;

    CHECK(lgn_bpe_pretokenize(text, capture_span, &capture), name);
    if (capture.count != expected_count) {
        fprintf(stderr,
                "FAIL: %s (got %zu spans, expected %zu)\n",
                name,
                capture.count,
                expected_count);
        failures++;
        return;
    }
    for (size_t i = 0; i < expected_count; i++) {
        if (capture.length[i] != expected[i].length ||
            memcmp(capture.bytes[i], expected[i].bytes, expected[i].length) != 0) {
            fprintf(stderr, "FAIL: %s (span %zu differs)\n", name, i);
            failures++;
        }
    }
}

#define EXPECT_LITERAL(value) \
    { (const unsigned char *)(value), sizeof(value) - 1u }

static void test_laguna_pretokenizer(void) {
    static const expected_span contractions[] = {
        EXPECT_LITERAL("can"),
        EXPECT_LITERAL("'t"),
        EXPECT_LITERAL("we"),
        EXPECT_LITERAL("'LL"),
        EXPECT_LITERAL("'re"),
        EXPECT_LITERAL("'M"),
        EXPECT_LITERAL("'d"),
    };
    check_pretokenized("ASCII words, contractions, and case folding",
                       "can'twe'LL're'M'd",
                       contractions,
                       sizeof(contractions) / sizeof(contractions[0]));

    static const expected_span words_numbers[] = {
        EXPECT_LITERAL("abc"),
        EXPECT_LITERAL("1"),
        EXPECT_LITERAL("2"),
        EXPECT_LITERAL("3"),
        EXPECT_LITERAL("bar"),
    };
    check_pretokenized("max_digits=1 keeps every ASCII digit separate",
                       "abc123bar",
                       words_numbers,
                       sizeof(words_numbers) / sizeof(words_numbers[0]));

    static const expected_span punctuation[] = {
        EXPECT_LITERAL("hello"),
        EXPECT_LITERAL(","),
        EXPECT_LITERAL(" world"),
        EXPECT_LITERAL("!!"),
    };
    check_pretokenized("punctuation runs and leading-space words",
                       "hello, world!!",
                       punctuation,
                       sizeof(punctuation) / sizeof(punctuation[0]));

    static const expected_span punctuation_before_word[] = {
        EXPECT_LITERAL("!word"),
        EXPECT_LITERAL("?"),
    };
    check_pretokenized("punctuation immediately before a letter joins that word",
                       "!word?",
                       punctuation_before_word,
                       sizeof(punctuation_before_word) /
                           sizeof(punctuation_before_word[0]));

    static const expected_span spaces_before_word[] = {
        EXPECT_LITERAL(" "),
        EXPECT_LITERAL(" abc"),
        EXPECT_LITERAL("  "),
    };
    check_pretokenized("multiple spaces split before a following word",
                       "  abc  ",
                       spaces_before_word,
                       sizeof(spaces_before_word) / sizeof(spaces_before_word[0]));

    static const expected_span spaces_before_punctuation[] = {
        EXPECT_LITERAL(" "),
        EXPECT_LITERAL(" !!"),
        EXPECT_LITERAL("x"),
    };
    check_pretokenized("spaces before punctuation stay with the punctuation run",
                       "  !!x",
                       spaces_before_punctuation,
                       sizeof(spaces_before_punctuation) /
                           sizeof(spaces_before_punctuation[0]));

    static const expected_span newline_runs[] = {
        EXPECT_LITERAL("foo"),
        EXPECT_LITERAL("\n\n"),
        EXPECT_LITERAL("bar"),
    };
    check_pretokenized("LF runs are one top-level span",
                       "foo\n\nbar",
                       newline_runs,
                       sizeof(newline_runs) / sizeof(newline_runs[0]));

    static const expected_span crlf[] = {
        EXPECT_LITERAL("foo"),
        EXPECT_LITERAL("\r"),
        EXPECT_LITERAL("\n"),
        EXPECT_LITERAL("bar"),
    };
    check_pretokenized("CR stays before the top-level LF span",
                       "foo\r\nbar",
                       crlf,
                       sizeof(crlf) / sizeof(crlf[0]));

    static const expected_span punctuation_crlf[] = {
        EXPECT_LITERAL("!\r"),
        EXPECT_LITERAL("\n"),
    };
    check_pretokenized("punctuation absorbs trailing CR but not LF",
                       "!\r\n",
                       punctuation_crlf,
                       sizeof(punctuation_crlf) / sizeof(punctuation_crlf[0]));

    static const expected_span whitespace_crlf[] = {
        EXPECT_LITERAL(" \r"),
        EXPECT_LITERAL("\n"),
    };
    check_pretokenized("whitespace keeps CR with its preceding spaces",
                       " \r\n",
                       whitespace_crlf,
                       sizeof(whitespace_crlf) / sizeof(whitespace_crlf[0]));

    static const expected_span unicode[] = {
        EXPECT_LITERAL("\xC3\xA9"),
        EXPECT_LITERAL("\xC2\xA9!"),
        EXPECT_LITERAL("\xC2\xA0x"),
    };
    check_pretokenized("Unicode letters, punctuation, and whitespace classify correctly",
                       "\xC3\xA9\xC2\xA9!\xC2\xA0x",
                       unicode,
                       sizeof(unicode) / sizeof(unicode[0]));

    static const char unicode_numbers_text[] =
        "\xD9\xA1\xD9\xA2\xEF\xBC\x93\x41";
    static const expected_span unicode_numbers[] = {
        EXPECT_LITERAL("\xD9\xA1"),
        EXPECT_LITERAL("\xD9\xA2"),
        EXPECT_LITERAL("\xEF\xBC\x93"),
        EXPECT_LITERAL("A"),
    };
    check_pretokenized("Unicode decimal digits also obey max_digits=1",
                       unicode_numbers_text,
                       unicode_numbers,
                       sizeof(unicode_numbers) / sizeof(unicode_numbers[0]));

    static const char unicode_space_text[] = "\xE2\x80\x83x";
    static const expected_span unicode_space[] = {
        EXPECT_LITERAL("\xE2\x80\x83x"),
    };
    check_pretokenized("Unicode whitespace before a letter joins that word",
                       unicode_space_text,
                       unicode_space,
                       sizeof(unicode_space) / sizeof(unicode_space[0]));

    static const char malformed_join_text[] = "\xC2\x41";
    static const expected_span malformed_join[] = {
        EXPECT_LITERAL("\xC2\x41"),
    };
    check_pretokenized("malformed UTF-8 keeps the original bytes and boundary",
                       malformed_join_text,
                       malformed_join,
                       sizeof(malformed_join) / sizeof(malformed_join[0]));

    static const char malformed_continuations_text[] = "\x80\x81!";
    static const expected_span malformed_continuations[] = {
        EXPECT_LITERAL("\x80\x81"),
        EXPECT_LITERAL("!"),
    };
    check_pretokenized("malformed continuation bytes remain lossless",
                       malformed_continuations_text,
                       malformed_continuations,
                       sizeof(malformed_continuations) /
                           sizeof(malformed_continuations[0]));

    static const char malformed_tail_text[] = "\xE2\x82";
    static const expected_span malformed_tail[] = {
        EXPECT_LITERAL("\xE2\x82"),
    };
    check_pretokenized("truncated UTF-8 at end remains one byte-preserving span",
                       malformed_tail_text,
                       malformed_tail,
                       sizeof(malformed_tail) / sizeof(malformed_tail[0]));

    span_capture failure;
    memset(&failure, 0, sizeof(failure));
    failure.fail_after = 1;
    CHECK(!lgn_bpe_pretokenize("first second", capture_span, &failure),
          "callback failure propagates");
    CHECK(failure.count == 1 &&
              failure.length[0] == sizeof("first") - 1u &&
              memcmp(failure.bytes[0], "first", sizeof("first") - 1u) == 0,
          "callback failure stops before the next span");

    span_capture empty;
    memset(&empty, 0, sizeof(empty));
    empty.fail_after = (size_t)-1;
    CHECK(lgn_bpe_pretokenize("", capture_span, &empty),
          "empty text pre-tokenizes successfully");
    CHECK(empty.count == 0, "empty text has no spans");
    CHECK(!lgn_bpe_pretokenize(NULL, capture_span, &empty),
          "NULL text is rejected");
    CHECK(!lgn_bpe_pretokenize("text", NULL, &empty),
          "NULL callback is rejected");
}

int main(void) {
    test_architecture();
    test_ladder_parser();
    test_ladder_formatter();
    test_s21_topology();
    test_laguna_pretokenizer();
    if (failures != 0) {
        fprintf(stderr, "test-lgn: %d failure(s)\n", failures);
        return 1;
    }
    puts("test-lgn: ok");
    return 0;
}
