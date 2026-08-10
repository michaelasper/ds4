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

int main(void) {
    test_ladder_parser();
    test_ladder_formatter();
    test_s21_topology();
    if (failures != 0) {
        fprintf(stderr, "test-lgn: %d failure(s)\n", failures);
        return 1;
    }
    puts("test-lgn: ok");
    return 0;
}
