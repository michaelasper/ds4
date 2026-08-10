#include "../ds4_gpu.h"

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

/* ds4_metal.m keeps this shared logging helper for diagnostics.  The focused
 * selector test never emits a tty-sensitive report, so a local test stub
 * keeps the test CPU-only and avoids linking the full engine. */
bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void check_int(const char *label, int got, int expected) {
    if (got == expected) return;
    fprintf(stderr, "FAIL: %s: got %d, expected %d\n", label, got, expected);
    g_failures++;
}

static void check_u64(const char *label, uint64_t got, uint64_t expected) {
    if (got == expected) return;
    fprintf(stderr, "FAIL: %s: got %llu, expected %llu\n",
            label, (unsigned long long)got, (unsigned long long)expected);
    g_failures++;
}

static void check_world(int world) {
    check_int("set TP world", ds4_gpu_q8_0_mv_dispatch_test_set_tp_world(world), 1);
}

static void check_nsg(const char *label, int expected) {
    check_int(label, ds4_gpu_q8_0_mv_dispatch_test_nsg(), expected);
}

static void repeat_current_selection(const char *label, int expected) {
    for (int i = 0; i < 10000; i++) {
        check_nsg(label, expected);
    }
}

static char *save_env(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    size_t bytes = strlen(value) + 1u;
    char *copy = (char *)malloc(bytes);
    if (!copy) {
        fprintf(stderr, "FAIL: unable to save %s\n", name);
        g_failures++;
        return NULL;
    }
    memcpy(copy, value, bytes);
    return copy;
}

static void restore_env(const char *name, char *saved) {
    if (saved) {
        (void)setenv(name, saved, 1);
        free(saved);
    } else {
        (void)unsetenv(name);
    }
}

int main(void) {
    const char *name = "DS4_METAL_Q8_MV_NSG";
    char *saved = save_env(name);

    /* Missing configuration keeps the original world-specific defaults.  A
     * repeated decode-shaped call sequence probes/configures each world once,
     * and switching back reuses its independent slot. */
    (void)unsetenv(name);
    ds4_gpu_q8_0_mv_dispatch_test_reset();
    check_world(1);
    check_nsg("world 1 default", 4);
    repeat_current_selection("world 1 cached default", 4);
    check_u64("world 1 parser probes", ds4_gpu_q8_0_mv_dispatch_test_probe_count(), 1);
    check_world(2);
    check_nsg("world 2 default", 2);
    repeat_current_selection("world 2 cached default", 2);
    check_u64("world 1/world 2 parser probes", ds4_gpu_q8_0_mv_dispatch_test_probe_count(), 2);
    check_world(1);
    check_nsg("world 1 remains cached after toggle", 4);
    check_u64("world toggle does not re-probe", ds4_gpu_q8_0_mv_dispatch_test_probe_count(), 2);

    /* A valid override is selected independently for both slots. */
    (void)setenv(name, "6", 1);
    ds4_gpu_q8_0_mv_dispatch_test_reset();
    check_world(1);
    check_nsg("explicit override world 1", 6);
    repeat_current_selection("explicit override world 1 cached", 6);
    check_world(2);
    check_nsg("explicit override world 2", 6);
    repeat_current_selection("explicit override world 2 cached", 6);
    check_u64("explicit override parser probes", ds4_gpu_q8_0_mv_dispatch_test_probe_count(), 2);

    /* Keep malformed, below-minimum, overflow, and above-maximum behavior
     * identical to ds4_gpu_env_u64: defaults for rejected values and a clamp
     * to 8 for a well-formed value above the accepted maximum. */
    static const struct {
        const char *value;
        int world1;
        int world2;
    } cases[] = {
        { "", 4, 2 },
        { "garbage", 4, 2 },
        { "0", 4, 2 },
        { "18446744073709551616", 4, 2 },
        { "9", 8, 8 },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        (void)setenv(name, cases[i].value, 1);
        ds4_gpu_q8_0_mv_dispatch_test_reset();
        check_world(1);
        check_nsg("parser fallback/clamp world 1", cases[i].world1);
        check_world(2);
        check_nsg("parser fallback/clamp world 2", cases[i].world2);
        check_u64("malformed/out-of-range parser probes",
                  ds4_gpu_q8_0_mv_dispatch_test_probe_count(), 2);
    }

    /* Environment changes are deliberately ignored until the lifecycle
     * boundary, then become visible after cleanup/reinitialization. */
    (void)setenv(name, "4", 1);
    ds4_gpu_q8_0_mv_dispatch_test_reset();
    check_world(1);
    check_nsg("pre-cleanup value", 4);
    (void)setenv(name, "7", 1);
    check_nsg("environment remains frozen while initialized", 4);
    check_u64("frozen environment probe count", ds4_gpu_q8_0_mv_dispatch_test_probe_count(), 1);
    ds4_gpu_cleanup();
    check_world(1);
    check_nsg("post-cleanup reinitialized value", 7);
    check_u64("reinitialized parser probes", ds4_gpu_q8_0_mv_dispatch_test_probe_count(), 1);

    restore_env(name, saved);
    if (g_failures != 0) {
        fprintf(stderr, "q8 dispatch cache: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("q8 dispatch cache: PASS (world defaults 4/2, two cached slots, deterministic probes)\n");
    return 0;
}
