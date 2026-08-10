/* Focused model-independent tests for the SSD streaming seams. */

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef ssize_t (*ds4_test_stream_pread_fn)(int, void *, size_t, off_t);

bool ds4_test_stream_readahead_enabled(void);
bool ds4_test_stream_madvise_willneed_enabled(void);
uint32_t ds4_test_stream_pread_chunk(void);
uint32_t ds4_test_stream_auto_preload_cap(void);
void ds4_test_stream_set_pread_hook(ds4_test_stream_pread_fn fn);
bool ds4_test_stream_pread_range_fd(int fd,
                                    uint64_t model_size,
                                    uint64_t offset,
                                    uint64_t size,
                                    uint8_t *scratch,
                                    uint64_t *read_bytes,
                                    uint8_t *sink);
uint32_t ds4_test_stream_readahead_coalesce(
        const uint64_t *offsets,
        const uint64_t *ends,
        uint32_t        n_spans,
        uint64_t       *out_offsets,
        uint64_t       *out_sizes,
        uint32_t        out_cap,
        uint64_t        model_size);
uint32_t ds4_test_stream_builtin_count(int variant);
uint32_t ds4_test_stream_builtin_target(int variant, uint32_t requested);
uint32_t ds4_test_stream_builtin_load_count(int variant, uint32_t requested);
uint32_t ds4_test_stream_hotlist_file_load_count(const char *path,
                                                  uint32_t    requested);
bool ds4_test_stream_hotlist_should_skip(bool     from_file,
                                         bool     refresh_builtin_glm,
                                         uint32_t current_count,
                                         uint32_t requested,
                                         int      variant);
bool ds4_test_stream_prepare_failure_cleanup(uint32_t n_jobs,
                                             uint32_t fail_at,
                                             uint32_t *started_out,
                                             uint32_t *joined_out);

static unsigned g_failures;

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        g_failures++; \
    } \
} while (0)

static char *save_env(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    const size_t len = strlen(value);
    char *copy = malloc(len + 1u);
    CHECK(copy != NULL);
    if (!copy) return NULL;
    memcpy(copy, value, len + 1u);
    return copy;
}

static void restore_env(const char *name, char *saved) {
    if (saved) {
        CHECK(setenv(name, saved, 1) == 0);
        free(saved);
    } else {
        CHECK(unsetenv(name) == 0);
    }
}

static void test_platform_gates(void) {
    char *saved_read_disable = save_env("DS4_METAL_DISABLE_STREAMING_READAHEAD");
    char *saved_read_enable = save_env("DS4_METAL_ENABLE_STREAMING_READAHEAD");
    char *saved_madv_disable = save_env("DS4_METAL_DISABLE_STREAMING_MADVISE_WILLNEED");
    char *saved_madv_enable = save_env("DS4_METAL_ENABLE_STREAMING_MADVISE_WILLNEED");
    char *saved_cap = save_env("DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP");

    unsetenv("DS4_METAL_DISABLE_STREAMING_READAHEAD");
    unsetenv("DS4_METAL_ENABLE_STREAMING_READAHEAD");
    unsetenv("DS4_METAL_DISABLE_STREAMING_MADVISE_WILLNEED");
    unsetenv("DS4_METAL_ENABLE_STREAMING_MADVISE_WILLNEED");
    unsetenv("DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP");
#if defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    CHECK(ds4_test_stream_readahead_enabled());
    CHECK(ds4_test_stream_madvise_willneed_enabled());
    CHECK(ds4_test_stream_pread_chunk() == 4u * 1024u * 1024u);
    CHECK(ds4_test_stream_auto_preload_cap() == 8192u);
#else
    CHECK(!ds4_test_stream_readahead_enabled());
    CHECK(!ds4_test_stream_madvise_willneed_enabled());
    CHECK(ds4_test_stream_pread_chunk() == 1u * 1024u * 1024u);
    CHECK(ds4_test_stream_auto_preload_cap() == 4096u);
#endif

    setenv("DS4_METAL_DISABLE_STREAMING_READAHEAD", "1", 1);
    setenv("DS4_METAL_ENABLE_STREAMING_READAHEAD", "1", 1);
    CHECK(!ds4_test_stream_readahead_enabled());
    setenv("DS4_METAL_DISABLE_STREAMING_MADVISE_WILLNEED", "1", 1);
    setenv("DS4_METAL_ENABLE_STREAMING_MADVISE_WILLNEED", "1", 1);
    CHECK(!ds4_test_stream_madvise_willneed_enabled());

    unsetenv("DS4_METAL_DISABLE_STREAMING_READAHEAD");
    unsetenv("DS4_METAL_DISABLE_STREAMING_MADVISE_WILLNEED");
    setenv("DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP", "123", 1);
    CHECK(ds4_test_stream_auto_preload_cap() == 123u);

    restore_env("DS4_METAL_DISABLE_STREAMING_READAHEAD", saved_read_disable);
    restore_env("DS4_METAL_ENABLE_STREAMING_READAHEAD", saved_read_enable);
    restore_env("DS4_METAL_DISABLE_STREAMING_MADVISE_WILLNEED", saved_madv_disable);
    restore_env("DS4_METAL_ENABLE_STREAMING_MADVISE_WILLNEED", saved_madv_enable);
    restore_env("DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP", saved_cap);
}

static void test_builtin_targets_and_sources(void) {
    const uint32_t pro_count = ds4_test_stream_builtin_count(1);
    const uint32_t flash_count = ds4_test_stream_builtin_count(0);
    const uint32_t glm_count = ds4_test_stream_builtin_count(2);
    CHECK(pro_count != 0);
    CHECK(flash_count != 0);
    CHECK(glm_count != 0);

    CHECK(ds4_test_stream_builtin_target(1, pro_count / 2u) == pro_count / 2u);
    CHECK(ds4_test_stream_builtin_target(1, pro_count + 1u) == pro_count);
    CHECK(ds4_test_stream_builtin_target(0, flash_count + 1u) == flash_count);
    CHECK(ds4_test_stream_builtin_load_count(1, pro_count + 1u) == pro_count);
    CHECK(ds4_test_stream_builtin_load_count(0, flash_count + 1u) == flash_count);

    /* A custom file remains a requested-entry source, while GLM's built-in
     * source intentionally refreshes even when the cache is already deep. */
    CHECK(ds4_test_stream_hotlist_should_skip(false, false,
                                              pro_count, pro_count + 1u, 1));
    CHECK(!ds4_test_stream_hotlist_should_skip(false, false,
                                               pro_count - 1u,
                                               pro_count + 1u,
                                               1));
    CHECK(!ds4_test_stream_hotlist_should_skip(true, false,
                                               UINT32_MAX, 7u, 1));
    CHECK(!ds4_test_stream_hotlist_should_skip(false, true,
                                               UINT32_MAX, 7u, 2));

    char path[] = "/tmp/ds4-hotlist-test-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) {
        const char text[] = "0 0 5\n1 1 4\n2 2 3\n";
        CHECK(write(fd, text, sizeof(text) - 1u) == (ssize_t)(sizeof(text) - 1u));
        CHECK(close(fd) == 0);
        CHECK(ds4_test_stream_hotlist_file_load_count(path, 99u) == 3u);
        CHECK(ds4_test_stream_hotlist_file_load_count(path, 2u) == 2u);
        CHECK(unlink(path) == 0);
    }
}

static void test_readahead_coalescing(void) {
    const uint64_t offsets[] = {0, 10, 15, 30, 40, 41, 90, 50, 70};
    const uint64_t ends[] =    {10, 20, 25, 40, 40, 45, 110, 60, 60};
    uint64_t out_offsets[8] = {0};
    uint64_t out_sizes[8] = {0};
    const uint32_t count = ds4_test_stream_readahead_coalesce(
            offsets, ends, (uint32_t)(sizeof(offsets) / sizeof(offsets[0])),
            out_offsets, out_sizes, 8u, 100u);
    CHECK(count == 4u);
    CHECK(out_offsets[0] == 0u && out_sizes[0] == 25u);
    CHECK(out_offsets[1] == 30u && out_sizes[1] == 10u);
    CHECK(out_offsets[2] == 41u && out_sizes[2] == 4u);
    CHECK(out_offsets[3] == 50u && out_sizes[3] == 10u);

    const uint64_t large_offsets[] = {UINT64_MAX - 16u, UINT64_MAX - 8u};
    const uint64_t large_ends[] = {UINT64_MAX - 8u, UINT64_MAX - 1u};
    CHECK(ds4_test_stream_readahead_coalesce(
                  large_offsets, large_ends, 2u,
                  out_offsets, out_sizes, 8u, UINT64_MAX) == 1u);
    CHECK(out_offsets[0] == UINT64_MAX - 16u && out_sizes[0] == 15u);

    CHECK(ds4_test_stream_readahead_coalesce(NULL, NULL, 0u,
                                             out_offsets, out_sizes, 8u, 100u) == 0u);
    CHECK(ds4_test_stream_readahead_coalesce(NULL, ends, 1u,
                                             out_offsets, out_sizes, 8u, 100u) == UINT32_MAX);
    CHECK(ds4_test_stream_readahead_coalesce(offsets, ends, 9u,
                                             NULL, NULL, 1u, 100u) == UINT32_MAX);
    CHECK(ds4_test_stream_readahead_coalesce(
                  offsets, ends, (uint32_t)(sizeof(offsets) / sizeof(offsets[0])),
                  out_offsets, out_sizes, 1u, 100u) == 4u);
    CHECK(out_offsets[0] == 0u && out_sizes[0] == 25u);
}

static int g_pread_calls;
static int g_pread_eintr_calls;
static size_t g_pread_max_return;

static ssize_t scripted_pread(int fd, void *buf, size_t count, off_t offset) {
    g_pread_calls++;
    if (g_pread_eintr_calls > 0) {
        g_pread_eintr_calls--;
        errno = EINTR;
        return -1;
    }
    const ssize_t n = pread(fd, buf, count, offset);
    if (n > 0 && g_pread_max_return != 0 && (size_t)n > g_pread_max_return) {
        return (ssize_t)g_pread_max_return;
    }
    return n;
}

static void test_pread_short_reads_and_eintr(void) {
    char path[] = "/tmp/ds4-pread-test-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd < 0) return;
    enum { DATA_BYTES = 8192 };
    uint8_t data[DATA_BYTES];
    for (uint32_t i = 0; i < DATA_BYTES; i++) data[i] = (uint8_t)(i * 13u + 7u);
    CHECK(write(fd, data, sizeof(data)) == (ssize_t)sizeof(data));
    uint8_t *scratch = malloc(ds4_test_stream_pread_chunk());
    CHECK(scratch != NULL);
    if (scratch) {
        g_pread_calls = 0;
        g_pread_eintr_calls = 2;
        g_pread_max_return = 97u;
        ds4_test_stream_set_pread_hook(scripted_pread);
        uint64_t read_bytes = 0;
        uint8_t sink = 0;
        CHECK(ds4_test_stream_pread_range_fd(fd,
                                             DATA_BYTES,
                                             0,
                                             DATA_BYTES,
                                             scratch,
                                             &read_bytes,
                                             &sink));
        ds4_test_stream_set_pread_hook(NULL);
        CHECK(read_bytes == DATA_BYTES);
        CHECK(g_pread_calls > 2);
        CHECK(sink != 0);
        CHECK(!ds4_test_stream_pread_range_fd(fd, DATA_BYTES,
                                              0, DATA_BYTES,
                                              NULL, NULL, NULL));
        CHECK(!ds4_test_stream_pread_range_fd(fd, DATA_BYTES,
                                              DATA_BYTES - 1u, 2u,
                                              scratch, NULL, NULL));
        free(scratch);
    }
    CHECK(close(fd) == 0);
    CHECK(unlink(path) == 0);
}

static void test_prepare_failure_cleanup(void) {
    uint32_t started = 0;
    uint32_t joined = 0;
    CHECK(ds4_test_stream_prepare_failure_cleanup(4u, 2u,
                                                  &started, &joined));
    CHECK(started == 2u);
    CHECK(joined == started);
    CHECK(ds4_test_stream_prepare_failure_cleanup(3u, 0u,
                                                  &started, &joined));
    CHECK(started == 0u);
    CHECK(joined == 0u);
}

int main(void) {
    test_platform_gates();
    test_builtin_targets_and_sources();
    test_readahead_coalescing();
    test_pread_short_reads_and_eintr();
    test_prepare_failure_cleanup();
    if (g_failures != 0) {
        fprintf(stderr, "ssd streaming hook tests: %u failure(s)\n", g_failures);
        return 1;
    }
    puts("ssd streaming hook tests: ok");
    return 0;
}
