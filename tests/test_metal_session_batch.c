/* Model-backed correctness oracle for native Metal session batching.
 *
 * Run with:
 *   LGN2_TEST_MODEL=/path/to/model.gguf make test-metal-session-batch
 */

#include "lgn2.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_SESSION_COUNT 16
#define DECODE_STEPS 6
#define MIXED_SUFFIX_TOKENS 8
#define TEST_CTX 512

static const char *prompts[MAX_SESSION_COUNT] = {
    "Write the integers from 1 to 80, separated by commas.",
    "Explain a binary search using one compact worked example.",
    "Give three concise reasons to test concurrent model sessions.",
    "Write a four-line description of merge sort.",
    "List five prime numbers and briefly define a prime number.",
    "Explain the difference between a stack and a queue in two sentences.",
    "Give a compact example of hexadecimal notation.",
    "Describe one invariant of a binary search tree.",
};

static void fail(const char *what, int session, int step) {
    fprintf(stderr, "FAIL: %s session=%d step=%d\n", what, session, step);
    exit(1);
}

static int session_count_from_env(void) {
    const char *value = getenv("LGN2_TEST_SESSION_COUNT");
    if (!value || !value[0]) return 2;
    char *end = NULL;
    long count = strtol(value, &end, 10);
    if (end == value || *end != '\0' || count < 2 ||
        count > MAX_SESSION_COUNT) {
        fprintf(stderr, "FAIL: invalid LGN2_TEST_SESSION_COUNT=%s\n", value);
        exit(1);
    }
    return (int)count;
}

static void archive_logits(lgn2_session *session, float *dst, int vocab,
                           int session_id, int step) {
    if (lgn2_session_copy_logits(session, dst, vocab) != vocab) {
        fail("copy logits", session_id, step);
    }
}

static void compare_logits(lgn2_session *session, const float *expected,
                           float *actual, int vocab, int expected_argmax,
                           int session_id, int step) {
    archive_logits(session, actual, vocab, session_id, step);
    float max_abs = 0.0f;
    int different = 0;
    int low_different = 0;
    int high_different = 0;
    for (int i = 0; i < vocab; i++) {
        if (memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
            different++;
            if (i < vocab / 2) low_different++;
            else high_different++;
        }
        float d = fabsf(actual[i] - expected[i]);
        if (!isfinite(d)) d = FLT_MAX;
        if (d > max_abs) max_abs = d;
    }
    int actual_argmax = lgn2_session_argmax(session);
    if (different != 0 || actual_argmax != expected_argmax) {
        fprintf(stderr,
                "FAIL: logits mismatch session=%d step=%d expected_top=%d "
                "actual_top=%d differing=%d low=%d high=%d max_abs=%g\n",
                session_id, step, expected_argmax, actual_argmax,
                different, low_different, high_different, max_abs);
        exit(1);
    }
}

int main(void) {
    const char *model = getenv("LGN2_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: LGN2_TEST_MODEL is not set\n");
        return 1;
    }
    setenv("LGN2_METAL_SESSION_BATCH_LOG", "1", 1);
    const int session_count = session_count_from_env();

    lgn2_engine_options opt = {
        .model_path = model,
        .n_threads = 1,
        .context_size = TEST_CTX,
    };
    lgn2_engine *engine = NULL;
    if (lgn2_engine_open(&engine, &opt) != 0) fail("engine open", -1, -1);

    lgn2_tokens prompt[MAX_SESSION_COUNT] = {0};
    lgn2_session *batched[MAX_SESSION_COUNT] = {0};
    char err[256] = {0};
    for (int i = 0; i < session_count; i++) {
        lgn2_encode_chat_prompt(engine, NULL, prompts[i % 8], LGN2_THINK_NONE,
                               &prompt[i]);
        if (lgn2_session_create(&batched[i], engine, TEST_CTX) != 0) {
            fail("session create", i, -1);
        }
        if (lgn2_session_sync(batched[i], &prompt[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: prefill session=%d: %s\n", i, err);
            return 1;
        }
    }

    const int vocab = lgn2_engine_vocab_size(engine);
    const size_t frontier_count =
        (size_t)session_count * (DECODE_STEPS + 1u);
    float *expected = malloc(frontier_count * (size_t)vocab * sizeof(float));
    float *actual = malloc((size_t)vocab * sizeof(float));
    int *argmax = malloc(frontier_count * sizeof(int));
    int generated[MAX_SESSION_COUNT][DECODE_STEPS];
    if (!expected || !actual || !argmax) fail("oracle allocation", -1, -1);

#define FRONTIER(step_, session_) \
    ((size_t)(step_) * (size_t)session_count + (size_t)(session_))
    for (int i = 0; i < session_count; i++) {
        size_t f = FRONTIER(0, i);
        archive_logits(batched[i], expected + f * (size_t)vocab,
                       vocab, i, 0);
        argmax[f] = lgn2_session_argmax(batched[i]);
    }

    for (int step = 0; step < DECODE_STEPS; step++) {
        lgn2_decode_item items[MAX_SESSION_COUNT];
        for (int row = 0; row < session_count; row++) {
            int i = (step & 1) ? session_count - 1 - row : row;
            int token = lgn2_session_argmax(batched[i]);
            generated[i][step] = token;
            items[row].session = batched[i];
            items[row].token = token;
        }
        if (lgn2_sessions_eval_batch(items, session_count,
                                    err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: batch step=%d: %s\n", step, err);
            return 1;
        }
        for (int i = 0; i < session_count; i++) {
            size_t f = FRONTIER(step + 1, i);
            archive_logits(batched[i], expected + f * (size_t)vocab,
                           vocab, i, step + 1);
            argmax[f] = lgn2_session_argmax(batched[i]);
        }
    }
    for (int i = 0; i < session_count; i++) {
        lgn2_session_free(batched[i]);
    }

    lgn2_tokens mixed_prompt = {0};
    lgn2_tokens suffix = {0};
    lgn2_tokens_copy(&mixed_prompt, &prompt[0]);
    lgn2_tokenize_text(engine,
                      " Continue with a concise verification example and conclusion.",
                      &suffix);
    if (suffix.len < MIXED_SUFFIX_TOKENS) {
        fail("mixed suffix tokenization", -1, -1);
    }
    for (int i = 0; i < MIXED_SUFFIX_TOKENS; i++) {
        lgn2_tokens_push(&mixed_prompt, suffix.v[i]);
    }

    lgn2_session *mixed_prefill = NULL;
    lgn2_session *mixed_decode[MAX_SESSION_COUNT] = {0};
    float *mixed_expected = malloc(
            (size_t)(session_count + 1) * (size_t)vocab * sizeof(float));
    int mixed_argmax[MAX_SESSION_COUNT + 1];
    if (!mixed_expected) fail("mixed oracle allocation", -1, -1);
    if (lgn2_session_create(&mixed_prefill, engine, TEST_CTX) != 0) {
        fail("mixed prefill create", -1, -1);
    }
    if (lgn2_session_sync(mixed_prefill, &prompt[0], err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: mixed base prefill: %s\n", err);
        return 1;
    }
    lgn2_decode_item mixed_items[MAX_SESSION_COUNT];
    for (int i = 0; i < session_count; i++) {
        if (lgn2_session_create(&mixed_decode[i], engine, TEST_CTX) != 0) {
            fail("mixed decode create", i, -1);
        }
        if (lgn2_session_sync(mixed_decode[i], &prompt[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: mixed decode prefill session=%d: %s\n", i, err);
            return 1;
        }
        mixed_items[i].session = mixed_decode[i];
        mixed_items[i].token = lgn2_session_argmax(mixed_decode[i]);
    }
    if (lgn2_sessions_eval_batch_with_prefill(
                mixed_items, session_count, mixed_prefill, &mixed_prompt,
                err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: Metal mixed batch: %s\n", err);
        return 1;
    }

    archive_logits(mixed_prefill, mixed_expected, vocab, -1, -1);
    mixed_argmax[0] = lgn2_session_argmax(mixed_prefill);
    for (int i = 0; i < session_count; i++) {
        archive_logits(mixed_decode[i],
                       mixed_expected + (size_t)(i + 1) * (size_t)vocab,
                       vocab, i, -1);
        mixed_argmax[i + 1] = lgn2_session_argmax(mixed_decode[i]);
    }

    lgn2_session *mixed_control = NULL;
    if (lgn2_session_create(&mixed_control, engine, TEST_CTX) != 0) {
        fail("mixed prefill control create", -1, -1);
    }
    if (lgn2_session_sync(mixed_control, &prompt[0], err, sizeof(err)) != 0 ||
        lgn2_session_sync(mixed_control, &mixed_prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: mixed prefill control: %s\n", err);
        return 1;
    }
    compare_logits(mixed_control, mixed_expected, actual, vocab,
                   mixed_argmax[0],
                   -1, -1);
    if (lgn2_session_pos(mixed_prefill) != mixed_prompt.len ||
        lgn2_session_pos(mixed_control) != mixed_prompt.len) {
        fail("mixed prefill checkpoint", -1, -1);
    }
    lgn2_session_free(mixed_control);

    for (int i = 0; i < session_count; i++) {
        mixed_control = NULL;
        if (lgn2_session_create(&mixed_control, engine, TEST_CTX) != 0) {
            fail("mixed decode control create", i, -1);
        }
        if (lgn2_session_sync(mixed_control, &prompt[i], err, sizeof(err)) != 0 ||
            lgn2_session_eval(mixed_control, mixed_items[i].token,
                             err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: mixed decode control session=%d: %s\n", i, err);
            return 1;
        }
        compare_logits(mixed_control,
                       mixed_expected + (size_t)(i + 1) * (size_t)vocab,
                       actual, vocab, mixed_argmax[i + 1], i, -1);
        if (lgn2_session_pos(mixed_decode[i]) != prompt[i].len + 1 ||
            lgn2_session_pos(mixed_control) != prompt[i].len + 1) {
            fail("mixed decode checkpoint", i, -1);
        }
        lgn2_session_free(mixed_control);
        lgn2_session_free(mixed_decode[i]);
    }
    lgn2_session_free(mixed_prefill);
    free(mixed_expected);
    lgn2_tokens_free(&suffix);
    lgn2_tokens_free(&mixed_prompt);

    for (int i = 0; i < session_count; i++) {
        lgn2_session *control = NULL;
        if (lgn2_session_create(&control, engine, TEST_CTX) != 0) {
            fail("control create", i, -1);
        }
        if (lgn2_session_sync(control, &prompt[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: control prefill session=%d: %s\n", i, err);
            return 1;
        }
        for (int step = 0; step <= DECODE_STEPS; step++) {
            size_t f = FRONTIER(step, i);
            compare_logits(control, expected + f * (size_t)vocab,
                           actual, vocab, argmax[f], i, step);
            if (step < DECODE_STEPS &&
                lgn2_session_eval(control, generated[i][step],
                                 err, sizeof(err)) != 0) {
                fprintf(stderr,
                        "FAIL: control eval session=%d step=%d: %s\n",
                        i, step, err);
                return 1;
            }
        }
        lgn2_session_free(control);
        lgn2_tokens_free(&prompt[i]);
    }

    free(argmax);
    free(actual);
    free(expected);
    lgn2_engine_close(engine);
    fprintf(stderr,
            "test_metal_session_batch PASS sessions=%d steps=%d mixed_suffix=%d exact_logits=1\n",
            session_count, DECODE_STEPS, MIXED_SUFFIX_TOKENS);
    return 0;
#undef FRONTIER
}
