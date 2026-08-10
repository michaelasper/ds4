# Contributing

This fork is intentionally specialized for Laguna S2.1 inference on Apple
Metal. Changes should make that product smaller, clearer, more correct, or
faster. CPU inference, CUDA, ROCm, DeepSeek, GLM, distributed execution,
tensor parallelism, multi-GPU placement, SSD expert streaming, MTP, and DSpark
are outside the supported boundary.

Include the commands you ran, the exact commit, macOS and hardware, model
filename and SHA-256, and any non-default selectors in a pull request or commit
handoff. Do not hide failed or skipped gates.

## Build and model-independent tests

Start with a clean Apple-silicon checkout:

```zsh
git diff --check
make clean
make -j8 test
```

`make test` is the default model-independent Laguna/Metal suite. Useful focused
checks include:

```zsh
make check-metal-sources
make test-laguna-cli-options
./ds4_test --laguna-architecture
./ds4_test --laguna-selector-parser
./ds4_test --laguna-metal-core
./ds4_test --server
./ds4-eval --self-test-extractors
```

Add or tighten a focused test whenever a bug could otherwise return silently to
an old route. An optimization test must prove both numerical behavior and that
the intended production path actually executed. Explicit selectors should fail
before graph, command-buffer, session, or KV mutation when unavailable or
malformed.

## Model-backed checks

Use the Laguna S2.1 GGUF affected by the change:

```zsh
export LAGUNA_TEST_MODEL=/absolute/path/to/laguna-s2.1.gguf
shasum -a 256 "$LAGUNA_TEST_MODEL"
make test-metal-laguna-integration LAGUNA_TEST_MODEL="$LAGUNA_TEST_MODEL"
```

For graph, kernel, tokenizer, template, or sampling changes, also run a fixed
greedy prompt through the raw CLI and session/server paths. Exercise both a
single-token decode and production prefill. Record route diagnostics and the
declared parity result.

For DFlash changes, compare enabled and disabled runs with a compatible support
GGUF. Record accepted tokens, fallbacks, verifier failures, output parity, and
throughput.

## Quality checks

Build the Metal quality scorer and use the Laguna fixture:

```zsh
make gguf-tools/quality-testing/score_official

gguf-tools/quality-testing/score_official OLD.gguf \
  gguf-tools/quality-testing/data/laguna-openrouter-100/manifest.tsv \
  /tmp/old.tsv 4096 --quality

gguf-tools/quality-testing/score_official NEW.gguf \
  gguf-tools/quality-testing/data/laguna-openrouter-100/manifest.tsv \
  /tmp/new.tsv 4096 --quality

python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

Lower average NLL is better, but inspect first-token, top-logprob, and ordering
metrics too. Compare the same model checkpoint, manifest, machine, and options.

## Performance checks

Use `ds4-bench` for exploratory throughput work:

```zsh
./ds4-bench \
  --metal \
  -m "$LAGUNA_TEST_MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/laguna-speed.csv
```

Compare prefill and steady decode separately on the same machine, model,
context frontiers, power/thermal state, and background load. Preserve raw CSVs
and commands. Performance claims or selector promotion require a committed,
reviewed protocol; use `BENCHMARK.md` when it covers the experiment rather than
inventing an ad-hoc gate.

## Review expectations

- Keep default-off experiments isolated and fail closed when requested.
- Do not add compatibility abstractions for unsupported models or backends.
- Prefer small commits that delete one obsolete surface or establish one clear
  Laguna boundary.
- Preserve attribution and licenses when deleting inherited implementation
  files.
- Run the full default suite after integration, not only a focused test.
- Follow `QA_BEFORE_RELEASES.md` before tagging or publishing artifacts.
