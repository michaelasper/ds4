# QA before releases

This checklist is the release gate for the Laguna S2.1 Apple Metal fork. The
supported product is deliberately narrow:

- model family: Laguna S2.1;
- host: Apple silicon running macOS;
- inference backend: Metal;
- optional speculative path: a compatible Laguna DFlash support GGUF.

CPU inference, CUDA, ROCm, multi-GPU placement, distributed and tensor-parallel
inference, SSD expert streaming, MTP, DSpark, DeepSeek, and GLM are not release
targets. Do not revive their tests or describe them as supported to make a
release gate pass.

Record the commit, macOS/Xcode versions, hardware, GGUF filename and SHA-256,
context size, and non-default selectors for every model-backed run. Never run
two large model processes concurrently during correctness or performance work.

## 1. Repository and build sanity

Start from the exact release commit in a clean checkout:

```zsh
git status --short --branch
git diff --check
git rev-parse HEAD
make clean
make -j8
```

The build must fail clearly on a non-Darwin host. All runtime Metal source files
and explicit source overrides must pass the fail-closed source check:

```zsh
make check-metal-sources
```

Review compiler output. Existing SDK deprecation diagnostics must be recorded;
new warnings are release blockers unless their cause and narrow suppression are
reviewed in the release commit.

Check that the public programs render their help and expose only Laguna/Metal
features:

```zsh
./ds4 --help all
./ds4-server --help all
./ds4-bench --help all
./ds4-eval --help all
```

## 2. Model-independent tests

Run the default Laguna suite from a clean build:

```zsh
make clean
make -j8 test
```

After parser, help, or frontend changes, run the public option contract
explicitly:

```zsh
make test-laguna-cli-options
./ds4-eval --self-test-extractors
```

After Metal kernel, graph, quantization, or scheduling changes, also run the
focused targets that cover the changed code. At minimum:

```zsh
make test-metal-laguna
make test-mxfp4-metal
```

Do not substitute `test-legacy` for the default Laguna suite. It is a temporary
transition aid and may exercise code scheduled for deletion.

## 3. Model-backed integration

Use the release Laguna S2.1 GGUF and record its SHA-256. The integration target
must inspect the model and exercise the session API. Unsupported-architecture
rejection remains part of the model-independent `make test` gate.

```zsh
export LAGUNA_TEST_MODEL=/absolute/path/to/laguna-s2.1.gguf
shasum -a 256 "$LAGUNA_TEST_MODEL"
make test-metal-laguna-integration LAGUNA_TEST_MODEL="$LAGUNA_TEST_MODEL"
```

For a release-affecting graph or kernel change, run at least one deterministic
raw CLI prompt and one session/server prompt with the same input. Require the
expected route diagnostics, valid text, and matching greedy output under the
change's declared parity contract. Exercise both short decode and a prompt long
enough to enter the production prefill route.

When DFlash changes, run the same deterministic prompt with DFlash disabled and
enabled. Record acceptance, fallbacks, verifier failures, output parity, and
throughput. A support/target compatibility failure must occur before session or
KV mutation.

## 4. Server and persistence

Run the built-in server tests whenever HTTP parsing, SSE, prompt rendering,
tool calls, sampling, batching, cancellation, or disk KV code changes:

```zsh
./ds4_test --server
```

Then start `ds4-server` with the release model and smoke-test every advertised
API family:

- `GET /v1/models`;
- OpenAI chat completions, streaming and non-streaming;
- Responses, streaming and non-streaming;
- Anthropic messages, streaming and non-streaming;
- cancellation followed by a successful request;
- two resident sessions when batching code changed;
- disk-KV save, load, repeated load, and corrupt-checkpoint rejection when
  persistence code changed.

Malformed requests must fail without terminating the server or contaminating a
following valid request. Thinking text and tool-call protocol markers must not
leak into final answer content.

## 5. Quality and performance

Run the official Laguna continuation fixture with the release model whenever
tokenization, prompt rendering, sampling, logits, quantization, or graph math
changes. Preserve the raw TSV, command, revision, model hash, and summary:

```zsh
make gguf-tools/quality-testing/score_official
gguf-tools/quality-testing/score_official \
  "$LAGUNA_TEST_MODEL" \
  gguf-tools/quality-testing/data/laguna-openrouter-100/manifest.tsv \
  /tmp/laguna-quality.tsv 4096 --quality
```

Performance claims require a committed protocol. For the current M5 Max
confirmation study, `BENCHMARK.md` is the complete and authoritative handoff;
run its committed driver without reconstructing commands or changing its
selectors. Never use an informal timing run as release evidence.

Compare against the declared baseline on the same machine and power state.
Report prefill and steady decode separately, retain rejected arms, and never
promote a selector that misses correctness, route, state, or statistical gates.

## 6. Release hygiene

Before tagging:

1. rerun `git diff --check`, `make clean && make -j8 test`, and the applicable
   model-backed gates;
2. verify the checkout and subcommands use the intended commit and clean tree;
3. confirm help, README, and release notes describe only currently supported
   behavior;
4. inspect packaged file names, tar headers, owners, extended attributes, and
   text contents for local usernames, paths, host identifiers, credentials,
   serials, UUIDs, and other private metadata;
5. generate a relative checksum manifest that covers every regular archived
   file, and verify it after extracting into a new temporary directory;
6. publish immutable command logs, revision and model hashes, test status, and
   the exact benchmark/runbook revision supporting every performance claim.

An unavailable optional model or machine is not a passing result. Mark the gate
not run, explain why, and do not make the corresponding compatibility or
performance claim.
