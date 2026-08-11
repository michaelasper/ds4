# LagoonNebula

Run a verified Laguna S2.1 model locally on Apple silicon with one coherent
CLI, HTTP, session, and evaluation workflow.

LagoonNebula is the product name and **`lgn2`** is the staged repository,
tool, API, environment, and local-state namespace. Local state lives under
`~/.lgn2`. This branch applies the clean break: old executable, API,
environment, cache, lock-file, and model-link names are not aliases for the
current interface.

[![License: MIT][license-shield]][license-url]

If a reproducible local model workflow matters more than a hosted service, this
checkout is for you. It deliberately targets Laguna S2.1 on Apple Metal and
keeps the runtime, server, benchmark, and quality tools on the same product
surface.

## Why LagoonNebula

Local model experiments often drift when every tool uses a different model
file, prompt path, or runtime configuration. LagoonNebula pins the model identity
and gives interactive generation, HTTP serving, session reuse, evaluation, and
benchmarking one documented path.

## Scope

This README documents one supported product: a Laguna S2.1 GGUF model (the
portable model-file format) running on Apple Metal with the whole model
memory-mapped. The supported commands are `lgn2`, `lgn2-server`, `lgn2-bench`,
and `lgn2-eval`; the public C/API namespace is `lgn2_*` and runtime
environment variables use `LGN2_*`. Unsupported model architectures, flags,
and execution modes are rejected rather than selecting an undocumented
fallback or compatibility alias.

## Features

- Interactive or one-shot generation from a local, hash-verified model.
- One HTTP server for chat completions, Responses, Anthropic messages,
  completions, streaming, tools, and resident session batching.
- Reusable prompt and session state through disk-backed KV (key/value)
  checkpoints.
- Optional Laguna DFlash speculative decoding support with explicit draft and
  probability controls.
- `lgn2-bench`, `lgn2-eval`, and the Laguna quality scorer for repeatable local
  experiments without promising a particular speed or memory result.

## Prerequisites

- Apple silicon running macOS with Metal support.
- Xcode Command Line Tools, `make`, Python 3.11+, `curl`, and `shasum`.
- Enough local storage for the pinned model and any generated cache files.

No memory or performance minimum is asserted here; measure the exact model and
machine with the supplied tools.

## Quick start

From a clean checkout:

```sh
make -j8
./download_model.sh laguna-q2-q3
./lgn2 --metal \
  -m gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf \
  -p "Explain this repository in one paragraph."
```

The model downloader resumes interrupted downloads and verifies the pinned hash.
The only convenience model link is `lgn2.gguf`; current tools do not consult
legacy model-link names.

## Install and build

The project is built from source, with an optional `make install` for the four
executables and their runtime Metal sources. The source rename is staged on
`refactor/laguna-metal-only`, but the GitHub
repository remains `michaelasper/ds4` until the final remote rename. Clone that
current URL and branch now; update the remote and clone URL only as part of the
final repository operation.

```sh
git clone --branch refactor/laguna-metal-only --single-branch \
  https://github.com/michaelasper/ds4.git LagoonNebula
cd LagoonNebula
make -j8
```

`make` builds the four executables: `lgn2`, `lgn2-server`, `lgn2-bench`, and
`lgn2-eval`. Run `make clean` before a clean rebuild when changing source or
compiler settings.

Each executable accepts the standalone `--version` option. It prints the
LagoonNebula product name, the invoked executable, the literal `development`
release label, and the 12-hex committed Git revision. Builds from source
archives report `unknown` for the revision; `-V` and other legacy aliases are
not accepted.

To stage or install the standalone runtime without installing a model:

```sh
make install PREFIX="$PWD/.local"
make uninstall PREFIX="$PWD/.local"
```

The installed executables discover Metal sources under the corresponding
`share/lgn2/metal` directory even when launched from another working directory.

## Pinned model download

`download_model.sh` accepts the `laguna-q2-q3` alias and stores the verified
file under `LGN2_GGUF_DIR` (default: `./gguf`). The current model identity is:

| File | SHA-256 |
| --- | --- |
| `laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf` | `61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903` |

The pinned file is available from the [Laguna S2.1 GGUF download][model-url].
To select a different output directory and verify the result explicitly:

```sh
export LGN2_GGUF_DIR="$PWD/gguf"
./download_model.sh laguna-q2-q3
MODEL="$LGN2_GGUF_DIR/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf"
shasum -a 256 "$MODEL"
```

## CLI

The `lgn2` command supports one-shot prompts, prompt files, and an interactive
session:

```sh
MODEL=gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf

./lgn2 --metal -m "$MODEL" -p "Summarize the model card."
./lgn2 --metal -m "$MODEL" --prompt-file prompt.txt --nothink
./lgn2 --metal -m "$MODEL"
```

Useful common options are:

| Option | Purpose |
| --- | --- |
| `-m, --model FILE` | Select the Laguna GGUF file. |
| `--metal` | Select the Apple Metal runtime explicitly. |
| `-c, --ctx N` | Set the context size. |
| `-p, --prompt TEXT` / `--prompt-file FILE` | Supply one-shot input. |
| `-n, --tokens N` | Set the generation limit. |
| `--temp`, `--top-k`, `--top-p`, `--min-p`, `--seed` | Control sampling. |
| `--think`, `--think-max`, `--nothink` | Select thinking behavior. |
| `--system TEXT` | Set a system message. |
| `--dflash FILE` | Load an optional DFlash support file. |
| `--quality` / `--warm-weights` | Select the corresponding runtime modes. |

In an interactive session, `/help`, `/think`, `/think-max`, `/nothink`,
`/ctx N`, `/read FILE`, `/quit`, and `/exit` are available. Use
`./lgn2 --help all` for diagnostics, logits/probability dumps, inspection, and
the complete option limits.

## DFlash

DFlash is an optional Laguna speculative-decoding path. Pass a compatible
support GGUF with
`--dflash FILE`; without it, the regular decode path remains available. The
draft count accepts `1` through `15`, and `--dflash-p-min` accepts `0` through
`1` (`0` selects fixed-width behavior):

```sh
MODEL=gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf

./lgn2 --metal -m "$MODEL" \
  --dflash /path/to/compatible-dflash.gguf \
  --dflash-draft 3 \
  --dflash-p-min 0.4 \
  -p "Give a short answer."
```

Treat DFlash settings as an experiment and compare them with a regular run;
this README makes no acceptance-rate or speed claim.

## HTTP server

`lgn2-server` exposes the loaded model on a local HTTP listener. The defaults
are `127.0.0.1:8000`:

```sh
MODEL=gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf
./lgn2-server --metal -m "$MODEL" --host 127.0.0.1 --port 8000
```

Available endpoints are:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

Discover the current model identifier, then send an OpenAI-compatible request:

```sh
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"laguna-s-2.1","messages":[{"role":"user","content":"Say hello."}]}'
```

Requests can use streamed responses and tool schemas where the selected
protocol supports them. Server controls include `--cors`, `--trace FILE`,
`--batched-session N`, and `--mixed-prefill-quantum N`. Run
`./lgn2-server --help all` for request, thinking, and cache policy details.

### Disk KV and sessions

Enable a disk-backed KV directory when prompt or session state should be
checkpointed outside the process:

```sh
./lgn2-server --metal -m "$MODEL" \
  --kv-disk-dir "$HOME/.lgn2/server-kv" \
  --kv-disk-space-mb 8192
```

The server also exposes cache-age, continuation, compatibility, and tool-memory
policies. Keep those defaults unless an experiment requires a documented
change; `./lgn2-server --help all` is the authoritative option reference.

## Benchmark, evaluation, and quality

`lgn2-bench` measures context-growth frontiers and generation into CSV. This is
an exploratory invocation:

```sh
MODEL=gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf

./lgn2-bench \
  --metal \
  -m "$MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/lgn2-speed.csv
```

Use the [benchmark runbook](BENCHMARK.md) for official comparisons, fixed
frontiers, parity checks, and interpretation. `BENCHMARK.md` and every file
under `benchmark/` are frozen historical pre-fork protocols. Their old `ds4`/
`DS4_*` commands and environment names are not current LagoonNebula interfaces;
do not copy them into new runs. A CSV from an ad-hoc run is not a published
performance result.

`lgn2-eval` provides local reasoning, math, science, and security cases:

```sh
./lgn2-eval --self-test-extractors
./lgn2-eval --metal -m "$MODEL" --questions 10 --plain
```

For deterministic continuation quality, build the scorer and compare the
tracked Laguna fixture:

```sh
make quality/score_official
quality/score_official \
  OLD.gguf \
  quality/data/laguna-openrouter-100/manifest.tsv \
  /tmp/old.tsv 4096 --quality
quality/score_official \
  NEW.gguf \
  quality/data/laguna-openrouter-100/manifest.tsv \
  /tmp/new.tsv 4096 --quality
python3 quality/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

The [quality guide](quality/README.md) explains the
fixture and score columns without turning a single run into a general claim.

## Build and test

Run the source and model-independent checks on supported Apple hardware:

```sh
git diff --check
make clean
make -j8 test
make check-metal-sources
make test-installed-resources
make test-laguna-cli-options
./lgn2_test --laguna-architecture
./lgn2_test --laguna-selector-parser
./lgn2_test --laguna-metal-core
./lgn2_test --server
```

For a model-backed integration gate, provide the exact model explicitly:

```sh
make test-metal-laguna-integration \
  LGN2_TEST_MODEL=/absolute/path/to/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for review evidence and focused test
guidance, and [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md) for the release
checklist.

## Support boundary

The supported product is Laguna S2.1 GGUF inference on Apple Metal, using the
whole model through a memory-mapped file. The CLI, HTTP server, session/KV
serialization, batching, sampling, and DFlash path are retained product
surfaces. Other model families, hardware paths, and execution topologies are
outside this release; do not infer compatibility from files in the repository
that are not part of the current lgn2 interface.

## Attribution and license

LagoonNebula is released under the [MIT License](LICENSE). It exists thanks to
the [llama.cpp](https://github.com/ggml-org/llama.cpp) and
[GGML](https://github.com/ggerganov/ggml) projects. The runtime and tools build
on that ecosystem's code, kernels, quantization formats, GGUF conventions, and
engineering; retained upstream notices and license terms must remain with any
redistribution. See [LICENSE](LICENSE) and the source headers for the complete
notices.

## AI disclosure

This repository has been developed with substantial assistance from AI coding
tools. Humans lead the design, review, testing, and release decisions.

## Further reading

- [Model card](MODEL_CARD.md): pinned file identity, checksum, and provenance.
- [Benchmark runbook](BENCHMARK.md): the reproducible comparison protocol.
- [Contributing guide](CONTRIBUTING.md): development and review expectations.
- [Release QA](QA_BEFORE_RELEASES.md): pre-release gates and evidence.
- [Refactor boundary](FORK.md): current scope and namespace decisions.
- [License](LICENSE): copyright and upstream attribution terms.

---

Crafted with [Readme Craft](https://github.com/motiful/readme-craft).

[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: LICENSE
[model-url]: https://huggingface.co/antirez/Laguna-S-2.1-GGUF/resolve/main/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf
