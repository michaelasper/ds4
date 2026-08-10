<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo.svg">
    <source media="(prefers-color-scheme: light)" srcset="logo.svg">
    <img src="logo.svg" alt="DwarfStar logo" width="220">
  </picture>
</p>

# DwarfStar

Run a verified Laguna S2.1 model locally on Apple silicon with one coherent
CLI, HTTP, session, and evaluation workflow.

[![License: MIT][license-shield]][license-url]

If a reproducible local model workflow matters more than a hosted service, this
checkout is for you. It deliberately targets Laguna S2.1 on Apple Metal and
keeps the runtime, server, benchmark, and quality tools on the same product
surface.

## Why DwarfStar

Local model experiments often drift when every tool uses a different model
file, prompt path, or runtime configuration. DwarfStar pins the model identity
and gives interactive generation, HTTP serving, session reuse, evaluation, and
benchmarking one documented path.

## Scope

This README documents one supported product: a Laguna S2.1 GGUF model (the
portable model-file format) running on Apple Metal with the whole model
memory-mapped. The public names `ds4`,
`ds4-server`, `ds4-bench`, and `ds4-eval` are retained temporarily; the rename
is intentionally deferred. Unsupported model architectures, flags, and
execution modes are rejected rather than selecting an undocumented fallback.

## Features

- Interactive or one-shot generation from a local, hash-verified model.
- One HTTP server for chat completions, Responses, Anthropic messages,
  completions, streaming, tools, and resident session batching.
- Reusable prompt and session state through disk-backed KV (key/value)
  checkpoints.
- Optional Laguna DFlash speculative decoding support with explicit draft and
  probability controls.
- `ds4-bench`, `ds4-eval`, and the Laguna quality scorer for repeatable local
  experiments without promising a particular speed or memory result.

## Prerequisites

- Apple silicon running macOS with Metal support.
- Xcode Command Line Tools, `make`, `curl`, and `shasum`.
- Enough local storage for the pinned model and any generated cache files.

No memory or performance minimum is asserted here; measure the exact model and
machine with the supplied tools.

## Quick start

From a clean checkout:

```sh
make -j8
./download_model.sh laguna-q2-q3
./ds4 --metal \
  -m gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf \
  -p "Explain this repository in one paragraph."
```

The model downloader resumes interrupted downloads, verifies the pinned hash,
and leaves an existing model file or `ds4flash.gguf` link untouched.

## Install and build

The project is built from source; there is no package-manager install step.

```sh
git clone https://github.com/michaelasper/ds4.git
cd ds4
make -j8
```

`make` builds the four retained executables: `ds4`, `ds4-server`, `ds4-bench`,
and `ds4-eval`. Run `make clean` before a clean rebuild when changing source
or compiler settings.

## Pinned model download

`download_model.sh` accepts the `laguna-q2-q3` alias and stores the verified
file under `DS4_GGUF_DIR` (default: `./gguf`). The current model identity is:

| File | SHA-256 |
| --- | --- |
| `laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf` | `61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903` |

The pinned file is available from the [Laguna S2.1 GGUF download][model-url].
To select a different output directory and verify the result explicitly:

```sh
export DS4_GGUF_DIR="$PWD/gguf"
./download_model.sh laguna-q2-q3
MODEL="$DS4_GGUF_DIR/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf"
shasum -a 256 "$MODEL"
```

## CLI

The `ds4` command supports one-shot prompts, prompt files, and an interactive
session:

```sh
MODEL=gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf

./ds4 --metal -m "$MODEL" -p "Summarize the model card."
./ds4 --metal -m "$MODEL" --prompt-file prompt.txt --nothink
./ds4 --metal -m "$MODEL"
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
`./ds4 --help all` for diagnostics, logits/probability dumps, inspection, and
the complete option limits.

## DFlash

DFlash is an optional Laguna speculative-decoding path. Pass a compatible
support GGUF with
`--dflash FILE`; without it, the regular decode path remains available. The
draft count accepts `1` through `15`, and `--dflash-p-min` accepts `0` through
`1` (`0` selects fixed-width behavior):

```sh
MODEL=gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf

./ds4 --metal -m "$MODEL" \
  --dflash /path/to/compatible-dflash.gguf \
  --dflash-draft 3 \
  --dflash-p-min 0.4 \
  -p "Give a short answer."
```

Treat DFlash settings as an experiment and compare them with a regular run;
this README makes no acceptance-rate or speed claim.

## HTTP server

`ds4-server` exposes the loaded model on a local HTTP listener. The defaults
are `127.0.0.1:8000`:

```sh
MODEL=gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf
./ds4-server --metal -m "$MODEL" --host 127.0.0.1 --port 8000
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
`./ds4-server --help all` for request, thinking, and cache policy details.

### Disk KV and sessions

Enable a disk-backed KV directory when prompt or session state should be
checkpointed outside the process:

```sh
./ds4-server --metal -m "$MODEL" \
  --kv-disk-dir "$HOME/.ds4/server-kv" \
  --kv-disk-space-mb 8192
```

The server also exposes cache-age, continuation, compatibility, and tool-memory
policies. Keep those defaults unless an experiment requires a documented
change; `./ds4-server --help all` is the authoritative option reference.

## Benchmark, evaluation, and quality

`ds4-bench` measures context-growth frontiers and generation into CSV. This is
an exploratory invocation:

```sh
MODEL=gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf

./ds4-bench \
  --metal \
  -m "$MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/laguna-speed.csv
```

Use the [benchmark runbook](BENCHMARK.md) for official comparisons, fixed
frontiers, parity checks, and interpretation. A CSV from an ad-hoc run is not
a published performance result.

`ds4-eval` provides local reasoning, math, science, and security cases:

```sh
./ds4-eval --self-test-extractors
./ds4-eval --metal -m "$MODEL" --questions 10 --plain
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
make test-laguna-cli-options
./ds4_test --laguna-architecture
./ds4_test --laguna-selector-parser
./ds4_test --laguna-metal-core
./ds4_test --server
```

For a model-backed integration gate, provide the exact model explicitly:

```sh
make test-metal-laguna-integration \
  LAGUNA_TEST_MODEL=/absolute/path/to/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for review evidence and focused test
guidance, and [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md) for the release
checklist.

## Support boundary

The supported product is Laguna S2.1 GGUF inference on Apple Metal, using the
whole model through a memory-mapped file. The CLI, HTTP server, session/KV
serialization, batching, sampling, and DFlash path are retained product
surfaces. Other model families, hardware paths, and execution topologies are
outside this release; do not infer compatibility from the temporary `ds4*`
names or from files in the repository that are not linked here.

## Attribution and license

DwarfStar is released under the [MIT License](LICENSE). It exists thanks to
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
- [Refactor boundary](FORK.md): current scope and deferred naming decisions.
- [License](LICENSE): copyright and upstream attribution terms.

---

Crafted with [Readme Craft](https://github.com/motiful/readme-craft).

[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: LICENSE
[model-url]: https://huggingface.co/antirez/Laguna-S-2.1-GGUF/resolve/main/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf
