# Agent Notes

These notes are authoritative for the Laguna Metal-only refactor. The product
target is Laguna S2.1 GGUF inference on Apple Metal; the historical ds4
multi-model and multi-backend paths are cleanup work, not compatibility
requirements.

## Product boundary

- Support Laguna S2.1 GGUF only. Model loading must require the Laguna
  architecture instead of falling back to DeepSeek or GLM metadata.
- Support Apple Metal only. The production path is whole-model,
  mmap-backed Metal inference; do not add eager copies or SSD expert
  streaming.
- Retain the CLI and server as supported product surfaces.
- Retain DFlash as an explicitly Laguna-specific optional path for now.
- Make no product promise for CPU, CUDA, ROCm, SSD streaming, distributed
  inference, tensor parallelism, multi-GPU placement, MTP, DSpark, steering,
  power controls, or custom prefill. Do not preserve these paths with new
  compatibility flags.
- Defer the ds4-to-Laguna rename until the implementation and documentation
  cleanup is complete. Existing names, cache paths, payload identifiers, and
  public symbols are temporary compatibility surfaces until that decision is
  recorded in `FORK.md`.

## Implementation rules

- Keep one canonical Laguna Metal execution path. Prefer deleting obsolete
  branches over adding another semantic variant behind a flag.
- Keep model loading mmap-backed and whole-model. Do not reintroduce an SSD,
  distributed, or host-inference fallback while simplifying the runtime.
- Keep Objective-C limited to the Metal runtime and use C for the rest of the
  implementation. Do not introduce C++.
- Keep public APIs narrow. CLI and server code should not know tensor or
  pipeline internals.
- Preserve correctness before speed. Any optimisation must retain attention,
  KV-cache, logits, sampling, and session correctness on the Laguna path.
- Retain DFlash only where it is exercised by Laguna and covered by a test;
  do not use it as a reason to retain unrelated DeepSeek, GLM, or backend
  compatibility code.
- Comments should explain non-obvious model mechanics, cache lifetime,
  memory policy, or API orchestration beside the implementation.

## Repository layout

- `ds4.c`: model loading, tokenizer, Metal graph scheduling, sessions, and
  disk-cache payload serialisation.
- `ds4_cli.c`: command line and interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic-compatible HTTP API, worker queue,
  streaming, tool-call mapping, and server-side KV-cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: Metal compute kernels.
- `tests/`: unit and live integration tests.
- `FORK.md`: refactor boundary, deletion order, guardrails, and deferred
  compatibility decisions.

## Testing

Use `make` for build validation and `make test` for the model-independent
Laguna Metal suite on Apple hardware. Run the model-backed gate explicitly as
`make test-metal-laguna-integration LAGUNA_TEST_MODEL=/absolute/model.gguf`;
it must never fall back to a default fixture or skip a missing model. At each
major refactor boundary, verify:

1. A valid Laguna S2.1 model loads through the whole-model mmap Metal path.
2. Non-Laguna architectures and unsupported backend/mode options are rejected
   clearly rather than selecting a legacy fallback.
3. CLI generation, server streaming, sessions, batching, and sampling remain
   correct.
4. The retained DFlash path passes its focused regression coverage.

Do not add new CPU, CUDA, ROCm, SSD, distributed, tensor-parallel, multi-GPU,
MTP, DSpark, steering, power, or custom-prefill test obligations. Keep the
frozen `laguna-s2.1` benchmark commit unchanged while this work proceeds on the
refactor branch; compare against it rather than changing it.

## Safety

- Do not run multiple huge model processes concurrently; the instance lock is
  intentional.
- Do not delete a Metal source or compatibility layer until its callers,
  runtime source list, build target, and focused test coverage have been
  removed or updated together.
- Do not commit or push refactor work unless explicitly requested.
