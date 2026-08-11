# LagoonNebula Agent Notes

These notes are authoritative for the LagoonNebula Laguna Metal-only refactor.
The product target is Laguna S2.1 GGUF inference on Apple Metal; historical
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
- Make no product promise for CPU, CUDA, ROCm, distributed
  inference, tensor parallelism, multi-GPU placement, MTP, DSpark, steering,
  power controls, or custom prefill. Do not preserve these paths with new
  compatibility flags.
- The product name is **LagoonNebula** and the repository, tool, API, and local
  state namespace is **`lgn2`**. Public commands are `lgn2`, `lgn2-server`,
  `lgn2-bench`, and `lgn2-eval`; public C/API names use `lgn2_*`, and runtime
  environment variables use `LGN2_*`; local state lives under `~/.lgn2`. The
  clean break retains no old command, symbol, environment, cache, lock-file, or
  model-link alias. The exact protocol identifiers that must remain stable are
  recorded in `FORK.md`.

## Implementation rules

- Keep one canonical Laguna Metal execution path. Prefer deleting obsolete
  branches over adding another semantic variant behind a flag.
- Keep model loading mmap-backed and whole-model. SSD expert streaming,
  simulated-memory, SSD expert-cache budget, and partial-load compatibility are
  retired; do not reintroduce them, a distributed path, or a host-inference
  fallback. Disk-KV persistence and its independent on-disk budget/eviction
  policy remain supported.
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

- `lgn.c` / `lgn.h`: narrow private LagoonNebula boundary and shared types.
- `lgn_model.c` / `lgn_model.h`: immutable Laguna S2.1 shape, admission,
  tensor-layout, and binding rules.
- `lgn_dflash.c` / `lgn_dflash.h`: private Laguna DFlash profile, metadata,
  tensor binding, and BF16 shadow-map conversion.
- `lgn_graph.c` / `lgn_graph.h`: private Laguna target-graph storage owner for
  base Metal scratch, persistent KV caches, and their allocation lifecycle.
- `lgn_dflash_graph.c` / `lgn_dflash_graph.h`: private DFlash support-graph
  storage owner for feature history, draft scratch, and six-layer KV caches.
- `lgn_dflash_exec.c` / `lgn_dflash_exec.h`: borrowed DFlash support-map
  execution and active-batch-only six-layer injection recording; it never owns
  scheduler, target-output, rollback, or command completion state.
- `lgn2_engine.c`: internal tokenizer, Laguna scheduling, sessions, and disk-cache
  payload serialisation. The unreachable private generic/raw graph
  implementation is deleted; public session/batch/speculative routes own only
  Laguna plus optional DFlash. The standalone GLM/DSA graph and all public
  raw-graph, imatrix, MTP, and DSpark orchestration are gone. The backend enum,
  CPU inference closure, non-Apple accelerator startup shell, steering, power,
  and custom-prefill APIs are also gone. Move supported Laguna code into
  `lgn_*` modules as the remaining dormant low-level backend conditionals and
  generic Metal compatibility paths are deleted.
- `lgn2_cli.c`: command-line and interactive transcript handling.
- `lgn2_server.c`: OpenAI/Anthropic-compatible HTTP API, worker queue,
  streaming, tool-call mapping, and server-side KV-cache policy.
- `lgn2_metal.m`: contracted Objective-C Metal runtime and kernel wrappers.
  Legacy HC, raw-KV, compressor, and generic graph wrappers are removed;
  retain only paths proven reachable from Laguna/DFlash or their correctness
  and lifecycle evidence.
- `metal/*.metal`: Metal compute kernels. Runtime and Makefile source lists
  must remain identical; mixed sources are pruned by symbol rather than by
  filename when Laguna still owns a kernel.
- `tests/`: unit and live integration tests.
- `FORK.md`: refactor boundary, deletion order, guardrails, and fixed namespace
  decisions.

## Testing

Use `make` for build validation and `make test` for the model-independent
Laguna Metal suite on Apple hardware. Run the model-backed gate explicitly as
`make test-metal-laguna-integration LGN2_TEST_MODEL=/absolute/model.gguf`;
it must never fall back to a default fixture or skip a missing model. At each
major refactor boundary, verify:

1. A valid Laguna S2.1 model loads through the whole-model mmap Metal path.
2. Non-Laguna architectures and unsupported backend/mode options are rejected
   clearly rather than selecting an undocumented fallback.
3. CLI generation, server streaming, sessions, batching, and sampling remain
   correct.
4. The retained DFlash path passes its focused regression coverage.

Do not add new CPU, CUDA, ROCm, SSD, distributed, tensor-parallel, multi-GPU,
MTP, DSpark, steering, power, or custom-prefill test obligations. Keep
`BENCHMARK.md` and `benchmark/**` unchanged: they are frozen pre-fork protocol
evidence whose pinned commits and releases remain in `michaelasper/ds4`.

## Safety

- Do not run multiple huge model processes concurrently; the instance lock is
  intentional.
- Do not delete a Metal source or compatibility layer until its callers,
  runtime source list, build target, and focused test coverage have been
  removed or updated together.
- Do not commit or push refactor work unless explicitly requested.
