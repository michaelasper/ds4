# LagoonNebula Metal-only Refactor

This file records the migration boundary for `refactor/laguna-metal-only`.
It is an implementation guide, not a replacement for the frozen benchmark
baseline. The frozen `laguna-s2.1` benchmark commit, its defaults, and its
benchmark data remain unchanged while this branch is cleaned up.

## Current boundary

The intended tool and product is **LagoonNebula**: one Laguna S2.1 GGUF model
family, executed on Apple Metal with the whole model mmap-backed. The
repository will be renamed to **`lgn2`** after the cleanup boundary is stable.
The CLI and server remain supported; DFlash remains a Laguna-specific optional
feature for now.

The following are explicitly outside the product boundary: CPU, CUDA, ROCm,
distributed inference, tensor parallelism, multi-GPU
placement, MTP, DSpark, steering, power controls, and custom prefill. Existing
code for those features is legacy removal material, even when it still builds.
The `ds4_*` names are retained temporarily; renaming is deliberately deferred.

## Completed boundary work

The refactor branch already:

- requires the literal Laguna architecture before engine creation;
- exposes a Metal-only CLI and server contract and rejects retired backend,
  topology, steering, MTP, and DSpark options;
- routes raw generation directly through the Laguna graph;
- removes the obsolete GLM one-shot generation graph, distributed runtime,
  and tensor-parallel transport/lifecycle;
- removes the obsolete multi-GPU layer planner/packer, placement test, and
  their build wiring; this boundary does not claim graph scalarization is
  complete;
- exposes DFlash as the only optional Laguna support-model API and removes
  the public MTP, GLM-MTP, and DSpark aliases/options;
- isolates immutable Laguna S2.1 shape, admission, tensor-layout, binding,
  and output-head rules in the private `lgn_model.c` / `lgn_model.h` module;
- isolates the retained DFlash support profile, metadata/layout validation,
  tensor binding, and parallel BF16 shadow-map conversion in the private
  `lgn_dflash.c` / `lgn_dflash.h` module;
- isolates Laguna target-graph base scratch, persistent KV storage, capacity
  accounting, and allocation/free lifecycle in the private `lgn_graph.c` /
  `lgn_graph.h` module while scheduler-owned speculative/evidence state remains
  in `ds4.c`;
- isolates the complete DFlash support-graph tensor owner and its fixed
  feature/draft/KV allocation lifecycle in the private `lgn_dflash_graph.c` /
  `lgn_dflash_graph.h` module while command scheduling and target capture stay
  in `ds4.c`;
- isolates borrowed DFlash support-model execution in the private
  `lgn_dflash_exec.c` / `lgn_dflash_exec.h` module: BF16 weights use the F16
  shadow map, quantized weights use the support GGUF map, and six-layer
  injection only records into an already-owned command batch; scheduler,
  target-output, rollback, and completion evidence remain in `ds4.c`;
- makes DFlash speculative command ownership explicit: the scheduler owns the
  snapshot, draft, verifier, rollback, and terminal command boundaries;
  submitted snapshots and accepted-prefix restores must complete successfully
  before checkpoint or DFlash state can be certified, and failures quarantine
  the session instead of restoring from untrusted backup data;
- enforces engine-outlives-session ownership and tears down command buffers,
  shared Metal tensors, backend caches, host aliases, and model mappings in
  lifetime-safe order, including partial initialization and GPU-error paths;
- fences Laguna session synchronization, argmax, and mixed-batch admission
  away from legacy GLM execution, using the serial public fallback where the
  generic optimized prefill path does not own the Laguna graph;
- deletes the unreachable standalone GLM/DSA graph, the private generic/raw
  graph implementation, session diagnostics, and CLI reasoning-prefix
  orchestration; six GLM-named Metal router/MoE helpers remain because Laguna
  decode, prefill, and DFlash verification still call them, and the Q2/Q3
  exactness fixture continues to cover that retained kernel surface;
- removes the remaining public raw-graph, first-token, output-head, and
  imatrix diagnostic APIs and CLI switches. Removed switches now fail during
  option parsing, before model I/O, and no deleted generic-graph entry point is
  left reachable from Laguna;
- contracts the public engine and session runtime to the Laguna target graph
  plus optional DFlash support: generic graph workspaces, MTP/DSpark support
  loading, native/backend batch dispatch, and legacy speculative schedulers no
  longer own a public execution route;
- preserves DSV4 v2 byte layout and logical KV-ring ordering, keeps DSVL
  fail-closed for Laguna, and invalidates DFlash state before payload restore
  validation so a partial load can never publish a mixed checkpoint;
- serializes public multi-session batches after validating the entire set,
  propagates mixed-prefill interruption before any decoder runs, and reduces
  the no-DFlash speculative fallback to exactly one ordinary target token;
- refuses engine teardown while sessions remain live, then drains and destroys
  Metal state while the main model, DFlash model, and optional F16 shadow maps
  are still mapped; DFlash graph storage is released before Laguna storage;
- retires inert expert-profile/hotlist controls and makes the historical power
  setting honestly full-power-only. The model-independent default gate now
  includes sampling, DFlash payload lifecycle, serial-route, and terminal
  drain-failure coverage;
- fixes the product name as **LagoonNebula** and the eventual repository name
  as **`lgn2`**, while deliberately postponing the mechanical identifier
  rename until unsupported implementation paths are gone;
- retains only Laguna quality fixtures and tooling under `quality/`;
- preserves normal DSV4 session payloads, disk KV persistence, batching,
  streaming responses, tool calls, and the optional Laguna DFlash path.
- removes SSD expert streaming, simulated-memory accounting, SSD expert-cache
  budget controls, and partial-load compatibility; supported model weights are
  whole-model mmap-backed while generic residency, warmup, and Q4 resident
  paths remain available. Disk-KV persistence, including its independent disk
  budget and eviction policy, remains supported;
- removes the public backend enum and selector, custom-prefill, steering, and
  power fields/APIs; engine and session startup now own one Apple Metal runtime
  directly, while the four retained frontends reject the retired backend
  spellings before model I/O;
- removes the unreachable CPU inference closure, non-Apple accelerator startup
  shell, obsolete multi-GPU public header, and stale GLM long-context smoke
  script, plus dormant ROCm/CUDA conditionals, test fixtures, and stale
  device-cache/tier declarations with no Laguna or DFlash caller. Host
  tokenizer, sampling, payload serialization, DFlash conversion, and Metal
  numerical reference helpers remain where the supported product or its
  correctness tests still use them;
- removes the uncalled tensor-parallel gate service, lifecycle/callback and
  keep-alive APIs, host service-thread/event state, and TP flag/keep-alive
  kernels. Whole-model mmap residency and warmup are now unconditional within
  their existing `DS4_METAL_NO_RESIDENCY`/`DS4_METAL_NO_MODEL_WARMUP` controls;
  the lifecycle-frozen world-1 `DS4_METAL_Q8_MV_NSG` override remains
  supported and world-2 dispatch geometry is retired;
  routed expert argument fields and range helpers remain as private ABI residue
  for the later TP-compatible shader cleanup;
- removes the unreachable legacy Metal HC, raw-KV, generic RoPE, concat,
  repeat, set-rows, softmax, and sum-rows shader families together with their
  public wrappers, pipeline state, runtime source registrations, and stale
  tests. Mixed copy, dense, router, attention, MoE, get-rows, and norm sources
  are reduced at symbol granularity. The six GLM-named helpers still called by
  Laguna, grouped Q2/Q3/Q4/Q5/Q6 and R1 routes, logits top-k/argsort, live
  MXFP4 product templates, command-completion evidence, and DFlash kernels are
  deliberately retained.

The private generic raw graph implementation and its public routes are now
deleted. Active routed expert argument fields/range helpers, tier-aware Metal
helpers, and a smaller set of shared FlashAttention, parallel-FFN,
quantized-MoE, and source-override internals still remain for later low-level
cleanup; no public engine, session, diagnostic, imatrix, or support-model
route owns them. Their presence is transitional and must not be interpreted
as supported behavior. A `glm_` name on one of the six retained router/MoE
Metal helpers describes inherited implementation naming, not GLM product
support.

## Staged deletion and extraction order

1. **Freeze the contract and baseline.** Record the canonical Laguna model
   alias, Metal-only startup behaviour, whole-model mmap policy, DFlash status,
   and the existing Laguna S2.1 benchmark/default baseline. Add focused gates
   before removing implementation paths.
2. **Fence the public surfaces.** Simplify CLI and server options to the
   supported Metal/Laguna modes. Keep HTTP protocol adapters and tool calls,
   but remove promises for unsupported backends and legacy model modes.
3. **Remove model-family compatibility.** Require `general.architecture =
   laguna`; remove DeepSeek Flash/Pro and GLM shapes, validators, graphs,
   prompt/tool branches, MTP, DSpark, steering, power, and custom-prefill
   paths. Keep Laguna DFlash code only where still reachable and tested.
4. **Remove topology compatibility.** Delete distributed, tensor-parallel,
   multi-GPU, layer-packing, GPU-placement, and GPU-argument APIs, then remove
   their transports, payload delegation, commands, and tests.
5. **Remove platform compatibility.** Delete CUDA/MMQ and ROCm/HIP layers and
   build targets. Remove the CPU inference/reference backend after host-side
   helpers needed by Metal diagnostics and serialisation are separated.
6. **Enforce the storage policy.** Completed on this branch: remove SSD
   streaming, simulated-memory, SSD expert-cache budget, and partial-load
   promises. Keep the independent disk-KV budget and eviction policy alongside
   the whole-model mmap map, generic residency/warmup, Q4 resident
   paths, DFlash, and session/KV persistence needed by the supported path.
7. **Slim Metal sources atomically.** Update Metal pipeline globals, runtime
   source loading, the Makefile source list, environment contracts, and
   callers together. Retain Laguna, DFlash, and genuinely shared kernels only
   after a call-graph check.
8. **Extract or remove adjacent tooling.** Keep CLI/server and their focused
   tests. Rewrite or remove DeepSeek/GLM/CUDA/ROCm/distributed fixtures,
   quantisation notes, model download cases, and stale documentation. Extract
   any still-useful Laguna-only tooling before deleting umbrella tooling.
9. **Rename last.** Apply the fixed LagoonNebula product identity and rename
   the repository to `lgn2`. Rename binaries, files, symbols, environment
   variables, cache paths, aliases, and any intentionally migrated payload
   identifiers only after the runtime and test contract is stable.

## Guardrails

- Work only on `refactor/laguna-metal-only`; do not modify or reinterpret the
  frozen `laguna-s2.1` benchmark commit.
- Make the smallest coherent change at each boundary. Do not mix a broad
  rename with backend or model deletion.
- Remove flags and dead branches instead of preserving unsupported behaviour
  behind new aliases. A compatibility alias requires an explicit decision.
- Change a Metal kernel source, `ds4_metal.m`, and its Makefile source entry
  as one unit; never leave the runtime loader naming a deleted source.
- Preserve mmap lifetime, Metal command/resource ownership, KV/session
  serialisation, server streaming, batching, sampling, and the retained
  DFlash path.
- Keep legal attribution for upstream GGUF/GGML/llama.cpp-derived code that
  remains after compatibility tooling is removed.

## Test gates

At each stage, the minimum gate is:

- `make` succeeds on the supported Apple Metal environment;
- `make test` passes the remaining Laguna Metal suite;
- a valid Laguna S2.1 GGUF loads through whole-model mmap;
- a non-Laguna GGUF is rejected with an architecture error;
- CLI generation, server streaming, sessions, batching, and sampling smoke
  tests pass;
- the retained DFlash regression passes.

Use the frozen Laguna S2.1 default-branch commit only as a comparison fixture.
Do not alter its model default, benchmark results, or runbook while
refactoring.
Deletion stages should also include a repository search for removed flags,
targets, source names, and documentation claims so stale compatibility does
not survive the code removal.

## Intentional source-ABI break

`refactor/laguna-metal-only` intentionally breaks the old public
`ds4_engine_options` layout and the multi-GPU source ABI. This is a deliberate
rename/boundary break, not an append-only API evolution: callers must rebuild
against the current headers. No reserved compatibility slots are promised;
retaining them would defeat the rename/break objective.

## Recorded ABI and rename decisions

The current public boundary records the following decisions:

- the final product name is **LagoonNebula** and the repository name is
  **`lgn2`**;
- DFlash remains a public, optional support-model API for Laguna S2.1;
- legacy MTP, GLM-MTP, and DSpark option fields and public aliases are
  intentionally absent; callers must use the DFlash fields and rebuild;
- DSV4/DSVL session payload magics, versions, and layouts remain fixed.

## Deferred ABI and rename decisions

The product and repository names are fixed. No decision has yet been made on
the exact migration spelling or compatibility policy for:

- renaming `ds4_*`/`DS4_*` symbols and executable names;
- changing `DS4_METAL_*` environment variables;
- changing model IDs, model aliases, default filenames, or `~/.ds4` cache paths;
- whether server protocol compatibility names remain unchanged;

Until those choices are recorded, preserve the existing identifiers where the
supported Laguna path still depends on them and avoid silent format changes;
this does not authorize the removed legacy option fields or aliases.
