# LagoonNebula Metal-only Refactor

This file records the migration boundary for `refactor/laguna-metal-only`.
It is an implementation guide, not a replacement for the frozen benchmark
baseline. The frozen `laguna-s2.1` benchmark commit, its defaults, and its
benchmark data remain unchanged while this branch is cleaned up.

## Current boundary

The product is **LagoonNebula**: one Laguna S2.1 GGUF model family, executed on
Apple Metal with the whole model mmap-backed. The staged repository, tool, API,
environment, and local-state namespace is **`lgn2`**. The supported commands
are `lgn2`, `lgn2-server`, `lgn2-bench`, and `lgn2-eval`; DFlash remains a
Laguna-specific optional feature for now. Local state lives under `~/.lgn2`.

The following are explicitly outside the product boundary: CPU, CUDA, ROCm,
distributed inference, tensor parallelism, multi-GPU
placement, MTP, DSpark, steering, power controls, and custom prefill. Existing
code for those features is legacy removal material, even when it still builds.
The public namespace is a clean break: no prior executable, API, environment,
cache, lock-file, or model-link alias is supported.

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
  in `lgn2_engine.c`;
- isolates the complete DFlash support-graph tensor owner and its fixed
  feature/draft/KV allocation lifecycle in the private `lgn_dflash_graph.c` /
  `lgn_dflash_graph.h` module while command scheduling and target capture stay
  in `lgn2_engine.c`;
- isolates borrowed DFlash support-model execution in the private
  `lgn_dflash_exec.c` / `lgn_dflash_exec.h` module: BF16 weights use the F16
  shadow map, quantized weights use the support GGUF map, and six-layer
  injection only records into an already-owned command batch; scheduler,
  target-output, rollback, and completion evidence remain in `lgn2_engine.c`;
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
- fixes the product name as **LagoonNebula** and records **`lgn2`** as the
  repository, tool, API, environment, and local-state namespace;
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
  their existing `LGN2_METAL_NO_RESIDENCY`/`LGN2_METAL_NO_MODEL_WARMUP` controls;
  the lifecycle-frozen world-1 `LGN2_METAL_Q8_MV_NSG` override remains
  supported and world-2 dispatch geometry is retired;
  active routed expert bindings now use the full model expert range and direct
  expert ids; the active world-1 TP ownership/range ABI residue is removed.
  Zeroed reserved pads preserve the versioned/current constant-buffer layout
  for compatible source overrides, while incompatible or unversioned stale
  overrides fail closed before active dispatch.  The marker is a compatibility
  fingerprint, not a cryptographic trust boundary for a deliberately lying
  override;
- removes the unreachable legacy Metal HC, raw-KV, generic RoPE, concat,
  repeat, set-rows, softmax, and sum-rows shader families together with their
  public wrappers, pipeline state, runtime source registrations, and stale
  tests. Mixed copy, dense, router, attention, MoE, get-rows, and norm sources
  are reduced at symbol granularity. The six GLM-named helpers still called by
  Laguna, grouped Q2/Q3/Q4/Q5/Q6 and R1 routes, logits top-k/argsort, live
  MXFP4 product templates, command-completion evidence, and DFlash kernels are
  deliberately retained.

The active MoE host/MSL ABI marker is
`kernel_laguna_moe_abi_v2_mulmmid104_routed96_stride48`: generation 2,
`mul_mm_id` size 104, routed-MoE size 96, and routed key/stride offset 48.
Every host/MSL layout change must rename or bump this marker.  The strict
checks are `make check-metal-sources`, `make test-laguna-q23-metal`,
`make test-metal-laguna`, and `make test-installed-resources` (also the default
`make test` for the first three).  The ABI gate inside
the latter runs `./lgn2_test --laguna-moe-abi`, whose default matrix creates a
current source and a marker-stripped stale fixture, plus the explicit current
source check:
`LGN2_METAL_MOE_SOURCE=metal/moe.metal LGN2_TEST_MOE_ABI_MODE=current ./lgn2_test --laguna-moe-abi`.
An exact pre-fingerprint parent/old override can be checked with
`LGN2_METAL_MOE_SOURCE=/path/to/pre-fingerprint/metal/moe.metal LGN2_TEST_MOE_ABI_MODE=old ./lgn2_test --laguna-moe-abi`;
all such incompatible or unversioned sources must fail before active work.

The private generic raw graph implementation and its public routes are now
deleted. The dormant generic matvec/sum6 family and its coupled Q4 expert
table/address/cache lifecycle are deleted as well. The active `mul_mm_id`
family, Laguna shared/exact Q2/Q3/Q4 routes, dense Q4/MXFP4 math, dequant
helpers, and whole-model residency remain. A smaller set of shared
FlashAttention, quantized-MoE, and source-override internals remains for
later low-level cleanup; no public engine, session,
diagnostic, imatrix, or support-model route owns them. Their presence is
transitional and must not be interpreted as supported behavior. A `glm_` name
on one of the retained router/MoE Metal helpers describes inherited
implementation naming, not GLM product support.

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
9. **Apply the clean-break namespace.** Use the fixed LagoonNebula product
   identity and `lgn2` repository namespace for binaries, public symbols,
   environment variables, cache paths, local state, and model links. Do not
   retain old aliases or fallbacks; apply this only after the runtime and test
   contract is stable.

## Guardrails

- Work only on `refactor/laguna-metal-only`; do not modify or reinterpret the
  frozen `laguna-s2.1` benchmark commit.
- Make the smallest coherent change at each boundary. Do not mix a broad
  rename with backend or model deletion.
- Remove flags and dead branches instead of preserving unsupported behaviour
  behind new aliases. A compatibility alias requires an explicit decision.
- Change a Metal kernel source, `lgn2_metal.m`, and its Makefile source entry
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

`refactor/laguna-metal-only` intentionally defines the public `lgn2_engine_options`
layout as a clean break from the pre-fork layout and multi-GPU source ABI. This
is a deliberate boundary break, not an append-only API evolution: callers must
rebuild against the current headers. No reserved compatibility slots are
promised; retaining them would defeat the rename/break objective.

## Recorded ABI and rename decisions

The current public boundary records the following decisions:

- the final product name is **LagoonNebula** and the repository name is
  **`lgn2`**;
- the four installed commands will be `lgn2`, `lgn2-server`, `lgn2-bench`,
  and `lgn2-eval`; the public C namespace will be `lgn2_*`/`LGN2_*`, runtime
  environment variables will use `LGN2_*`, and local state will use the
  `~/.lgn2` namespace;
- the rename is a clean break: no old executable, C symbol, environment,
  history, cache, lock-file, or model-symlink alias is retained.  The former
  engine implementation becomes `lgn2_engine.c` so it does not collide with
  the existing private `lgn.c` Laguna module;
- the only convenience model link is `lgn2.gguf`; no legacy model-link name is
  consulted by the renamed tools;
- DFlash remains a public, optional support-model API for Laguna S2.1;
- legacy MTP, GLM-MTP, and DSpark option fields and public aliases are
  intentionally absent; callers must use the DFlash fields and rebuild;
- the canonical Laguna model ids remain `laguna-s-2.1` and its documented
  Laguna aliases. Those are model/API identities, not tool-namespace
  compatibility aliases;
- the GGUF architecture literals `laguna` and `dflash`, every `laguna.*` and
  `dflash.*` metadata key, the pinned GGUF filename/hash/source, GGUF numeric
  codes, and Laguna family/model numeric identities 2/3 remain fixed;
- DSV4/DSVL session payload magics, versions, field order, and layouts remain
  fixed.  The 48-byte KVC header, KVC/DSV4/DSVL version bytes, and serialized
  Laguna model-id byte 3 remain fixed;
- OpenAI-, Anthropic-, and Responses-compatible HTTP routes, request/response
  JSON fields, SSE event shapes, `[DONE]`, and DSML/tokenizer markers remain
  byte-compatible;
- all active Metal `host_name` values and runtime lookup names remain fixed,
  including `kernel_laguna_moe_abi_v2_mulmmid104_routed96_stride48`.  Host and
  shader implementation identifiers may adopt `lgn2`, but their externally
  looked-up entry-point strings do not;
- `BENCHMARK.md` and every file under `benchmark/` are frozen historical
  pre-fork protocol evidence and remain byte-for-byte unchanged. Their old
  `ds4`/`DS4_*` commands and environment names apply only to their pinned
  revisions and are not current LagoonNebula interfaces;
- retained upstream copyright and license attribution is not rewritten as
  new LagoonNebula authorship.

## Repository rename timing

Source and local namespaces can be changed and reviewed on the refactor branch
before the GitHub repository itself moves.  Update clone URLs, Git remotes,
default-branch settings, and release coordinates only as one explicit final
repository operation; do not publish a half-renamed remote.  Until that
operation, `fork/refactor/laguna-metal-only` is the integration branch and the
frozen `laguna-s2.1` branch remains untouched.
