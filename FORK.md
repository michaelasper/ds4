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
SSD expert streaming, distributed inference, tensor parallelism, multi-GPU
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
- fixes the product name as **LagoonNebula** and the eventual repository name
  as **`lgn2`**, while deliberately postponing the mechanical identifier
  rename until unsupported implementation paths are gone;
- retains only Laguna quality fixtures and tooling under `quality/`;
- preserves normal DSV4 session payloads, disk KV persistence, batching,
  streaming responses, tool calls, and the optional Laguna DFlash path.

Low-level CUDA-oriented tensor-parallel/tier-aware graph helpers, SSD expert
streaming, legacy model helpers, and broad shared-backend Metal code still
remain internally. Graph scalarization is not complete. Their presence is
transitional and must not be interpreted as supported behavior.

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
6. **Enforce the storage policy.** Remove SSD streaming and its cache/planning
   promises if whole-model mmap is the final policy. Keep only the mmap model
   map and session/KV persistence that the supported path needs.
7. **Slim Metal sources atomically.** Update Metal pipeline globals, runtime
   source loading, the Makefile source list, environment contracts, and
   callers together. Retain Laguna, DFlash, and genuinely shared kernels only
   after a call-graph check.
8. **Extract or remove adjacent tooling.** Keep CLI/server and their focused
   tests. Rewrite or remove DeepSeek/GLM/CUDA/ROCm/SSD/distributed fixtures,
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
