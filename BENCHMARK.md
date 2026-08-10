# Laguna S 2.1 M5 Max benchmark runbook

This bounded how-to guide measures resident Laguna S 2.1 on an M5 Max.  It is
for reproducible evidence, not a single headline number.  Timed arms use the
pristine final baseline commit and change one opt-in selector at a time.  A
failed arm is packaged as a semantic reject so that the rest of the matrix and
its evidence survive.

The first pass is finite:

- Main decode workload: D8, an 8,192-token prompt, 128 greedy tokens, and
  ctx-alloc=8321.
- Prefill workloads: P8K and P16K, separate fresh processes at the 8,192-token
  and 16,384-token frontiers, with no generation.  The K is deliberate: these
  are not 8-token and 16-token tests.
- Route probes: one completion-scoped companion for every cell on the workload
  that actually exercises it; SIMD32/atlas cells prove both decode and prefill.
- Screen order: mirrored/reversed A-B-B-A.  At most three finalists enter ten
  paired randomised blocks, the minimum that makes the predeclared exact
  sign-flip/Holm gate attainable.
The runbook does not run PR8 GPU sampling, the lm-head safe-math screen, or SSD
streaming as a Laguna resident-model cell.  Those are different questions and
would contaminate this matrix.

## Return contract

Return all of these, including when a cell is rejected:

1. results/SUMMARY.md with revision, model SHA-256, workload, cell, effect,
   confidence interval, correction, gate, and reject tables.
2. results/ containing raw commands, environment files, stdout, stderr,
   per-workload CSVs, parity hashes, route-proof logs, and host evidence.
3. laguna-m5-max-results.tar.gz and its SHA-256 beside results/.

Do not delete rejected arms.  A reject is evidence and must be visible in the
summary rather than hidden by an early shell exit.

## Fixed revisions

These are immutable comparison points.  Never substitute a branch name or a
newer tip.

| Label | Commit | Role |
| --- | --- | --- |
| pre-rebase-basic | 448d5695d1c86401a4e9447c440feb983b73e6de | Basic control before the main rebase |
| clean-pre-opt-rebase | 729e0cedf54dccfef93fac5138b26d1157aea56a | Clean pre-optimisation rebase control |
| final-baseline | c8f25ee7ef1b6f16eff470ccf337d1546e314906 | Final baseline; all experiments off, R1 anchor on |
| trace-companion | 4322b2ca7664040b811de4426cc791cb499b2866 | Untimed, completion-scoped route proof only |

The clean controls use the R1 anchor environment, DS4_METAL_GLM_QMV_R1=1,
where the revision understands it.  Older revisions may ignore that variable;
recording the exact SHA and complete environment still makes the comparison
auditable.  The final baseline means only R1 is enabled: no cell selector,
trace selector, sampling selector, fallback selector, or swap/streaming mode.

Report these control deltas separately:

- pre-rebase-basic to clean-pre-opt-rebase: rebase effect;
- clean-pre-opt-rebase to final-baseline: source-series effect;
- pre-rebase-basic to final-baseline: total effect.

## Machine and path preflight

Run on a plugged-in M5 Max with Low Power Mode disabled.  Keep display layout,
power mode, macOS build, Xcode/Clang, and the terminal session fixed.  Stop
other GPU, CPU, disk, and memory work.  Wait for nominal thermal pressure
before every screen and record the state before and after every arm.

This setup is zsh-safe and intentionally does not enable ERR_EXIT.  A
non-zero benchmark process is recorded by run_arm; it does not erase earlier
results.  Use fresh absolute paths for every run.  Do not point
DS4_BENCH_ROOT at an existing checkout or results directory.

~~~zsh
set -u
set -o pipefail

export DS4_BENCH_ROOT="${DS4_BENCH_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/ds4-laguna-m5.XXXXXX")}"
export DS4_GGUF_DIR="${DS4_GGUF_DIR:-$DS4_BENCH_ROOT/models}"
export DS4_JOBS="${DS4_JOBS:-8}"

require_absolute() {
  local name=$1
  local value=$2
  case "$value" in
    /*) ;;
    *) print -u2 -- "$name must be an absolute path: $value"; return 1 ;;
  esac
}

preflight_paths() {
  require_absolute DS4_BENCH_ROOT "$DS4_BENCH_ROOT" || return 1
  require_absolute DS4_GGUF_DIR "$DS4_GGUF_DIR" || return 1
  if [[ -e "$DS4_BENCH_ROOT/control" || -e "$DS4_BENCH_ROOT/results" ]]; then
    print -u2 -- "stop: $DS4_BENCH_ROOT already contains control/ or results/"
    return 1
  fi
  mkdir -p "$DS4_BENCH_ROOT" "$DS4_GGUF_DIR" "$DS4_BENCH_ROOT/results/metadata"
}
preflight_paths || { print -u2 -- 'preflight failed; no clone, build, or benchmark was started'; return 2; }

export DS4_CONTROL_REPO="$DS4_BENCH_ROOT/control"
export DS4_RESULTS="$DS4_BENCH_ROOT/results"
export DS4_PRE_SHA=448d5695d1c86401a4e9447c440feb983b73e6de
export DS4_BASE_SHA=729e0cedf54dccfef93fac5138b26d1157aea56a
export DS4_FINAL_SHA=c8f25ee7ef1b6f16eff470ccf337d1546e314906
export DS4_TRACE_SHA=4322b2ca7664040b811de4426cc791cb499b2866
export DS4_PRE_WT="$DS4_BENCH_ROOT/pre-rebase-basic"
export DS4_BASE_WT="$DS4_BENCH_ROOT/clean-pre-opt-rebase"
export DS4_FINAL_WT="$DS4_BENCH_ROOT/final-baseline"
export DS4_TRACE_WT="$DS4_BENCH_ROOT/trace-companion"
export DS4_ISOLATED_HOME="$DS4_BENCH_ROOT/isolated-home"
export DS4_LONG_PROMPT="$DS4_FINAL_WT/speed-bench/promessi_sposi.txt"
export DS4_SHORT_PROMPT="$DS4_FINAL_WT/tests/long_context_story_prompt.txt"
mkdir -p "$DS4_ISOLATED_HOME"
~~~

## Clean worktrees and model identity

Clone the michaelasper/ds4 fork and fetch all branch tips.  Fetching all refs
is intentional: the rebase control is not an ancestor of the final feature
series, so fetching only the current branch can leave 729e0ced unavailable.

~~~zsh
if [[ ! -d "$DS4_CONTROL_REPO/.git" ]]; then
  git clone --no-checkout https://github.com/michaelasper/ds4.git "$DS4_CONTROL_REPO"
  git -C "$DS4_CONTROL_REPO" fetch --no-tags origin \
    '+refs/heads/*:refs/remotes/origin/*'
fi

missing_commit=0
for sha in "$DS4_PRE_SHA" "$DS4_BASE_SHA" "$DS4_FINAL_SHA" "$DS4_TRACE_SHA"; do
  if ! git -C "$DS4_CONTROL_REPO" cat-file -e "$sha^{commit}"; then
    print -u2 -- "missing required commit $sha; fetch the source fork before continuing"
    missing_commit=1
  fi
done
if (( missing_commit )); then
  print -u2 -- 'stop: one or more fixed revisions are unavailable'
  return 2
fi

git -C "$DS4_CONTROL_REPO" worktree add --detach "$DS4_PRE_WT" "$DS4_PRE_SHA"
git -C "$DS4_CONTROL_REPO" worktree add --detach "$DS4_BASE_WT" "$DS4_BASE_SHA"
git -C "$DS4_CONTROL_REPO" worktree add --detach "$DS4_FINAL_WT" "$DS4_FINAL_SHA"
git -C "$DS4_CONTROL_REPO" worktree add --detach "$DS4_TRACE_WT" "$DS4_TRACE_SHA"

labels=(pre-rebase-basic clean-pre-opt-rebase final-baseline trace-companion)
worktrees=("$DS4_PRE_WT" "$DS4_BASE_WT" "$DS4_FINAL_WT" "$DS4_TRACE_WT")
shas=("$DS4_PRE_SHA" "$DS4_BASE_SHA" "$DS4_FINAL_SHA" "$DS4_TRACE_SHA")
for index in 1 2 3 4; do
  label=${labels[index]}
  wt=${worktrees[index]}
  sha=${shas[index]}
  actual=$(git -C "$wt" rev-parse HEAD)
  print -- "$label,$actual,$sha" >> "$DS4_RESULTS/metadata/revisions.csv"
  git -C "$wt" status --porcelain=v1 > "$DS4_RESULTS/metadata/$label-status-before-build.txt"
done

export DS4_Q23_MODEL="$DS4_GGUF_DIR/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf"
if [[ ! -f "$DS4_Q23_MODEL" ]]; then
  ( cd "$DS4_FINAL_WT" && DS4_GGUF_DIR="$DS4_GGUF_DIR" ./download_model.sh laguna-q2-q3 )
fi
missing_input=0
for path in "$DS4_LONG_PROMPT" "$DS4_SHORT_PROMPT" "$DS4_Q23_MODEL"; do
  if [[ ! -f "$path" ]]; then
    print -u2 -- "missing required file: $path"
    missing_input=1
  fi
done
if (( missing_input )); then
  print -u2 -- 'stop: required prompts/model are absent'
  return 2
fi

export DS4_MODEL_SHA256=$(shasum -a 256 "$DS4_Q23_MODEL" | awk '{print $1}')
shasum -a 256 "$DS4_Q23_MODEL" "$DS4_LONG_PROMPT" "$DS4_SHORT_PROMPT" \
  > "$DS4_RESULTS/metadata/input-sha256.txt"
print -- "model SHA-256: $DS4_MODEL_SHA256"
~~~

The model hash is measured once before timing and copied into every arm's
metadata.  Do not hash a resident model between paired arms.  A changed model
path or hash invalidates the run and requires a fresh benchmark root.

## Build evidence and host-state capture

Build each fixed revision in its own clean worktree.  Build products are not
part of the source comparison.  Retain the command, SHA, stdout, stderr,
compiler identity, and post-build worktree status.

~~~zsh
capture_state() {
  local out=$1
  mkdir -p "$out"
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ' > "$out/time.txt" 2>&1 || true
  # Never archive raw profiler output: it can contain serials, UUIDs, UDIDs,
  # display identifiers, or other host identifiers.  Redact structurally
  # before the JSON reaches a file.
  profiler_json() {
    local destination=$1
    shift
    system_profiler -json "$@" 2>/dev/null |
      python3 -c '
import json, re, sys
key_re = re.compile(r"serial|uuid|udid|identifier|hardware.?address|platform.?id", re.I)
uuid_re = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
def scrub(value, key=""):
    if key_re.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): scrub(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return uuid_re.sub("<redacted-uuid>", value)
    return value
try:
    data = json.load(sys.stdin)
except Exception:
    data = {"status": "profiler-unavailable"}
json.dump(scrub(data), sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\n")
' > "$destination" || true
  }
  profiler_json "$out/system-profiler-sanitized.json" \
    SPHardwareDataType SPSoftwareDataType
  profiler_json "$out/power-display-profiler-sanitized.json" \
    SPPowerDataType SPDisplaysDataType
  pmset -g batt > "$out/pmset-batt.txt" 2>&1 || true
  pmset -g custom > "$out/pmset-custom.txt" 2>&1 || true
  pmset -g therm > "$out/pmset-therm.txt" 2>&1 || true
  sysctl hw.model hw.ncpu hw.memsize kern.osproductversion kern.osversion > "$out/sysctl-host.txt" 2>&1 || true
  sysctl vm.swapusage > "$out/vm-swapusage.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure.txt" 2>&1 || true
  vm_stat > "$out/vm-stat.txt" 2>&1 || true
  df -h "$DS4_BENCH_ROOT" "$DS4_GGUF_DIR" > "$out/df.txt" 2>&1 || true
}

thermal_is_nominal() {
  local report=$1
  local lower=${report:l}
  [[ "$lower" == *"no thermal warning level has been recorded"* ]] &&
  [[ "$lower" == *"no performance warning level has been recorded"* ]] &&
  [[ "$lower" == *"no cpu power status has been recorded"* ]] &&
  [[ "$lower" != *"cpu power notify"* ]]
}

wait_for_nominal() {
  local destination=$1
  local attempt report
  mkdir -p "${destination:h}"
  for attempt in {1..30}; do
    report=$(pmset -g therm 2>&1)
    print -r -- "$report" > "$destination"
    if thermal_is_nominal "$report"; then return 0; fi
    sleep 10
  done
  print -u2 -- "thermal state did not return to nominal; see $destination"
  return 1
}

verify_m5_host() {
  local out="$DS4_RESULTS/metadata/host-gate"
  mkdir -p "$out"
  capture_state "$out/state"
  local chip mem batt custom therm
  chip=$(system_profiler SPHardwareDataType -json 2>/dev/null |
    python3 -c 'import json,sys; d=json.load(sys.stdin); print(" ".join(str(v) for x in d.values() for r in (x if isinstance(x,list) else []) for k,v in r.items() if "chip" in k.lower()))')
  mem=$(sysctl -n hw.memsize 2>/dev/null)
  batt=$(pmset -g batt 2>&1)
  custom=$(pmset -g custom 2>&1)
  therm=$(pmset -g therm 2>&1)
  print -r -- "$chip" > "$out/chip.txt"
  print -r -- "$mem" > "$out/memsize.txt"
  print -r -- "$batt" > "$out/battery.txt"
  print -r -- "$custom" > "$out/power-settings.txt"
  print -r -- "$therm" > "$out/thermal.txt"
  [[ "$chip" == *"Apple M5 Max"* ]] || { print -u2 -- "requires Apple M5 Max, found: $chip"; return 1; }
  [[ "$mem" == 137438953472 ]] || { print -u2 -- "requires 128 GiB, found hw.memsize=$mem"; return 1; }
  [[ "$batt" == *"AC Power"* ]] || { print -u2 -- 'AC power is required'; return 1; }
  print -r -- "$custom" | awk '
    /AC Power:/ { ac=1; next }
    /^[^[:space:]]/ && !/AC Power:/ { ac=0 }
    ac && $1 == "lowpowermode" && $2 == 0 { ok=1 }
    END { exit(ok ? 0 : 1) }
  ' || { print -u2 -- 'Low Power Mode must be off in the AC profile'; return 1; }
  thermal_is_nominal "$therm" || { print -u2 -- 'thermal/performance/CPU-power state is not nominal'; return 1; }
}

verify_m5_host || { print -u2 -- 'host gate failed; benchmark not started'; return 2; }

build_revision() {
  local label=$1
  local wt=$2
  local sha=$3
  local out="$DS4_RESULTS/metadata/build-$label"
  mkdir -p "$out"
  {
    print -r -- "cwd=$wt"
    print -r -- "git_sha=$sha"
    print -r -- "sanitized make -C $wt -j$DS4_JOBS ds4 ds4-bench"
  } > "$out/command.txt"
  git -C "$wt" rev-parse HEAD > "$out/revision.txt"
  git -C "$wt" status --porcelain=v1 > "$out/status-before.txt"
  capture_state "$out/state-before"
  ( cd "$wt" && env -i HOME="$DS4_ISOLATED_HOME" \
      PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C make -j"$DS4_JOBS" ds4 ds4-bench ) \
      > "$out/stdout.log" 2> "$out/stderr.log"
  local rc=$?
  git -C "$wt" status --porcelain=v1 > "$out/status-after.txt"
  cc --version > "$out/compiler.txt" 2>&1 || true
  capture_state "$out/state-after"
  if (( rc == 0 )); then
    print -- PASS > "$out/status.txt"
  else
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- "build exit=$rc" > "$out/semantic-reject.txt"
  fi
  return 0
}

build_revision pre-rebase-basic "$DS4_PRE_WT" "$DS4_PRE_SHA"
build_revision clean-pre-opt-rebase "$DS4_BASE_WT" "$DS4_BASE_SHA"
build_revision final-baseline "$DS4_FINAL_WT" "$DS4_FINAL_SHA"
build_revision trace-companion "$DS4_TRACE_WT" "$DS4_TRACE_SHA"

# The final and companion revisions own the feature tests used by this matrix.
( cd "$DS4_FINAL_WT" && env -i HOME="$DS4_ISOLATED_HOME" \
    PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
    make -j"$DS4_JOBS" ds4_test && ./ds4_test --metal-kernels ) \
  > "$DS4_RESULTS/metadata/final-metal-tests.stdout.log" \
  2> "$DS4_RESULTS/metadata/final-metal-tests.stderr.log"
[[ $? == 0 ]] || { print -u2 -- 'final Metal test suite failed'; return 2; }
( cd "$DS4_TRACE_WT" && env -i HOME="$DS4_ISOLATED_HOME" \
    PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
    make -j"$DS4_JOBS" ds4_test && \
    ./ds4_test --laguna-selector-parser --laguna-bench-trace-transaction ) \
  > "$DS4_RESULTS/metadata/trace-tests.stdout.log" \
  2> "$DS4_RESULTS/metadata/trace-tests.stderr.log"
[[ $? == 0 ]] || { print -u2 -- 'trace companion tests failed'; return 2; }
~~~

If a build is rejected, keep its logs and do not call it a performance result.
The final package may contain the other revisions, with the missing comparison
explicitly marked SEMANTIC_REJECT.

## Workloads and exactly 15 optimisation cells

Every ordinary arm starts with the same R1 anchor and receives only the cell's
listed overrides.  Timed arms contain no trace selector, profiler, logit dump,
or route-trace variable.  A fresh process is the selector reset boundary;
never mutate an experiment variable while a process is running.

~~~zsh
typeset -a R1_ENV D8_ARGS P8K_ARGS P16K_ARGS M64_ARGS
R1_ENV=(DS4_METAL_GLM_QMV_R1=1)
D8_ARGS=(
  --metal -m "$DS4_Q23_MODEL" --prompt-file "$DS4_LONG_PROMPT"
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8321
  --step-incr 1 --gen-tokens 128 --warm-weights
)
P8K_ARGS=(
  --metal -m "$DS4_Q23_MODEL" --prompt-file "$DS4_LONG_PROMPT"
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8193
  --step-incr 1 --gen-tokens 0 --warm-weights
)
P16K_ARGS=(
  --metal -m "$DS4_Q23_MODEL" --prompt-file "$DS4_LONG_PROMPT"
  --ctx-start 16384 --ctx-max 16384 --ctx-alloc 16385
  --step-incr 1 --gen-tokens 0 --warm-weights
)
M64_ARGS=(
  --metal -m "$DS4_Q23_MODEL" --prompt-file "$DS4_LONG_PROMPT"
  --ctx-start 64 --ctx-max 64 --ctx-alloc 65
  --step-incr 1 --gen-tokens 0 --prefill-chunk 64 --warm-weights
)
~~~

D8 primary is gen_steady_tps at ctx_tokens=8192; its first-token measurement
is gen_first_ms.  P8K and P16K are separate fresh prefill processes at exactly
8,192 and 16,384 tokens; each reports its own prefill_tps.  M64 is a separate
64-token prefill used by the MoE threshold cell.  Never use one incremental
8,192-to-16,384 process: it would measure only the suffix and change the graph
row count.

The table below is the complete optimisation matrix.  It has exactly 15 cells;
do not add an unlisted cell to the first pass.

| # | Cell | Timed override(s), in addition to R1 | Main exercised path |
| ---: | --- | --- | --- |
| 1 | GQA9 | DS4_METAL_LAGUNA_SWA_GQA9=1 | SWA grouped-attention decode |
| 2 | fused router | DS4_METAL_LAGUNA_ROUTER_DECODE_FUSED=1 | One-row routed decode router |
| 3 | residual fusion | DS4_METAL_LAGUNA_DECODE_RESIDUAL_NORM=1 | Decode add/residual norm |
| 4 | dense-Q8 decode | DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU=1; DS4_METAL_Q8_MV_ROWS=2 | Leading dense Q8 gate/up + SwiGLU |
| 5 | canonical ladder | DS4_METAL_LAGUNA_DECODE_LADDER=7,15,23,31,39,47 | Six exact M5 decode ladder boundaries |
| 6 | direct KV prefill | DS4_METAL_LAGUNA_DIRECT_KV_PREFILL=1 | Direct KV store at P8K/P16K |
| 7 | batched dense-Q8 prefill | DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_BATCH=1 | Batched dense Q8 at P8K/P16K |
| 8 | router SIMD top-k | DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK=1 | Standalone routed decode selector |
| 9 | SIMD32 Q/K | DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32=1 | SIMD32 Q/K norm + RoPE |
| 10 | RoPE atlas | DS4_METAL_LAGUNA_ROPE_ATLAS=1 | Strict global/SWA RoPE angle atlas |
| 11 | SIMD32×atlas | DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32=1; DS4_METAL_LAGUNA_ROPE_ATLAS=1 | Combined Q/K SIMD32 and atlas |
| 12 | GQA3 | DS4_METAL_LAGUNA_SWA_GQA3=1 | Full-ring SWA GQA3 |
| 13 | MoE threshold32 | DS4_METAL_GLM_GROUPED_MOE_MIN_TOKENS=32 | Grouped MoE prefill at M64 |
| 14 | front-rung ladder | DS4_METAL_LAGUNA_DECODE_LADDER=1,7,15,23,31,39,47 | Seven-rung decode schedule |
| 15 | adaptive champion | Only accepted, recorded single-cell overrides selected after analysis | Final bounded combination; never invent a new flag |

The requested output-head concept is intentionally not a Laguna cell: c8's
Laguna raw/session path never consumes the generic HC output-head selector.
Cell 8 is the distinct router SIMD top-k replacement and is tested separately
from the fused-router cell; never combine those selectors in a champion.

Cell 5 is the canonical six-rung schedule 7,15,23,31,39,47.  Cell 14 is the
distinct seven-rung front schedule 1,7,15,23,31,39,47.  The latter is the
bounded replacement for any unavailable optional schedule.  Cell 15 remains
empty until accepted cells and their exact environment strings are copied into
CHAMPION_ENV; an empty champion is a packaged NOT_ELIGIBLE result.

## State-safe timed arm runner

This is the only timed entry point.  It uses env -i with absolute model,
prompt, binary, CSV, and output paths.  /usr/bin/time -l keeps maximum RSS in
raw stderr; extracted max-rss.txt is convenience evidence, not a replacement
for stderr.  The runner always writes status.txt and returns zero to the shell.

~~~zsh
capture_arm_state() { capture_state "$1"; }

arm_bad_text() {
  local out=$1
  local found=0
  local path
  : > "$out/unexpected-diagnostics.txt"
  for path in "$out"/*.stderr.log(N); do
    if rg -i 'fallback|unavailable|paused|failed|invalid|mismatch|refusal' \
        "$path" >> "$out/unexpected-diagnostics.txt" 2>&1; then
      found=1
    fi
  done
  (( found == 1 ))
}

run_workload() {
  local out=$1
  local name=$2
  local wt=$3
  shift 3
  local -a args
  args=("$@")
  local csv="$out/$name.csv"
  {
    print -r -- "cwd=$wt"
    print -r -- "binary=$wt/ds4-bench"
    printf 'arg=%q\n' "$wt/ds4-bench" "${args[@]}" --csv "$csv"
  } > "$out/$name.command.txt"
  /usr/bin/time -l env -i \
    HOME="$DS4_ISOLATED_HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
    TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C AppleLanguages='(en)' \
    "${ARM_ENV[@]}" "$wt/ds4-bench" "${args[@]}" --csv "$csv" \
    > "$out/$name.stdout.log" 2> "$out/$name.stderr.log"
  local rc=$?
  if [[ -s "$csv" ]]; then
    shasum -a 256 "$csv" > "$out/$name.csv.sha256"
    python3 - "$csv" <<'PY' || rc=65
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1], newline='')))
required = ('ctx_tokens','prefill_tokens','prefill_tps','gen_tokens',
            'gen_tps','gen_first_ms','gen_steady_tokens','gen_steady_tps')
if not rows or any(k not in rows[-1] for k in required):
    raise SystemExit(1)
for key in required:
    value = float(rows[-1][key])
    if not math.isfinite(value) or value < 0:
        raise SystemExit(1)
PY
  elif (( rc == 0 )); then
    rc=65
  fi
  rg -i 'maximum resident set size|peak resident|resident set' \
    "$out/$name.stderr.log" > "$out/$name.max-rss.txt" 2>&1 || true
  print -- "$rc" > "$out/$name.exit-code.txt"
  return "$rc"
}

run_arm() {
  local rel=$1
  local wt=$2
  local label=$3
  local mode=$4
  shift 4
  local -a overrides
  overrides=("$@")
  local out="$DS4_RESULTS/$rel"
  mkdir -p "$out"
  git -C "$wt" rev-parse HEAD > "$out/revision.txt"
  print -r -- "$label" > "$out/arm-label.txt"
  print -r -- "$mode" > "$out/workload-mode.txt"
  print -rl -- "${overrides[@]}" > "$out/cell-overrides.txt"
  print -rl -- "${R1_ENV[@]}" "${overrides[@]}" > "$out/env-overrides.txt"
  print -r -- "${DS4_MODEL_SHA256:-missing}" > "$out/model-sha256.txt"
  {
    print -r -- "cwd=$wt"
    print -r -- "binary=$wt/ds4-bench"
    print -r -- "workload_mode=$mode"
    print -r -- 'timed_trace=off'
    for value in "${R1_ENV[@]}" "${overrides[@]}"; do print -r -- "env=$value"; done
  } > "$out/command.txt"
  typeset -g -a ARM_ENV
  ARM_ENV=("${R1_ENV[@]}" "${overrides[@]}")
  for value in "${ARM_ENV[@]}"; do
    if [[ "$value" == *TRACE*=* || "$value" == DS4_TRACE_TOP=* ]]; then
      print -- SEMANTIC_REJECT > "$out/status.txt"
      print -- 'timed arm contains a trace selector' > "$out/semantic-reject.txt"
      return 0
    fi
  done
  if [[ -z "$DS4_MODEL_SHA256" ]]; then
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- 'model SHA-256 was not captured before timing' > "$out/semantic-reject.txt"
    return 0
  fi
  if ! wait_for_nominal "$out/cooldown-before.txt"; then
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- 'host thermal state was not nominal before the arm' > "$out/semantic-reject.txt"
    return 0
  fi
  capture_arm_state "$out/state-before"
  local rc=0
  local one_rc=0
  case "$mode" in
    decode)
      run_workload "$out" d8 "$wt" "${D8_ARGS[@]}" || rc=$?
      ;;
    prefill)
      run_workload "$out" p8k "$wt" "${P8K_ARGS[@]}" || rc=$?
      run_workload "$out" p16k "$wt" "${P16K_ARGS[@]}" || one_rc=$?
      if (( rc == 0 && one_rc != 0 )); then rc=$one_rc; fi
      ;;
    moe64)
      run_workload "$out" m64 "$wt" "${M64_ARGS[@]}" || rc=$?
      ;;
    both)
      run_workload "$out" d8 "$wt" "${D8_ARGS[@]}" || rc=$?
      run_workload "$out" p8k "$wt" "${P8K_ARGS[@]}" || one_rc=$?
      if (( rc == 0 && one_rc != 0 )); then rc=$one_rc; fi
      run_workload "$out" p16k "$wt" "${P16K_ARGS[@]}" || one_rc=$?
      if (( rc == 0 && one_rc != 0 )); then rc=$one_rc; fi
      ;;
    full)
      run_workload "$out" d8 "$wt" "${D8_ARGS[@]}" || rc=$?
      run_workload "$out" p8k "$wt" "${P8K_ARGS[@]}" || one_rc=$?
      if (( rc == 0 && one_rc != 0 )); then rc=$one_rc; fi
      run_workload "$out" p16k "$wt" "${P16K_ARGS[@]}" || one_rc=$?
      if (( rc == 0 && one_rc != 0 )); then rc=$one_rc; fi
      run_workload "$out" m64 "$wt" "${M64_ARGS[@]}" || one_rc=$?
      if (( rc == 0 && one_rc != 0 )); then rc=$one_rc; fi
      ;;
    *)
      rc=64
      print -- "unknown workload mode: $mode" > "$out/semantic-reject.txt"
      ;;
  esac
  capture_arm_state "$out/state-after"
  git -C "$wt" status --porcelain=v1 > "$out/worktree-status.txt"
  if (( rc != 0 )); then
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- "benchmark process exit=$rc" > "$out/semantic-reject.txt"
  elif arm_bad_text "$out"; then
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- 'unexpected fallback/unavailable/failure/mismatch diagnostic' > "$out/semantic-reject.txt"
  else
    print -- PASS > "$out/status.txt"
  fi
  return 0
}
~~~

run_arm intentionally does not call set -e, and it does not turn an unavailable
optional path into a stock result.  An explicit selector that cannot be
honoured must fail closed in the binary and be packaged as a reject.

## Mirrored/reversed controls and cell screens

Run the fixed-revision control in reversed order.  A, B, and C are the three
immutable revisions; the first and last arm are the same revision and bound
thermal/drift changes.

~~~zsh
run_arm controls/01-pre-A "$DS4_PRE_WT" A both
run_arm controls/02-base-B "$DS4_BASE_WT" B both
run_arm controls/03-final-C "$DS4_FINAL_WT" C both
run_arm controls/04-final-C "$DS4_FINAL_WT" C both
run_arm controls/05-base-B "$DS4_BASE_WT" B both
run_arm controls/06-pre-A "$DS4_PRE_WT" A both

run_cell_screen() {
  local cell=$1
  local mode=$2
  shift 2
  local -a flags
  flags=("$@")
  run_arm "warmup/$cell/B" "$DS4_FINAL_WT" B "$mode" "${flags[@]}"
  run_arm "screen/$cell/01-A" "$DS4_FINAL_WT" A "$mode"
  run_arm "screen/$cell/02-B" "$DS4_FINAL_WT" B "$mode" "${flags[@]}"
  run_arm "screen/$cell/03-B" "$DS4_FINAL_WT" B "$mode" "${flags[@]}"
  run_arm "screen/$cell/04-A" "$DS4_FINAL_WT" A "$mode"
}

run_cell_screen 01-gqa9 decode DS4_METAL_LAGUNA_SWA_GQA9=1
run_cell_screen 02-fused-router decode DS4_METAL_LAGUNA_ROUTER_DECODE_FUSED=1
run_cell_screen 03-residual-fusion decode DS4_METAL_LAGUNA_DECODE_RESIDUAL_NORM=1
run_cell_screen 04-dense-q8-decode decode \
  DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU=1 DS4_METAL_Q8_MV_ROWS=2
run_cell_screen 05-canonical-ladder decode \
  DS4_METAL_LAGUNA_DECODE_LADDER=7,15,23,31,39,47
run_cell_screen 06-direct-kv-prefill prefill DS4_METAL_LAGUNA_DIRECT_KV_PREFILL=1
run_cell_screen 07-batched-dense-q8-prefill prefill \
  DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_BATCH=1
run_cell_screen 08-router-simd-topk decode DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK=1
run_cell_screen 09-simd32-qk both DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32=1
run_cell_screen 10-rope-atlas both DS4_METAL_LAGUNA_ROPE_ATLAS=1
run_cell_screen 11-simd32-atlas both \
  DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32=1 \
  DS4_METAL_LAGUNA_ROPE_ATLAS=1
run_cell_screen 12-gqa3 decode DS4_METAL_LAGUNA_SWA_GQA3=1
run_cell_screen 13-moe-threshold32 moe64 DS4_METAL_GLM_GROUPED_MOE_MIN_TOKENS=32
run_cell_screen 14-front-rung-ladder decode \
  DS4_METAL_LAGUNA_DECODE_LADDER=1,7,15,23,31,39,47
~~~

Each cell has one candidate warm-up that is not analysed, then exactly mirrored
A-B-B-A arms. A is R1 on pristine c8; B is R1 plus exactly the cell overrides.

## Completion-scoped route-proof companions

These runs are untimed and use only the pinned trace-companion SHA. Timed data
always comes from pristine `c8f25ee`; never time or substitute the companion
binary. Each probe must end with one waited-success graph record and the exact
route counts below.

~~~zsh
typeset -a TRACE_D8_ARGS
TRACE_D8_ARGS=(
  --metal -m "$DS4_Q23_MODEL" --prompt-file "$DS4_LONG_PROMPT"
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8194
  --step-incr 1 --gen-tokens 1 --warm-weights
)

run_route_probe() {
  local cell=$1
  local mode=$2
  shift 2
  local -a flags args trace_flags
  flags=("$@")
  trace_flags=()
  case "$mode" in
    decode) args=("${TRACE_D8_ARGS[@]}") ;;
    prefill) args=("${P8K_ARGS[@]}") ;;
    moe64) args=("${M64_ARGS[@]}") ;;
    *) print -u2 -- "bad route mode: $mode"; return 1 ;;
  esac
  if [[ "$cell" == 08-router-simd-topk ]]; then
    trace_flags=(DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK_TRACE=1)
  fi
  local out="$DS4_RESULTS/route-proof/$cell-$mode"
  mkdir -p "$out"
  print -rl -- DS4_LAGUNA_BENCH_TRACE=1 "${R1_ENV[@]}" \
    "${trace_flags[@]}" "${flags[@]}" > "$out/env-overrides.txt"
  capture_arm_state "$out/state-before"
  env -i HOME="$DS4_ISOLATED_HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
    TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C AppleLanguages='(en)' \
    DS4_LAGUNA_BENCH_TRACE=1 "${R1_ENV[@]}" "${trace_flags[@]}" \
    "${flags[@]}" "$DS4_TRACE_WT/ds4-bench" "${args[@]}" \
    --csv "$out/metrics.csv" > "$out/stdout.log" 2> "$out/stderr.log"
  local rc=$?
  capture_arm_state "$out/state-after"
  if (( rc == 0 )); then
    python3 - "$cell" "$mode" "$out/stderr.log" <<'PY' || rc=66
import re, sys
cell, mode, path = sys.argv[1:]
lines = open(path, errors='replace').read().splitlines()
records = []
for line in lines:
    if 'ds4-laguna-bench-trace ' not in line:
        continue
    rec = dict(re.findall(r'([A-Za-z0-9_]+)=([^ ]+)', line))
    records.append(rec)
if not records or any(r.get('status') != 'ok' for r in records):
    raise SystemExit('missing or failed completion-scoped trace record')
graph = 'decode' if mode == 'decode' else 'prefill'
matches = [r for r in records if r.get('graph') == graph]
if not matches:
    raise SystemExit(f'missing graph={graph} record')
r = matches[-1]
expected_tokens = {'decode':'1','prefill':'8192','moe64':'64'}[mode]
if r.get('completion') != 'waited' or r.get('n_tokens') != expected_tokens:
    raise SystemExit('wrong completion or token count')
need = {
 '01-gqa9': {'gqa9':36, 'global_grouped':12},
 '02-fused-router': {'router_fused':47, 'router_stock':0},
 '04-dense-q8-decode': {'dense_decode_fused':1, 'dense_decode_stock':0},
 '06-direct-kv-prefill': {'direct_kv':12, 'staged_kv':36},
 '07-batched-dense-q8-prefill': {'dense_prefill_fused':1, 'dense_prefill_stock':0},
 '08-router-simd-topk': {'router_stock':47, 'router_fused':0},
 '09-simd32-qk': {'simd32':48, 'ordinary_qk':0},
 '10-rope-atlas': {'consumers':48, 'family0':12, 'family1':36},
 '11-simd32-atlas': {'simd32':48, 'consumers':48, 'family0':12, 'family1':36},
 '12-gqa3': {'gqa3':36, 'global_grouped':12},
 '13-moe-threshold32': {'moe_grouped':47, 'moe_grouped_rows':3008},
}.get(cell, {})
for key, value in need.items():
    if int(r.get(key, '-1')) != value:
        raise SystemExit(f'{cell}: expected {key}={value}, got {r.get(key)}')
text = '\n'.join(lines)
if cell == '03-residual-fusion' and not re.search(
        r'residual\+RMS fusion enabled .*completion=waited', text):
    raise SystemExit('missing waited residual-fusion diagnostic')
if cell == '05-canonical-ladder' and not re.search(
        r'layers=7,15,23,31,39,47; flushes=6; completion=waited', text):
    raise SystemExit('wrong canonical ladder diagnostic')
if cell == '14-front-rung-ladder' and not re.search(
        r'layers=1,7,15,23,31,39,47; flushes=7; completion=waited', text):
    raise SystemExit('wrong front-rung ladder diagnostic')
if cell == '08-router-simd-topk' and not re.search(
        r'router SIMD top-k trace optimized=47 fallback=0 encoded_rows=47 encoded_dispatches=47', text):
    raise SystemExit('wrong router SIMD trace')
PY
  fi
  if (( rc == 0 )); then
    print -- PASS > "$out/status.txt"
  else
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- "route proof exit=$rc" > "$out/semantic-reject.txt"
  fi
  return 0
}

run_route_probe 01-gqa9 decode DS4_METAL_LAGUNA_SWA_GQA9=1
run_route_probe 02-fused-router decode DS4_METAL_LAGUNA_ROUTER_DECODE_FUSED=1
run_route_probe 03-residual-fusion decode DS4_METAL_LAGUNA_DECODE_RESIDUAL_NORM=1
run_route_probe 04-dense-q8-decode decode \
  DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU=1 DS4_METAL_Q8_MV_ROWS=2
run_route_probe 05-canonical-ladder decode \
  DS4_METAL_LAGUNA_DECODE_LADDER=7,15,23,31,39,47
run_route_probe 06-direct-kv-prefill prefill DS4_METAL_LAGUNA_DIRECT_KV_PREFILL=1
run_route_probe 07-batched-dense-q8-prefill prefill \
  DS4_METAL_LAGUNA_DENSE_Q8_GATE_UP_SWIGLU_BATCH=1
run_route_probe 08-router-simd-topk decode DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK=1
run_route_probe 09-simd32-qk decode DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32=1
run_route_probe 09-simd32-qk prefill DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32=1
run_route_probe 10-rope-atlas decode DS4_METAL_LAGUNA_ROPE_ATLAS=1
run_route_probe 10-rope-atlas prefill DS4_METAL_LAGUNA_ROPE_ATLAS=1
run_route_probe 11-simd32-atlas decode \
  DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32=1 DS4_METAL_LAGUNA_ROPE_ATLAS=1
run_route_probe 11-simd32-atlas prefill \
  DS4_METAL_LAGUNA_QK_NORM_ROPE_SIMD32=1 DS4_METAL_LAGUNA_ROPE_ATLAS=1
run_route_probe 12-gqa3 decode DS4_METAL_LAGUNA_SWA_GQA3=1
run_route_probe 13-moe-threshold32 moe64 DS4_METAL_GLM_GROUPED_MOE_MIN_TOKENS=32
run_route_probe 14-front-rung-ladder decode \
  DS4_METAL_LAGUNA_DECODE_LADDER=1,7,15,23,31,39,47
~~~

Every timed cell must have a `PASS` route-proof status before finalist
selection. The trace branch is intentionally companion-only and rejects
DFlash; this bounded benchmark does not download or run DFlash.

## Finalists: ten paired randomised blocks

A cell becomes a finalist only after its mirrored screen passes hard semantic
gates and its primary estimate is at least +1.5%.  Select at most three
finalists.  Copy exact accepted selector strings; do not type a different
spelling when composing the champion.

~~~zsh
typeset -i DS4_RANDOM_SEED=${DS4_RANDOM_SEED:-20260809}

run_finalist_blocks() {
  local finalist=$1
  local mode=$2
  shift 2
  local -a flags
  flags=("$@")
  local order_file="$DS4_RESULTS/finalists/$finalist/random-order.csv"
  mkdir -p "${order_file:h}"
  print -- 'block,order' > "$order_file"
  local block
  for block in {1..10}; do
    local order
    order=$(python3 - "$DS4_RANDOM_SEED" "$block" <<'PY'
import random
import sys
r = random.Random(int(sys.argv[1]) + int(sys.argv[2]) * 1009)
arms = ['A', 'B']
r.shuffle(arms)
print(' '.join(arms))
PY
)
    print -- "$block,$order" >> "$order_file"
    local arm
    for arm in ${(z)order}; do
      if [[ "$arm" == A ]]; then
        run_arm "finalists/$finalist/block-$block-A" "$DS4_FINAL_WT" A "$mode"
      else
        run_arm "finalists/$finalist/block-$block-B" "$DS4_FINAL_WT" B "$mode" "${flags[@]}"
      fi
    done
  done
}
~~~

Cell 15 is the adaptive champion.  Populate a bounded array from accepted
cells only, then run exactly the same ten blocks on fresh holdout observations.
The empty default writes
NOT_ELIGIBLE instead of accidentally measuring an unreviewed combination.

~~~zsh
typeset -a CHAMPION_ENV
CHAMPION_ENV=()
# Example after review, never before the ledger:
# CHAMPION_ENV=(
#   DS4_METAL_LAGUNA_SWA_GQA9=1
#   DS4_METAL_LAGUNA_ROUTER_SIMD_TOPK=1
# )
if (( ${#CHAMPION_ENV[@]} == 0 )); then
  mkdir -p "$DS4_RESULTS/finalists/15-adaptive-champion"
  print -- NOT_ELIGIBLE > "$DS4_RESULTS/finalists/15-adaptive-champion/status.txt"
  print -- 'adaptive champion awaits accepted single-cell flags' \
    > "$DS4_RESULTS/finalists/15-adaptive-champion/semantic-reject.txt"
else
  run_finalist_blocks 15-adaptive-champion full "${CHAMPION_ENV[@]}"
fi
~~~

For each selected single-cell finalist, call `run_finalist_blocks CELL both
FLAGS...`; use `full` for cell 13 so M64 is present. These 10 pairs are fresh
confirmation data—do not reuse the A-B-B-A screen in the exact test. At most
three single cells advance, limiting expensive reloads while retaining a
valid two-sided sign-flip minimum p-value of 2/1024.

## Analysis, Holm correction, and gates

The 14 single-cell experiments are the predeclared Holm family. Cell 15 is an
adaptive combination evaluated only on fresh holdout pairs and is never added
to that family. Decode cells use D8 `gen_steady_tps` as primary; cells 6 and 7
use P16K `prefill_tps`; cell 13 uses M64 `prefill_tps`. Ten confirmation pairs
give a minimum two-sided exact sign-flip p-value of 2/1024=.001953, sufficient
for the first Holm threshold .05/14=.003571. Screen observations are selection
data only and never enter confirmation.

The script below reads the paired finalist directories, writes one effect per
block, assigns p=1 to every unconfirmed/rejected family member, applies the
step-down Holm cumulative maximum, and bootstraps a deterministic 95% lower
bound. It intentionally ignores cell 15.

~~~zsh
mkdir -p "$DS4_RESULTS/analysis"
python3 - "$DS4_RESULTS" <<'PY'
import csv, math, os, random, statistics, sys
root = sys.argv[1]
family = [
 '01-gqa9','02-fused-router','03-residual-fusion','04-dense-q8-decode',
 '05-canonical-ladder','06-direct-kv-prefill','07-batched-dense-q8-prefill',
 '08-router-simd-topk','09-simd32-qk','10-rope-atlas','11-simd32-atlas',
 '12-gqa3','13-moe-threshold32','14-front-rung-ladder']
primary_metric = {c:('d8.csv','gen_steady_tps') for c in family}
primary_metric['06-direct-kv-prefill'] = ('p16k.csv','prefill_tps')
primary_metric['07-batched-dense-q8-prefill'] = ('p16k.csv','prefill_tps')
primary_metric['13-moe-threshold32'] = ('m64.csv','prefill_tps')

def value(path, key):
    rows = list(csv.DictReader(open(path, newline='')))
    x = float(rows[-1][key])
    if not math.isfinite(x) or x <= 0: raise ValueError(path)
    return x
def effect(b, a): return 100.0 * (b / a - 1.0)
effects=[]; grouped={c:[] for c in family}
for cell in family:
    for block in range(1, 11):
        a=os.path.join(root,'finalists',cell,f'block-{block}-A')
        b=os.path.join(root,'finalists',cell,f'block-{block}-B')
        try:
            pf, pk = primary_metric[cell]
            row={'cell':cell,'block':block,
                 'primary_effect_pct':effect(value(os.path.join(b,pf),pk),value(os.path.join(a,pf),pk)),
                 'd8_pct':effect(value(os.path.join(b,'d8.csv'),'gen_steady_tps'),value(os.path.join(a,'d8.csv'),'gen_steady_tps')),
                 'p8_pct':effect(value(os.path.join(b,'p8k.csv'),'prefill_tps'),value(os.path.join(a,'p8k.csv'),'prefill_tps')),
                 'p16_pct':effect(value(os.path.join(b,'p16k.csv'),'prefill_tps'),value(os.path.join(a,'p16k.csv'),'prefill_tps')),
                 'first_token_pct':effect(value(os.path.join(b,'d8.csv'),'gen_first_ms'),value(os.path.join(a,'d8.csv'),'gen_first_ms'))}
        except (OSError, KeyError, ValueError, IndexError):
            continue
        effects.append(row); grouped[cell].append(row)
with open(os.path.join(root,'analysis','effects.csv'),'w',newline='') as f:
    fields=['cell','block','primary_effect_pct','d8_pct','p8_pct','p16_pct','first_token_pct']
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(effects)

def signflip(v):
    if len(v) != 10: return 1.0
    obs=abs(statistics.mean(v)); hits=0
    for mask in range(1 << len(v)):
        x=statistics.mean(x if (mask>>i)&1 else -x for i,x in enumerate(v))
        hits += abs(x) >= obs - 1e-12
    return hits/(1 << len(v))
def lower(v, seed):
    if len(v) != 10: return float('-inf')
    r=random.Random(seed); means=[]
    for _ in range(10000): means.append(statistics.mean(r.choice(v) for _ in v))
    means.sort(); return means[249]

summary=[]
for i,cell in enumerate(family):
    rows=grouped[cell]; v=[r['primary_effect_pct'] for r in rows]
    summary.append({'cell':cell,'n':len(v),
      'primary_median_pct':statistics.median(v) if v else float('-inf'),
      'primary_lower_ci_pct':lower(v,1729+i),'p_value':signflip(v),
      'd8_min_pct':min((r['d8_pct'] for r in rows),default=float('-inf')),
      'p8_min_pct':min((r['p8_pct'] for r in rows),default=float('-inf')),
      'p16_min_pct':min((r['p16_pct'] for r in rows),default=float('-inf')),
      'first_token_max_pct':max((r['first_token_pct'] for r in rows),default=float('inf'))})
ordered=sorted(summary,key=lambda r:(r['p_value'],r['cell'])); running=0.0
for rank,row in enumerate(ordered):
    running=max(running,min(1.0,row['p_value']*(len(family)-rank)))
    row['holm_p_adj']=running
with open(os.path.join(root,'analysis','holm.csv'),'w',newline='') as f:
    fields=['cell','n','primary_median_pct','primary_lower_ci_pct','p_value','holm_p_adj',
            'd8_min_pct','p8_min_pct','p16_min_pct','first_token_max_pct']
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(sorted(summary,key=lambda r:r['cell']))
PY
~~~

Hard gates:

- Primary: median speedup ≥ +1.5%, 95% lower bound > +1.0%, exactly 10 valid
  pairs, and Holm-adjusted p ≤ .05. Primary is D8 steady decode except P16K
  prefill for cells 6/7 and M64 prefill for cell 13.
- Secondary: every measured D8/P8K/P16K effect is ≥ −1%. A missing or malformed
  CSV is a semantic reject, not a zero.
- First token: report `gen_first_ms`, but do not use it as a promotion gate:
  existing completion diagnostics perturb the first evaluation of residual
  and ladder processes. `gen_steady_tps` remains clean and primary for decode.
- Output: generated text is identical; P16K logits are bitwise identical; the
  8192→8193 one-row decode logits are bitwise identical except GQA3/GQA9,
  which require the same finite winner plus max-abs ≤5e-4 and RMS ≤1e-4.
- Exact path: the pinned trace-companion route proof is `PASS` for the cell's
  actual workload (both decode and prefill for cells 9–11).
- No fallback: an explicit selector must not print fallback, unavailable,
  paused, failed, invalid, or refusal diagnostics.  Explicit requests fail
  closed rather than silently selecting stock code.
- No swap: before/after vm.swapusage, memory_pressure, and vm_stat show no new
  swap or pressure event.  A new swap event rejects the arm.
- State: power remains connected, Low Power Mode remains disabled, and thermal
  pressure is nominal.  Record excursions and rerun affected arms in a fresh
  root.

The two observations for each A and B arm in the screen must each drift by no
more than 3%; otherwise rerun the entire A-B-B-A cell in a fresh result path.
A trace-enabled companion must never be mixed into a timed arm.

## Exact parity probe

Timed ds4-bench arms do not use show-output, logit dumps, or tracing.  After a
cell/finalist has a valid timed screen, run one separate parity probe for R1
and candidate, save raw logs, and compare generated text and frontier hashes.

~~~zsh
run_parity_probe() {
  local rel=$1
  shift
  local -a flags
  flags=("$@")
  local out="$DS4_RESULTS/parity/$rel"
  mkdir -p "$out/decode-logits" "$out/prefill-logits"
  capture_arm_state "$out/state-before"
  ( cd "$DS4_FINAL_WT" && env -i \
      HOME="$DS4_ISOLATED_HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
      TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
      DS4_METAL_GLM_QMV_R1=1 "${flags[@]}" \
      ./ds4-bench --metal -m "$DS4_Q23_MODEL" \
      --prompt-file "$DS4_LONG_PROMPT" --ctx-start 8192 --ctx-max 8193 \
      --ctx-alloc 8194 --step-incr 1 --gen-tokens 0 --warm-weights \
      --dump-frontier-logits-f32-dir "$out/decode-logits" \
      --csv "$out/decode.csv" > "$out/decode.stdout.log" \
      2> "$out/decode.stderr.log" )
  local rc=$?
  ( cd "$DS4_FINAL_WT" && env -i \
      HOME="$DS4_ISOLATED_HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
      TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
      DS4_METAL_GLM_QMV_R1=1 "${flags[@]}" \
      ./ds4-bench --metal -m "$DS4_Q23_MODEL" \
      --prompt-file "$DS4_LONG_PROMPT" --ctx-start 16384 --ctx-max 16384 \
      --ctx-alloc 16385 --step-incr 1 --gen-tokens 0 --warm-weights \
      --dump-frontier-logits-f32-dir "$out/prefill-logits" \
      --csv "$out/prefill.csv" > "$out/prefill.stdout.log" \
      2> "$out/prefill.stderr.log" ) || rc=$?
  ( cd "$DS4_FINAL_WT" && env -i \
      HOME="$DS4_ISOLATED_HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
      TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
      DS4_METAL_GLM_QMV_R1=1 "${flags[@]}" \
      ./ds4-bench "${D8_ARGS[@]}" --show-output --csv "$out/generated.csv" \
      > "$out/generated.stdout.log" 2> "$out/generated.stderr.log" ) || rc=$?
  capture_arm_state "$out/state-after"
  rg 'gen\[ctx=8192\] decoded text:' "$out/generated.stderr.log" \
    > "$out/generated-line.txt" 2>&1 || rc=67
  shasum -a 256 "$out"/*.csv "$out"/*-logits/*.f32 \
    "$out/generated-line.txt" > "$out/artifacts.sha256" 2>/dev/null || rc=67
  if (( rc != 0 )); then print -- SEMANTIC_REJECT > "$out/status.txt"; else print -- PASS > "$out/status.txt"; fi
  return 0
}
run_parity_probe r1
# After a finalist passes all gates, run the same probe with its exact flags:
# run_parity_probe 01-gqa9 DS4_METAL_LAGUNA_SWA_GQA9=1
~~~

For every candidate, `cmp` its `generated-line.txt`, P16K F32 file, and (except
GQA3/GQA9) `decode-logits/frontier_008193.logits.f32` against `parity/r1`.
For GQA3/GQA9, compare the decode F32 arrays with Python `struct.iter_unpack`
and record finite argmax equality, max absolute difference, and RMS; enforce
the numeric limits above. A speed CSV is never parity evidence.

## Summary and archive

Create the final summary after the analysis and parity ledgers. Name every
cell, including NOT_ELIGIBLE and SEMANTIC_REJECT.
Keep paths absolute in the manifest.

~~~zsh
python3 - "$DS4_RESULTS/SUMMARY.md" "$DS4_RESULTS/analysis/holm.csv" <<'PY'
import csv
import os
import sys

summary_path, holm_path = sys.argv[1:3]
rows = list(csv.DictReader(open(holm_path, newline=''))) if os.path.isfile(holm_path) else []
cells = [
    '01-gqa9', '02-fused-router', '03-residual-fusion',
    '04-dense-q8-decode', '05-canonical-ladder', '06-direct-kv-prefill',
    '07-batched-dense-q8-prefill', '08-router-simd-topk', '09-simd32-qk',
    '10-rope-atlas', '11-simd32-atlas', '12-gqa3', '13-moe-threshold32',
    '14-front-rung-ladder', '15-adaptive-champion',
]
indexed = {row.get('cell'): row for row in rows}
with open(summary_path, 'w') as handle:
    handle.write('# Laguna S 2.1 M5 Max benchmark summary\n\n')
    handle.write('Generated from raw evidence; semantic rejects are retained.\n\n')
    handle.write('## Identity\n\n')
    handle.write('- pre-rebase/basic: 448d5695d1c86401a4e9447c440feb983b73e6de\n')
    handle.write('- clean pre-opt rebase: 729e0cedf54dccfef93fac5138b26d1157aea56a\n')
    handle.write('- final baseline: c8f25ee7ef1b6f16eff470ccf337d1546e314906; R1 on, others off\n\n')
    handle.write('- untimed trace companion: 4322b2ca7664040b811de4426cc791cb499b2866\n\n')
    handle.write('## Cell decisions\n\n')
    handle.write('| Cell | Primary median % | Lower CI % | Holm p | P8 min % | P16 min % | First-token max % | Status |\n')
    handle.write('| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n')
    for cell in cells:
        row = indexed.get(cell)
        if row:
            handle.write('| {cell} | {primary_median_pct} | {primary_lower_ci_pct} | {holm_p_adj} | {p8_min_pct} | {p16_min_pct} | {first_token_max_pct} | review hard gates |\n'.format(**row))
        else:
            handle.write(f'| {cell} | — | — | — | — | — | — | PENDING/REJECTED |\n')
    handle.write('\n## Gates\n\n')
    handle.write('- Primary ≥ +1.5%, lower CI > +1.0%; D8/P8K/P16K secondaries ≥ −1%.\n')
    handle.write('- First-token latency is reported but non-gating because existing one-shot diagnostics perturb it.\n')
    handle.write('- Exact output/path, no fallback, and no new swap are mandatory.\n')
    handle.write('- Raw evidence is in metadata/, controls/, screen/, finalists/, route-proof/, and parity/.\n')
    handle.write('- Holm correction uses the 14 predeclared single-cell hypotheses; cell 15 is fresh holdout.\n')
PY

[[ -s "$DS4_RESULTS/SUMMARY.md" ]] || { print -u2 -- 'SUMMARY.md missing'; return 2; }
capture_state "$DS4_RESULTS/metadata/final-state"
find "$DS4_RESULTS" -type f -print | LC_ALL=C sort > "$DS4_RESULTS/MANIFEST.txt"
export DS4_ARCHIVE="$DS4_BENCH_ROOT/laguna-m5-max-results.tar.gz"
COPYFILE_DISABLE=1 tar -C "$DS4_BENCH_ROOT" -czf "$DS4_ARCHIVE" results
shasum -a 256 "$DS4_ARCHIVE" > "$DS4_ARCHIVE.sha256"
print -- "SUMMARY=$DS4_RESULTS/SUMMARY.md"
print -- "ARCHIVE=$DS4_ARCHIVE"
print -- "ARCHIVE_SHA256=$DS4_ARCHIVE.sha256"
~~~

Before hand-off, check SUMMARY.md, MANIFEST.txt, the archive, and its SHA are
non-empty; check every status.txt is PASS, SEMANTIC_REJECT, or NOT_ELIGIBLE;
and capture final-state/ with the same host/thermal/
power/memory function.  Preserve DS4_BENCH_ROOT until the archive has been
copied and its SHA-256 verified.
