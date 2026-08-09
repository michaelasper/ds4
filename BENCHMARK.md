# Laguna S 2.1 benchmark protocol for M5 Max

Use this runbook to compare the preserved pre-rebase Laguna branch, the
rebased baseline, and the current optimisation candidate on a 128 GB M5 Max.
The first return is deliberately bounded: it establishes correctness and a
directional performance result before spending time on the full 65K-context
sweep.

Run every binary from its own worktree. DS4 loads `metal/*.metal` relative to
the current working directory; invoking a binary by absolute path from another
checkout can silently combine a binary from one revision with Metal source
from another.

## Return contract

Return both of these:

1. A `SUMMARY.md` containing the required tables and observations in
   [Report the result](#report-the-result).
2. The compressed raw-results archive produced in
   [Package the evidence](#package-the-evidence), including its SHA-256.

Do not return only a headline tokens/second number. A result is usable only
when its revision, model, command, raw CSV, stdout, stderr, parity evidence,
and machine state are present.

## Fixed revisions

| Role | Immutable commit | Purpose |
| --- | --- | --- |
| Pre-rebase/basic | `448d5695d1c86401a4e9447c440feb983b73e6de` | Preserved Laguna lineage before rebasing onto `main` |
| Rebased baseline | `729e0cedf54dccfef93fac5138b26d1157aea56a` | Same Laguna tip rebased onto `main`, before the new optimisation series |
| Candidate | `0e6c49fcf59a0d391e363c3ea990478db8303254` | Current source candidate, including the exact-on and opt-in experiments |

Report these deltas separately:

- pre-rebase/basic to rebased baseline: effect of integrating current `main`;
- rebased baseline to candidate: effect of the new optimisation series;
- pre-rebase/basic to candidate: total branch improvement.

Never substitute a branch name or a newer branch tip for these commits.

## Prepare the machine

Use the M5 Max while plugged into power, with Low Power Mode disabled and no
other GPU-, CPU-, disk-, or memory-intensive work running. Let the machine
return to nominal thermal pressure before starting. Keep the same power mode,
display arrangement, and terminal session for the complete first pass.

The required models are:

- `laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf`;
- `laguna-s-2.1-Q4_K_M.gguf`;
- `laguna-s-2.1-DFlash-Q8_0.gguf`.

Allow at least 130 GB of free disk space for the three models, worktrees,
builds, and results. Set an absolute model directory before continuing:

```zsh
export DS4_GGUF_DIR=/absolute/path/to/laguna-gguf
case "$DS4_GGUF_DIR" in
  /*) ;;
  *) print -u2 'DS4_GGUF_DIR must be an absolute path'; return 2 ;;
esac
mkdir -p "$DS4_GGUF_DIR"
```

## Create immutable worktrees

Use a fresh control clone so local edits and build products cannot enter the
comparison. Keep this shell open: later commands use the exported paths.

```zsh
set -euo pipefail

export DS4_BENCH_ROOT="${DS4_BENCH_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/ds4-m5-max.XXXXXX")}"
case "$DS4_BENCH_ROOT" in
  /*) ;;
  *) print -u2 'DS4_BENCH_ROOT must be an absolute path'; return 2 ;;
esac
export DS4_CONTROL_REPO="$DS4_BENCH_ROOT/control"
export DS4_RESULTS="$DS4_BENCH_ROOT/results"
export DS4_PRE_SHA=448d5695d1c86401a4e9447c440feb983b73e6de
export DS4_BASE_SHA=729e0cedf54dccfef93fac5138b26d1157aea56a
export DS4_CAND_SHA=0e6c49fcf59a0d391e363c3ea990478db8303254
export DS4_PRE_WT="$DS4_BENCH_ROOT/pre-rebase"
export DS4_BASE_WT="$DS4_BENCH_ROOT/rebased-baseline"
export DS4_CAND_WT="$DS4_BENCH_ROOT/candidate"

git clone --no-checkout https://github.com/michaelasper/ds4.git "$DS4_CONTROL_REPO"
git -C "$DS4_CONTROL_REPO" fetch origin \
  refs/heads/laguna-s2.1 \
  refs/heads/backup/laguna-s2.1-pre-main-rebase-20260808

for sha in "$DS4_PRE_SHA" "$DS4_BASE_SHA" "$DS4_CAND_SHA"; do
  git -C "$DS4_CONTROL_REPO" cat-file -e "$sha^{commit}"
done

git -C "$DS4_CONTROL_REPO" worktree add --detach "$DS4_PRE_WT" "$DS4_PRE_SHA"
git -C "$DS4_CONTROL_REPO" worktree add --detach "$DS4_BASE_WT" "$DS4_BASE_SHA"
git -C "$DS4_CONTROL_REPO" worktree add --detach "$DS4_CAND_WT" "$DS4_CAND_SHA"
mkdir -p "$DS4_RESULTS/metadata"
git -C "$DS4_CONTROL_REPO" rev-parse origin/laguna-s2.1 \
  > "$DS4_RESULTS/metadata/runbook-commit.txt"
git -C "$DS4_CONTROL_REPO" show origin/laguna-s2.1:BENCHMARK.md \
  > "$DS4_RESULTS/metadata/BENCHMARK.md"

for item in \
  "pre-rebase:$DS4_PRE_WT:$DS4_PRE_SHA" \
  "rebased-baseline:$DS4_BASE_WT:$DS4_BASE_SHA" \
  "candidate:$DS4_CAND_WT:$DS4_CAND_SHA"; do
  label=${item%%:*}
  rest=${item#*:}
  wt=${rest%%:*}
  expected=${rest##*:}
  actual=$(git -C "$wt" rev-parse HEAD)
  [[ "$actual" == "$expected" ]]
  print "$label,$actual" >> "$DS4_RESULTS/metadata/revisions.csv"
done

print "Benchmark root: $DS4_BENCH_ROOT"
```

If the models are not already present, download them once through the
candidate worktree:

```zsh
(cd "$DS4_CAND_WT" && DS4_GGUF_DIR="$DS4_GGUF_DIR" ./download_model.sh laguna-q2-q3)
(cd "$DS4_CAND_WT" && DS4_GGUF_DIR="$DS4_GGUF_DIR" ./download_model.sh laguna-q4)
(cd "$DS4_CAND_WT" && DS4_GGUF_DIR="$DS4_GGUF_DIR" ./download_model.sh laguna-dflash)

export DS4_Q23_MODEL="$DS4_GGUF_DIR/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf"
export DS4_Q4_MODEL="$DS4_GGUF_DIR/laguna-s-2.1-Q4_K_M.gguf"
export DS4_DFLASH_MODEL="$DS4_GGUF_DIR/laguna-s-2.1-DFlash-Q8_0.gguf"
export DS4_PROMPT="$DS4_CAND_WT/tests/long_context_story_prompt.txt"
export DS4_LONG_PROMPT="$DS4_CAND_WT/speed-bench/promessi_sposi.txt"
export DS4_RAW_PROMPT="$DS4_RESULTS/metadata/promessi-first-600-lines.txt"

for file_path in "$DS4_Q23_MODEL" "$DS4_Q4_MODEL" "$DS4_DFLASH_MODEL" \
                 "$DS4_PROMPT" "$DS4_LONG_PROMPT"; do
  [[ -f "$file_path" ]] || { print -u2 "Missing required file: $file_path"; return 2; }
done

[[ "$(shasum -a 256 "$DS4_PROMPT" | awk '{print $1}')" == \
   29363eab21bbbccaeea8e13f669e7ce05e8eafc48e31fcf9b725edabb2058666 ]]
[[ "$(shasum -a 256 "$DS4_LONG_PROMPT" | awk '{print $1}')" == \
   f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f ]]
sed -n '1,600p' "$DS4_LONG_PROMPT" > "$DS4_RAW_PROMPT"
[[ "$(shasum -a 256 "$DS4_RAW_PROMPT" | awk '{print $1}')" == \
   a50224adb48545ded1a76ab176543812a1afb5e76f413dbd7fe8cf256cb9c1c2 ]]

shasum -a 256 "$DS4_Q23_MODEL" "$DS4_Q4_MODEL" "$DS4_DFLASH_MODEL" \
  "$DS4_PROMPT" "$DS4_LONG_PROMPT" "$DS4_RAW_PROMPT" \
  > "$DS4_RESULTS/metadata/input-sha256.txt"
```

Model hashing reads more than 100 GB once. Complete it before timing rather
than between measured arms.

## Record the host and build each revision

```zsh
{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  uname -a
  sw_vers
  sysctl -n machdep.cpu.brand_string
  sysctl -n hw.memsize
  sysctl -n hw.logicalcpu
  xcodebuild -version
  cc --version
  system_profiler SPHardwareDataType SPDisplaysDataType SPPowerDataType
  pmset -g custom
  pmset -g batt
  pmset -g therm
  memory_pressure -Q
  vm_stat
  df -h "$DS4_BENCH_ROOT" "$DS4_GGUF_DIR"
} > "$DS4_RESULTS/metadata/host-before.txt" 2>&1

JOBS=$(sysctl -n hw.logicalcpu)
for wt in "$DS4_PRE_WT" "$DS4_BASE_WT" "$DS4_CAND_WT"; do
  (cd "$wt" && make -j"$JOBS" ds4 ds4-bench)
done

(cd "$DS4_CAND_WT" && make -B -j"$JOBS" ds4_test test-glm-q23-metal)
(cd "$DS4_CAND_WT" && ./ds4_test --metal-kernels) \
  > "$DS4_RESULTS/metadata/metal-kernels.stdout" \
  2> "$DS4_RESULTS/metadata/metal-kernels.stderr"
(cd "$DS4_CAND_WT" && ./tests/test_glm_q23_metal) \
  > "$DS4_RESULTS/metadata/glm-q23.stdout" \
  2> "$DS4_RESULTS/metadata/glm-q23.stderr"

for wt in "$DS4_PRE_WT" "$DS4_BASE_WT" "$DS4_CAND_WT"; do
  git -C "$wt" status --short --branch
done > "$DS4_RESULTS/metadata/worktree-status-after-build.txt"
```

Stop if either Metal test exits non-zero. Compiler warnings may be recorded,
but a missing optional pipeline, staged-SWA fallback in require mode, numerical
mismatch, or test failure invalidates the candidate run.

## Cool down before benchmark warm-ups

The build and Metal smoke tests can heat the machine. Do not start any warm-up
until the thermal state is nominal again. This bounded wait is unattended; it
polls `pmset` for up to one hour, then records the exact start state used by the
first matrix.

```zsh
for attempt in {1..120}; do
  benchmark_start_therm=$(pmset -g therm 2>&1 || true)
  if print -r -- "$benchmark_start_therm" | grep -Eiq \
      'nominal|thermal level: 0|no thermal warning level has been recorded'; then
    break
  fi
  if (( attempt == 120 )); then
    print -u2 'thermal state did not return to nominal within one hour'
    return 1
  fi
  sleep 30
done
{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  print -r -- "$benchmark_start_therm"
  memory_pressure -Q
  vm_stat
} > "$DS4_RESULTS/metadata/benchmark-start.txt" 2>&1
```

## Install the run helpers

The helpers start each process with a small, explicit environment. This keeps
unrelated `DS4_*` debug and experiment variables out of the measurement.
`DS4_METAL_DISABLE_Q23_EXACT_MULTIROW` is presence-tested: even a value of `0`
disables the optimisation, so it must be absent from control runs.

```zsh
extract_bench_text() {
  perl -0777 -e '
    my $s = do { local $/; <> };
    $s =~ /ds4-bench: gen\[ctx=8192\] decoded text: "(.*)"\n/s
      or die "generated benchmark text was not found\n";
    print $1;
  ' "$1"
}

run_bench() {
  local run_id=$1
  local wt=$2
  local model=$3
  shift 3
  local out="$DS4_RESULTS/$run_id"
  mkdir -p "$out/logits"
  {
    printf 'cwd=%q\n' "$wt"
    printf 'env_override=%q\n' "$@"
    printf 'binary=%q\n' ./ds4-bench
    printf 'arg=%q\n' --metal -m "$model" --prompt-file "$DS4_PROMPT" \
      --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8321 \
      --gen-tokens 128 --warm-weights --show-output \
      --dump-frontier-logits-dir "$out/logits" --csv "$out/metrics.csv"
  } > "$out/command.txt"
  print -rl -- "$@" > "$out/env-overrides.txt"
  git -C "$wt" rev-parse HEAD > "$out/revision.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/start-utc.txt"
  pmset -g therm > "$out/thermal-before.txt" 2>&1 || true
  vm_stat > "$out/vm-before.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-before.txt" 2>&1 || true
  (
    cd "$wt"
    /usr/bin/time -p env -i \
      HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C \
      "$@" \
      ./ds4-bench --metal \
        -m "$model" \
        --prompt-file "$DS4_PROMPT" \
        --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8321 \
        --gen-tokens 128 --warm-weights --show-output \
        --dump-frontier-logits-dir "$out/logits" \
        --csv "$out/metrics.csv"
  ) > "$out/stdout.log" 2> "$out/stderr.log"
  extract_bench_text "$out/stderr.log" > "$out/decoded.txt"
  pmset -g therm > "$out/thermal-after.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-after.txt" 2>&1 || true
  vm_stat > "$out/vm-after.txt" 2>&1 || true
  shasum -a 256 "$out/metrics.csv" "$out/decoded.txt" \
    "$out/logits/frontier_008192.logits.json" > "$out/output-sha256.txt"
}

run_decode_probe() {
  local run_id=$1
  local wt=$2
  local model=$3
  shift 3
  local out="$DS4_RESULTS/$run_id"
  mkdir -p "$out/logits"
  {
    printf 'cwd=%q\n' "$wt"
    printf 'env_override_count=%q\n' "$#"
    for env_name in "$@"; do
      printf 'env_override=%q\n' "$env_name"
    done
    printf 'binary=%q\n' ./ds4-bench
    printf 'arg=%q\n' --metal -m "$model" --prompt-file "$DS4_PROMPT" \
      --ctx-start 8192 --ctx-max 8193 --step-incr 1 --ctx-alloc 8194 \
      --gen-tokens 0 --dump-frontier-logits-dir "$out/logits" \
      --csv "$out/metrics.csv"
  } > "$out/command.txt"
  print -rl -- "$@" > "$out/env-overrides.txt"
  git -C "$wt" rev-parse HEAD > "$out/revision.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/start-utc.txt"
  pmset -g therm > "$out/thermal-before.txt" 2>&1 || true
  vm_stat > "$out/vm-before.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-before.txt" 2>&1 || true
  (
    cd "$wt"
    /usr/bin/time -p env -i \
      HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C \
      "$@" \
      ./ds4-bench --metal \
        -m "$model" --prompt-file "$DS4_PROMPT" \
        --ctx-start 8192 --ctx-max 8193 --step-incr 1 --ctx-alloc 8194 \
        --gen-tokens 0 --dump-frontier-logits-dir "$out/logits" \
        --csv "$out/metrics.csv"
  ) > "$out/stdout.log" 2> "$out/stderr.log"
  pmset -g therm > "$out/thermal-after.txt" 2>&1 || true
  vm_stat > "$out/vm-after.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-after.txt" 2>&1 || true
  shasum -a 256 "$out/metrics.csv" \
    "$out/logits/frontier_008193.logits.json" > "$out/output-sha256.txt"
}

compare_decode_logits() {
  local off_run=$1
  local on_run=$2
  local report_rel=$3
  local off_json="$DS4_RESULTS/$off_run/logits/frontier_008193.logits.json"
  local on_json="$DS4_RESULTS/$on_run/logits/frontier_008193.logits.json"
  local report="$DS4_RESULTS/$report_rel"
  mkdir -p "${report:h}"
  {
    printf 'off_json=%q\n' "$off_json"
    printf 'on_json=%q\n' "$on_json"
    printf 'report=%q\n' "$report"
  } > "$report.command.txt"
  python3 - "$off_json" "$on_json" "$report" <<'PY'
import csv
import json
import math
import sys

off_path, on_path, report_path = sys.argv[1:]

def read_logits(path):
    with open(path, encoding="utf-8") as stream:
        document = json.load(stream)
    logits = document.get("logits")
    vocab = document.get("vocab")
    if not isinstance(logits, list) or not logits:
        raise SystemExit(f"missing logits in {path}")
    if not isinstance(vocab, int) or vocab != len(logits):
        raise SystemExit(f"vocab/logits length mismatch in {path}")
    values = []
    for value in logits:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"non-numeric logit in {path}")
        value = float(value)
        if not math.isfinite(value):
            raise SystemExit(f"non-finite logit in {path}")
        values.append(value)
    top = sorted(range(vocab), key=lambda index: (-values[index], index))
    computed_argmax = top[0]
    if document.get("argmax_id") != computed_argmax:
        raise SystemExit(f"argmax metadata mismatch in {path}")
    return values, vocab, top

off, off_vocab, off_top = read_logits(off_path)
on, on_vocab, on_top = read_logits(on_path)
if off_vocab != on_vocab:
    raise SystemExit(f"vocab differs: {off_vocab} != {on_vocab}")
if len(off) != len(on):
    raise SystemExit("logit vector lengths differ")
if off_top[0] != on_top[0]:
    raise SystemExit(f"argmax differs: {off_top[0]} != {on_top[0]}")

absolute = [abs(left - right) for left, right in zip(off, on)]
ordered = sorted(absolute)
p99_index = max(0, math.ceil(0.99 * len(ordered)) - 1)
max_abs = max(absolute)
rms = math.sqrt(sum(error * error for error in absolute) / len(absolute))
denominator = math.sqrt(sum(value * value for value in off)) * math.sqrt(
    sum(value * value for value in on)
)
if denominator == 0.0:
    raise SystemExit("cannot compute cosine for zero-norm logits")
cosine = sum(left * right for left, right in zip(off, on)) / denominator
off_margin = off[off_top[0]] - off[off_top[1]]
on_margin = on[on_top[0]] - on[on_top[1]]
investigate = max_abs > 1.0e-3 or rms > 1.0e-4
fields = [
    "off_json", "on_json", "vocab", "finite", "argmax_same",
    "off_argmax_id", "on_argmax_id", "off_argmax_logit",
    "on_argmax_logit", "off_margin", "on_margin", "max_abs", "rms",
    "p99_abs", "cosine", "top10_overlap", "investigation_flag",
]
row = {
    "off_json": off_path,
    "on_json": on_path,
    "vocab": off_vocab,
    "finite": True,
    "argmax_same": True,
    "off_argmax_id": off_top[0],
    "on_argmax_id": on_top[0],
    "off_argmax_logit": off[off_top[0]],
    "on_argmax_logit": on[on_top[0]],
    "off_margin": off_margin,
    "on_margin": on_margin,
    "max_abs": max_abs,
    "rms": rms,
    "p99_abs": ordered[p99_index],
    "cosine": cosine,
    "top10_overlap": len(set(off_top[:10]) & set(on_top[:10])) / 10.0,
    "investigation_flag": "investigate" if investigate else "within-first-pass-bound",
}
with open(report_path, "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerow(row)
print(f"{report_path}: max_abs={max_abs:.9g} rms={rms:.9g} "
      f"p99_abs={ordered[p99_index]:.9g} cosine={cosine:.9g} "
      f"argmax={off_top[0]} flag={row['investigation_flag']}")
if investigate:
    print("warning: numeric drift exceeds the uncalibrated first-pass bound; "
          "investigate before promotion", file=sys.stderr)
PY
  shasum -a 256 "$report" > "$report.sha256"
}

run_raw() {
  local run_id=$1
  local wt=$2
  local model=$3
  shift 3
  local out="$DS4_RESULTS/$run_id"
  mkdir -p "$out"
  {
    printf 'cwd=%q\n' "$wt"
    printf 'env_override=%q\n' "$@"
    printf 'binary=%q\n' ./ds4
    printf 'arg=%q\n' --metal -m "$model" --prompt-file "$DS4_RAW_PROMPT" \
      --raw-prompt --ctx 65536 --tokens 256 --temp 0 --nothink \
      --warm-weights
  } > "$out/command.txt"
  print -rl -- "$@" > "$out/env-overrides.txt"
  git -C "$wt" rev-parse HEAD > "$out/revision.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/start-utc.txt"
  pmset -g therm > "$out/thermal-before.txt" 2>&1 || true
  vm_stat > "$out/vm-before.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-before.txt" 2>&1 || true
  (
    cd "$wt"
    /usr/bin/time -p env -i \
      HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C \
      "$@" \
      ./ds4 --metal \
        -m "$model" --prompt-file "$DS4_RAW_PROMPT" --raw-prompt \
        --ctx 65536 --tokens 256 --temp 0 --nothink --warm-weights
  ) > "$out/generated.txt" 2> "$out/stderr.log"
  pmset -g therm > "$out/thermal-after.txt" 2>&1 || true
  vm_stat > "$out/vm-after.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-after.txt" 2>&1 || true
  shasum -a 256 "$out/generated.txt" > "$out/output-sha256.txt"
}

run_dflash() {
  local run_id=$1
  shift
  local out="$DS4_RESULTS/$run_id"
  mkdir -p "$out"
  {
    printf 'cwd=%q\n' "$DS4_CAND_WT"
    printf 'env_override=%q\n' DS4_DFLASH_TIMING=1 "$@"
    printf 'binary=%q\n' ./ds4
    printf 'arg=%q\n' --metal -m "$DS4_Q23_MODEL" \
      --dflash "$DS4_DFLASH_MODEL" --dflash-draft 3 --dflash-p-min 0 \
      --prompt-file "$DS4_RAW_PROMPT" --raw-prompt --ctx 65536 \
      --tokens 256 --temp 0 --nothink --warm-weights
  } > "$out/command.txt"
  print -rl -- "$@" > "$out/env-overrides.txt"
  git -C "$DS4_CAND_WT" rev-parse HEAD > "$out/revision.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/start-utc.txt"
  pmset -g therm > "$out/thermal-before.txt" 2>&1 || true
  vm_stat > "$out/vm-before.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-before.txt" 2>&1 || true
  (
    cd "$DS4_CAND_WT"
    /usr/bin/time -p env -i \
      HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C DS4_DFLASH_TIMING=1 \
      "$@" \
      ./ds4 --metal \
        -m "$DS4_Q23_MODEL" --dflash "$DS4_DFLASH_MODEL" \
        --dflash-draft 3 --dflash-p-min 0 \
        --prompt-file "$DS4_RAW_PROMPT" --raw-prompt \
        --ctx 65536 --tokens 256 --temp 0 --nothink --warm-weights
  ) > "$out/generated.txt" 2> "$out/stderr.log"
  pmset -g therm > "$out/thermal-after.txt" 2>&1 || true
  vm_stat > "$out/vm-after.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-after.txt" 2>&1 || true
  shasum -a 256 "$out/generated.txt" > "$out/output-sha256.txt"
}
```

Do not add flags to the clean control arm. In particular, do not write
`DS4_METAL_DISABLE_Q23_EXACT_MULTIROW=0`.

## Run the mandatory first pass

Run one unreported warm-up for every distinct arm, then preserve the specified
order. The mirrored three-way order controls first-to-last drift; every
two-arm experiment uses ABBA.

### 1. Default three-way comparison

Run this matrix for both resident quants:

```zsh
# Q2/Q3 warm-ups.
run_bench warmup/default-q23-pre  "$DS4_PRE_WT"  "$DS4_Q23_MODEL"
run_bench warmup/default-q23-base "$DS4_BASE_WT" "$DS4_Q23_MODEL"
run_bench warmup/default-q23-cand "$DS4_CAND_WT" "$DS4_Q23_MODEL"

# Q2/Q3 mirrored A-B-C-C-B-A measurements.
run_bench default-q23/01-pre-A  "$DS4_PRE_WT"  "$DS4_Q23_MODEL"
run_bench default-q23/02-base-B "$DS4_BASE_WT" "$DS4_Q23_MODEL"
run_bench default-q23/03-cand-C "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_bench default-q23/04-cand-C "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_bench default-q23/05-base-B "$DS4_BASE_WT" "$DS4_Q23_MODEL"
run_bench default-q23/06-pre-A  "$DS4_PRE_WT"  "$DS4_Q23_MODEL"

# Q4 warm-ups.
run_bench warmup/default-q4-pre  "$DS4_PRE_WT"  "$DS4_Q4_MODEL"
run_bench warmup/default-q4-base "$DS4_BASE_WT" "$DS4_Q4_MODEL"
run_bench warmup/default-q4-cand "$DS4_CAND_WT" "$DS4_Q4_MODEL"

# Q4 mirrored A-B-C-C-B-A measurements.
run_bench default-q4/01-pre-A  "$DS4_PRE_WT"  "$DS4_Q4_MODEL"
run_bench default-q4/02-base-B "$DS4_BASE_WT" "$DS4_Q4_MODEL"
run_bench default-q4/03-cand-C "$DS4_CAND_WT" "$DS4_Q4_MODEL"
run_bench default-q4/04-cand-C "$DS4_CAND_WT" "$DS4_Q4_MODEL"
run_bench default-q4/05-base-B "$DS4_BASE_WT" "$DS4_Q4_MODEL"
run_bench default-q4/06-pre-A  "$DS4_PRE_WT"  "$DS4_Q4_MODEL"
```

For each quant, all six `decoded.txt` files and all six frontier-logit JSON
files must be byte-identical. Stop and report the mismatch if either parity
gate fails:

```zsh
for matrix in default-q23 default-q4; do
  ref="$DS4_RESULTS/$matrix/01-pre-A"
  for run in "$DS4_RESULTS/$matrix"/0*; do
    cmp "$ref/decoded.txt" "$run/decoded.txt"
    cmp "$ref/logits/frontier_008192.logits.json" \
        "$run/logits/frontier_008192.logits.json"
  done
done
```

### 2. Resident one-row-per-SIMD QMV

Measure Q2/Q3 and Q4 separately. A is the candidate's ordinary QMV path; B
sets `DS4_METAL_GLM_QMV_R1=1` and changes no other variable.

The `run_bench` throughput arm records the prefill frontier at 8192; that JSON
does not by itself exercise or prove the decode-only R1 path. First run the
correctness-only full-model one-token probe below. It advances from 8192 to
8193 with no generated tokens, preserves command/environment/revision/stdout/
stderr/CSV evidence, and requires byte-exact frontier-logit parity between a
clean arm and `DS4_METAL_GLM_QMV_R1=1` for both resident quants.

```zsh
run_decode_probe r1-probe-q23/off "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_decode_probe r1-probe-q23/on  "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_GLM_QMV_R1=1
run_decode_probe r1-probe-q4/off  "$DS4_CAND_WT" "$DS4_Q4_MODEL"
run_decode_probe r1-probe-q4/on   "$DS4_CAND_WT" "$DS4_Q4_MODEL" \
  DS4_METAL_GLM_QMV_R1=1

cmp "$DS4_RESULTS/r1-probe-q23/off/logits/frontier_008193.logits.json" \
    "$DS4_RESULTS/r1-probe-q23/on/logits/frontier_008193.logits.json"
cmp "$DS4_RESULTS/r1-probe-q4/off/logits/frontier_008193.logits.json" \
    "$DS4_RESULTS/r1-probe-q4/on/logits/frontier_008193.logits.json"

# Throughput arms, in ABBA order, after the correctness probe passes.
run_bench warmup/r1-q23-A "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_bench warmup/r1-q23-B "$DS4_CAND_WT" "$DS4_Q23_MODEL" DS4_METAL_GLM_QMV_R1=1
run_bench r1-q23/01-off-A "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_bench r1-q23/02-on-B  "$DS4_CAND_WT" "$DS4_Q23_MODEL" DS4_METAL_GLM_QMV_R1=1
run_bench r1-q23/03-on-B  "$DS4_CAND_WT" "$DS4_Q23_MODEL" DS4_METAL_GLM_QMV_R1=1
run_bench r1-q23/04-off-A "$DS4_CAND_WT" "$DS4_Q23_MODEL"

run_bench warmup/r1-q4-A "$DS4_CAND_WT" "$DS4_Q4_MODEL"
run_bench warmup/r1-q4-B "$DS4_CAND_WT" "$DS4_Q4_MODEL" DS4_METAL_GLM_QMV_R1=1
run_bench r1-q4/01-off-A "$DS4_CAND_WT" "$DS4_Q4_MODEL"
run_bench r1-q4/02-on-B  "$DS4_CAND_WT" "$DS4_Q4_MODEL" DS4_METAL_GLM_QMV_R1=1
run_bench r1-q4/03-on-B  "$DS4_CAND_WT" "$DS4_Q4_MODEL" DS4_METAL_GLM_QMV_R1=1
run_bench r1-q4/04-off-A "$DS4_CAND_WT" "$DS4_Q4_MODEL"
```

Within each matrix, compare every `decoded.txt` and frontier-logit JSON with
`01-off-A` before using its timings.

The R1 gate is therefore the decoded-output parity from the measured arm, the
8193 frontier probe above, and the exact Metal-kernel tests above; do not describe the 8192 frontier JSON alone as an R1 correctness
result.

### 3. Full-ring Laguna SWA GQA3

This is an isolated candidate experiment for the production Laguna SWA shape:
72 query heads, 8 KV heads, head dimension 128, and a 512-slot ring at a full
512-key decode. `DS4_METAL_LAGUNA_SWA_GQA3=1` is default-off and reuses the
existing grouped GQA3 decode pipeline. The clean arm has no override; B sets
only this flag. Every helper uses `env -i`, so `DS4_METAL_LAGUNA_STAGED_SWA` is
absent even if it was exported by the calling shell. Never combine the flags:
when both are exported in an ad hoc run, staged SWA takes precedence and emits
a diagnostic, so such a run is not evidence for grouped GQA3 speed.

First run the correctness probes, which advance from 8192 to 8193 without
generated tokens. The 8193 vectors are allowed to differ in low bits because
the ordinary and grouped reductions use different partition/order; the Python
helper requires equal vocabulary, finite values, and equal argmax, then writes
the measured drift to a preserved CSV and SHA-256 sidecar. It does not fail
solely on the uncalibrated first-pass magnitude bound.

```zsh
run_decode_probe swa-gqa3-probe-q23/off "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_decode_probe swa-gqa3-probe-q23/on  "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_SWA_GQA3=1
run_decode_probe swa-gqa3-probe-q4/off  "$DS4_CAND_WT" "$DS4_Q4_MODEL"
run_decode_probe swa-gqa3-probe-q4/on   "$DS4_CAND_WT" "$DS4_Q4_MODEL" \
  DS4_METAL_LAGUNA_SWA_GQA3=1

compare_decode_logits \
  swa-gqa3-probe-q23/off swa-gqa3-probe-q23/on \
  swa-gqa3-q23/8193-drift.csv
compare_decode_logits \
  swa-gqa3-probe-q4/off swa-gqa3-probe-q4/on \
  swa-gqa3-q4/8193-drift.csv
```

Then measure the 8192-prefill control and generation in ABBA order. The
frontier JSON at 8192 must remain byte-identical across the four measured arms;
all generated token text must also remain byte-identical. Only B receives the
GQA3 override.

```zsh
run_bench warmup/swa-gqa3-q23-off "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_bench warmup/swa-gqa3-q23-on  "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_SWA_GQA3=1
run_bench swa-gqa3-q23/01-off-A "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_bench swa-gqa3-q23/02-on-B  "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_SWA_GQA3=1
run_bench swa-gqa3-q23/03-on-B  "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_SWA_GQA3=1
run_bench swa-gqa3-q23/04-off-A "$DS4_CAND_WT" "$DS4_Q23_MODEL"

run_bench warmup/swa-gqa3-q4-off "$DS4_CAND_WT" "$DS4_Q4_MODEL"
run_bench warmup/swa-gqa3-q4-on  "$DS4_CAND_WT" "$DS4_Q4_MODEL" \
  DS4_METAL_LAGUNA_SWA_GQA3=1
run_bench swa-gqa3-q4/01-off-A "$DS4_CAND_WT" "$DS4_Q4_MODEL"
run_bench swa-gqa3-q4/02-on-B  "$DS4_CAND_WT" "$DS4_Q4_MODEL" \
  DS4_METAL_LAGUNA_SWA_GQA3=1
run_bench swa-gqa3-q4/03-on-B  "$DS4_CAND_WT" "$DS4_Q4_MODEL" \
  DS4_METAL_LAGUNA_SWA_GQA3=1
run_bench swa-gqa3-q4/04-off-A "$DS4_CAND_WT" "$DS4_Q4_MODEL"

for quant in q23 q4; do
  for run in 01-off-A 02-on-B 03-on-B 04-off-A; do
    cmp "$DS4_RESULTS/swa-gqa3-$quant/$run/decoded.txt" \
        "$DS4_RESULTS/swa-gqa3-$quant/01-off-A/decoded.txt"
    cmp "$DS4_RESULTS/swa-gqa3-$quant/$run/logits/frontier_008192.logits.json" \
        "$DS4_RESULTS/swa-gqa3-$quant/01-off-A/logits/frontier_008192.logits.json"
  done
done
```

Record `gen_steady_tps` as the primary performance metric, with `prefill_tps`
as the unaffected control. Record both B observations and their median against
both A observations and their median. Also copy the Q2/Q3 and Q4 8193 CSV rows
into `SUMMARY.md`, including max absolute drift, RMS, p99 absolute drift,
cosine, argmax IDs/logits/margins, top-10 overlap, and the helper's
`investigation_flag`.

### 4. Paired Laguna prefill Q/K norm and RoPE

A is ordinary prefill; B sets only
`DS4_LAGUNA_PREFILL_QK_NORM_ROPE_PAIRED=1`.

```zsh
run_bench warmup/paired-prefill-A "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_bench warmup/paired-prefill-B "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_LAGUNA_PREFILL_QK_NORM_ROPE_PAIRED=1
run_bench paired-prefill/01-off-A "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_bench paired-prefill/02-on-B  "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_LAGUNA_PREFILL_QK_NORM_ROPE_PAIRED=1
run_bench paired-prefill/03-on-B  "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_LAGUNA_PREFILL_QK_NORM_ROPE_PAIRED=1
run_bench paired-prefill/04-off-A "$DS4_CAND_WT" "$DS4_Q23_MODEL"
```

Compare every `decoded.txt` and frontier-logit JSON with `01-off-A`. The
primary metric here is `prefill_tps`; generation must not regress beyond the
noise threshold.

### 5. Direct raw GPU argmax

This experiment must use `ds4`, not `ds4-bench`: the optimisation applies to
the direct, raw, single-tier Metal greedy path. A performs the ordinary logits
readback and CPU argmax; B sets only `DS4_METAL_LAGUNA_GPU_ARGMAX=1`.

```zsh
run_raw warmup/gpu-argmax-A "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_raw warmup/gpu-argmax-B "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1
run_raw gpu-argmax/01-off-A "$DS4_CAND_WT" "$DS4_Q23_MODEL"
run_raw gpu-argmax/02-on-B  "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1
run_raw gpu-argmax/03-on-B  "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1
run_raw gpu-argmax/04-off-A "$DS4_CAND_WT" "$DS4_Q23_MODEL"
```

All four `generated.txt` files must be byte-identical. Both B logs must contain
`Laguna GPU argmax enabled (single-dispatch)`; neither A log may contain it.
Use the `ds4: Laguna prefill: ..., generation: ...` line for throughput.

### 6. DFlash Q2/Q3 verifier batching and staged SWA

The long prompt places the target verifier beyond the 512-token SWA wrap.
Draft width is fixed at three and the confidence cutoff is zero so verifier
shape cannot vary between arms.

- A sets `DS4_METAL_DISABLE_Q23_EXACT_MULTIROW=1`, restoring the legacy Q2/Q3
  verifier row loop.
- B is the clean candidate default: exact multirow verifier batching is on and
  staged SWA is off.
- C adds staged SWA and require mode. Require mode turns a missing staged path
  into an error rather than silently falling back.

```zsh
run_dflash warmup/dflash-A DS4_METAL_DISABLE_Q23_EXACT_MULTIROW=1
run_dflash warmup/dflash-B
run_dflash warmup/dflash-C DS4_METAL_LAGUNA_STAGED_SWA=1 DS4_METAL_LAGUNA_REQUIRE_STAGED_SWA=1

run_dflash dflash/01-legacy-multirow-A DS4_METAL_DISABLE_Q23_EXACT_MULTIROW=1
run_dflash dflash/02-default-B
run_dflash dflash/03-staged-C \
  DS4_METAL_LAGUNA_STAGED_SWA=1 DS4_METAL_LAGUNA_REQUIRE_STAGED_SWA=1
run_dflash dflash/04-staged-C \
  DS4_METAL_LAGUNA_STAGED_SWA=1 DS4_METAL_LAGUNA_REQUIRE_STAGED_SWA=1
run_dflash dflash/05-default-B
run_dflash dflash/06-legacy-multirow-A DS4_METAL_DISABLE_Q23_EXACT_MULTIROW=1
```

All six `generated.txt` files must be byte-identical. Every measured log must
contain multiple `DFlash cycle drafted=... verified=... accepted=...
pipeline=... ms` lines. Reject the matrix if any log reports DFlash being
paused, a required staged pipeline being unavailable, verifier failure, or
fallback. Compare A to B for exact Q2/Q3 multirow batching, and B to C for
staged SWA.

## Check noise and correctness

Before calculating speedups, apply all of these gates:

- every command exited zero;
- all required byte-parity comparisons passed (the 8193 GQA3 probe uses the
  numeric CSV helper rather than a byte comparison);
- the GQA3 8193 helper reports equal vocabulary, finite values, and equal
  argmax IDs for both Q2/Q3 and Q4; a drift flag is observational on this
  first M5 pass, but must be investigated before promotion;
- GQA3 measured arms have byte-identical generated text and 8192 frontier JSON
  between off/on; no staged-SWA override appears in their environment files;
- `kvcache_bytes`, generated-token counts, and DFlash drafted/verified/accepted
  counts agree between paired arms;
- no measured stderr contains an unexpected `failed`, `unavailable`,
  `fallback`, `paused`, `mismatch`, or invalid-token message;
- thermal pressure remained nominal and no swap or memory-pressure event
  occurred during one arm but not its peer;
- the first and last A observations differ by no more than 3% for the primary
  metric.

If the two A observations drift by more than 3%, discard the matrix, cool the
machine, and rerun its warm-ups and ABBA/ABCCBA sequence. If an apparent delta
is below 1%, treat it as noise until at least five alternating observations per
arm reproduce the same direction.

## Metrics to calculate

For every `ds4-bench` arm, preserve both observations and report their median:

- `prefill_tps`;
- `gen_tps`;
- `gen_first_ms`;
- `gen_steady_tps`;
- `kvcache_bytes`;
- percentage delta of B or C relative to its named control.

For full-ring SWA GQA3, report Q2/Q3 and Q4 independently:

- off/on `gen_steady_tps` observations, medians, and B-versus-A percentage;
- `prefill_tps` from the 8192 unaffected-prefill control;
- 8193 `max_abs`, RMS, p99 absolute drift, cosine, argmax IDs/logits/margins,
  and top-10 overlap from the preserved CSV;
- whether the 8192 frontier and generated text parity gates passed;
- whether the numeric helper set `investigation_flag`.

For GPU argmax, report both raw observations of Laguna generation tokens/s,
their median, and B versus A percentage change. Also report Laguna prefill
tokens/s as a control metric.

For every DFlash arm, report:

- overall prefill and generation tokens/s from the final DS4 timing line;
- number of DFlash cycles;
- sum of `drafted`, `verified`, and `accepted`;
- accepted/verified ratio;
- median and p95 `pipeline` milliseconds;
- generated-output SHA-256;
- whether speculation remained active for the whole run.

Use `(new / control - 1) * 100` for throughput deltas and
`(control / new - 1) * 100` for latency improvements. Do not average tokens/s
across different contexts or quants.

## Report the result

Create `SUMMARY.md` in `$DS4_RESULTS` with these tables:

1. Host identity: exact Mac model, chip, GPU cores, RAM, macOS build, Xcode/Clang
   version, power mode, display configuration, and thermal status.
2. Input identity: all three source SHAs and all model/prompt SHA-256 values.
3. Default Q2/Q3 and Q4: the two observations and median for each revision,
   followed by the three separately labelled deltas.
4. Isolated experiments: R1 Q2/Q3, R1 Q4, full-ring SWA GQA3 Q2/Q3, full-ring
   SWA GQA3 Q4, paired prefill, and GPU argmax, with A/B observations,
   medians, percentage deltas, parity status, and the 8193 numeric CSV fields.
5. DFlash: the A/B/C metrics above, with A to B and B to C deltas.
6. A decision for each opt-in path: `promote`, `keep opt-in`, `reject`, or
   `rerun`, with one sentence grounded in the measured metric and noise gate.
7. Every warning, fallback, thermal excursion, failed parity check, or rerun.

As initial decision thresholds, promote an experiment only when parity passes
and its relevant median improves by at least 1.5% without a greater than 1%
regression in an unaffected primary metric. A smaller positive result remains
opt-in pending more repetitions. For staged SWA, require both lower DFlash
pipeline latency and higher end-to-end generation throughput. For full-ring
SWA GQA3, reject promotion if generated text or argmax differs, any probe
value is non-finite, the numeric helper reports material drift (`max_abs` >
1e-3 or RMS > 1e-4), or the median `gen_steady_tps` improvement is below
1.5%. The first M5 pass must still record and return a numeric drift result
instead of aborting solely on those uncalibrated magnitude thresholds; mark it
for investigation and keep the flag opt-in until calibrated.

## Package the evidence

Record final host state, add a manifest, and archive only results—not models or
worktrees:

```zsh
{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  pmset -g batt
  pmset -g therm
  memory_pressure -Q
  vm_stat
} > "$DS4_RESULTS/metadata/host-after.txt" 2>&1

find "$DS4_RESULTS" -type f -print | LC_ALL=C sort \
  > "$DS4_RESULTS/MANIFEST.txt"
ARCHIVE="$DS4_BENCH_ROOT/ds4-laguna-m5-max-results.tar.gz"
tar -C "$DS4_BENCH_ROOT" -czf "$ARCHIVE" results
shasum -a 256 "$ARCHIVE" | tee "$ARCHIVE.sha256"
print "Return $DS4_RESULTS/SUMMARY.md, $ARCHIVE, and $ARCHIVE.sha256"
```

Keep `$DS4_BENCH_ROOT` until the archive has been received and verified.

## Extended sweep after the first return

Do not block the first return on this section. Once the winning candidate flag
set is agreed, rerun the three default revisions and the winning candidate at
the repository's standard context frontiers using
`speed-bench/promessi_sposi.txt`:

```zsh
# Keep this as an array: an empty array is the clean default arm, while each
# candidate override is one complete NAME=VALUE argument to env.
typeset -a approved_candidate_env=()
# Example winning candidate arm (approved flags may be combined except staged SWA):
# approved_candidate_env=(
#   DS4_METAL_GLM_QMV_R1=1
#   DS4_LAGUNA_PREFILL_QK_NORM_ROPE_PAIRED=1
#   DS4_METAL_LAGUNA_SWA_GQA3=1
# )
# Never put DS4_METAL_LAGUNA_SWA_GQA3=1 and DS4_METAL_LAGUNA_STAGED_SWA=1
# in the same approved array. Staged SWA takes precedence for the production
# 512-slot route and emits a diagnostic; benchmark and promote them separately.

worktree="$DS4_CAND_WT"
model="$DS4_Q4_MODEL"
output_dir="$DS4_RESULTS/extended/candidate-q4-approved"
output_csv="$output_dir/metrics.csv"
mkdir -p "$output_dir"
{
  printf 'cwd=%q\n' "$worktree"
  printf 'binary=%q\n' ./ds4-bench
  printf 'arg=%q\n' --metal -m "$model" --prompt-file "$DS4_LONG_PROMPT" \
    --ctx-start 2048 --ctx-max 65536 --ctx-alloc 65665 \
    --step-incr 2048 --gen-tokens 128 --warm-weights --csv "$output_csv"
  printf 'env_override_count=%q\n' "${#approved_candidate_env[@]}"
  for env_name in "${approved_candidate_env[@]}"; do
    printf 'env_override=%q\n' "$env_name"
  done
} > "$output_dir/command.txt"
print -rl -- "${approved_candidate_env[@]}" > "$output_dir/env-overrides.txt"
git -C "$worktree" rev-parse HEAD > "$output_dir/revision.txt"

(
  cd "$worktree"
  env -i \
    HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
    "${approved_candidate_env[@]}" \
    ./ds4-bench --metal -m "$model" \
      --prompt-file "$DS4_LONG_PROMPT" \
      --ctx-start 2048 --ctx-max 65536 --ctx-alloc 65665 \
      --step-incr 2048 --gen-tokens 128 --warm-weights \
      --csv "$output_csv"
) > "$output_dir/stdout.log" 2> "$output_dir/stderr.log"
```

Use absolute `worktree`, `model`, and `output_csv` paths. Run clean default
revisions with no candidate flags. Use mirrored order, preserve raw logs, and
report each context separately at 2K increments. If a promising delta remains
below 1% in the first pass, use five alternating repetitions at the 2K, 16K,
32K, and 65K frontiers before making a promotion decision.

Keep GPU argmax on the direct `ds4` command, and keep staged-SWA and Q2/Q3
multirow DFlash extensions on the raw/DFlash commands above; those paths are
not `ds4-bench` options.
