# Laguna S 2.1 benchmark protocol for M5 Max

Use this runbook to compare the preserved pre-rebase Laguna branch, the
rebased baseline, and the current optimisation candidate on a 128 GB M5 Max.
The current requested return is deliberately bounded: run the clean Q2/Q3
three-way comparison and the candidate-only Laguna Q8 lm-head experiment.
The older R1, GQA3, paired-prefill, raw-argmax, Q4, and DFlash procedures are
retained below as optional historical material and are not part of this
return.

Run every binary from its own worktree. DS4 loads `metal/*.metal` relative to
the current working directory; invoking a binary by absolute path from another
checkout can silently combine a binary from one revision with Metal source
from another.

## Return contract

Return both of these:

1. A `SUMMARY.md` containing the required tables and observations in
   [Report the result](#report-the-result).  For this return, that means only
   the clean Q2/Q3 three-way and candidate lm-head A–E tables.
2. The compressed raw-results archive produced in
   [Package the evidence](#package-the-evidence), including its SHA-256.

Do not return only a headline tokens/second number. A result is usable only
when its revision, model, command, raw CSV or applicable timing log, stdout,
stderr, parity evidence, and machine state are present.

## Fixed revisions

| Role | Immutable commit | Purpose |
| --- | --- | --- |
| Pre-rebase/basic | `448d5695d1c86401a4e9447c440feb983b73e6de` | Preserved Laguna lineage before rebasing onto `main` |
| Rebased baseline | `729e0cedf54dccfef93fac5138b26d1157aea56a` | Same Laguna tip rebased onto `main`, before the new optimisation series |
| Candidate | `202a46b5158bff38ba36e066d71bc5b242b53a4a` | Current source candidate, including the certified Q8 lm-head screen |

Report these deltas separately for the clean Q2/Q3 matrix:

- pre-rebase/basic to rebased baseline: effect of integrating current `main`;
- rebased baseline to candidate: effect of the new optimisation series;
- pre-rebase/basic to candidate: total branch improvement.

Never substitute a branch name or a newer branch tip for these commits.

## Prepare the machine

Use the M5 Max while plugged into power, with Low Power Mode disabled and no
other GPU-, CPU-, disk-, or memory-intensive work running. Let the machine
return to nominal thermal pressure before starting. Keep the same power mode,
display arrangement, and terminal session for the complete first pass.

The required model for this bounded return is:

- `laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf`;

The Q4 and DFlash models remain optional for the historical procedures below;
do not download or require them for this return.

Allow ample free disk space for the Q2/Q3 model, worktrees, builds, and
results. Set an absolute model directory before continuing:

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
export DS4_CAND_SHA=202a46b5158bff38ba36e066d71bc5b242b53a4a
export DS4_PRE_WT="$DS4_BENCH_ROOT/pre-rebase"
export DS4_BASE_WT="$DS4_BENCH_ROOT/rebased-baseline"
export DS4_CAND_WT="$DS4_BENCH_ROOT/candidate"

[[ -d "$DS4_BENCH_ROOT" ]] || mkdir -p "$DS4_BENCH_ROOT" || {
  print -u2 "cannot create DS4_BENCH_ROOT: $DS4_BENCH_ROOT"
  return 2
}
if [[ -e "$DS4_CONTROL_REPO" || -e "$DS4_RESULTS" ]]; then
  print -u2 'DS4_BENCH_ROOT already contains control/ or results; use a fresh root'
  return 2
fi
if [[ -n "$(command ls -A "$DS4_BENCH_ROOT" 2>/dev/null)" ]]; then
  print -u2 'DS4_BENCH_ROOT must be empty before creating the benchmark run'
  return 2
fi

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

typeset -a revision_labels revision_worktrees revision_shas
revision_labels=(pre-rebase rebased-baseline candidate)
revision_worktrees=("$DS4_PRE_WT" "$DS4_BASE_WT" "$DS4_CAND_WT")
revision_shas=("$DS4_PRE_SHA" "$DS4_BASE_SHA" "$DS4_CAND_SHA")
for index in {1..3}; do
  label=${revision_labels[index]}
  wt=${revision_worktrees[index]}
  expected=${revision_shas[index]}
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
export DS4_Q23_MODEL="$DS4_GGUF_DIR/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf"
export DS4_Q4_MODEL="$DS4_GGUF_DIR/laguna-s-2.1-Q4_K_M.gguf"
export DS4_DFLASH_MODEL="$DS4_GGUF_DIR/laguna-s-2.1-DFlash-Q8_0.gguf"
export DS4_PROMPT="$DS4_CAND_WT/tests/long_context_story_prompt.txt"
export DS4_LONG_PROMPT="$DS4_CAND_WT/speed-bench/promessi_sposi.txt"
export DS4_RAW_PROMPT="$DS4_RESULTS/metadata/promessi-first-600-lines.txt"

for file_path in "$DS4_Q23_MODEL" "$DS4_PROMPT" "$DS4_LONG_PROMPT"; do
  [[ -f "$file_path" ]] || { print -u2 "Missing required file: $file_path"; return 2; }
done

[[ "$(shasum -a 256 "$DS4_PROMPT" | awk '{print $1}')" == \
   29363eab21bbbccaeea8e13f669e7ce05e8eafc48e31fcf9b725edabb2058666 ]]
[[ "$(shasum -a 256 "$DS4_LONG_PROMPT" | awk '{print $1}')" == \
   f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f ]]
sed -n '1,600p' "$DS4_LONG_PROMPT" > "$DS4_RAW_PROMPT"
[[ "$(shasum -a 256 "$DS4_RAW_PROMPT" | awk '{print $1}')" == \
   a50224adb48545ded1a76ab176543812a1afb5e76f413dbd7fe8cf256cb9c1c2 ]]

shasum -a 256 "$DS4_Q23_MODEL" "$DS4_PROMPT" "$DS4_LONG_PROMPT" \
  "$DS4_RAW_PROMPT" \
  > "$DS4_RESULTS/metadata/input-sha256.txt"

print "Optional historical models, if already available:"
print "  Q4: $DS4_Q4_MODEL"
print "  DFlash: $DS4_DFLASH_MODEL"
```

Model hashing reads the required Q2/Q3 model (roughly 45 GiB) once. Complete
it before timing rather than between measured arms.

## Record the host and build each revision

```zsh
capture_profiler_json() {
  local output=$1
  shift
  env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
    LANG=C LC_ALL=C AppleLanguages='(en)' \
    system_profiler -json "$@" 2>&1 |
    python3 -c '
import json
import re
import sys

secret_keys = {
    "serial_number", "platform_uuid", "provisioning_udid",
    "sppower_battery_serial_number", "sppower_ac_charger_serial_number",
    "_spdisplays_display_serial_number", "computer_name", "host_name",
    "hostname", "network_name",
}

def key_name(key):
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")

def redact(value):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            normalized = key_name(key)
            private = normalized in secret_keys or (
                "serial" in normalized and "number" in normalized
            ) or normalized.endswith("_uuid") or normalized.endswith("_udid")
            result[key] = "<redacted>" if private else redact(child)
        return result
    if isinstance(value, list):
        return [redact(child) for child in value]
    return value

try:
    document = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError):
    document = {"_unavailable": True}
    parse_ok = False
else:
    parse_ok = True
json.dump(redact(document), sys.stdout, sort_keys=True, indent=2)
sys.stdout.write("\n")
if not parse_ok:
    raise SystemExit(1)
' > "$output"
}

thermal_is_nominal() {
  local text=$1
  print -r -- "$text" | awk '
    BEGIN {
      thermal_ok = 0
      thermal_bad = 0
      performance_ok = 0
      performance_bad = 0
      cpu_ok = 0
      cpu_bad = 0
      saw_thermal = 0
      saw_performance = 0
      saw_cpu = 0
    }
    {
      line = tolower($0)
      if (line ~ /thermal pressure:/) {
        saw_thermal = 1
        if (line ~ /nominal/) {
          if (!thermal_bad) thermal_ok = 1
        } else {
          thermal_bad = 1
          thermal_ok = 0
        }
      }
      if (line ~ /thermal level:/) {
        saw_thermal = 1
        if (line ~ /: *0([[:space:]]|$)/) {
          if (!thermal_bad) thermal_ok = 1
        } else {
          thermal_bad = 1
          thermal_ok = 0
        }
      }
      if (line ~ /no thermal warning level has been recorded/) {
        saw_thermal = 1
        if (!thermal_bad) thermal_ok = 1
      }
      if (line ~ /thermal warning level/ &&
          line !~ /no thermal warning level has been recorded/) {
        saw_thermal = 1
        thermal_bad = 1
        thermal_ok = 0
      }
      if (line ~ /performance:/) {
        saw_performance = 1
        if (line ~ /nominal|normal|: *0([[:space:]]|$)/) {
          if (!performance_bad) performance_ok = 1
        } else {
          performance_bad = 1
          performance_ok = 0
        }
      }
      if (line ~ /no performance warning level has been recorded/) {
        saw_performance = 1
        if (!performance_bad) performance_ok = 1
      }
      if (line ~ /performance warning level/ &&
          line !~ /no performance warning level has been recorded/) {
        saw_performance = 1
        performance_bad = 1
        performance_ok = 0
      }
      if (line ~ /cpu power:/) {
        saw_cpu = 1
        if (line ~ /warning|reduced|critical|serious|fair|throttl|elevated/) {
          cpu_bad = 1
          cpu_ok = 0
        } else if (line ~ /nominal|normal|: *0([[:space:]]|$)/ && !cpu_bad) {
          cpu_ok = 1
        }
      }
      if (line ~ /no cpu power status has been recorded/) {
        saw_cpu = 1
        if (!cpu_bad) cpu_ok = 1
      }
      if (line ~ /cpu power status/ &&
          line !~ /no cpu power status has been recorded/) {
        saw_cpu = 1
        cpu_bad = 1
        cpu_ok = 0
      }
      if (line ~ /cpu power/ && line !~ /cpu power:|no cpu power status has been recorded/) {
        saw_cpu = 1
        if (line !~ /no cpu power status has been recorded/) {
          cpu_bad = 1
          cpu_ok = 0
        }
      }
    }
    END {
      exit !(thermal_ok && !thermal_bad && saw_thermal &&
             performance_ok && !performance_bad && saw_performance &&
             cpu_ok && !cpu_bad && saw_cpu)
    }
  '
}

capture_machine_state() {
  local output_dir=$1
  local phase=$2
  pmset -g custom > "$output_dir/power-custom-$phase.txt" 2>&1 || true
  pmset -g batt > "$output_dir/battery-$phase.txt" 2>&1 || true
  pmset -g therm > "$output_dir/thermal-$phase.txt" 2>&1 || true
  capture_profiler_json "$output_dir/power-profile-$phase.txt" \
    SPPowerDataType
  sysctl vm.swapusage > "$output_dir/vm-swapusage-$phase.txt" 2>&1 || true
  vm_stat > "$output_dir/vm-$phase.txt" 2>&1 || true
  memory_pressure -Q > "$output_dir/memory-pressure-$phase.txt" 2>&1 || true
}

verify_power_and_thermals() {
  local custom batt therm power_profile
  custom=$(pmset -g custom 2>&1) || return 1
  batt=$(pmset -g batt 2>&1) || return 1
  therm=$(pmset -g therm 2>&1) || return 1
  power_profile=$(capture_profiler_json /dev/stdout SPPowerDataType) || return 1
  print -r -- "$custom"
  print -r -- "$batt"
  print -r -- "$therm"
  print -r -- "$power_profile"
  if ! print -r -- "$power_profile" | python3 -c '
import json
import re
import sys

document = json.load(sys.stdin)
high = False
low = False
current_ac = False

def walk(node, in_ac=False):
    global high, low, current_ac
    if isinstance(node, dict):
        names = [str(value).strip().lower() for key, value in node.items()
                 if str(key).lower() in {"_name", "name"}]
        ac = in_ac or "ac power" in names
        for key, value in node.items():
            normalized = re.sub(r"[^a-z0-9]+", "", str(key).lower())
            child_ac = ac or normalized == "acpower"
            value_text = str(value).strip().lower()
            if child_ac and normalized in {"highpowermode", "highpowermodeenabled"}:
                high = high or value_text == "yes"
            if child_ac and normalized in {"lowpowermode", "lowpowermodeenabled"}:
                low = low or value_text == "no"
            if child_ac and normalized in {"currentpowersource", "currentpowersourceconnected"}:
                current_ac = current_ac or value_text in {"yes", "true", "1", "ac power"}
            walk(value, child_ac)
    elif isinstance(node, list):
        for child in node:
            walk(child, in_ac)

walk(document)
raise SystemExit(0 if high and low and current_ac else 1)
'; then
    print -u2 'AC profile does not verify current AC, High Power Mode=Yes, and Low Power Mode=No'
    return 1
  fi
  thermal_is_nominal "$therm" || {
    print -u2 'thermal state is not nominal'
    return 1
  }
}

wait_for_nominal() {
  local output=$1
  local benchmark_start_therm=''
  for attempt in {1..120}; do
    benchmark_start_therm=$(pmset -g therm 2>&1 || true)
    if thermal_is_nominal "$benchmark_start_therm"; then
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
    pmset -g custom
    pmset -g batt
    sysctl vm.swapusage
    memory_pressure -Q
    vm_stat
  } > "$output" 2>&1
}

verify_power_and_thermals > "$DS4_RESULTS/metadata/power-before.txt" 2>&1 || return 1
{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  uname -srvmp
  sw_vers
  sysctl -n machdep.cpu.brand_string
  sysctl -n hw.memsize
  sysctl -n hw.logicalcpu
  env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
    LANG=C LC_ALL=C xcodebuild -version
  env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
    LANG=C LC_ALL=C cc --version
  capture_profiler_json /dev/stdout \
    SPHardwareDataType SPDisplaysDataType SPPowerDataType
  pmset -g custom
  pmset -g batt
  pmset -g therm
  sysctl vm.swapusage
  memory_pressure -Q
  vm_stat
  df -h "$DS4_BENCH_ROOT" "$DS4_GGUF_DIR"
} > "$DS4_RESULTS/metadata/host-before.txt" 2>&1

# System-profiler identity fields are replaced with the literal <redacted>
# placeholder before any snapshot is retained or packaged.
host_chip=$(sysctl -n machdep.cpu.brand_string)
host_memsize=$(sysctl -n hw.memsize)
[[ "$host_chip" == *'Apple M5 Max'* ]] || {
  print -u2 "requires Apple M5 Max, found: $host_chip"
  return 1
}
[[ "$host_memsize" == 137438953472 ]] || {
  print -u2 "requires exactly 137438953472 bytes RAM, found: $host_memsize"
  return 1
}

JOBS=$(sysctl -n hw.logicalcpu)
build_revision() {
  local label=$1
  local wt=$2
  shift 2
  local out="$DS4_RESULTS/metadata/build-$label"
  local -a make_args
  make_args=("$@")
  mkdir -p "$out"
  {
    printf 'cwd=%q\n' "$wt"
    git -C "$wt" rev-parse HEAD
    printf 'env=HOME=%q PATH=%q TMPDIR=%q LANG=C LC_ALL=C\n' \
      "$HOME" "$PATH" "${TMPDIR:-/tmp}"
    printf 'arg='; printf '%q ' make "${make_args[@]}"; print
  } > "$out/command.txt"
  git -C "$wt" rev-parse HEAD > "$out/revision.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/start-utc.txt"
  (
    cd "$wt"
    env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C make "${make_args[@]}"
  ) > "$out/stdout.log" 2> "$out/stderr.log"
  shasum -a 256 "$out/stdout.log" "$out/stderr.log" > "$out/output-sha256.txt"
}

build_revision pre-rebase "$DS4_PRE_WT" -j"$JOBS" ds4 ds4-bench
build_revision rebased-baseline "$DS4_BASE_WT" -j"$JOBS" ds4 ds4-bench
build_revision candidate "$DS4_CAND_WT" -j"$JOBS" ds4 ds4-bench
build_revision candidate-ds4-test "$DS4_CAND_WT" -B -j"$JOBS" ds4_test

metal_dir="$DS4_RESULTS/metadata/metal-kernels"
mkdir -p "$metal_dir"
{
  printf 'cwd=%q\n' "$DS4_CAND_WT"
  git -C "$DS4_CAND_WT" rev-parse HEAD
  printf 'env=HOME=%q PATH=%q TMPDIR=%q LANG=C LC_ALL=C\n' \
    "$HOME" "$PATH" "${TMPDIR:-/tmp}"
  printf 'binary=%q\n' ./ds4_test
  printf 'arg=%q\n' --metal-kernels
} > "$metal_dir/command.txt"
print -rl -- HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
  > "$metal_dir/env-overrides.txt"
git -C "$DS4_CAND_WT" rev-parse HEAD > "$metal_dir/revision.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$metal_dir/start-utc.txt"
(
  cd "$DS4_CAND_WT"
  env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
    LANG=C LC_ALL=C ./ds4_test --metal-kernels
) > "$metal_dir/stdout.log" 2> "$metal_dir/stderr.log"
shasum -a 256 "$metal_dir/stdout.log" "$metal_dir/stderr.log" \
  > "$metal_dir/output-sha256.txt"

smoke_dir="$DS4_RESULTS/metadata/candidate-lmhead-smoke"
mkdir -p "$smoke_dir"
typeset -a smoke_env
smoke_env=(
  DS4_TEST_LAGUNA_Q8_LMHEAD_SCREEN=1
  DS4_METAL_MATH_SAFE=1
)
{
  printf 'cwd=%q\n' "$DS4_CAND_WT"
  git -C "$DS4_CAND_WT" rev-parse HEAD
  printf 'env_override_count=%q\n' "${#smoke_env[@]}"
  for env_name in "${smoke_env[@]}"; do printf 'env_override=%q\n' "$env_name"; done
  printf 'binary=%q\n' ./ds4_test
  printf 'arg=%q\n' --metal-laguna-q8-lmhead-screen
} > "$smoke_dir/command.txt"
print -rl -- "${smoke_env[@]}" > "$smoke_dir/env-overrides.txt"
git -C "$DS4_CAND_WT" rev-parse HEAD > "$smoke_dir/revision.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$smoke_dir/start-utc.txt"
capture_machine_state "$smoke_dir" before
(
  cd "$DS4_CAND_WT"
  env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
    LANG=C LC_ALL=C "${smoke_env[@]}" \
    ./ds4_test --metal-laguna-q8-lmhead-screen
) > "$smoke_dir/stdout.log" 2> "$smoke_dir/stderr.log"
capture_machine_state "$smoke_dir" after
shasum -a 256 "$smoke_dir/stdout.log" "$smoke_dir/stderr.log" \
  > "$smoke_dir/output-sha256.txt"
wait_for_nominal "$smoke_dir/cooldown-after.txt" || return 1
verify_power_and_thermals > "$smoke_dir/power-after-cooldown.txt" 2>&1 || return 1

for wt in "$DS4_PRE_WT" "$DS4_BASE_WT" "$DS4_CAND_WT"; do
  git -C "$wt" status --short --branch
done > "$DS4_RESULTS/metadata/worktree-status-after-build.txt"
```

Stop if the Metal kernel suite or focused lm-head smoke exits non-zero.
Compiler warnings may be recorded,
but a missing optional pipeline, staged-SWA fallback in require mode, numerical
mismatch, or test failure invalidates the candidate run.

## Cool down before benchmark warm-ups

The build and Metal smoke tests can heat the machine. Do not start any warm-up
until the thermal state is nominal again. This bounded wait is unattended; it
polls `pmset` for up to one hour, then records the exact start state used by the
first matrix.

```zsh
wait_for_nominal "$DS4_RESULTS/metadata/benchmark-start.txt" || return 1
verify_power_and_thermals > "$DS4_RESULTS/metadata/power-at-benchmark-start.txt" 2>&1 || return 1
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

require_gpu_argmax_diagnostic() {
  local stderr_path=$1
  grep -Fq 'Laguna GPU argmax enabled (single-dispatch)' "$stderr_path" || {
    semantic_fail "GPU argmax diagnostic missing: $stderr_path"
  }
}

typeset -ga semantic_failures
semantic_failures=()
semantic_fail() {
  local message=$*
  semantic_failures+=("$message")
  print -u2 "semantic gate failed (evidence retained): $message"
  return 0
}

compare_files_semantic() {
  local label=$1
  local left=$2
  local right=$3
  if cmp -s "$left" "$right"; then
    return 0
  else
    local cmp_rc=$?
  fi
  if (( cmp_rc == 1 )); then
    semantic_fail "$label"
    return 0
  fi
  print -u2 "fatal evidence read failure comparing $left and $right"
  return 1
}

run_bench() {
  local run_id=$1
  local wt=$2
  local model=$3
  shift 3
  local -a env_overrides
  env_overrides=("$@")
  local out="$DS4_RESULTS/$run_id"
  mkdir -p "$out/logits"
  {
    printf 'cwd=%q\n' "$wt"
    printf 'env_override_count=%q\n' "${#env_overrides[@]}"
    for env_name in "${env_overrides[@]}"; do
      printf 'env_override=%q\n' "$env_name"
    done
    printf 'binary=%q\n' ./ds4-bench
    printf 'arg=%q\n' --metal -m "$model" --prompt-file "$DS4_PROMPT" \
      --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8321 \
      --gen-tokens 128 --warm-weights --show-output \
      --dump-frontier-logits-dir "$out/logits" --csv "$out/metrics.csv"
  } > "$out/command.txt"
  print -rl -- "${env_overrides[@]}" > "$out/env-overrides.txt"
  git -C "$wt" rev-parse HEAD > "$out/revision.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/start-utc.txt"
  pmset -g batt > "$out/battery-before.txt" 2>&1 || true
  pmset -g therm > "$out/thermal-before.txt" 2>&1 || true
  pmset -g custom > "$out/power-custom-before.txt" 2>&1 || true
  capture_profiler_json "$out/power-profile-before.txt" \
    SPPowerDataType
  sysctl vm.swapusage > "$out/vm-swapusage-before.txt" 2>&1 || true
  vm_stat > "$out/vm-before.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-before.txt" 2>&1 || true
  (
    cd "$wt"
    /usr/bin/time -p env -i \
      HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C \
      "${env_overrides[@]}" \
      ./ds4-bench --metal \
        -m "$model" \
        --prompt-file "$DS4_PROMPT" \
        --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8321 \
        --gen-tokens 128 --warm-weights --show-output \
        --dump-frontier-logits-dir "$out/logits" \
        --csv "$out/metrics.csv"
  ) > "$out/stdout.log" 2> "$out/stderr.log"
  extract_bench_text "$out/stderr.log" > "$out/decoded.txt"
  pmset -g batt > "$out/battery-after.txt" 2>&1 || true
  pmset -g therm > "$out/thermal-after.txt" 2>&1 || true
  pmset -g custom > "$out/power-custom-after.txt" 2>&1 || true
  capture_profiler_json "$out/power-profile-after.txt" \
    SPPowerDataType
  sysctl vm.swapusage > "$out/vm-swapusage-after.txt" 2>&1 || true
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
  local -a env_overrides
  env_overrides=("$@")
  local out="$DS4_RESULTS/$run_id"
  mkdir -p "$out/logits"
  {
    printf 'cwd=%q\n' "$wt"
    printf 'env_override_count=%q\n' "${#env_overrides[@]}"
    for env_name in "${env_overrides[@]}"; do
      printf 'env_override=%q\n' "$env_name"
    done
    printf 'binary=%q\n' ./ds4-bench
    printf 'arg=%q\n' --metal -m "$model" --prompt-file "$DS4_PROMPT" \
      --ctx-start 8192 --ctx-max 8193 --step-incr 1 --ctx-alloc 8194 \
      --gen-tokens 0 --dump-frontier-logits-dir "$out/logits" \
      --csv "$out/metrics.csv"
  } > "$out/command.txt"
  print -rl -- "${env_overrides[@]}" > "$out/env-overrides.txt"
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
      "${env_overrides[@]}" \
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
  local -a env_overrides
  env_overrides=("$@")
  local out="$DS4_RESULTS/$run_id"
  mkdir -p "$out"
  {
    printf 'cwd=%q\n' "$wt"
    printf 'env_override_count=%q\n' "${#env_overrides[@]}"
    for env_name in "${env_overrides[@]}"; do
      printf 'env_override=%q\n' "$env_name"
    done
    printf 'binary=%q\n' ./ds4
    printf 'arg=%q\n' --metal -m "$model" --prompt-file "$DS4_RAW_PROMPT" \
      --raw-prompt --ctx 65536 --tokens 256 --temp 0 --nothink \
      --warm-weights
  } > "$out/command.txt"
  print -rl -- "${env_overrides[@]}" > "$out/env-overrides.txt"
  git -C "$wt" rev-parse HEAD > "$out/revision.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/start-utc.txt"
  pmset -g batt > "$out/battery-before.txt" 2>&1 || true
  pmset -g therm > "$out/thermal-before.txt" 2>&1 || true
  pmset -g custom > "$out/power-custom-before.txt" 2>&1 || true
  capture_profiler_json "$out/power-profile-before.txt" \
    SPPowerDataType
  sysctl vm.swapusage > "$out/vm-swapusage-before.txt" 2>&1 || true
  vm_stat > "$out/vm-before.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-before.txt" 2>&1 || true
  (
    cd "$wt"
    /usr/bin/time -p env -i \
      HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C \
      "${env_overrides[@]}" \
      ./ds4 --metal \
        -m "$model" --prompt-file "$DS4_RAW_PROMPT" --raw-prompt \
        --ctx 65536 --tokens 256 --temp 0 --nothink --warm-weights
  ) > "$out/stdout.log" 2> "$out/stderr.log"
  cp "$out/stdout.log" "$out/generated.txt"
  pmset -g batt > "$out/battery-after.txt" 2>&1 || true
  pmset -g therm > "$out/thermal-after.txt" 2>&1 || true
  pmset -g custom > "$out/power-custom-after.txt" 2>&1 || true
  capture_profiler_json "$out/power-profile-after.txt" \
    SPPowerDataType
  sysctl vm.swapusage > "$out/vm-swapusage-after.txt" 2>&1 || true
  vm_stat > "$out/vm-after.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-after.txt" 2>&1 || true
  print 'requested_tokens=256' > "$out/token-count.txt"
  wc -c < "$out/generated.txt" | awk '{print "output_bytes=" $1}' \
    >> "$out/token-count.txt"
  shasum -a 256 "$out/generated.txt" > "$out/output-sha256.txt"
}

run_lmhead_init_probe() {
  local run_id=$1
  local wt=$2
  local model=$3
  shift 3
  local -a env_overrides
  env_overrides=("$@")
  local out="$DS4_RESULTS/$run_id"
  mkdir -p "$out"
  local -a args
  args=(
    --metal -m "$model" --prompt-file "$DS4_RAW_PROMPT" --raw-prompt
    --ctx 65536 --tokens 1 --temp 0 --nothink --warm-weights
  )
  {
    printf 'cwd=%q\n' "$wt"
    printf 'env_override_count=%q\n' "${#env_overrides[@]}"
    for env_name in "${env_overrides[@]}"; do
      printf 'env_override=%q\n' "$env_name"
    done
    printf 'binary=%q\n' ./ds4
    printf 'arg='
    printf '%q ' "${args[@]}"
    print
  } > "$out/command.txt"
  print -rl -- "${env_overrides[@]}" > "$out/env-overrides.txt"
  git -C "$wt" rev-parse HEAD > "$out/revision.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/start-utc.txt"
  pmset -g batt > "$out/battery-before.txt" 2>&1 || true
  pmset -g therm > "$out/thermal-before.txt" 2>&1 || true
  pmset -g custom > "$out/power-custom-before.txt" 2>&1 || true
  capture_profiler_json "$out/power-profile-before.txt" \
    SPPowerDataType
  sysctl vm.swapusage > "$out/vm-swapusage-before.txt" 2>&1 || true
  vm_stat > "$out/vm-before.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-before.txt" 2>&1 || true
  (
    cd "$wt"
    /usr/bin/time -l env -i \
      HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C "${env_overrides[@]}" \
      ./ds4 "${args[@]}"
  ) > "$out/stdout.log" 2> "$out/stderr.log"
  cp "$out/stdout.log" "$out/generated.txt"
  pmset -g batt > "$out/battery-after.txt" 2>&1 || true
  pmset -g therm > "$out/thermal-after.txt" 2>&1 || true
  pmset -g custom > "$out/power-custom-after.txt" 2>&1 || true
  capture_profiler_json "$out/power-profile-after.txt" \
    SPPowerDataType
  sysctl vm.swapusage > "$out/vm-swapusage-after.txt" 2>&1 || true
  vm_stat > "$out/vm-after.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-after.txt" 2>&1 || true
  grep -E 'Laguna Q8 lm-head screen (prepared|stats)' "$out/stderr.log" \
    > "$out/screen-diagnostic.txt" || true
  grep -Ei 'maximum resident|max rss|maximum resident set size' \
    "$out/stderr.log" > "$out/max-rss.txt" || true
  shasum -a 256 "$out/generated.txt" "$out/stderr.log" \
    > "$out/output-sha256.txt"
}

parse_lmhead_trace() {
  local stderr_path=$1
  local report_path=$2
  python3 - "$stderr_path" "$report_path" <<'PY'
import csv
import math
import re
import struct
import sys

stderr_path, report_path = sys.argv[1:]
pattern = re.compile(
    r"screen stats screen_calls=(\d+) candidates=(\d+) "
    r"row_blocks=(\d+) exact_row_blocks=(\d+) nonfinite=(\d+) "
    r"packed_bytes=(\d+) sidecopy_init_ms=([^ ]+) winner=(-?\d+) "
    r"value=([^ ]+) winner_bits=0x([0-9a-fA-F]+)"
)
total_rows = 100352
blocks_per_row = 96
total_row_blocks = total_rows * blocks_per_row
raw_head_bytes = 327548928
packed_bytes = 173408256
rows = []
semantic_failures = []

def semantic(message):
    semantic_failures.append(message)

with open(stderr_path, encoding="utf-8", errors="replace") as stream:
    for line in stream:
        if "Laguna Q8 lm-head screen stats" not in line:
            continue
        match = pattern.search(line)
        if not match:
            raise SystemExit(f"unparseable lm-head trace line: {line.rstrip()}")
        screen_calls, candidates, candidate_blocks, exact_blocks, nonfinite, \
            packed, init_ms, winner, value, winner_bits = match.groups()
        packed = int(packed)
        exact_blocks = int(exact_blocks)
        try:
            sidecopy_init_ms = float(init_ms)
        except ValueError as error:
            raise SystemExit(f"unparseable sidecopy init time: {init_ms}") from error
        try:
            parsed_winner_value = float(value)
        except ValueError as error:
            raise SystemExit(f"unparseable winner value: {value}") from error
        screen_calls = int(screen_calls)
        candidates = int(candidates)
        candidate_blocks = int(candidate_blocks)
        nonfinite = int(nonfinite)
        winner = int(winner)
        row_failures = []
        if packed != packed_bytes:
            row_failures.append(f"packed bytes {packed} != {packed_bytes}")
        if not 0 <= nonfinite <= total_rows:
            row_failures.append(f"coarse nonfinite count out of bounds: {nonfinite}")
        if not math.isfinite(sidecopy_init_ms) or sidecopy_init_ms < 0.0:
            row_failures.append(
                f"sidecopy init time must be finite/nonnegative: {init_ms}"
            )
        if not 1 <= candidates < total_rows:
            row_failures.append(f"candidate row count out of bounds: {candidates}")
        if candidate_blocks != candidates * blocks_per_row:
            row_failures.append(
                f"candidate row-block mismatch: {candidate_blocks} != "
                f"{candidates}*{blocks_per_row}"
            )
        if not 0 <= candidate_blocks <= total_row_blocks:
            row_failures.append(
                f"candidate row-block count out of bounds: {candidate_blocks}"
            )
        if exact_blocks % 192 != 0:
            row_failures.append(
                f"exact row-block count is not NR2-pair aligned: {exact_blocks}"
            )
        if exact_blocks < 192:
            row_failures.append(
                f"exact row-block count lacks mandatory seed pair: {exact_blocks}"
            )
        if not candidate_blocks <= exact_blocks < total_row_blocks:
            row_failures.append(
                f"exact row-block count out of bounds: {exact_blocks} "
                f"(candidate={candidate_blocks}, total={total_row_blocks})"
            )
        if not 0 <= winner < total_rows:
            row_failures.append(f"winner index out of bounds: {winner}")
        winner_bits_value = int(winner_bits, 16)
        if not 0 <= winner_bits_value <= 0xffffffff:
            row_failures.append(f"winner bits out of uint32 range: {winner_bits}")
        if math.isnan(parsed_winner_value):
            row_failures.append("winner value is NaN")
        if 0 <= winner_bits_value <= 0xffffffff:
            winner_bits_value_float = struct.unpack(
                "<f", struct.pack("<I", winner_bits_value)
            )[0]
            if math.isnan(winner_bits_value_float):
                row_failures.append("winner bits encode NaN")
        semantic_failures.extend(row_failures)
        rows.append({
            "screen_calls": screen_calls,
            "candidate_rows": candidates,
            "candidate_row_blocks": candidate_blocks,
            "exact_row_blocks": exact_blocks,
            "coarse_nonfinite": nonfinite,
            "packed_bytes": int(packed),
            "sidecopy_init_ms": sidecopy_init_ms,
            "winner_index": winner,
            "winner_value": value,
            "winner_bits": f"0x{winner_bits_value:08x}",
            "row_semantic_failures": "; ".join(row_failures),
            "candidate_percent": 100.0 * candidates / total_rows,
            "exact_physical_row_block_percent":
                100.0 * exact_blocks / total_row_blocks,
            "exact_q8_bytes": exact_blocks * 34,
            "estimated_weight_bytes": packed_bytes + exact_blocks * 34,
            "traffic_break_even_exact_row_blocks": 4533550,
            "logical_candidate_heuristic_percent": 27.24,
            "total_rows": total_rows,
            "blocks_per_row": blocks_per_row,
            "raw_head_bytes": raw_head_bytes,
        })
if not rows:
    semantic("no Laguna Q8 lm-head trace lines found")
if len(rows) != 16:
    semantic(f"expected exactly 16 lm-head trace lines, got {len(rows)}")
expected_calls = list(range(1, 17))
actual_calls = [row["screen_calls"] for row in rows]
if actual_calls != expected_calls:
    semantic(
        f"expected screen_calls sequence 1..16, got {actual_calls}"
    )
status = "PASS" if not semantic_failures else "FAIL"
fields = [
    "screen_calls", "candidate_rows", "candidate_row_blocks",
    "exact_row_blocks", "coarse_nonfinite", "packed_bytes",
    "sidecopy_init_ms", "winner_index", "winner_value", "winner_bits",
    "row_semantic_failures", "candidate_percent",
    "exact_physical_row_block_percent", "exact_q8_bytes",
    "estimated_weight_bytes", "traffic_break_even_exact_row_blocks",
    "logical_candidate_heuristic_percent", "total_rows", "blocks_per_row",
    "raw_head_bytes",
]
with open(report_path, "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
summary_path = report_path + ".summary.txt"
with open(summary_path, "w", encoding="utf-8") as stream:
    stream.write(f"status={status}\n")
    stream.write(f"rows={len(rows)}\n")
    stream.write(f"screen_calls={actual_calls}\n")
    for failure in semantic_failures:
        stream.write(f"failure={failure}\n")
print(
    f"parsed {len(rows)} lm-head trace lines into {report_path} "
    f"status={status}"
)
PY
  shasum -a 256 "$report_path" "$report_path.summary.txt" \
    > "$report_path.sha256"
}

run_lmhead_trace_probe() {
  local run_id=$1
  local wt=$2
  local model=$3
  shift 3
  local -a env_overrides
  env_overrides=("$@")
  local trace_requested=0
  for env_name in "${env_overrides[@]}"; do
    if [[ "$env_name" == DS4_METAL_LAGUNA_Q8_LMHEAD_TRACE=1 ]]; then
      trace_requested=1
    fi
  done
  local out="$DS4_RESULTS/$run_id"
  mkdir -p "$out"
  local -a args
  args=(
    --metal -m "$model" --prompt-file "$DS4_RAW_PROMPT" --raw-prompt
    --ctx 65536 --tokens 16 --temp 0 --nothink --warm-weights
  )
  {
    printf 'cwd=%q\n' "$wt"
    printf 'env_override_count=%q\n' "${#env_overrides[@]}"
    for env_name in "${env_overrides[@]}"; do
      printf 'env_override=%q\n' "$env_name"
    done
    printf 'binary=%q\n' ./ds4
    printf 'arg='
    printf '%q ' "${args[@]}"
    print
  } > "$out/command.txt"
  print -rl -- "${env_overrides[@]}" > "$out/env-overrides.txt"
  git -C "$wt" rev-parse HEAD > "$out/revision.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/start-utc.txt"
  pmset -g batt > "$out/battery-before.txt" 2>&1 || true
  pmset -g therm > "$out/thermal-before.txt" 2>&1 || true
  pmset -g custom > "$out/power-custom-before.txt" 2>&1 || true
  capture_profiler_json "$out/power-profile-before.txt" \
    SPPowerDataType
  sysctl vm.swapusage > "$out/vm-swapusage-before.txt" 2>&1 || true
  vm_stat > "$out/vm-before.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-before.txt" 2>&1 || true
  (
    cd "$wt"
    env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C "${env_overrides[@]}" \
      ./ds4 "${args[@]}"
  ) > "$out/stdout.log" 2> "$out/stderr.log"
  cp "$out/stdout.log" "$out/generated.txt"
  pmset -g batt > "$out/battery-after.txt" 2>&1 || true
  pmset -g therm > "$out/thermal-after.txt" 2>&1 || true
  pmset -g custom > "$out/power-custom-after.txt" 2>&1 || true
  capture_profiler_json "$out/power-profile-after.txt" \
    SPPowerDataType
  sysctl vm.swapusage > "$out/vm-swapusage-after.txt" 2>&1 || true
  vm_stat > "$out/vm-after.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-after.txt" 2>&1 || true
  if (( trace_requested )); then
    parse_lmhead_trace "$out/stderr.log" "$out/lmhead-trace.csv"
  fi
  shasum -a 256 "$out/generated.txt" "$out/stderr.log" \
    > "$out/output-sha256.txt"
}

run_lmhead_winner_probe() {
  local run_id=$1
  local wt=$2
  local model=$3
  shift 3
  local -a env_overrides
  env_overrides=("$@")
  local out="$DS4_RESULTS/$run_id"
  mkdir -p "$out"
  local -a args
  args=(
    --metal -m "$model" --prompt-file "$DS4_RAW_PROMPT" --raw-prompt
    --ctx 65536 --tokens 1 --temp 0 --nothink --warm-weights
  )
  {
    printf 'cwd=%q\n' "$wt"
    printf 'env_override_count=%q\n' "${#env_overrides[@]}"
    for env_name in "${env_overrides[@]}"; do
      printf 'env_override=%q\n' "$env_name"
    done
    printf 'binary=%q\n' ./ds4
    printf 'arg='
    printf '%q ' "${args[@]}"
    print
  } > "$out/command.txt"
  print -rl -- "${env_overrides[@]}" > "$out/env-overrides.txt"
  git -C "$wt" rev-parse HEAD > "$out/revision.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/start-utc.txt"
  pmset -g batt > "$out/battery-before.txt" 2>&1 || true
  pmset -g therm > "$out/thermal-before.txt" 2>&1 || true
  pmset -g custom > "$out/power-custom-before.txt" 2>&1 || true
  capture_profiler_json "$out/power-profile-before.txt" \
    SPPowerDataType
  sysctl vm.swapusage > "$out/vm-swapusage-before.txt" 2>&1 || true
  vm_stat > "$out/vm-before.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-before.txt" 2>&1 || true
  (
    cd "$wt"
    env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C "${env_overrides[@]}" \
      ./ds4 "${args[@]}"
  ) > "$out/stdout.log" 2> "$out/stderr.log"
  cp "$out/stdout.log" "$out/generated.txt"
  pmset -g batt > "$out/battery-after.txt" 2>&1 || true
  pmset -g therm > "$out/thermal-after.txt" 2>&1 || true
  pmset -g custom > "$out/power-custom-after.txt" 2>&1 || true
  capture_profiler_json "$out/power-profile-after.txt" \
    SPPowerDataType
  sysctl vm.swapusage > "$out/vm-swapusage-after.txt" 2>&1 || true
  vm_stat > "$out/vm-after.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure-after.txt" 2>&1 || true
  shasum -a 256 "$out/generated.txt" "$out/stderr.log" \
    > "$out/output-sha256.txt"
}

compare_lmhead_winner() {
  local stock_run=$1
  local screen_run=$2
  local report_rel=$3
  local stock="$DS4_RESULTS/$stock_run"
  local screen="$DS4_RESULTS/$screen_run"
  local report="$DS4_RESULTS/$report_rel"
  local status_path="$report.status.txt"
  mkdir -p "${report:h}"
  python3 - "$stock" "$screen" "$report" <<'PY'
import csv
import math
import re
import struct
import sys

stock_dir, screen_dir, report_path = sys.argv[1:]
status_path = report_path + ".status.txt"
stock_text = open(f"{stock_dir}/stderr.log", encoding="utf-8", errors="replace").read()
screen_text = open(f"{screen_dir}/stderr.log", encoding="utf-8", errors="replace").read()
marker = "ds4: top logits Laguna step 0:\n"
marker_pos = stock_text.find(marker)
if marker_pos < 0:
    raise SystemExit("stock Laguna step-0 top-logits marker not found")
stock_section = stock_text[marker_pos + len(marker):].split("\nds4:", 1)[0]
stock_match = re.search(
    r"^\s+0\s+(-?\d+)\s+([^ \r\n]+)", stock_section, re.MULTILINE
)
if not stock_match:
    raise SystemExit("stock Laguna step-0 rank-0 line not found")
stock_id = int(stock_match.group(1))
stock_printed = float(stock_match.group(2))
if not 0 <= stock_id < 100352:
    raise SystemExit(f"stock winner ID out of range: {stock_id}")
stock_bits = struct.unpack("<I", struct.pack("<f", stock_printed))[0]
screen_matches = re.findall(
    r"screen stats screen_calls=(\d+) .*?winner=(-?\d+) "
    r".*?winner_bits=0x([0-9a-fA-F]+)",
    screen_text,
)
if len(screen_matches) != 1:
    raise SystemExit(
        f"expected exactly one screen winner trace line, got {len(screen_matches)}"
    )
screen_calls, screen_id, screen_bits_text = screen_matches[0]
if int(screen_calls) != 1:
    raise SystemExit(f"winner probe requires screen_calls=1, got {screen_calls}")
screen_id = int(screen_id)
screen_bits = int(screen_bits_text, 16)
if not 0 <= screen_id < 100352:
    raise SystemExit(f"screen winner ID out of range: {screen_id}")
if not 0 <= screen_bits <= 0xffffffff:
    raise SystemExit(f"screen winner bits out of range: {screen_bits_text}")
screen_value = struct.unpack("<f", struct.pack("<I", screen_bits))[0]
winner_values_not_nan = not math.isnan(stock_printed) and not math.isnan(screen_value)
generated_equal = open(f"{stock_dir}/generated.txt", "rb").read() == \
    open(f"{screen_dir}/generated.txt", "rb").read()
row = {
    "stock_top0_id": stock_id,
    "stock_top0_printed": stock_match.group(2),
    "stock_top0_float32_bits": f"0x{stock_bits:08x}",
    "screen_winner_id": screen_id,
    "screen_winner_bits": f"0x{screen_bits:08x}",
    "winner_values_not_nan": winner_values_not_nan,
    "winner_id_exact": stock_id == screen_id,
    "winner_value_bits_exact": stock_bits == screen_bits,
    "generated_output_byte_exact": generated_equal,
}
row["parity_status"] = "pass" if all((
    row["winner_values_not_nan"], row["winner_id_exact"],
    row["winner_value_bits_exact"],
    row["generated_output_byte_exact"],
)) else "fail"
with open(report_path, "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
with open(status_path, "w", encoding="utf-8") as stream:
    stream.write(f"status={row['parity_status']}\n")
print(row)
if row["parity_status"] != "pass":
    print("lm-head winner parity failed (evidence retained)", file=sys.stderr)
PY
  shasum -a 256 "$report" "$status_path" > "$report.sha256"
}

run_dflash() {
  local run_id=$1
  shift
  local -a env_overrides
  env_overrides=("$@")
  local out="$DS4_RESULTS/$run_id"
  mkdir -p "$out"
  {
    printf 'cwd=%q\n' "$DS4_CAND_WT"
    printf 'env_override_count=%q\n' "${#env_overrides[@]}"
    printf 'env_override=%q\n' DS4_DFLASH_TIMING=1
    for env_name in "${env_overrides[@]}"; do
      printf 'env_override=%q\n' "$env_name"
    done
    printf 'binary=%q\n' ./ds4
    printf 'arg=%q\n' --metal -m "$DS4_Q23_MODEL" \
      --dflash "$DS4_DFLASH_MODEL" --dflash-draft 3 --dflash-p-min 0 \
      --prompt-file "$DS4_RAW_PROMPT" --raw-prompt --ctx 65536 \
      --tokens 256 --temp 0 --nothink --warm-weights
  } > "$out/command.txt"
  print -rl -- DS4_DFLASH_TIMING=1 "${env_overrides[@]}" \
    > "$out/env-overrides.txt"
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
      "${env_overrides[@]}" \
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

## Run the current bounded first pass

Run one unreported warm-up for every distinct arm, then preserve the specified
order. The mirrored three-way order controls first-to-last drift; every
two-arm timing experiment uses ABBA. The init/RSS, trace, and winner probes
are single non-timing observations.

### 1. Default three-way comparison

Run this matrix for the required Q2/Q3 resident quant only:

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

```

All six Q2/Q3 `decoded.txt` files and frontier-logit JSON files are expected to
be byte-identical. Record any parity mismatch in the semantic-gate ledger and
continue collecting the bounded evidence:

```zsh
ref="$DS4_RESULTS/default-q23/01-pre-A"
for run in "$DS4_RESULTS/default-q23"/0*; do
  compare_files_semantic "clean decoded output mismatch: $run" \
    "$ref/decoded.txt" "$run/decoded.txt"
  compare_files_semantic "clean 8192 frontier mismatch: $run" \
    "$ref/logits/frontier_008192.logits.json" \
    "$run/logits/frontier_008192.logits.json"
done

if python3 - "$DS4_RESULTS/default-q23" <<'PY'
import csv
import math
import os
import statistics
import sys

root = sys.argv[1]
groups = {
    "pre": ["01-pre-A", "06-pre-A"],
    "base": ["02-base-B", "05-base-B"],
    "cand": ["03-cand-C", "04-cand-C"],
}
metric_names = ["prefill_tps", "gen_tps", "gen_steady_tps", "gen_first_ms"]
data = {}
drift_failures = []
for label, runs in groups.items():
    observations = []
    for run in runs:
        metrics = os.path.join(root, run, "metrics.csv")
        try:
            with open(metrics, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        except OSError as error:
            raise SystemExit(2) from error
        if not rows:
            raise SystemExit(2)
        row = rows[-1]
        values = {}
        for name in metric_names:
            text = row.get(name)
            if text is None:
                raise SystemExit(2)
            try:
                value = float(text)
            except ValueError as error:
                raise SystemExit(2) from error
            if not math.isfinite(value) or value <= 0:
                raise SystemExit(2)
            values[name] = value
        observations.append(values)
    data[label] = observations
    drift = abs(observations[1]["gen_steady_tps"] /
                observations[0]["gen_steady_tps"] - 1.0)
    if drift > 0.03:
        drift_failures.append(f"{label} drift={drift:.3%}")
    print(
        f"{label}: {observations[0]['gen_steady_tps']:.9g},"
        f"{observations[1]['gen_steady_tps']:.9g} drift={drift:.3%}"
    )

medians = {
    label: {name: statistics.median(row[name] for row in rows)
            for name in metric_names}
    for label, rows in data.items()
}
metrics_report = os.path.join(root, "clean-q23-metrics.csv")
metric_fields = ["revision", "observation", "row_type"] + metric_names
with open(metrics_report, "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=metric_fields)
    writer.writeheader()
    for label, rows in data.items():
        for index, values in enumerate(rows, 1):
            writer.writerow({"revision": label, "observation": index,
                             "row_type": "observation", **values})
        writer.writerow({"revision": label, "observation": "median",
                         "row_type": "median", **medians[label]})

delta_report = os.path.join(root, "clean-q23-deltas.csv")
delta_fields = ["comparison", "metric", "left_median", "right_median",
                "delta_percent", "formula"]
comparisons = [
    ("pre_to_base", "pre", "base"),
    ("base_to_candidate", "base", "cand"),
    ("pre_to_candidate", "pre", "cand"),
]
with open(delta_report, "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=delta_fields)
    writer.writeheader()
    for comparison, left, right in comparisons:
        for name in metric_names:
            before = medians[left][name]
            after = medians[right][name]
            if name == "gen_first_ms":
                delta = (before / after - 1.0) * 100.0
                formula = "(left_median/right_median-1)*100"
            else:
                delta = (after / before - 1.0) * 100.0
                formula = "(right_median/left_median-1)*100"
            writer.writerow({"comparison": comparison, "metric": name,
                             "left_median": before, "right_median": after,
                             "delta_percent": delta, "formula": formula})
if drift_failures:
    print("clean Q2/Q3 drift gate failed: " + "; ".join(drift_failures),
          file=sys.stderr)
    raise SystemExit(1)
PY
then
  :
else
  drift_rc=$?
  if (( drift_rc == 1 )); then
    semantic_fail 'clean Q2/Q3 within-revision drift gate failed; record the failure and start a fresh DS4_BENCH_ROOT for any rerun'
  else
    print -u2 'clean Q2/Q3 drift evidence is malformed or unreadable'
    return 1
  fi
fi
wait_for_nominal "$DS4_RESULTS/metadata/cooldown-after-default-q23.txt" || return 1
verify_power_and_thermals > \
  "$DS4_RESULTS/metadata/power-after-default-q23.txt" 2>&1 || return 1
```

### 2. Candidate Laguna Q8 lm-head experiment

All five matrices below run only in `DS4_CAND_WT` at candidate SHA
`202a46b5158bff38ba36e066d71bc5b242b53a4a`. They use the Q2/Q3 model and the
existing raw Laguna 65K-context/256-token workload where throughput is
measured. Each arm records its exact environment, revision, stdout, stderr,
output hash, and battery/thermal/power-profile/vm-stat/vm.swapusage/
memory-pressure state before and after. Cool down between matrices and do not
use trace or screen flags in a
timed arm unless that matrix explicitly says so.

#### A. Safe-mode tax

Run one warm-up per arm, then A-B-B-A. A is the current fast GPU-argmax path;
B adds safe Metal math. The lm-head screen is absent from both arms.

```zsh
run_raw warmup/lmhead-safe-tax-A "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1
run_raw warmup/lmhead-safe-tax-B "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1
run_raw lmhead-safe-tax/01-fast-A "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1
run_raw lmhead-safe-tax/02-safe-B "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1
run_raw lmhead-safe-tax/03-safe-B "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1
run_raw lmhead-safe-tax/04-fast-A "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1
for run in 01-fast-A 02-safe-B 03-safe-B 04-fast-A; do
  compare_files_semantic "safe-tax generated output mismatch: $run" \
    "$DS4_RESULTS/lmhead-safe-tax/$run/generated.txt" \
    "$DS4_RESULTS/lmhead-safe-tax/01-fast-A/generated.txt"
  compare_files_semantic "safe-tax token/output-byte mismatch: $run" \
    "$DS4_RESULTS/lmhead-safe-tax/$run/token-count.txt" \
    "$DS4_RESULTS/lmhead-safe-tax/01-fast-A/token-count.txt"
done
for run in warmup/lmhead-safe-tax-A lmhead-safe-tax/01-fast-A \
           lmhead-safe-tax/04-fast-A; do
  require_gpu_argmax_diagnostic \
    "$DS4_RESULTS/$run/stderr.log"
  grep -Eiq 'math_safe=off' \
    "$DS4_RESULTS/$run/stderr.log" || {
    semantic_fail "fast math diagnostic missing from A arm: $run"
  }
  if grep -Eiq 'math_safe=on|shader library math mode = safe' \
      "$DS4_RESULTS/$run/stderr.log"; then
    semantic_fail "safe math unexpectedly enabled in A arm: $run"
  fi
done
for run in warmup/lmhead-safe-tax-B lmhead-safe-tax/02-safe-B \
           lmhead-safe-tax/03-safe-B; do
  require_gpu_argmax_diagnostic \
    "$DS4_RESULTS/$run/stderr.log"
  grep -Fq 'Metal shader library math mode = safe' \
    "$DS4_RESULTS/$run/stderr.log" || {
    semantic_fail "safe math diagnostic missing from B arm: $run"
  }
done
wait_for_nominal "$DS4_RESULTS/metadata/cooldown-after-lmhead-safe-tax.txt" || return 1
verify_power_and_thermals > \
  "$DS4_RESULTS/metadata/power-after-lmhead-safe-tax.txt" 2>&1 || return 1
```

Require all four `generated.txt` files and their requested-token/output-byte
records to match; report the safe tax separately from screen speed.

#### B. Screen throughput

Run one warm-up per arm, then A-B-B-A. A is safe math with GPU argmax and no
screen. B adds only `DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN=1`. Trace is absent
from both timed arms. Use the raw Laguna Q2/Q3 65K/256 workload from
`run_raw`; report prefill and generation t/s, output SHA, the requested-token
and output-byte record, and the `/usr/bin/time -p` evidence.

```zsh
run_raw warmup/lmhead-screen-A "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1
run_raw warmup/lmhead-screen-B "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1 \
  DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN=1
run_raw lmhead-screen/01-safe-off-A "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1
run_raw lmhead-screen/02-safe-on-B "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1 \
  DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN=1
run_raw lmhead-screen/03-safe-on-B "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1 \
  DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN=1
run_raw lmhead-screen/04-safe-off-A "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1
for run in 01-safe-off-A 02-safe-on-B 03-safe-on-B 04-safe-off-A; do
  compare_files_semantic "screen generated output mismatch: $run" \
    "$DS4_RESULTS/lmhead-screen/$run/generated.txt" \
    "$DS4_RESULTS/lmhead-screen/01-safe-off-A/generated.txt"
  compare_files_semantic "screen token/output-byte mismatch: $run" \
    "$DS4_RESULTS/lmhead-screen/$run/token-count.txt" \
    "$DS4_RESULTS/lmhead-screen/01-safe-off-A/token-count.txt"
  if grep -q 'DS4_METAL_LAGUNA_Q8_LMHEAD_TRACE' \
      "$DS4_RESULTS/lmhead-screen/$run/env-overrides.txt"; then
    print -u2 "trace must be absent from timed lm-head arm: $run"
    semantic_fail "trace must be absent from timed lm-head arm: $run"
  fi
  if grep -Eiq 'screen stats|fallback|unavailable|failed' \
      "$DS4_RESULTS/lmhead-screen/$run/stderr.log"; then
    print -u2 "unexpected diagnostic in timed lm-head arm: $run"
    semantic_fail "unexpected diagnostic in timed lm-head arm: $run"
  fi
done
for run in warmup/lmhead-screen-A lmhead-screen/01-safe-off-A \
           lmhead-screen/04-safe-off-A; do
  require_gpu_argmax_diagnostic \
    "$DS4_RESULTS/$run/stderr.log"
  if grep -Eiq 'Laguna Q8 lm-head screen (prepared|enabled|stats)' \
      "$DS4_RESULTS/$run/stderr.log"; then
    print -u2 "screen must be absent from timed A arm: $run"
    semantic_fail "screen must be absent from timed A arm: $run"
  fi
  grep -Fq 'Metal shader library math mode = safe' \
    "$DS4_RESULTS/$run/stderr.log" || {
    semantic_fail "safe math diagnostic missing from timed A arm: $run"
  }
done
for run in warmup/lmhead-screen-B lmhead-screen/02-safe-on-B \
           lmhead-screen/03-safe-on-B; do
  require_gpu_argmax_diagnostic \
    "$DS4_RESULTS/$run/stderr.log"
  grep -Fq 'Laguna Q8 lm-head screen prepared' \
    "$DS4_RESULTS/$run/stderr.log" || {
    semantic_fail "screen prepared diagnostic missing from timed B arm: $run"
  }
  grep -Fq 'Laguna Q8 lm-head screen enabled (exact top-1 only)' \
    "$DS4_RESULTS/$run/stderr.log" || {
    semantic_fail "screen enable diagnostic missing from timed B arm: $run"
  }
  grep -Fq 'Metal shader library math mode = safe' \
    "$DS4_RESULTS/$run/stderr.log" || {
    semantic_fail "safe math diagnostic missing from timed B arm: $run"
  }
done
wait_for_nominal "$DS4_RESULTS/metadata/cooldown-after-lmhead-screen.txt" || return 1
verify_power_and_thermals > \
  "$DS4_RESULTS/metadata/power-after-lmhead-screen.txt" 2>&1 || return 1
```

Compare the four generated outputs and requested-token/output-byte records
byte-for-byte.
Use median `generation` t/s as the primary screen metric and `prefill` t/s as
the control. The screen path must not emit fallback, unavailable, or failed
diagnostics. The token-count file records the requested token count and output
byte count; it is not an observed-token counter.

#### C. Non-timing one-token init/RSS probe

Run one safe-off and one safe-on raw-CLI probe with `--tokens 1`. The CLI
rejects `--tokens 0`, so this is a one-token startup/RSS probe, not a
throughput arm. `/usr/bin/time -l`, maximum RSS, vm-stat, memory pressure,
and the packed-sidecopy diagnostic are the evidence. `sidecopy_init_ms` means
allocation plus sidecopy only; it excludes pipeline compilation and model
wrapping. The safe-on arm must report the packed sidecopy and its exact byte
size.

```zsh
run_lmhead_init_probe lmhead-init/safe-off "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1
run_lmhead_init_probe lmhead-init/safe-on "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1 \
  DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN=1
for run in safe-off safe-on; do
  require_gpu_argmax_diagnostic \
    "$DS4_RESULTS/lmhead-init/$run/stderr.log"
  grep -Fq 'Metal shader library math mode = safe' \
    "$DS4_RESULTS/lmhead-init/$run/stderr.log" || {
    semantic_fail "safe Metal library diagnostic missing from init arm: $run"
  }
done
grep -q 'Laguna Q8 lm-head screen prepared' \
  "$DS4_RESULTS/lmhead-init/safe-on/screen-diagnostic.txt" ||
  semantic_fail 'screen prepared diagnostic missing from init safe-on arm'
grep -q '173408256 bytes' \
  "$DS4_RESULTS/lmhead-init/safe-on/screen-diagnostic.txt" ||
  semantic_fail 'packed sidecopy byte diagnostic missing from init safe-on arm'
grep -Fq 'Laguna Q8 lm-head screen enabled (exact top-1 only)' \
  "$DS4_RESULTS/lmhead-init/safe-on/stderr.log" || {
  semantic_fail 'screen enable diagnostic missing from init safe-on arm'
}
for run in safe-off safe-on; do
  [[ -s "$DS4_RESULTS/lmhead-init/$run/max-rss.txt" ]] || {
    print -u2 "missing maximum-RSS evidence for lm-head init arm: $run"
    return 1
  }
done
if grep -Eiq 'Laguna Q8 lm-head screen (prepared|enabled|stats)' \
    "$DS4_RESULTS/lmhead-init/safe-off/stderr.log"; then
  semantic_fail 'screen unexpectedly active in safe-off init control'
fi
wait_for_nominal "$DS4_RESULTS/metadata/cooldown-after-lmhead-init.txt" || return 1
verify_power_and_thermals > \
  "$DS4_RESULTS/metadata/power-after-lmhead-init.txt" 2>&1 || return 1
```

#### D. Screen trace probe

Run one non-timing 16-token raw-CLI probe with safe math, GPU argmax, screen,
and trace enabled. The parser consumes every trace line, requires exactly 16
rows with `screen_calls=1..16`, and writes a CSV with
`screen_calls`, `candidate_rows`, `candidate_row_blocks`, `exact_row_blocks`
(including the mandatory seed pair and pair expansion), `coarse_nonfinite`,
winner ID/bits, candidate percentage, exact physical row-block percentage,
exact Q8 bytes, and estimated weight bytes.
Run a paired 16-token trace-off control with the same safe screen flags and
compare its generated output byte-for-byte; this proves trace collection does
not alter the screen route beyond the one-token winner probe.

```zsh
run_lmhead_trace_probe lmhead-trace/q23 "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1 \
  DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN=1 \
  DS4_METAL_LAGUNA_Q8_LMHEAD_TRACE=1
run_lmhead_trace_probe lmhead-trace/q23-control "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1 \
  DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN=1
require_gpu_argmax_diagnostic \
  "$DS4_RESULTS/lmhead-trace/q23/stderr.log"
grep -Fq 'Metal shader library math mode = safe' \
  "$DS4_RESULTS/lmhead-trace/q23/stderr.log" ||
  semantic_fail 'safe Metal library diagnostic missing from trace arm'
grep -Fq 'Laguna Q8 lm-head screen prepared' \
  "$DS4_RESULTS/lmhead-trace/q23/stderr.log" ||
  semantic_fail 'screen prepared diagnostic missing from trace arm'
grep -Fq 'Laguna Q8 lm-head screen enabled (exact top-1 only)' \
  "$DS4_RESULTS/lmhead-trace/q23/stderr.log" ||
  semantic_fail 'screen enable diagnostic missing from trace arm'
require_gpu_argmax_diagnostic \
  "$DS4_RESULTS/lmhead-trace/q23-control/stderr.log"
grep -Fq 'Metal shader library math mode = safe' \
  "$DS4_RESULTS/lmhead-trace/q23-control/stderr.log" ||
  semantic_fail 'safe Metal library diagnostic missing from trace-off control'
grep -Fq 'Laguna Q8 lm-head screen prepared' \
  "$DS4_RESULTS/lmhead-trace/q23-control/stderr.log" ||
  semantic_fail 'screen prepared diagnostic missing from trace-off control'
grep -Fq 'Laguna Q8 lm-head screen enabled (exact top-1 only)' \
  "$DS4_RESULTS/lmhead-trace/q23-control/stderr.log" ||
  semantic_fail 'screen enable diagnostic missing from trace-off control'
if grep -Fq 'Laguna Q8 lm-head screen stats' \
    "$DS4_RESULTS/lmhead-trace/q23-control/stderr.log"; then
  semantic_fail 'trace-off control unexpectedly emitted screen stats'
fi
compare_files_semantic '16-token trace-on/off generated output mismatch' \
  "$DS4_RESULTS/lmhead-trace/q23/generated.txt" \
  "$DS4_RESULTS/lmhead-trace/q23-control/generated.txt"
awk -F, 'NR == 1 || $5 == 0 {next} {bad=1} END {exit bad}' \
  "$DS4_RESULTS/lmhead-trace/q23/lmhead-trace.csv" || {
  semantic_fail 'real-prompt lm-head trace reported coarse_nonfinite != 0'
}
[[ -s "$DS4_RESULTS/lmhead-trace/q23/lmhead-trace.csv.summary.txt" ]] || {
  print -u2 'trace parser summary is missing or empty'
  return 1
}
if grep -q '^status=FAIL$' \
    "$DS4_RESULTS/lmhead-trace/q23/lmhead-trace.csv.summary.txt"; then
  while IFS= read -r failure; do
    [[ "$failure" == failure=* ]] || continue
    semantic_fail "trace parser: ${failure#failure=}"
  done < "$DS4_RESULTS/lmhead-trace/q23/lmhead-trace.csv.summary.txt"
elif ! grep -q '^status=PASS$' \
    "$DS4_RESULTS/lmhead-trace/q23/lmhead-trace.csv.summary.txt"; then
  print -u2 'trace parser summary status is malformed'
  return 1
fi
wait_for_nominal "$DS4_RESULTS/metadata/cooldown-after-lmhead-trace.txt" || return 1
verify_power_and_thermals > \
  "$DS4_RESULTS/metadata/power-after-lmhead-trace.txt" 2>&1 || return 1
```

The parser uses `total_rows=100352`, `blocks_per_row=96`, raw head bytes
`327548928`, and packed sidecopy bytes `173408256`. Exact Q8 bytes are
`exact_row_blocks*34`; estimated weight bytes are
`173408256+exact_row_blocks*34`. Pure traffic break-even is
`exact_row_blocks < 4,533,550` (about 47.06% of raw head traffic); the
logical-candidate independence heuristic is about 27.24%. All real-prompt
trace rows must have `coarse_nonfinite=0`.

#### E. Exact one-token winner-value probe

Run the stock full-logit arm and screen arm separately. The stock arm uses
safe math plus `DS4_TRACE_TOP=1` and no GPU argmax. The screen arm uses safe
math, GPU argmax, screen, and trace. The parser converts the stock rank-0
value printed with nine significant digits back to float32 bits, then compares
token ID and winner-value bits exactly with the screen trace. Generated output
must also be byte-identical; the parser and output hashes are preserved.

```zsh
run_lmhead_winner_probe lmhead-winner/stock "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_MATH_SAFE=1 DS4_TRACE_TOP=1
grep -Fq 'Metal shader library math mode = safe' \
  "$DS4_RESULTS/lmhead-winner/stock/stderr.log" || {
  semantic_fail 'safe Metal library diagnostic missing from stock winner arm'
}
run_lmhead_winner_probe lmhead-winner/screen "$DS4_CAND_WT" "$DS4_Q23_MODEL" \
  DS4_METAL_LAGUNA_GPU_ARGMAX=1 DS4_METAL_MATH_SAFE=1 \
  DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN=1 \
  DS4_METAL_LAGUNA_Q8_LMHEAD_TRACE=1
require_gpu_argmax_diagnostic \
  "$DS4_RESULTS/lmhead-winner/screen/stderr.log"
grep -Fq 'Metal shader library math mode = safe' \
  "$DS4_RESULTS/lmhead-winner/screen/stderr.log" ||
  semantic_fail 'safe Metal library diagnostic missing from screen winner arm'
grep -Fq 'Laguna Q8 lm-head screen prepared' \
  "$DS4_RESULTS/lmhead-winner/screen/stderr.log" ||
  semantic_fail 'screen prepared diagnostic missing from winner arm'
grep -Fq 'Laguna Q8 lm-head screen enabled (exact top-1 only)' \
  "$DS4_RESULTS/lmhead-winner/screen/stderr.log" ||
  semantic_fail 'screen enable diagnostic missing from winner arm'
compare_lmhead_winner lmhead-winner/stock lmhead-winner/screen \
  lmhead-winner/winner-parity.csv
[[ -s "$DS4_RESULTS/lmhead-winner/winner-parity.csv.status.txt" ]] || {
  print -u2 'winner parity status sidecar is missing or empty'
  return 1
}
if grep -q '^status=fail$' \
    "$DS4_RESULTS/lmhead-winner/winner-parity.csv.status.txt"; then
  semantic_fail 'winner ID/value-bit/output parity gate failed'
elif ! grep -q '^status=pass$' \
    "$DS4_RESULTS/lmhead-winner/winner-parity.csv.status.txt"; then
  print -u2 'winner parity status sidecar is malformed'
  return 1
fi
wait_for_nominal "$DS4_RESULTS/metadata/cooldown-after-lmhead-winner.txt" || return 1
verify_power_and_thermals > \
  "$DS4_RESULTS/metadata/power-after-lmhead-winner.txt" 2>&1 || return 1
```

Compute the net generation and prefill gates from medians, not from additive
percentages or cross-matrix averages. The safe-mode tax and screen throughput
are separate experiments. The explicit ratios are
`net_ratio=(safe_tax_safe_median/fast_median)*(screen_on_median/screen_safe_off_median)`
and
`net_prefill_ratio=(safe_tax_safe_prefill/fast_prefill)*(screen_on_prefill/screen_safe_off_prefill)`.
Require both net ratios to be at least 0.99; retain the incremental screen
prefill ratio as a diagnostic and report all component ratios in `SUMMARY.md`.

```zsh
python3 - "$DS4_RESULTS" <<'PY'
import csv
import math
import re
import statistics
import sys

root = sys.argv[1]
pattern = re.compile(
    r"^ds4: Laguna prefill: ([0-9.eE+-]+) t/s, "
    r"generation: ([0-9.eE+-]+) t/s$",
    re.MULTILINE,
)

def read_tps(rel):
    path = f"{root}/{rel}/stderr.log"
    text = open(path, encoding="utf-8", errors="replace").read()
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise SystemExit(f"expected one Laguna throughput line in {path}, got {len(matches)}")
    prefill, generation = (float(value) for value in matches[0])
    if not all(math.isfinite(value) and value > 0 for value in (prefill, generation)):
        raise SystemExit(f"invalid throughput in {path}: {matches[0]}")
    return prefill, generation

fast_runs = ["lmhead-safe-tax/01-fast-A", "lmhead-safe-tax/04-fast-A"]
safe_runs = ["lmhead-safe-tax/02-safe-B", "lmhead-safe-tax/03-safe-B"]
screen_off_runs = ["lmhead-screen/01-safe-off-A", "lmhead-screen/04-safe-off-A"]
screen_on_runs = ["lmhead-screen/02-safe-on-B", "lmhead-screen/03-safe-on-B"]
series = {name: [read_tps(run) for run in runs] for name, runs in {
    "fast": fast_runs,
    "safe": safe_runs,
    "screen_off": screen_off_runs,
    "screen_on": screen_on_runs,
}.items()}
fast = statistics.median(value[1] for value in series["fast"])
safe = statistics.median(value[1] for value in series["safe"])
screen_off = statistics.median(value[1] for value in series["screen_off"])
screen_on = statistics.median(value[1] for value in series["screen_on"])
fast_prefill = statistics.median(value[0] for value in series["fast"])
safe_prefill = statistics.median(value[0] for value in series["safe"])
fast_a_drift = abs(series["fast"][1][1] / series["fast"][0][1] - 1.0)
screen_a_drift = abs(
    series["screen_off"][1][1] / series["screen_off"][0][1] - 1.0
)
screen_prefill_off = statistics.median(
    value[0] for value in series["screen_off"]
)
screen_prefill_on = statistics.median(
    value[0] for value in series["screen_on"]
)
safe_ratio = safe / fast
safe_prefill_ratio = safe_prefill / fast_prefill
screen_ratio = screen_on / screen_off
screen_prefill_ratio = screen_prefill_on / screen_prefill_off
net_ratio = safe_ratio * screen_ratio
net_prefill_ratio = safe_prefill_ratio * screen_prefill_ratio
gate = {
    "safe_tax_a_drift_le_3pct": fast_a_drift <= 0.03,
    "screen_a_drift_le_3pct": screen_a_drift <= 0.03,
    "screen_generation_ratio_ge_1_015": screen_ratio >= 1.015,
    "screen_prefill_ratio_ge_0_99": screen_prefill_ratio >= 0.99,
    "net_ratio_ge_0_99": net_ratio >= 0.99,
    "net_prefill_ratio_ge_0_99": net_prefill_ratio >= 0.99,
}
performance_pass = all(gate.values())
report = f"{root}/metadata/lmhead-net-gate.csv"
with open(report, "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=[
        "fast_prefill_median", "safe_prefill_median",
        "fast_generation_median", "safe_generation_median",
        "screen_safe_off_generation_median", "screen_on_generation_median",
        "screen_safe_off_prefill_median", "screen_on_prefill_median",
        "safe_tax_ratio", "safe_prefill_ratio", "screen_generation_ratio",
        "screen_prefill_ratio", "net_ratio", "net_prefill_ratio",
        "safe_tax_a_drift", "screen_a_drift",
        *gate,
        "performance_gate",
    ])
    writer.writeheader()
    writer.writerow({
        "fast_prefill_median": fast_prefill,
        "safe_prefill_median": safe_prefill,
        "fast_generation_median": fast,
        "safe_generation_median": safe,
        "screen_safe_off_generation_median": screen_off,
        "screen_on_generation_median": screen_on,
        "screen_safe_off_prefill_median": screen_prefill_off,
        "screen_on_prefill_median": screen_prefill_on,
        "safe_tax_ratio": safe_ratio,
        "safe_prefill_ratio": safe_prefill_ratio,
        "screen_generation_ratio": screen_ratio,
        "screen_prefill_ratio": screen_prefill_ratio,
        "net_ratio": net_ratio,
        "net_prefill_ratio": net_prefill_ratio,
        "safe_tax_a_drift": fast_a_drift,
        "screen_a_drift": screen_a_drift,
        **gate,
        "performance_gate": performance_pass,
    })
with open(report + ".status.txt", "w", encoding="utf-8") as stream:
    stream.write(f"status={'pass' if performance_pass else 'fail'}\n")
print(
    f"safe_tax_ratio={safe_ratio:.9f} safe_prefill_ratio={safe_prefill_ratio:.9f} "
    f"screen_generation_ratio={screen_ratio:.9f} "
    f"screen_prefill_ratio={screen_prefill_ratio:.9f} "
    f"net_ratio={net_ratio:.9f} net_prefill_ratio={net_prefill_ratio:.9f} "
    f"performance_gate={performance_pass} "
    f"report={report}"
)
for name, passed in gate.items():
    if not passed:
        print(f"performance gate failed (evidence retained): {name}", file=sys.stderr)
PY
shasum -a 256 "$DS4_RESULTS/metadata/lmhead-net-gate.csv" \
  "$DS4_RESULTS/metadata/lmhead-net-gate.csv.status.txt" \
  > "$DS4_RESULTS/metadata/lmhead-net-gate.csv.sha256"
[[ -s "$DS4_RESULTS/metadata/lmhead-net-gate.csv.status.txt" ]] || {
  print -u2 'lm-head performance status sidecar is missing or empty'
  return 1
}
if grep -q '^status=fail$' \
    "$DS4_RESULTS/metadata/lmhead-net-gate.csv.status.txt"; then
  semantic_fail 'one or more executable lm-head performance thresholds failed'
elif ! grep -q '^status=pass$' \
    "$DS4_RESULTS/metadata/lmhead-net-gate.csv.status.txt"; then
  print -u2 'lm-head performance status sidecar is malformed'
  return 1
fi
semantic_ledger="$DS4_RESULTS/metadata/semantic-gates.txt"
{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  if (( ${#semantic_failures[@]} == 0 )); then
    print 'status=PASS'
  else
    print 'status=FAIL'
    for failure in "${semantic_failures[@]}"; do
      print "failure=$failure"
    done
    print 'decision_must_not_be=promote'
  fi
} > "$semantic_ledger"
```

The helper exits non-zero only for malformed timing evidence. A failed
performance threshold is written as `performance_gate=False` and must produce
an explicit `reject` or `keep opt-in` decision in `SUMMARY.md`; retain and
package the valid correctness evidence regardless.

The old one-row, GQA3, paired-prefill, raw-argmax, Q4, and DFlash sections
below are optional historical experiments. Do not run them for this return.

## Optional / historical experiments (not part of this return)

### Resident one-row-per-SIMD QMV

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

### Full-ring Laguna SWA GQA3 (historical)

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

### Paired Laguna prefill Q/K norm and RoPE (historical)

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

### Direct raw GPU argmax (historical)

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

### DFlash Q2/Q3 verifier batching and staged SWA (historical)

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

For the bounded return, apply only these gates before calculating speedups:

- every command exited zero; a process failure, invalid-token error, or missing
  winner is an incomplete/fatal arm, not a reportable semantic reject, and must
  be diagnosed in a fresh bounded run;
- all six clean Q2/Q3 decoded outputs and 8192 frontier JSON files are
  byte-identical;
- every lm-head timed arm records `requested_tokens=256` plus an output-byte
  count, and generated output is byte-identical across A-B-B-A;
- the screen throughput B arm has no trace, fallback, unavailable, or failed
  diagnostic, and its sidecopy is not created in the A arm;
- the screen trace CSV contains every emitted stats line, with
  `coarse_nonfinite=0` for every real-prompt row;
- the exact winner probe reports equal token ID, equal float32 winner-value
  bits, and byte-identical generated output;
- both lm-head init arms have non-empty maximum-RSS evidence and every new arm
  has redacted power-profile plus `vm.swapusage` before/after snapshots;
- no measured stderr contains an unexpected `failed`, `unavailable`,
  `fallback`, `paused`, `mismatch`, or invalid-token message (the focused
  smoke's intentional one-call-per-batch diagnostic is expected);
- thermal pressure remained nominal and no swap or memory-pressure event
  occurred during one arm but not its peer; High Power remained enabled, Low
  Power remained disabled, and AC remained connected;
- the first and last A observations in each A-B-B-A matrix differ by no more
  than 3% for the primary metric.
- `lmhead-net-gate.csv` records and evaluates safe-tax A drift ≤3%, screen A
  drift ≤3%, screen generation ratio ≥1.015, screen prefill ratio ≥0.99, and
  both median net ratios (generation and prefill) ≥0.99; a performance failure
  is reported and packaged.

If the two A observations drift by more than 3%, record the semantic failure,
cool the machine, and do not rerun in this results directory: start a fresh
`DS4_BENCH_ROOT` and repeat the complete bounded return so stale failures cannot
be mixed with replacement observations. A positive result below 1.5% remains
opt-in and requires at least five alternating observations per arm before
promotion consideration.

## Metrics to calculate

For the clean Q2/Q3 three-way, preserve both observations and the median for
each revision in `clean-q23-metrics.csv` for all four ds4-bench metrics:
`prefill_tps` (control), `gen_tps`, `gen_steady_tps` (primary metric and drift
gate), and `gen_first_ms`. Write `clean-q23-deltas.csv` with all four metrics
for pre-rebase/basic → rebased baseline, rebased baseline → candidate, and
pre-rebase/basic → candidate. Use `(right/left-1)*100` for throughput metrics
and `(left/right-1)*100` for the `gen_first_ms` latency delta.

For lm-head A–E, report:

- A safe-mode tax: raw Laguna prefill/generation t/s and median B versus A;
- B screen throughput: raw Laguna prefill/generation t/s, median B versus A,
  output SHA, requested token count plus output-byte evidence, and
  `/usr/bin/time -p`;
- safe-tax generation ratio, safe-tax prefill ratio, screen generation ratio,
  screen prefill ratio, and the explicit median net generation ratio
  `(safe_tax_safe_median/fast_median) * (screen_on_median/screen_safe_off_median)`;
  also report `net_prefill_ratio=(safe_tax_safe_prefill/fast_prefill) *
  (screen_on_prefill/screen_safe_off_prefill)`, written by the net-gate helper;
- C safe-off/safe-on max RSS, vm-stat, memory-pressure, sidecopy bytes, and
  `sidecopy_init_ms` (allocations + sidecopy only, excluding pipeline/model
  wrapping);
- D every trace row's `screen_calls`, candidate rows/row-blocks,
  `exact_row_blocks`, coarse-nonfinite count, winner ID/bits, candidate and
  physical exact percentages, exact Q8 bytes, and estimated weight bytes;
- E stock rank-0 ID/printed value/float32 bits, screen winner ID/bits, exactness
  booleans, generated-output byte parity, and the winner-parity/report SHA
  (`winner-parity.csv.sha256`).

Use `(new / control - 1) * 100` for throughput deltas and
`(control / new - 1) * 100` for latency improvements. Do not average tokens/s
across revisions, contexts, or quants.

Promotion gates for this return are: generated output byte exact; exact probe
ID and winner-value bits exact; all real-prompt `coarse_nonfinite=0`; no
fallback/unavailable/failed diagnostics; A-arm drift ≤3%; screen versus
safe-off median generation t/s ≥+1.5%; prefill regression ≤1%;
`net_ratio=(safe_tax_safe_median/fast_median) *
(screen_on_median/screen_safe_off_median) >= 0.99`; and
`net_prefill_ratio=(safe_tax_safe_prefill/fast_prefill) *
(screen_on_prefill/screen_safe_off_prefill) >= 0.99`. Report both component
ratios and both net ratios, with the safe-mode tax separately. Reject parity, nonfinite,
all-row-pruning, thermal/swap/memory-pressure, or net-regression failures. A
process/invalid-token/no-winner failure aborts the arm before packaging and must
be diagnosed in a fresh bounded run. A positive result below +1.5% stays opt-in
and needs ≥5 alternating observations.

## Report the result

Create `SUMMARY.md` in `$DS4_RESULTS` with these tables:

1. Host identity: exact Mac model, chip, GPU cores, RAM, macOS build, Xcode/Clang
   version, power mode, display configuration, and thermal status.
2. Input identity: the three source SHAs (including candidate
   `202a46b5158bff38ba36e066d71bc5b242b53a4a`) and mandatory Q2/Q3/model/prompt
   SHA-256 values.
3. Clean Q2/Q3: both observations and median for pre-rebase/basic, rebased
   baseline, and candidate, with all three labelled deltas.
4. Candidate lm-head A–E: safe tax, screen throughput, init/RSS, trace CSV,
   exact winner parity, output hashes, requested-token/output-byte records,
   and state evidence.
5. Decisions: `promote`, `keep opt-in`, `reject`, or `rerun` for the screen,
   grounded in the gates above; report the safe tax separately. A `rerun`
   means a fresh `DS4_BENCH_ROOT` and complete bounded return, never an
   in-place overwrite of this results directory.
6. Every warning, intentional diagnostic, fallback, thermal excursion, failed
   parity check, or rerun. Explicitly state that excluded historical sections
   were not run.

The `metadata/semantic-gates.txt` ledger is authoritative for measured
correctness/performance rejects. If it is `status=FAIL`, the decision cannot
be `promote`; retain the full evidence and choose `reject`, `keep opt-in`, or
`rerun` as appropriate. Any rerun starts a fresh root as described above.

Do not add Q4, R1, GQA3, paired-prefill, raw-argmax legacy, or DFlash rows to
the current return tables. They may be reported only if a later optional
historical run is explicitly requested.

## Package the evidence

Record final host state, add a manifest, and archive only the bounded-return
results—not models or worktrees. All paths in the returned command and manifest
must be absolute:

```zsh
[[ -s "$DS4_RESULTS/metadata/semantic-gates.txt" ]] || {
  print -u2 'semantic gate ledger is missing or empty'
  return 1
}
verify_power_and_thermals > "$DS4_RESULTS/metadata/power-final.txt" 2>&1 || return 1
{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  pmset -g custom
  pmset -g batt
  pmset -g therm
  capture_profiler_json /dev/stdout SPPowerDataType
  capture_profiler_json /dev/stdout SPDisplaysDataType
  sysctl vm.swapusage
  memory_pressure -Q
  vm_stat
} > "$DS4_RESULTS/metadata/host-after.txt" 2>&1

typeset -a required_artifacts
required_artifacts=(
  "$DS4_RESULTS/SUMMARY.md"
  "$DS4_RESULTS/metadata/host-before.txt"
  "$DS4_RESULTS/metadata/host-after.txt"
  "$DS4_RESULTS/metadata/power-before.txt"
  "$DS4_RESULTS/metadata/worktree-status-after-build.txt"
  "$DS4_RESULTS/metadata/benchmark-start.txt"
  "$DS4_RESULTS/metadata/power-at-benchmark-start.txt"
  "$DS4_RESULTS/metadata/cooldown-after-default-q23.txt"
  "$DS4_RESULTS/metadata/power-after-default-q23.txt"
  "$DS4_RESULTS/metadata/cooldown-after-lmhead-safe-tax.txt"
  "$DS4_RESULTS/metadata/power-after-lmhead-safe-tax.txt"
  "$DS4_RESULTS/metadata/cooldown-after-lmhead-screen.txt"
  "$DS4_RESULTS/metadata/power-after-lmhead-screen.txt"
  "$DS4_RESULTS/metadata/cooldown-after-lmhead-init.txt"
  "$DS4_RESULTS/metadata/power-after-lmhead-init.txt"
  "$DS4_RESULTS/metadata/cooldown-after-lmhead-trace.txt"
  "$DS4_RESULTS/metadata/power-after-lmhead-trace.txt"
  "$DS4_RESULTS/metadata/cooldown-after-lmhead-winner.txt"
  "$DS4_RESULTS/metadata/power-after-lmhead-winner.txt"
  "$DS4_RESULTS/metadata/power-final.txt"
  "$DS4_RESULTS/metadata/semantic-gates.txt"
  "$DS4_RESULTS/metadata/candidate-lmhead-smoke/cooldown-after.txt"
  "$DS4_RESULTS/metadata/candidate-lmhead-smoke/power-after-cooldown.txt"
  "$DS4_RESULTS/metadata/revisions.csv"
  "$DS4_RESULTS/metadata/runbook-commit.txt"
  "$DS4_RESULTS/metadata/BENCHMARK.md"
  "$DS4_RESULTS/metadata/input-sha256.txt"
  "$DS4_RESULTS/metadata/metal-kernels/output-sha256.txt"
  "$DS4_RESULTS/metadata/candidate-lmhead-smoke/output-sha256.txt"
  "$DS4_RESULTS/metadata/build-pre-rebase/output-sha256.txt"
  "$DS4_RESULTS/metadata/build-rebased-baseline/output-sha256.txt"
  "$DS4_RESULTS/metadata/build-candidate/output-sha256.txt"
  "$DS4_RESULTS/metadata/build-candidate-ds4-test/output-sha256.txt"
  "$DS4_RESULTS/clean-q23-metrics.csv"
  "$DS4_RESULTS/clean-q23-deltas.csv"
  "$DS4_RESULTS/lmhead-init/safe-off/max-rss.txt"
  "$DS4_RESULTS/lmhead-init/safe-on/max-rss.txt"
  "$DS4_RESULTS/lmhead-init/safe-on/screen-diagnostic.txt"
  "$DS4_RESULTS/lmhead-trace/q23/lmhead-trace.csv"
  "$DS4_RESULTS/lmhead-trace/q23/lmhead-trace.csv.summary.txt"
  "$DS4_RESULTS/lmhead-trace/q23/lmhead-trace.csv.sha256"
  "$DS4_RESULTS/lmhead-winner/winner-parity.csv"
  "$DS4_RESULTS/lmhead-winner/winner-parity.csv.status.txt"
  "$DS4_RESULTS/lmhead-winner/winner-parity.csv.sha256"
  "$DS4_RESULTS/metadata/lmhead-net-gate.csv"
  "$DS4_RESULTS/metadata/lmhead-net-gate.csv.status.txt"
  "$DS4_RESULTS/metadata/lmhead-net-gate.csv.sha256"
)

typeset -a required_files required_nonempty_files
required_files=()
required_nonempty_files=()
append_arm_files() {
  local rel=$1
  local phase
  required_nonempty_files+=(
    "$DS4_RESULTS/$rel/command.txt"
    "$DS4_RESULTS/$rel/revision.txt"
    "$DS4_RESULTS/$rel/start-utc.txt"
    "$DS4_RESULTS/$rel/output-sha256.txt"
  )
  required_files+=(
    "$DS4_RESULTS/$rel/env-overrides.txt"
    "$DS4_RESULTS/$rel/stdout.log"
    "$DS4_RESULTS/$rel/stderr.log"
  )
  for phase in before after; do
    required_nonempty_files+=(
      "$DS4_RESULTS/$rel/battery-$phase.txt"
      "$DS4_RESULTS/$rel/thermal-$phase.txt"
      "$DS4_RESULTS/$rel/power-custom-$phase.txt"
      "$DS4_RESULTS/$rel/power-profile-$phase.txt"
      "$DS4_RESULTS/$rel/vm-swapusage-$phase.txt"
      "$DS4_RESULTS/$rel/vm-$phase.txt"
      "$DS4_RESULTS/$rel/memory-pressure-$phase.txt"
    )
  done
}

required_files+=(
  "$DS4_RESULTS/metadata/metal-kernels/env-overrides.txt"
  "$DS4_RESULTS/metadata/metal-kernels/stdout.log"
  "$DS4_RESULTS/metadata/metal-kernels/stderr.log"
  "$DS4_RESULTS/metadata/candidate-lmhead-smoke/env-overrides.txt"
  "$DS4_RESULTS/metadata/candidate-lmhead-smoke/stdout.log"
  "$DS4_RESULTS/metadata/candidate-lmhead-smoke/stderr.log"
)
required_nonempty_files+=(
  "$DS4_RESULTS/metadata/metal-kernels/command.txt"
  "$DS4_RESULTS/metadata/metal-kernels/revision.txt"
  "$DS4_RESULTS/metadata/metal-kernels/start-utc.txt"
  "$DS4_RESULTS/metadata/metal-kernels/output-sha256.txt"
  "$DS4_RESULTS/metadata/candidate-lmhead-smoke/command.txt"
  "$DS4_RESULTS/metadata/candidate-lmhead-smoke/revision.txt"
  "$DS4_RESULTS/metadata/candidate-lmhead-smoke/start-utc.txt"
  "$DS4_RESULTS/metadata/candidate-lmhead-smoke/output-sha256.txt"
)
for build in pre-rebase rebased-baseline candidate candidate-ds4-test; do
  required_files+=(
    "$DS4_RESULTS/metadata/build-$build/stdout.log"
    "$DS4_RESULTS/metadata/build-$build/stderr.log"
  )
  required_nonempty_files+=(
    "$DS4_RESULTS/metadata/build-$build/command.txt"
    "$DS4_RESULTS/metadata/build-$build/revision.txt"
    "$DS4_RESULTS/metadata/build-$build/start-utc.txt"
    "$DS4_RESULTS/metadata/build-$build/output-sha256.txt"
  )
done
for run in 01-pre-A 02-base-B 03-cand-C 04-cand-C 05-base-B 06-pre-A; do
  append_arm_files "default-q23/$run"
  required_artifacts+=(
    "$DS4_RESULTS/default-q23/$run/metrics.csv"
    "$DS4_RESULTS/default-q23/$run/logits/frontier_008192.logits.json"
  )
  required_artifacts+=("$DS4_RESULTS/default-q23/$run/decoded.txt")
done
for run in 01-fast-A 02-safe-B 03-safe-B 04-fast-A; do
  append_arm_files "lmhead-safe-tax/$run"
  required_artifacts+=("$DS4_RESULTS/lmhead-safe-tax/$run/generated.txt"
                       "$DS4_RESULTS/lmhead-safe-tax/$run/token-count.txt")
done
for run in 01-safe-off-A 02-safe-on-B 03-safe-on-B 04-safe-off-A; do
  append_arm_files "lmhead-screen/$run"
  required_artifacts+=("$DS4_RESULTS/lmhead-screen/$run/generated.txt"
                       "$DS4_RESULTS/lmhead-screen/$run/token-count.txt")
done
for run in safe-off safe-on; do
  append_arm_files "lmhead-init/$run"
  required_files+=("$DS4_RESULTS/lmhead-init/$run/generated.txt"
                   "$DS4_RESULTS/lmhead-init/$run/screen-diagnostic.txt")
  required_artifacts+=("$DS4_RESULTS/lmhead-init/$run/max-rss.txt")
done
append_arm_files lmhead-trace/q23
required_files+=("$DS4_RESULTS/lmhead-trace/q23/generated.txt")
append_arm_files lmhead-trace/q23-control
required_files+=("$DS4_RESULTS/lmhead-trace/q23-control/generated.txt")
for run in stock screen; do
  append_arm_files "lmhead-winner/$run"
  required_files+=("$DS4_RESULTS/lmhead-winner/$run/generated.txt")
done
for artifact in "${required_artifacts[@]}"; do
  [[ -s "$artifact" ]] || {
    print -u2 "missing required bounded-return artifact: $artifact"
    return 1
  }
done
for artifact in "${required_nonempty_files[@]}"; do
  [[ -s "$artifact" ]] || {
    print -u2 "missing or empty required bounded-return evidence: $artifact"
    return 1
  }
done
for artifact in "${required_files[@]}"; do
  [[ -f "$artifact" ]] || {
    print -u2 "missing required bounded-return file: $artifact"
    return 1
  }
done

find "$DS4_RESULTS" -type f -print | LC_ALL=C sort \
  > "$DS4_RESULTS/MANIFEST.txt"
ARCHIVE="$DS4_BENCH_ROOT/ds4-laguna-m5-max-results.tar.gz"
COPYFILE_DISABLE=1 tar -C "$DS4_BENCH_ROOT" -czf "$ARCHIVE" results
shasum -a 256 "$ARCHIVE" | tee "$ARCHIVE.sha256"
print "Return $DS4_RESULTS/SUMMARY.md, $ARCHIVE, and $ARCHIVE.sha256"
```

Keep `$DS4_BENCH_ROOT` until the archive has been received and verified.

## Optional extended sweep after the first return

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
