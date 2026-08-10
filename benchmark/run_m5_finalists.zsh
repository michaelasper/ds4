#!/bin/zsh

# Reproducible M5 Max confirmation runner for the Laguna S 2.1 finalists.
# Timed processes use c8f25ee. The runner and analysis are provenance-only.

emulate -LR zsh
setopt NO_UNSET PIPE_FAIL NULL_GLOB
unsetopt ERR_EXIT

readonly RUNNER_PATH=${${(%):-%N}:A}
readonly RUNBOOK_REPO=${RUNNER_PATH:h:h}
readonly ANALYSER_PATH="$RUNBOOK_REPO/benchmark/m5_finalist_analysis.py"

readonly TIMED_SHA=c8f25ee7ef1b6f16eff470ccf337d1546e314906
readonly TRACE_SHA=4322b2ca7664040b811de4426cc791cb499b2866
readonly EXPECTED_MODEL_SHA=61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903
readonly EXPECTED_LONG_PROMPT_SHA=f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f
readonly DISCOVERY_RELEASE=bench-laguna-s2.1-m5-max-2026-08-10
readonly DISCOVERY_ARCHIVE_SHA=0298bf9b3ca01a2bdba44b0dd356156eacd0de26589fc51105136f844e053146

typeset -gr BENCH_ROOT=${DS4_BENCH_ROOT:-}
typeset -gr GGUF_DIR=${DS4_GGUF_DIR:-}
typeset -gir JOBS=${DS4_JOBS:-8}
typeset -gir RANDOM_SEED=20260810
typeset -gir COOLDOWN_SECONDS=${DS4_COOLDOWN_SECONDS:-30}

typeset -g CONTROL_REPO RESULTS TIMED_WT TRACE_WT ISOLATED_HOME MODEL_SOURCE MODEL_LINK
typeset -g RUNBOOK_SHA MODEL_SHA MODEL_STAT
typeset -ga R1_ENV D8_ARGS TRACE_D8_ARGS PARITY_DECODE_ARGS PARITY_PREFILL_ARGS
typeset -ga SELECTOR_ENV ACTIVE_ARMS
typeset -gra CANDIDATE_ARMS=(
  05-canonical-ladder
  12-gqa3
  14-front-rung-ladder
  15-canonical-ladder-gqa3
  16-front-rung-ladder-gqa3
)

R1_ENV=(DS4_METAL_GLM_QMV_R1=1)

usage() {
  print -u2 -- "usage: ${RUNNER_PATH:t} prepare|prove|measure|analyse|package|all|upload-draft"
  print -u2 -- 'required environment: DS4_BENCH_ROOT=/private/tmp/fresh-dir DS4_GGUF_DIR=/absolute/model/dir'
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    print -u2 -- "missing required command: $1"
    return 1
  }
}

load_paths() {
  [[ -n "$BENCH_ROOT" ]] || { print -u2 -- 'DS4_BENCH_ROOT is required'; return 1; }
  [[ -n "$GGUF_DIR" ]] || { print -u2 -- 'DS4_GGUF_DIR is required'; return 1; }
  [[ "$BENCH_ROOT" == /* ]] || { print -u2 -- 'DS4_BENCH_ROOT must be absolute'; return 1; }
  [[ "$GGUF_DIR" == /* ]] || { print -u2 -- 'DS4_GGUF_DIR must be absolute'; return 1; }
  case "$BENCH_ROOT" in
    /|/tmp|/private/tmp|"$HOME")
      print -u2 -- "DS4_BENCH_ROOT is too broad: $BENCH_ROOT"
      return 1
      ;;
  esac
  (( JOBS >= 1 )) || { print -u2 -- 'DS4_JOBS must be at least 1'; return 1; }
  (( COOLDOWN_SECONDS >= 0 )) || { print -u2 -- 'DS4_COOLDOWN_SECONDS must not be negative'; return 1; }
  CONTROL_REPO="$BENCH_ROOT/control"
  RESULTS="$BENCH_ROOT/results"
  TIMED_WT="$BENCH_ROOT/timed"
  TRACE_WT="$BENCH_ROOT/trace"
  ISOLATED_HOME="$BENCH_ROOT/isolated-home"
  MODEL_SOURCE="$GGUF_DIR/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf"
  MODEL_LINK="$BENCH_ROOT/input/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf"
  D8_ARGS=(
    --metal -m "$MODEL_LINK" --prompt-file "$TIMED_WT/speed-bench/promessi_sposi.txt"
    --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8321
    --step-incr 1 --gen-tokens 128 --warm-weights
  )
  TRACE_D8_ARGS=(
    --metal -m "$MODEL_LINK" --prompt-file "$TIMED_WT/speed-bench/promessi_sposi.txt"
    --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8194
    --step-incr 1 --gen-tokens 1 --warm-weights
  )
  PARITY_DECODE_ARGS=(
    --metal -m "$MODEL_LINK" --prompt-file "$TIMED_WT/speed-bench/promessi_sposi.txt"
    --ctx-start 8192 --ctx-max 8193 --ctx-alloc 8194
    --step-incr 1 --gen-tokens 0 --warm-weights
  )
  PARITY_PREFILL_ARGS=(
    --metal -m "$MODEL_LINK" --prompt-file "$TIMED_WT/speed-bench/promessi_sposi.txt"
    --ctx-start 16384 --ctx-max 16384 --ctx-alloc 16385
    --step-incr 1 --gen-tokens 0 --warm-weights
  )
}

set_selector_env() {
  local arm=$1
  SELECTOR_ENV=()
  case "$arm" in
    00-r1-baseline) ;;
    05-canonical-ladder)
      SELECTOR_ENV=(DS4_METAL_LAGUNA_DECODE_LADDER=7,15,23,31,39,47)
      ;;
    12-gqa3)
      SELECTOR_ENV=(DS4_METAL_LAGUNA_SWA_GQA3=1)
      ;;
    14-front-rung-ladder)
      SELECTOR_ENV=(DS4_METAL_LAGUNA_DECODE_LADDER=1,7,15,23,31,39,47)
      ;;
    15-canonical-ladder-gqa3)
      SELECTOR_ENV=(
        DS4_METAL_LAGUNA_DECODE_LADDER=7,15,23,31,39,47
        DS4_METAL_LAGUNA_SWA_GQA3=1
      )
      ;;
    16-front-rung-ladder-gqa3)
      SELECTOR_ENV=(
        DS4_METAL_LAGUNA_DECODE_LADDER=1,7,15,23,31,39,47
        DS4_METAL_LAGUNA_SWA_GQA3=1
      )
      ;;
    *) print -u2 -- "unknown arm: $arm"; return 1 ;;
  esac
}

all_candidates() {
  print -rl -- "${CANDIDATE_ARMS[@]}"
}

has_gqa3() {
  [[ "$1" == 12-gqa3 || "$1" == 15-canonical-ladder-gqa3 || "$1" == 16-front-rung-ladder-gqa3 ]]
}

has_canonical_ladder() {
  [[ "$1" == 05-canonical-ladder || "$1" == 15-canonical-ladder-gqa3 ]]
}

has_front_ladder() {
  [[ "$1" == 14-front-rung-ladder || "$1" == 16-front-rung-ladder-gqa3 ]]
}

redacted_profiler_json() {
  local destination=$1
  shift
  system_profiler -json "$@" 2>/dev/null |
    python3 -c '
import json, re, sys
key_re = re.compile(
    r"serial|uuid|udid|identifier|hardware.?address|platform.?id|"
    r"local.?host.?name|user.?name|computer.?name|host.?name",
    re.I,
)
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
    payload = json.load(sys.stdin)
except Exception:
    payload = {"status": "profiler-unavailable"}
json.dump(scrub(payload), sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\\n")
' > "$destination"
}

capture_state() {
  local out=$1
  mkdir -p "$out"
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ' > "$out/time.txt" 2>&1 || true
  redacted_profiler_json "$out/host-profiler-sanitised.json" \
    SPHardwareDataType SPSoftwareDataType || true
  redacted_profiler_json "$out/power-display-profiler-sanitised.json" \
    SPPowerDataType SPDisplaysDataType || true
  pmset -g batt 2>&1 | /usr/bin/sed -E 's/id=[0-9]+/id=<redacted>/g' \
    > "$out/pmset-batt.txt" || true
  pmset -g custom > "$out/pmset-custom.txt" 2>&1 || true
  pmset -g therm > "$out/pmset-therm.txt" 2>&1 || true
  sysctl hw.model hw.ncpu hw.memsize kern.osproductversion kern.osversion \
    > "$out/sysctl-host.txt" 2>&1 || true
  sysctl vm.swapusage > "$out/vm-swapusage.txt" 2>&1 || true
  vm_stat > "$out/vm-stat.txt" 2>&1 || true
  memory_pressure -Q > "$out/memory-pressure.txt" 2>&1 || true
  df -h "$BENCH_ROOT" > "$out/df.txt" 2>&1 || true
  print -- 'informational_only=power_mode,ac_state,thermal_state,display_state' \
    > "$out/gating-policy.txt"
}

validate_memory_state() {
  local before=$1
  local after=$2
  local report=$3
  python3 - "$before" "$after" "$report" <<'PY'
import re, sys
from pathlib import Path

before, after, report = map(Path, sys.argv[1:])

def text(root, name):
    value = (root / name).read_text(errors="replace")
    if not value.strip():
        raise ValueError(f"empty {root / name}")
    return value

def vm_stat(root):
    result = {}
    for line in text(root, "vm-stat.txt").splitlines():
        match = re.match(r"([^:]+):\s*([0-9]+)", line)
        if match:
            result[match.group(1).strip().lower()] = int(match.group(2))
    missing = {"pageouts", "swapouts"} - result.keys()
    if missing:
        raise ValueError(f"missing {sorted(missing)} in {root / 'vm-stat.txt'}")
    return result

def used_mib(root):
    value = text(root, "vm-swapusage.txt")
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", value, re.I)
    if not match:
        raise ValueError(f"cannot parse used swap in {root / 'vm-swapusage.txt'}")
    amount = float(match.group(1))
    scale = {"K": 1 / 1024, "M": 1, "G": 1024}[match.group(2).upper()]
    return amount * scale

try:
    bstat, astat = vm_stat(before), vm_stat(after)
    bswap, aswap = used_mib(before), used_mib(after)
    pageouts = astat["pageouts"] - bstat["pageouts"]
    swapouts = astat["swapouts"] - bstat["swapouts"]
    swap_growth = aswap - bswap
    clean = pageouts == 0 and swapouts == 0 and swap_growth <= 1.0
    report.write_text(
        f"pageouts_delta={pageouts}\n"
        f"swapouts_delta={swapouts}\n"
        f"swap_used_mib_before={bswap:.3f}\n"
        f"swap_used_mib_after={aswap:.3f}\n"
        f"swap_used_mib_delta={swap_growth:.3f}\n"
        f"status={'PASS' if clean else 'REJECT'}\n"
    )
except Exception as error:
    report.write_text(f"status=ERROR\nerror={error}\n")
    raise SystemExit(2)
raise SystemExit(0 if clean else 1)
PY
}

validate_diagnostics() {
  local out=$1
  python3 - "$out" <<'PY'
import glob, re, sys
from pathlib import Path

root = Path(sys.argv[1])
paths = [Path(value) for value in glob.glob(str(root / "*.stderr.log"))]
report = root / "unexpected-diagnostics.txt"
pattern = re.compile(
    r"fallback|unavailable|unsupported|paused|fail(?:ed|ure)?|invalid|"
    r"mismatch|refusal|error|fault|reject(?:ed|ion)?",
    re.I,
)
try:
    if not paths:
        raise RuntimeError("no stderr logs were available to scan")
    matches = []
    for stderr_file in paths:
        operational = stderr_file.read_bytes().split(
            b'ds4-bench: gen[ctx=8192] decoded text: "', 1
        )[0].decode(errors="replace")
        for number, line in enumerate(operational.splitlines(), 1):
            if re.fullmatch(r"\s*[0-9]+\s+page faults?\s*", line, re.I):
                continue
            if pattern.search(line):
                matches.append(f"{stderr_file.name}:{number}:{line}")
    report.write_text("\n".join(matches) + ("\n" if matches else ""))
except Exception as error:
    report.write_text(f"scanner_error={error}\n")
    raise SystemExit(2)
raise SystemExit(1 if matches else 0)
PY
}

cool_down() {
  if (( COOLDOWN_SECONDS > 0 )); then
    sleep "$COOLDOWN_SECONDS"
  fi
}

refuse_directory() {
  local out=$1
  if [[ -e "$out" ]]; then
    print -u2 -- "refusing to overwrite existing evidence: $out"
    return 1
  fi
  mkdir -p "$out"
}

model_stat_value() {
  stat -f '%d:%i:%z:%m:%c' "$MODEL_SOURCE"
}

verify_model_identity() {
  local expected_stat actual_stat actual_link
  [[ -f "$MODEL_SOURCE" && -L "$MODEL_LINK" ]] || {
    print -u2 -- 'model source or pinned model symlink is missing'
    return 1
  }
  actual_link=$(readlink "$MODEL_LINK") || return 1
  [[ "$actual_link" == "$MODEL_SOURCE" ]] || {
    print -u2 -- "model symlink changed: $actual_link"
    return 1
  }
  expected_stat=$(<"$RESULTS/metadata/model-stat.txt") || return 1
  actual_stat=$(model_stat_value) || return 1
  [[ "$actual_stat" == "$expected_stat" ]] || {
    print -u2 -- "model file identity changed: expected $expected_stat, found $actual_stat"
    return 1
  }
}

verify_binary_identity() {
  local wt=$1
  local ledger actual expected
  if [[ "$wt" == "$TIMED_WT" ]]; then
    ledger="$RESULTS/metadata/timed-binary-sha256.txt"
  elif [[ "$wt" == "$TRACE_WT" ]]; then
    ledger="$RESULTS/metadata/trace-binary-sha256.txt"
  else
    print -u2 -- "binary identity requested for unknown worktree: $wt"
    return 1
  fi
  [[ -x "$wt/ds4-bench" && -s "$ledger" ]] || return 1
  expected=$(<"$ledger")
  actual=$(shasum -a 256 "$wt/ds4-bench" | awk '{print $1}') || return 1
  [[ "$actual" == "$expected" ]] || {
    print -u2 -- "ds4-bench binary changed in $wt"
    return 1
  }
}

verify_prompt_identity() {
  local prompt="$TIMED_WT/speed-bench/promessi_sposi.txt"
  local actual
  actual=$(shasum -a 256 "$prompt" | awk '{print $1}') || return 1
  [[ "$actual" == "$EXPECTED_LONG_PROMPT_SHA" ]] || {
    print -u2 -- "prompt identity changed: $actual"
    return 1
  }
}

verify_runtime_identity() {
  verify_model_identity && verify_prompt_identity && verify_binary_identity "$1"
}

write_identity() {
  local out=$1
  local wt=$2
  local arm=$3
  local expected actual
  if [[ "$wt" == "$TIMED_WT" ]]; then
    expected=$TIMED_SHA
  elif [[ "$wt" == "$TRACE_WT" ]]; then
    expected=$TRACE_SHA
  else
    print -u2 -- "identity requested for unknown worktree: $wt"
    return 1
  fi
  actual=$(git -C "$wt" rev-parse HEAD) || return 1
  [[ "$actual" == "$expected" ]] || {
    print -u2 -- "worktree revision changed: expected $expected, found $actual"
    return 1
  }
  [[ "$MODEL_SHA" == "$EXPECTED_MODEL_SHA" ]] || {
    print -u2 -- "model identity changed: $MODEL_SHA"
    return 1
  }
  verify_runtime_identity "$wt" || return 1
  print -r -- "$actual" > "$out/revision.txt"
  print -r -- "$MODEL_SHA" > "$out/model-sha256.txt"
  print -r -- "$arm" > "$out/arm.txt"
  print -rl -- "${R1_ENV[@]}" "${SELECTOR_ENV[@]}" > "$out/env-overrides.txt"
}

verify_protocol_snapshot() {
  local metadata="$RESULTS/metadata"
  [[ -s "$metadata/protocol-files.sha256" && -s "$metadata/runbook-revision.txt" ]] || {
    print -u2 -- 'protocol snapshot is missing'
    return 1
  }
  ( cd "$metadata" && shasum -c protocol-files.sha256 >/dev/null ) || {
    print -u2 -- 'archived protocol files no longer match their hashes'
    return 1
  }
  cmp "$RUNBOOK_REPO/BENCHMARK.md" "$metadata/BENCHMARK.md" >/dev/null || return 1
  cmp "$RUNNER_PATH" "$metadata/run_m5_finalists.zsh" >/dev/null || return 1
  cmp "$ANALYSER_PATH" "$metadata/m5_finalist_analysis.py" >/dev/null || return 1
  local expected actual runbook_status
  expected=$(<"$metadata/runbook-revision.txt")
  actual=$(git -C "$RUNBOOK_REPO" rev-parse HEAD) || return 1
  runbook_status=$(git -C "$RUNBOOK_REPO" status --porcelain=v1) || return 1
  [[ "$actual" == "$expected" && -z "$runbook_status" ]] || {
    print -u2 -- 'live runbook checkout changed after prepare'
    return 1
  }
}

run_d8_arm() {
  local rel=$1
  local arm=$2
  local out="$RESULTS/$rel"
  set_selector_env "$arm" || return 1
  refuse_directory "$out" || return 1
  write_identity "$out" "$TIMED_WT" "$arm" || return 1
  {
    print -r -- "cwd=$TIMED_WT"
    print -r -- "binary=$TIMED_WT/ds4-bench"
    print -r -- 'workload=D8'
    print -r -- 'trace=off'
    printf 'env=%q\n' "${R1_ENV[@]}" "${SELECTOR_ENV[@]}"
    printf 'arg=%q\n' "$TIMED_WT/ds4-bench" "${D8_ARGS[@]}" --csv "$out/d8.csv"
  } > "$out/command.txt"
  capture_state "$out/state-before"
  : > "$out/d8.csv"
  ( cd "$TIMED_WT" && /usr/bin/time -l env -i \
      HOME="$ISOLATED_HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
      TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C AppleLanguages='(en)' \
      "${R1_ENV[@]}" "${SELECTOR_ENV[@]}" \
      ./ds4-bench "${D8_ARGS[@]}" --csv "$out/d8.csv" \
      > "$out/d8.stdout.log" 2> "$out/d8.stderr.log" )
  local rc=$?
  verify_runtime_identity "$TIMED_WT" || return 1
  print -- "$rc" > "$out/d8.exit-code.txt"
  capture_state "$out/state-after"
  git -C "$TIMED_WT" status --porcelain=v1 > "$out/worktree-status.txt" || rc=69
  if (( rc == 0 )) && [[ -s "$out/worktree-status.txt" ]]; then
    rc=69
  fi
  if (( rc == 0 )) && [[ -s "$out/d8.csv" ]]; then
    python3 - "$out/d8.csv" <<'PY' || rc=65
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
required = ("ctx_tokens", "prefill_tokens", "prefill_tps", "gen_tokens",
            "gen_tps", "gen_first_ms", "gen_steady_tokens", "gen_steady_tps")
if len(rows) != 1 or any(key not in rows[0] for key in required):
    raise SystemExit(1)
row = rows[0]
for key in required:
    value = float(row[key])
    if not math.isfinite(value) or value < 0:
        raise SystemExit(1)
for key in ("prefill_tps", "gen_tps", "gen_first_ms", "gen_steady_tps"):
    if float(row[key]) <= 0:
        raise SystemExit(1)
expected = {
    "ctx_tokens": 8192,
    "prefill_tokens": 8192,
    "gen_tokens": 128,
    "gen_steady_tokens": 127,
}
if any(float(row[key]) != value for key, value in expected.items()):
    raise SystemExit(1)
PY
  elif (( rc == 0 )); then
    rc=65
  fi
  ( cd "$out" && shasum -a 256 d8.csv > d8.csv.sha256 ) 2>/dev/null || rc=65
  /usr/bin/grep -Ei 'maximum resident set size|peak resident|resident set' \
    "$out/d8.stderr.log" > "$out/d8.max-rss.txt" 2>&1 || true
  local scan_rc=0
  local memory_rc=0
  validate_diagnostics "$out"
  scan_rc=$?
  validate_memory_state "$out/state-before" "$out/state-after" "$out/memory-gate.txt"
  memory_rc=$?
  if (( rc != 0 )); then
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- "benchmark process or CSV validation exit=$rc" > "$out/reject-reason.txt"
  elif (( scan_rc != 0 )); then
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- "diagnostic scan rejected or errored, exit=$scan_rc" > "$out/reject-reason.txt"
  elif (( memory_rc != 0 )); then
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- "memory-state gate rejected or errored, exit=$memory_rc" > "$out/reject-reason.txt"
  else
    print -- PASS > "$out/status.txt"
  fi
  return 0
}

build_revision() {
  local label=$1
  local wt=$2
  local sha=$3
  local out="$RESULTS/metadata/build-$label"
  refuse_directory "$out" || return 1
  print -r -- "$sha" > "$out/revision.txt"
  git -C "$wt" status --porcelain=v1 > "$out/status-before.txt" || return 1
  {
    print -r -- "cwd=$wt"
    print -r -- "sanitised make -j$JOBS ds4 ds4-bench ds4_test"
  } > "$out/command.txt"
  ( cd "$wt" && env -i HOME="$ISOLATED_HOME" \
      PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" \
      LANG=C LC_ALL=C make -j"$JOBS" ds4 ds4-bench ds4_test ) \
      > "$out/stdout.log" 2> "$out/stderr.log"
  local rc=$?
  git -C "$wt" status --porcelain=v1 > "$out/status-after.txt" || rc=69
  if (( rc == 0 )) && [[ -s "$out/status-after.txt" ]]; then
    rc=69
  fi
  if (( rc == 0 )); then
    print -- PASS > "$out/status.txt"
  else
    print -- SEMANTIC_REJECT > "$out/status.txt"
  fi
  return "$rc"
}

verify_host_identity() {
  local out="$RESULTS/metadata/host"
  refuse_directory "$out" || return 1
  capture_state "$out/state"
  local chip mem
  chip=$(system_profiler SPHardwareDataType -json 2>/dev/null |
    python3 -c 'import json,sys; d=json.load(sys.stdin); print(" ".join(str(v) for x in d.values() for r in (x if isinstance(x,list) else []) for k,v in r.items() if "chip" in k.lower()))')
  mem=$(sysctl -n hw.memsize 2>/dev/null)
  print -r -- "$chip" > "$out/chip.txt"
  print -r -- "$mem" > "$out/memory-bytes.txt"
  print -- 'Power, AC/battery, Energy Mode, Low Power Mode, thermal, and display data are informational only.' \
    > "$out/power-policy.txt"
  [[ "$chip" == *'Apple M5 Max'* ]] || { print -u2 -- "requires Apple M5 Max, found: $chip"; return 1; }
  [[ "$mem" == 137438953472 ]] || { print -u2 -- "requires 128 GiB, found: $mem"; return 1; }
}

prepare() {
  load_paths || return 1
  for tool_name in git make python3 shasum tar system_profiler sysctl vm_stat memory_pressure stat readlink; do
    require_command "$tool_name" || return 1
  done
  [[ -f "$ANALYSER_PATH" ]] || { print -u2 -- "missing analyser: $ANALYSER_PATH"; return 1; }
  RUNBOOK_SHA=$(git -C "$RUNBOOK_REPO" rev-parse HEAD) || return 1
  local runbook_status
  runbook_status=$(git -C "$RUNBOOK_REPO" status --porcelain=v1) || return 1
  if [[ -n "$runbook_status" ]]; then
    print -u2 -- 'runbook checkout is dirty; commit or discard its changes before benchmarking'
    return 1
  fi
  if [[ -e "$CONTROL_REPO" || -e "$RESULTS" || -e "$TIMED_WT" || -e "$TRACE_WT" ]]; then
    print -u2 -- 'benchmark root is not fresh; choose a new DS4_BENCH_ROOT'
    return 1
  fi
  if [[ -d "$BENCH_ROOT" && -n "$(command ls -A "$BENCH_ROOT" 2>/dev/null)" ]]; then
    print -u2 -- 'DS4_BENCH_ROOT must be empty before prepare'
    return 1
  fi
  mkdir -p "$BENCH_ROOT" "$GGUF_DIR" "$RESULTS/metadata" "$ISOLATED_HOME" "$BENCH_ROOT/input"
  git clone --no-checkout https://github.com/michaelasper/ds4.git "$CONTROL_REPO" || return 1
  git -C "$CONTROL_REPO" fetch --no-tags origin '+refs/heads/*:refs/remotes/origin/*' || return 1
  for sha in "$RUNBOOK_SHA" "$TIMED_SHA" "$TRACE_SHA"; do
    git -C "$CONTROL_REPO" cat-file -e "$sha^{commit}" || return 1
  done
  git -C "$CONTROL_REPO" worktree add --detach "$TIMED_WT" "$TIMED_SHA" || return 1
  git -C "$CONTROL_REPO" worktree add --detach "$TRACE_WT" "$TRACE_SHA" || return 1
  if [[ ! -f "$MODEL_SOURCE" ]]; then
    ( cd "$TIMED_WT" && DS4_GGUF_DIR="$GGUF_DIR" ./download_model.sh laguna-q2-q3 ) || return 1
  fi
  [[ -f "$MODEL_SOURCE" ]] || { print -u2 -- "missing model: $MODEL_SOURCE"; return 1; }
  ln -s "$MODEL_SOURCE" "$MODEL_LINK" || return 1
  local model_stat_before model_stat_after
  model_stat_before=$(model_stat_value) || return 1
  MODEL_SHA=$(shasum -a 256 "$MODEL_LINK" | awk '{print $1}')
  model_stat_after=$(model_stat_value) || return 1
  [[ "$model_stat_before" == "$model_stat_after" ]] || {
    print -u2 -- 'model file changed while it was being hashed'
    return 1
  }
  MODEL_STAT=$model_stat_after
  [[ "$MODEL_SHA" == "$EXPECTED_MODEL_SHA" ]] || {
    print -u2 -- "wrong model SHA-256: $MODEL_SHA"
    return 1
  }
  local prompt_sha
  prompt_sha=$(shasum -a 256 "$TIMED_WT/speed-bench/promessi_sposi.txt" | awk '{print $1}')
  [[ "$prompt_sha" == "$EXPECTED_LONG_PROMPT_SHA" ]] || {
    print -u2 -- "wrong prompt SHA-256: $prompt_sha"
    return 1
  }
  print -r -- "$RUNBOOK_SHA" > "$RESULTS/metadata/runbook-revision.txt"
  print -r -- "$TIMED_SHA" > "$RESULTS/metadata/timed-revision.txt"
  print -r -- "$TRACE_SHA" > "$RESULTS/metadata/trace-revision.txt"
  print -r -- "$MODEL_SHA" > "$RESULTS/metadata/model-sha256.txt"
  print -r -- "$MODEL_STAT" > "$RESULTS/metadata/model-stat.txt"
  print -r -- "$prompt_sha" > "$RESULTS/metadata/prompt-sha256.txt"
  print -r -- "$DISCOVERY_RELEASE" > "$RESULTS/metadata/discovery-release.txt"
  print -r -- "$DISCOVERY_ARCHIVE_SHA" > "$RESULTS/metadata/discovery-archive-sha256.txt"
  print -r -- "$RANDOM_SEED" > "$RESULTS/metadata/random-seed.txt"
  print -r -- "$COOLDOWN_SECONDS" > "$RESULTS/metadata/cooldown-seconds.txt"
  verify_model_identity || return 1
  cp "$RUNBOOK_REPO/BENCHMARK.md" "$RESULTS/metadata/BENCHMARK.md"
  cp "$RUNNER_PATH" "$RESULTS/metadata/run_m5_finalists.zsh"
  cp "$ANALYSER_PATH" "$RESULTS/metadata/m5_finalist_analysis.py"
  ( cd "$RESULTS/metadata" && shasum -a 256 \
      BENCHMARK.md run_m5_finalists.zsh m5_finalist_analysis.py \
      > protocol-files.sha256 ) || return 1
  verify_protocol_snapshot || return 1
  verify_host_identity || return 1
  build_revision timed "$TIMED_WT" "$TIMED_SHA" || return 1
  build_revision trace "$TRACE_WT" "$TRACE_SHA" || return 1
  {
    print -r -- "cwd=$TIMED_WT"
    print -r -- "revision=$TIMED_SHA"
    print -r -- 'env -i HOME=<isolated> PATH=/usr/bin:/bin:/usr/sbin:/sbin LANG=C LC_ALL=C ./ds4_test --metal-kernels'
  } > "$RESULTS/metadata/timed-tests.command.txt"
  ( cd "$TIMED_WT" && env -i HOME="$ISOLATED_HOME" \
      PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
      ./ds4_test --metal-kernels ) > "$RESULTS/metadata/timed-tests.stdout.log" \
      2> "$RESULTS/metadata/timed-tests.stderr.log" || return 1
  print -- PASS > "$RESULTS/metadata/timed-tests.status.txt"
  {
    print -r -- "cwd=$TRACE_WT"
    print -r -- "revision=$TRACE_SHA"
    print -r -- 'env -i HOME=<isolated> PATH=/usr/bin:/bin:/usr/sbin:/sbin LANG=C LC_ALL=C ./ds4_test --laguna-selector-parser --laguna-bench-trace-transaction'
  } > "$RESULTS/metadata/trace-tests.command.txt"
  ( cd "$TRACE_WT" && env -i HOME="$ISOLATED_HOME" \
      PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
      ./ds4_test --laguna-selector-parser --laguna-bench-trace-transaction ) \
      > "$RESULTS/metadata/trace-tests.stdout.log" \
      2> "$RESULTS/metadata/trace-tests.stderr.log" || return 1
  print -- PASS > "$RESULTS/metadata/trace-tests.status.txt"
  git -C "$TIMED_WT" status --porcelain=v1 > "$RESULTS/metadata/timed-status-after-tests.txt" || return 1
  git -C "$TRACE_WT" status --porcelain=v1 > "$RESULTS/metadata/trace-status-after-tests.txt" || return 1
  [[ ! -s "$RESULTS/metadata/timed-status-after-tests.txt" && \
     ! -s "$RESULTS/metadata/trace-status-after-tests.txt" ]] || {
    print -u2 -- 'a fixed worktree became dirty during build/tests'
    return 1
  }
  shasum -a 256 "$TIMED_WT/ds4-bench" | awk '{print $1}' \
    > "$RESULTS/metadata/timed-binary-sha256.txt" || return 1
  shasum -a 256 "$TRACE_WT/ds4-bench" | awk '{print $1}' \
    > "$RESULTS/metadata/trace-binary-sha256.txt" || return 1
  verify_binary_identity "$TIMED_WT" || return 1
  verify_binary_identity "$TRACE_WT" || return 1
  print -- PASS > "$RESULTS/metadata/prepare-status.txt"
  print -- "prepared $BENCH_ROOT"
}

run_route_probe() {
  local arm=$1
  local out="$RESULTS/route-proof/$arm"
  set_selector_env "$arm" || return 1
  refuse_directory "$out" || return 1
  write_identity "$out" "$TRACE_WT" "$arm" || return 1
  {
    print -r -- "cwd=$TRACE_WT"
    print -r -- "binary=$TRACE_WT/ds4-bench"
    printf 'env=%q\n' DS4_LAGUNA_BENCH_TRACE=1 "${R1_ENV[@]}" "${SELECTOR_ENV[@]}"
    printf 'arg=%q\n' "$TRACE_WT/ds4-bench" "${TRACE_D8_ARGS[@]}" --csv "$out/metrics.csv"
  } > "$out/command.txt"
  capture_state "$out/state-before"
  ( cd "$TRACE_WT" && env -i HOME="$ISOLATED_HOME" \
      PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
      DS4_LAGUNA_BENCH_TRACE=1 "${R1_ENV[@]}" "${SELECTOR_ENV[@]}" \
      ./ds4-bench "${TRACE_D8_ARGS[@]}" --csv "$out/metrics.csv" \
      > "$out/stdout.log" 2> "$out/stderr.log" )
  local rc=$?
  verify_runtime_identity "$TRACE_WT" || return 1
  capture_state "$out/state-after"
  git -C "$TRACE_WT" status --porcelain=v1 > "$out/worktree-status.txt" || rc=69
  if (( rc == 0 )) && [[ -s "$out/worktree-status.txt" ]]; then rc=69; fi
  if (( rc == 0 )); then
    python3 - "$arm" "$out/stderr.log" "$out/metrics.csv" <<'PY' || rc=66
import csv, math, re, sys
arm, source, metrics = sys.argv[1:]
with open(metrics, newline="") as handle:
    metric_rows = list(csv.DictReader(handle))
if len(metric_rows) != 1:
    raise SystemExit(f"expected one route metric row, found {len(metric_rows)}")
metric = metric_rows[0]
expected_metrics = {"ctx_tokens": 8192, "prefill_tokens": 8192, "gen_tokens": 1}
for key, expected_value in expected_metrics.items():
    value = float(metric.get(key, "nan"))
    if not math.isfinite(value) or value != expected_value:
        raise SystemExit(f"wrong route metric {key}={value}")
lines = open(source, errors="replace").read().splitlines()
records = []
for line in lines:
    if "ds4-laguna-bench-trace " in line:
        records.append(dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", line)))
if not records or any(row.get("status") != "ok" for row in records):
    raise SystemExit("missing or non-ok completion-scoped trace record")
matches = [row for row in records if row.get("graph") == "decode"]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one decode trace, found {len(matches)}")
row = matches[0]
if row.get("status") != "ok" or row.get("completion") != "waited" or row.get("n_tokens") != "1":
    raise SystemExit("wrong completion-scoped trace identity")
if "gqa3" in arm:
    expected = {"ordinary": 0, "gqa3": 36, "gqa9": 0, "global_grouped": 12}
    if any(int(row.get(key, -1)) != value for key, value in expected.items()):
        raise SystemExit("wrong GQA3/global route counts")
else:
    expected = {"ordinary": 36, "gqa3": 0, "gqa9": 0, "global_grouped": 12}
    if any(int(row.get(key, -1)) != value for key, value in expected.items()):
        raise SystemExit("wrong ordinary/global route counts")
text = "\n".join(lines)
safe_zero_counter = re.compile(
    r"\b(?:fallback|unavailable|unsupported|paused|failed|failures|invalid|"
    r"mismatch|refusal|errors|faults|rejected)=0\b",
    re.I,
)
for number, line in enumerate(lines, 1):
    diagnostic = safe_zero_counter.sub("", line)
    if re.search(
        r"fallback|unavailable|unsupported|paused|fail(?:ed|ure)?|invalid|"
        r"mismatch|refusal|error|fault|reject(?:ed|ion)?",
        diagnostic,
        re.I,
    ):
        raise SystemExit(f"forbidden diagnostic at line {number}: {line}")
if "canonical-ladder" in arm and not re.search(
    r"layers=7,15,23,31,39,47; flushes=6; completion=waited", text
):
    raise SystemExit("wrong canonical ladder diagnostic")
if "front-rung-ladder" in arm and not re.search(
    r"layers=1,7,15,23,31,39,47; flushes=7; completion=waited", text
):
    raise SystemExit("wrong front-rung ladder diagnostic")
PY
  fi
  if (( rc == 0 )); then
    print -- PASS > "$out/status.txt"
  else
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- "route probe exit=$rc" > "$out/reject-reason.txt"
  fi
}

extract_generated_output() {
  local stderr_file=$1
  local output_file=$2
  python3 - "$stderr_file" "$output_file" <<'PY'
import sys
source, destination = sys.argv[1:]
data = open(source, "rb").read()
marker = b'ds4-bench: gen[ctx=8192] decoded text: "'
if data.count(marker) != 1:
    raise SystemExit("expected exactly one decoded-text marker")
payload = data.split(marker, 1)[1]
if not payload.endswith(b'"\n'):
    raise SystemExit("decoded text was not the final stderr record")
decoded = payload[:-2]
if not decoded:
    raise SystemExit("decoded output is empty")
open(destination, "wb").write(decoded)
PY
}

run_parity_probe() {
  local arm=$1
  local out="$RESULTS/parity/$arm"
  set_selector_env "$arm" || return 1
  refuse_directory "$out" || return 1
  mkdir -p "$out/decode-logits" "$out/prefill-logits"
  write_identity "$out" "$TIMED_WT" "$arm" || return 1
  {
    print -r -- "cwd=$TIMED_WT"
    print -r -- "binary=$TIMED_WT/ds4-bench"
    printf 'env=%q\n' "${R1_ENV[@]}" "${SELECTOR_ENV[@]}"
    print -r -- 'probe=decode frontier 8192 to 8193, P16 logits, and full 128-token generated bytes'
  } > "$out/command.txt"
  capture_state "$out/state-before"
  local rc=0 one_rc=0
  ( cd "$TIMED_WT" && env -i HOME="$ISOLATED_HOME" \
      PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
      "${R1_ENV[@]}" "${SELECTOR_ENV[@]}" \
      ./ds4-bench "${PARITY_DECODE_ARGS[@]}" \
      --dump-frontier-logits-f32-dir "$out/decode-logits" --csv "$out/decode.csv" \
      > "$out/decode.stdout.log" 2> "$out/decode.stderr.log" ) || rc=$?
  verify_runtime_identity "$TIMED_WT" || return 1
  one_rc=0
  ( cd "$TIMED_WT" && env -i HOME="$ISOLATED_HOME" \
      PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
      "${R1_ENV[@]}" "${SELECTOR_ENV[@]}" \
      ./ds4-bench "${PARITY_PREFILL_ARGS[@]}" \
      --dump-frontier-logits-f32-dir "$out/prefill-logits" --csv "$out/prefill.csv" \
      > "$out/prefill.stdout.log" 2> "$out/prefill.stderr.log" ) || one_rc=$?
  verify_runtime_identity "$TIMED_WT" || return 1
  (( rc == 0 && one_rc != 0 )) && rc=$one_rc
  one_rc=0
  ( cd "$TIMED_WT" && env -i HOME="$ISOLATED_HOME" \
      PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" LANG=C LC_ALL=C \
      "${R1_ENV[@]}" "${SELECTOR_ENV[@]}" \
      ./ds4-bench "${D8_ARGS[@]}" --show-output --csv "$out/generated.csv" \
      > "$out/generated.stdout.log" 2> "$out/generated.stderr.log" ) || one_rc=$?
  verify_runtime_identity "$TIMED_WT" || return 1
  (( rc == 0 && one_rc != 0 )) && rc=$one_rc
  extract_generated_output "$out/generated.stderr.log" "$out/generated.bin" || rc=67
  capture_state "$out/state-after"
  git -C "$TIMED_WT" status --porcelain=v1 > "$out/worktree-status.txt" || rc=69
  if (( rc == 0 )) && [[ -s "$out/worktree-status.txt" ]]; then rc=69; fi
  local decode_file="$out/decode-logits/frontier_008193.logits.f32"
  local prefill_file="$out/prefill-logits/frontier_016384.logits.f32"
  [[ -s "$decode_file" && -s "$prefill_file" && -s "$out/generated.bin" ]] || rc=67
  validate_diagnostics "$out" || rc=68
  ( cd "$out" && shasum -a 256 \
      decode.csv prefill.csv generated.csv generated.bin \
      decode-logits/frontier_008193.logits.f32 \
      prefill-logits/frontier_016384.logits.f32 > artifacts.sha256 ) || rc=67
  if (( rc == 0 )); then
    print -- PASS > "$out/execution-status.txt"
  else
    print -- SEMANTIC_REJECT > "$out/execution-status.txt"
    print -- "parity execution exit=$rc" > "$out/reject-reason.txt"
  fi
}

compare_parity() {
  local arm=$1
  local out="$RESULTS/parity/$arm"
  local baseline="$RESULTS/parity/00-r1-baseline"
  if [[ "$(<"$out/execution-status.txt")" != PASS || "$(<"$baseline/execution-status.txt")" != PASS ]]; then
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- 'baseline or candidate parity execution failed' >> "$out/reject-reason.txt"
    return 0
  fi
  local mode=exact
  has_gqa3 "$arm" && mode=gqa3
  python3 - "$mode" "$baseline" "$out" <<'PY'
import array, csv, math, os, sys
mode, baseline, candidate = sys.argv[1:]

def same(name):
    return open(os.path.join(baseline, name), "rb").read() == open(os.path.join(candidate, name), "rb").read()

generated_equal = same("generated.bin")
prefill_equal = same("prefill-logits/frontier_016384.logits.f32")
decode_equal = same("decode-logits/frontier_008193.logits.f32")
same_argmax = False
max_abs = 0.0
rms = 0.0
numeric_ok = decode_equal
if mode == "gqa3":
    left, right = array.array("f"), array.array("f")
    left.frombytes(open(os.path.join(baseline, "decode-logits/frontier_008193.logits.f32"), "rb").read())
    right.frombytes(open(os.path.join(candidate, "decode-logits/frontier_008193.logits.f32"), "rb").read())
    if len(left) != len(right) or not left:
        numeric_ok = False
    elif not all(math.isfinite(value) for value in left) or not all(math.isfinite(value) for value in right):
        numeric_ok = False
    else:
        left_argmax = max(range(len(left)), key=left.__getitem__)
        right_argmax = max(range(len(right)), key=right.__getitem__)
        same_argmax = left_argmax == right_argmax
        differences = [abs(a - b) for a, b in zip(left, right)]
        max_abs = max(differences)
        rms = math.sqrt(sum(value * value for value in differences) / len(differences))
        numeric_ok = same_argmax and max_abs <= 5e-4 and rms <= 1e-4
passed = generated_equal and prefill_equal and numeric_ok
with open(os.path.join(candidate, "comparison.csv"), "w", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["generated_equal", "prefill_equal", "decode_bit_equal", "same_finite_argmax", "max_abs", "rms", "status"])
    writer.writerow([generated_equal, prefill_equal, decode_equal, same_argmax, max_abs, rms, "PASS" if passed else "REJECT"])
raise SystemExit(0 if passed else 1)
PY
  if (( $? == 0 )); then
    print -- PASS > "$out/status.txt"
  else
    print -- SEMANTIC_REJECT > "$out/status.txt"
    print -- 'candidate parity comparison failed' >> "$out/reject-reason.txt"
  fi
}

prove() {
  load_paths || return 1
  [[ -f "$RESULTS/metadata/prepare-status.txt" && \
     "$(<"$RESULTS/metadata/prepare-status.txt")" == PASS ]] || {
    print -u2 -- 'prepare phase is incomplete; use a fresh benchmark root'
    return 1
  }
  verify_protocol_snapshot || return 1
  MODEL_SHA=$(<"$RESULTS/metadata/model-sha256.txt")
  run_parity_probe 00-r1-baseline || return 1
  cp "$RESULTS/parity/00-r1-baseline/execution-status.txt" \
    "$RESULTS/parity/00-r1-baseline/status.txt"
  local arm
  for arm in "${CANDIDATE_ARMS[@]}"; do
    run_route_probe "$arm" || return 1
    run_parity_probe "$arm" || return 1
    compare_parity "$arm"
  done
  print -- COMPLETE > "$RESULTS/metadata/prove-phase.txt"
}

proof_passes() {
  local arm=$1
  local parity_status="$RESULTS/parity/$arm/status.txt"
  local route_status="$RESULTS/route-proof/$arm/status.txt"
  [[ -f "$parity_status" && -f "$route_status" ]] || return 1
  [[ "$(<"$parity_status")" == PASS && "$(<"$route_status")" == PASS ]]
}

phase_complete() {
  local phase_file="$RESULTS/metadata/$1-phase.txt"
  [[ -f "$phase_file" && "$(<"$phase_file")" == COMPLETE ]]
}

balanced_order() {
  local block=$1
  shift
  (( $# >= 2 )) || { print -u2 -- 'balanced_order needs baseline plus at least one candidate'; return 1; }
  python3 - "$RANDOM_SEED" "$block" "$@" <<'PY'
import random, sys
seed, block, *arms = sys.argv[1:]
pattern = [0]
low, high = 1, len(arms) - 1
while len(pattern) < len(arms):
    pattern.append(low)
    low += 1
    if len(pattern) < len(arms):
        pattern.append(high)
        high -= 1
orders = []
for shift in range(5):
    order = [arms[(index + shift) % len(arms)] for index in pattern]
    orders.extend((order, list(reversed(order))))
random.Random(int(seed)).shuffle(orders)
index = int(block) - 1
if not 0 <= index < len(orders):
    raise SystemExit("block must be in 1..10")
print("\n".join(orders[index]))
PY
}

mark_not_eligible() {
  local arm=$1
  local reason=$2
  local out="$RESULTS/not-eligible/$arm"
  mkdir -p "$out"
  print -- NOT_ELIGIBLE > "$out/status.txt"
  print -r -- "$reason" >> "$out/reason.txt"
}

measure() {
  load_paths || return 1
  phase_complete prove || {
    print -u2 -- 'prove phase is incomplete; use a fresh benchmark root'
    return 1
  }
  verify_protocol_snapshot || return 1
  MODEL_SHA=$(<"$RESULTS/metadata/model-sha256.txt")
  local arm
  local -a proof_eligible warm_arms warm_order active_candidates order
  proof_eligible=()
  for arm in "${CANDIDATE_ARMS[@]}"; do
    if proof_passes "$arm"; then
      proof_eligible+=("$arm")
    else
      mark_not_eligible "$arm" 'parity or route proof did not pass; timed confirmation skipped'
    fi
  done
  refuse_directory "$RESULTS/warmup" || return 1
  warm_arms=(00-r1-baseline "${proof_eligible[@]}")
  print -rl -- "${warm_arms[@]}" > "$RESULTS/warmup/arms.txt"
  if (( ${#warm_arms[@]} > 1 )); then
    warm_order=("${(@f)$(balanced_order 1 "${warm_arms[@]}")}") || return 1
  else
    warm_order=(00-r1-baseline)
  fi
  print -rl -- "${warm_order[@]}" > "$RESULTS/warmup/order.txt"
  for arm in "${warm_order[@]}"; do
    run_d8_arm "warmup/$arm" "$arm" || return 1
  done
  active_candidates=()
  if [[ "$(<"$RESULTS/warmup/00-r1-baseline/status.txt")" == PASS ]]; then
    for arm in "${proof_eligible[@]}"; do
      if [[ "$(<"$RESULTS/warmup/$arm/status.txt")" == PASS ]]; then
        active_candidates+=("$arm")
      else
        mark_not_eligible "$arm" 'warm-up was a semantic reject; timed confirmation skipped'
      fi
    done
  else
    for arm in "${proof_eligible[@]}"; do
      mark_not_eligible "$arm" 'R1 baseline warm-up was a semantic reject; timed confirmation skipped'
    done
  fi
  ACTIVE_ARMS=(00-r1-baseline "${active_candidates[@]}")
  refuse_directory "$RESULTS/confirmation" || return 1
  print -rl -- "${ACTIVE_ARMS[@]}" > "$RESULTS/confirmation/active-arms.txt"
  print -- 'block,position,arm' > "$RESULTS/confirmation/balanced-order.csv"
  if (( ${#active_candidates[@]} == 0 )); then
    print -- COMPLETE > "$RESULTS/metadata/measure-phase.txt"
    print -u2 -- 'no candidate was eligible after proofs and warm-ups; packaging a no-promotion result'
    return 0
  fi
  local block position
  for block in {1..10}; do
    cool_down
    order=("${(@f)$(balanced_order "$block" "${ACTIVE_ARMS[@]}")}") || return 1
    position=0
    for arm in "${order[@]}"; do
      (( position++ ))
      print -- "$block,$position,$arm" >> "$RESULTS/confirmation/balanced-order.csv"
      run_d8_arm "confirmation/block-${(l:2::0:)block}/$arm" "$arm" || return 1
    done
  done
  print -- COMPLETE > "$RESULTS/metadata/measure-phase.txt"
}

analyse() {
  load_paths || return 1
  phase_complete measure || {
    print -u2 -- 'measure phase is incomplete; use a fresh benchmark root'
    return 1
  }
  verify_protocol_snapshot || return 1
  python3 "$RESULTS/metadata/m5_finalist_analysis.py" "$RESULTS" || return 1
  print -- COMPLETE > "$RESULTS/metadata/analyse-phase.txt"
}

validate_inventory() {
  local root=$1
  python3 - "$root" <<'PY'
import sys
import hashlib
from pathlib import Path

root = Path(sys.argv[1])
baseline = "00-r1-baseline"
candidates = [
    "05-canonical-ladder", "12-gqa3", "14-front-rung-ladder",
    "15-canonical-ladder-gqa3", "16-front-rung-ladder-gqa3",
]
failures = []

def require(path, nonempty=True):
    if not path.is_file():
        failures.append(f"missing:{path.relative_to(root)}")
    elif nonempty and path.stat().st_size == 0:
        failures.append(f"empty:{path.relative_to(root)}")

def require_text(path, expected):
    require(path)
    if path.is_file() and path.read_text(errors="replace").strip() != expected:
        failures.append(f"wrong:{path.relative_to(root)}")

def require_state(directory):
    for name, nonempty in (
        ("time.txt", True), ("host-profiler-sanitised.json", True),
        ("power-display-profiler-sanitised.json", True),
        ("pmset-batt.txt", False), ("pmset-custom.txt", False),
        ("pmset-therm.txt", False), ("sysctl-host.txt", True),
        ("vm-swapusage.txt", True), ("vm-stat.txt", True),
        ("memory-pressure.txt", True), ("df.txt", True),
        ("gating-policy.txt", True),
    ):
        require(directory / name, nonempty)

def verify_sha_ledger(directory, ledger_name):
    ledger = directory / ledger_name
    require(ledger)
    if not ledger.is_file():
        return
    for line in ledger.read_text(errors="replace").splitlines():
        try:
            digest, name = line.split("  ", 1)
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe path")
            target = directory / relative
            computed = hashlib.sha256(target.read_bytes()).hexdigest()
            if computed != digest:
                failures.append(f"hash-mismatch:{target.relative_to(root)}")
        except (OSError, ValueError) as error:
            failures.append(f"bad-ledger:{ledger.relative_to(root)}:{error}")

def status(directory):
    path = directory / "status.txt"
    require(path)
    return path.read_text(errors="replace").strip() if path.is_file() else "MISSING"

def plain_status(directory):
    try:
        return (directory / "status.txt").read_text(errors="replace").strip()
    except OSError:
        return "MISSING"

def require_timed_arm(directory):
    value = status(directory)
    for name, nonempty in (
        ("command.txt", True), ("revision.txt", True), ("model-sha256.txt", True),
        ("d8.exit-code.txt", True), ("d8.csv", value == "PASS"),
        ("d8.stdout.log", False), ("d8.stderr.log", False),
        ("d8.max-rss.txt", True), ("d8.csv.sha256", True),
        ("unexpected-diagnostics.txt", False),
        ("memory-gate.txt", True), ("worktree-status.txt", False),
        ("arm.txt", True), ("env-overrides.txt", True),
    ):
        require(directory / name, nonempty)
    require_state(directory / "state-before")
    require_state(directory / "state-after")
    verify_sha_ledger(directory, "d8.csv.sha256")
    if value == "PASS":
        try:
            memory = (directory / "memory-gate.txt").read_text(errors="replace")
            if "status=PASS" not in memory:
                failures.append(f"memory-not-pass:{directory.relative_to(root)}")
        except OSError:
            pass
    elif value == "SEMANTIC_REJECT":
        require(directory / "reject-reason.txt")
    else:
        failures.append(f"invalid-status:{directory.relative_to(root)}={value}")

metadata = root / "metadata"
require_text(metadata / "prepare-status.txt", "PASS")
for marker in ("prove-phase.txt", "measure-phase.txt", "analyse-phase.txt"):
    require_text(metadata / marker, "COMPLETE")
for name in (
    "BENCHMARK.md", "run_m5_finalists.zsh", "m5_finalist_analysis.py",
    "protocol-files.sha256", "runbook-revision.txt", "timed-revision.txt",
    "trace-revision.txt", "timed-binary-sha256.txt", "trace-binary-sha256.txt",
    "model-sha256.txt", "model-stat.txt", "prompt-sha256.txt",
    "discovery-release.txt", "discovery-archive-sha256.txt",
    "random-seed.txt", "cooldown-seconds.txt", "analysis-recheck.txt",
):
    require(metadata / name)
for label in ("timed", "trace"):
    build = metadata / f"build-{label}"
    require_text(build / "status.txt", "PASS")
    for name, nonempty in (
        ("revision.txt", True), ("command.txt", True), ("stdout.log", False),
        ("stderr.log", False), ("status-before.txt", False),
        ("status-after.txt", False),
    ):
        require(build / name, nonempty)
    require_text(metadata / f"{label}-tests.status.txt", "PASS")
    require(metadata / f"{label}-tests.command.txt")
    require(metadata / f"{label}-tests.stdout.log", False)
    require(metadata / f"{label}-tests.stderr.log", False)
    require(metadata / f"{label}-status-after-tests.txt", False)
host = metadata / "host"
require(host / "chip.txt")
require(host / "memory-bytes.txt")
require(host / "power-policy.txt")
require_state(host / "state")
require_state(metadata / "final-state")
for arm in [baseline, *candidates]:
    parity = root / "parity" / arm
    value = status(parity)
    for name, nonempty in (
        ("execution-status.txt", True), ("command.txt", True),
        ("revision.txt", True), ("model-sha256.txt", True),
        ("arm.txt", True), ("env-overrides.txt", True),
        ("decode.stdout.log", False), ("prefill.stdout.log", False),
        ("generated.stdout.log", False),
        ("decode.stderr.log", False), ("prefill.stderr.log", False),
        ("generated.stderr.log", False),
    ):
        require(parity / name, nonempty)
    if value == "PASS":
        require(parity / "generated.bin")
        require(parity / "artifacts.sha256")
        if arm != baseline:
            require(parity / "comparison.csv")
        verify_sha_ledger(parity, "artifacts.sha256")
    else:
        require(parity / "reject-reason.txt")
    require_state(parity / "state-before")
    require_state(parity / "state-after")
for arm in candidates:
    route = root / "route-proof" / arm
    value = status(route)
    for name, nonempty in (
        ("command.txt", True), ("revision.txt", True), ("model-sha256.txt", True),
        ("arm.txt", True), ("env-overrides.txt", True),
        ("stdout.log", False), ("stderr.log", False), ("worktree-status.txt", False),
    ):
        require(route / name, nonempty)
    if value == "PASS":
        require(route / "metrics.csv")
        require(route / "stderr.log")
    else:
        require(route / "reject-reason.txt")
    require_state(route / "state-before")
    require_state(route / "state-after")

warmup = root / "warmup"
require(warmup / "arms.txt")
require(warmup / "order.txt")
warm_arms = [line for line in (warmup / "arms.txt").read_text().splitlines() if line] if (warmup / "arms.txt").is_file() else []
if not warm_arms or warm_arms[0] != baseline or not set(warm_arms[1:]).issubset(candidates):
    failures.append(f"invalid-warm-arms:{warm_arms}")
expected_warm_arms = [
    baseline,
    *(
        arm for arm in candidates
        if plain_status(root / "parity" / arm) == "PASS"
        and plain_status(root / "route-proof" / arm) == "PASS"
    ),
]
if warm_arms != expected_warm_arms:
    failures.append(f"warm-arms:{warm_arms} expected={expected_warm_arms}")
actual_warm_dirs = {path.name for path in warmup.iterdir() if path.is_dir()}
if actual_warm_dirs != set(warm_arms):
    failures.append(f"warm-dir-set:{sorted(actual_warm_dirs)} expected={sorted(warm_arms)}")
for arm in warm_arms:
    require_timed_arm(warmup / arm)

confirmation = root / "confirmation"
require(confirmation / "active-arms.txt")
require(confirmation / "balanced-order.csv")
active_arms = [line for line in (confirmation / "active-arms.txt").read_text().splitlines() if line] if (confirmation / "active-arms.txt").is_file() else []
if not active_arms or active_arms[0] != baseline or not set(active_arms[1:]).issubset(candidates):
    failures.append(f"invalid-active-arms:{active_arms}")
expected_active_arms = [
    baseline,
    *(arm for arm in candidates if plain_status(root / "not-eligible" / arm) != "NOT_ELIGIBLE"),
]
if active_arms != expected_active_arms:
    failures.append(f"active-arms:{active_arms} expected={expected_active_arms}")
for arm in set(candidates) - set(active_arms):
    require_text(root / "not-eligible" / arm / "status.txt", "NOT_ELIGIBLE")
    require(root / "not-eligible" / arm / "reason.txt")
expected_blocks = {f"block-{block:02d}" for block in range(1, 11)} if len(active_arms) > 1 else set()
actual_blocks = {path.name for path in confirmation.glob("block-*") if path.is_dir()}
if actual_blocks != expected_blocks:
    failures.append(f"confirmation block set={sorted(actual_blocks)}, expected={sorted(expected_blocks)}")
for block in expected_blocks:
    block_dir = confirmation / block
    actual_arm_dirs = {path.name for path in block_dir.iterdir() if path.is_dir()}
    if actual_arm_dirs != set(active_arms):
        failures.append(f"{block}: arm dirs={sorted(actual_arm_dirs)}, expected={sorted(active_arms)}")
    for arm in active_arms:
        require_timed_arm(block_dir / arm)
for name in ("SUMMARY.md", "RELEASE.md", "analysis/block-effects.csv", "analysis/decisions.csv", "analysis/component-comparisons.csv"):
    require(root / name)
    verify_sha_ledger((root / name).parent, (root / name).name + ".sha256")
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
print("PASS")
PY
}

privacy_scan() {
  local root=$1
  python3 - "$root" <<'PY'
import os, re, sys
root = os.path.realpath(sys.argv[1])
patterns = {
    "user_path": re.compile(rb"/" rb"Users/[^/\s]+"),
    "email": re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "mac": re.compile(rb"(?i)(?<![0-9a-f])(?:[0-9a-f]{2}:){5}[0-9a-f]{2}(?![0-9a-f])"),
    "uuid": re.compile(rb"(?i)(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])"),
}
failures = []
for base, _, names in os.walk(root):
    for name in names:
        if name.endswith(".f32"):
            continue
        path = os.path.join(base, name)
        try:
            data = open(path, "rb").read()
        except OSError as error:
            failures.append(f"unreadable:{os.path.relpath(path, root)}:{error}")
            continue
        for label, pattern in patterns.items():
            if pattern.search(data):
                failures.append(f"{label}:{os.path.relpath(path, root)}")
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
print("PASS")
PY
}

make_manifest() {
  local root=$1
  python3 - "$root" <<'PY'
import hashlib, os, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
manifest = root / "MANIFEST.sha256"
entries = []
for path in sorted(root.rglob("*")):
    if path == manifest:
        continue
    if path.is_symlink():
        raise SystemExit(f"symlink is not allowed in results: {path.relative_to(root)}")
    if path.is_dir():
        continue
    if not path.is_file():
        raise SystemExit(f"non-regular result member: {path.relative_to(root)}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    entries.append(f"{digest.hexdigest()}  {path.relative_to(root).as_posix()}")
manifest.write_text("\n".join(entries) + "\n")
PY
}

verify_manifest() {
  local root=$1
  python3 - "$root" <<'PY'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
manifest = root / "MANIFEST.sha256"
expected = {}
for line in manifest.read_text().splitlines():
    digest, name = line.split("  ", 1)
    if name in expected:
        raise SystemExit(f"duplicate manifest member: {name}")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"unsafe manifest member: {name}")
    expected[name] = digest
actual = set()
for path in root.rglob("*"):
    if path == manifest:
        continue
    if path.is_symlink():
        raise SystemExit(f"symlink is not allowed: {path.relative_to(root)}")
    if path.is_dir():
        continue
    if not path.is_file():
        raise SystemExit(f"non-regular member: {path.relative_to(root)}")
    actual.add(path.relative_to(root).as_posix())
if actual != set(expected):
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    raise SystemExit(f"manifest file-set mismatch missing={missing} extra={extra}")
for name, digest in expected.items():
    path = root / name
    computed = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            computed.update(chunk)
    if computed.hexdigest() != digest:
        raise SystemExit(f"manifest mismatch: {name}")
print("PASS")
PY
}

verify_archive_headers() {
  local archive=$1
  python3 - "$archive" <<'PY'
import sys, tarfile
from pathlib import PurePosixPath

archive = sys.argv[1]
with tarfile.open(archive, "r:gz") as handle:
    members = handle.getmembers()
if not members:
    raise SystemExit("archive has no members")
for member in members:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "results":
        raise SystemExit(f"unsafe archive member: {member.name}")
    if not (member.isfile() or member.isdir()):
        raise SystemExit(f"non-regular archive member: {member.name}")
    if member.uid != 0 or member.gid != 0 or member.uname != "root" or member.gname != "wheel":
        raise SystemExit(
            f"non-normalised archive ownership: {member.name} "
            f"{member.uid}:{member.gid} {member.uname}:{member.gname}"
        )
    if any("xattr" in key.lower() or "acl" in key.lower() for key in member.pax_headers):
        raise SystemExit(f"extended metadata in archive: {member.name}")
print("PASS")
PY
}

verify_analysis_reproducible() {
  local expected_dir rc=0
  expected_dir=$(mktemp -d /private/tmp/ds4-analysis-verify.XXXXXX) || return 1
  mkdir -p "$expected_dir/analysis"
  local relative
  for relative in \
      SUMMARY.md SUMMARY.md.sha256 RELEASE.md RELEASE.md.sha256 \
      analysis/block-effects.csv analysis/block-effects.csv.sha256 \
      analysis/decisions.csv analysis/decisions.csv.sha256 \
      analysis/component-comparisons.csv analysis/component-comparisons.csv.sha256; do
    cp "$RESULTS/$relative" "$expected_dir/$relative" || rc=1
  done
  if (( rc == 0 )); then
    python3 "$RESULTS/metadata/m5_finalist_analysis.py" "$RESULTS" \
      > "$BENCH_ROOT/analysis-recheck.stdout.log" \
      2> "$BENCH_ROOT/analysis-recheck.stderr.log" || rc=1
  fi
  if (( rc == 0 )); then
    for relative in \
        SUMMARY.md SUMMARY.md.sha256 RELEASE.md RELEASE.md.sha256 \
        analysis/block-effects.csv analysis/block-effects.csv.sha256 \
        analysis/decisions.csv analysis/decisions.csv.sha256 \
        analysis/component-comparisons.csv analysis/component-comparisons.csv.sha256; do
      cmp "$RESULTS/$relative" "$expected_dir/$relative" >/dev/null || rc=1
    done
  fi
  /bin/rm -rf -- "$expected_dir"
  if (( rc == 0 )); then
    print -- PASS > "$RESULTS/metadata/analysis-recheck.txt"
  fi
  return "$rc"
}

validate_statuses() {
  local root=$1
  python3 - "$root" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
allowed = {"PASS", "SEMANTIC_REJECT", "NOT_ELIGIBLE"}
paths = list(root.rglob("status.txt"))
if not paths:
    raise SystemExit("no status files")
bad = []
for path in paths:
    value = path.read_text().strip()
    if value not in allowed:
        bad.append(f"{path.relative_to(root)}={value!r}")
if bad:
    raise SystemExit("\n".join(bad))
print(f"PASS statuses={len(paths)}")
PY
}

package_results() {
  load_paths || return 1
  verify_protocol_snapshot || return 1
  local archive="$BENCH_ROOT/laguna-m5-max-finalists-results.tar.gz"
  local sidecar="$archive.sha256"
  [[ ! -e "$archive" && ! -e "$sidecar" && ! -e "$BENCH_ROOT/package-complete.txt" ]] || {
    print -u2 -- 'package output already exists; use a fresh benchmark root rather than overwriting it'
    return 1
  }
  [[ -s "$RESULTS/SUMMARY.md" && -s "$RESULTS/RELEASE.md" && -s "$RESULTS/analysis/decisions.csv" ]] || {
    print -u2 -- 'run analyse first; summary/release/decisions are missing'
    return 1
  }
  phase_complete analyse || {
    print -u2 -- 'analyse phase is incomplete; refusing to package'
    return 1
  }
  verify_analysis_reproducible || {
    print -u2 -- 'analysis is not reproducible from the archived analyser and raw evidence'
    return 1
  }
  capture_state "$RESULTS/metadata/final-state"
  validate_inventory "$RESULTS" > "$RESULTS/metadata/inventory-validation.txt" || return 1
  validate_statuses "$RESULTS" > "$RESULTS/metadata/status-validation.txt" || return 1
  print -- 'MANIFEST.sha256 must match the exact regular-file set; symlinks and extra files are rejected.' \
    > "$RESULTS/metadata/manifest-policy.txt"
  privacy_scan "$RESULTS" > "$RESULTS/metadata/privacy-validation.txt" || return 1
  make_manifest "$RESULTS" || return 1
  verify_manifest "$RESULTS" > "$BENCH_ROOT/manifest-verification.txt" || return 1
  COPYFILE_DISABLE=1 tar --no-xattrs --no-acls --no-fflags \
    --uid 0 --gid 0 --uname root --gname wheel \
    -C "$BENCH_ROOT" -czf "$archive" results || return 1
  verify_archive_headers "$archive" > "$BENCH_ROOT/archive-header-verification.txt" || return 1
  ( cd "$BENCH_ROOT" && shasum -a 256 "${archive:t}" > "${sidecar:t}" ) || return 1
  local verify_dir
  verify_dir=$(mktemp -d "$BENCH_ROOT/archive-verify.XXXXXX") || return 1
  tar -C "$verify_dir" -xzf "$archive" || return 1
  verify_manifest "$verify_dir/results" > "$BENCH_ROOT/archive-verification.txt" || return 1
  local archive_digest manifest_digest
  archive_digest=$(awk '{print $1}' "$sidecar") || return 1
  manifest_digest=$(shasum -a 256 "$RESULTS/MANIFEST.sha256" | awk '{print $1}') || return 1
  {
    print -- 'status=COMPLETE'
    print -r -- "archive_sha256=$archive_digest"
    print -r -- "manifest_sha256=$manifest_digest"
  } > "$BENCH_ROOT/package-complete.txt"
  print -- "SUMMARY=$RESULTS/SUMMARY.md"
  print -- "RELEASE_NOTES=$RESULTS/RELEASE.md"
  print -- "ARCHIVE=$archive"
  print -- "ARCHIVE_SHA256=$sidecar"
}

verify_local_package() {
  local archive="$BENCH_ROOT/laguna-m5-max-finalists-results.tar.gz"
  local sidecar="$archive.sha256"
  local complete="$BENCH_ROOT/package-complete.txt"
  [[ -s "$archive" && -s "$sidecar" && -s "$complete" ]] || return 1
  ( cd "$BENCH_ROOT" && shasum -c "${sidecar:t}" ) >/dev/null || return 1
  verify_manifest "$RESULTS" >/dev/null || return 1
  verify_archive_headers "$archive" >/dev/null || return 1
  local verify_dir rc=0 archive_digest manifest_digest
  verify_dir=$(mktemp -d /private/tmp/ds4-local-package-verify.XXXXXX) || return 1
  tar -C "$verify_dir" -xzf "$archive" || rc=1
  if (( rc == 0 )); then verify_manifest "$verify_dir/results" >/dev/null || rc=1; fi
  if (( rc == 0 )); then cmp "$RESULTS/MANIFEST.sha256" "$verify_dir/results/MANIFEST.sha256" >/dev/null || rc=1; fi
  if (( rc == 0 )); then cmp "$RESULTS/SUMMARY.md" "$verify_dir/results/SUMMARY.md" >/dev/null || rc=1; fi
  archive_digest=$(awk '{print $1}' "$sidecar") || rc=1
  manifest_digest=$(shasum -a 256 "$RESULTS/MANIFEST.sha256" | awk '{print $1}') || rc=1
  if (( rc == 0 )); then
    /usr/bin/grep -qx "status=COMPLETE" "$complete" || rc=1
    /usr/bin/grep -qx "archive_sha256=$archive_digest" "$complete" || rc=1
    /usr/bin/grep -qx "manifest_sha256=$manifest_digest" "$complete" || rc=1
  fi
  /bin/rm -rf -- "$verify_dir"
  return "$rc"
}

upload_draft() {
  load_paths || return 1
  verify_protocol_snapshot || return 1
  require_command gh || return 1
  local tag=${DS4_RELEASE_TAG:-}
  [[ -n "$tag" ]] || { print -u2 -- 'DS4_RELEASE_TAG is required for upload-draft'; return 1; }
  [[ "$tag" == [A-Za-z0-9]* && "$tag" != *[^A-Za-z0-9._-]* ]] || {
    print -u2 -- 'DS4_RELEASE_TAG must use only letters, digits, dot, underscore, and hyphen'
    return 1
  }
  local archive="$BENCH_ROOT/laguna-m5-max-finalists-results.tar.gz"
  local sidecar="$archive.sha256"
  [[ -s "$archive" && -s "$sidecar" && -s "$RESULTS/SUMMARY.md" && -s "$RESULTS/MANIFEST.sha256" ]] || {
    print -u2 -- 'package artifacts are missing; run package first'
    return 1
  }
  verify_local_package || { print -u2 -- 'local package verification failed'; return 1; }
  RUNBOOK_SHA=$(<"$RESULTS/metadata/runbook-revision.txt")
  local remote_refs
  remote_refs=$(git ls-remote --tags https://github.com/michaelasper/ds4.git \
    "refs/tags/$tag" "refs/tags/$tag^{}") || return 1
  if [[ -n "$remote_refs" ]]; then
    print -u2 -- "remote git tag already exists; refusing to reuse it: $tag"
    return 1
  fi
  if gh release view "$tag" --repo michaelasper/ds4 >/dev/null 2>&1; then
    print -u2 -- "release tag already exists; refusing to modify it: $tag"
    return 1
  fi
  gh auth status >/dev/null || return 1
  gh release create "$tag" --repo michaelasper/ds4 --target "$RUNBOOK_SHA" \
    --title 'Laguna S 2.1 M5 Max finalist confirmation' \
    --notes-file "$RESULTS/RELEASE.md" --draft --prerelease \
    "$archive#Raw results archive" \
    "$sidecar#Portable archive SHA-256" \
    "$RESULTS/SUMMARY.md#Benchmark summary" \
    "$RESULTS/MANIFEST.sha256#Per-file SHA-256 manifest" \
    "$RESULTS/metadata/BENCHMARK.md#Exact runbook" \
    "$RESULTS/metadata/runbook-revision.txt#Runbook commit" || return 1
  local verify_dir
  verify_dir=$(mktemp -d /private/tmp/ds4-release-verify.XXXXXX) || return 1
  gh release download "$tag" --repo michaelasper/ds4 --dir "$verify_dir" || return 1
  cmp "$archive" "$verify_dir/${archive:t}" || return 1
  cmp "$sidecar" "$verify_dir/${sidecar:t}" || return 1
  cmp "$RESULTS/SUMMARY.md" "$verify_dir/SUMMARY.md" || return 1
  cmp "$RESULTS/MANIFEST.sha256" "$verify_dir/MANIFEST.sha256" || return 1
  cmp "$RESULTS/metadata/BENCHMARK.md" "$verify_dir/BENCHMARK.md" || return 1
  cmp "$RESULTS/metadata/runbook-revision.txt" "$verify_dir/runbook-revision.txt" || return 1
  ( cd "$verify_dir" && shasum -c "${sidecar:t}" ) || return 1
  verify_archive_headers "$verify_dir/${archive:t}" >/dev/null || return 1
  local release_json="$verify_dir/release.json"
  gh release view "$tag" --repo michaelasper/ds4 \
    --json url,tagName,targetCommitish,isDraft,isPrerelease,assets > "$release_json" || return 1
  python3 - "$release_json" "$tag" "$RUNBOOK_SHA" <<'PY' || return 1
import json, sys
path, expected_tag, expected_sha = sys.argv[1:]
release = json.load(open(path))
expected_assets = {
    "laguna-m5-max-finalists-results.tar.gz",
    "laguna-m5-max-finalists-results.tar.gz.sha256",
    "SUMMARY.md", "MANIFEST.sha256", "BENCHMARK.md", "runbook-revision.txt",
}
actual_assets = {asset["name"] for asset in release.get("assets", [])}
checks = {
    "tagName": release.get("tagName") == expected_tag,
    "targetCommitish": release.get("targetCommitish") == expected_sha,
    "isDraft": release.get("isDraft") is True,
    "isPrerelease": release.get("isPrerelease") is True,
    "assetInventory": actual_assets == expected_assets,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"release metadata verification failed: {failed}; release={release}")
print(release["url"])
PY
  print -- "Verified release metadata:"
  print -r -- "$(<"$release_json")"
  print -- "Draft verified. Read SUMMARY.md, then publish with:"
  print -- "gh release edit ${(q)tag} --repo michaelasper/ds4 --draft=false --prerelease"
}

all_phases() {
  prepare || return 1
  prove || return 1
  measure || return 1
  analyse || return 1
  package_results || return 1
}

main() {
  local action=${1:-}
  case "$action" in
    prepare) prepare ;;
    prove) load_paths && MODEL_SHA=$(<"$RESULTS/metadata/model-sha256.txt") && prove ;;
    measure) load_paths && MODEL_SHA=$(<"$RESULTS/metadata/model-sha256.txt") && measure ;;
    analyse) analyse ;;
    package) package_results ;;
    all) all_phases ;;
    upload-draft) upload_draft ;;
    *) usage; return 2 ;;
  esac
}

if [[ "${DS4_BENCH_RUNNER_SOURCE_ONLY:-0}" != 1 ]]; then
  main "$@"
fi
