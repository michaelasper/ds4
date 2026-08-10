#!/bin/zsh

emulate -LR zsh
setopt NO_UNSET PIPE_FAIL

export DS4_BENCH_RUNNER_SOURCE_ONLY=1
export DS4_BENCH_ROOT=/private/tmp/ds4-runner-test-root
export DS4_GGUF_DIR=/private/tmp/ds4-runner-test-models
source "${0:A:h}/run_m5_finalists.zsh"

fail() {
  print -u2 -- "FAIL: $*"
  exit 1
}

test_root=$(mktemp -d /private/tmp/ds4-runner-unit.XXXXXX) || exit 1
trap 'rm -r -- "$test_root"' EXIT

clean="$test_root/clean"
mkdir -p "$clean"
print -- 'ordinary diagnostic' > "$clean/d8.stderr.log"
print -- '        0 page faults' >> "$clean/d8.stderr.log"
validate_diagnostics "$clean" || fail 'clean diagnostic rejected'
[[ ! -s "$clean/unexpected-diagnostics.txt" ]] || fail 'clean report is not empty'

bad="$test_root/bad"
mkdir -p "$bad"
print -- 'nonfinite rows use stock fallback' > "$bad/d8.stderr.log"
validate_diagnostics "$bad"
[[ $? == 1 ]] || fail 'forbidden diagnostic did not return 1'
/usr/bin/grep -q 'fallback' "$bad/unexpected-diagnostics.txt" || fail 'forbidden line not recorded'
print -- 'route failure: unsupported kernel' > "$bad/d8.stderr.log"
validate_diagnostics "$bad"
[[ $? == 1 ]] || fail 'failure/unsupported diagnostic did not reject'
print -- 'GPU address fault' > "$bad/d8.stderr.log"
validate_diagnostics "$bad"
[[ $? == 1 ]] || fail 'GPU fault diagnostic did not reject'

missing="$test_root/missing"
mkdir -p "$missing"
validate_diagnostics "$missing"
[[ $? == 2 ]] || fail 'missing stderr did not fail closed'

generated="$test_root/generated"
mkdir -p "$generated"
print -- 'ordinary diagnostic' > "$generated/generated.stderr.log"
print -- 'ds4-bench: gen[ctx=8192] decoded text: "fallback is generated text"' \
  >> "$generated/generated.stderr.log"
validate_diagnostics "$generated" || fail 'generated text was incorrectly scanned as a diagnostic'

before="$test_root/before"
after="$test_root/after"
mkdir -p "$before" "$after"
print -- 'Pageouts: 100.' > "$before/vm-stat.txt"
print -- 'Swapouts: 20.' >> "$before/vm-stat.txt"
print -- 'Pageouts: 100.' > "$after/vm-stat.txt"
print -- 'Swapouts: 20.' >> "$after/vm-stat.txt"
print -- 'total = 4096.00M  used = 100.00M  free = 3996.00M' > "$before/vm-swapusage.txt"
print -- 'total = 4096.00M  used = 100.50M  free = 3995.50M' > "$after/vm-swapusage.txt"
validate_memory_state "$before" "$after" "$test_root/memory-clean.txt" || fail 'clean memory delta rejected'
print -- 'Pageouts: 101.' > "$after/vm-stat.txt"
print -- 'Swapouts: 20.' >> "$after/vm-stat.txt"
validate_memory_state "$before" "$after" "$test_root/memory-bad.txt"
[[ $? == 1 ]] || fail 'pageout growth did not reject'
print -- 'Pageouts: 100.' > "$after/vm-stat.txt"
validate_memory_state "$before" "$after" "$test_root/memory-missing.txt"
[[ $? == 2 ]] || fail 'missing swapout counter did not fail closed'

set_selector_env 15-canonical-ladder-gqa3 || fail 'combination selector missing'
[[ ${#SELECTOR_ENV[@]} == 2 ]] || fail 'combination selector count is not two'
[[ "$SELECTOR_ENV[1]" == DS4_METAL_LAGUNA_DECODE_LADDER=7,15,23,31,39,47 ]] || fail 'wrong ladder selector'
[[ "$SELECTOR_ENV[2]" == DS4_METAL_LAGUNA_SWA_GQA3=1 ]] || fail 'wrong GQA3 selector'
set_selector_env 00-r1-baseline || fail 'baseline selector missing'
[[ ${#SELECTOR_ENV[@]} == 0 ]] || fail 'baseline unexpectedly has a selector'
set_selector_env 05-canonical-ladder || fail 'canonical selector missing'
[[ "$SELECTOR_ENV[1]" == DS4_METAL_LAGUNA_DECODE_LADDER=7,15,23,31,39,47 ]] || fail 'canonical selector changed'
set_selector_env 12-gqa3 || fail 'GQA3 selector missing'
[[ "$SELECTOR_ENV[1]" == DS4_METAL_LAGUNA_SWA_GQA3=1 ]] || fail 'GQA3 selector changed'
set_selector_env 14-front-rung-ladder || fail 'front selector missing'
[[ "$SELECTOR_ENV[1]" == DS4_METAL_LAGUNA_DECODE_LADDER=1,7,15,23,31,39,47 ]] || fail 'front selector changed'
set_selector_env 16-front-rung-ladder-gqa3 || fail 'front combination selector missing'
[[ ${#SELECTOR_ENV[@]} == 2 ]] || fail 'front combination selector count is not two'

typeset -a order
order=("${(@f)$(balanced_order 1 00-r1-baseline "${CANDIDATE_ARMS[@]}")}")
[[ ${#order[@]} == 6 ]] || fail 'random block does not contain six arms'
[[ ${#${(u)order}[@]} == 6 ]] || fail 'random block contains duplicate arms'

manifest_root="$test_root/manifest"
mkdir -p "$manifest_root/subdir"
print -- alpha > "$manifest_root/a.txt"
print -- beta > "$manifest_root/subdir/b.txt"
make_manifest "$manifest_root" || fail 'manifest creation failed'
verify_manifest "$manifest_root" >/dev/null || fail 'exact manifest rejected'
print -- extra > "$manifest_root/extra.txt"
verify_manifest "$manifest_root" >/dev/null 2>&1
[[ $? == 1 ]] || fail 'unlisted manifest member did not reject'

archive_source="$test_root/archive-source"
mkdir -p "$archive_source/results"
print -- payload > "$archive_source/results/file.txt"
COPYFILE_DISABLE=1 tar --no-xattrs --no-acls --no-fflags \
  --uid 0 --gid 0 --uname root --gname wheel \
  -C "$archive_source" -czf "$test_root/normalised.tar.gz" results \
  || fail 'normalised archive creation failed'
verify_archive_headers "$test_root/normalised.tar.gz" >/dev/null \
  || fail 'normalised archive headers rejected'

print -- PASS
