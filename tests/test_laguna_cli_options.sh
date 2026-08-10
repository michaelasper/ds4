#!/usr/bin/env bash
# Model-independent parser/help contract for the Laguna Metal CLI and server.
set -euo pipefail

cd "$(dirname "$0")/.."

test_tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-laguna-cli.XXXXXX")
trap 'rm -rf "$test_tmp_dir"' EXIT

pass_count=0
fail_count=0

pass() {
    pass_count=$((pass_count + 1))
    echo "ok $1"
}

fail() {
    fail_count=$((fail_count + 1))
    echo "FAIL $1" >&2
}

contains_option() {
    local name=$1 option=$2 file=$3
    if grep -Eq -- "^[[:space:]]{2}${option}([[:space:]]|$)" "$file"; then
        pass "$name"
    else
        fail "$name (missing option row: $option)"
        sed -n '1,80p' "$file" >&2
    fi
}

omits_token() {
    local name=$1 token=$2 file=$3
    if grep -Fq -- "$token" "$file"; then
        fail "$name (unexpected help token: $token)"
    else
        pass "$name"
    fi
}

assert_help_contract() {
    local name=$1 bin=$2
    local out="$test_tmp_dir/${name}.help"

    if "$bin" --help >"$out" 2>&1; then
        pass "$name --help exits zero"
    else
        fail "$name --help exits nonzero"
        return
    fi

    contains_option "$name advertises Metal" "--metal" "$out"
    contains_option "$name advertises DFlash" "--dflash" "$out"
    contains_option "$name advertises DFlash draft width" "--dflash-draft" "$out"
    contains_option "$name advertises DFlash probability floor" "--dflash-p-min" "$out"

    local forbidden=(
        "--backend" "--cpu" "--cuda" "--rocm"
        "--gpu-vram" "--gpu-devices" "--cuda-tensor-parallel"
        "--ssd-streaming" "--ssd-streaming-cold"
        "--ssd-streaming-cache-experts" "--ssd-streaming-full-layers"
        "--ssd-streaming-preload-experts" "--simulate-used-memory"
        "--prefill-chunk" "--power"
        "--dir-steering-file" "--dir-steering-ffn" "--dir-steering-attn"
        "--mtp" "--mtp-draft" "--mtp-margin" "--glm-mtp"
        "--glm-mtp-timing" "--dspark" "--dspark-confidence"
        "--dspark-strict" "--role" "--layers" "--listen"
        "--coordinator" "--dist-prefill-chunk" "--dist-prefill-window"
        "--dist-activation-bits" "--dist-replay-check" "--debug"
        "--tensor-parallel" "--transport" "--rdma-device"
        "--rdma-gid-index" "--tensor-parallel-token-prefill"
        "--debug-hash" "--first-token-test"
    )
    local pattern
    for pattern in "${forbidden[@]}"; do
        omits_token "$name omits $pattern from help" "$pattern" "$out"
    done

    out="$test_tmp_dir/${name}.distributed.help"
    "$bin" --help distributed >"$out" 2>&1 || true
    for pattern in "--role" "--tensor-parallel" "--dir-steering-file"; do
        omits_token "$name rejects obsolete help topic content: $pattern" "$pattern" "$out"
    done
}

assert_help_contract ds4 ./ds4
assert_help_contract ds4-server ./ds4-server

# Public engine selection is intentionally DFlash-only in this product slice.
# Check the linked CLI surface as well as the source-level environment contract;
# this keeps an old MTP spelling from silently returning through a frontend.
for symbol in _ds4_engine_has_dflash _ds4_engine_dflash_draft_tokens; do
    if nm -gU ./ds4 | grep -F -- "$symbol" >/dev/null &&
       nm -gU ./ds4-server | grep -F -- "$symbol" >/dev/null; then
        pass "public DFlash symbol $symbol"
    else
        fail "missing public DFlash symbol $symbol"
    fi
done
legacy_symbols=(
    "_ds4_engine_has_""mtp"
    "_ds4_engine_""mtp_draft_tokens"
)
for symbol in "${legacy_symbols[@]}"; do
    if nm -gU ./ds4 | grep -F -- "$symbol" >/dev/null ||
       nm -gU ./ds4-server | grep -F -- "$symbol" >/dev/null; then
        fail "legacy public symbol $symbol is still exported"
    else
        pass "legacy public symbol $symbol absent"
    fi
done
legacy_spec_disable_env="DS4""_MTP""_SPEC_DISABLE"
if grep -Fq -- 'DS4_DFLASH_SPEC_DISABLE' ds4_cli.c ds4_server.c &&
   ! grep -Fq -- "$legacy_spec_disable_env" ds4_cli.c ds4_server.c; then
    pass "DFlash speculative disable environment contract"
else
    fail "DFlash speculative disable environment contract"
fi
contains_option "server preserves HTTP batching" "--mixed-prefill-quantum" \
    "$test_tmp_dir/ds4-server.help"
contains_option "server preserves disk KV" "--kv-disk-dir" \
    "$test_tmp_dir/ds4-server.help"

server_parser_out="$test_tmp_dir/ds4-server.retained"
if ./ds4-server --host 127.0.0.1 --port 8000 --cors \
    --trace /tmp/laguna-trace --batched-session 2 --mixed-prefill-quantum 64 \
    --kv-disk-dir /tmp/laguna-kv --kv-disk-space-mb 64 \
    --kv-cache-min-tokens 1 --kv-cache-cold-max-tokens 1 \
    --kv-cache-continued-interval-tokens 1 --kv-cache-boundary-trim-tokens 1 \
    --kv-cache-boundary-align-tokens 1 --kv-cache-reject-different-quant \
    --disable-exact-dsml-tool-replay --tool-memory-max-ids 1 --help \
    >"$server_parser_out" 2>&1; then
    pass "server retains HTTP/KV/batching parser contract"
else
    fail "server rejected retained HTTP/KV/batching parser contract"
    sed -n '1,80p' "$server_parser_out" >&2
fi

# --backend is intentionally hidden, but its retained Metal spelling must
# still parse without attempting to open a model.  DFlash options must likewise
# parse through their normal value checks before --help exits.
for spec_name in ds4 ds4-server; do
    bin=./$spec_name
    out="$test_tmp_dir/${spec_name}.positive"
    if "$bin" --backend metal --metal \
        --dflash /tmp/laguna-dflash.gguf --dflash-draft 3 \
        --dflash-p-min 0.4 --help >"$out" 2>&1; then
        pass "$spec_name accepts hidden Metal backend and DFlash options"
    else
        fail "$spec_name rejected retained Metal/DFlash parser contract"
        sed -n '1,80p' "$out" >&2
    fi
done

unsupported=(
    "--cpu"
    "--cuda"
    "--rocm"
    "--backend cuda"
    "--gpu-vram 1"
    "--gpu-devices 0"
    "--cuda-tensor-parallel"
    "--ssd-streaming"
    "--ssd-streaming-cold"
    "--ssd-streaming-cache-experts 2"
    "--ssd-streaming-full-layers 1"
    "--ssd-streaming-preload-experts 1"
    "--simulate-used-memory 1GB"
    "--prefill-chunk 128"
    "--power 50"
    "--dir-steering-file direction.bin"
    "--dir-steering-ffn 1"
    "--dir-steering-attn 1"
    "--mtp support.gguf"
    "--mtp-draft 2"
    "--mtp-margin 3"
    "--glm-mtp"
    "--glm-mtp-timing"
    "--dspark"
    "--dspark-confidence 0.5"
    "--dspark-strict"
    "--role worker"
    "--layers 0:1"
    "--listen 127.0.0.1 9000"
    "--coordinator 127.0.0.1 9000"
    "--dist-prefill-chunk 1"
    "--dist-prefill-window 1"
    "--dist-activation-bits 16"
    "--dist-replay-check"
    "--debug"
    "--tensor-parallel"
    "--transport tcp"
    "--rdma-device test"
    "--rdma-gid-index 0"
    "--tensor-parallel-token-prefill"
    "--debug-hash 1"
    "--first-token-test"
)

for spec_name in ds4 ds4-server; do
    bin=./$spec_name
    for spec in "${unsupported[@]}"; do
        read -r -a argv <<< "$spec"
        out="$test_tmp_dir/${spec_name}.negative"
        if "$bin" "${argv[@]}" -m /dev/null >"$out" 2>&1; then
            fail "$spec_name accepts unsupported option: $spec"
            continue
        fi
        if grep -Fq -- "Laguna S2.1 on Apple Metal only" "$out"; then
            pass "$spec_name rejects unsupported option: $spec"
        else
            fail "$spec_name gives non-product diagnostic for: $spec"
            sed -n '1,40p' "$out" >&2
        fi
    done
done

echo "test_laguna_cli_options: PASS=$pass_count FAIL=$fail_count"
test "$fail_count" -eq 0
