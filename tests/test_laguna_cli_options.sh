#!/usr/bin/env bash
# Model-independent parser/help contract for the Laguna Metal CLI and server.
set -euo pipefail

cd "$(dirname "$0")/.."

test_tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/lgn2-laguna-cli.XXXXXX")
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

assert_version_contract() {
    local name=$1 bin=$2 expected_revision=$3
    local out="$test_tmp_dir/${name}.version"
    local expected="LagoonNebula ${name} development (revision ${expected_revision})"

    if "$bin" --version >"$out" 2>&1; then
        pass "$name --version exits zero"
    else
        fail "$name --version exits nonzero"
        sed -n '1,20p' "$out" >&2
        return
    fi
    if test "$(wc -l <"$out" | tr -d ' ')" -eq 1 &&
       grep -Fqx -- "$expected" "$out"; then
        pass "$name --version identity and revision"
    else
        fail "$name --version identity and revision"
        sed -n '1,20p' "$out" >&2
    fi

    out="$test_tmp_dir/${name}.version-alias"
    if "$bin" -V >"$out" 2>&1; then
        fail "$name rejects legacy -V alias"
    elif grep -Fqx -- "$expected" "$out"; then
        fail "$name rejects legacy -V alias (printed version)"
    else
        pass "$name rejects legacy -V alias"
    fi

    out="$test_tmp_dir/${name}.version-extra"
    if "$bin" --version extra >"$out" 2>&1; then
        fail "$name requires standalone --version"
    elif grep -Fqx -- "$expected" "$out"; then
        fail "$name requires standalone --version (printed version)"
    else
        pass "$name requires standalone --version"
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
    contains_option "$name advertises version" "--version" "$out"
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
        "--expert-profile"
        "--dir-steering-file" "--dir-steering-ffn" "--dir-steering-attn"
        "--mtp" "--mtp-draft" "--mtp-margin" "--glm-mtp"
        "--glm-mtp-timing" "--dspark" "--dspark-confidence"
        "--dspark-strict" "--role" "--layers" "--listen"
        "--coordinator" "--dist-prefill-chunk" "--dist-prefill-window"
        "--dist-activation-bits" "--dist-replay-check" "--debug"
        "--tensor-parallel" "--transport" "--rdma-device"
        "--rdma-gid-index" "--tensor-parallel-token-prefill"
        "--debug-hash" "--first-token-test"
        "--imatrix-dataset" "--imatrix-out" "--imatrix-max-prompts"
        "--imatrix-max-tokens" "--head-test" "--metal-graph-test"
        "--metal-graph-full-test" "--metal-graph-prompt-test"
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

assert_help_contract lgn2 ./lgn2
assert_help_contract lgn2-server ./lgn2-server
for spec_name in lgn2-bench lgn2-eval; do
    out="$test_tmp_dir/${spec_name}.help"
    if ./$spec_name --help >"$out" 2>&1; then
        pass "$spec_name --help exits zero"
        contains_option "$spec_name advertises version" "--version" "$out"
    else
        fail "$spec_name --help exits nonzero"
    fi
done

git_revision=$(git rev-parse --short=12 HEAD)
assert_version_contract lgn2 ./lgn2 "$git_revision"
assert_version_contract lgn2-server ./lgn2-server "$git_revision"
assert_version_contract lgn2-bench ./lgn2-bench "$git_revision"
assert_version_contract lgn2-eval ./lgn2-eval "$git_revision"

# --version is only the standalone identity query.  Do not consume it as an
# option value, and do not let it silently override another command spelling.
for spec_name in lgn2 lgn2-server lgn2-bench lgn2-eval; do
    out="$test_tmp_dir/${spec_name}.version-value"
    if ./"$spec_name" --model --version >"$out" 2>&1; then
        fail "$spec_name does not consume --version as a model value"
    elif grep -Eq '^LagoonNebula .* development \(revision [^)]*\)$' "$out"; then
        fail "$spec_name does not consume --version as a model value (printed version)"
    else
        pass "$spec_name keeps --version as an option value"
    fi
done

# A git archive has no .git directory.  Build the four binaries from that
# archive and require the deterministic source-archive fallback revision.
archive_dir="$test_tmp_dir/source-archive"
mkdir -p "$archive_dir"
if git archive --format=tar HEAD | tar -x -C "$archive_dir" &&
   (cd "$archive_dir" && make -j8 lgn2 lgn2-server lgn2-bench lgn2-eval) \
       >"$test_tmp_dir/source-archive.build" 2>&1; then
    for spec_name in lgn2 lgn2-server lgn2-bench lgn2-eval; do
        out="$test_tmp_dir/source-archive.${spec_name}.version"
        if (cd "$archive_dir" && ./"$spec_name" --version) >"$out" 2>&1 &&
           test "$(wc -l <"$out" | tr -d ' ')" -eq 1 &&
           grep -Fqx -- "LagoonNebula ${spec_name} development (revision unknown)" "$out"; then
            pass "source archive $spec_name reports unknown revision"
        else
            fail "source archive $spec_name reports unknown revision"
            sed -n '1,20p' "$out" >&2
        fi
    done
else
    fail "source archive builds four versioned executables"
    sed -n '1,120p' "$test_tmp_dir/source-archive.build" >&2
fi

# Public engine selection is intentionally DFlash-only in this product slice.
# Check the linked CLI surface as well as the source-level environment contract;
# this keeps an old MTP spelling from silently returning through a frontend.
for symbol in _lgn2_engine_has_dflash _lgn2_engine_dflash_draft_tokens; do
    if nm -gU ./lgn2 | grep -F -- "$symbol" >/dev/null &&
       nm -gU ./lgn2-server | grep -F -- "$symbol" >/dev/null; then
        pass "public DFlash symbol $symbol"
    else
        fail "missing public DFlash symbol $symbol"
    fi
done
legacy_symbols=(
    "_lgn2_engine_has_""mtp"
    "_lgn2_engine_""mtp_draft_tokens"
)
legacy_namespace="d""s4"
legacy_symbols+=(
    "_${legacy_namespace}_engine_has_dflash"
    "_${legacy_namespace}_engine_dflash_draft_tokens"
    "_${legacy_namespace}_engine_has_""mtp"
    "_${legacy_namespace}_engine_""mtp_draft_tokens"
)
for symbol in "${legacy_symbols[@]}"; do
    if nm -gU ./lgn2 | grep -F -- "$symbol" >/dev/null ||
       nm -gU ./lgn2-server | grep -F -- "$symbol" >/dev/null; then
        fail "legacy public symbol $symbol is still exported"
    else
        pass "legacy public symbol $symbol absent"
    fi
done
removed_symbols=(
    _lgn2_engine_has_output_head
    _lgn2_session_layer_payload_bytes
    _lgn2_session_save_layer_payload
    _lgn2_session_load_layer_payload
)
for symbol in "${removed_symbols[@]}"; do
    if nm -gU ./lgn2 | grep -F -- "$symbol" >/dev/null ||
       nm -gU ./lgn2-server | grep -F -- "$symbol" >/dev/null; then
        fail "removed public symbol $symbol is still exported"
    else
        pass "removed public symbol $symbol absent"
    fi
done
legacy_spec_disable_env="LGN2""_MTP""_SPEC_DISABLE"
if grep -Fq -- 'LGN2_DFLASH_SPEC_DISABLE' lgn2_cli.c lgn2_server.c &&
   ! grep -Fq -- "$legacy_spec_disable_env" lgn2_cli.c lgn2_server.c; then
    pass "DFlash speculative disable environment contract"
else
    fail "DFlash speculative disable environment contract"
fi
contains_option "server preserves HTTP batching" "--mixed-prefill-quantum" \
    "$test_tmp_dir/lgn2-server.help"
contains_option "server preserves disk KV" "--kv-disk-dir" \
    "$test_tmp_dir/lgn2-server.help"

server_parser_out="$test_tmp_dir/lgn2-server.retained"
if ./lgn2-server --host 127.0.0.1 --port 8000 --cors \
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

# The executable is Metal-only, so there is no backend selector.  The explicit
# --metal spelling remains a harmless product assertion.  DFlash options must
# likewise parse through their normal value checks before --help exits.
for spec_name in lgn2 lgn2-server; do
    bin=./$spec_name
    out="$test_tmp_dir/${spec_name}.positive"
    if "$bin" --metal \
        --dflash /tmp/laguna-dflash.gguf --dflash-draft 3 \
        --dflash-p-min 0.4 --help >"$out" 2>&1; then
        pass "$spec_name accepts Metal assertion and DFlash options"
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
    "--prefill-chunk 128"
    "--power 50"
    "--expert-profile profile.json"
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

for spec_name in lgn2 lgn2-server; do
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

for spec_name in lgn2-bench lgn2-eval; do
    bin=./$spec_name
    out="$test_tmp_dir/${spec_name}.expert-profile.negative"
    if "$bin" --expert-profile profile.json -m /dev/null >"$out" 2>&1; then
        fail "$spec_name accepts retired --expert-profile"
        continue
    fi
    if grep -Fq -- "$spec_name: unsupported option --expert-profile" "$out" &&
       ! grep -Eq -- "failed to open|unsupported model|architecture" "$out"; then
        pass "$spec_name rejects --expert-profile before model work"
    else
        fail "$spec_name gives non-product diagnostic for --expert-profile"
        sed -n '1,40p' "$out" >&2
    fi
done

# SSD expert-streaming switches were retired with the whole-model mmap
# contract.  They must follow the ordinary unknown-option path rather than a
# product-specific compatibility rejection.
retired_options=(
    "--ssd-streaming"
    "--ssd-streaming-cold"
    "--ssd-streaming-cache-experts 2"
    "--ssd-streaming-full-layers 1"
    "--ssd-streaming-preload-experts 1"
    "--simulate-used-memory 1GB"
)
for spec_name in lgn2 lgn2-server lgn2-bench lgn2-eval; do
    bin=./$spec_name
    for spec in "${retired_options[@]}"; do
        read -r -a argv <<< "$spec"
        out="$test_tmp_dir/${spec_name}.retired"
        if "$bin" "${argv[@]}" -m /dev/null >"$out" 2>&1; then
            fail "$spec_name accepts retired option: $spec"
            continue
        fi
        if grep -Fq -- "unknown option" "$out" &&
           ! grep -Eq -- "failed to open|unsupported model|architecture" "$out"; then
            pass "$spec_name treats retired option as unknown: $spec"
        else
            fail "$spec_name gives non-unknown diagnostic for retired option: $spec"
            sed -n '1,40p' "$out" >&2
        fi
    done
done

# Legacy raw diagnostics must fail during CLI parsing, before a model is
# opened. Exercise both bare switches and every argument-taking spelling with
# an intentionally invalid model path to make an accidental dispatch obvious.
removed_options=(
    "--imatrix-dataset /tmp/laguna-imatrix.txt"
    "--imatrix-out /tmp/laguna-imatrix.dat"
    "--imatrix-max-prompts 1"
    "--imatrix-max-tokens 1"
    "--head-test"
    "--metal-graph-test"
    "--metal-graph-full-test"
    "--metal-graph-prompt-test"
)
for spec in "${removed_options[@]}"; do
    read -r -a argv <<< "$spec"
    out="$test_tmp_dir/lgn2.removed"
    if ./lgn2 "${argv[@]}" -m /dev/null >"$out" 2>&1; then
        fail "lgn2 accepts removed raw diagnostic option: $spec"
        continue
    fi
    if grep -Fq -- "lgn2: unknown option: ${argv[0]}" "$out" &&
       ! grep -Eq -- "failed to open|unsupported model|architecture" "$out"; then
        pass "lgn2 rejects removed option before model work: $spec"
    else
        fail "lgn2 dispatches removed option or gives wrong diagnostic: $spec"
        sed -n '1,40p' "$out" >&2
    fi
done

echo "test_laguna_cli_options: PASS=$pass_count FAIL=$fail_count"
test "$fail_count" -eq 0
