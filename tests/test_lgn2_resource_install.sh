#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 PREFIX INSTALLED_TEST_BINARY" >&2
    exit 2
fi

prefix=$1
installed_test=$2
repo_root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
source_root="$repo_root/metal"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/lgn2-resource-relocation.XXXXXX")

cleanup() {
    rm -rf "$tmp"
}
trap cleanup EXIT HUP INT TERM

metal_sources='flash_attn.metal
dense.metal
moe.metal
unary.metal
dsv4_misc.metal
laguna.metal
dflash.metal
argsort.metal
cpy.metal
get_rows.metal
glu.metal
norm.metal
bin.metal'

for source in $metal_sources; do
    test -f "$source_root/$source"
    test -r "$prefix/share/lgn2/metal/$source"
done

for binary in lgn2 lgn2-server lgn2-bench lgn2-eval; do
    test -x "$prefix/bin/$binary"
done
test -x "$installed_test"

# An explicit non-empty source path remains exclusive in the loader.  The
# relocation gate exercises only the default lookup contract.
unset LGN2_METAL_FLASH_ATTN_SOURCE LGN2_METAL_DENSE_SOURCE \
    LGN2_METAL_MOE_SOURCE LGN2_METAL_UNARY_SOURCE \
    LGN2_METAL_DSV4_MISC_SOURCE LGN2_METAL_LAGUNA_SOURCE \
    LGN2_METAL_DFLASH_SOURCE LGN2_METAL_ARGSORT_SOURCE \
    LGN2_METAL_CPY_SOURCE LGN2_METAL_GET_ROWS_SOURCE \
    LGN2_METAL_GLU_SOURCE LGN2_METAL_NORM_SOURCE \
    LGN2_METAL_BIN_SOURCE

run_kernel_group() {
    label=$1
    binary=$2
    cwd=$3
    output="$tmp/$label.out"
    error="$tmp/$label.err"
    if ! (cd "$cwd" && "$binary" --metal-kernels >"$output" 2>"$error"); then
        cat "$output" "$error" >&2
        echo "resource relocation group failed: $label" >&2
        return 1
    fi
}

# The installed PREFIX/bin layout must discover PREFIX/share/lgn2/metal even
# when launched from an unrelated working directory.
mkdir -p "$tmp/installed-cwd"
run_kernel_group installed "$installed_test" "$tmp/installed-cwd"

# A copied bundle with metal/ beside its executable must work from a relocated
# working directory without consulting the source checkout.
mkdir -p "$tmp/adjacent/bin/metal" "$tmp/adjacent/cwd"
cp -p "$installed_test" "$tmp/adjacent/bin/lgn2_test"
for source in $metal_sources; do
    cp -p "$source_root/$source" "$tmp/adjacent/bin/metal/$source"
done
run_kernel_group adjacent "$tmp/adjacent/bin/lgn2_test" "$tmp/adjacent/cwd"

# A symlinked launcher must retain the resolved executable directory as a
# fallback, so resources beside the real copied executable remain discoverable.
mkdir -p "$tmp/symlink/cwd"
ln -s "$tmp/adjacent/bin/lgn2_test" "$tmp/symlink/lgn2_test"
run_kernel_group symlink "$tmp/symlink/lgn2_test" "$tmp/symlink/cwd"

# Finally, retain the source-tree/CWD path as the highest-priority default.
mkdir -p "$tmp/cwd-first/metal"
for source in $metal_sources; do
    cp -p "$source_root/$source" "$tmp/cwd-first/metal/$source"
done
run_kernel_group cwd-first "$installed_test" "$tmp/cwd-first"

echo "lgn2: staged Metal source relocation checks passed"
