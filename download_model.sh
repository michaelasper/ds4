#!/bin/sh
set -e

MODEL_ALIAS="laguna-q2-q3"
MODEL_FILE="laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf"
MODEL_SHA256="61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903"
MODEL_REPOSITORY="antirez/Laguna-S-2.1-GGUF"
MODEL_REVISION="main"
MODEL_URL="https://huggingface.co/$MODEL_REPOSITORY/resolve/$MODEL_REVISION/$MODEL_FILE"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${LGN2_GGUF_DIR:-}" ]; then
    OUT_DIR=$LGN2_GGUF_DIR
else
    OUT_DIR=$ROOT/gguf
fi
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac

MODEL_PATH="$OUT_DIR/$MODEL_FILE"
PART_PATH="$MODEL_PATH.part"
LINK_PATH="$ROOT/lgn2.gguf"
TOKEN=${HF_TOKEN:-}
DRY_RUN=0

usage() {
    cat <<EOF
Laguna S 2.1 GGUF downloader

Usage:
  ./download_model.sh $MODEL_ALIAS [--token TOKEN] [--dry-run]
  ./download_model.sh --help

The downloader fetches this exact benchmark model:
  file:       $MODEL_FILE
  SHA-256:    $MODEL_SHA256
  source:     $MODEL_URL

Environment:
  LGN2_GGUF_DIR   Directory for the model (default: ./gguf)
  HF_TOKEN       Optional Hugging Face access token

Existing model files and lgn2.gguf links are never replaced. A partial
download is resumed in place and remains on disk if verification fails.
EOF
}

die() {
    echo "download_model.sh: $*" >&2
    exit 1
}

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        die "need shasum or sha256sum to verify $1"
    fi
}

verify_model() {
    actual=$(sha256_file "$1") || die "could not hash $1"
    [ "$actual" = "$MODEL_SHA256" ] || die "refusing unexpected SHA-256 for $1: $actual"
}

load_token() {
    if [ -n "$TOKEN" ] || [ -z "${HOME:-}" ]; then
        return
    fi
    token_file="$HOME/.cache/huggingface/token"
    if [ -s "$token_file" ]; then
        TOKEN=$(sed -n '1p' "$token_file")
    fi
}

download_model() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "Dry run; no files changed."
        echo "Would download: $MODEL_URL"
        echo "Would verify:   $MODEL_SHA256"
        echo "Would save to:  $MODEL_PATH"
        return
    fi

    mkdir -p "$OUT_DIR"

    if [ -f "$MODEL_PATH" ]; then
        verify_model "$MODEL_PATH"
        echo "Already present and verified: $MODEL_PATH"
        return
    fi
    if [ -L "$MODEL_PATH" ] || [ -e "$MODEL_PATH" ]; then
        die "refusing to replace existing non-regular model path: $MODEL_PATH"
    fi
    if [ -L "$PART_PATH" ]; then
        die "refusing to write through symlink partial path: $PART_PATH"
    fi

    command -v curl >/dev/null 2>&1 || die "curl is required to download the model"
    load_token
    if [ -e "$PART_PATH" ]; then
        echo "Resuming partial download: $PART_PATH"
    else
        echo "Downloading $MODEL_FILE"
    fi
    echo "from $MODEL_URL"
    echo "The approximately 45 GiB download can be resumed by rerunning this command."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - \
            -H "Authorization: Bearer $TOKEN" \
            -o "$PART_PATH" "$MODEL_URL"
    else
        curl -fL --progress-meter -C - -o "$PART_PATH" "$MODEL_URL"
    fi

    [ -s "$PART_PATH" ] || die "download completed without creating $PART_PATH"
    verify_model "$PART_PATH" || die "downloaded file failed verification; leaving $PART_PATH"
    if [ -L "$MODEL_PATH" ] || [ -e "$MODEL_PATH" ]; then
        die "model appeared during download; leaving verified partial at $PART_PATH"
    fi
    mv "$PART_PATH" "$MODEL_PATH"
    echo "Saved and verified: $MODEL_PATH"
}

link_model() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "Would link:      $LINK_PATH -> $MODEL_PATH (only if the link is absent)"
        return
    fi

    if [ -L "$LINK_PATH" ]; then
        current=$(readlink "$LINK_PATH") || current=""
        if [ "$current" = "$MODEL_PATH" ]; then
            echo "Link already points to the verified model: $LINK_PATH"
        else
            echo "Existing link left untouched: $LINK_PATH -> $current" >&2
            echo "Use the downloaded model directly: $MODEL_PATH" >&2
        fi
        return
    fi
    if [ -e "$LINK_PATH" ]; then
        echo "Existing file left untouched: $LINK_PATH" >&2
        echo "Use the downloaded model directly: $MODEL_PATH" >&2
        return
    fi

    ln -s "$MODEL_PATH" "$LINK_PATH"
    echo "Linked $LINK_PATH -> $MODEL_PATH"
}

MODEL=${1:-}
case "$MODEL" in
    -h|--help|help)
        usage
        exit 0
        ;;
    "$MODEL_ALIAS")
        shift
        ;;
    "")
        usage >&2
        exit 2
        ;;
    *)
        echo "Unknown model target: $MODEL" >&2
        usage >&2
        exit 2
        ;;
esac

while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        --token)
            shift
            [ "$#" -gt 0 ] || die "missing value after --token"
            TOKEN=$1
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
    shift
done

download_model
link_model

echo "Done."
