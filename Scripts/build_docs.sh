#!/usr/bin/env bash
# Build static DocC site(s) for one or more SPM targets into ./docs
# (GitHub Pages-ready). Modeled after ml-explore/mlx-swift's
# tools/build-documentation.sh, with optional LLM-Markdown export.
#
# Usage:
#   Scripts/build_docs.sh                # build all $TARGETS into ./docs
#   Scripts/build_docs.sh preview        # local preview (first target only)
#   Scripts/build_docs.sh -f             # bypass gh-pages branch guard
#   EMIT_LLMS_TXT=1 Scripts/build_docs.sh   # also emit docs/llms.txt
#                                           # (needs a swift.org toolchain;
#                                           #  ~/.swiftly main-snapshot works)
#
# Env (edit defaults below or pass at call site):
#   TARGETS              Space-separated target names. Default: MLXVis.
#   HOSTING_BASE_PATH    Repo name on GitHub Pages. Default: mlx-swift-vis.
#   REPO_URL             https URL to the GitHub repo (no trailing slash).
#   REPO_BRANCH          Branch the source links point at. Default: main.
#   OUTPUT_DIR           Default: docs
#   REQUIRE_GH_PAGES=1   Refuse to build off the gh-pages branch unless -f.
#   EMIT_MARKDOWN=1      Pass experimental Markdown-output flags (per-symbol .md).
#   EMIT_LLMS_TXT=1      Above + concatenate the .md into <OUTPUT_DIR>/llms.txt.
#   TOOLCHAIN            Build with a specific toolchain (identifier or "swift").
#                        Honors TOOLCHAINS too. If neither is set and swiftly is
#                        installed, ~/.swiftly/env.sh is sourced so the toolchain
#                        you `swiftly use` is picked up automatically.
set -euo pipefail

cd "$(dirname "$0")/.."

TARGETS="${TARGETS:-MLXVis}"
HOSTING_BASE_PATH="${HOSTING_BASE_PATH:-mlx-swift-vis}"
REPO_URL="${REPO_URL:-https://github.com/mnmly/mlx-swift-vis}"
REPO_BRANCH="${REPO_BRANCH:-main}"
OUTPUT_DIR="${OUTPUT_DIR:-docs}"

# Toolchain selection. The experimental Markdown flags require a recent swift-docc
# (newer than the one bundled with current Xcode). Let callers point at one without
# prefixing every invocation, and fall back to a swiftly selection if present.
if [[ -z "${TOOLCHAINS:-}" && -n "${TOOLCHAIN:-}" ]]; then
    export TOOLCHAINS="$TOOLCHAIN"
fi
if [[ -z "${TOOLCHAINS:-}" && -f "$HOME/.swiftly/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.swiftly/env.sh"
fi

FORCE=0
MODE="build"
for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=1 ;;
        preview)    MODE="preview" ;;
    esac
done

if [[ "${REQUIRE_GH_PAGES:-0}" == "1" && "$MODE" == "build" && $FORCE -eq 0 ]]; then
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo)"
    if [[ "$branch" != "gh-pages" ]]; then
        echo "Refusing to build off branch '$branch'. Use -f to override."
        exit 1
    fi
fi

export DOCC_JSON_PRETTYPRINT=YES

# Preview: first target only — `swift package preview-documentation` is
# single-target and interactive.
if [[ "$MODE" == "preview" ]]; then
    first_target="${TARGETS%% *}"
    exec swift package --disable-sandbox \
        preview-documentation --target "$first_target"
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Resolve the same docc the plugin will use: prefer one on PATH (swiftly puts the
# selected toolchain's binaries there), else fall back to xcrun (honors TOOLCHAINS).
# The experimental flags only exist in recent swift-docc; probe and skip-with-warning
# rather than hard-failing on an unknown option (the Xcode-bundled docc lacks them).
DOCC_BIN="$(command -v docc 2>/dev/null || xcrun --find docc 2>/dev/null || true)"
docc_supports() {
    [[ -n "$DOCC_BIN" ]] && "$DOCC_BIN" convert --help 2>&1 | grep -q -- "$1"
}

EXTRA_FLAGS=()
if [[ "${EMIT_MARKDOWN:-0}" == "1" || "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
    for flag in --enable-experimental-markdown-output \
                --enable-experimental-markdown-output-manifest; do
        if docc_supports "$flag"; then
            EXTRA_FLAGS+=("$flag")
        else
            echo "warning: active docc does not support '$flag' — skipping." >&2
            echo "         Use a recent swift.org toolchain (e.g. via swiftly:" >&2
            echo "         'swiftly install main-snapshot && swiftly use main-snapshot')," >&2
            echo "         or pass TOOLCHAIN=<id>. The Xcode-bundled docc lacks it." >&2
        fi
    done
fi

SOURCE_FLAGS=()
if [[ -n "$REPO_URL" ]]; then
    SOURCE_FLAGS+=(
        --source-service github
        --source-service-base-url "${REPO_URL%/}/blob/${REPO_BRANCH}"
        --checkout-path "$(pwd)"
    )
fi

for TARGET in $TARGETS; do
    slug="$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')"
    out="$OUTPUT_DIR/$TARGET"
    mkdir -p "$out"

    echo ">> Building DocC for $TARGET → $out"
    swift package --allow-writing-to-directory "$out" \
        generate-documentation \
        --target "$TARGET" \
        --fallback-bundle-identifier "${HOSTING_BASE_PATH}.${slug}" \
        --output-path "$out" \
        --emit-digest \
        --disable-indexing \
        --transform-for-static-hosting \
        --hosting-base-path "${HOSTING_BASE_PATH}/${TARGET}" \
        ${SOURCE_FLAGS[@]+"${SOURCE_FLAGS[@]}"} \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
done

if [[ "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
    LLMS="$OUTPUT_DIR/llms.txt"
    {
        echo "# ${HOSTING_BASE_PATH} — DocC export for LLM consumption"
        echo
        echo "Generated $(date -u +%FT%TZ) from swift-docc."
        echo "Targets: $TARGETS"
        echo
        for TARGET in $TARGETS; do
            find "$OUTPUT_DIR/$TARGET/data" -name '*.md' -type f 2>/dev/null \
                | sort \
                | while IFS= read -r f; do
                    rel="${f#$OUTPUT_DIR/}"
                    echo
                    echo "---"
                    echo "## $rel"
                    echo
                    cat "$f"
                done
        done
    } > "$LLMS"
    echo "Wrote $LLMS ($(wc -l < "$LLMS" | tr -d ' ') lines)."
fi

# Write a top-level redirect index.html so the Pages root URL lands on the
# first target's documentation instead of returning 404.
first_target="${TARGETS%% *}"
first_slug="$(echo "$first_target" | tr '[:upper:]' '[:lower:]')"
redirect_url="/${HOSTING_BASE_PATH}/${first_target}/documentation/${first_slug}/"
cat > "$OUTPUT_DIR/index.html" <<HTML
<!doctype html>
<meta charset="utf-8">
<title>${HOSTING_BASE_PATH}</title>
<meta http-equiv="refresh" content="0; url=${redirect_url}">
<link rel="canonical" href="${redirect_url}">
<p>Redirecting to <a href="${redirect_url}">${redirect_url}</a>.</p>
HTML

echo
echo "Docs written to $OUTPUT_DIR/. Open $OUTPUT_DIR/<Target>/index.html"
echo "or push to gh-pages."
