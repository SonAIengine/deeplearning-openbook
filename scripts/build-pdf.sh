#!/usr/bin/env bash
# Build a single PDF from all markdown content (README + 13 chapters + topics/).
#
# Dependencies (Ubuntu):
#   sudo apt-get install -y pandoc texlive-xetex texlive-fonts-recommended \
#     texlive-lang-cjk texlive-latex-extra fonts-nanum fonts-nanum-coding \
#     imagemagick librsvg2-bin
#
# Dependencies (macOS):
#   brew install pandoc imagemagick librsvg
#   brew install --cask mactex-no-gui   # or basictex + tlmgr install xecjk
#   # Korean fonts: install NanumGothic via Font Book or `brew install --cask font-nanum-gothic`
#
# Usage:
#   ./scripts/build-pdf.sh              # → deeplearning-openbook.pdf
#   OUTPUT=foo.pdf ./scripts/build-pdf.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT="${OUTPUT:-deeplearning-openbook.pdf}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# ---------------------------------------------------------------------------
# 1. Convert GIF first-frames → PNG so xelatex can embed them.
# ---------------------------------------------------------------------------
mkdir -p assets/gif-stills
shopt -s nullglob
for gif in assets/gifs/*.gif; do
  base="$(basename "$gif" .gif)"
  out="assets/gif-stills/${base}.png"
  if [ ! -f "$out" ] || [ "$gif" -nt "$out" ]; then
    echo "  convert $gif[0] → $out"
    if command -v magick >/dev/null 2>&1; then
      magick "${gif}[0]" "$out"
    else
      convert "${gif}[0]" "$out"
    fi
  fi
done
shopt -u nullglob

# ---------------------------------------------------------------------------
# 2. File order: README → 본문 13 chapters → topics/ as appendix.
# ---------------------------------------------------------------------------
chapters=(
  "README.md"
  "01_ML_Overview.md"
  "02_FNNs.md"
  "03_DNNs.md"
  "04_CNNs.md"
  "05_RNNs.md"
  "06_LSTM_GRU.md"
  "07_Cross_Architecture_Design.md"
  "08_WhatIf_Ablation.md"
  "09_Architecture_Hyperparameter_Decisions.md"
  "10_Training_Diagnosis.md"
  "11_Comparison_Decision.md"
  "12_BigPicture_FAQ.md"
  "13_Task_Design_Playbook.md"
)

topics=(
  "topics/EDA/README.md"
  "topics/EDA/00_quick_reference.md"
  "topics/EDA/01_philosophy.md"
  "topics/EDA/02_anscombe_datasaurus.md"
  "topics/EDA/03_distributions.md"
  "topics/EDA/04_missing_data.md"
  "topics/hyperparameters/README.md"
  "topics/hyperparameters/loss.md"
  "topics/hyperparameters/optimizer.md"
  "topics/hyperparameters/regularization.md"
  "topics/hyperparameters/lr_schedule.md"
  "topics/regression_loss/README.md"
  "topics/regression_loss/01_geometry.md"
)

# ---------------------------------------------------------------------------
# 3. Pre-process each markdown file:
#    - Rewrite assets/gifs/*.gif  →  assets/gif-stills/*.png
#    - Resolve relative image paths from topics/*/file.md to repo root
#      so pandoc finds them regardless of `--resource-path`.
# ---------------------------------------------------------------------------
preprocess() {
  local input="$1"
  local output="$2"
  local dir
  dir="$(dirname "$input")"

  # Compute relative prefix to repo root from the file's directory.
  # README.md / 01_*.md → "" ;  topics/EDA/README.md → "../../"
  local prefix=""
  if [ "$dir" != "." ]; then
    local depth
    depth="$(echo "$dir" | awk -F/ '{print NF}')"
    for ((i=0; i<depth; i++)); do prefix="${prefix}../"; done
  fi

  python3 - "$input" "$output" "$prefix" <<'PY'
import re, sys, pathlib
src, dst, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
text = pathlib.Path(src).read_text(encoding="utf-8")

# 1. Drop YAML front-matter if present (would confuse pandoc when concatenated).
text = re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)

# 2. Replace .gif refs with .png stills.
text = re.sub(r"assets/gifs/([^)\s]+)\.gif", r"assets/gif-stills/\1.png", text)

# 3. Rewrite relative asset paths so pandoc resolves them from repo root.
#    Only touches paths that don't already start with "/", "http", or repo-root prefixes.
def fix_path(match):
    alt = match.group(1)
    path = match.group(2)
    if path.startswith(("http://", "https://", "/", "data:")):
        return match.group(0)
    # If the path already starts with a repo-root folder, leave it.
    if path.startswith(("assets/", "topics/")):
        return match.group(0)
    # Otherwise prepend the prefix that walks up to repo root.
    return f"![{alt}]({prefix}{path})"

text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", fix_path, text)

pathlib.Path(dst).write_text(text, encoding="utf-8")
PY
}

# ---------------------------------------------------------------------------
# 4. Concatenate everything into one big markdown file with section dividers.
# ---------------------------------------------------------------------------
combined="$TMPDIR/combined.md"
: > "$combined"

emit_divider() {
  printf '\n\n\\newpage\n\n' >> "$combined"
}

emit_part() {
  local title="$1"
  printf '\n\n\\part{%s}\n\n' "$title" >> "$combined"
}

emit_part "본문"
for f in "${chapters[@]}"; do
  echo "  + $f"
  out="$TMPDIR/$(echo "$f" | tr '/' '_')"
  preprocess "$f" "$out"
  cat "$out" >> "$combined"
  emit_divider
done

emit_part "부록 — 심화 자료 (topics/)"
for f in "${topics[@]}"; do
  echo "  + $f"
  out="$TMPDIR/$(echo "$f" | tr '/' '_')"
  preprocess "$f" "$out"
  cat "$out" >> "$combined"
  emit_divider
done

# ---------------------------------------------------------------------------
# 5. Run pandoc → PDF (xelatex for Korean + math).
# ---------------------------------------------------------------------------
echo "→ pandoc → $OUTPUT"
pandoc "$combined" \
  --from=gfm+tex_math_dollars+raw_tex+yaml_metadata_block+pipe_tables+backtick_code_blocks+fenced_code_blocks+task_lists \
  --pdf-engine=xelatex \
  --metadata-file=pdf/metadata.yaml \
  --top-level-division=chapter \
  --highlight-style=tango \
  --resource-path=".:assets:assets/images:assets/gif-stills" \
  -V mainfont="DejaVu Serif" \
  -V CJKmainfont="NanumGothic" \
  -V CJKmonofont="NanumGothicCoding" \
  -o "$OUTPUT"

echo
echo "✓ Built $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
