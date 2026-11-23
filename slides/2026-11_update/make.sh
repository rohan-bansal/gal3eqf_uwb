#!/usr/bin/env bash
# Build the update deck. Run on a compute node:
#   srun -p rl2-lab -A rl2-lab -c 4 --mem 8G -t 00:15:00 slides/2026-11_update/make.sh [figs|deck|all]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV="$HERE/../.env"
BUILD="$HERE/build"
PPTX_SKILL="$HOME/.claude/skills/synced/a1e1ea47-6147-4b0c-8258-d89a38b50227_3880d501-2c80-4180-8ffa-2ee964d7d248/pptx"
export PATH="$HOME/.nvm/versions/node/v22.23.2/bin:$ENV/py/bin:$PATH"
export NODE_PATH="$HERE/../.tools/node_modules"
export PLAYWRIGHT_BROWSERS_PATH="$ENV/pw"
export FONTCONFIG_FILE="$ENV/fonts.conf"   # adds Liberation Sans (Arial metrics)
export LD_LIBRARY_PATH="$ENV/libroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
what="${1:-all}"
mkdir -p "$BUILD"

if [[ $what == figs || $what == all ]]; then
  python "$HERE/src/make_pages.py" "$BUILD"
  python "$HERE/src/render.py" "$BUILD"
fi

if [[ $what == deck || $what == all ]]; then
  (cd "$BUILD" && node "$HERE/src/build.js")
  python -m pip show python-pptx >/dev/null 2>&1 || python -m pip install -q python-pptx defusedxml lxml Pillow
  python "$PPTX_SKILL/scripts/office/validate.py" "$BUILD/gal3uwb_update.pptx"
  rm -rf "$BUILD/render" && mkdir -p "$BUILD/render"
  "$ENV/lo/opt/libreoffice25.8/program/soffice" --headless -env:UserInstallation=file://"$BUILD/lo_profile" \
    --convert-to pdf --outdir "$BUILD/render" "$BUILD/gal3uwb_update.pptx" >/dev/null 2>&1
  pdftoppm -jpeg -r 110 "$BUILD/render/gal3uwb_update.pdf" "$BUILD/render/slide"
  python "$PPTX_SKILL/scripts/slide_text.py" "$BUILD/gal3uwb_update.pptx" > "$BUILD/render/slide_text.md"
  ls "$BUILD/render"
fi
