#!/bin/bash
# Usage: ./build_pdf.sh nairr_startup_proposal   (reads NAME.md, writes NAME.pdf)
set -e
cd "$(dirname "$0")"
NAME="$1"
pandoc "$NAME.md" -s --metadata pagetitle="$NAME" -c style.css --embed-resources -o "$NAME.html" 2>/dev/null || pandoc "$NAME.md" -s --metadata pagetitle="$NAME" -c style.css --self-contained -o "$NAME.html"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$NAME.pdf" "file://$PWD/$NAME.html" >/dev/null 2>&1
rm -f "$NAME.html"
echo "built $NAME.pdf"
