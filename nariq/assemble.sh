#!/usr/bin/env bash
# Put the release page's files together locally; nothing is uploaded (nariq/README.md).
#
#   assemble.sh page LINUX_DIR WINDOWS_DIR OUT   # build_linux.sh's archives, the Windows workflow's zips (its artifact,
#                                                # unzipped; "-" for none yet), run.sh, run.ps1 and one SHA256SUMS
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

case "${1:-}" in
page)
    [ $# -eq 4 ] || { echo "usage: assemble.sh page LINUX_DIR WINDOWS_DIR|- OUT" >&2; exit 2; }
    exec python3 "$HERE/assemble_page.py" "$2" "$3" "$4"
    ;;
*)
    echo "usage: assemble.sh page LINUX_DIR WINDOWS_DIR|- OUT" >&2; exit 2 ;;
esac
