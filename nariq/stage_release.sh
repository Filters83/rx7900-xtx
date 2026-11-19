#!/usr/bin/env bash
# Rebuild the folder that is uploaded to the GitHub release; nothing is uploaded.
#
#   nariq/stage_release.sh LINUX_DIR [WINDOWS_DIR] [OUT]
#
# LINUX_DIR holds build_linux.sh's two archives and SHA256SUMS; WINDOWS_DIR the Windows workflow's two zips and
# SHA256SUMS ("-" or omitted: a Linux-only release without run.ps1). OUT defaults to nariq-release-stage/TAG at the
# repository root and is replaced as a whole. The release assets are the archives, run.sh, run.ps1 (with Windows) and
# one SHA256SUMS over them, built by assemble_page.py; RELEASE_NOTES.md is the release page's text, made from
# nariq/RELEASE_NOTES.md with the block for the platforms present. Every hash is checked again afterwards, including
# with the parsers run.sh and run.ps1 use.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TAG="$(cat "$HERE/TAG")"
[ $# -ge 1 ] && [ $# -le 3 ] || { echo "usage: stage_release.sh LINUX_DIR [WINDOWS_DIR|-] [OUT]" >&2; exit 2; }
linux="$1" windows="${2:--}" out="${3:-$ROOT/nariq-release-stage/$TAG}"
new="$out.new"
git -C "$ROOT" diff --quiet HEAD -- nariq/run.sh nariq/run.ps1 nariq/RELEASE_NOTES.md nariq/TAG \
    || echo "stage_release.sh: warning: launchers, notes or TAG differ from HEAD; commit them before publishing" >&2
rm -rf "$new"
python3 "$HERE/assemble_page.py" "$linux" "$windows" "$new" > /dev/null

# The notes keep the block of the platforms present: <!-- windows --> ... <!-- /windows --> with Windows,
# <!-- linux-only --> ... <!-- /linux-only --> without. The marker lines themselves are dropped.
keep=windows drop=linux-only
[ "$windows" = - ] && keep=linux-only drop=windows
sed -e "/^<!-- $drop -->$/,/^<!-- \/$drop -->$/d" -e "/^<!-- \/\{0,1\}$keep -->$/d" \
    "$HERE/RELEASE_NOTES.md" > "$new/RELEASE_NOTES.md"
! grep -n -e '<!--' -e 'TODO' "$new/RELEASE_NOTES.md" || { echo "stage_release.sh: markers left in the notes" >&2; exit 1; }
grep -q "$TAG" "$new/RELEASE_NOTES.md" || { echo "stage_release.sh: the notes do not name $TAG" >&2; exit 1; }

cd "$new"
! grep -q $'\r' SHA256SUMS || { echo "stage_release.sh: SHA256SUMS has CR line endings" >&2; exit 1; }
sha256sum --quiet -c SHA256SUMS
# exactly the listed files, SHA256SUMS and the notes
diff <(ls -1 | sort) <({ cut -d' ' -f3 SHA256SUMS; echo SHA256SUMS; echo RELEASE_NOTES.md; } | sort) \
    || { echo "stage_release.sh: unexpected or missing files" >&2; exit 1; }
# run.sh's parser (sed, whole lines) and run.ps1's (regex per line) find each archive they download
for archive in "nari-llama.cpp-$TAG-bin-linux-cuda-12.8-x64" "cudart-nari-llama.cpp-$TAG-linux-cuda-12.8-x64"; do
    [ -n "$(sed -n "s/^\([0-9a-f]\{64\}\)  $archive.tar.gz$/\1/p" SHA256SUMS)" ] \
        || { echo "stage_release.sh: run.sh would not find $archive.tar.gz" >&2; exit 1; }
done
if [ "$windows" != - ]; then
    for archive in "nari-llama.cpp-$TAG-bin-win-cuda-12.8-x64" "cudart-nari-llama.cpp-$TAG-win-cuda-12.8-x64"; do
        grep -Eq "^[0-9a-f]{64}  ${archive//./\\.}\.zip$" SHA256SUMS \
            || { echo "stage_release.sh: run.ps1 would not find $archive.zip" >&2; exit 1; }
    done
fi
cd - > /dev/null
rm -rf "$out"
mv "$new" "$out"
echo "staged $out:"
(cd "$out" && ls -l --time-style=+ | sed 1d && echo && cat SHA256SUMS)
