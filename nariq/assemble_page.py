"""Verify matching platform archives and assemble a local release candidate."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

here = Path(__file__).resolve().parent
repo = Path(os.environ.get("NARIQ_SOURCE_REPO", here.parent))
tag = (here / "TAG").read_text().strip()
# provenance.py hashes these; a build without Git metadata is checked against them.
SOURCE_DIRS = ("app", "cmake", "common", "conversion", "ggml", "gguf-py/gguf", "include", "scripts", "src", "tools", "vendor",
               "nariq")
ROOT_FILES = ("CMakeLists.txt", "convert_hf_to_gguf.py", "LICENSE")
GENERATED = ("tools/ui/dist/",)  # prepare_ui.py's output, checked against ui.json's sha256 at build time
# Paths that enter neither build: the Linux and Windows archives may come from revisions that differ only here.
# CI workflow files and the package smoke check do not change the sources either build compiles.
NON_BUILD = re.compile(r"\.gitignore|\.dockerignore|.*\.md|\.github/workflows/.*|nariq/(run\.sh|run\.ps1|Dockerfile|Dockerfile\.dockerignore"
                       r"|assemble\.sh|assemble_page\.py|stage_release\.sh|smoke_package\.py|tests/.*)")


def git(*args, data=None):
    return subprocess.run(["git", "-C", str(repo), *args], input=data, capture_output=True, check=True).stdout


def commit_of(revision):
    try:
        return git("rev-parse", "--verify", "--quiet", revision + "^{commit}").decode().strip()
    except subprocess.CalledProcessError:
        return None


def verify_sources(name, revision, recorded):
    """A build without Git metadata (a Docker COPY of a worktree) records the short revision and dirty = null. Accept it
    only when every recorded source hash equals that commit's file and no source file of the commit is missing."""
    full = commit_of(revision)
    if full is None:
        raise SystemExit(f"assemble.sh: {name}: source revision {revision} is not in {repo}")
    blobs = {}
    for entry in git("ls-tree", "-r", "-z", full).decode().split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            if path in ROOT_FILES or path.startswith(tuple(d + "/" for d in SOURCE_DIRS)):
                blobs[path] = meta.split()[2]
    extra = [p for p in recorded if p not in blobs and not p.startswith(GENERATED)]
    missing = [p for p in blobs if p not in recorded]
    if extra or missing:
        raise SystemExit(f"assemble.sh: {name}: sources differ from {full[:12]}: "
                         f"not in the commit {extra[:5]}, not in the build {missing[:5]}")
    paths = sorted(p for p in recorded if p in blobs)
    out = git("cat-file", "--batch", data="".join(blobs[p] + "\n" for p in paths).encode())
    offset = 0
    for path in paths:
        header_end = out.index(b"\n", offset)
        size = int(out[offset:header_end].split()[2])
        content = out[header_end + 1:header_end + 1 + size]
        offset = header_end + 1 + size + 1
        if hashlib.sha256(content).hexdigest() != recorded[path]:
            raise SystemExit(f"assemble.sh: {name}: {path} does not match revision {full[:12]}")
    print(f"assemble.sh: {name}: {len(paths)} source files match {full}", file=sys.stderr)
    return full


def source_revision(name, metadata):
    revision = metadata.get("revision", "")
    if re.fullmatch(r"[0-9a-f]{40}", revision) and metadata.get("dirty") is False:
        return revision
    if re.fullmatch(r"[0-9a-f]{7,40}", revision) and metadata.get("dirty") is None and metadata.get("source_sha256"):
        return verify_sources(name, revision, metadata["source_sha256"])
    raise SystemExit("assemble.sh: candidate requires a clean, identified source revision")


linux, windows, output = sys.argv[1:]
out = Path(output)
if out.exists() and any(out.iterdir()):
    raise SystemExit("assemble.sh: output must be empty")
archives = []
revisions = set()
for directory, platform, ext in ((linux, "linux", "tar.gz"), (windows, "win", "zip")):
    if directory == "-":
        continue
    folder = Path(directory)
    program = f"nari-llama.cpp-{tag}-bin-{platform}-cuda-12.8-x64"
    runtime = f"cudart-nari-llama.cpp-{tag}-{platform}-cuda-12.8-x64"
    expected = {f"{program}.{ext}", f"{runtime}.{ext}"}
    sums = {}
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if not match or match[2] in sums:
            raise SystemExit("assemble.sh: invalid or duplicate checksum entry")
        sums[match[2]] = match[1]
    if set(sums) != expected:
        raise SystemExit(f"assemble.sh: expected both {tag} {platform} archives, got {sorted(sums)}")
    for name, sha in sums.items():
        path = folder / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != sha:
            raise SystemExit(f"assemble.sh: {name} does not match its SHA256SUMS")
        archives.append((path, sha))
    path = folder / f"{program}.{ext}"
    if platform == "linux":
        with tarfile.open(path) as archive:
            metadata = json.load(archive.extractfile(f"{program}/build-info.json"))
    else:
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("build-info.json").decode("utf-8-sig"))
    revisions.add(source_revision(path.name, metadata))
if len(revisions) > 1:
    first, second = sorted(revisions)
    try:
        changed = git("diff", "--name-only", first, second).decode().split()
    except subprocess.CalledProcessError:
        changed = None
    if changed is None or any(not NON_BUILD.fullmatch(p) for p in changed):
        shown = "unknown revisions" if changed is None else [p for p in changed if not NON_BUILD.fullmatch(p)][:5]
        raise SystemExit(f"assemble.sh: Linux and Windows archives have different source revisions ({shown})")
    print(f"assemble.sh: the archives' revisions {first[:12]} and {second[:12]} differ only in {changed}", file=sys.stderr)
out.mkdir(parents=True, exist_ok=True)
lines = []
for path, sha in archives:
    try:
        os.link(path, out / path.name)  # the archives are large and never modified in place
    except OSError:
        shutil.copy2(path, out / path.name)
    lines.append(f"{sha}  {path.name}")
for name in ("run.sh", "run.ps1") if windows != "-" else ("run.sh",):
    shutil.copy2(here / name, out / name)
    lines.append(f"{hashlib.sha256((out / name).read_bytes()).hexdigest()}  {name}")
(out / "SHA256SUMS").write_text("\n".join(lines) + "\n", newline="\n")
print("\n".join(lines))
