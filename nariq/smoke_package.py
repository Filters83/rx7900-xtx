"""Check and extract final archives, then run model-free CPU startup checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

folder = Path(sys.argv[1]).resolve()
windows = sys.platform == "win32"


def diagnose_cpu_backends(bench, env):
    """Print why no CPU variant loads: the CPU, each variant's own load error and llama-bench with it forced."""
    if windows:
        query = "Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors"
        subprocess.run(["powershell", "-NoProfile", "-Command", query], timeout=60)
    else:
        print(next((line for line in Path("/proc/cpuinfo").read_text().splitlines() if "model name" in line), ""),
              flush=True)
    print("package:", " ".join(sorted(p.name for p in bench.parent.iterdir())), flush=True)
    for lib in sorted(bench.parent.glob("*ggml-cpu*")):
        # a child process, so that no library stays loaded from the folder that is removed afterwards
        load = subprocess.run([sys.executable, "-I", "-c", "import ctypes, sys; ctypes.CDLL(sys.argv[1])", str(lib)],
                              env=env, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        forced = subprocess.run([str(bench), "--list-devices"], env={**env, "GGML_BACKEND_PATH": str(lib)},
                                timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        status = "loads" if load.returncode == 0 else "does not load: " + load.stdout.strip().splitlines()[-1:][0]
        print(f"{lib.name}: {status}; forced with GGML_BACKEND_PATH: exit {forced.returncode}", flush=True)
        print("    " + forced.stdout.strip().replace("\n", "\n    "), flush=True)


suffix = ".zip" if windows else ".tar.gz"
paths = sorted([*folder.glob("nari-llama.cpp-*" + suffix), *folder.glob("cudart-nari-llama.cpp-*" + suffix)])
if len(paths) != 2:
    raise SystemExit("Expected a program archive and a CUDA runtime archive")
sums = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in (folder / "SHA256SUMS").read_text().splitlines()}
with tempfile.TemporaryDirectory() as tmp:
    dest = Path(tmp)
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if sums.get(path.name) != digest.hexdigest():
            raise SystemExit(f"Checksum mismatch: {path.name}")
        if windows:
            with zipfile.ZipFile(path) as archive:
                archive.extractall(dest)
        else:
            with tarfile.open(path) as archive:
                archive.extractall(dest)
    executable = "llama-server.exe" if windows else "llama-server"
    binaries = list(dest.rglob(executable))
    if len(binaries) != 1:
        raise SystemExit("Missing or ambiguous server in package")
    binary = binaries[0]
    info = json.loads((binary.parent / "build-info.json").read_text(encoding="utf-8-sig"))
    if "tools/ui/dist/index.html" not in info["source_sha256"]:
        raise SystemExit("Package does not record pinned UI assets")
    runtime_dirs = [str(p.parent) for p in dest.rglob("libcudart.so.12")]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": os.pathsep.join([str(binary.parent), *runtime_dirs])}
    subprocess.run([str(binary), "--version"], env=env, check=True, timeout=60)
    # llama-bench loads the backends before it installs a logger, so ggml's own logger prints "loaded CPU backend"
    # synchronously. llama-cli logs through common_log's worker thread, which Windows ends at exit(0) mid-line
    # (common/log.cpp leaks the logger on purpose, upstream issue 22142): its message may never arrive.
    bench = binary.parent / ("llama-bench.exe" if windows else "llama-bench")
    devices = subprocess.run([str(bench), "--list-devices"], env=env, check=True,
                             timeout=60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(devices.stdout, end="")
    if "loaded CPU backend" not in devices.stdout:
        diagnose_cpu_backends(bench, env)
        raise SystemExit("Packaged CPU backend could not be loaded")
    if not list(binary.parent.glob("*ggml-cpu*")):
        raise SystemExit("Package contains no CPU backend")
print("Archive checksums, extraction, provenance, UI and CPU startup checks passed")
