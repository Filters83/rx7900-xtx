import functools
import hashlib
import http.server
import io
import json
import zipfile
import os
from pathlib import Path
import platform
import re
import shutil
import ssl
import subprocess
import tarfile
import tempfile
import threading
import unittest

RELEASE = Path(__file__).resolve().parents[1]


def command(*args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


@unittest.skipUnless(platform.system() == "Linux" and platform.machine() == "x86_64", "Linux launcher")
class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = {**os.environ, "TRELLIS_HOME": str(self.root / "cache")}
        self.tag = (RELEASE / "TAG").read_text().strip()

    def test_invalid_arguments_do_not_download(self):
        for args in [("--bogus",), ("1.5",), ("--port",), ("--host",)]:
            result = command("bash", str(RELEASE / "run.sh"), *args, env=self.env)
            self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((self.root / "cache").exists())

    def test_unsupported_architecture_stops_before_download(self):
        mock = self.root / "mock"
        mock.mkdir()
        (mock / "uname").write_text('#!/bin/sh\nif [ "$1" = -s ]; then echo Linux; else echo aarch64; fi\n')
        (mock / "uname").chmod(0o755)
        self.env["PATH"] = str(mock) + os.pathsep + self.env["PATH"]
        result = command("bash", str(RELEASE / "run.sh"), env=self.env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Linux x86-64", result.stderr)
        self.assertFalse((self.root / "cache").exists())

    def archives(self, directory, platform_name, crlf=False, info=None):
        directory.mkdir()
        ext = "zip" if platform_name == "win" else "tar.gz"
        names = [f"nari-llama.cpp-{self.tag}-bin-{platform_name}-cuda-12.8-x64.{ext}",
                 f"cudart-nari-llama.cpp-{self.tag}-{platform_name}-cuda-12.8-x64.{ext}"]
        lines = []
        for name in names:
            data = json.dumps(info or {"revision": "a" * 40, "dirty": False}).encode()
            if ext == "zip":
                with zipfile.ZipFile(directory / name, "w") as archive:
                    archive.writestr("build-info.json", data)
            else:
                with tarfile.open(directory / name, "w:gz") as archive:
                    entry = tarfile.TarInfo(name.removesuffix(".tar.gz") + "/build-info.json")
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
            lines.append(f"{hashlib.sha256((directory / name).read_bytes()).hexdigest()}  {name}")
        (directory / "SHA256SUMS").write_bytes((("\r\n" if crlf else "\n").join(lines) + "\n").encode())
        return names

    def test_assemble_checks_inputs_and_combines_windows_checksums(self):
        linux, windows, out = [self.root / x for x in ("linux", "windows", "out")]
        names = self.archives(linux, "linux") + self.archives(windows, "win", crlf=True)
        result = command("bash", str(RELEASE / "assemble.sh"), "page", str(linux), str(windows), str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(b"\r", (out / "SHA256SUMS").read_bytes())
        self.assertEqual(command("sha256sum", "-c", "SHA256SUMS", cwd=out).returncode, 0)
        self.assertTrue((out / "run.ps1").is_file())
        (linux / names[0]).write_bytes(b"corrupt")
        bad = command("bash", str(RELEASE / "assemble.sh"), "page", str(linux), str(windows), str(self.root / "bad"))
        self.assertNotEqual(bad.returncode, 0)

    def test_linux_only_candidate_omits_windows_launcher(self):
        linux, out = self.root / "linux", self.root / "out"
        self.archives(linux, "linux")
        result = command("bash", str(RELEASE / "assemble.sh"), "page", str(linux), "-", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((out / "run.ps1").exists())

    def test_assembly_rejects_incomplete_and_mixed_revision_archives(self):
        linux, windows = self.root / "linux", self.root / "windows"
        self.archives(linux, "linux")
        names = self.archives(windows, "win")
        program = windows / names[0]
        with zipfile.ZipFile(program, "w") as archive:
            archive.writestr("build-info.json", json.dumps({"revision": "b" * 40, "dirty": False}))
        (windows / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((windows / n).read_bytes()).hexdigest()}  {n}\n" for n in names))
        mixed = command("bash", str(RELEASE / "assemble.sh"), "page", str(linux), str(windows), str(self.root / "out"))
        self.assertNotEqual(mixed.returncode, 0)
        self.assertIn("different source revisions", mixed.stderr)
        lines = (linux / "SHA256SUMS").read_text().splitlines()
        (linux / "SHA256SUMS").write_text(lines[0] + "\n")
        missing = command("bash", str(RELEASE / "assemble.sh"), "page", str(linux), "-", str(self.root / "out"))
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("expected both", missing.stderr)

    def source_repo(self):
        repo = self.root / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src/a.c").write_text("int a;\n")
        git = ("git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test", "-c", "commit.gpgsign=false")
        self.assertEqual(command("git", "init", "-q", str(repo)).returncode, 0)

        def commit():
            command(*git, "add", "-A")
            self.assertEqual(command(*git, "commit", "-qm", "c").returncode, 0)
            return command(*git, "rev-parse", "HEAD").stdout.strip()
        self.env["NARIQ_SOURCE_REPO"] = str(repo)
        return repo, commit

    def no_git_build(self, revision, sources):
        return {"revision": revision[:7], "dirty": None,
                "source_sha256": {p: hashlib.sha256(d).hexdigest() for p, d in sources.items()}}

    def test_assembly_checks_a_build_without_git_metadata_against_its_commit(self):
        repo, commit = self.source_repo()
        revision = commit()
        good = self.no_git_build(revision, {"src/a.c": b"int a;\n", "tools/ui/dist/index.html": b"ui"})
        self.archives(self.root / "linux", "linux", info=good)
        result = command("bash", str(RELEASE / "assemble.sh"), "page", str(self.root / "linux"), "-",
                         str(self.root / "out"), env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(revision, result.stderr)
        for case, sources in (("changed", {"src/a.c": b"int b;\n"}), ("extra", {"src/a.c": b"int a;\n", "src/b.c": b""}),
                              ("missing", {"tools/ui/dist/index.html": b"ui"})):
            self.archives(self.root / case, "linux", info=self.no_git_build(revision, sources))
            bad = command("bash", str(RELEASE / "assemble.sh"), "page", str(self.root / case), "-",
                          str(self.root / f"out-{case}"), env=self.env)
            self.assertNotEqual(bad.returncode, 0, case)

    def test_assembly_accepts_revisions_that_differ_only_outside_the_build(self):
        repo, commit = self.source_repo()
        first = commit()
        (repo / "nariq").mkdir()
        (repo / "nariq/PUBLISH.md").write_text("notes\n")
        (repo / "nariq/smoke_package.py").write_text("print()\n")
        (repo / ".github/workflows").mkdir(parents=True)
        (repo / ".github/workflows/windows.yml").write_text("on: workflow_dispatch\n")
        second = commit()
        (repo / "src/a.c").write_text("int c;\n")
        third = commit()
        self.archives(self.root / "linux", "linux", info=self.no_git_build(first, {"src/a.c": b"int a;\n"}))
        self.archives(self.root / "docs", "win", info={"revision": second, "dirty": False})
        result = command("bash", str(RELEASE / "assemble.sh"), "page", str(self.root / "linux"), str(self.root / "docs"),
                         str(self.root / "out"), env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.archives(self.root / "code", "win", info={"revision": third, "dirty": False})
        mixed = command("bash", str(RELEASE / "assemble.sh"), "page", str(self.root / "linux"), str(self.root / "code"),
                        str(self.root / "out2"), env=self.env)
        self.assertNotEqual(mixed.returncode, 0)
        self.assertIn("different source revisions", mixed.stderr)

    def test_stage_release_builds_the_upload_folder(self):
        self.archives(self.root / "linux", "linux")
        out = self.root / "stage"
        (out).mkdir()
        (out / "stale.txt").write_text("replaced as a whole")
        result = command("bash", str(RELEASE / "stage_release.sh"), str(self.root / "linux"), "-", str(out), env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         sorted([f"nari-llama.cpp-{self.tag}-bin-linux-cuda-12.8-x64.tar.gz",
                                 f"cudart-nari-llama.cpp-{self.tag}-linux-cuda-12.8-x64.tar.gz",
                                 "run.sh", "SHA256SUMS", "RELEASE_NOTES.md"]))
        notes = (out / "RELEASE_NOTES.md").read_text()
        self.assertNotIn("<!--", notes)
        self.assertNotIn("win-cuda", notes)
        self.archives(self.root / "windows", "win", crlf=True)
        result = command("bash", str(RELEASE / "stage_release.sh"), str(self.root / "linux"), str(self.root / "windows"),
                         str(out), env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((out / "run.ps1").is_file())
        self.assertIn("win-cuda", (out / "RELEASE_NOTES.md").read_text())
        self.assertEqual(command("sha256sum", "-c", "SHA256SUMS", cwd=out).returncode, 0)

    def test_launcher_download_extract_hash_and_forward_arguments(self):
        web = self.root / "web"
        web.mkdir()
        model = b"small test fixture, not model weights"
        sha = hashlib.sha256(model).hexdigest()
        script = (RELEASE / "run.sh").read_text()
        script = re.sub(r"[a-f0-9]{64}", sha, script)
        launcher = self.root / "run.sh"
        launcher.write_text(script)
        (web / "Qwen3.8-27B-NariQ-3.0bpw.gguf").write_bytes(model)
        sums = []
        for prefix in (f"nari-llama.cpp-{self.tag}-bin", f"cudart-nari-llama.cpp-{self.tag}"):
            name = f"{prefix}-linux-cuda-12.8-x64"
            path = web / f"{name}.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                data = b'#!/bin/sh\nprintf "%s\\n" "$@"\n'
                entry = tarfile.TarInfo(f"{name}/" + ("llama-server" if prefix.endswith("-bin") else "runtime.txt"))
                entry.size = len(data)
                entry.mode = 0o755
                archive.addfile(entry, io.BytesIO(data))
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        (web / "SHA256SUMS").write_text("\n".join(sums) + "\n")

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(web)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}"
        self.env.update(TRELLIS_RELEASE_URL=url, TRELLIS_MODEL_URL=url)
        result = command("bash", str(launcher), "3bit", "--no-vision", "--", "-t", "2", "--alias", "test model", env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--alias\ntest model\n", result.stdout)
        self.assertEqual(result.stdout.splitlines().count("-t"), 1)
        self.assertEqual((self.root / "cache/models/Qwen3.8-27B-NariQ-3.0bpw.gguf").read_bytes(), model)
        shutil.rmtree(self.root / "cache")
        (web / "Qwen3.8-27B-NariQ-3.0bpw.gguf").write_bytes(b"corrupt")
        bad = command("bash", str(launcher), "--no-vision", env=self.env)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("does not match its sha256", bad.stderr)

    def fixture_launcher(self, web, files):
        """run.sh with every pinned sha256 replaced by the fixture's, a folder with the fixture files and the two
        archives (a stand-in llama-server that prints its arguments), and a stand-in nvidia-smi on PATH."""
        model = b"small test fixture, not model weights"
        sha = hashlib.sha256(model).hexdigest()
        launcher = self.root / "run.sh"
        launcher.write_text(re.sub(r"[a-f0-9]{64}", sha, (RELEASE / "run.sh").read_text()))
        web.mkdir(parents=True)
        for name in files:
            (web / name).write_bytes(model)
        sums = []
        for prefix in (f"nari-llama.cpp-{self.tag}-bin", f"cudart-nari-llama.cpp-{self.tag}"):
            name = f"{prefix}-linux-cuda-12.8-x64"
            path = web / f"{name}.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                data = b'#!/bin/sh\nprintf "%s\\n" "$@"\n'
                entry = tarfile.TarInfo(f"{name}/" + ("llama-server" if prefix.endswith("-bin") else "runtime.txt"))
                entry.size = len(data)
                entry.mode = 0o755
                archive.addfile(entry, io.BytesIO(data))
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        (web / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        mock = self.root / "mock"
        mock.mkdir(exist_ok=True)
        (mock / "nvidia-smi").write_text("#!/bin/sh\necho 'NVIDIA Test GPU, 580.65.06, 16384 MiB'\n")
        (mock / "nvidia-smi").chmod(0o755)
        self.env["PATH"] = str(mock) + os.pathsep + self.env["PATH"]
        return launcher, model

    def serve(self, directory, tls=None):
        """A local HTTP(S) server over `directory`; returns (port, the headers of each request)."""
        seen = []

        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                seen.append((self.path, dict(self.headers)))
                if self.path.startswith("/redirect/"):  # as huggingface.co sends file downloads to another host
                    self.send_response(302)
                    self.send_header("Location", self.server.redirect + self.path.removeprefix("/redirect/"))
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                super().do_GET()

            def log_message(self, *_):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(directory)))
        if tls:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(*tls)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, seen

    def test_docker_mode_runs_the_image_with_read_only_models(self):
        web = self.root / "web"
        names = ["Qwen3.8-27B-NariQ-3.0bpw.gguf", "Qwen3.8-27B-NariQ-mmproj-BF16.gguf"]
        launcher, model = self.fixture_launcher(web, names)
        server, _ = self.serve(web)
        calls = self.root / "docker-calls"
        (self.root / "mock" / "docker").write_text(
            '#!/bin/sh\n{ for a in "$@"; do printf "%s\\0" "$a"; done; printf "\\n"; } >> "$DOCKER_CALLS"\n')
        (self.root / "mock" / "docker").chmod(0o755)
        image = "registry.invalid/nari-llama.cpp:test"
        self.env.update(TRELLIS_MODEL_URL=f"http://127.0.0.1:{server.server_port}", TRELLIS_IMAGE=image,
                        DOCKER_CALLS=str(calls))
        result = command("bash", str(launcher), "3.0", "--docker", "--port", "9000", "--",
                         "-t", "2", "--alias", "test model", env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        home = self.root / "cache"
        self.assertEqual(calls.read_text().splitlines()[0].split("\0")[:-1], [
            "run", "--rm", "--gpus", "all", "-p", "127.0.0.1:9000:8080", "-v", f"{home}/models:/models:ro", image,
            "-m", "/models/Qwen3.8-27B-NariQ-3.0bpw.gguf",
            "--mmproj", "/models/Qwen3.8-27B-NariQ-mmproj-BF16.gguf", "--fit-target", "2560",
            "--host", "0.0.0.0", "--port", "8080", "-t", "2", "--alias", "test model"])
        for name in names:
            self.assertEqual((home / "models" / name).read_bytes(), model)
        self.assertFalse((home / "bin").exists())  # the image brings its own binaries
        calls.unlink()
        result = command("bash", str(launcher), "2.5", "--docker", "--download-only", env=self.env)
        self.assertNotEqual(result.returncode, 0)  # the fixture has no 2.5 file: nothing is pulled before it
        self.assertFalse(calls.exists())
        result = command("bash", str(launcher), "--docker", "--download-only", env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls.read_text().splitlines()[0].split("\0")[:-1], ["pull", image])

    @unittest.skipUnless(shutil.which("openssl"), "needs openssl for the test certificate")
    def test_hf_token_goes_to_huggingface_only(self):
        web, hf = self.root / "web", self.root / "hf"
        launcher, model = self.fixture_launcher(web, ["Qwen3.8-27B-NariQ-3.0bpw.gguf"])
        hf.mkdir()
        cert, key = self.root / "cert.pem", self.root / "key.pem"
        made = command("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj",
                       "/CN=huggingface.co", "-addext", "subjectAltName=DNS:huggingface.co",
                       "-keyout", str(key), "-out", str(cert))
        self.assertEqual(made.returncode, 0, made.stderr)
        files, files_seen = self.serve(web)
        hub, hub_seen = self.serve(hf, tls=(str(cert), str(key)))
        hub.redirect = f"http://127.0.0.1:{files.server_port}/"
        # curl's own configuration routes huggingface.co:443 to the local TLS server and trusts its certificate
        (self.root / ".curlrc").write_text(f'connect-to = "huggingface.co:443:127.0.0.1:{hub.server_port}"\n'
                                           f'cacert = "{cert}"\n')
        token = "hf_test_token_not_a_secret"
        self.env.update(CURL_HOME=str(self.root), HF_TOKEN=token,
                        TRELLIS_MODEL_URL="https://huggingface.co/redirect",
                        TRELLIS_RELEASE_URL=f"http://127.0.0.1:{files.server_port}")
        result = command("bash", str(launcher), "3.0", "--no-vision", env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "cache/models/Qwen3.8-27B-NariQ-3.0bpw.gguf").read_bytes(), model)
        self.assertNotIn(token, result.stdout + result.stderr)
        self.assertEqual([h.get("Authorization") for p, h in hub_seen],
                         [f"Bearer {token}"])
        self.assertEqual({p for p, _ in files_seen} >= {"/Qwen3.8-27B-NariQ-3.0bpw.gguf", "/SHA256SUMS"}, True)
        self.assertEqual([p for p, h in files_seen if "Authorization" in h], [])
        shutil.rmtree(self.root / "cache")
        hub_seen.clear()
        del self.env["HF_TOKEN"]
        result = command("bash", str(launcher), "3.0", "--no-vision", "--download-only", env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([h.get("Authorization") for p, h in hub_seen], [None])


class CrossPlatformTests(unittest.TestCase):
    def test_launchers_pin_identical_files_and_tag(self):
        sh, ps = [(RELEASE / name).read_text() for name in ("run.sh", "run.ps1")]
        self.assertEqual(set(re.findall(r"[a-f0-9]{64}", sh)), set(re.findall(r"[a-f0-9]{64}", ps)))
        tag = (RELEASE / "TAG").read_text().strip()
        self.assertIn(tag, sh)
        self.assertIn(tag, ps)


if __name__ == "__main__":
    unittest.main()
