# Run Qwen3.8-27B NariQ on an NVIDIA GPU with one command (Windows x86-64, PowerShell 5.1 or 7; experimental): the
# Windows twin of run.sh. Downloads the nari-llama.cpp binaries and CUDA runtime, the GGUF and the vision projector
# once, checks each file's sha256, and starts llama-server with an OpenAI-compatible API and a chat page at
# http://127.0.0.1:8080.
#
#   powershell -ExecutionPolicy Bypass -File run.ps1                 # the 3.0 file, with vision, the longest context
#   powershell -ExecutionPolicy Bypass -File run.ps1 -Budget 2.5            # or 2.0; 3, 3.0bpw, 3bit and the like too
#   powershell -ExecutionPolicy Bypass -File run.ps1 -Port 9000 -ListenHost 0.0.0.0   # another port, all interfaces
#   powershell -ExecutionPolicy Bypass -Command "& .\run.ps1 -NoVision -Extra '-np','4','-c','131072'"
#   (with -File, PowerShell passes a list as one string, so -Extra needs -Command)
#   powershell -ExecutionPolicy Bypass -File run.ps1 -Budget 2.5 -Wsl        # the Linux build in WSL2 (run.sh)
#
# Options: -Budget, -Port (8080), -ListenHost (127.0.0.1), -NoVision, -DownloadOnly, -Extra (llama-server arguments),
# -Wsl (run the release's run.sh in WSL2 instead of the Windows build; -Distro picks the WSL distribution).
# Files go to %LOCALAPPDATA%\nari-llama.cpp (about 13 GB with the 3.0 file); delete that folder to remove them.
# With -Wsl, the model and binaries go to ~/.cache/nari-llama.cpp inside WSL.
# Environment: TRELLIS_HOME (another download folder); TRELLIS_RELEASE_URL, TRELLIS_MODEL_URL
# and TRELLIS_TAG (mirrors and tests); HF_TOKEN (sent to https://huggingface.co only, for gated or private model
# repositories); TRELLIS_NO_HF=1 (download the model files with curl even when the hf CLI is installed; hf is
# faster). Needs Windows 10 1803 or newer (curl.exe) and an NVIDIA driver of 572 or newer.
# Without a GPU it runs on the CPU, slowly.
param(
    [ValidateSet("3.0", "2.5", "2.0", "3", "2", "3.0bpw", "2.5bpw", "2.0bpw", "3bit", "2.5bit", "2bit")][string]$Budget = "3.0",
    [int]$Port = 8080,
    [string]$ListenHost = "127.0.0.1",
    [switch]$NoVision,
    [switch]$DownloadOnly,
    [string[]]$Extra = @(),
    [switch]$Wsl,
    [string]$Distro = ""
)
$ErrorActionPreference = "Stop"
$Aliases = @{ "3" = "3.0"; "2" = "2.0"; "3.0bpw" = "3.0"; "2.5bpw" = "2.5"; "2.0bpw" = "2.0"; "3bit" = "3.0"; "2.5bit" = "2.5"; "2bit" = "2.0" }
if ($Aliases.ContainsKey($Budget)) { $Budget = $Aliases[$Budget] }

$Tag = if ($env:TRELLIS_TAG) { $env:TRELLIS_TAG } else { "b12431-nariv0" }  # the release tag
$ReleaseUrl = if ($env:TRELLIS_RELEASE_URL) { $env:TRELLIS_RELEASE_URL } else { "https://github.com/nari-labs/nari-llama.cpp/releases/download/$Tag" }
$ModelUrl = if ($env:TRELLIS_MODEL_URL) { $env:TRELLIS_MODEL_URL } else { "https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF/resolve/main" }
$HomeDir = if ($env:TRELLIS_HOME) { $env:TRELLIS_HOME } else { Join-Path $env:LOCALAPPDATA "nari-llama.cpp" }
$Cuda = "12.8"

# file, sha256 and size in bytes per budget; the layout of these files is frozen
$Files = @{
    "3.0" = @("Qwen3.8-27B-NariQ-3.0bpw.gguf", "1eaad934aa05579b8eba47ed9b26450392f9875fc2354efe8cfa8b29be8f4419", 10123626592)
    "2.5" = @("Qwen3.8-27B-NariQ-2.5bpw.gguf", "2445372e6de20620b940fbfab53aa718af9ba741f202da30cd2f02b5df78d05c", 8443119712)
    "2.0" = @("Qwen3.8-27B-NariQ-2.0bpw.gguf", "3a9e4955329dba8ba0be5509a9552220e5ccdae2f66bcc72c21c4580cd49bbb2", 6753954304)
}
$Mmproj = @("Qwen3.8-27B-NariQ-mmproj-BF16.gguf", "8832425f593068d3278e08eda4c1591072c7a2d8dc90f21e4f86394e4b2eb002", 931145760)

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$gpu = $null
if ($smi) {
    try {
        # Capture first: a pipeline into Select-Object leaves $LASTEXITCODE unset or stale.
        $out = @(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null)
        if ($LASTEXITCODE -eq 0 -and $out.Count -gt 0) { $gpu = $out[0] }
    } catch {
        # Windows PowerShell 5.1 can turn native stderr into a terminating error.
        $gpu = $null
    }
}
if ($gpu -and $gpu -match '^.+,\s*(\d+)\.\d+.*,') {
    Write-Host "GPU: $gpu"
    $driver = [int]$Matches[1]
    if ($driver -lt 572) { throw "NVIDIA driver $driver is too old for CUDA $Cuda; install 572 or newer" }
} else {
    Write-Warning "no NVIDIA GPU found; llama-server will run on the CPU, which is slow for a 27B model"
}

function Test-Sha([string]$File, [string]$Sha) {
    (Test-Path $File) -and ((Get-FileHash $File -Algorithm SHA256).Hash.ToLower() -eq $Sha)
}

function Fetch([string]$Url, [string]$Path, [string]$Sha, [long]$Size = 0) {
    if ((Test-Path $Path) -and (Test-Path "$Path.ok")) { return }
    New-Item -ItemType Directory -Force -Path (Split-Path $Path) | Out-Null
    $have = if (Test-Path "$Path.part") { (Get-Item "$Path.part").Length } else { 0 }
    if ($Size -gt 0 -and $have -gt $Size) { Remove-Item "$Path.part"; $have = 0 }
    if ($Size -gt 0 -and $have -eq $Size) { Fetch-Check $Path $Sha; return }  # finished before, never checked
    Write-Host "downloading $(Split-Path $Path -Leaf)"
    # HF_TOKEN goes to Hugging Face only, from a file so it is not on a command line; curl does not send it on to the
    # hosts a download redirects to
    $auth = @(); $header = $null
    if ($env:HF_TOKEN -and ($Url -like "https://huggingface.co/*" -or $Url -like "https://hf.co/*")) {
        $header = New-TemporaryFile
        Set-Content -Path $header -Value "Authorization: Bearer $env:HF_TOKEN" -NoNewline
        $auth = @("-H", "@$($header.FullName)")
    }
    try {
        for ($attempt = 1; $attempt -le 5; $attempt++) {
            # a transfer slower than 100 KB/s for 60 s fails, so a stalled connection is resumed on a new one
            $code = & curl.exe -fL -# --retry 5 --retry-delay 3 --speed-limit 102400 --speed-time 60 -C - @auth -o "$Path.part" -w "%{http_code}" $Url
            if ($LASTEXITCODE -eq 0) { break }
            # resuming a part that is already complete fails (HTTP 416): keep it if it checks out
            if ((Test-Path "$Path.part") -and ($Size -eq 0 -or (Get-Item "$Path.part").Length -eq $Size) -and (Test-Sha "$Path.part" $Sha)) { break }
            if ($code -in "401", "403") { throw "$Url refused access (HTTP $code); the repository may be private or gated. Set HF_TOKEN to a token that can read it" }
            if ($code -eq "404") { throw "$Url was not found (HTTP 404)" }
            if ($attempt -eq 5) { throw "could not download $Url (see the curl error above); run again to resume" }
            Write-Host "the download of $(Split-Path $Path -Leaf) stopped; resuming"
        }
    } finally {
        if ($header) { Remove-Item -Force $header }
    }
    Fetch-Check $Path $Sha
}

function Fetch-Hf([string]$File, [string]$Sha) {
    # hf downloads in parallel, often much faster than one curl stream; on any failure curl takes over
    if ($env:TRELLIS_NO_HF -or -not (Get-Command hf -ErrorAction SilentlyContinue)) { return }
    $m = [regex]::Match($ModelUrl, '^https://(huggingface\.co|hf\.co)/([^/]+/[^/]+)/resolve/([^/]+)$')
    if (-not $m.Success) { return }
    $models = Join-Path $HomeDir "models"
    if (Test-Path (Join-Path $models "$File.ok")) { return }
    $staging = Join-Path $models ".hf"
    Write-Host "downloading $File with hf"
    & hf download $m.Groups[2].Value $File --revision $m.Groups[3].Value --local-dir $staging | Out-Null
    if ($LASTEXITCODE -eq 0 -and (Test-Path (Join-Path $staging $File))) {
        Move-Item -Force (Join-Path $staging $File) (Join-Path $models "$File.part")
        try { Fetch-Check (Join-Path $models $File) $Sha } catch { Write-Warning "$File from hf does not match its sha256; using curl" }
    } else {
        Write-Warning "hf could not download $File; using curl"
    }
}

function Get-ReleaseSha([string]$Name) {
    # the sha256 of a release file, from the release's SHA256SUMS
    $downloads = Join-Path $HomeDir "downloads"
    New-Item -ItemType Directory -Force -Path $downloads | Out-Null
    $sums = Join-Path $downloads "SHA256SUMS-$Tag"
    & curl.exe -fsSL --retry 5 -o $sums "$ReleaseUrl/SHA256SUMS"
    if ($LASTEXITCODE -ne 0) { throw "download of $ReleaseUrl/SHA256SUMS failed" }
    $line = Get-Content $sums | Where-Object { $_ -match "^[0-9a-f]{64}  $([regex]::Escape($Name))$" } | Select-Object -First 1
    if (-not $line) { throw "$Name is not in the release's SHA256SUMS" }
    $line.Substring(0, 64)
}

function Fetch-Check([string]$Path, [string]$Sha) {
    Write-Host "checking $(Split-Path $Path -Leaf)"
    if (-not (Test-Sha "$Path.part" $Sha)) {
        Remove-Item "$Path.part"
        throw "$(Split-Path $Path -Leaf) does not match its sha256; deleted, run again to download it again"
    }
    Move-Item -Force "$Path.part" $Path
    New-Item -ItemType File -Force -Path "$Path.ok" | Out-Null
}

if ($Wsl) {
    # the Linux build in WSL2: it is the tested build, and Smart App Control does not apply to it
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { throw "WSL is not installed; install it with: wsl --install" }
    $distroArgs = if ($Distro) { @("-d", $Distro) } else { @() }
    $kernel = & wsl.exe @distroArgs -e uname -r
    if ($LASTEXITCODE -ne 0) { throw "could not start WSL (see the error above); set up a Linux distribution with: wsl --install" }
    if ($kernel -notmatch "microsoft-standard") { throw "this WSL distribution runs on WSL 1, which has no GPU access; convert it with: wsl --set-version <name> 2" }
    $runSh = Join-Path $HomeDir "downloads\run-$Tag.sh"
    Fetch "$ReleaseUrl/run.sh" $runSh (Get-ReleaseSha "run.sh")
    $wslRunSh = (& wsl.exe @distroArgs -e wslpath -a $runSh | Out-String).Trim()
    $shArgs = @($Budget, "--port", $Port, "--host", $ListenHost)
    if ($NoVision) { $shArgs += "--no-vision" }
    if ($DownloadOnly) { $shArgs += "--download-only" }
    if ($Extra.Count -gt 0) { $shArgs += @("--") + $Extra }
    # WSLENV lists the Windows variables that WSL passes on; TRELLIS_HOME is a Windows path, so it stays here
    $pass = @("HF_TOKEN", "TRELLIS_TAG", "TRELLIS_RELEASE_URL", "TRELLIS_MODEL_URL", "TRELLIS_NO_HF") | Where-Object { Test-Path "env:$_" }
    if ($pass) { $env:WSLENV = (@($env:WSLENV) + $pass | Where-Object { $_ }) -join ":" }
    Write-Host "running run.sh in WSL2; its files go to ~/.cache/nari-llama.cpp inside WSL"
    & wsl.exe @distroArgs -e bash $wslRunSh @shArgs
    exit $LASTEXITCODE
}

$file, $sha, $size = $Files[$Budget]
Write-Host "files: $HomeDir"

$bin = Join-Path $HomeDir "bin\$Tag"
if (-not (Test-Path (Join-Path $bin "llama-server.exe"))) {
    $downloads = Join-Path $HomeDir "downloads"
    $part = "$bin.part"
    if (Test-Path $part) { Remove-Item -Recurse -Force $part }
    foreach ($archive in @("nari-llama.cpp-$Tag-bin-win-cuda-$Cuda-x64", "cudart-nari-llama.cpp-$Tag-win-cuda-$Cuda-x64")) {
        Fetch "$ReleaseUrl/$archive.zip" (Join-Path $downloads "$archive.zip") (Get-ReleaseSha "$archive.zip")
        Expand-Archive -Force (Join-Path $downloads "$archive.zip") $part
    }
    # check that the binaries run here before the model download (llama-server --version needs no GPU)
    # (llama-server logs to stderr, which Windows PowerShell 5.1 turns into errors under "Stop")
    $ErrorActionPreference = "Continue"
    & (Join-Path $part "llama-server.exe") --version *> $null
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    # -1058471934 is 0xC0E90002: Code Integrity blocked an unsigned DLL
    if ($code -eq -1058471934) { throw "Windows Smart App Control blocked the release binaries (they are not code-signed). Run the Linux build in WSL2 instead: powershell -ExecutionPolicy Bypass -File run.ps1 -Budget $Budget -Wsl" }
    if ($code -ne 0) { throw "the release binaries do not run on this system (llama-server --version exited with $code)" }
    if (Test-Path $bin) { Remove-Item -Recurse -Force $bin }
    Move-Item $part $bin
}
Fetch-Hf $file $sha
Fetch "$ModelUrl/$file" (Join-Path $HomeDir "models\$file") $sha $size
if (-not $NoVision) {
    Fetch-Hf $Mmproj[0] $Mmproj[1]
    Fetch "$ModelUrl/$($Mmproj[0])" (Join-Path $HomeDir "models\$($Mmproj[0])") $Mmproj[1] $Mmproj[2]
}
$staging = Join-Path $HomeDir "models\.hf"
if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
if ($DownloadOnly) { exit 0 }

# the GGUFs carry the model's sampling settings except min_p, which llama-server would otherwise set to 0.05
$serverArgs = @("-m", (Join-Path $HomeDir "models\$file"), "--min-p", "0")
# the projector is loaded after the context is fitted to free memory: keep room for it and its image batch
if (-not $NoVision) { $serverArgs += @("--mmproj", (Join-Path $HomeDir "models\$($Mmproj[0])"), "--fit-target", "2560") }
Write-Host "starting llama-server ($Budget) on http://${ListenHost}:$Port"
& (Join-Path $bin "llama-server.exe") @serverArgs --host $ListenHost --port $Port @Extra
exit $LASTEXITCODE
