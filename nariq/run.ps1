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
#
# Options: -Budget, -Port (8080), -ListenHost (127.0.0.1), -NoVision, -DownloadOnly, -Extra (llama-server arguments).
# Files go to %LOCALAPPDATA%\nari-llama.cpp (about 13 GB with the 3.0 file); delete that folder to remove them.
# Environment: TRELLIS_HOME (another download folder); TRELLIS_RELEASE_URL, TRELLIS_MODEL_URL
# and TRELLIS_TAG (mirrors and tests); HF_TOKEN (sent to https://huggingface.co only, for gated or private model
# repositories). Needs Windows 10 1803 or newer (curl.exe) and an NVIDIA driver of 572 or newer.
# Without a GPU it runs on the CPU, slowly.
param(
    [ValidateSet("3.0", "2.5", "2.0", "3", "2", "3.0bpw", "2.5bpw", "2.0bpw", "3bit", "2.5bit", "2bit")][string]$Budget = "3.0",
    [int]$Port = 8080,
    [string]$ListenHost = "127.0.0.1",
    [switch]$NoVision,
    [switch]$DownloadOnly,
    [string[]]$Extra = @()
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
        & curl.exe -fL -# --retry 5 --retry-delay 3 -C - @auth -o "$Path.part" $Url
    } finally {
        if ($header) { Remove-Item -Force $header }
    }
    # resuming a part that is already complete fails (HTTP 416): keep it if it checks out
    if ($LASTEXITCODE -ne 0 -and -not (Test-Sha "$Path.part" $Sha)) {
        throw "could not download $Url (see the curl error above); run again to resume"
    }
    Fetch-Check $Path $Sha
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

$file, $sha, $size = $Files[$Budget]
Write-Host "files: $HomeDir"

$bin = Join-Path $HomeDir "bin\$Tag"
if (-not (Test-Path (Join-Path $bin "llama-server.exe"))) {
    $downloads = Join-Path $HomeDir "downloads"
    New-Item -ItemType Directory -Force -Path $downloads | Out-Null
    $sums = Join-Path $downloads "SHA256SUMS-$Tag"
    & curl.exe -fsSL --retry 5 -o $sums "$ReleaseUrl/SHA256SUMS"
    if ($LASTEXITCODE -ne 0) { throw "download of $ReleaseUrl/SHA256SUMS failed" }
    $part = "$bin.part"
    if (Test-Path $part) { Remove-Item -Recurse -Force $part }
    foreach ($archive in @("nari-llama.cpp-$Tag-bin-win-cuda-$Cuda-x64", "cudart-nari-llama.cpp-$Tag-win-cuda-$Cuda-x64")) {
        $line = Get-Content $sums | Where-Object { $_ -match "^([0-9a-f]{64})  $([regex]::Escape($archive)).zip$" }
        if (-not $line) { throw "$archive.zip is not in the release's SHA256SUMS" }
        Fetch "$ReleaseUrl/$archive.zip" (Join-Path $downloads "$archive.zip") $Matches[1]
        Expand-Archive -Force (Join-Path $downloads "$archive.zip") $part
    }
    # check that the binaries run here before the model download (llama-server --version needs no GPU)
    # (llama-server logs to stderr, which Windows PowerShell 5.1 turns into errors under "Stop")
    $ErrorActionPreference = "Continue"
    & (Join-Path $part "llama-server.exe") --version *> $null
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) { throw "the release binaries do not run on this system (llama-server --version exited with $code)" }
    if (Test-Path $bin) { Remove-Item -Recurse -Force $bin }
    Move-Item $part $bin
}
Fetch "$ModelUrl/$file" (Join-Path $HomeDir "models\$file") $sha $size
if (-not $NoVision) { Fetch "$ModelUrl/$($Mmproj[0])" (Join-Path $HomeDir "models\$($Mmproj[0])") $Mmproj[1] $Mmproj[2] }
if ($DownloadOnly) { exit 0 }

# the GGUFs carry the model's sampling settings except min_p, which llama-server would otherwise set to 0.05
$serverArgs = @("-m", (Join-Path $HomeDir "models\$file"), "--min-p", "0")
# the projector is loaded after the context is fitted to free memory: keep room for it and its image batch
if (-not $NoVision) { $serverArgs += @("--mmproj", (Join-Path $HomeDir "models\$($Mmproj[0])"), "--fit-target", "2560") }
Write-Host "starting llama-server ($Budget) on http://${ListenHost}:$Port"
& (Join-Path $bin "llama-server.exe") @serverArgs --host $ListenHost --port $Port @Extra
exit $LASTEXITCODE
