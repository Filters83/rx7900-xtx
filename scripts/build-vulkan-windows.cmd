@echo off
rem Windows build (MSVC + Vulkan SDK) of the Vulkan backend. See docs/rdna3/WINDOWS.md.
rem Requires Visual Studio with the C++ toolchain and the Vulkan SDK (VULKAN_SDK set by its installer).
rem Set VCVARS if Visual Studio is installed elsewhere. Set LLAMA_BUILD_UI=ON to build the Web UI with Node.js.
setlocal
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo vcvars64.bat not found: set VCVARS to its path
    exit /b 1
)
if not defined VULKAN_SDK (
    echo VULKAN_SDK is not set: install the Vulkan SDK
    exit /b 1
)
call "%VCVARS%" >nul || exit /b 1
set "PATH=%VULKAN_SDK%\Bin;%PATH%"
if not defined LLAMA_BUILD_UI set "LLAMA_BUILD_UI=OFF"
cd /d "%~dp0.." || exit /b 1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release ^
    -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF -DGGML_AVX_VNNI=OFF ^
    -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF ^
    -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=%LLAMA_BUILD_UI% || exit /b 2
cmake --build build --config Release --target llama-server llama-bench llama-perplexity test-backend-ops test-trellis test-trellis-metadata test-trellis-model || exit /b 3
echo BUILD_OK
