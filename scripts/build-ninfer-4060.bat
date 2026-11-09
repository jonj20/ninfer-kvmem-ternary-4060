@echo off
setlocal enabledelayedexpansion
rem Build NInfer for the RTX 4060 (sm_89) on Windows with MSVC + CUDA + Ninja.
rem Produces build-backends\ninfer\apps\ninfer-serve.exe.
rem The configure step runs on every build, so the coverage profile below is always the one the
rem build directory carries; only the compile step is incremental. Pass -clean to rebuild from
rem scratch.
rem ASCII-only comments on purpose: cmd.exe reads .bat with the OEM codepage
rem (936 here), so UTF-8 non-ASCII comments corrupt the parser.
rem
rem Usage: scripts\build-ninfer-4060.bat [-clean] [-jobs N] [-target T]
rem                                       [-cudaPath PATH] [-vcpkgTree PATH] [-buildDir DIR]
rem                                       [-configureOnly] [-buildOnly]
rem   -clean        delete the build directory first (full rebuild)
rem   -jobs N       parallel compile jobs (default 8)
rem   -target T     build target (default ninfer-serve)
rem   -cudaPath     CUDA toolkit dir (default D:\dev\nvkit)
rem   -vcpkgTree    vcpkg installed triplet dir (default D:\dev\vcpkg\installed\x64-windows)
rem   -buildDir     build directory (default build-backends\ninfer)
rem   -configureOnly only run cmake configure, then stop
rem   -buildOnly     only run cmake --build, skip configure
rem
rem RTX 4060 coverage profile (sm_89 only): the KV cache codings and head geometry this card and
rem the launcher (scripts\start-ninfer-4060.bat) actually run:
rem   codings   : bf16 and int8 (always compiled) + rk8v4, rk4v4, rk4v4-e8 and nvfp4, the
rem               launcher's choices. rk8v4 is the fastest of them, rk4v4/rk4v4-e8 keep rk8v4's
rem               values with 4-bit keys (smaller, and better perplexity than nvfp4), nvfp4 is the
rem               smallest and the slowest.
rem   geometries: h24 (24 query heads over 4 KV heads, head_dim 256). The Bonsai 2 27B artifact
rem               has 24 query heads over 4 KV heads, so h16 and g4 are unreachable for it; those
rem               units are the largest single lever on the binary's size and on compile time.
rem rk2v4-e8, fp8 and k8v4 are not compiled.
rem The server refuses a --kv-dtype this build omits, by name, at startup, so a launcher that asks
rem for one of them fails immediately instead of inside the first generation. Turn a switch back
rem ON here (and rebuild) to serve that coding; the server then accepts it again.

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "BLD=%ROOT%\build-backends\ninfer"
set "VCVARS=D:\dev\VS2022\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA=D:\dev\nvkit"
set "VCPKG=D:\dev\vcpkg\installed\x64-windows"
set "JOBS=8"
set "TARGET=ninfer-serve"
set "CLEAN="
set "CFG_ONLY="
set "BLD_ONLY="

:parseargs
if "%~1"=="" goto parsed
if /i "%~1"=="-clean" (set "CLEAN=1" & shift & goto parseargs)
if /i "%~1"=="-jobs" (set "JOBS=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-target" (set "TARGET=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-cudaPath" (set "CUDA=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-vcpkgTree" (set "VCPKG=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-buildDir" (set "BLD=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-configureOnly" (set "CFG_ONLY=1" & shift & goto parseargs)
if /i "%~1"=="-buildOnly" (set "BLD_ONLY=1" & shift & goto parseargs)
echo Unknown switch: %~1
exit /b 2
:parsed

if not exist "%ROOT%\CMakeLists.txt" (
  echo ERROR: NInfer source root not found at %ROOT%
  exit /b 1
)
if not exist "%VCVARS%" (
  echo ERROR: vcvars64.bat not found at %VCVARS%
  echo Edit VCVARS in this script to point at your Visual Studio install.
  exit /b 1
)
if not exist "%CUDA%\bin\nvcc.exe" (
  echo ERROR: nvcc not found at %CUDA%\bin\nvcc.exe
  echo Edit CUDA in this script to point at your CUDA install.
  exit /b 1
)

if defined CLEAN (
  echo Removing %BLD% ...
  if exist "%BLD%" rmdir /s /q "%BLD%"
)

call "%VCVARS%" >nul || exit /b 1
set "PATH=D:\dev\python\Scripts;%CUDA%\bin;%CUDA%\bin\x64;%VCPKG%\bin;%PATH%"

rem A build directory configured for another checkout cannot be reused.
if exist "%BLD%\CMakeCache.txt" (
  for /f "tokens=2 delims==" %%H in ('findstr /b /c:"CMAKE_HOME_DIRECTORY:INTERNAL=" "%BLD%\CMakeCache.txt"') do set "CACHED_SRC=%%H"
  if defined CACHED_SRC (
    set "CACHED_SRC=!CACHED_SRC:/=\!"
    if /i not "!CACHED_SRC!"=="%ROOT%" (
      echo Build directory was configured for another source tree; recreating it.
      rmdir /s /q "%BLD%"
    )
  )
)

if not defined BLD_ONLY (
  echo === configuring %BLD% ===
  cmake -S "%ROOT%" -B "%BLD%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES=89 ^
    -DCUDAToolkit_ROOT="%CUDA%" ^
    -DCMAKE_CUDA_COMPILER="%CUDA%\bin\nvcc.exe" ^
    -DNINFER_KVMEM_SOURCE_DIR="%ROOT%\kvmem" ^
    -DNINFER_VCPKG_TREE="%VCPKG%" ^
    -DNINFER_BUILD_APPS=ON ^
    -DNINFER_BUILD_BENCHMARKS=OFF ^
    -DNINFER_CUDA_PRUNE_PTX=ON ^
    -DNINFER_KV_ATTN_CODING_RK8V4=ON ^
    -DNINFER_KV_ATTN_CODING_RK4V4=ON ^
    -DNINFER_KV_ATTN_CODING_RK4V4E8=ON ^
    -DNINFER_KV_ATTN_CODING_RK2V4E8=OFF ^
    -DNINFER_KV_ATTN_FP8=OFF ^
    -DNINFER_KV_ATTN_NVFP4=ON ^
    -DNINFER_KV_ATTN_K8V4=OFF ^
    -DNINFER_KV_ATTN_GEOM_H24=ON ^
    -DNINFER_KV_ATTN_GEOM_H16=OFF ^
    -DNINFER_KV_ATTN_GEOM_G4=OFF
  if errorlevel 1 (
    echo.
    echo === CONFIGURE FAILED ===
    exit /b 1
  )
  echo === configure OK ===
  echo.
)

if not defined CFG_ONLY (
  echo === building %TARGET% with -j %JOBS% ===
  cmake --build "%BLD%" --parallel %JOBS% --target %TARGET%
  set "RC=!ERRORLEVEL!"
  echo.
  if not "!RC!"=="0" (
    echo === BUILD FAILED exit code !RC! ===
    exit /b !RC!
  )
  echo === BUILD OK ===
)

set "EXE=%BLD%\apps\ninfer-serve.exe"
if not defined CFG_ONLY (
  if exist "!EXE!" (
    for %%F in ("!EXE!") do echo ninfer-serve.exe   %%~zF bytes
  ) else (
    echo WARNING: !EXE! was not produced
    exit /b 1
  )
)
exit /b 0
