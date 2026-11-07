@echo off
rem Build NInfer as a native Windows x64 executable (MSVC + CUDA + Ninja).
rem Produces build_4060\apps\*.exe.
rem ASCII-only on purpose: cmd.exe reads .bat with the OEM codepage (936 here), so
rem UTF-8 non-ASCII comments corrupt the parser.
rem
rem Usage: build_4060_win.bat [-configure] [-clean] [-jobs N]
rem
rem   -configure  force a reconfigure even when a CMake cache already exists
rem   -clean      delete build_4060 first (full rebuild from scratch)
rem   -jobs N     parallel compile jobs (default 16)
rem
rem Default behaviour with no switches is an incremental build: configure runs only
rem when build_4060\CMakeCache.txt is missing, otherwise it skips straight to compiling.
rem
rem Discovered paths on this machine (2026-10-03). Adjust if a component moves:
rem   MSVC     D:\dev\VS2022                       (Visual Studio Community 2022 17.14)
rem   CUDA     D:\dev\nvkit                        (13.4.92; the default
rem              C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4 is an
rem              empty stub with no nvcc, so D:\dev\nvkit is the real install)
rem   vcpkg    D:\dev\vcpkg                        (curl, ffmpeg, pkgconf)
rem   Ninja    D:\dev\cygwin64\bin
rem   CMake    comes from vcvars64.bat (VS-bundled 3.31.6)
rem
rem Two settings that are easy to get wrong and both fail opaquely:
rem   1. -DVCPKG_MANIFEST_MODE=OFF. The repo root carries a vcpkg.json with a
rem      builtin-baseline, and resolving that baseline needs a full git history in
rem      the vcpkg checkout. The packages are already installed into
rem      D:\dev\vcpkg\installed, so classic lookup is both correct and independent
rem      of git. Leaving manifest mode on fails during CMake configure with
rem      "not a git repository".
rem   2. -DCMAKE_CUDA_ARCHITECTURES=89 is validated by the top-level CMakeLists and
rem      only sm_89 is accepted. 89 covers both the RTX 4090 and the RTX 4060.
setlocal enabledelayedexpansion

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "BLD=%ROOT%\build_4060"
set "VCVARS=D:\dev\VS2022\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA=D:\dev\nvkit"
set "VCPKG=D:\dev\vcpkg"
set "NINJA=D:\dev\cygwin64\bin\ninja.exe"
set "JOBS=16"
set "FORCE_CFG="
set "CLEAN="

:parseargs
if "%~1"=="" goto parsed
if /i "%~1"=="-configure" (set "FORCE_CFG=1" & shift & goto parseargs)
if /i "%~1"=="-clean" (set "CLEAN=1" & shift & goto parseargs)
if /i "%~1"=="-jobs" (set "JOBS=%~2" & shift & shift & goto parseargs)
echo Unknown switch: %~1
echo Usage: build_4060_win.bat [-configure] [-clean] [-jobs N]
exit /b 2
:parsed

if not exist "%VCVARS%" (
  echo ERROR: vcvars64.bat not found at:
  echo        %VCVARS%
  echo Edit VCVARS in this script to point at your Visual Studio install.
  exit /b 1
)
if not exist "%CUDA%\bin\nvcc.exe" (
  echo ERROR: nvcc not found at %CUDA%\bin\nvcc.exe. Edit CUDA in this script.
  exit /b 1
)
if not exist "%VCPKG%\scripts\buildsystems\vcpkg.cmake" (
  echo ERROR: vcpkg toolchain not found. Edit VCPKG in this script.
  exit /b 1
)

if defined CLEAN (
  echo Removing %BLD% ...
  if exist "%BLD%" rmdir /s /q "%BLD%"
)

call "%VCVARS%" || exit /b 1
rem Keep /showIncludes in English so Ninja records header dependencies. A localized
rem MSVC emits a translated prefix that Ninja drops, leaving depfiles empty and
rem letting objects with a stale struct layout survive header edits.
set "VSLANG=1033"
set "PATH=%CUDA%\bin;%VCPKG%\installed\x64-windows\bin;%PATH%"

set "NEED_CFG="
if defined FORCE_CFG set "NEED_CFG=1"
if not exist "%BLD%\CMakeCache.txt" set "NEED_CFG=1"

if defined NEED_CFG (
  echo === configuring %BLD% ===
  cmake -S "%ROOT%" -B "%BLD%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG%\scripts\buildsystems\vcpkg.cmake" ^
    -DVCPKG_TARGET_TRIPLET=x64-windows ^
    -DVCPKG_MANIFEST_MODE=OFF ^
    -DCMAKE_CUDA_ARCHITECTURES=89 ^
    -DCUDAToolkit_ROOT="%CUDA%" ^
    -DCMAKE_CUDA_COMPILER="%CUDA%\bin\nvcc.exe"
  if errorlevel 1 (
    echo.
    echo === CONFIGURE FAILED ===
    exit /b 1
  )
  echo === configure OK ===
  echo.
) else (
  echo === reusing existing CMake cache; pass -configure to force ===
  echo.
)

echo === building with -j %JOBS% ===
cmake --build "%BLD%" -j %JOBS%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
  echo === BUILD FAILED  exit code %RC% ===
  exit /b %RC%
)

echo === BUILD OK ===
if exist "%BLD%\apps\ninfer-serve.exe" (
  for %%F in ("%BLD%\apps\ninfer-serve.exe") do echo   ninfer-serve.exe  %%~zF bytes
  for %%F in ("%BLD%\apps\ninfer.exe") do echo   ninfer.exe         %%~zF bytes
  for %%F in ("%BLD%\apps\ninfer-perplexity.exe") do echo   ninfer-perplexity.exe  %%~zF bytes
) else (
  echo WARNING: %BLD%\apps\ninfer-serve.exe was not produced
  exit /b 1
)
echo.
echo Start the server with: scripts\start-ninfer-4060.bat
exit /b 0