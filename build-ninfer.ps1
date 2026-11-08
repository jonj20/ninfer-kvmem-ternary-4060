#Requires -Version 5.1
<#
.SYNOPSIS
    Build the NInfer worker (ninfer-serve) repeatedly.

.DESCRIPTION
    Configures and builds backends/ninfer into build-backends/ninfer.
    This script does NOT fetch sources or apply patches; it only compiles
    the current working tree.

.EXAMPLES
    # Incremental build (default sm_89, 8 jobs)
    .\build-ninfer.ps1

    # Clean rebuild
    .\build-ninfer.ps1 -Clean

    # Rebuild and strip PTX to reduce binary size
    .\build-ninfer.ps1 -PrunePtx

    # Only reconfigure / only compile
    .\build-ninfer.ps1 -ConfigureOnly
    .\build-ninfer.ps1 -BuildOnly
#>
[CmdletBinding()]
param(
    [string]$CudaPath = $(if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'D:/dev/nvkit' }),
    [string]$VcpkgTree = 'D:/dev/vcpkg/installed/x64-windows',
    [string]$Architectures = '89',
    [ValidateRange(1, 64)][int]$Jobs = 8,
    [string]$BuildDir = '',
    [switch]$Clean,
    [switch]$PrunePtx,
    [switch]$ConfigureOnly,
    [switch]$BuildOnly,
    [string[]]$CMakeArgs = @()
)

$ErrorActionPreference = 'Stop'
if ($ConfigureOnly -and $BuildOnly) {
    throw 'Choose -ConfigureOnly or -BuildOnly, not both.'
}

$root = $PSScriptRoot
# This checkout nests ninfer at the repository root beside kvmem.
$ninferSrc = $root
$kvmemSrc = Join-Path $root 'kvmem'
if (!$BuildDir) { $BuildDir = Join-Path $root 'build-backends/ninfer' }
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
$ninferSrc = [IO.Path]::GetFullPath($ninferSrc)
$kvmemSrc = [IO.Path]::GetFullPath($kvmemSrc)

if (!(Test-Path (Join-Path $ninferSrc 'CMakeLists.txt'))) {
    throw "NInfer source not found: $ninferSrc"
}
if (!(Test-Path (Join-Path $kvmemSrc 'include/kvmem/memory_contract.hpp'))) {
    throw "KVMem core not found: $kvmemSrc"
}
if (!(Test-Path (Join-Path $CudaPath 'bin/nvcc.exe'))) {
    throw "CUDA toolkit not found: $CudaPath (pass -CudaPath)"
}
if (!(Test-Path (Join-Path $VcpkgTree 'share/curl/CURLConfig.cmake'))) {
    throw "vcpkg triplet tree not found: $VcpkgTree (pass -VcpkgTree)"
}

# Make sure cmake/ninja are reachable. This machine keeps them in the Python Scripts dir.
if (!(Get-Command cmake -ErrorAction SilentlyContinue) -or
    !(Get-Command ninja -ErrorAction SilentlyContinue)) {
    $toolHint = 'D:/dev/python/Scripts'
    if (Test-Path (Join-Path $toolHint 'cmake.exe')) {
        $env:PATH = "$toolHint;$env:PATH"
    }
}
if (!(Get-Command cmake -ErrorAction SilentlyContinue)) { throw 'cmake not found on PATH.' }
if (!(Get-Command ninja -ErrorAction SilentlyContinue)) { throw 'ninja not found on PATH.' }

# Enter the VS2022 x64 developer environment when started from a plain shell.
if (!$env:VSCMD_VER) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (!(Test-Path $vswhere)) {
        throw 'vswhere.exe not found. Install Visual Studio 2022 C++ Build Tools.'
    }
    $vs = & $vswhere -version '[17.0,18.0)' -latest -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$vs) {
        throw 'Visual Studio 2022 C++ Build Tools not found.'
    }
    & (Join-Path $vs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
}

$env:PATH = "$CudaPath/bin;$CudaPath/bin/x64;$env:PATH"

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Removing build directory: $BuildDir"
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}

# A build directory configured for another checkout cannot be reused.
$cache = Join-Path $BuildDir 'CMakeCache.txt'
if (Test-Path $cache) {
    $line = Select-String -Path $cache -Pattern '^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$' |
        Select-Object -First 1
    if ($line) {
        $cachedSrc = $line.Matches[0].Groups[1].Value -replace '/', '\'
        if ($cachedSrc.TrimEnd('\') -ne $ninferSrc.TrimEnd('\')) {
            Write-Host "Build directory was configured for another source tree; recreating it."
            Remove-Item -LiteralPath $BuildDir -Recurse -Force
        }
    }
}

if (!$BuildOnly) {
    $configureArgs = @(
        '-S', $ninferSrc,
        '-B', $BuildDir,
        '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release',
        "-DNINFER_KVMEM_SOURCE_DIR=$kvmemSrc",
        '-DNINFER_BUILD_APPS=ON',
        '-DNINFER_BUILD_BENCHMARKS=OFF',
        "-DCMAKE_CUDA_ARCHITECTURES=$Architectures",
        "-DCMAKE_CUDA_COMPILER=$CudaPath/bin/nvcc.exe",
        "-DCUDAToolkit_ROOT=$CudaPath",
        "-DNINFER_VCPKG_TREE=$VcpkgTree"
    )
    if ($PrunePtx) {
        $configureArgs += '-DNINFER_CUDA_PRUNE_PTX=ON'
    }
    $configureArgs += $CMakeArgs

    Write-Host ''
    Write-Host '=== Configure ==='
    Write-Host ("cmake " + ($configureArgs -join ' '))
    & cmake @configureArgs
    if ($LASTEXITCODE -ne 0) { throw "Configure failed: $LASTEXITCODE" }
}

if (!$ConfigureOnly) {
    $buildArgs = @(
        '--build', $BuildDir,
        '--parallel', "$Jobs",
        '--target', 'ninfer-serve'
    )

    Write-Host ''
    Write-Host '=== Build ==='
    Write-Host ("cmake " + ($buildArgs -join ' '))
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & cmake @buildArgs
    $code = $LASTEXITCODE
    $sw.Stop()
    if ($code -ne 0) { throw "Build failed: $code" }
    Write-Host ("Build finished in {0:n1} min." -f $sw.Elapsed.TotalMinutes)
}

$exe = Join-Path $BuildDir 'apps/ninfer-serve.exe'
if (Test-Path $exe) {
    $item = Get-Item $exe
    Write-Host ''
    Write-Host '=== Result ==='
    Write-Host ("Exe   : {0}" -f $item.FullName)
    Write-Host ("Size  : {0:n0} bytes ({1:n1} MB)" -f $item.Length, ($item.Length / 1MB))
    Write-Host ("Built : {0}" -f $item.LastWriteTime)
    & $exe --version
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "ninfer-serve --version exited with $LASTEXITCODE"
    }
} elseif (!$ConfigureOnly) {
    throw "Build finished but ninfer-serve.exe was not found: $exe"
}
