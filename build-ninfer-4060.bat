@echo off
REM build-ninfer-4060.bat
REM Fixed configuration for RTX 4060 (sm_89), single standalone exe, no kernel DLL.
REM - Prunes PTX to shrink the CUDA archives (smaller ninfer-serve.exe).
REM - Builds only sm_89 so no other architecture code is carried.
call "D:\dev\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CUDACXX=D:\dev\nvkit\bin\nvcc.exe"
powershell -NoProfile -ExecutionPolicy Bypass -File "build-ninfer.ps1" -Clean -Architectures 89 -Jobs 8 -PrunePtx -CudaPath "D:/dev/nvkit" -VcpkgTree "D:/dev/vcpkg/installed/x64-windows"
echo BUILD_EXIT=%ERRORLEVEL%
