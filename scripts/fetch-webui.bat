@echo off
setlocal enabledelayedexpansion
rem Unpacks llama.cpp's released WebUI into a directory for -DNINFER_WEBUI_DIR. The build itself
rem never downloads anything, so offline and hermetic builds stay that way unless this is run
rem first. This is the Windows counterpart of scripts\fetch-webui.sh.
rem
rem Usage: scripts\fetch-webui.bat [tag] [dir]
rem
rem tag defaults to the release below; dir defaults to third_party\webui\<tag> (git-ignored).
rem
rem The bundle is llama.cpp's own WebUI: it talks to the server's llama.cpp-shaped routes
rem (/props, /slots, /v1/models, /v1/chat/completions, ...), all of which ninfer-serve serves
rem from its own port. The patched bundle in the release package calls /api/chat/* and
rem /api/launcher/* instead, which need the release's Python launcher - a second web service.

set "TAG=%~1"
if "%TAG%"=="" set "TAG=b11165"
set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "DIR=%~2"
if "%DIR%"=="" set "DIR=%ROOT%\third_party\webui\%TAG%"
set "URL=https://github.com/ggml-org/llama.cpp/releases/download/%TAG%/llama-%TAG%-ui.tar.gz"

echo === fetching %URL% ===
if not exist "%DIR%" mkdir "%DIR%"
curl -fL --retry 3 -o "%DIR%\ui.tar.gz" "%URL%"
if errorlevel 1 (
  echo ERROR: download failed: %URL%
  exit /b 1
)
tar -xzf "%DIR%\ui.tar.gz" -C "%DIR%"
if errorlevel 1 (
  echo ERROR: extraction failed
  exit /b 1
)
del "%DIR%\ui.tar.gz"

rem The bundle's index.html may sit in a subdirectory; the build wants the directory holding it.
set "INDEX="
for /f "delims=" %%F in ('dir /b /s "%DIR%\index.html" 2^>nul') do if not defined INDEX set "INDEX=%%F"
if not defined INDEX (
  echo ERROR: no index.html under %DIR%
  exit /b 1
)
for %%I in ("%INDEX%") do set "WEBUI_DIR=%%~dpI"
if "%WEBUI_DIR:~-1%"=="\" set "WEBUI_DIR=%WEBUI_DIR:~0,-1%"
echo.
echo WebUI unpacked. Configure with:
echo   -DNINFER_WEBUI_DIR="%WEBUI_DIR%"
exit /b 0
