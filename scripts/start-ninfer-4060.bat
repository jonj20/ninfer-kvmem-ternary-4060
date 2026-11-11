@echo off
setlocal enabledelayedexpansion
rem Start NInfer server (RTX 4060) - full current binary parameter coverage
rem ASCII-only comments on purpose: cmd.exe reads .bat with the OEM codepage
rem (936 here), so UTF-8 non-ASCII comments corrupt the parser.
rem
rem Usage: scripts\start-ninfer-4060.bat [-model PATH] [-port N] [-host H] [-ctx N]
rem                                      [-kvc N] [-budget N] [-reserve N] [-hostmib N]
rem                                      [-sink N] [-recent N] [-sessions N]
rem                                      [-prefill N] [-drafts N] [-kvtype T]
rem                                      [-concurrency N] [-device N] [-tp] [-tp-split A,B]
rem                                      [-expert-residency device|host|disk]
rem                                      [-fast-prefill] [-prefill-cublas] [-no-prefill-a8]
rem                                      [-mtp-q4] [-gdn-fp16] [-rope-yarn] [-yarn-factor F]
rem                                      [-cuda-policy default|mixed|strict]
rem                                      [-stats-port N] [-api-key K] [-cors] [-no-webui]
rem                                      [-log-level L] [-log-colours on|off] [-dryrun]
rem
rem Defaults for 27B MoE + KVMem + MTP on RTX 4060 (8 GB):
rem   ctx=262144, budget=23040, reserve=2048, hostmib=12288, sink=12288, recent=1024,
rem   sessions=1, prefill=256, drafts=2, kvtype=rk4v4-e8, concurrency=1, device=0,
rem   expert-residency=device, expert-cache-mib=auto, fast-prefill=off, mtp-q4=off,
rem   gdn-fp16=off, cuda-policy=default, stats-port=off, cors=off, log-level=info
rem
rem The kvtype must be one the build carries: scripts\build-ninfer-4060.bat compiles rk8v4,
rem rk4v4, rk4v4-e8 and nvfp4 (and always bf16 and int8). The server refuses any other coding by
rem name at startup, so a build and a launcher that disagree fail there, not inside the first
rem generation.
rem rk4v4-e8 is the default: measured on this card at a 17,088-token KVMem budget (one prompt,
rem one 160-token answer each), it needs 925.6 MiB of KV runtime, the same as rk4v4 and nvfp4,
rem and decodes at 68.3 tok/s against nvfp4's 66.5 and rk8v4's 64.4 - and rk8v4's 8-bit keys
rem cost 14% more memory for no speed. docs\cli.md has the formats' size and quality numbers.

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "EXE=%ROOT%\build-backends\ninfer\apps\ninfer-serve.exe"

set "MODEL=E:\gguf\qwen3.6-35b-a3b\kvmem-v0.19.1-windows-x64-nvidia-40-unified\models\Bonsai2-T5-MTP-r3-Q4.ninfer"
set "PORT=18200"
set "HOST=127.0.0.1"
set "CTX=262144"
set "KVC="
set "BUDGET=23040"
set "RESERVE=2048"
set "HOSTMIB=12288"
set "SINK=12288"
set "RECENT=1024"
set "SESSIONS=1"
set "PRE=256"
set "DRAFTS=2"
set "KVTYPE=rk4v4-e8"
set "CONCURRENCY=1"
set "DEVICE=0"
set "DEVICES="
set "STAGE_LAYERS="
set "TENSOR_PARALLEL="
set "TENSOR_SPLIT="
set "EXPERT_RESIDENCY=device"
set "EXPERT_CACHE=auto"
set "FAST_PREFILL="
set "PREFILL_CUBLAS="
set "NO_PREFILL_A8="
set "MTP_Q4="
set "GDN_FP16="
set "ROPE_YARN="
set "YARN_FACTOR=1"
set "CUDA_POLICY=default"
set "STATS_PORT="
set "API_KEY="
set "CORS="
set "NO_WEBUI="
set "LOG_LEVEL=info"
set "LOG_COLOURS=on"
set "DRYRUN="

:parseargs
if "%~1"=="" goto parsed
if /i "%~1"=="-model"      (set "MODEL=%~2"       & shift & shift & goto parseargs)
if /i "%~1"=="-port"      (set "PORT=%~2"        & shift & shift & goto parseargs)
if /i "%~1"=="-host"      (set "HOST=%~2"        & shift & shift & goto parseargs)
if /i "%~1"=="-ctx"       (set "CTX=%~2"         & shift & shift & goto parseargs)
if /i "%~1"=="-kvc"       (set "KVC=%~2"         & shift & shift & goto parseargs)
if /i "%~1"=="-budget"    (set "BUDGET=%~2"      & shift & shift & goto parseargs)
if /i "%~1"=="-reserve"   (set "RESERVE=%~2"     & shift & shift & goto parseargs)
if /i "%~1"=="-hostmib"   (set "HOSTMIB=%~2"     & shift & shift & goto parseargs)
if /i "%~1"=="-sink"      (set "SINK=%~2"        & shift & shift & goto parseargs)
if /i "%~1"=="-recent"    (set "RECENT=%~2"      & shift & shift & goto parseargs)
if /i "%~1"=="-sessions"  (set "SESSIONS=%~2"    & shift & shift & goto parseargs)
if /i "%~1"=="-prefill"   (set "PRE=%~2"         & shift & shift & goto parseargs)
if /i "%~1"=="-drafts"    (set "DRAFTS=%~2"      & shift & shift & goto parseargs)
if /i "%~1"=="-kvtype"    (set "KVTYPE=%~2"      & shift & shift & goto parseargs)
if /i "%~1"=="-concurrency"(set "CONCURRENCY=%~2"& shift & shift & goto parseargs)
if /i "%~1"=="-device"    (set "DEVICE=%~2"      & shift & shift & goto parseargs)
if /i "%~1"=="-devices"   (set "DEVICES=%~2"     & shift & shift & goto parseargs)
if /i "%~1"=="-stage-layers"(set "STAGE_LAYERS=%~2"& shift & shift & goto parseargs)
if /i "%~1"=="-tp"        (set "TENSOR_PARALLEL=1" & shift & goto parseargs)
if /i "%~1"=="-tp-split"  (set "TENSOR_SPLIT=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-expert-residency" (set "EXPERT_RESIDENCY=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-expert-cache" (set "EXPERT_CACHE=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-fast-prefill" (set "FAST_PREFILL=1" & shift & goto parseargs)
if /i "%~1"=="-no-fast-prefill" (set "FAST_PREFILL=" & shift & goto parseargs)
if /i "%~1"=="-prefill-cublas" (set "PREFILL_CUBLAS=1" & shift & goto parseargs)
if /i "%~1"=="-no-prefill-a8"  (set "NO_PREFILL_A8=1" & shift & goto parseargs)
if /i "%~1"=="-mtp-q4"    (set "MTP_Q4=1"        & shift & goto parseargs)
if /i "%~1"=="-gdn-fp16"  (set "GDN_FP16=1"      & shift & goto parseargs)
if /i "%~1"=="-rope-yarn" (set "ROPE_YARN=1"     & shift & goto parseargs)
if /i "%~1"=="-yarn-factor"(set "YARN_FACTOR=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-cuda-policy" (set "CUDA_POLICY=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-stats-port" (set "STATS_PORT=%~2"  & shift & shift & goto parseargs)
if /i "%~1"=="-api-key"    (set "API_KEY=%~2"     & shift & shift & goto parseargs)
if /i "%~1"=="-cors"       (set "CORS=1"          & shift & goto parseargs)
if /i "%~1"=="-no-webui"   (set "NO_WEBUI=1"      & shift & goto parseargs)
if /i "%~1"=="-log-level"  (set "LOG_LEVEL=%~2"   & shift & shift & goto parseargs)
if /i "%~1"=="-log-colours" (set "LOG_COLOURS=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-dryrun"     (set "DRYRUN=1"        & shift & goto parseargs)
echo Unknown switch: %~1
exit /b 2
:parsed

if not exist "%EXE%" (
  echo ERROR: %EXE% not found.
  echo Build it first with scripts\build_4060_win.bat
  exit /b 1
)
if not exist "%MODEL%" (
  echo ERROR: model artifact not found:
  echo        %MODEL%
  echo Pass -model PATH to point at a .ninfer file.
  exit /b 1
)

rem MTP pool preflight
set /a POOLNEED=BUDGET+PRE+DRAFTS
if "%DRAFTS%"=="0" goto poolcheck
if %DRAFTS% LSS 1 goto bad_drafts
if %DRAFTS% GTR 15 goto bad_drafts
:poolcheck
if "%KVC%"=="" goto after_preflight
if %KVC% LSS %POOLNEED% goto pool_short
goto after_preflight
:bad_drafts
echo ERROR: -drafts %DRAFTS% outside MTP domain (0=off, 1..15=on).
exit /b 2
:pool_short
echo ERROR: -kvc %KVC% cannot cover kvmem window + MTP lead.
echo        Needed: budget %BUDGET% + prefill %PRE% + drafts %DRAFTS% = %POOLNEED%.
exit /b 2
:after_preflight

set "SPEC="
if not "%DRAFTS%"=="0" set "SPEC=--spec mtp --draft-tokens %DRAFTS%"

set "KV_ARGS=--kvmem --kvmem-budget %BUDGET% --kvmem-gen-reserve %RESERVE% --kvmem-host-mib %HOSTMIB% --kvmem-sink-tokens %SINK% --kvmem-recent-tokens %RECENT% --kvmem-sessions %SESSIONS% --kvmem-query-scorer page-softmax --kvmem-score-agg mean --kvmem-graph-alpha 0.2 --no-kvmem-lrfu --kvmem-disk-mib 0 --kvmem-query-max-tokens 512 --kvmem-query-replay-max-tokens 2048"
if not "%KVC%"=="" set "KV_ARGS=%KV_ARGS% --kv-capacity %KVC%"
set "DEVICE_ARGS="
if not "%DEVICES%"=="" (
  set "DEVICE_ARGS=--devices %DEVICES%"
) else if not "%STAGE_LAYERS%"=="" (
  set "DEVICE_ARGS=--devices %DEVICE% --stage-layers %STAGE_LAYERS%"
) else (
  set "DEVICE_ARGS=--device %DEVICE%"
)
if "%TENSOR_PARALLEL%"=="1" set "DEVICE_ARGS=%DEVICE_ARGS% --tensor-parallel"
if not "%TENSOR_SPLIT%"=="" set "DEVICE_ARGS=%DEVICE_ARGS% --tensor-split %TENSOR_SPLIT%"

set "EXPERT_ARGS=--expert-residency %EXPERT_RESIDENCY% --expert-cache-mib %EXPERT_CACHE%"
if "%FAST_PREFILL%"=="1" set "EXPERT_ARGS=%EXPERT_ARGS% --fast-prefill-kernel"
if "%PREFILL_CUBLAS%"=="1" set "EXPERT_ARGS=%EXPERT_ARGS% --prefill-cublas"
if "%NO_PREFILL_A8%"=="1" set "EXPERT_ARGS=%EXPERT_ARGS% --no-prefill-a8"
if "%MTP_Q4%"=="1" set "EXPERT_ARGS=%EXPERT_ARGS% --mtp-experts-q4"
if "%GDN_FP16%"=="1" set "EXPERT_ARGS=%EXPERT_ARGS% --gdn-state-fp16"
if "%ROPE_YARN%"=="1" set "EXPERT_ARGS=%EXPERT_ARGS% --rope-yarn --rope-yarn-factor %YARN_FACTOR%"
if not "%CUDA_POLICY%"=="default" set "EXPERT_ARGS=%EXPERT_ARGS% --cuda-memory-policy %CUDA_POLICY%"

set "SPEC_ARGS=%SPEC%"
if not "%STATS_PORT%"=="" set "SPEC_ARGS=%SPEC_ARGS% --stats-port %STATS_PORT%"
if not "%API_KEY%"=="" set "SPEC_ARGS=%SPEC_ARGS% --api-key %API_KEY%"
if "%CORS%"=="1" set "SPEC_ARGS=%SPEC_ARGS% --cors"
if "%NO_WEBUI%"=="1" set "SPEC_ARGS=%SPEC_ARGS% --no-webui"
if not "%LOG_LEVEL%"=="info" set "SPEC_ARGS=%SPEC_ARGS% --log-level %LOG_LEVEL%"
if not "%LOG_COLOURS%"=="on" set "SPEC_ARGS=%SPEC_ARGS% --log-colours %LOG_COLOURS%"

echo === ninfer-serve (RTX 4060 full params) ===
echo   exe          : %EXE%
echo   model        : %MODEL%
echo   context      : %CTX%
echo   kv pool      : %KVC%
echo   budget       : %BUDGET%
echo   reserve      : %RESERVE%
echo   host MiB     : %HOSTMIB%
echo   sink tokens  : %SINK%
echo   recent tokens: %RECENT%
echo   sessions     : %SESSIONS%
echo   prefill      : %PRE%
echo   drafts       : %DRAFTS%
echo   kvtype       : %KVTYPE%
echo   concurrency  : %CONCURRENCY%
echo   device(s)    : %DEVICE_ARGS%
echo   tp           : %TENSOR_PARALLEL%
echo   expert res   : %EXPERT_RESIDENCY%
echo   expert cache : %EXPERT_CACHE%
echo   fast prefill : %FAST_PREFILL%
echo   prefill cublas: %PREFILL_CUBLAS%
echo   no prefill a8: %NO_PREFILL_A8%
echo   mtp q4       : %MTP_Q4%
echo   gdn fp16     : %GDN_FP16%
echo   rope yarn    : %ROPE_YARN%
echo   cuda policy  : %CUDA_POLICY%
echo   port         : %PORT%
echo   host         : %HOST%
echo.

if "%DRYRUN%"=="1" goto dryrun
echo Log follows. Ctrl+C stops the server.
echo ==========================================================
echo.

taskkill /f /im ninfer-serve.exe >nul 2>&1
ping -n 3 127.0.0.1 >nul

"%EXE%" "%MODEL%" ^
  --host %HOST% --port %PORT% ^
  --max-context %CTX% ^
  --kv-dtype %KVTYPE% ^
  --max-concurrency %CONCURRENCY% --default-max-tokens %CTX% ^
  --prefill-chunk %PRE% ^
  %KV_ARGS% ^
  %DEVICE_ARGS% ^
  %EXPERT_ARGS% ^
  %SPEC_ARGS%

set "RC=%ERRORLEVEL%"
echo.
echo Server stopped with code %RC%.
pause
exit /b %RC%

:dryrun
echo   dry run: server was not started.
echo.
echo   Command line:
echo   "%EXE%" "%MODEL%" --host %HOST% --port %PORT% --max-context %CTX% --kv-dtype %KVTYPE% --max-concurrency %CONCURRENCY% --default-max-tokens %CTX% --prefill-chunk %PRE% %KV_ARGS% %DEVICE_ARGS% %EXPERT_ARGS% %SPEC_ARGS%
echo.
exit /b 0