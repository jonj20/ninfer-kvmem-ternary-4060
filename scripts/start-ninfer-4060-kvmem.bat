@echo off
rem Windows-native launcher that turns on the KVMem bounded KV working set, so a
rem context far larger than VRAM can be served. Runs the .exe directly - no WSL.
rem ASCII-only on purpose: cmd.exe reads .bat with the OEM codepage (936 here), so
rem UTF-8 non-ASCII comments corrupt the parser.
rem
rem MODEL: the default artifact is the one measured with this build. The newer
rem Ternary-Bonsai-2-T5-MTP-r3-Q4.ninfer is NOT loadable by this build - startup
rem fails with "weight/000192: RowSplit requires a grouped integer format"; use it
rem
rem WHAT THIS BUYS YOU
rem   Default here is a 262144-token logical context with only 40960 tokens of
rem   Device KV. The rest lives in the Host KV tier and comes back through a
rem   per-round retrieval window instead of staying resident. Measured on an RTX
rem   4060 Laptop (8188 MiB) on
rem   Ternary-Bonsai-2-27B-PTQ1_0-vl_mtp_q4q5.ninfer:
rem
rem     prompt_n = 258,939 tokens prefilled and generated in one request (364 s),
rem     no context_length_exceeded, while device KV residency fell from 1024
rem     pages (65536 tokens) to 559 and Host KV grew to 3.89 GiB.
rem
rem   That same context placed densely needs 4.25 GiB of KV plus 5.52 GiB of
rem   weights, which does not fit in 8.58 GiB. That gap is the whole point.
rem
rem   MTP x KVMM IN THIS BUILD (stage 6a)
rem   --spec mtp and a real --kvmem-budget now run TOGETHER: MTP target
rem   verification consumes the renumbered window through the cache-position
rem   pipeline and --kvmem-select stays live. Only the dflash family keeps the
rem   identity window (not offered here). The pool must therefore cover the
rem   draft lead:
rem     kv_capacity >= budget + prefill_chunk + max_concurrency * draft_window
rem   (lanes = 1 here, so + drafts). This script pre-checks it before launching.
rem   NOTE: the engine --help text for --kvmem-budget still says "Ignored while
rem   --spec is enabled"; that string is stale. docs/cli.md and docs/serving.md
rem   carry the current contract.
rem
rem THE FOUR NUMBERS
rem   ctx      262144  logical ceiling (--max-context)
rem   kvc       40960  Device KV pool, the resident working set (--kv-capacity)
rem   budget    32768  tokens the KVMem window keeps Device-resident (--kvmem-budget)
rem   drafts        1  MTP draft tokens (--draft-tokens), 0 = MTP off
rem
rem   gen_reserve is not a flag: it is derived as kv_capacity - budget = 8192.
rem
rem   kvc defaults to 40960, not 65536, because MTP drafts need the verify
rem   round, its graphs and the backend pool on top of the KV pool: at kvc
rem   65536 with -drafts 3 startup refuses with
rem     "requested Engine runtime reservation requires 1638109952 bytes, but only
rem      1245708288 bytes are available for runtime capacity" (short 374 MiB).
rem   kvc 40960 + drafts 3 starts, warms up and serves on this card.
rem
rem WHY --kv-ring IS MANDATORY HERE
rem   startup.cpp requires kv_capacity >= max_context unless --kv-ring is set, and
rem   --kvmem-budget does NOT relax that (ring_requested() reads only kv_ring).
rem   --kv-ring is what lets the Device pool (40960) be smaller than the logical
rem   context (262144). It only lowers the pool-size floor; the windowing itself is
rem   driven by --kvmem-budget. Without it startup refuses with
rem   "kv_capacity must be at least max_context".
rem
rem WHY -hostkv IS NOT PASSED BY DEFAULT
rem   With --kv-ring and --kvmem-budget both on, "--host-kv-mib 0" makes startup
rem   auto-size the Host tier by the ring rule (logical - pool = 3072 pages) while
rem   KVMem needs logical - window (3584 pages) and then refuses. Leaving the flag
rem   off uses the ContextCacheOptions default: 8 GiB addressable, pinned lazily.
rem   Pass -hostkv only when you must pin an explicit size (>= 3809 at 256K).
rem   Under MTP the demoted backend pages join the same tier, so an explicit pin
rem   must also cover the MTP backend stride.
rem
rem MTP DRAFT WIDTH - MEASURED ON THIS CARD
rem   Same request (76-token prompt, 256 output, temp 0), decode tok/s:
rem     -drafts 0  37.15      -drafts 1  43.90      -drafts 2  36.92
rem     -drafts 3  34.61      -drafts 4  41.43
rem   One draft is the sweet spot: it accepts ~60% of drafts and pays one
rem   verify token per round, so it beats plain decode by ~18%. Wider windows
rem   accept fewer drafts each (39.9% at 2, 30.8% at 4) and the extra verify
rem   work cancels the gain, which is why the default is 1. Pass -drafts 3/4/5
rem   only to measure your own workload.
rem
rem RETRIEVAL CAVEAT
rem   -select retrieval scores blocks by mean-key similarity. Since stage 6a it
rem   stays live under MTP drafts (verify pre-RoPE keys feed the same score
rem   store). -drafts 0 keeps the ordinary-decode path, which is the measured
rem   pure-KVMem configuration above.
rem
rem Usage: start-ninfer-4060-kvmem.bat [-ctx N] [-kvc N] [-budget N] [-noring]
rem                                         [-select recency|retrieval] [-nothink]
rem                                         [-drafts N] [-hostkv MIB] [-prefill N]
rem                                         [-port N] [-model PATH] [-dryrun]
rem
rem   -ctx N        logical context ceiling (default 262144; multiple of 128)
rem   -kvc N        Device KV pool in tokens (default 40960). Must be >=
rem                  budget + prefill chunk + drafts, and < ctx, which is why
rem                  --kv-ring is passed unless -noring is given.
rem   -budget N     resident KVMem window in tokens (default 32768; multiple of 128)
rem   -noring       drop --kv-ring. Legal only when -kvc >= -ctx, which makes the
rem                  pool cover the whole context and needs far more VRAM; that
rem                  defeats KVMem, so this is for A/B comparison only.
rem   -select M     recency (default) or retrieval
rem   -nothink      disable thinking. Costs no extra VRAM.
rem   -drafts N     MTP draft tokens (default 1; 0 = MTP off). N in 1..5; the
rem                  engine has dedicated CUDA graph shapes at 3/4/5. The pool
rem                  must cover budget + prefill chunk + N (pre-checked here).
rem   -hostkv MIB   pin the Host KV arena to MIB MiB (>= 3809 at 256K context).
rem   -prefill N    prefill chunk in tokens (default 1024; multiple of 128)
rem   -port N       listen port (default 8080)
rem   -model PATH   .ninfer artifact. Overrides the default below.
rem   -dryrun       print the resolved settings and command line, then exit.
rem                  Nothing is killed and the server is not started.
rem
rem MEMORY - this card is an RTX 4060 Laptop with 8188 MiB.
rem   VRAM: the vl artifact measured weights 5.78 GiB loaded + KV pool ~0.67 GiB
rem        at kvc 40960 plus startup state, KV page tables, MTP verify graphs and
rem        the draft backend pool; the engine reported
rem        runtime_reservation_bytes=1183552256 (1.10 GiB) with
rem        available_after_weights_bytes=1245708288 (1.16 GiB). -drafts N adds
rem        the MTP backend KV (capped by the pool) and the verify round state.
rem        Scaling kvc up scales VRAM linearly at 17408 bytes/token, and kvc
rem        65536 with drafts 3 does NOT fit (see above).
rem        A refusal is not a crash - the log names the exact shortfall, e.g.
rem          "requested Engine runtime reservation requires N bytes, but only M ..."
rem          "--kv-capacity must cover --kvmem-budget plus one --prefill-chunk ..."
rem          "--kvmem-budget needs a Host KV tier ..."
rem        Cut -ctx / -kvc / -budget / -drafts to fit.
rem   HOST: startup pins 1.15 GiB of host state regardless of -hostkv, and text
rem        Host KV reached ~3.9 GiB at 256K context (drafts-off measurement).
rem        Under MTP the demoted backend pages share that tier. 32 GB of system
rem        RAM is comfortable; 16 GB is tight (measured OK at 15.7 GB total /
rem        8.9 GB free); 12 GB will not hold a 256K context. Lower -ctx to trade
rem        context for RAM.
rem
rem Usage notes:
rem   - Ctrl+C in this window stops the server (it runs in the foreground).
rem   - Any stale ninfer-serve.exe is killed first, so rerunning is safe.
rem   - WebUI http://localhost:8080/   API http://localhost:8080/v1
rem     Monitor http://localhost:8080/monitor
setlocal enabledelayedexpansion

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "APPS=%ROOT%\build_4060\apps"
set "EXE=%APPS%\ninfer-serve.exe"
set "MODEL=E:\gguf\qwen3.6-35b-a3b\ninfer-bonsai-27b\Ternary-Bonsai-2-27B-PTQ1_0-vl_mtp_q4q5.ninfer"

set "VCPKG=D:\dev\vcpkg"
set "CUDA=D:\dev\nvkit"

set "CTX="
set "KVC="
set "BUDGET="
set "RING=1"
set "SELECT="
set "THINK=1"
set "DRAFTS="
set "PRE="
set "PORT="
set "HOSTKV="
set "DRYRUN="

:parseargs
if "%~1"=="" goto parsed
if /i "%~1"=="-ctx" (set "CTX=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-kvc" (set "KVC=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-budget" (set "BUDGET=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-noring" (set "RING=0" & shift & goto parseargs)
if /i "%~1"=="-select" (set "SELECT=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-think" (set "THINK=1" & shift & goto parseargs)
if /i "%~1"=="-nothink" (set "THINK=" & shift & goto parseargs)
if /i "%~1"=="-drafts" (set "DRAFTS=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-hostkv" (set "HOSTKV=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-prefill" (set "PRE=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-port" (set "PORT=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-model" (set "MODEL=%~2" & shift & shift & goto parseargs)
if /i "%~1"=="-dryrun" (set "DRYRUN=1" & shift & goto parseargs)
echo Unknown switch: %~1
echo Usage: start-ninfer-4060-kvmem.bat [-ctx N] [-kvc N] [-budget N] [-noring]
echo                                          [-select recency^|retrieval] [-nothink]
echo                                          [-drafts N] [-hostkv MIB] [-prefill N]
echo                                          [-port N] [-model PATH] [-dryrun]
exit /b 2
:parsed

if not exist "%EXE%" (
  echo ERROR: %EXE% not found.
  echo Build it first with the script in the repo: scripts\build_4060_win.bat
  exit /b 1
)
if not exist "%MODEL%" (
  echo ERROR: model artifact not found:
  echo        %MODEL%
  echo Pass -model PATH to point at a .ninfer file.
  exit /b 1
)

if "%CTX%"=="" set "CTX=262144"
if "%KVC%"=="" set "KVC=40960"
if "%BUDGET%"=="" set "BUDGET=32768"
if "%SELECT%"=="" set "SELECT=recency"
if "%DRAFTS%"=="" set "DRAFTS=1"
if "%PRE%"=="" set "PRE=1024"
if "%PORT%"=="" set "PORT=8080"
if "%DRYRUN%"=="" set "DRYRUN=0"

rem --- draft domain and pool preflight -------------------------------
rem The engine refuses exactly this at startup; failing here saves one model load.
rem Lanes are fixed at 1 (--max-concurrency 1), so the MTP draft lead = DRAFTS.
set /a POOLNEED=BUDGET+PRE+DRAFTS
if "%DRAFTS%"=="0" goto poolcheck
if %DRAFTS% LSS 1 goto bad_drafts
if %DRAFTS% GTR 5 goto bad_drafts
:poolcheck
if %KVC% LSS %POOLNEED% goto pool_short
goto after_preflight

:bad_drafts
echo ERROR: -drafts %DRAFTS% is outside the MTP draft window domain (1..5).
echo        Use 0 to switch MTP off, or 1..5 to enable it.
exit /b 2

:pool_short
echo ERROR: -kvc %KVC% cannot cover the KVMem window plus the MTP draft lead.
echo        Needed: budget %BUDGET% + prefill %PRE% + drafts %DRAFTS% = %POOLNEED%.
echo        Raise -kvc to at least %POOLNEED%, or lower -budget / -drafts / -prefill.
exit /b 2

:after_preflight
set /a GENRESERVE=KVC-BUDGET
set /a KVMB=(KVC*17408)/1048576

rem --kv-ring is what permits kv_capacity < max_context; without it startup rejects
rem the pool with "kv_capacity must be at least max_context".
set "RINGFLAG="
if "%RING%"=="1" set "RINGFLAG=--kv-ring"

rem Omitted on purpose; see WHY -hostkv IS NOT PASSED BY DEFAULT above.
set "HOSTKVFLAG="
if not "%HOSTKV%"=="" set "HOSTKVFLAG=--host-kv-mib %HOSTKV%"

rem --- sampling and thinking switches -------------------------------
if defined THINK (
  set "SAMPLING=--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --presence-penalty 0.0"
  set "THINKFLAG="
) else (
  set "SAMPLING=--temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 --presence-penalty 1.5"
  set "THINKFLAG=--no-thinking"
)

rem --- speculation --------------------------------------------------
rem MTP rides the KVMem window in this build (stage 6a): the draft round
rem reselects through the cache-position pipeline and selection stays live.
rem -drafts 0 drops speculation and restores the measured pure-KVMem path.
set "SPEC=--spec mtp --draft-tokens %DRAFTS%"
if "%DRAFTS%"=="0" set "SPEC="

echo === ninfer-serve (KVMem bounded KV working set + MTP) ===
echo   exe      : %EXE%
echo   model    : %MODEL%
if defined THINK (echo   mode     : THINK) else (echo   mode     : no-thinking)
echo   context  : %CTX%   (logical ceiling)
echo   kv pool  : %KVC%   (device working set)
echo   window   : %BUDGET%   (kvmem resident budget)
echo   drafts   : %DRAFTS%   (MTP lead; pool must cover %POOLNEED% = budget+chunk+lead)
echo   reserve  : %GENRESERVE%   (= kv_capacity - budget)
echo   select   : %SELECT%
echo   ring     : %RINGFLAG%
echo   spec     : %SPEC%
echo   kv       : rk4v4-e8   lanes 1   prefill %PRE%
echo   port     : %PORT%
echo.
echo   VRAM need : weights 5.78 GiB (vl) + kv pool ~%KVMB% MiB at kvc %KVC% (17408 B/token)
echo   HOST need : ~1.15 GiB startup + ~3.9 GiB KV at ctx %CTX%
echo.
echo   WebUI   http://localhost:%PORT%/
echo   API     http://localhost:%PORT%/v1
echo   Monitor http://localhost:%PORT%/monitor
echo.
if "%DRYRUN%"=="1" goto dryrun
echo Log follows. Ctrl+C stops the server.
echo ==========================================================
echo.

rem --- stop any stale server ----------------------------------------
taskkill /f /im ninfer-serve.exe >nul 2>&1
ping -n 3 127.0.0.1 >nul

rem --- runtime DLLs: ffmpeg + curl come from vcpkg, cudart from CUDA --
set "PATH=%VCPKG%\installed\x64-windows\bin;%CUDA%\bin;%PATH%"

"%EXE%" "%MODEL%" ^
  --host 0.0.0.0 --port %PORT% ^
  --max-context %CTX% --kv-capacity %KVC% %RINGFLAG% ^
  --kvmem-budget %BUDGET% --kvmem-select %SELECT% ^
  --kv-dtype rk4v4-e8 ^
  --max-concurrency 1 --default-max-tokens 262144 ^
  --prefill-chunk %PRE% ^
  %HOSTKVFLAG% ^
  --device-state-slots 0 ^
  %THINKFLAG% %SAMPLING% %SPEC%

set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" if not "%RC%"=="1" echo === server exited with code %RC% ===
echo Server stopped.
pause
exit /b %RC%

:dryrun
echo   dry run: nothing killed, the server was not started.
echo.
echo   Command line:
echo   "%EXE%" "%MODEL%" --host 0.0.0.0 --port %PORT% --max-context %CTX% --kv-capacity %KVC% %RINGFLAG% --kvmem-budget %BUDGET% --kvmem-select %SELECT% --kv-dtype rk4v4-e8 --max-concurrency 1 --default-max-tokens 262144 --prefill-chunk %PRE% %HOSTKVFLAG% --device-state-slots 0 %THINKFLAG% %SAMPLING% %SPEC%
echo.
exit /b 0
