# NInfer KVMem for the Ternary 4060

A single-card, Windows-first deployment of [NInfer-all](https://github.com/iamwavecut/ninfer-all)'s
KVMem line, built for an 8 GB NVIDIA card: one RTX 4060 (Laptop) running Ternary Bonsai 2 27B with
the KVMem context cache, MTP speculation, llama.cpp's WebUI and a monitoring view of this fork's
own, all from one process on one port.

This repository is a fork, not a new engine. It carries upstream's code and changes only what this
deployment needs.

- **Upstream:** [NInfer-all](https://github.com/iamwavecut/ninfer-all) (which continues
  [NInfer](https://github.com/Neroued/ninfer) through the NInfer-3090 line), at base commit
  `d0da39796697b79fd08fc570e1781d75943dddb1`.
- **On top of it:** the KVMem product line's 27 patches (`kvmem`, `ptq1`, `t5-performance`,
  `nvfp4-*`, `monitor-events`, `memory-planning`, `heads-tp`, ...), applied in order.
- **And this branch's own work**, listed below.

The upstream README describes the whole product: several model families, every GPU from Volta to
consumer Blackwell, pipeline stages, Docker images and native releases. None of that is measured
here. What follows is what this fork is and what it does.

[What this branch changes](#what-this-branch-changes) | [Quick start](#quick-start)
[Measured here](#measured-on-an-rtx-4060-laptop-gpu)
[KV attention coverage](#kv-attention-coverage) | [Repository layout](#repository-layout)
[Known limits](#known-limits) | [Documentation](#documentation)

## What this branch changes

| Area | Change | Where |
|---|---|---|
| KV attention coverage | The KV cache codings and the head geometries a build carries are compile-time options (`NINFER_KV_ATTN_CODING_*`, `NINFER_KV_ATTN_GEOM_*`). Code is kept, not deleted: a build without a coding links a throwing stub and rejects that `--kv-dtype` | `src/ops/softmax_attention/kv_attn_coverage.h`, `src/ops/CMakeLists.txt` |
| Startup admission | The Engine refuses a `--kv-dtype` this build does not carry, by name and with the list it does carry, before a weight is uploaded. The device-route calibration times only the codings the build carries | `src/runtime/engine/model_instance.cpp`, `src/calibration/device_calibration.cu` |
| 4060 profile | `sm_89` only; h24 plus bf16, int8, rk8v4, rk4v4, rk4v4-e8 and nvfp4. The binary is 276 MB against 482 MB for the full coverage, and 285 MB with the WebUI embedded | `scripts/build-ninfer-4060.bat` |
| Windows build and launch | One script builds the server (MSVC + CUDA + Ninja, incremental), one starts it with this deployment's KVMem, MTP and memory flags | `scripts/build-ninfer-4060.bat`, `scripts/start-ninfer-4060.bat` |
| WebUI | llama.cpp's released WebUI, embedded into the server and served from its own port. `scripts/fetch-webui.bat` unpacks it; the build never downloads anything | `scripts/fetch-webui.bat`, `src/serve/webui/embed.cpp` |
| Monitoring view | A first-party page at `/kvmem/` (and `/monitor/`): the card's load, the KV occupancy on the device and in host memory, the KVMem window, sessions, history and payload tiers, the scheduler's queue and the live requests. It reads the server's own `/stats`, `/props` and `/api/monitor/snapshot` | `src/serve/webui/kvmem/index.html` |
| llama.cpp compatibility | `GET /tools` answers the server-side tool list (empty: NInfer's tools travel in the chat request), and `/props` reports the `kv_dtype` the server runs with | `src/serve/http_server.cpp` |
| Model support | The flint and gemma4 families are removed; this deployment serves the Qwen3.5-core models with the KVMem cache | commit `a8dc79b` |

Everything else is upstream's, and upstream's documentation still describes it.

## Quick start

### Prerequisites

- Windows 10 or newer, an NVIDIA GPU with 8 GB or more (this deployment: an RTX 4060 Laptop GPU).
- Visual Studio 2022 build tools (x64), CUDA 12.8 or newer (13.x for Blackwell; this branch builds
  `sm_89`), and a vcpkg tree with curl and FFmpeg.
- The scripts default to `D:\dev\nvkit` (CUDA) and `D:\dev\vcpkg\installed\x64-windows` (vcpkg);
  `-cudaPath` and `-vcpkgTree` override them.

### Build

```bat
scripts\fetch-webui.bat                 rem once, for the chat UI (see "Known limits")
scripts\build-ninfer-4060.bat           rem incremental; -clean for a full rebuild
```

The build configures first, so the coverage profile in the script is always the one the build
directory carries, then compiles `ninfer-serve` into `build-backends\ninfer\apps\ninfer-serve.exe`.
`-jobs N` sets the parallelism (8 by default), `-target T` another target, `-buildDir DIR` another
directory.

### Serve

```bat
scripts\start-ninfer-4060.bat
```

The script starts the server on `http://127.0.0.1:18200` with this deployment's flags: `--kv-dtype
rk4v4-e8`, KVMem with a 23,040-token budget and a 12 GiB host tier, MTP with two drafts, and the
device expert residency. `-model PATH` points at another artifact, `-kvtype` at another coding the
build carries, `-port N` at another port, and `-dryrun` prints the command line without starting
anything.

Two pages come with it, from the same process and the same port:

| Page | Address |
|---|---|
| llama.cpp's chat UI | `http://127.0.0.1:18200/` |
| This fork's monitoring view | `http://127.0.0.1:18200/kvmem/` |

## Measured on an RTX 4060 Laptop GPU

One card (8 GB, 24 SMs), CUDA 13.4, `sm_89`, the Bonsai2-T5-MTP-r3-Q4 artifact (Ternary Bonsai 2
27B, PTQ1 with MTP r3, 24 query heads over 4 KV heads, head_dim 256). One prompt and one 160-token
answer per KV coding, at a 17,088-token KVMem budget:

| KV coding | KV runtime | free VRAM | prefill | decode | TTFT |
|---|---|---|---|---|---|
| `rk4v4-e8` | 925.6 MiB | 522 MiB | 213.1 tok/s | **68.3 tok/s** | 349 ms |
| `rk4v4` | 925.6 MiB | 522 MiB | 213.4 tok/s | 66.3 tok/s | 343 ms |
| `nvfp4` | 935.5 MiB | 512 MiB | 198.8 tok/s | 66.5 tok/s | 369 ms |
| `rk8v4` | **1.06 GiB** | 362 MiB | 183.5 tok/s | 64.4 tok/s | 407 ms |

`rk4v4-e8` is the launcher's default: it needs the same memory as the other 4-bit-key codings and
decodes fastest here, while `rk8v4`'s 8-bit keys cost 14% more memory for no speed. The build
carries all four, so `-kvtype` selects any of them.

Startup and serving, from a cold start:

| Step | Value |
|---|---|
| Device-route calibration | 2 min 23 s the first time on a card, then the measured profile is reused |
| Weights | 5.75 GiB, 2.4 GiB/s from the artifact |
| Engine ready | 2 min 27 s from the start of the process |
| KV pool | 25,088 tokens in 392 pages of 64 tokens |
| Runtime reservation | 935 MiB (nvfp4) to 1.06 GiB (rk8v4), with 360-520 MiB of the card left free |
| One request | TTFT 321-407 ms, decode 64-68 tok/s, MTP accepted 81-82 of 93 drafts |

The card is nearly full at the end. The Engine's memory planner refuses a budget that does not fit
and names the ones that do:

```
requested Engine runtime reservation requires 1302903040 bytes, but only 1251999744 bytes are
available for runtime capacity | KVMEM_MEMORY_PLAN {"options":[{"id":"nvfp4","budget":23040,...},
{"id":"rk8v4","budget":17088,...}]}
```

So `rk8v4` needs `--kvmem-budget 17088` on this card, while `nvfp4` and the 4-bit-key codings take
the launcher's 23,040.

## KV attention coverage

A build compiles a subset of the KV cache codings and of the head geometries, chosen with CMake
options:

```
-DNINFER_KV_ATTN_CODING_RK8V4=ON   -DNINFER_KV_ATTN_CODING_RK4V4=ON
-DNINFER_KV_ATTN_CODING_RK4V4E8=ON -DNINFER_KV_ATTN_CODING_RK2V4E8=OFF
-DNINFER_KV_ATTN_FP8=OFF           -DNINFER_KV_ATTN_NVFP4=ON
-DNINFER_KV_ATTN_K8V4=OFF          -DNINFER_KV_ATTN_GEOM_H24=ON
-DNINFER_KV_ATTN_GEOM_H16=OFF      -DNINFER_KV_ATTN_GEOM_G4=OFF
```

The product default is every coding and geometry ON. The values reach the units that read them as
compile definitions, and a unit that reads them without them is a compile error rather than a
silent "everything is compiled": a unit that believed in a coding this build does not carry would
select a kernel that was never compiled.

**The build profile and the launcher must agree.** A `--kv-dtype` the build does not carry is
refused at startup, before any weight is uploaded:

```
FATAL server failed during startup | this build was compiled without the fp8 KV cache coding
(compiled: bf16, int8, rk8v4, nvfp4, rk4v4, rk4v4-e8)
```

Turning a coding on is one flag in `scripts/build-ninfer-4060.bat` and a rebuild; the rebuild
recompiles the KV attention units, not the whole engine.

The head geometries follow the model: this artifact has 24 query heads over 4 KV heads (group 6),
which is the h24 geometry, so h16 and g4 are unreachable here and their kernels are left out.

## Repository layout

```
scripts/            the Windows build, launch and WebUI fetch; upstream's release and pod scripts
src/ops/softmax_attention/   the KV attention: codings, geometries and their coverage switches
src/serve/         the HTTP surface: OpenAI, Anthropic, llama.cpp and the monitoring endpoints
src/serve/webui/   the WebUI embedder and this fork's monitoring view
src/runtime/       the Engine: admission, scheduling, the context cache and the KV cache
kvmem/             the portable KVMem core (upstream submodule checkout)
docs/              upstream's documentation; this branch's changes are noted in the tables above
third_party/       vendored dependencies; third_party/webui holds the unpacked llama.cpp WebUI
```

The architecture, the ownership boundaries and how to change an Op are in
[docs/maintainer/engine-architecture.md](docs/maintainer/engine-architecture.md) and
[docs/maintainer/op-development.md](docs/maintainer/op-development.md).

## Known limits

- **The WebUI is not downloaded by the build.** `scripts/fetch-webui.bat` unpacks llama.cpp's
  released WebUI into `third_party/webui` (git-ignored) and prints the directory to configure with;
  the build script then finds it by itself. The build stays hermetic, so a machine without the
  bundle gets the monitoring view and no chat UI. Upstream's `scripts/fetch-webui.sh` does the same
  on Linux.
- **Three llama.cpp-native routes are not implemented**: `/v1/chat/completions/control` (stop or
  steer a running generation), `/v1/stream` and `/v1/streams/lookup`. The chat itself uses
  `/v1/chat/completions`, which works; a cancel is a client disconnect, as it is everywhere else in
  this engine.
- **The monitoring view shows what the server already exposes.** The KVMem resident window's token
  count, the per-session list of retained sessions and the per-tier page counts are not in the
  Engine's snapshot yet, so the page cannot show them.
- **Multi-GPU is untested here.** The pipeline stages, tensor parallelism and expert parallelism
  are upstream's and untouched; this fork is single-card.
- **The test suite does not run in this profile** (`BUILD_TESTING=OFF` in the 4060 build). The
  changes in this branch are compile-time and CLI-level, and the affected runtime paths were
  exercised by hand: startup, a streamed chat completion, the refusal of an omitted coding and the
  monitoring view's own script.
- The card is nearly full at the launcher's defaults; a larger budget, a second lane or another
  resident model does not fit.

## Documentation

Upstream's documentation describes the whole product and is unchanged by this branch:

| Document | What it covers |
|---|---|
| [docs/README.md](docs/README.md) | the documentation map |
| [docs/serving.md](docs/serving.md) | the HTTP surface, the WebUI and the monitoring view, KVMem |
| [docs/cli.md](docs/cli.md) | every flag, the KV formats and their measured size and speed |
| [docs/build-windows.md](docs/build-windows.md) | the Windows toolchain |
| [docs/gpus.md](docs/gpus.md) | every card, with what has run on it |
| [docs/performance.md](docs/performance.md) | the published measurements |
| [docs/recipes.md](docs/recipes.md) | launch recipes from an 8 GB card up |
| [docs/maintainer/](docs/maintainer/) | the engine's architecture, its Op contracts, the build system |

The KV format numbers this branch measured on its own card are in
[docs/cli.md](docs/cli.md#context-and-memory) upstream; the ones above were measured here.

## Credits and license

This fork keeps upstream's [LICENSE](LICENSE) and [NOTICE](NOTICE), and credits
[CONTRIBUTORS.md](CONTRIBUTORS.md). NInfer-all continues [NInfer](https://github.com/Neroued/ninfer)
by Neroued; the KVMem line and this deployment carry the work of several forks.
