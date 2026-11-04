# NInfer Ternary KVMem for Windows

A C++20/CUDA inference engine for **one NVIDIA GeForce RTX 4060 Laptop** (`sm_89`), built and run
natively on **Windows 11** (MSVC + CUDA; no WSL, no Docker). It serves Qwen3.8-27B and Ternary
Bonsai 2 27B through a CLI and an OpenAI- and Anthropic-compatible HTTP server.

**KVMem** is the feature that makes an 8 GB card usable at long context: the logical context may be
far larger than the pool that stays resident on the GPU, and each round retrieves the window it
needs. Measured on an RTX 4060 Laptop (8188 MiB) with
`scripts/start-ninfer-4060-kvmem.bat`: **258,939 tokens** prefilled and generated in one request
(364 s, no `context_length_exceeded`, 37 tok/s decode without MTP) while Device KV residency fell
from 1024 pages (65,536 tokens) to 559 and Host KV grew to 3.89 GiB. The same context placed
densely needs 4.25 GiB of KV plus 5.52 GiB of weights and does not fit.

Ternary Bonsai 2 27B (Prism ML, ternary weights, text + vision, 6.4 GiB) is the recommended model
here. The Qwen3.8-27B artifacts are 19.0 GiB and need KVMem + a Host KV tier on this card; the
int8-prefill one runs prompt processing on int8 tensor cores. **Every figure below was measured on
an RTX 4060 Laptop (8188 MiB) under Windows 11.**

- [Quick start](#quick-start)
- [Measured performance](#measured-performance)
- [Settings](#settings)
- [Limits](#limits)
- [Further reading](#further-reading)

## Quick start

| | |
|---|---|
| GPU | RTX 4060 Laptop, 8188 MiB (`sm_89`). The build targets this card only. |
| OS | Windows 11 x64 |
| Toolchain | MSVC Build Tools (2026 validated), CUDA 12.8+ (13.4 validated), CMake 3.28+, Ninja |
| Libraries | [vcpkg](https://github.com/microsoft/vcpkg) (`curl`, `ffmpeg`, `pkgconf`) |
| Download tool | [`hf`](https://huggingface.co/docs/huggingface_hub/guides/cli) |

### Prebuilt

Download `ninfer-4060-windows-x64-<date>.zip` from the
[latest release](https://github.com/JGamboa/ninfer-4060-windows/releases/latest). It needs only an
NVIDIA driver 595+ and the
[VC++ 2015-2022 x64 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe); CUDA is
statically linked.

### Build from source

```bat
git clone https://github.com/jonj20/ninfer-kvmem-ternary-4060
cd ninfer-kvmem-ternary-4060
scripts\build_4060_win.bat
```

The script locates MSVC, CUDA, vcpkg and Ninja, then configures and builds into `build_4060\`.
It is incremental by default and accepts `-configure`, `-clean` and `-jobs N`; the toolchain paths
are constants at the top of the script. Produces `build_4060\apps\ninfer.exe`,
`build_4060\apps\ninfer-serve.exe` and `build_4060\apps\ninfer-perplexity.exe`. See
[WINDOWS_PORT.md](WINDOWS_PORT.md) for the Windows-specific changes.

### Download a model

```bat
hf download jgamboa/Ternary-Bonsai-2-27B-NInfer-4090 bonsai2_27b_vl_mtp_q4q5.ninfer --local-dir E:\LLM
```

Verify each download against the checksums on its model card.

### Command line

```bat
build_4060\apps\ninfer.exe E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer ^
  --prompt "Write a Python function that merges two sorted lists." ^
  --max-context 8192 --max-new 1024 --spec mtp --draft-tokens 2 --lm-head-draft
```

The summary reports prefill and decode speed, MTP acceptance and memory use. Chat histories,
images and video go through `--messages FILE.json`
([examples/cli/messages](examples/cli/messages)). All options: `ninfer.exe --help`,
[docs/cli.md](docs/cli.md).

### Server

```bat
build_4060\apps\ninfer-serve.exe E:\LLM\qwen3_8_27b_a8.ninfer ^
  --host 127.0.0.1 --port 8080 --model-id qwen3.8-27b ^
  --max-context 100000 --kv-capacity 65536 --kv-ring --kvmem-budget 32768 ^
  --kv-dtype rk4v4-e8 --max-concurrency 3 ^
  --max-pending-requests 10 --pending-timeout-ms 600000 --prefill-chunk 1408 ^
  --spec mtp --draft-tokens 3 --lm-head-draft --ngram chain --preserve-thinking ^
  --device-state-slots 3 --host-state-slots 4 --host-kv-mib 4096
```

The memory plan is checked before the socket opens, so an infeasible configuration fails at startup
rather than at request time.

- `http://127.0.0.1:8080/v1` (OpenAI Chat Completions, Responses) and
  `http://127.0.0.1:8080/v1/messages` (Anthropic Messages).
- `/monitor` dashboard, `/metrics` (Prometheus), `/slots` (llama.cpp-style table).

## Measured performance

`scripts\start-ninfer-4060-kvmem.bat` with `bonsai2_27b_vl_mtp_q4q5.ninfer`, Windows 11, CUDA 13.4,
no display attached. The script defaults to `-drafts 0` and a 32,768-token window.

```bat
scripts\start-ninfer-4060-kvmem.bat
scripts\start-ninfer-4060-kvmem.bat -select retrieval
```

| Measurement | Result |
|---|---:|
| Decode, no MTP | **37 tok/s** |
| Logical context prefilled and generated in one request | **258,939 tokens** |
| Wall time | 364 s |
| `context_length_exceeded` | none |
| Fact placed at the start, slid out of the window | still answered correctly |
| Device KV pages, start to end | 1024 -> 559 (pool cap 1024 = 65,536 tokens) |
| Host KV at end | 3.89 GiB |
| Prefill throughput, average vs peak | 711 against 1020 tok/s |

Config: `--max-context 262144 --kv-capacity 65536 --kv-ring --kvmem-budget 32768`. A 36,639-token
run gave the same answer after two window slides.

For per-model tok/s, measure locally with `build_4060\bench\ninfer_bench.exe` rather than trusting a
number from another card; see [bench/README.md](bench/README.md).

## Settings

**Speculation.** Drafted tokens are verified by the model, so quality is unchanged and only speed
moves. Bonsai: `--spec mtp --draft-tokens 2 --lm-head-draft --ngram chain`. Qwen3.8:
`--draft-tokens 3`. `--draft-tokens 3` is faster on code and math, slower on prose. DFlash2 is
faster on prose but loads 1.6 GiB more weights, which is tight at 8 GB, and it does not support
Bonsai's ternary output head. `--ngram chain` copies context text into drafts (useful for code, JSON
and tool calls) at the cost of a second set of CUDA graphs.

**KV cache.**

| `--kv-dtype` | Use it for |
|---|---|
| `rk4v4-e8` (4-bit E8-lattice keys, 4-bit values) | Default here. Best balance for 8 GB. |
| `rk2v4-e8` | More headroom (2-bit keys), about 10 % slower decode |
| `int8` | Maximum precision; very tight with Qwen3.8 |

**KVMem.** `--kvmem-budget N` (default 32768, `0` disables windowing) bounds the Device KV window to
the selected blocks and keeps the rest in the pinned Host KV tier. Recency keeps a sink prefix plus
the newest tail; `--kvmem-select retrieval` also returns the best-scoring interior blocks, scored by
mean-key similarity to the newest completed block. Retrieval acts on ordinary decode only; prefill
and MTP verify force recency because their contiguous call ranges must stay in one retained run.

| Constraint | Why |
|---|---|
| `--kv-capacity >= budget + --prefill-chunk` | window plus the chunk in flight must fit |
| `--kvmem-budget >= 2 * 128 + max(prefill chunk, verify range)` | one contiguous call must fit the recency tail |
| `--kv-capacity < --max-context` requires `--kv-ring` | otherwise startup rejects the pool |
| Host KV tier required | a demoted page needs a Host replica; `--host-kv-mib 4096` sufficed at 256K |

`gen_reserve` is derived: `gen_reserve = --kv-capacity - --kvmem-budget`. It adds no reusable
checkpoints, so prefix-reusing requests stay bounded by `--kv-capacity`.

**Concurrency.** `--max-concurrency` (1-8) sets the lanes; two lanes is a practical maximum for
Bonsai, one for Qwen3.8 with KVMem. Prefill runs one request at a time, so a long prompt delays the
other lanes.

**Display.** If the 4060 Laptop also drives the monitor, the Windows compositor takes the GPU from
CUDA every frame, costing 15-20 % of decode. Route the monitor to the integrated GPU if you can,
otherwise use 60 Hz and keep animated windows still. Compare tok/s only between runs with the same
display setup.

## Limits

- One RTX 4060 Laptop, one process, one resident model. No multi-GPU, no weight offload, no request
  preemption or priorities.
- Qwen3.8-27B (19.0 GiB) needs KVMem + Host KV; Bonsai (6.4 GiB) is the recommended model here.
- `--kvmem-budget` keeps a bounded Device window. Retrieval acts on ordinary decode only; prefill and
  MTP verify stay on recency, and DFlash draft pools are not windowed.
- Window slides move pages synchronously across the Device/Host boundary, about 30 % of prefill
  throughput in the 256K run.
- `--host-kv-mib 0` does not auto-size with `--kv-ring` plus `--kvmem-budget`; pass an explicit size.
- NVFP4/W4A4 needs Blackwell tensor cores and is unavailable on `sm_89`.

## Further reading

| Topic | Document |
|---|---|
| All documentation | [docs/README.md](docs/README.md) |
| CLI options | [docs/cli.md](docs/cli.md) |
| HTTP server and protocols | [docs/serving.md](docs/serving.md) |
| Windows build and measurements | [WINDOWS_PORT.md](WINDOWS_PORT.md) |
| KVMem design and the 256K run | [docs/maintainer/kvmem-port.md](docs/maintainer/kvmem-port.md) |
| Ternary Bonsai design and kernels | [docs/maintainer/bonsai-ternary-design.md](docs/maintainer/bonsai-ternary-design.md) |
| Model conversion | [docs/maintainer/bonsai-ternary-conversion.md](docs/maintainer/bonsai-ternary-conversion.md), [docs/weight-conversion.md](docs/weight-conversion.md) |
| Benchmarks and tests | [bench/README.md](bench/README.md), [tests/README.md](tests/README.md) |

## Credits

Native Windows build for the RTX 4060 Laptop, with Ternary Bonsai 2 27B (converter, ternary format
and kernels, vision), n-gram speculation, concurrent lanes, DFlash2 verification routes, and the
**KVMem** bounded Device KV working set with Host KV tiering.

Ternary Bonsai 2 27B, its packings and Hadamard rotation are by
[Prism ML](https://huggingface.co/prism-ml); their
[llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) defined the formats this branch reads.

## License

Apache License 2.0. See [LICENSE](LICENSE).