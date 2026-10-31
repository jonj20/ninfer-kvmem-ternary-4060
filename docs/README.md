# NInfer documentation

Start with the [project README](../README.md) to download the prebuilt Windows binaries or build
NInfer, download a model, and run the CLI or HTTP server. This fork targets one NVIDIA GeForce
RTX 4090 (`sm_89`) on native Windows.

## User guides

| Document | Purpose |
|---|---|
| [Native Windows port](../WINDOWS_PORT.md) | Windows build details, Qwen3.8 measurements and experiments on the RTX 4090 |
| [Prebuilt releases](https://github.com/JGamboa/ninfer-4090-windows/releases) | Windows x64 binaries with DLLs, launchers and licenses |
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Comparison with llama.cpp](llamacpp-comparison.md) | same-machine prefill comparison on the RTX 4090 (2026-09-26) and the earlier Linux comparison |
| [Performance (upstream, RTX 5090)](performance.md) | the upstream engine's RTX 5090 serving measurements and publication rules; RTX 4090 figures are in the README and WINDOWS_PORT |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

Artifacts published for this fork (RTX 4090):

| Model | Weights | Download |
|---|---|---|
| Ternary Bonsai 2 27B (text + vision + MTP) | `t5_g128_fp16` ternary | [Hugging Face](https://huggingface.co/jgamboa/Ternary-Bonsai-2-27B-NInfer-4090) |
| Qwen3.8-27B, int8 prefill | official `groupwise-int` weights with `AllowA8` | [Hugging Face](https://huggingface.co/jgamboa/Qwen3.8-27B-NInfer-4090) |
| Swift 1.5 Qwen3.8-27B, int8 prefill | `groupwise-int` with `AllowA8` | [Hugging Face](https://huggingface.co/jgamboa/Swift-1.5-Qwen3.8-27B-NInfer-4090) |

Upstream artifacts (the `groupwise-int` ones run on the RTX 4090; `nvfp4` needs Blackwell):

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Ternary Bonsai design notes](maintainer/bonsai-ternary-design.md) | ternary format, kernels, speculation work and every measurement of this fork (section 9.1) |
| [Ternary Bonsai conversion](maintainer/bonsai-ternary-conversion.md) | Prism GGUF reading, tensor mapping and conversion checks |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | candidate selection, retention, materialization and Device/Host checkpoint policy |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear benchmark](maintainer/linear-benchmark.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |
| [N-gram speculation plan](maintainer/ngram-speculation-plan.md) | temporary: where phase 1 (`--ngram chain`) is documented, and the optional n-gram-only backend and verification wider than 16 |
| [MTP × KVMem windowing](maintainer/kvmem-port.md) | staged: `--spec mtp` shares the retrieval window with the host KV tier (kvmem-port.md stage 6a landing record) |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.
