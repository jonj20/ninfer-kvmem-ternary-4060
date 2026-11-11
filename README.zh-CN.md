# NInfer KVMem · Ternary 4060

[NInfer-all](https://github.com/iamwavecut/ninfer-all) 的 KVMem 产品线在 **8 GB 单卡**上的部署形态：
一张 RTX 4060（Laptop）跑 Ternary Bonsai 2 27B，带 KVMem 上下文缓存、MTP 投机解码、llama.cpp 的
WebUI，以及本分支自己做的监控页 —— 全部在**同一个进程、同一个端口**上。

本仓库是 fork，不是另起炉灶的引擎：代码来自上游，只改这个部署真正需要的部分。

> **旧分支已废弃。** 早期版本留在 `main` 分支（<https://github.com/jonj20/ninfer-kvmem-ternary-4060/tree/main>），
> 不再维护；当前的开发在 `master`。下面的内容只描述 `master`。

- **上游：** [NInfer-all](https://github.com/iamwavecut/ninfer-all)（它延续
  [NInfer](https://github.com/Neroued/ninfer) 的 NInfer-3090 线），基线提交
  `d0da39796697b79fd08fc570e1781d75943dddb1`。
- **其上：** KVMem 产品线的 27 个补丁（`kvmem`、`ptq1`、`t5-performance`、`nvfp4-*`、
  `monitor-events`、`memory-planning`、`heads-tp` 等），按序应用。
- **再加上：** 本分支自己的改动，见下表。

上游 README 描述的是整个产品形态（多个模型家族、Volta 到消费级 Blackwell 的全部显卡、流水线阶段、
Docker 镜像、原生发布包）。那些都不是本仓库的内容，也没有在本机验证过。下面只讲这个 fork 是什么、
它做什么。

[本分支改了什么](#本分支改了什么) | [快速开始](#快速开始) | [本机实测](#本机实测) |
[KV attention 覆盖开关](#kv-attention-覆盖开关) | [目录结构](#目录结构) | [已知限制](#已知限制) | [文档](#文档)

## 本分支改了什么

| 方面 | 改动 | 位置 |
|---|---|---|
| KV attention 覆盖 | 一个 build 编译哪些 KV cache coding、哪些头几何，是编译期选项（`NINFER_KV_ATTN_CODING_*`、`NINFER_KV_ATTN_GEOM_*`）。代码保留而不删除：没编的 coding 链接一个抛异常的 stub，并拒绝该 `--kv-dtype` | `src/ops/softmax_attention/kv_attn_coverage.h`、`src/ops/CMakeLists.txt` |
| 启动期准入 | Engine 在上传任何权重之前，就按名字拒绝本 build 不带的 `--kv-dtype`，并列出它带了哪些；设备路由标定也只测本 build 编译了的 coding | `src/runtime/engine/model_instance.cpp`、`src/calibration/device_calibration.cu` |
| 4060 profile | 只编 `sm_89`；h24 几何 + bf16、int8、rk8v4、rk4v4、rk4v4-e8、nvfp4。产物 276 MB，全量覆盖是 482 MB，嵌入 WebUI 后 285 MB | `scripts/build-ninfer-4060.bat` |
| Windows 构建与启动 | 一个脚本构建服务端（MSVC + CUDA + Ninja，增量），一个脚本按本部署的 KVMem、MTP、内存参数启动它 | `scripts/build-ninfer-4060.bat`、`scripts/start-ninfer-4060.bat` |
| WebUI | 上游 llama.cpp 发布的 WebUI，嵌进服务端、由它自己的端口提供。`scripts/fetch-webui.bat` 负责解包；构建过程本身不联网 | `scripts/fetch-webui.bat`、`src/serve/webui/embed.cpp` |
| 监控页 | 本分支自研的页面，路径 `/kvmem/`（以及 `/monitor/`）：显卡负载、设备端与 host 端 KV 占用、KVMem 窗口/会话/历史/各层载荷、调度队列、在跑的请求。数据来自服务端自己的 `/stats`、`/props`、`/api/monitor/snapshot` | `src/serve/webui/kvmem/index.html` |
| llama.cpp 兼容 | `GET /tools` 回答服务端工具列表（空列表：NInfer 的工具随聊天请求一起传）；`/props` 报告当前运行所用的 `kv_dtype` | `src/serve/http_server.cpp` |
| 模型支持 | 去掉 flint 与 gemma4 家族；本部署服务的是带 KVMem 缓存的 Qwen3.5-core 模型 | 提交 `a8dc79b` |

其余都是上游的，上游文档仍然照旧描述它们。

## 快速开始

### 前置

- Windows 10 及以上，8 GB 及以上的 NVIDIA 显卡（本部署：RTX 4060 Laptop GPU）。
- Visual Studio 2022 生成工具（x64）、CUDA 12.8 或更新（Blackwell 需 13.x；本分支只编 `sm_89`），
  以及带 curl 与 FFmpeg 的 vcpkg 目录。
- 脚本默认用 `D:\dev\nvkit`（CUDA）和 `D:\dev\vcpkg\installed\x64-windows`（vcpkg），
  `-cudaPath`、`-vcpkgTree` 可覆盖。

### 构建

```bat
scripts\fetch-webui.bat                 rem 只需一次，为了聊天 UI（见“已知限制”）
scripts\build-ninfer-4060.bat           rem 增量构建；-clean 为全量重建
```

脚本总是先 configure，所以脚本里的覆盖 profile 永远是构建目录里生效的那一份；然后编译
`ninfer-serve` 到 `build-backends\ninfer\apps\ninfer-serve.exe`。`-jobs N` 设并行度（默认 8），
`-target T` 换目标，`-buildDir DIR` 换目录。

### 启动

```bat
scripts\start-ninfer-4060.bat
```

脚本以本部署的参数在 `http://127.0.0.1:18200` 启动：`--kv-dtype rk4v4-e8`、KVMem 23,040 token
预算与 12 GiB host 层、MTP 两个草稿、专家常驻设备。`-model PATH` 换 artifact，`-kvtype` 换本
build 编译了的另一种 coding，`-port N` 换端口，`-dryrun` 只打印命令行不启动。

两个页面随它一起来，同进程同端口：

| 页面 | 地址 |
|---|---|
| llama.cpp 聊天 UI | `http://127.0.0.1:18200/` |
| 本分支的监控页 | `http://127.0.0.1:18200/kvmem/` |

## 本机实测

一张卡（8 GB、24 SM）、CUDA 13.4、`sm_89`，artifact 为 Bonsai2-T5-MTP-r3-Q4（Ternary Bonsai 2
27B，PTQ1 带 MTP r3，24 个 query head 对 4 个 KV head，head_dim 256）。每种 KV coding 用同一个
提示词、生成 160 token，KVMem 预算 17,088 token：

| KV coding | KV 运行时占用 | 余显存 | prefill | decode | TTFT |
|---|---|---|---|---|---|
| `rk4v4-e8` | 925.6 MiB | 522 MiB | 213.1 tok/s | **68.3 tok/s** | 349 ms |
| `rk4v4` | 925.6 MiB | 522 MiB | 213.4 tok/s | 66.3 tok/s | 343 ms |
| `nvfp4` | 935.5 MiB | 512 MiB | 198.8 tok/s | 66.5 tok/s | 369 ms |
| `rk8v4` | **1.06 GiB** | 362 MiB | 183.5 tok/s | 64.4 tok/s | 407 ms |

启动器默认 `rk4v4-e8`：它与另外两种 4-bit key 的 coding 占一样显存，而在这里解码最快；`rk8v4`
的 8-bit key 要多花 14% 显存却不快。四种都编进了 build，`-kvtype` 可任选。

冷启动到可服务：

| 步骤 | 数值 |
|---|---|
| 设备路由标定 | 该卡第一次 2 分 23 秒，之后复用已测好的 profile |
| 权重 | 5.75 GiB，从 artifact 读入 2.4 GiB/s |
| Engine ready | 进程启动后 2 分 27.4 秒 |
| KV 池 | 25,088 token，392 个 64-token 页 |
| 运行时预留 | 935 MiB（nvfp4）到 1.06 GiB（rk8v4），卡上余 360–520 MiB |
| 单次请求 | TTFT 321–407 ms，decode 64–68 tok/s，93 个草稿接受 81–82 个 |

到最后这张卡基本占满。预算放不下时 Engine 的内存规划器会拒绝，并直接告诉你哪些可行：

```
requested Engine runtime reservation requires 1302903040 bytes, but only 1251999744 bytes are
available for runtime capacity | KVMEM_MEMORY_PLAN {"options":[{"id":"nvfp4","budget":23040,...},
{"id":"rk8v4","budget":17088,...}]}
```

也就是说 `rk8v4` 在这张卡上要 `--kvmem-budget 17088`，而 `nvfp4` 和两种 4-bit key 的 coding 可以用
启动器默认的 23,040。

## KV attention 覆盖开关

一个 build 编译哪些 KV cache coding 和哪些头几何，由 CMake 选项决定：

```
-DNINFER_KV_ATTN_CODING_RK8V4=ON   -DNINFER_KV_ATTN_CODING_RK4V4=ON
-DNINFER_KV_ATTN_CODING_RK4V4E8=ON -DNINFER_KV_ATTN_CODING_RK2V4E8=OFF
-DNINFER_KV_ATTN_FP8=OFF           -DNINFER_KV_ATTN_NVFP4=ON
-DNINFER_KV_ATTN_K8V4=OFF          -DNINFER_KV_ATTN_GEOM_H24=ON
-DNINFER_KV_ATTN_GEOM_H16=OFF      -DNINFER_KV_ATTN_GEOM_G4=OFF
```

产品默认是所有 coding 与几何全开。这些值以编译定义的形式送到读取它们的编译单元；**读不到这些值
的编译单元直接编译报错**，而不是静默地以为“全都有”：一个以为本 build 带了某 coding 的单元，会
去选一个根本没编出来的 kernel。

**build 的 profile 与启动脚本必须一致。** 本 build 没带的 `--kv-dtype` 会在启动时、上传任何权重
之前被拒：

```
FATAL server failed during startup | this build was compiled without the fp8 KV cache coding
(compiled: bf16, int8, rk8v4, nvfp4, rk4v4, rk4v4-e8)
```

打开某个 coding 就是改 `scripts/build-ninfer-4060.bat` 里一个 flag 再重新编译；重编只重编 KV
attention 相关的单元，不是整个引擎。

头几何跟着模型走：本 artifact 是 24 个 query head 对 4 个 KV head（group 6），对应 h24 几何，
h16 与 g4 在这里走不到，所以它们的 kernel 没有编进来。

## 目录结构

```
scripts/            Windows 的构建、启动、拉 WebUI；上游的发布与 pod 脚本
src/ops/softmax_attention/   KV attention：coding、几何与它们的覆盖开关
src/serve/         HTTP 层：OpenAI、Anthropic、llama.cpp 以及监控用的端点
src/serve/webui/   WebUI 嵌入器和本分支的监控页
src/runtime/       Engine：准入、调度、上下文缓存与 KV 缓存
kvmem/             可移植的 KVMem 核心（上游的子模块检出）
docs/              上游文档；本分支的改动在上面的表里标出
third_party/       内置依赖；third_party/webui 放解包后的 llama.cpp WebUI
```

架构、职责边界与怎么改一个 Op，见
[docs/maintainer/engine-architecture.md](docs/maintainer/engine-architecture.md) 和
[docs/maintainer/op-development.md](docs/maintainer/op-development.md)。

## 已知限制

- **构建不会自动下载 WebUI。** `scripts/fetch-webui.bat` 把 llama.cpp 发布的 WebUI 解包到
  `third_party/webui`（被 git 忽略）并打印要配置的目录；构建脚本随后会自己找到它。构建保持
  hermetic，所以没有这个包的机器只有监控页、没有聊天 UI。上游的 `scripts/fetch-webui.sh` 在
  Linux 上做同一件事。
- **三个 llama.cpp 原生路由没有实现**：`/v1/chat/completions/control`（停止或引导在途生成）、
  `/v1/stream`、`/v1/streams/lookup`。聊天本身走 `/v1/chat/completions`，可用；取消的方式是
  客户端断开，这与本引擎其它地方一致。
- **监控页只能显示服务端已经暴露的东西。** KVMem 常驻窗口的 token 数、每个保留会话的明细、
  各层页数都还不在 Engine 的快照里，页面因此显示不了。
- **多卡在这里没有验证。** 流水线阶段、张量并行、专家并行都是上游的、本分支没动；这个 fork 是
  单卡。
- **测试套件在这个 profile 下不跑**（4060 构建里 `BUILD_TESTING=OFF`）。本分支的改动集中在编译期
  和 CLI 层，受影响的运行时路径是手工验证的：启动、流式聊天回复、拒绝未编译的 coding、以及监控页
  自己的脚本。
- 卡在启动器默认参数下基本占满；更大的预算、第二条并发通道或另一个常驻模型都放不下。

## 文档

上游文档描述整个产品，本分支没有改动它们：

| 文档 | 内容 |
|---|---|
| [docs/README.md](docs/README.md) | 文档地图 |
| [docs/serving.md](docs/serving.md) | HTTP 面、WebUI 与监控页、KVMem |
| [docs/cli.md](docs/cli.md) | 全部参数、KV 格式及其实测体积与速度 |
| [docs/build-windows.md](docs/build-windows.md) | Windows 工具链 |
| [docs/gpus.md](docs/gpus.md) | 每一张卡，以及在它上面跑过什么 |
| [docs/performance.md](docs/performance.md) | 已发布的实测数据 |
| [docs/recipes.md](docs/recipes.md) | 从 8 GB 卡到最大上下文的启动配方 |
| [docs/maintainer/](docs/maintainer/) | 引擎架构、Op 契约、构建系统 |

本分支自己在这张卡上测到的 KV 格式数据在上面的表格里；上游在 3090 上的那份见
[docs/cli.md](docs/cli.md#context-and-memory)。

## 致谢与许可

本 fork 保留上游的 [LICENSE](LICENSE)、[NOTICE](NOTICE) 与 [CONTRIBUTORS.md](CONTRIBUTORS.md)。
NInfer-all 延续 Neroued 的 [NInfer](https://github.com/Neroued/ninfer)；KVMem 产品线与本部署承载了
多个 fork 的工作。

[English](README.md)
