# KVMem 移植

本文是唯一权威：(1) 参考实现 KVMem 的设计；(2) 移植到 NInfer 的计划。
Part 1 是外部设计的稳定参考，Part 2 是进行中的移植计划。

参考源：`llamacpp/llama.cpp-kvmem-bonsai2-mtp/runtime-kvmem/kvmem-v0.16.0-rc3-source/source`
（`kvmem/` 主机侧逻辑、`src/adapter/` llama.cpp 接入、`docs/architecture.md`）。

---

## Part 1 — KVMem 设计总结

### 1.1 一句话

不把全量历史装进显存，而是让 attention 只看一个**有界的工作集**，历史全放主机。

### 1.2 问题定位与两条路线

长 agent 会话（几百 K ~ 1 M token）的 KV 显存开销，小卡装不下。两种思路：

- **KV streaming**（全量 attention）：显存放一部分、其余在主机，逐层预取。保精度，但
  attention 工作量和 PCIe 流量随上下文线性增长，decode 逐渐变慢。
- **KVMem**（检索式）：显存只放一个有界窗口，每轮用当前 query 检索出相关历史块。
  attention 成本不再随历史长度增长。

论文（KVMem, arXiv 2609.04852）：256K 历史、只留 **32K GPU 活动上下文**，
LongMemEval-S 85.6% vs 86.6%（全量），AgentLongBench 60.9% vs 59.5% —— 接近无损。

### 1.3 三层存储与分块粒度

```
GPU   有界 block-slot 池，大小 = --kvmem-budget + --kvmem-gen-reserve
CPU   pinned 主机 RAM，存已完成的历史块
SSD   NVMe 层 —— 本移植版未实现
```

- 逻辑块 = **128 token**（`--kvmem-block-tokens`），独立于 llama.cpp 的 16 token 物理页。
- 每个逻辑块占池中 `block_tokens` 个 cell；cell 的 `pos` = **原始单调 token 位置**，
  槽号不是 RoPE 坐标。
- 块填满时把 packed KV **异步拷到主机**，与后续 prefill 重叠；驱逐时若拷贝已在则跳过。

### 1.4 选择（核心）

每轮 agent step 用当前 query 对历史块打分，取 top-k 放进 GPU 工作集，**按时间顺序**排列：

| 信号 | 行为 |
|---|---|
| Retrieval（默认） | query × 块 **mean-K** 相似度，可复活已丢弃的块（Quest/InfLLM 式） |
| H2O | 窗口内累积 attention 热度，只能保留、不能复活 |
| Recency | 无信号，只留 sink + recent |

- **sink** = 永远保留的前缀块；**recent** = 永远保留的尾部块。
- 打分用 **mean-K**（F32、pre-RoPE、首次写入时捕获）—— 便宜。
- 打分、块表、迁移 diff 全是**纯主机逻辑**：不碰 GPU，可单测。

### 1.5 迁移与复用

- 每轮产出 `KvMemPlan` **diff**：常驻选中块保持原槽不动，只有 `stage_out` 被释放、
  只有 `stage_in` 被拷贝 —— 重选只搬变化的块。
- 冷 stage-in 走 **32 MiB GPU slab** 批量 H2D；stage-out 是 gather kernel 打包后一次 D2H。
- 块在 GPU 里是**无拷贝的仓库**：窗口槽变化时原地 re-RoPE（`rope_block_remap`）；
  旋转次数或累计位移超阈值就从主机 raw-K 镜像重建（`immutable_source_k` 刷新策略）。
- **Flash Attention 不改。**
- 只有标准 attention 层参与；DeltaNet/GDN 循环层（Qwen3.6/3.8 的 3/4 层）存 O(1)
  状态，完全不动。

### 1.6 刻意的取舍（理解这些才懂它的边界）

1. **attention 不覆盖全量历史** —— 覆盖的是检索出的窗口。这是拿质量换显存的交易，有论文背书。
2. **`gen_reserve` 卡死单次生成长度**（IQ4 配方 12288 token，含 thinking）。检索会
   **pin** 住窗口，新 token 只能用预留槽；槽满直接报错，而不是驱逐检索块。
3. **decode 期间窗口被 pin**，不做 recency 重选 —— 否则刚捞出来的"针"会立刻被丢掉。
4. **跨轮复用**：客户端每轮发全量 `messages`，server 按 token LCP 复用 GPU 前缀。

### 1.7 对"单请求 262144"目标的含义

`-c 262144` 是**逻辑工作区**（历史存主机）；显存只承载 `budget + gen_reserve`。
上下文长度与显存**解耦** —— 但 attention 覆盖的是检索窗口，不是全部 262144 token。

---

## Part 2 — 移植计划

### 2.1 目标与验收标准（先定死，避免跑偏）

**目标语义**：单请求的**逻辑上下文**可达 262144；显存只常驻一个有界工作集
（`budget`），其余历史在主机；每轮用检索决定哪些块进窗口。

**明确不是**：把 262144 token 的全量 attention 塞进 8 GB（物理上不可能，
262144 × 17408 B = 4.25 GiB KV + 5.52 GiB 权重 > 7.99 GiB）。

**验收**（按序达成）：
1. 单请求 `--max-context 262144` 能 prefill 完整 262144 token 并生成，不报
   `context_length_exceeded`。
2. 任一时刻 device KV 常驻 ≤ `budget + gen_reserve` 页，host 池懒 pin 有上界。
3. 检索正确性有独立 oracle 覆盖（见 2.8）。
4. 窗口外的历史可被后续轮次重新检索到（跨轮复用成立）。

### 2.2 现状盘点：NInfer 已经具备什么

调研结论（每条都有代码位置）：

| 能力 | 状态 | 证据 |
|---|---|---|
| stage-out（device→host D2H） | **已有** | `paged_kv_cache.cpp:605-617`（`cudaMemcpy2DAsync`），调用方 `materialization.cpp:1574-1575`（压力降级）、`:1162-1163`（前缀 fork 保留尾部） |
| stage-in（host→device H2D） | **已有** | `paged_kv_cache.cpp:658-670`，调用方 `materialization.cpp:1029-1036`，先 `reserve_device_replica`（`kv_store.h:458-470`）拿到**池中全新的物理页** |
| 逻辑存活期间驱逐 device 副本 | **已有** | `kv_store.h:494-514` `drop_device_replica`：只要求 `active_references==0 && writer_references==0`，**不要求 `address_references==0`**；host 副本维持描述符存活（`kv_store.h:702-708`） |
| 端到端验证 | **已有** | `test_engine_prefix_real.cpp:555-590` 断言 `main_kv_d2h_pages` / `main_kv_h2d_pages` 增长 |
| pinned 懒增长 arena | **已有** | `src/core/host_kv_arena.*`（本轮 `--kv-ring` 工作已改） |
| 逻辑页状态模型 | **已有** | `kv_store.h:243-832`：generation + content_epoch + committed_columns + 三态副本（device/pending/host）+ 各类引用/锁 |
| 主机策略"无保留/无调度策略" | **已有** | `host_kv_store.h:50-51` 明确声明，职责边界干净 |

**结论**：KVMem 的 **CPU 层和搬运机制 NInfer 基本已经有了**，而且比 KVMem 更严谨
（epoch/coverage 校验、RAII、事务化）。缺的不是搬运，是**选择**和**可见性**。

**已有但受限的两处**：
- 放置只在 admission/capture/finish/explicit-release 边界发生，
  **decode 期间不做周期性升降级**（`resource-scheduling-and-context-cache.md:931-932`）。
- **正在活跃解码的序列的页永远不会被降级**（`materialization.cpp:1549-1551` 拒绝
  `has_active_reference`）。KVMem 要求 decode 期间窗口可被替换 —— 这是必须改的行为。

### 2.3 关键缺口（按阻塞程度排序）

1. **attention 的可见集是稠密前缀 `[0,p]`**
   契约在 `include/ninfer/ops/softmax_attention.h:15-16, 118-124`：
   "query head h attends cache rows **[0,p]** through table row kv_table_rows[b]"。
   key 的位置**由逻辑行号隐含给出**，没有 per-key 位置或有效性输入。这是最上游的阻塞。
2. **映射是 append-only 稠密前缀**
   `storage/kv_store.h:1440-1477` `ensure_mapped_to_tokens`：`if (target <= page_count) return;`
   只往前长；`commit_frontier` 单调（`:1479-1497`）；`destructive_truncate` 只裁尾。
   **活跃序列的逻辑页永不移动。** 重选需要新的 store 原语，不是一个开关。
3. **没有任何检索打分机制**
   全 `src/` 下 grep `attn_score|mean_k|heat|retrieval_score` 为空。
   唯一相关的是 `state/decoder_state.cpp:27-29` 的预告注释：
   "ring leaves unmapped logical pages for the retrieval mask to hide"
   —— 说明原作者已经预留了 retrieval mask 的位置，但 mask 本身不存在。
4. **没有每轮 reselect 钩子**
   `engine_core.h:1738-1747` `ensure_base_plan()` 把 `request->base_plan` 记忆化，
   **只在 admission 时跑一次**。调度器每轮重算成员（`scheduler.h:146`），
   但没有 per-request 的重选点。
5. **页数是 frontier 的函数**
   `TargetKVRequirement{main_pages, backend_pages}`（`program.h:97-105`）、
   `text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens)`
   （`request_plan.cpp:270-279`）、`kv_pages_for_frontier`（`pressure.cpp:503`）、
   目录有效性 `required_kv.main_pages != 0`（`resource_manager.h:1549`）。
   有界工作集下这套记账全部要重新定义。
6. **MTP draft KV 是独立的稠密池**
   `startup.cpp:145-146`（`text_physical_page_groups` / `mtp_physical_page_groups` 分离）、
   `decoder_state.cpp:70,74`、`text.cpp:376-391` draft 也读 `[0,p]`。
   **只缩小 target 工作集，draft 池会立刻成为新的显存天花板。**
7. **CUDA Graph —— 这是好消息**
   block table 的**内容**在 graph 外上传（`paged_kv_cache.cpp:790-805` `publish_indices`），
   graph 只捕获**矩阵地址**（`decoder_state.cpp:134`），入口是 `decode.cpp:34` 的
   `kernel_copy_async`（ingress，也在图外写好）。⇒ **重选逻辑→物理映射不需要重捕获。**
   唯一约束：`causal_softmax_attention_topology_class`（`softmax_attention.h:162-164`）
   由 head 数/宽度/batch/storage/**`envelope.max_visible_keys`** 决定
   （`causal_softmax_attention.cpp:345-386`）—— envelope 跨阈值会翻转 class，图就不对了。

### 2.4 核心设计决策：可见性怎么表达（二选一）

这是整个移植的技术分岔点，必须先定。

#### 路线 A：窗口重排 + re-RoPE（KVMem 的做法）

把选中块**按时间顺序打包进连续窗口 `[0, W)`**，并把 K 从原位置重新旋转到窗口槽位。

- ✅ **attention 契约完全不变**（仍是稠密前缀 `[0,p]`），不新增 Op、不改 oracle。
- ✅ 与 NInfer 现有 `ensure_mapped_to_tokens`、graph profile、entitlement 记账天然兼容。
- ❌ 需要新增 re-RoPE CUDA op（`de-rotate(baked_pos) + re-rotate(new_pos)`）。
- ❌ **每次搬动都有数值损耗**：旋转会重复舍入。稀疏选择还会改变**跨块相对距离**
  （真位置相差 4000 的两块，打包后只差 128），RoPE 分数就变了 —— 这是**有意的近似**。
  KVMem 用 `remap_count` / `remap_abs_delta` 阈值触发从 raw-K 镜像重建来兜底
  （`kvmem_store.hpp:57-64, 282-284`）。
- 补充事实：RoPE 是**相对**的（`R_m^T R_n = R_{n-m}`），所以**常数平移不改变任何分数**。
  若窗口是真位置的连续区间，重排是**精确**的；只有**非连续**选择才有损。

#### 路线 B：给 attention 传可见数量/位置（数值精确）

K 按真位置**只烘焙一次，永不再旋**；Q 同样按真位置烘焙，点积精确。
只需让 causal mask 用真位置而不是逻辑行号。

- 关键化简：选中块按真位置**升序**排列 ⇒ 每个 query 的可见集是该序列的**前缀**
  ⇒ 传一个 **`visible_count[T]`**（每 query 一个 int32）就足够，不需要 per-key 位置数组。
  密集场景下 `visible_count[r] = r + 1`，与现有语义等价。
- ✅ **零数值损耗**，无 re-RoPE 漂移，不需要 raw-K 镜像，实现反而更简单。
- ✅ prefill 压力窗口（sink + 已处理尾部）同样是升序 ⇒ 仍是前缀，成立。
- ❌ **改公开 Op 契约**：`positions` 输入的语义要拆分（现在它同时承担
  "query 位置 → key 行上界"和"融合 append 的缓存寻址"）。
  按 `AGENTS.md` / `op-development.md`：需要新 oracle 判据、新 topology class、
  新 graph profile、同步更新所有调用点与文档。
- ❌ blast radius 大：`causal_softmax_attention`（prefill/decode/verify）、
  `causal_softmax_attention_cached`、MTP 路由（`text.cpp:376-391`）都要过一遍。

#### 判定倾向

| 维度 | 路线 A | 路线 B |
|---|---|---|
| 数值正确性 | 近似（重排有损，有兜底机制） | 精确 |
| 新增 CUDA op | 1 个（re-RoPE，需完整 oracle 资格） | 0 个 |
| 改公开契约 | 不改 | 改（判据/类/图全跟着改） |
| 与现有记账兼容 | 完全兼容 | 要改 5 处稠密记账 |
| 与 KVMem 参考实现一致性 | 高（同一思想） | 低（但更正确） |

**倾向：路线 B。** 理由：NInfer 的 `positions` 本来就已经是 per-query 张量，
把"上界"从 `p` 换成 `visible_count` 是**局部语义变更**而非新增维度；
换来的是零漂移（不需要 raw-K 镜像、不需要 refresh 策略、不需要 re-RoPE op 的完整
数值资格）。而路线 A 的 re-RoPE op 要按 `op-development.md` 走 FP64 oracle 全流程，
代价并不小，且**永久引入一个有损环节**。

> 此决策待用户确认（见 2.9 开放问题 Q1）。

### 2.5 目标架构（按路线 B 陈述）

```
                        ┌─────────────────────────────────────────┐
   每轮 agent step      │  Host (纯逻辑, 可单测)                   │
  ┌──────────────────┐  │  BlockStore: 128-token 块表, 真位置      │
  │ query (最后 user) │─▶│  Selector: sink + recent + top-k         │
  └──────────────────┘  │  ScoreStore: mean-K 累积 / 检索打分        │
                        │  Plan = {stage_in, stage_out, visible[]} │
                        └───────────────┬─────────────────────────┘
                                        │ plan
                        ┌───────────────▼─────────────────────────┐
                        │  Program 事务层 (已有)                   │
                        │  ensure_mapped_subset()  ← 新原语         │
                        │  D2H 降级 / H2D 升级  (已有)              │
                        │  publish_indices 到 block table (已有)    │
                        └───────────────┬─────────────────────────┘
                                        │ 不重捕获 (图只存矩阵地址)
                        ┌───────────────▼─────────────────────────┐
                        │  Attention (契约扩展)                    │
                        │  visible_count[T] 替代 positions 作为上界  │
                        │  K/Q 仍按真位置烘焙 —— 零重旋转            │
                        └─────────────────────────────────────────┘
   GDN/循环层: 完全不动 (48/64 层 O(1) 状态)
```

存储分层沿用现有两级：**device 池（工作集）+ host pinned arena（历史）**。
KVMem 的 NVMe 第三级不在范围内。

### 2.6 分阶段实施（每阶段有独立可验证产出）

> **当前进度**（2026-10-05）：阶段 0/1/2/3 + **阶段 4** + **阶段 5 全部** + **层级释放** 已完成 ——
> `KVMemBlockStore`（阶段 1）、`cache_positions` 窗口坐标与 attention 契约注释（阶段 2）、
> `set_window` 执行表重发布（阶段 3）、每轮 reselect 钩子与 rel delta 全链路、
> `--kvmem-budget` CLI（0=恒等窗口，非 0 需 128 正倍数）、层级释放：
> 窗口洞页的 stage-in/stage-out 执行器 `apply_window_transition`（`context.cpp`，
> 单次 `transfer_stream` 同步）、capture 快照前的 `stage_in_all_holes` + `hold_window_holes`
> （abort 走 `release_window_holes`）、阶段 0 记账字段落地（见 2.10）、
> 以及阶段 5 后半的长 prefill：entitlement 逻辑/物理拆分、resident-aware 准入、
> 渐进映射与启动校验（落地记录见 2.6 阶段 5）。
> `--kvmem-budget 0` ⇒ 恒等窗口，行为与移植前逐位一致；**2026-10-05 起 `ninfer-serve` 默认
> 32768**（Q4，见该表），`--kvmem-budget 0` 可显式退回稠密。
> **阶段 4（打分采集）已完成**（落地记录见 2.6）：Op `block_sum_by_position`、pre-RoPE K 捕获
> 接线（prefill `Single` / decode `SequenceBatch`，decode 走 CUDA Graph 捕获）、Program 侧
> `accumulate`/`finalize` 合入、`kvmem_block_shares` 块分区、FP64 oracle 与位置不变性单测。
> 未做：阶段 5 的长 prefill 压力选择 `pick_prefill_pressure_blocks`
> **按 W7 决定仅做文档映射**（见 2.6），搬运**按 W6 决定保持同步**、不引入异步 round 等待条件；
> 阶段 6；阶段 7 的文档与 `--help` 已完成。
> **窗口滑动首次真机跑通并修掉两个必崩缺陷**（见 2.6 阶段 5 落地记录追加段）；
> **2.1 四条端到端验收全部通过**（258,939 token 一次性 prefill + 生成，device 常驻
> 始终 ≤ 65536 token = 1.14 GiB，其余 3.89 GiB 落 Host；跨轮检索复用成立）。
> resident bound（`request_plan.cpp:222`）已按 2.10 更新：staging 生效时放宽到逻辑容量，
> 非 staging（`--kvmem-budget 0`，含 `--kv-ring`）保留原拒绝行为。
> 证据：`test_context_store.cpp` 新增 staged 场景（capped 激活、渐进映射、窗口滑动与
> stage-out、池耗尽 `bad_alloc`、释放归零），全量 `ctest` 134 项中 132 通过
> （2 项失败为本机缺 `jinja2` 与 GDN kernel cooperative launch 超限，均与本改动无交集）；
> 端到端 262144 prefill 实测因本机无 `out/qwen3_6_27b.ninfer` 未运行。

**阶段 0 — 冻结语义与记账口径**
- 写清：`budget`（检索窗口 token 数）、`gen_reserve`（生成预留）、
  `block_tokens`（=128 = 2 个 64-token 逻辑页，与 NInfer 页粒度对齐）、
  `logical_capacity`（= `max_context`）。
- 重定义页记账：`pages` 不再是 `frontier` 的函数，而是 `resident_pages` +
  `host_only_pages` 之和；改 `TargetKVRequirement`、entitlement、`kv_pages_for_frontier`。
- 产出：本文件 2.6 定稿 + `request_plan.cpp` / `program.h` 的记账口径说明。
- **验收**：设计文档能回答"任意时刻 device 需要多少页、由谁决定"。

**阶段 1 — 纯主机模块（不碰 GPU）**
- 新模块 `BlockStore` / `Selector` / `ScoreStore`：块表、真位置、sink/recent、
  top-k 选择、`KvMemPlan` diff（stage_in / stage_out / visible_count 推导）。
- 与 `HostKVExtentStore` 的职责区分：extents 只管字节，block store 只管语义。
- **验收**：新增纯主机测试（仿 `tests/test_kv_cache.cpp` 的 `check()`/`expect()` 风格，
  注册进 `tests/cmake/CoreTests.cmake`），覆盖：追加/截断、sink/recent 边界、
  top-k 预算、diff 只产生增量、升序→前缀可见性推导正确。

**阶段 2 — 把窗口坐标喂进 `cache_positions`（路线 B，见 2.12）**
- 已完成：`KVMemWindow`（`block_tokens`/`total_tokens`/`retained_pages`/
  `row_base_by_block`/`rel()`/`retains()`）由
  `KVMemBlockStore::window_for(plan, call_min, call_max)` 给出，含 I1/I2/预算三重断言，
  单测覆盖"重编号前缀 ⟺ 绝对保留集"的双向序关系、恒等退化、洞内调用拒绝、
  frontier 单块预算、interior top-k、非 128 块宽、以及块起点页对齐。
- `TextContext` 侧：用选中集算窗口，把 `rel` 写进 `active_cache_positions_`；
  `active_rope_positions_` 保持真位置。
- 改 `include/ninfer/ops/softmax_attention.h` 的**契约注释**（`positions` 由"绝对位置"
  改为"窗口坐标"，并写明真位置只走 `rope_positions`）；**打分/mask 代码不改**。
- envelope：`max_visible_keys = rel(max_p) + 1 <= window_tokens + 1 <= budget + 1`，
  `min_visible_keys` 按窗口下界给（`causal_softmax_attention.cpp:178` 要求 `> 0`）；
  topology class 可能翻转 ⇒ `planning/graph_profiles.cpp` 按窗口尺寸覆盖。
- **验收**：窗口 == 全上下文时与改前**逐位一致**（此时 `is_identity()`，`rel ≡ 真位置`）；
  graph profile 校验通过。

**阶段 3 — 按块映射发布执行表 + 三个不变量断言**
- `publish_window` 现在按逻辑页升序遍历，把所有 retained page 连续发布到
  `block_table[0, retained_pages)`；页的窗口行由 `window_page()` 给出（块行基 +
  块内页偏移）。**不搬任何 GPU 字节** —— 只改 `block_table` 的 4 字节发布
  （I2 保证页内偏移不变）。
- `ensure_mapped_to_tokens` 仍只追加尾部页：identity 用 `begin`，洞窗口用
  `retained_pages + (begin - pages_for_tokens(total))`，因此重复增长不重发布也不会覆盖行。
- 加 I1（调用区间内每个块都 retained，已在 `KVMemBlockStore::window_for` 断言）、
  I4（retained page 数 == 发布页数，`publish_window` 断言）与"块起点页对齐"
  （I2，已在 `validate_config` 与 `set_window` 断言）。
  违规直接抛，不留静默错误。
- 打通 decode 期间的降级：`materialization.cpp:1549-1551` 的
  `has_active_reference` 限制要为"本序列自己的页"放行（KVMem 的窗口替换正发生在这）。
- 保持：entitlement 检查、事务 phase machine、epoch/coverage 校验、
  "republish 不进图"（图只存矩阵地址）。
- **验收**：`test_context_store.cpp` 增加"活跃序列页被替换后仍正确"的用例；
  `test_engine_prefix_real.cpp` 的 d2h/h2d 断言仍通过。

**阶段 4 — 打分采集**
- **mean-K 捕获**：在 K 首次写入（`kv_cache_append` 前）时，对 pre-RoPE K 求块均值，
  下载到主机（每 128 token 一次，可与既有 D2H 合并）。
- **query 侧打分**：检索用当前 query 的 pre-RoPE K（或 mean）与历史块 mean-K 内积，
  softmax over blocks 取 top-k（对应 KVMem 的 `MeanK`）。
- 第一版**先做 Recency + sink/recent**（无打分，纯位置），把链路跑通，
  再上 Retrieval 打分 —— 这样阶段 4 可拆成 4a/4b，风险前移。
- **验收**：4a 用 Recency 能端到端跑通阶段 0 的验收 1/2；4b 有打分正确性单测。

**阶段 4a/4b 落地记录**（2026-10-05）：

- **Op `block_sum_by_position`**（`include/ninfer/ops/block_sum.h`，kernel/wrapper/launcher 三层）：
  对 pre-RoPE K `[W,T]` 按绝对序列位置求块内和，输出 `[W, block_count]` FP32；`block_base`/
  `block_count` 由 `first_position`/`T`/`block_tokens` **推导并校验**，Op 不接受调用方传入的块布局，
  因此"Op 写了哪些块"和"主机认为哪些块被写了"只有一个定义。位置口径是**绝对序列 token 索引**
  （不是窗口行、不是 RoPE 位置），依赖 `fill_i32_positions` 已有的连续性。
  契约明确：无 workspace、无 host 同步、可进 CUDA Graph。
- **F32 阶梯**（冻结）：BF16 K → F32 块内求和 → F32 D2H → 主机 F32 累积 → F32 均值 → F64 打分。
  128 项求和需 ~15 位尾数，F32(24 位)近乎无损；主机接口定死 `float`
  （`KVMemScoreStore::accumulate(const float*)`）。
- **只出和、不出均值**：块跨 prefill→decode 轮次，`finalize` 满额后才除 `block_tokens`。
- **层选择**：`run_layers` 只给 Full Attention 块编号，故 compact 索引 0 = 第一个 Full Attention 层。
  任意单层都自洽 —— 检索 query 取**最新已完成块的均值自身**（`select_window_blocks`，
  `context.cpp`），不是活的 query 向量，所以不需要多层一致。
- **捕获接线**（`TextContext::capture_key_sums`，`text.cpp` `attn_mix` 内，key rmsnorm 之后、
  `text_rope` 之前）：两个模式
  - `Single`（text prefill）：真实 `first_position = base + t0`（逐子块重绑），每个 reduction 记一条
    segment；`prefill_impl` 结尾已 synchronize，故 chunk 返回后 segment 即可读。
  - `SequenceBatch`（ordinary decode）：每 lane 一个 token，按 `lane * key_width` 固定列。
    **图内捕获的正确性论证**：`T=1` ⇒ `block_count≡1`，kernel 的 `t_begin/t_end` 被夹到 `0/1`，
    结果与 `first_position` 取值**无关**；图捕获时烧死的常量因此在每次 replay 上都正确。
    主机从各 lane 的 `execution_frontier` 推真实块号，不依赖图内标量。
- **暂存所有权**：device + pinned host 缓冲由 `ProgramImpl` 一次性分配
  （`prepare_kvmem_capture`，仅 Retrieval 激活时），**不是** per-round workspace —— 图要烧地址，
  缓冲必须活得比每次 traversal 长；pinned 的理由与既有 ordinary egress 相同（图内 memcpy 节点）。
- **合入时机**：prefill 在 `prefill_text_chunk` 返回后（已同步）；decode 在 `ordinary_decode_batch`
  的 `device.synchronize()` 之后、下一轮 reselect 之前，用该 lane 的**步前** `execution_frontier`
  （此时尚未自增）。两处都保证"本轮捕获的贡献在下一轮 reselect 前已进 ScoreStore"。
- **`kvmem_block_shares`**（`kvmem_store.h`，纯主机）：把"某个 range 触及哪些块、每块几个 token"
  这段算术从 Program 收敛到 store 里，与 Op 用同一套推导，杜绝两侧描述漂移。
- **验收状态**：`ninfer_block_sum_test`（FP64 oracle，含非对齐起点/跨块/`block_tokens=1`/拒绝用例/
  图捕获重放，以及**decode 位置不变性**用例：9 个 `first_position` 上单 token 归约结果逐位相同、
  声明形状恒为 `[W,1]` —— 上面那条图捕获论证因此是被测属性而非仅推导）；
  `ninfer_qwen3_5_kvmem_store_test` 新增 block-share 分区用例 + 既有 mean-K FP64 oracle 用例；
  `ninfer_qwen3_5_context_store_test`、`ninfer_qwen3_5_runtime_mechanisms_test` 通过。
  **端到端（2026-10-05，`Ternary-Bonsai-2-27B-PTQ1_0-vl_mtp_q4q5.ninfer`，4060 Laptop 8 GB，
  `--kv-dtype rk4v4-e8 --max-concurrency 1 --spec off`）**：2.1 四条**全部通过**。
  - **第 4 条 跨轮检索复用**：36639 token 的 prompt（超 budget 32768，窗口滑动两次）仍能准确
    回答埋在**开头、已被挤出窗口**的事实。
  - **第 1 条 262144 prefill**：`--max-context 262144 --kv-capacity 65536 --kv-ring
    --kvmem-budget 32768`（即 `gen_reserve = kv_capacity - budget = 32768`）下，实测
    **prompt_n = 258,939 token** 一次性 prefill 并生成，耗时 364 s，**无**
    `context_length_exceeded`，同样准确答出开头的事实。
  - **第 2 条 device 常驻上界**：该 258,939 token 运行中，`device_main_kv_pages` 从池上限
    1024 页（= 65536 token = 1.14 GiB）**单调递减**到 559 页，同期 `host_kv_bytes` 从 0
    增长到 3.89 GiB。device 常驻**始终不超过 65536 token**，即 KVMem 的核心命题
    "有限显存跑远大于显存的上下文"成立 —— 同样 258,939 token 稠密放置需要 4.25 GiB KV，
    加 5.52 GiB 权重超过本卡 8.58 GiB。
  **注**：`--kvmem-budget` **不**放宽 `kv_capacity >= max_context` 这条启动校验
  （`startup.cpp` 的 `ring_requested` 只看 `kv_ring`），所以 `kv_capacity(65536) <
  max_context(262144)` 的配置**必须同时给 `--kv-ring`**；`--kv-ring` 只放宽池尺寸下界，
  窗口化仍由 `--kvmem-budget` 驱动。
  **仍待**：`capture_key_sums` 本身需要真实 `Parameters`/权重构造 `TextContext`，**无法单测**；
  其可测部分（Op 归约、图重放、位置不变性、块分区、mean-K 打分）已分别覆盖。

**阶段 5 窗口滑动首次真机暴露的两个缺陷（2026-10-05 修复）**

窗口滑动这条路径此前只有单测覆盖块记账与发布原语，没有跑过"真实 prefill 跨窗口 + 收尾"。
端到端第一次跑就暴露两个必崩缺陷，二者都不是 mean-K 捕获引入的：

1. **`device_sources` 尺寸不匹配**（`context.cpp`，`stage_out_window_holes`）：
   把 `push_back` 刚写入的空 vector 传给要求调用方预先 resize 的双参数重载，
   `out.size() != extent.page_count` 必然抛 `"Host KV device-source output has the wrong size"`
   —— 窗口滑动必崩。改用会自行 resize 的单参数重载。另两处调用点
   （`materialization.cpp:1159` 的 `array<...,1>`、`:1570` 的显式 size 校验）本来就是对的。
2. **窗口洞页无法通过释放判定**（`kv_store.h`，`can_release_after_deactivate`）：
   判定要求 `active_references != 0`，但窗口洞页在 `set_window` 就已合法交还活动引用。
   **根因不是记账泄漏，而是判定与执行不一致**：`deactivate` 早已用 `demoted_` 标志
   跳过重复释放（那里写着"Window holes already released their active reference at set_window"），
   作者想到了这点，**前置判定却漏了同一个分支**。修法是让判定用同一个前提
   （`demoted_ != 0` 时改用 `can_release_reference`，即 `release()` 本身随后施加的守卫），
   而不是放宽全局不变量或让窗口保留活动引用。
   症状是 `finish()` 静默失败 → `abort()` 也失败 →
   `"Program could neither retain nor discard terminal sequence"`（500）。

  定位方法：`can_clear_lane_strict` 有 5 个静默早退点，手工推不动；临时在判定里加页级
  诊断输出，一次复现即读到 `idx=2 refs=1 act=0 hostrep=1`。**诊断代码已全部删除**
  （提交前确认无 `DIAG`/`describe_reference` 残留）。教训：静默判定函数是这类问题的
  主要排查障碍，值得考虑让失败原因可观测。

**阶段 5 — 每轮 reselect 钩子 + 压力 prefill**

reselect 的产物只有两样，形状都已经确定：

1. **`KVMemWindow`** ⇒ `kv_store.set_window(handle, window, stream)` 重发布执行表；
2. **`cache_position_delta = rel(call_min) - call_min`** ⇒ 让 Op 看到的 `positions` 变成 `rel`。

`rel` 对**单 token 调用**是仿射的（`abs → abs + delta`），所以 decode 不需要逐 token 换算：
- decode 的 `cache_positions[row]` 是主机侧写的（`decode.cpp:325-328`），主机加 `delta` 即可；
- envelope 取 `rel(p_max) + 1`。

prefill chunk 与 MTP verify 是**连续区间**，而 `rel` 在多个保留块之间不再是单一仿射段，
因此这两条路径强制 recency：连续区间落在 sink+tail 的单个 run 内，仍然可以用一个 `delta`。
检索只作用于单 token 的普通 decode。identity 窗口下 `rel ≡ abs`，`delta` 为 **0**——
分支留在 Program（窗口的拥有者），执行层保持无状态。

**已确认的事实**（决定了这个方案可行）：
- `causal_softmax_attention.cpp:132`：`capacity = logical_pages * 64 = max_context`
  ⇒ envelope 收窄到 `budget + 1` 不会被 `max_visible_keys > capacity` 拒绝；
- `startup.cpp:122/225`：`logical_pages = page_count(plan.capacity)` = 4096 @262144
  ⇒ 执行表首维同时容纳绝对页号与窗口页号；
- `kv_store.h:1846 publish_membership` 原本就是"全量重发布"，正是滑窗需要的原语。

- 块镜像与 frontier 的同步：`SequenceState.kvmem`（`KVMemBlockStore`）按
  `total_tokens() == frontier + 1` 对齐（**+1 因为 query 自己的块也要被选中**，否则
  `window_for` 的 I1 会拒绝本次调用）。frontier 增长就 `register_append`，回退就 `truncate_to`。
- 在 `decode.cpp:282` 算 `maximum_frontier` 之前逐 lane 跑 reselect；prefill 侧在 chunk 循环前跑。
- 长 prefill：`pick_prefill_pressure_blocks`（sink + 已处理尾部），边 prefill 边把完成的块
  stage-out，防止 device 池被撑爆 ——**W7 决定第一版不实现**，由渐进映射 + reselect 承担。
- 搬运是异步的，需要新的 round 等待条件（现在 `ExecutionAction` 没有这个状态，
  `scheduler.h:240-246`）——**W6 决定不引入异步等待**，见下方落地记录。
- **验收**：262144 prefill 全程 device 常驻 ≤ budget；无 `context_length_exceeded`。

**阶段 5 落地记录**（2026-10-04，含长 prefill；staging 生效条件 =
`--kvmem-budget != 0 && --spec off`，由 `ProgramImpl::resident_staging_active()` 定义）：

- **entitlement 逻辑/物理拆分**（`kv_store.h`）：`Address` 加 `logical_entitlement`，
  `entitlement()` 对 active 地址返回该字段、inactive 保持旧派生式；`prepare_activation` 的
  physical claim 改为 `min(required, pool.available_pages())`（capped），`commit_activation`
  的相等校验放宽为 `<= expected`，`commit_prefix_fork` / `commit_active_snapshot` /
  `commit_activation` 三条激活路径都写逻辑字段；`resize_entitlement` /
  `ensure_mapped_to_tokens` 在 materialize 前把 claim top-up 到
  `physical_tail_target()`（`min(logical 缺口, reservation + available)`），不足则
  `std::bad_alloc`。于是**逻辑覆盖可以大于物理池**（262144 覆盖 vs 数万页池）。
- **resident-aware 准入**（`request_plan.cpp`）：staging ⇒ `active_capacity = capacity`
  （否则 `min(capacity, kv_capacity)`，`--kv-ring` 拒绝行为原样保留）；
  `root_active.main_kv_pages = min(entitlement, pages_for_tokens(kv_capacity))`；
  非 Root 复用路径 staging 下要求 `text_kv_page_entitlement <= pages_for_tokens(kv_capacity)`，
  越界直接拒绝（恢复前缀先把每页发布为 Device 常驻，窗口无法缩小这一步）；
  staging 禁 `publish_continuation` 与 capture（`prepare_active_snapshot` 要求全页 Device，
  窗口刻意把页放在 Host tier）。
- **渐进映射**（`prefill.cpp`）：启动处 `ensure_sequence_kv_mapped(prompt_tokens)` 仅当
  `!staging || reuse != Root` 执行；`advance_prefill` 在每次 reselect **之前**
  ensure 到 `cursor + nominal`（顺序保证：窗口只覆盖已映射页，`set_window` 的
  "claims more tokens than it maps" 检查）。dense 配置下是 no-op，行为不变。
- **启动校验**（`startup.cpp:1125-`）：`kv_capacity >= budget`、
  `kv_capacity >= budget + prefill_chunk`（瞬时峰值 = 窗口 + 在飞 chunk 的由来）、
  Host tier 必需 —— `--host-kv-mib 0` 时按 `page_count(capacity) - page_count(budget)`
  个 demoted 页自动填，不足则抛（staging out 需要 Host 副本，不能静默超预算）。
- **`--host-kv-mib 0` 与 `--kv-ring` 同开时的已知限制**（2026-10-05 实测，刻意保持 fail-fast）：
  `--kv-ring` 分块先按 `logical - pool` 填 `host_kv_capacity_bytes`，于是 KVMem 分块看到值已非零
  而跳过自己的填充，随后按 `logical - window` 校验必然偏小 —— 实测
  `host_pages=3072 required_pages=3584`（ctx 262144 / pool 65536 / window 32768）直接拒绝启动。
  **把派生值抬到 `logical - window` 可以过校验，但运行时仍失败**：reselect 先算入窗口、
  再降级出窗口，稳态公式表达不了这个重叠，实测 258,979 token 时 Host 实际用到 3.89 GiB
  而派生上限只有 3.72 GiB，最终 `std::bad_alloc`（`HostKVExtentStore::prepare` 失败）。
  即该公式是**稳态下界而非上界**，把它调大只是把"启动即报错"降级成"跑到 25 万 token 才崩"。
  正确修法需要一个真实的重叠上界（而不是更大的常数），在此之前：
  **显式传 `--host-kv-mib`（256K 上下文 ≥ 3809 MiB），或整个省略该 flag**
  —— 省略即取 `ContextCacheOptions` 默认 8 GiB 可寻址容量、按需惰性 pin，这是实测通过的路径
  （`scripts\start-ninfer-4060-kvmem.bat` 即此配置）。
- **resident 峰值口径**：稳态窗口驻留 ≤ budget；滑动前映射新 chunk 的瞬时峰值
  ≤ budget + prefill_chunk（由启动校验覆盖）。
- **W6/W7 决定**：窗口搬运用既有同步路径（单次 `transfer_stream` 同步，不引入
  异步 `ExecutionAction`）；`pick_prefill_pressure_blocks` 第一版不实现，
  由"启动 ensure 全量 + 每轮 reselect 后释放"的组合承担同一职责。
- **验收状态**：单测见 2.6 头部证据行；端到端 262144 实测待模型产物。

**阶段 6a — MTP × KVMem 窗口化 落地记录**（2026-10-29；`--spec mtp` 与真实 `--kvmem-budget` 并存，
DFlash/DFlash2 保持恒等窗口，留阶段 6b）：

- **门禁**（D2/P2）：`resident_staging_active()` / `retrieval_selection_active()` 放开 `--spec mtp`
 （新增 `kvmem_resident_staging(backend)` 在 `src/models/load_options.h:55`，`program_impl.h` 两 gate、
  `model_instance.cpp:157` 与之同步改）；`storage/context.cpp:1645` reselect 门改为
  `is_masked_draft_backend` 硬拒绝（DFlash 系恒等），MTP 通过窗口行移消费表。
- **位置管线**（D4/P3）：`speculative_prepare_verify_inputs` 与 `mtp_prepare_next_round` 各增
  `cache_position_deltas [B]` 输入（每 lane 窗口行移，为 0 时恒等），分别输出
  `cache_positions [W,B]` / `ar_cache_positions [B,S]`（注意力与 KV 唯一行基）；RoPE 与
  `rope_deltas` 不动；dflash 路径传空 deltas；`MtpDecodeIngress` 增 `cache_position_deltas`
  字段；op 单元测试（`tests/ops/test_speculative_round.cpp`、`tests/ops/test_mtp_round.cpp`）
  已扩展并通过（GPU 实测 PASS）。
- **MTP 轮 reselect**（D3/P4）：`decode_mtp_batch` 逐 lane 先 `ensure_sequence_kv_mapped(
  frontier + extent + 1, min(capacity, frontier + extent + draft_window))`，再
  `reselect_sequence_window(sequence, frontier, frontier + extent + draft_window)`，返回行移
  写入 `cache_position_deltas[row]`；拒绝时下一轮 `truncate_to` 收回（既有逻辑复用）；
  `apply_window_transition` 对 text 与 backend 各调一次 `apply_window_space_transition`（P4
  纯重构，text 行为逐位不变）；`mtp_kv_table_rows` 已由 bound row 覆盖，不动。
- **启动校验**（D6/P2）：`kv_capacity ≥ budget + prefill_chunk + max_concurrency × draft_window`
 （MTP 导线）；Host tier 要求按两空间字节统筹（`demoted_pages × (text_stride +
  backend_stride)`，`--host-kv-mib 0` 口径与 2.6 阶段 5 一致，显式传或省略均可）。
- **准入**（P2）：staging 下 `main_kv_pages = min(text entitlement, pool_pages)`，
  `backend_kv_pages = min(backend entitlement, backend_pool_pages)`，其中 `backend_pool_pages`
  取自该空间的 Device 池容量（`page_pool().capacity_pages()`，即 `pages_for_tokens(kv_capacity)`
  加上 MTP draft lead 页）。用 text 池页数封顶会让超池 prompt 的激活多领一页，materialization
  随即按"取消"回滚（HTTP 200、`finish_reason:"stop"`、0 输出）；已按上述口径修正。
- **捕获**（P5）：`KVMemCaptureMode::MtpVerify`，verify 预 RoPE K 按 flat 列
  `lane × W + j` 分列单 token 求和入 staging（`text.cpp:839` 扩展），`prepare_kvmem_capture`
  容量加 `8 × 16 = 128` 列；decode 后按 `licensed_counts` 只折叠 committed 列（`accumulate_
  kvmem_decode_scores` 改按 `column` 索引 staging，ordinary decode `column = lane`，逐位不变）。
- **文档**：`docs/cli.md` / `docs/serving.md` `--kvmem-budget` 行已更新（“ignored while --spec
  enabled” 改为仅 dflash 系忽略，mtp 加入 staging；kv-capacity 下界注明 MTP draft lead）。
- **验收**：`ninfer_model_runtime` 全量编译通过；`ninfer_mtp_round_test`、`ninfer_speculative_round_test`
  GPU 实测 PASS；`git diff --check` 干净。端到端 262144 验收 1/2 待模型产物（阶段 7）。

原占位块（"MTP backend KV 是独立稠密池……若不可行，第一版默认 --spec off"）已作废。


**阶段 7 — 文档与端到端**（文档部分 2026-10-04 已完成；端到端未跑）
- **已完成**：`docs/serving.md` 选项行 + Execution behavior 的 `--kvmem-budget` 段落；
  `docs/maintainer/paged-kv-cache.md` §3.5 有界工作集契约 + §5.2 窗口降级语义；
  `README.md`（Choosing settings 能力说明 + Limits 限制行）；可执行 `--help` 同步
  （`serve_options.cpp` 新增 `--kvmem-budget` 描述行，并把 `--host-kv-mib 0` 的
  自动计费说明扩到 `--kvmem-budget`）。
- **未跑**：2.1 四条端到端验收 —— 第 1/2/4 条需模型产物实测，第 3 条依赖阶段 4 打分的
  独立 oracle。
- **验收**：`git diff --check` 干净（已过）；受影响测试 `ninfer_serve_options_test` 通过。

### 2.7 与已有 `--kv-ring` 的关系（必须处理）

- 现状：`--kv-ring` = "device 池 < 逻辑容量 + **resident bound 拒绝 prompt > kv_capacity**"
  （`request_plan.cpp:222` 的 `active_capacity = min(capacity, kv_capacity)`、
  `engine.cpp` 的 `context_length_exceeded`）。
- KVMem：同一个底层能力（device < 逻辑），但**让 prompt > kv_capacity 变得可行**。
- 按 `AGENTS.md` 的 change consistency：被取代的行为要在同一契约里一起删。
  **不能**留着 "ring 拒绝超长 prompt" 和 "kvmem 接受超长 prompt" 两条并存分支。
- 建议口径：`--kv-ring` 保留为"device 池小于逻辑容量"的**开关**，
  检索窗口作为其下的可见性策略；resident bound 的拒绝信息改成指向检索窗口的上限。
- 注意 `state/decoder_state.cpp:27-29` 已有 "retrieval mask" 预告注释 —— 语义归属要理顺。

**落地口径**（2026-10-04）：按 staging 模式互斥分流，同一配置下不存在两条并存分支 ——

- `--kvmem-budget 0`（默认，含 `--kv-ring`）：`active_capacity = min(capacity, kv_capacity)`，
  拒绝行为与移植前逐位一致；拒绝信息已更新（`request_plan.cpp:227-230`）指明
  `--kvmem-budget` 只对**无可复用前缀**的 prompt 放宽。
- `--kvmem-budget != 0 && --spec off`（staging）：新鲜（Root）prompt 上限 = `capacity`，
  非 Root 复用路径仍以池为界并显式拒绝（拒绝信息带 `text_kv_page_entitlement` 数值），
  capture / `publish_continuation` 关闭（见 2.6 阶段 5 落地记录）。
- 即 Q5 的分工成立：`--kv-ring` 是"device 池小于逻辑容量"的池开关，拒绝语义只在它
  自己的（非 staging）模式里生效；拒绝信息已按本节建议指向 `--kvmem-budget`。

### 2.8 验证方案

按 `AGENTS.md` 的证据表选取，不跑全表：

| 改动 | 证据 |
|---|---|
| 纯主机选择逻辑 | 新增 CoreTests：块表/选择/diff/可见性前缀推导 |
| attention 契约（阶段 2） | 独立 FP32/FP64 oracle：密集对拍改前逐位一致；稀疏可见集按 `op-development.md` 判据 |
| 事务/驱逐/映射（阶段 3） | `test_context_store.cpp` + `test_engine_prefix_real.cpp` 受影响用例 |
| 记账/容量（阶段 0） | 受影响的 capacity/planning 测试 |
| 端到端（阶段 5/7） | 262144 prefill+decode、跨轮检索复用、host 池懒 pin 上界、device 常驻不超 budget |
| 文档 | 受影响链接/引用 + `git diff --check` |

**不做**：性能声明（先不做 benchmark；若做了要按 `bench/README.md` 记录目标/硬件/命令）。
**多并发不测**（用户明确要求）；阶段 1-5 均在 `--max-concurrency 1` 下验证。

### 2.9 决策（已定，不再讨论）

| # | 决策 |
|---|---|
| Q1 | **可见性走路线 B**：K 按真位置只烘焙一次，零漂移；可见集用**重编号前缀**表达（2.12），不新增 Op 输入 |
| Q2 | **第一版只做 Recency**（sink + recent + recency），阶段 4 拆 4a/4b；mean-K Retrieval 放 4b |
| Q3 | **第一版 `--spec off`**，draft/MTP 池列为已知限制，留到阶段 6 |
| Q4 | `--kvmem-budget` 默认 **32768**（论文无损点）；`gen_reserve = kv_capacity - budget`。**2026-10-05 已启用（用户决定，覆盖原门禁）**：`ServeOptions` 默认 32768，同时 `max_context`/`kv_capacity` 默认抬到 65536 使默认组合自洽（kvmem 要求 `kv_capacity >= budget + prefill_chunk`，只改 budget 会让默认启动直接拒绝）；库 API `EngineOptions` 保持 0（嵌入式调用方自决），`CausalScoring` 强制 0。阶段 4b 与阶段 5 的 2.1 四条端到端验收**均已通过**（2026-10-05，258,939 token 实测），默认值不再是未验收状态 |
| Q5 | 新增 `--kvmem-*` 一组；`--kv-ring` 保留为"device 池小于逻辑容量"的池开关 |

### 2.10 术语与记账口径（阶段 0 冻结）

**参数**

| 词 | 含义 | 来源 |
|---|---|---|
| `logical_capacity` | 逻辑地址空间可命名的 token 数 = `--max-context` | 现有 |
| `kv_capacity` | device 池 token 容量 = `--kv-capacity`，**= budget + gen_reserve** | 现有，语义不变 |
| `budget` | 检索工作集可占用的 token 数 | 新，`--kvmem-budget` |
| `gen_reserve` | 为本轮回合新生成 token 预留的 device 槽 | 新，派生 = `kv_capacity - budget` |
| `block_tokens` | 选择/迁移的逻辑块粒度 = **128 token = 2 个 64-token 逻辑页** | 新，`--kvmem-block-tokens` |
| `sink_tokens` | 永远保留的前缀 | 新，`--kvmem-sink-tokens` |
| `recent_blocks` | 永远保留的尾部块数（第一版 Recency 用它近似） | 新，`--kvmem-recent-blocks` |

`block_tokens=128` 与 NInfer 页粒度 64 的关系：**选择/打分/迁移按 128-token 块，
物理存储与搬运仍按 64-token 逻辑页**（一个块 = 2 个页，一次 extent 成员）。
KVMem 用 128 是为了摊薄打分与传输开销；NInfer 不改页几何。

**记账（替换"页数 = frontier 的函数"）**

```
logical_pages  = page_count(logical_capacity)          // 4096 @262144
pool_pages     = kv_capacity / 64                      // device 池
host_capacity  >= logical_pages - pool_pages           // 启动护栏（已实现）

resident_pages(t)   <= pool_pages                      // 本刻 device 上映射的页
host_only_pages(t)  <= logical_pages - resident_pages  // 只有 host 副本的页
not_materialized(t)  = logical_pages - resident - host_only   // 还没算过的页
```

不变量：
1. `resident_pages + host_only_pages <= logical_pages`（可以不铺满——未算过的页不存在）
2. `resident_pages <= pool_pages`
3. 每刻 `resident_pages` **不必是前缀**；可见集由 `visible_count` 单独表达

受影响并需重定义的位置（阶段 0 清单）：
- `program.h:97-105` `TargetKVRequirement{main_pages, backend_pages}` → 加
  `resident_pages` / `host_only_pages` 两个量，`main_pages` 的"必须全部 device"语义放宽
- `request_plan.cpp:270-279` `text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens)`
  → entitlement 改为**逻辑**页数（记账上限），device 需求另算 `resident_pages`
- `pressure.cpp:503` `kv_pages_for_frontier` → 拆成"逻辑页"和"device 页"两个函数
- `context_work.cpp:93-105` `required_kv` 合并、`resource_manager.h:1549,1595`
  目录有效性 `required_kv.main_pages != 0`
- `request_plan.cpp:222` resident bound（B 语义）→ 上限改为 `logical_capacity`，
  device 侧不足由 reselect + stage-in 处理，不再靠拒绝 prompt
  （**已落地**：staging 分流见 2.7 落地口径）

**阶段 0 完成判据**：以上每处都写清"这个数字在有界工作集下等于什么、由谁决定"。

**阶段 0 落地记录**（2026-10-04，五处全部完成）：

- `TargetKVRequirement` 已加 `resident_pages`/`host_only_pages`（`program.h` 注释即口径）：
  `main_pages = resident + host_only = kv_pages_for_frontier(main_frontier)`（逻辑 claim）；
  构造点在 `context.cpp` 的 `tiered_kv_claim`（demoted 洞页 = host_only 半，无 KV 时退化为
  全 resident 的 rebuild claim），`context_work.cpp` 合并对 survivor 求 resident 的 max、
  host 取合并后 claim 的余数以保持等式精确；backend 半恒全 resident（spec 门控下无窗口）。
- entitlement（`request_plan.cpp`）：数值本就是 `pages_for_tokens` 的**逻辑**页数，已补口径注释；
  store 侧的 `entitlement = page_count + reservation.pages` 派生式已随阶段 5 改为
  **逻辑/物理拆分**（active 地址读 `logical_entitlement`，claim 被池 capped 并按需 top-up，
  见 2.6 阶段 5 落地记录）；`root_active` 已按 resident-aware 准入落地为
  `min(entitlement, pages_for_tokens(kv_capacity))`。
- `pressure.cpp` protection 边界（488/503）：确认为**逻辑**口径 —— 保护的是 reuse 的逻辑页范围，
  drop 路径各自查 device 驻留；device 口径由新函数 `kv_resident_pages_for_window` 单独给出。
- `resource_manager.h:1549,1595`：有效性仍只读 `main_pages != 0`（逻辑 claim ⇔ 至少命名一页），
  已加注释说明 resident/host 是规划边界而非有效性输入。
- `request_plan.cpp:222` resident bound：**staging 下已放宽**（2026-10-04）—— 激活路径已
  窗口化（`prepare_activation` claim 被池 capped、随映射 top-up，`commit_activation` 接受
  子声明），新鲜 prompt 上限变为 `logical_capacity`；`prepare_kv_restores` 仍整前缀重建，
  故非 Root 复用路径保留池界并显式拒绝；非 staging 配置（`--kvmem-budget 0`，含
  `--kv-ring`）的拒绝分支原样保留，见 2.7 落地口径。

已知限制（随层级释放引入）：
- 窗口洞页上的**部分列** trim 显式抛出（`destructive_truncate` "partially release a hole page"）；
  生产路径不可达：prefill trim 在激活之后（恒等窗口），decode trim 落在 recent 尾。
- capture（active snapshot）必须先 `stage_in_all_holes` + `hold_window_holes` 恢复洞页的
  Device 驻留与持有权（`prepare_active_snapshot` 要求全页 device + 单 active ref + 单 writer）；
  abort 走 `release_window_holes` 还原，成功 commit 则按既有约定把引用转给 destination。
- stage-out 不向 reservation 返还 claim（与既有 drop 行为一致），stage-in/新映射时再按
  `physical_tail_target` top-up；`logical_entitlement` 是独立存储字段（阶段 5 起），
  不再由 `page_count + reservation` 派生，drop 因此不需要回写 entitlement。
  共享/pinned 页 stage-out 门控失败时留在 device（有界超支，非错误）。

### 2.11 关键发现：可见性由 `cache_positions` 承载，无需新增 Op 输入

调研 `execution/text.cpp:877-882` 发现 **`cache_positions` 与 `rope_positions` 本来就是两个独立张量**，
且已分别可被覆盖：

```cpp
const Tensor& cache_positions = active_cache_positions_ ? *active_cache_positions_ : io_.pos;
const Tensor& rope_positions  = active_rope_positions_  ? *active_rope_positions_  : io_.rope_pos;
text_rope(rope_for_op, ..., qn, kn, s);          // rope_positions 只管烘焙
ops::causal_softmax_attention(..., cache_positions, ...); // cache_positions 管 mask + append 寻址
```

于是 Q1 的落地形式比原设想更小：

| | 输入 | 值 |
|---|---|---|
| RoPE 烘焙 | `rope_positions` | **真单调位置**，永不改变 ⇒ 零漂移 |
| mask 上界 + append 寻址 | `cache_positions` | **窗口行号**（选中块按真位置升序打包后的行） |
| block_table | 执行表行 `r` → 该窗口行的物理页 | 每次重选 republish，**不进 CUDA Graph** |

dense 模式下窗口行 ≡ 真位置，两者相同 ⇒ **现有行为逐位不变**。
因此**不需要给 Op 新增 `visible_count` 张量**——`cache_positions` 就是可见性：
mask 读 `row <= cache_positions[t]`，而喂进窗口行号即等价于"可见 k 行 + 自己"。

**必须成立的规划不变量（I1）**：每次 attention 调用时，该序列窗口内**每一行的真位置都严格小于
本批每个 query 的真位置**。
- decode：query 是最新的，自动成立
- prefill：窗口只放真位置 < 当前 chunk 首位的历史
- 检索：只从 query 之前的**历史**里选，自动成立
- **分歧重 prefill**：窗口里若残留真位置更靠后的块，必须先 stage-out —— NInfer 现有的
  `destructive_truncate` 已经在做这件事

违反 I1 的后果是静默过度可见（结果错误、不崩），不是崩溃。因此阶段 3 要把 I1 做成
**显式断言**，而不是隐含约定。

### 2.12 关键设计：重编号前缀（为什么不能直接照抄参考实现）

**参考实现怎么做的。** 本地 KVMem 参考（`llama.cpp-kvmem-.../kvmem-v0.16.0-rc3-source`）用的是
**按 cell 索引的槽池**，不是按位置索引的前缀：

- `cell = gpu_slot * block_tokens + (orig_pos - orig_pos_start)`
  （`src/adapter/llama-kvmem-batch.h:10-13`，计算在 `llama-kvmem-batch.cpp:39-54`）；
  `gpu_slot` 在重选时**不变**，幸存块一个字节都不移动
  （`docs/architecture.md:29-32`："resident selected blocks stay in their slot"）。
- 可见集 = **稠密读 `[0, n_kv)` + 每 cell 的位置 mask**（`n_kv = used_max_p1`，
  keep/drop 由 `cells.pos` 决定，见 `patches/0003-kv-cache-skip-hole-purge.patch:89-110`；
  实测日志 `docs/milestones/v0.3.0-complete_retest.log:27`：
  `mask last_pos=836 n_kv=256 keep=229 holes=1`——229 个 cell 跨真位置 0..836，中间有大洞）。
- Flash Attention **不改**（`docs/architecture.md:30-31`）。
- 写寻址（`slot_info.idxs`）与读可见性是**两套地址空间**（结论见 2.13 对照表）。

**NInfer 不能直接照抄的原因。** NInfer 的 `causal_softmax_attention` 只有一个可见性输入：
`positions`，其语义是**稠密前缀 `[0, p]`**（`include/ninfer/ops/softmax_attention.h:118-124`），
kernel 侧就是 `block_table[tile_k0 >> 6]` + `page_offset = position & 63`
（`prompt_fp8.cuh:176-177`、`small_t_fp8.cuh:194-196`）。**没有 per-row 位置 mask**，
也没有"写行号"与"读行号"的分离。若照抄 cell 池，就必须给 8 个 kernel 变体各加一个
`key_positions` 张量 + 一个 `write_rows` 张量 —— 这是重型 Op 契约改动。

**NInfer 的解法：把任意保留块集重编号成稠密前缀。**

窗口是**任意一组升序保留块**（Recency 只是"sink 头 + 最新尾"的特例；top-k 可以选任意
interior 块）。`KVMemWindow` 记录：

- `block_tokens`：选择块宽（生产值 128，必须是 64 的整数倍）；
- `total_tokens`：选择时块表覆盖的上下文长度；
- `window_tokens`：保留块持有的 token 总数；
- `retained_pages`：保留块覆盖的 64-token 页数，也是执行表要连续发布的行数；
- `row_base_by_block[id]`：块 `id` 第一 token 的窗口行；被丢弃的块为 `UINT32_MAX`。
  **空 map 表示 identity**（所有块都保留），走稠密快路径。

定义

```
rel(abs) = row_base_by_block[abs / block_tokens] + abs % block_tokens   // 保留块
         = 未定义                                                       // 洞
```

则可见集在 `rel` 空间里是**一个稠密前缀 `[0, rel(p)]`**，且 `rel` 在保留位置上**严格单调**，
所以

```
rel(k) <= rel(p)   ⟺   k <= p
```

mask 数学**完全不变**——不需要 per-row mask，不需要改 kernel 的打分逻辑。
top-k 选中的 interior 块只是 map 里的另一段行基，执行表按真位置升序连续发布即可。

**为什么不用搬字节（不变量 I2）。** 块起点总是 64 页对齐：块从 0 连续排布，
`truncate_to` 只缩短尾块、不改变任何 `orig_pos_start`，补齐后下一块仍从
`(k+1)*block_tokens` 开始；`block_tokens` 是 64 的倍数。于是同一块内页偏移
`rel & 63 == abs & 63`，字节停在原地，**只有 `block_table` 的行发布要改**
（`KVExecutionTablePool::publish(row, logical_begin, ...)`，`logical_begin` 本来就是任意页号）。

所以 **阶段 3 不存在"重排/搬移 GPU 字节"**，只有"改发布到执行表的页号"。

**新增不变量**

先纠正一个先前的误判：**不需要"窗口内每行真位置都小于 query"**。因为 `rel` 在保留位置上
严格单调，mask 判据是 `rel(k) <= rel(p)`，等价于 `k <= p`；真位置大于 query 的行自然落在
`rel(p)` 之上、根本不会被读。所以窗口里残留未来的块**不会**造成过度可见。
真正会静默出错的是**query 自己所在的块没被保留**——`rel(p)` 无定义，或 `rel(p)` 覆盖的行没发布。

| # | 不变量 | 谁保证 | 违反后果 |
|---|---|---|---|
| I1 | 下一次 attention 调用的位置区间**触及的每个块**都 retained | `window_for()` 遍历 `[first_block,last_block]` 断言 | `rel(p)` 无定义 → 静默错结果，不崩 |
| I2 | `block_tokens` 是 64-token 页的倍数 | `validate_config` + `set_window()` 断言 | 页内偏移错位 ⇒ 读到错的 K/V |
| I3 | 包含 frontier 的块必须被选中 | 选择器（`select_recency`/`select_topk` 强制保留尾块） | 前缀预算下无法 decode |
| I4 | `retained_pages` 与按 map 数出的页数一致，且全部已发布 | `window_for()` + `set_window()` + `publish_window()` 断言 | 读到未发布/已释放的物理页 |

I2 的前提"块起点总在 `block_tokens` 边界"由块表结构保证：块从 0 连续排布，`truncate_to`
只缩短尾块、不移动任何 `orig_pos_start`，补齐后下一块仍落在 `(k+1)*block_tokens`。已有单测覆盖
（`test_block_starts_stay_page_aligned`、`test_window_supports_blocks_wider_than_one_page_pair`）。

**对阶段 2 / 阶段 3 的影响**

- **阶段 2**：仍是"喂窗口行号 + envelope 按窗口尺寸"，Op 的**打分与 mask 代码不改**，
  oracle 不改（数学没变）。但必须同步改 `softmax_attention.h` 的**契约注释**：
  现在写的是"absolute position p"，落地后 `positions` 是**窗口坐标**，
  真位置只走 `rope_positions` —— 这是最容易被后人踩中的坑，不写清等于埋雷。
  还需：`envelope.max_visible_keys = rel(max_p) + 1 <= window_tokens + 1 <= budget + 1`（有界），
  `min_visible_keys` 同步按窗口下界给（`causal_softmax_attention.cpp:178` 要求 `> 0`）；
  topology class 可能翻转 ⇒ `graph_profiles` 要按窗口尺寸覆盖。
- **阶段 3**：`publish_window` 按逻辑页升序把 retained page 连续发布到
  `[0, retained_pages)`；`ensure_mapped_to_tokens` 在洞窗口下用
  `retained_pages + (begin - pages_for_tokens(total))` 继续追加尾部页。
  **没有字节搬运原语。**
- **阶段 4b（检索）**：top-k 的 interior 块直接进入 `row_base_by_block`，不再需要
  "两段/多段 delta"或 per-cell mask。约束是调用区间必须落在同一个连续保留 run 内：
  普通 decode 是单 token，自然满足；prefill chunk 与 MTP verify 是连续区间，所以这两条路径
  强制 recency，由 startup 的 `budget >= call_span + 2*block_tokens` 校验保证区间落在
  sink+tail 的 tail run 内。
