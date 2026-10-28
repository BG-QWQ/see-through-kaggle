# 推理验证记录

## 2026-10-27：分辨率、步数、显存与精度

### 结论与适用范围

- `common/assets/test_image.png` 已完成 **2048 / 30 steps / FP16 UNet 与文本编码器 / FP32 VAEs** 的完整分层、深度及 PSD 流程，seed 42、43 均未出现本次追踪的黑脸、脸部透明洞与灰雾。
- **[INFERENCE]** 本样例 2048 / 10 steps 的主要故障属于低步数设置下的采样／图层分解失败。加大显存压力没有改变输出；在相同输入与噪声下，FP32 head 阶段仍有同样的透明洞，不能归因于 FP16 独有故障。
- 当前路径 2560 先发生 CUDA OOM，没有可评价模型质量的有效输出。不能据此宣称模型的分辨率硬上限是 2048，也不能宣称 2560 的模型质量已验证。
- 这是单张样例的受控诊断，不是连续 Web 服务、双 worker 高分辨率并发、任意 seed 或所有隐藏图层的质量认证。**本轮没有修改正式推理、服务或 PSD 导出代码。**

### 环境与复现控制

- Kaggle T4×2，单次推理仅使用 GPU0；两张卡的显存没有合并。单卡报告总显存约 14.56 GiB，共享主机内存上限 30 GiB。
- 隔离环境：PyTorch `2.8.0+cu128`、CUDA 12.8、Gradio `6.12.0`；其余推理依赖见 [`requirements-kaggle.txt`](../requirements-kaggle.txt)。没有修改 Kaggle 基础环境，未升级 Ego。
- LayerDiff 不量化。FP16／FP32 对照改变 UNet 与文本编码器计算精度，两种 VAE 始终 FP32。完整流程中的 Marigold 固定 FP16 UNet、FP32 VAE、768 分辨率。
- seed 在模型构建、tag embedding 缓存之后重置。精度对照不仅匹配 seed，还匹配输入 RGBA、FP32 VAE conditioning latent、显式 Generator、CPU/CUDA RNG 状态、FP16 采样后转为目标 dtype 的初始噪声、原始 FP32 SDE 噪声、prompt/group 与 scheduler 配置。
- 诊断流程使用单进程单 case、分段加载管线、UNet CPU-first 注册 group offload，以及释放管线后的 glibc 空闲堆归还。这些加载／内存驻留干预不能当作原始服务的连续运行证明。完整 notebook 保存了实际干预代码。
- 精度参考固定 LayerDiff revision `4477e6ce529bc6a141732e1a55a4932176db3b89`；深度权重缓存 revision 为 `4f4ffc46050b6feb764b628859968a26e76e6b6a`。

被测正式源码 SHA256：

| 文件 | SHA256 |
|---|---|
| `inference/scripts/app_gradio.py` | `457936b64e41ad84dcd46883171cf38a51a027011e0a2103bca54e3c5d14adb2` |
| `inference/scripts/inference_psd_quantized.py` | `e992c4b6fd7b7faf2789bd364d66fdc8fd2b9169f5a1a3e95b98b35112dee7b9` |
| `common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py` | `16d39952a973f13df0d0f31bcc2d63415eda652b6569bc914e715ae44990297d` |

### 完整流程的分辨率／步数对照

显存为 PyTorch 峰值分配／峰值保留，不包含所有驱动开销；失败行的数字只是中止前观测值，不是该配置完成推理所需的容量。

| 分辨率 | UNet/text 精度 | steps | seed／附加条件 | 结果 | 分配 GiB | 保留 GiB |
|---|---|---|---|---|---:|---:|
| 1280 | FP16 | 30 | 42 | 完成，无本次目标缺陷 | 8.48 | 9.67 |
| 1536 | FP16 | 30 | 42 | 完成，无本次目标缺陷 | 11.47 | 13.08 |
| 2048 | FP16 | 30 | 42 | 完成，无本次目标缺陷 | 12.50 | 14.06 |
| 2048 | FP16 | 10 | 42 | 完成，但脸部透明洞、灰雾 | 12.50 | 14.06 |
| 2048 | FP32 | 10 | 42，原 block offload | 首次 UNet forward CUDA OOM，无有效输出 | 13.19 | 14.11 |
| 2048 | FP16 | 10 | 42，额外持有 1 GiB 显存 | 完成，最终 RGBA 与未加压坏图逐字节相同 | 13.50 | 14.06 |
| 2048 | FP16 | 30 | 43 | 完成，无本次目标缺陷 | 12.50 | 14.06 |
| 2560 | FP16 | 10 | 42 | conditioning latent 捕获前 CUDA OOM，无有效输出 | 13.14 | 13.19 |

减少 steps 在这条路径上没有降低峰值显存，不能作为提高分辨率时的显存解决方案。2048 / 30 steps 本轮约 57 分钟／图，包含模型加载与诊断开销，不是纯去噪性能基准。

### 缺陷首次出现的位置

正常 head 输入 → head 透明 decoder 原始 RGBA 已有脸部透明洞 → 下方黑色头发层露出 → 最终合成显示黑脸。缺陷先于尺寸还原、Marigold 与 PSD 导出。

固定整图可见脸部肤色 ROI 的透明洞比例在 2048 / 10 steps 为 95.15%，两次 2048 / 30 steps 均为 0%。该指标只统计局部可见肤色，不是整图或隐藏图层质量分数。完整 ROI、RGB 筛选规则与 alpha 阈值保存在分析 JSON。

[10／30 步的输入、原始 face、alpha 与独立合成对照](../workspace/kaggle-quality-controls-20261026T211949Z/analysis/head-stage-proof.png)

加压对照的正确单次 alpha 合成 RGBA PNG SHA256 均为 `3462fcb1054c49016d140d3f24ac1cee0ca86b58db246b99d57fbd3694cc75af`。

### 完整 head 的精度对照

整图 FP32 的 13 图层 body 阶段无法稳定容纳于 T4，因此精度参考只重放已定位缺陷的**完整 11 图层 head 阶段**。不减少分辨率、steps、部位或联合注意力；两种精度均使用 leaf offload、逐行独立 FeedForward 四块计算、仅去噪时将透明 decoder 驻留 CPU，并在构建后与每步归还空闲 CPU 堆。

| head 配置 | 完成结果 | 峰值分配 GiB | 中央可见脸颊 ROI 的透明洞比例 |
|---|---|---:|---:|
| 2048 / 10 steps / FP16 | 完成，透明洞仍在 | 12.45 | 100% |
| 2048 / 10 steps / FP32 | 完成，透明洞仍在 | 14.33 | 100% |

- 新执行方式的 FP16 与原始 FP16 head 的 **全部 11 个 RGBA 图层逐字节一致**。
- FP16／FP32 的输入像素、conditioning latent、初始及每步噪声、prompt/group、scheduler 均一致；FP32 改变了像素细节，但没有消除本次故障。
- 两种精度各完成 53 次 UNet 输入／输出、scheduler latent、VAE／透明 decoder 裁剪前输出的有限性检查，未见 NaN／Inf。完整流程中六个已完成 case 合计 1308 次检查也未见 NaN／Inf；这不是检查了每一个内部算子。
- 本节 ROI 位于 head 输入的中央可见脸颊，共 58,344 像素；与上一节整图 ROI 不是同一个区域，不混合比较比例。

[同输入、同噪声的 FP16／FP32 原始 face 与 alpha 对照](../workspace/kaggle-quality-controls-20261026T211949Z/head-precision-analysis/matched-head-precision.png)

### 内存中断与导出毛边应分开处理

- 一个 FP32 head 参考在第 3 次 UNet 调用期间被系统 SIGKILL，`oom_kill` 增量为 1，证实触发宿主 30 GiB RAM 上限。加入仅诊断用的空闲堆归还后，同输入参考完整完成。此结果不能证明历史每次 Web 退出都是同一原因，也不是已验证的正式服务修复。
- 原有 PSD 导出存在独立 alpha／预览问题：被测图层的用户 mask 重复应用像素 alpha；缓存合成预览又将低 alpha RGB 当作不透明像素显示。独立原始 alpha 合成可去除代表性边缘亮刺，但**没有修复原 PSD**。`force=True` 也不是完整修复。
- 禁止用侵蚀轮廓掩盖毛边；禁止把缓存预览或重复 alpha 的 PSD 合成结果当成模型质量基准。

### 原始证据与复现入口

本机附件根目录：[`workspace/kaggle-quality-controls-20261026T211949Z`](../workspace/kaggle-quality-controls-20261026T211949Z)。大型产物不保证随仓库分发；远端 notebook 为账号私有资源，需要相应权限。

- [最终诊断与证据索引](../workspace/kaggle-quality-controls-20261026T211949Z/final-diagnosis.json)
- [完整流程原始矩阵](../workspace/kaggle-quality-controls-20261026T211949Z/runtime/quality_controls/matrix-results.json)
- [精度与执行等价性逐图层分析](../workspace/kaggle-quality-controls-20261026T211949Z/head-precision-analysis/controlled-quality-analysis.json)
- 完整流程：私有 Kaggle 验证 notebook（version 1）。本机 `submission/` 保存实际 notebook 与 metadata。
- 原始 FP16 精度参考：私有 Kaggle 验证 notebook（version 3）。本机 `precision-two-chunk-submission/` 保存实际提交。
- FP32 head／匹配 FP16 head：私有 Kaggle 验证 notebook（version 4／5）。本机分别为 `fp32-head-trim-reference-submission/`、`fp16-head-matched-reference-submission/`。
- [此前 CPU 原生堆与 PSD alpha 诊断](../workspace/kaggle-diagnostics-20261026T192040Z/diagnosis.json)。

OOM、超时、权重缓存加载失败或仅部分迭代的尝试均不算有效质量结果。最终四个来源版本均已观察到 Kaggle `COMPLETE`；未产出日志／结果的额外补充 notebook 已删除，不纳入结论。

### 对正式服务的当前决定

保持正式 Web 上限默认 1536。后续正式服务已完成同进程连续请求实测：2K 单图及重复输出可用，但长驻进程稳定性失败，详见下一节。控件上限、单次成功或降低分辨率都不等于服务已稳定；不要同时运行双 worker 批处理与 Web 服务。

## 2026-10-27：正式服务同进程稳定性

### 总体结果：未通过

本轮计划 13 个真实 HTTP 请求，实际执行 10 个：6 个成功、1 个预期 CUDA OOM、3 个非预期失败。缓存场景的后续 3 个请求因服务进程已被杀而未执行，**不计为通过**。Kaggle `COMPLETE` 只代表采集脚本完成收尾，不代表稳定性通过。

本轮使用正式 `app_gradio.py`，保留原始 builder、加载顺序、seed 位置、attention、group offload 与释放逻辑。没有 `malloc_trim`、CPU-first 加载干预、逐请求重启或推理重试。只加入请求／阶段记录、只读内存端点、模型弱引用与资源采样。一个场景内始终使用同一服务 PID；场景之间才重建服务用于隔离。

环境、源码 SHA256 与权重 revision 与前述质量验证一致：T4×2 中仅 GPU0 可见，FP16 UNet/text、FP32 VAEs、Marigold 768、30 steps、expandable segments。权重预热并核对后离线加载；客户端通过容器内真实 Gradio HTTP API 请求，不使用公网隧道。

### 场景结果

| 场景 | 同进程实际执行 | 结果 |
|---|---|---|
| `KEEP_RESIDENT=False` 连续请求，PID 308 | 1536、1536、2048、2048、1536 换图、1536 返回原图 | 前 5 次成功；第 6 次在重新构建 LayerDiff 期间被 SIGKILL，cgroup `oom_kill` 增量 1 |
| `KEEP_RESIDENT=True`，两个并发 HTTP 客户端，PID 562 | 两个 1024 请求 | 队列观察到 `max_active=1`；先完成 1 个请求，另 1 个进入 LayerDiff 后被宿主 RAM OOM 杀死 |
| `KEEP_RESIDENT=False`，真实 OOM 后恢复，PID 645 | 2560 → 1536 | 2560 返回显存不足；同进程随后的 1536 仍显存不足，恢复失败；无宿主 OOM kill，服务可关闭 |

所有成功 PSD 均实际下载、解析并核对目标尺寸。1536 相同输入／seed 的两次输出，以及 2048 相同输入／seed 的两次输出，其原始图层 RGBA、几何、可见性与 opacity 指纹分别一致。2048 正式服务请求本轮约 66.7 分钟／次，包含加载、HTTP 输出处理与诊断开销，不是纯去噪性能基准。

首个 PSD 在后续请求完成后、缓存年龄约 2182 秒时仍可通过文件 URL 下载，SHA256 未变化；该检查发生在配置的一小时有效期内。

### 故障一：成功请求结束后，CPU 原生堆仍滞留

以下为低内存场景在 HTTP 请求完成、队列空闲时的服务进程快照，不是阶段中途的工作集：

| 已完成请求 | 分辨率／输入 | RSS GiB | glibc 报告空闲堆 GiB | CUDA 已分配 MiB | 已注册活跃模型对象 |
|---|---|---:|---:|---:|---:|
| 1 | 1536／原图 | 14.52 | 12.53 | 8.13 | 0 |
| 2 | 1536／原图重复 | 14.36 | 12.59 | 8.13 | 0 |
| 3 | 2048／原图 | 14.48 | 12.67 | 8.13 | 0 |
| 4 | 2048／原图重复 | 25.46 | 23.69 | 16.25 | 0 |
| 5 | 1536／另一张图 | 26.39 | 24.60 | 16.25 | 0 |

第 6 次开始重新构建 LayerDiff 后，没有构建完成事件；采样到服务 RSS 约 29.35 GiB，宿主 30 GiB 上限触发 OOM kill，服务退出码 `-9`。这是宿主 RAM 故障，不是显存不足返回错误，也不是某张输出图片被静默降质。

**[INFERENCE]** 结合此前空闲堆归还的因果控制，正常请求的主要滞留来自原生分配器空闲页与下一次加载的瞬时峰值，而不是仍持有整套正常请求模型。`fordblks` 不是可回收 RSS 的逐字节等价量，不能把该计数直接当作可回收物理内存保证。本轮没有将堆归还作为补丁加入基线。

缓存模式也未解决问题：第一个请求结束后，两条管线仍按设计缓存，同时存在大量空闲堆；第二个请求执行期间发生宿主 OOM。串行排队能防止两个模型调用重叠，不能保证一个长驻进程的内存安全。

### 故障二：CUDA OOM 后进程仍活着，但没有恢复可用

| 同进程步骤 | 结果 | 返回后 CUDA 已分配 GiB | 已注册活跃模型对象 | 管线缓存 |
|---|---|---:|---:|---|
| 2560 请求 | 返回显存不足 | 10.01 | 4 | 空 |
| 随后 1536 请求 | 在新构建阶段再次显存不足 | 4.40 | 0 | 空 |

这不是只有 CUDA allocator 的 reserved cache：`memory_allocated()` 本身仍有大量活跃分配。仅清空 `_PIPES`、调用 `gc.collect()`／`empty_cache()` 或降低下一次分辨率，没有使该进程恢复。

**[INFERENCE]** 异常路径保留的活跃张量／引用污染了下一次构建。异常 traceback 可能参与，但本轮未完成具体所有者的引用链定位，不能声称已证明是哪一个 Gradio 对象持有资源。弱引用仅覆盖构建完成后已注册的管线／模块；失败构建中的对象未必已注册，因此最后一行的 0 不能解释为没有任何模型／张量残留。

### 日志停在 286.7 秒，而页面仍显示 Running

用户观察到页面日志停在 286.7 秒、运行计时已达 7162 秒。现场读取实时 API 后发现：

- 286.5／286.6 秒对应权重预热完成。
- 后续真实请求完成事件分别出现在约 2477、4653 秒；当时正在第 3 次 2048 请求。
- 实时心跳已到 7369 秒以后，服务仍在，现场 OOM kill 计数尚为 0。后续故障发生得更晚。
- 日志客户端还独立发生过 `ChunkedEncodingError / Response ended prematurely`，不能将日志传输断开等同于模型退出。

因此该次“停在 286.7”是页面日志未同步到后续实时事件的表现，不是停在权重下载。心跳只证明进程存在，必须同时看请求开始／结束、阶段日志与资源记录，不能只依据 Running 时长判定进展。原始证据见附件中的 `log-stall-diagnosis.json` 与 `live-stall-inspection.txt`。

### 记录边界与当前处置

- 当前不能将 `KEEP_RESIDENT=True` 或 False 的单个长驻进程宣称为稳定服务；降低到 1536／1024 也不是已验证的连续运行保证。
- OOM 后不要仅降低分辨率并继续热重试该进程；本轮实际恢复失败，需要退出有残留的推理进程。
- **修复方向 [INFERENCE]**：父 Web 保留队列／结果，每次推理使用隔离 worker 生命周期，使 CPU／GPU 资源随 worker 退出回收。该方向尚未实现或验证为正式修复；本轮没有修改产品推理代码。
- 客户端也运行在该 Kaggle 宿主，cgroup 包含基础内核及客户端开销；记录了服务自身 RSS，不能把 cgroup 总量等同服务 RSS。
- 缓存场景某个客户端返回时，另一个排队请求已开始；`active>0` 时出现的临时目录／显存不判为上一请求泄漏。低内存成功请求的空闲快照中，临时请求目录均已清理。
- cgroup peak 是整个内核生命周期累计值，不把前一场景的峰值当作后一场景独立峰值。两个有限样例不是无限期或任意输入保证。
- 独立 alpha 参考图仅用于检查模型结果，不能宣称此前 PSD mask／缓存预览问题已修复。

### 稳定性原始证据

附件根目录：[`workspace/kaggle-service-stability-20261027T083238Z`](../workspace/kaggle-service-stability-20261027T083238Z)。远端为 私有 Kaggle 验证 notebook（version 1），提交超时参数为 32400 秒。大型附件与私有 notebook 的分发限制同前述质量验证。

- [稳定性诊断与逐请求结论](../workspace/kaggle-service-stability-20261027T083238Z/stability-diagnosis.json)
- [试验方案与通过条件](../workspace/kaggle-service-stability-20261027T083238Z/protocol.json)
- [可重放提交 notebook](../workspace/kaggle-service-stability-20261027T083238Z/submission/see_through_kaggle.ipynb)
- [原始场景矩阵](../workspace/kaggle-service-stability-20261027T083238Z/runtime/service_stability/matrix-results.json)
- 每场景的 `server-events.jsonl`、`host-samples.jsonl`、`gpu.csv` 与 `server.log` 位于 `runtime/service_stability/<scenario>/`。
- [正式服务实际完成的 2K 参考图](../workspace/kaggle-service-stability-20261027T083238Z/runtime/service_stability/lowmem-soak/requests/s03-r2048/reference-black.png)
- [页面日志停滞现场诊断](../workspace/kaggle-service-stability-20261027T083238Z/log-stall-diagnosis.json)

所有三个实验服务进程均已结束，最终 Kaggle 状态为 `COMPLETE`；稳定性结论仍是 **FAILED**。

## 2026-10-28：逐请求推理进程隔离

### 实现与不变项

- `app_gradio.py` 仅拥有 Gradio 队列、临时请求目录、输出缓存与进程监督，不导入 torch 或推理 runner；每次请求通过 `inference_process.py` 启动同一解释器的全新 `inference_worker.py`，不 fork 已初始化的 CUDA。
- worker 仍依次执行正式 LayerDiff、Marigold、PSD 组装；模型 ID、加载顺序、group offload、dtype、seed 位置、steps、attention 与导出算法不变。没有 `malloc_trim`、自动重试、降分辨率或降精度回退，也没有阶段间再分进程。
- Gallery 在 PSD 组装修改原 PNG 之前按字节复制；worker 提交原子结果清单后退出。父进程等待退出并回收，再将 PSD 与 gallery 文件复制进 Gradio 缓存，最后清理请求目录。
- UI 增加取消按钮；异步请求的取消与超时均等待 worker 终止、必要时强杀并回收，再释放下一请求。SIGTERM／SIGINT 停止接单并回收，SIGUSR1 等待当前请求和 Gradio 输出交接，之后保留有上限的下载宽限。
- 父进程独占 stdin 管道写端；父进程强杀或意外退出会触发 worker 的 EOF 守卫，使其进程组退出。它不是独立 cgroup／RAM 沙箱，也不能保证宿主 OOM 时父 Web 一定存活；单次请求峰值、双卡显存不合并、PSD alpha／预览问题均未改变。

### 本地检查

环境为 macOS arm64、uv Python 3.12、Gradio 6.12.0，无本地 CUDA。以下只证明对应本地生命周期，不替代真实 GPU 模型验收。

| 实际检查 | 结果 |
|---|---|
| stdlib OS 进程回归 | 6 项通过：取消、排队中取消不影响当前 worker、崩溃后新 PID、超时强杀、排空拒绝新任务、父进程强杀触发 EOF 守卫 |
| 真实 Gradio HTTP API，0.001 秒请求期限 | 连续两次返回超时；每次之后 Web HTTP 200 |
| Ego 实际页面 | Run／Cancel 可见，未上传图片时显示明确错误；截图已人工读取检查 |
| 重复 SIGTERM、真实不响应 TERM 的 OS 子进程 | 初版 handler 重入清理锁，12 秒未退出，线程转储定位 `_termination_lock`；改为清理线程后父进程 5.66 秒退出，返回码 0，worker 已退出 |
| 当前产品 Python 与 Notebook 代码格 | 编译通过 |

另一次本地 HTTP 冒烟实际返回 `No module named torch`，仅证明子进程导入失败没有杀死父服务，**不计为超时或模型推理通过**。重复信号检查使用真实 Gradio 与 supervisor、只等待信号的 OS 进程，不生成假 PSD／图像。

附件根目录：[`workspace/kaggle-request-isolation-20261027T171858Z`](../workspace/kaggle-request-isolation-20261027T171858Z)（目录时间为 UTC）。本地结果见 [`local-verification.json`](../workspace/kaggle-request-isolation-20261027T171858Z/local-verification.json)、[`local-process-tests.txt`](../workspace/kaggle-request-isolation-20261027T171858Z/local-process-tests.txt) 与 [`local-ui.png`](../workspace/kaggle-request-isolation-20261027T171858Z/local-ui.png)。真实 GPU 的协议与版本另外冻结在 [`gpu-protocol-v2.json`](../workspace/kaggle-request-isolation-20261027T171858Z/gpu-protocol-v2.json)，不能把提交成功算作验收通过。

### 真实 GPU 结果：本轮有界验收通过

远端 私有 Kaggle 验证 notebook（version 2） 使用私有源码 Dataset version 2。所有场景启动时核对了 app、supervisor、worker 与正式 runner 的 SHA256，权重 revision 与前述诊断一致。环境为隔离 Python 3.13、torch 2.8.0+cu128、CUDA 12.8、Gradio 6.12.0；本轮 `pip check` 通过。仍只用 GPU0、FP16 UNet/text、FP32 VAEs、Marigold 768。

**19 个预定真实 HTTP 请求全部执行：13 个成功产物、6 个预期故障／停止、0 个非预期失败。** 主场景使用同一父 PID **364**，连续执行 14 个请求，存活约 6.30 小时；19 次请求分别使用新 worker。另四个场景才更换父进程，用于不同超时配置与关闭行为。没有测试替身或模型重试；Kaggle `COMPLETE` 不是通过依据。

| 场景 | 实际结果 |
|---|---|
| 1536、重复 1536、2048、重复 2048、换图 1536、返回原图 1536 | 全部成功；两次 2K 分别 3895.3／3897.2 秒，均为 2048×2048、17 层 PSD、24 项 gallery |
| 2560 CUDA OOM → 1536 | 约 63.5 秒明确返回显存不足；同父进程的新 worker 随后成功，未自动重试或偷偷降分辨率 |
| 两个同时提交的 1024 HTTP 客户端 | 两个结果均成功，`max_active=1`，模型调用没有重叠 |
| 强杀活跃 GPU worker → 1024 | SIGKILL 明确返回；下一 worker 成功 |
| 实际 `Job.cancel()` → 1024 | cancel 返回 True、旧 worker 已退出；下一 worker 成功 |
| 600 秒期限的 2048 → 512 / 10 steps | 603.2 秒客户端收到超时；同父进程的新请求成功，289.9 秒 |
| 活跃请求期间 SIGUSR1 排空 | 512 / 10 steps 完成，PSD 与 24 项 gallery 已下载；保留 10 秒下载端点后父进程返回 0 |
| 活跃 GPU worker 期间 SIGTERM／SIGKILL 父进程 | 两个 worker 均实测已退出；父进程分别返回 0／-9。后者为有意注入，不是宿主 OOM |

所有场景 cgroup `oom_kill` 增量均为 **0**。故障控制均在实际观测到 GPU 使用后执行；不是只杀一个尚未进入 CUDA 的空 Python。512 / 10 steps 两次产生 18 层 PSD，只作超时／交接生命周期证明，**不作为低步数质量通过**。

### 输出未因隔离而变化

- 1536 与 2048 均与旧正式服务的成功结果 **PSD 文件逐字节相同**；逐层 RGBA、名字、几何、可见性、opacity 也完全相同。
- 1536 重复、换图后返回原图、OOM 后恢复，以及 1024 崩溃／取消后恢复，均与对应正常结果一致；每组 24 个 gallery 的 RGBA 像素也一致。
- worker 在正式 `further_extr` 之前执行 gallery 快照；manifest 保存了原子结果清单与快照文件 SHA256。父缓存不再引用已销毁的请求目录。
- 第一个 PSD 在完成第二次请求后、文件年龄约 2110.4 秒时重新通过公开文件路由下载，SHA256 未变。
- 全部 **13 份 PSD 实际下载到本机，独立重新解析并核对文件 SHA256、每一层像素与元数据**，分辨率均与请求相同。实际查看了新的 [2K 独立 alpha 参考图](../workspace/kaggle-request-isolation-20261027T171858Z/runtime/service_stability/isolated-primary/requests/s03-r2048/reference-black.png)；这不等同修复 PSD 的重复 alpha mask／缓存预览。

### 资源与未覆盖边界

| 本轮观测量 | 结果 |
|---|---:|
| 主场景 14 次请求返回时父进程 RSS | 150.21–174.63 MiB |
| 主场景父进程观测 HWM 峰值 | 174.63 MiB |
| 模型 worker 观测 HWM 峰值 | 22.93 GiB |
| GPU0 NVML 采样峰值 | 14531 MiB |
| Web 存活、worker 空闲时 GPU0 采样 | 3 MiB |

父进程所有观测点均未导入 torch／推理 runner，主场景每次请求返回时 worker 均已回收、请求目录均已清理。客户端返回后若另一排队请求已经开始，出现该新请求的目录不判为泄漏。父进程 RSS 仍有轻量增长，有限序列不能证明无限期无泄漏。CUDA 所有权依据模块边界、worker 生命周期与显存回落；未单独采集 PID 级 NVML 上下文表。

单次 worker 的 CPU 峰值仍大，共享 cgroup 包含基础内核、客户端与文件缓存，累计 peak 到达约 30 GiB 上限且出现回收压力；零 OOM kill **不代表有内存余量**。进程隔离不是硬内存沙箱，宿主 OOM 仍可能杀死父 Web。父进程强杀时 EOF 守卫保证本轮 worker 退出，但不能保证临时文件清理或代替宿主 init 回收僵尸 PID。

本轮 GPU 请求走容器内真实 Gradio API；UI 在本地 Ego 浏览器做了实际视觉检查，未重新外推为新架构的公网隧道或 cloudflared 验收。排空下载宽限本轮设为 10 秒，产品默认 60 秒；它不是大文件必然传完的保证。GPU 每 2 秒采样可能遗漏短时分配，故障动作另有即时 GPU 使用记录。

2560 仍是真实 CUDA OOM，没有有效质量输出；两个样例与有限 seed 不能认证任意图像。更高分辨率需要另行改变显存方案，不能把 T4×2 当作合并显存。原始模型质量与 PSD alpha／预览问题仍与稳定性分开处理。

初次 version 1 在配置格缺少 `Path` 导入，未进入任何模型请求，未计入本轮通过数；version 2 补齐导入并完成矩阵。日志传输曾出现 SSL／ChunkedEncodingError，只重连日志，没有重新提交模型请求。验收结束时 GPU 配额剩余 2.37h（当周 30h，2026-10-31 刷新）。

### 隔离验收原始证据

- [逐项结论、源码版本与全部 19 请求](../workspace/kaggle-request-isolation-20261027T171858Z/isolation-diagnosis.json)
- [全部 13 份实际 PSD 的本机独立复核](../workspace/kaggle-request-isolation-20261027T171858Z/artifact-integrity.json)
- [冻结的 v2 协议](../workspace/kaggle-request-isolation-20261027T171858Z/gpu-protocol-v2.json) 与 [可重放提交 notebook](../workspace/kaggle-request-isolation-20261027T171858Z/submission/see_through_kaggle.ipynb)
- [原始矩阵](../workspace/kaggle-request-isolation-20261027T171858Z/runtime/service_stability/matrix-results.json)；每场景 `server-events.jsonl`、`host-samples.jsonl`、`gpu.csv`、`server.log` 与客户端结果位于相邻目录。
- [原生实时日志（含传输重连）](../workspace/kaggle-request-isolation-20261027T171858Z/live-follow-v2.log) 与 [源码快照 manifest](../workspace/kaggle-request-isolation-20261027T171858Z/source-manifest-v2.json)

五个远端服务均已结束，Kaggle 最终状态 `COMPLETE`；本轮产品推理源码版本的结论为 **PASSED_BOUNDED_VALIDATION**，不是全输入或永久在线保证。

## 2026-10-28：main Run All 未出现公网链接

### 根因与对照

读取用户实际运行的 `已脱敏的 Kaggle notebook` 源码和原生日志后，确认其配置为 `ENABLE_WEBUI=True`、`TUNNEL='gradio'`，并非关闭了公网模式。约 235.5 秒的日志已进入 Cell 5：GPU0 分到 3 张测试图，GPU1 分到 2 张；采集到的日志尚无 Cell 8 启动记录。

当时的 `main`（`19547e1`）没有批处理总开关。Run All 必须先在 Cell 5 等待所有批处理 worker 退出，再执行 Cell 6 校验、Cell 7 打包，最后才在 Cell 8 创建公网链接。因此“开启 Web”不等于“马上启动 Web”，即使分享配置正确，也会先被五张图片的批处理阻塞。

此前成功的公网测试 notebook 删除了批处理单元，依赖安装后直接启动 Web；最近 request-isolation 验收则使用容器内 HTTP API、关闭 share。核心推理源码相同，不足以证明完整 main Run All 的公网启动流程相同。这是此前验收遗漏的入口流程差异。

### 修复与验证

- 新增 `RUN_BATCH=False` 默认配置；Cell 5、6、7 分别受该开关控制。默认 Run All 完成依赖安装后直接进入 Web 启动；需要批处理时显式设为 True。`ENABLE_WEBUI` 继续独立控制 Web，不改模型、dtype 或进程隔离实现。
- 全部 notebook 代码格编译通过；三个启用分支的 AST 与原批处理／校验／归档实现完全一致。本轮没有重新跑启用状态下的 GPU 批处理。
- 在 私有 Kaggle 验证 notebook（version 1） 使用独立 CPU 实例、最小 Gradio 6.12.0 环境，执行修复后的真实 Cell 2、3、5–9。没有启动批处理 worker、生成批处理文件或发起模型推理。
- 专项测试约 53.8 秒生成 `gradio.live` 链接，已从本机 Ego 浏览器实际打开，看到完整上传／分辨率／Run 界面，起始分辨率 1024、上限 2048。该时间不包含正式路径的完整 CUDA PyTorch 依赖安装，不能当作用户 Run All 的总耗时保证。
- 测试设置 180 秒保活，随后走产品默认 60 秒下载宽限，Web 返回码 0，Kaggle 最终 `COMPLETE`。推理核心四个文件 SHA256 与此前 GPU 验收完全相同；没有占用额外 GPU 任务，也未停止或重启用户正在运行的 notebook。

### 使用边界与证据

GitHub 更新不会替换已导入 Kaggle 的单元。需重新导入更新后的 notebook，保留自己的 Cell 2 配置；仅给旧版配置添加 `RUN_BATCH=False` 不足以跳过未受控的旧 Cell 5–7。若不想中断当前批处理，可以等其完成再运行 Cell 8；本次没有代替用户取消任务。

本地 macOS 的完整流程探针曾遇到非本机 `/kaggle/working` 缓存路径及新缓存缺少 frpc 下载文件，未计为成功；远端 CPU 专项使用可写缓存并完成了真实公网访问。静态 Kaggle logs 接口在运行中仅返回换行，原生实时流才提供执行阶段；日志采集端的读取超时不等同服务退出。cloudflared 与正式 fp32 回退仍未完成端到端验收。

原始附件位于 [`workspace/kaggle-public-link-20261028T073158Z`](../workspace/kaggle-public-link-20261028T073158Z)：[根因与验证结论](../workspace/kaggle-public-link-20261028T073158Z/public-link-diagnosis.json)、[用户实际运行日志](../workspace/kaggle-public-link-20261028T073158Z/native-live-log-probe.txt)、[远端完整收尾日志](../workspace/kaggle-public-link-20261028T073158Z/cpu-startup-finished.log)、[远端结果 JSON](../workspace/kaggle-public-link-20261028T073158Z/remote-output/public-startup-result.json)、[实际公网页面截图](../workspace/kaggle-public-link-20261028T073158Z/remote-public-ui.png) 与 `cpu-submission/` 可重放 notebook。测试公网链接已随正常关闭失效，不作为常驻服务入口。

## 2026-10-28：Gradio 真实采样步进度

### 根因与修复

旧 UI 的 `0/3` 统计 LayerDiff、深度和 PSD 三个大阶段，而不是采样步。LayerDiff 内部还包含身体与头部两次采样，因此推理正常时也会长时间显示 `0/3`。原生日志能看到采样进展，不代表这些步数已经传给 Gradio。

LayerDiff 与 Marigold 的 scheduler loop 新增可选 CPU 整数回调；runner 区分身体、头部和深度阶段，独立 worker 通过结构化 stdout 消息实时回传。UI 按当前阶段显示真实步数；加载模型、准备输入、解码／保存和 PSD 组装显示阶段文字，不用计时器估算百分比。没有读取进度张量或主动同步 CUDA，不改变模型计算、dtype、seed、offload 或 PSD 算法。

还复现了合并 stdout／stderr 时的分帧问题：原生 tqdm 的未换行文本会与进度 JSON 粘连，使监督进程忽略更新。worker 现在将小型进度记录加前后换行，并以单次原子 pipe write 发出。新增的真实 OS 进程回归在修复前等待中间进度超时，修复后通过；覆盖的是推理退出前的消息送达，不是假模型输出测试。

### 实际验收

在 私有 Kaggle 验证 notebook（version 2） 使用 Tesla T4 GPU0、torch `2.8.0+cu128`、CUDA 12.8、Gradio `6.12.0`，从非 Kaggle Ego 浏览器实际上传 `common/assets/test_image.png`，点击 Run **一次**。参数为 512 / LayerDiff 10 steps / seed 42 / 不拆左右；深度仍使用原有 4-step preset、768 分辨率。专项直接启动 app，未改变生产 notebook 的 `WEB_MAX_RESOLUTION=2048`。

| 采样阶段 | worker 完整回传 | 浏览器实际观察 | 视觉证据 |
| --- | --- | --- | --- |
| 身体 | 0–10 / 10，11 条 | 0–9 / 10；截图为 60% | [身体进度](../workspace/kaggle-progress-20261028T090221Z/browser-body-progress.png) |
| 头部 | 0–10 / 10，11 条 | 0–9 / 10；截图为 10% | [头部进度](../workspace/kaggle-progress-20261028T090221Z/browser-head-progress.png) |
| 深度 | 0–4 / 4，5 条 | 0–3 / 4；截图为 25% | [深度进度](../workspace/kaggle-progress-20261028T090221Z/browser-depth-progress.png) |

三种采样的中间更新均早于结果返回。Gradio 会合并紧邻的最终步／解码阶段更新，不保证每个百分比都单独渲染；阶段切换会重置当前阶段百分比，它不是整次请求的总耗时比例。单个采样步内部仍没有更细进度。

从请求进入到产生输出约 **265.7 秒**；gallery 24 项。通过真实浏览器下载并在本机独立解析 **512×512、18 个像素图层**的 PSD：二进制 SHA256 为 `410628fed43efe18607ac663deca5cb1f00b02a78ebd6c794addd4a160728440`，图层 RGBA／名称／位置／可见性指纹为 `285ba2a80ef2a98a1ca30e5e5079d5311272ef2dc0022c2d5fc292551072602e`，均与此前相同参数的 request-isolation 基线完全一致；远端保存的 PSD 也逐字节一致。

远端校验冻结 manifest 中的全部 99 个文件；8 个本轮修改的推理／Web／测试文件均与实际运行源码哈希一致。本地 11 个 Python 文件语法检查通过，7 项 OS 进程回归全部通过，包括取消、排队取消、超时强制回收、SIGKILL 后新请求、父进程死亡、正常排空，以及未换行 stderr 下的实时进度送达。此次 CPU 回归不替代 GPU 推理证明。

完成后保留公网下载端点 120 秒，再正常关闭，服务返回码 0，Kaggle 最终 `COMPLETE`；浏览器测试空间已关闭。没有停止或重启用户正在运行的 notebook。

### 范围与原始证据

本次只验证一个 512 请求的真实进度与输出不变性，没有重跑 1536／2048 稳定性或质量矩阵，也没有新增 cloudflared／完整 fp32 回退保证。首次 version 1 因 Kaggle 自动展开上传的 zip 而在测试 bootstrap 停止，未进入任何推理；version 2 改为校验展开后的 manifest 与逐文件哈希。日志采集曾遇到 SSL record-layer failure，只重连日志，没有重复提交模型请求。

- [结论、参数、源码哈希与测试边界](../workspace/kaggle-progress-20261028T090221Z/progress-diagnosis.json)
- [远端最终 verdict](../workspace/kaggle-progress-20261028T090221Z/remote-output/progress-verification/verdict.json)、[真实 worker 事件](../workspace/kaggle-progress-20261028T090221Z/remote-output/progress-verification/events.jsonl) 与同目录源码／模型 attestations、资源采样和服务日志
- [浏览器实际步数记录](../workspace/kaggle-progress-20261028T090221Z/browser-progress.json)、[完成界面](../workspace/kaggle-progress-20261028T090221Z/browser-completed.png) 与[浏览器下载 PSD 的独立复核](../workspace/kaggle-progress-20261028T090221Z/browser-output-validation.json)
- [自包含验证 notebook](../workspace/kaggle-progress-20261028T090221Z/see_through_progress.ipynb) 与 [CLI metadata](../workspace/kaggle-progress-20261028T090221Z/kernel-metadata.json)；测试公网链接已随正常关闭失效。
