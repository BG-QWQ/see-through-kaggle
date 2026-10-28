# Kaggle T4×2 推理与公网 Web UI

仅推理：图片 → LayerDiff 分层 → Marigold 深度 → PSD。不包含训练。

## 使用

1. 在 Kaggle 网页 Import Notebook 上传 [`see_through_kaggle.ipynb`](see_through_kaggle.ipynb)，确认 **GPU T4 x2**、**Internet On**。
2. **只改 Cell 2 配置区**。默认 `REPO_URL` 为 [`BG-QWQ/see-through-kaggle`](https://github.com/BG-QWQ/see-through-kaggle)，包含本集成；也可改成其他完整 fork，或上传完整仓库为 Kaggle Dataset 并设置 `REPO_DIR`。重跑 Cell 3 会将已有工作副本的 origin 对齐 `REPO_URL`，只允许 fast-forward 更新；分叉或本地冲突会停止，不强制覆盖。Cell 3 检查 dtype CLI、`app_gradio.py`、`inference_worker.py`、`inference_process.py` 和依赖文件，缺失时停止，避免旧 Dataset 静默运行旧服务。只读 Dataset 会复制到 `DATASET_COPY_DIR`，避免 editable install 写入失败。
3. 默认 `RUN_BATCH=False`、`ENABLE_WEBUI=True`，Run All 会在 Cell 1–4 安装、检查依赖后跳过 Cell 5–7，直接在 Cell 8 创建公网链接，Cell 9 保活。只做批处理时设 `RUN_BATCH=True`、`ENABLE_WEBUI=False`，运行 Cell 1–7。两者都开启时，必须等所有批处理及校验、打包结束才会出现 Web 链接；不要同时跑批处理与 Web UI。依赖安装到 `ENV_DIR` 独立 venv，不修改基础 kernel，也不需要重启 kernel。
4. CLI 提交：修改 [`kernel-metadata.json`](kernel-metadata.json) 的 `<username>`，在本目录运行 `kaggle kernels push -p .`。挂载 Dataset 时也要配置 `dataset_sources`。CLI 用 `machine_shape: "NvidiaTeslaT4"`；网页导入使用 notebook 的 `metadata.kaggle.accelerator: "nvidiaTeslaT4"`。两种路径都需确认实际分配了两张 T4。

### 配置入口

| 变量 | 含义 |
|---|---|
| `REPO_URL` / `REPO_BRANCH` / `REPO_DIR` | 默认本集成 fork、分支或 commit、仓库路径；重跑 Cell 3 对齐 origin 并更新 |
| `HF_TOKEN` | 可选；模型公开，空字符串即可；打印脱敏，仅通过子进程环境传递，不保存登录凭据 |
| `DTYPE` | T4 用 `fp16`，VAE 保持 `fp32`；数值异常回退 `fp32` |
| `RESOLUTION` / `RESOLUTION_DEPTH` / `STEPS` / `SEED` | 批处理参数；steps/seed/depth 也用于 Web UI |
| `RUN_BATCH` | 默认 False，跳过 Cell 5–7；True 才执行批处理、校验与打包。开启后会延后公网服务启动 |
| `INPUT_DIR` / `SAVE_DIR` / `ARCHIVE_PATH` | 输入、输出、ZIP；输入为空时用仓库的测试图片 |
| `NUM_WORKERS` | 一卡一个进程；自动限制为可用 GPU 数，不跨卡拆模型 |
| `BATCH_GROUP_OFFLOAD` | 默认 False；双 worker 保持权重在各自 GPU，避免耗尽共享主机内存 |
| `CUDA_ALLOCATOR_CONF` | 默认 expandable segments；用于子进程，避免 1280 解码显存碎片 |
| `ENABLE_WEBUI` / `TUNNEL` | 是否启动；`gradio`（默认）、`cloudflared`（备用）、`none`（仅容器内） |
| `WEB_REQUEST_TIMEOUT` | 每次 Web 推理上限，默认 5400 秒；传入 `SEETHROUGH_REQUEST_TIMEOUT`，超时终止该请求的 worker |
| `WEB_DRAIN_GRACE` | 正常排空并完成输出交接后保留下载端点，默认 60 秒；传入 `SEETHROUGH_DRAIN_GRACE`，0 表示不额外等待 |
| `WEB_SESSION_SECONDS` | Cell 9 为 CLI/Save & Run All 保活，默认 10800 秒（3h）；0 跳过，正常到期停止接单并排空当前请求 |
| `PROCESS_STOP_TIMEOUT` | 中断/重跑时等待进程退出，默认 15 秒；给 app 默认 5 秒的 worker teardown 留出余量 |
| `WEB_RESOLUTION` / `WEB_MAX_RESOLUTION` / `WEB_PORT` | Web 初始分辨率 / 滑块上限 / 服务端口；初始默认 1024，上限默认 2048，须 ≥512 且为 64 的倍数 |
| `WORK_DIR` / `HF_HOME` / `DATASET_COPY_DIR` | 日志、模型缓存、Dataset 可写副本路径 |
| `ENV_DIR` | 干净的运行时 venv，默认 `/tmp/see-through-venv`；不继承系统 site-packages，不计入 notebook 输出归档 |
| `BOOTSTRAP_PACKAGES` / `TORCH_PACKAGES` / `TORCH_INDEX_URL` | Kaggle 启动钩子依赖、固定 CUDA PyTorch/torchvision wheel、安装源；不安装未使用的 torchaudio |

超时、轮询、显存/alpha 验收阈值、运行时依赖和 cloudflared 下载地址也集中在 Cell 2。修改服务配置后重跑 Cell 8，cloudflared 模式再重跑 Cell 9；启动格子会关闭旧服务，避免同端口与显存冲突。`ENABLE_WEBUI=False` 重跑 Cell 8 会关闭旧服务。

已有 Kaggle Notebook 的单元不会随 GitHub 更新自动替换。要使用新的默认 Web 流程，需重新导入更新后的 `.ipynb` 并恢复自己的 Cell 2 配置；仅在旧配置里添加 `RUN_BATCH=False` 不会跳过旧版未加开关的 Cell 5–7。正在运行的旧批处理可等待完成后再启动 Web；不要在批处理进程仍占 GPU 时另起 Web 推理。

批处理实际使用 `--quant_mode none --dtype fp16`，不需要 bitsandbytes。模型仓库中的 NF4 配置使用 bf16 compute，不适合直接在 T4 上跑。两张 16GB 卡不是一个 32GB 显存池；Web UI 只使用 GPU0。

Web 父进程仅负责队列、结果缓存与轻量监督；每个请求启动全新解释器，在 GPU0 分阶段加载 LayerDiff 与 Marigold，完成、报错或超时后退出，不跨请求缓存管线。请求串行执行。该隔离用于避免旧长驻推理进程的宿主空闲堆／异常张量污染下一请求，不降低单次请求的 CPU／GPU 峰值，也不修复 PSD alpha／预览毛边。连续请求及故障恢复已完成真实 T4 的有限序列验收，范围与原始证据见[验证记录](VALIDATION.md)，不代表任意输入或无限期服务保证。

## 依赖隔离

不要把 `requirements-kaggle.txt` 安装到 Kaggle 基础 kernel。预装的 bigframes、dopamine-rl、moviepy 等不属于推理依赖；修补它们的版本会扩大冲突面。Cell 4 创建不继承系统包的 venv，通过基础解释器的 `pip --python` 管理目标环境；批处理、PSD 验收和 Web 服务都使用 `runtime_python`。基础 kernel 只负责 GPU 枚举、调度与 notebook 展示。

Kaggle Python 的标准库 `sitecustomize` 需要 `wrapt`，即使启用 venv 也会执行。因此先用基础 pip 的 `--target` 将真实的 `wrapt` 包装入目标环境，再首次执行其解释器；不禁用平台钩子。子进程设 `MPLBACKEND=Agg`，不继承 notebook 的 inline 后端；同时移除 `PYTHONPATH` / `PYTHONHOME`，禁用用户 site-packages。

Cell 4 自动执行目标环境的 `pip check`，期望输出 `No broken requirements found.`，随后检查 CUDA、关键导入与推理 CLI。修改运行时版本后重跑 Cell 4，再重启批处理或 Web 子进程；不用重启基础 kernel。首次安装另需下载 CUDA wheel，占用额外临时磁盘空间。

## 分辨率与显存

这里没有 Hugging Face Space 的 `spaces.GPU` 限制，但仍受单张 T4 的显存、Kaggle 主机内存和 session 时长约束。两张 T4 用于两张图片并行，不能把同一张图片当作 32GB 显存来推理。

Web 初始分辨率保留 1024；滑块上限默认 `WEB_MAX_RESOLUTION=2048`，不会在后端偷偷降回 1280。Notebook 将此上限通过 `SEETHROUGH_MAX_RESOLUTION` 传入 app；单独运行 app 时需自行设置该环境变量，未设置时 app 自身的回退上限仍为 1536。若希望页面打开时直接选择已经重复跑通的 2048 路径，在 Cell 2 设置：

```python
WEB_MAX_RESOLUTION = 2048
WEB_RESOLUTION = 2048
STEPS = 30
```

10 steps 只适合链路冒烟，不能据其结果判断高分辨率质量；减少 steps 主要缩短耗时，不能保证解决峰值显存不足。当前 worker 在 LayerDiff 与 Marigold 之间释放管线，每次请求结束后退出。新隔离服务两次 2048 / 30 steps 均成功，约 65 分钟／次，PSD 与修复前成功结果逐字节相同；2560 仍发生真实 CUDA OOM，随后新 worker 的 1536 请求成功。旧长驻服务连续请求曾因宿主空闲堆滞留而 OOM kill，且 CUDA OOM 后同进程降分辨率未恢复；不要混用两版结论。方法、原始数据与边界见[推理验证记录](VALIDATION.md)。Notebook 默认上限已调整为 2048，但逐请求隔离不是单次峰值或长期稳定保证，不要同时启动双 worker 批处理。

## 文件归属与冲突面

集成新增文件：

- 本目录的 notebook、CLI metadata、本维护手册。
- 根目录 [`requirements-kaggle.txt`](../requirements-kaggle.txt)。
- [`inference/scripts/app_gradio.py`](../inference/scripts/app_gradio.py)：复用[作者维护的 Space 布局与流程](https://huggingface.co/spaces/24yearsold/see-through-demo/blob/main/app.py)，不依赖 `spaces.GPU`。
- [`inference/scripts/inference_process.py`](../inference/scripts/inference_process.py)：轻量请求进程监督、超时与停止管理。
- [`inference/scripts/inference_worker.py`](../inference/scripts/inference_worker.py)：每请求独立解释器入口，复用现有推理 API 并输出 PSD/gallery。

推理代码只修改一个上游文件：[`inference_psd_quantized.py`](../inference/scripts/inference_psd_quantized.py)，加入 `--dtype`、组件 dtype 选择、stats dtype 和局部 Marigold 混合精度边界。默认 `bf16` 不变；已有 blockswap 调用不提供 dtype 字段时仍沿用 bf16。根 README 另加一节入口链接（纯文档，不属于推理 patch）。没有修改 `common/`、`inference_psd.py`、`inference_psd_blockswap.py` 或 annotators。

**混合精度边界不能省略**：LayerDiff 已在 VAE 编码后把 conditioning 转回 UNet dtype；当前 Marigold 原实现没有这样做。fp32 VAE conditioning 与 fp16 latent 拼接会提升到 fp32，导致 fp16 UNet 报 dtype mismatch。本集成在同一推理文件内用 `_MixedDtypeMarigoldPipeline.single_infer()` 做一次入口转换；bf16/同 dtype 仍使用原类。此边界已通过 CPU 真实小型模型验证，以及 Kaggle T4×2 完整分层、深度与 PSD 导出验收。

**加载时也要选择 dtype**：T4×2 共享主机内存，不能让两个 worker 各自先构建完整 fp32 UNet/文本编码器再转 fp16。`_load_dtype_kwargs()` 在非 bf16 模式下按组件直接加载目标 dtype；VAE 从加载到推理均保留 fp32，不经过 fp16 舍入。默认 bf16 仍不向加载器传 dtype 参数，保留原加载路径。本地真实 checkpoint 冒烟已验证 fp32 VAE 权重逐项一致。

2026-10-26 的 Kaggle 双 worker 实测还发现：group offload 会耗尽共享 30 GiB 主机内存并触发 OS OOM kill；关闭后，1280 解码需要 expandable segments 避免 CUDA 分配碎片。因此 notebook 批处理默认 `BATCH_GROUP_OFFLOAD=False`，并将 `CUDA_ALLOCATOR_CONF` 传入子进程；PyTorch 2.11+ 使用新环境变量名，旧版本使用兼容名称。Web 每次请求在独立进程中使用分阶段 group offload，不改变推理数学、精度放置、seed 位置、采样、attention 或 PSD alpha 算法。

同步上游：

```bash
# remote 已存在时不用重复 add
 git remote add upstream https://github.com/shitagaki-lab/see-through.git
 git pull upstream main
```

新文件通常不与上游冲突；推理冲突面只有 `inference_psd_quantized.py`，根 README 的入口节也可能遇到文档冲突。冲突时保留上游逻辑和本集成 dtype 相关行，不要用整文件覆盖上游版本。上游未来若新增同名集成文件，也应逐项合并。

## 每次同步检查清单

- [ ] `inference_psd_quantized.py` 的 `REPO_MAP['none']` 模型 ID 是否改变？同步 `inference_worker.py` 的模型参数。notebook 不硬编码模型 ID。Marigold 旧地址目前会重定向到 `layerdifforg/seethroughv0.0.1_marigold`，未来若失效需同步 worker。
- [ ] `build_layerdiff_pipeline` / `build_marigold_pipeline` / `run_layerdiff` / `run_marigold` / `further_extr` 签名是否改变？同步 worker 调用。Marigold `single_infer` 签名或内部 dtype 边界改变时，检查局部子类是否仍必要、是否仍正确。
- [ ] 保留非 bf16 模式的按组件 dtype 加载；不要把 pipeline 整体先转 fp16 再把 VAE 转回 fp32，这会丢失 VAE 权重精度，也不能降低首次 fp32 加载的共享内存峰值。
- [ ] [官方 Space app.py](https://huggingface.co/spaces/24yearsold/see-through-demo/blob/main/app.py) 是否有值得复用的 UI 改进？[Space README](https://huggingface.co/spaces/24yearsold/see-through-demo/blob/main/README.md) 的 `sdk_version` 是否改变？同步 `requirements-kaggle.txt` 的 Gradio pin；当前为 `6.12.0`。
- [ ] 根 `requirements.txt` 的 transformers / diffusers / accelerate 等 pin 是否更新？同步精简依赖，避免引入 Qt、annotators 或训练包。
- [ ] 运行 `python -m py_compile inference/scripts/inference_psd_quantized.py inference/scripts/app_gradio.py inference/scripts/inference_worker.py inference/scripts/inference_process.py`，在装好精简依赖后实际运行推理 CLI `--help` 和本地 Gradio 页面。
- [ ] 运行 `python -m unittest discover -s inference/tests -v` 检查 OS 进程生命周期；它不需要模型依赖，不替代真实 GPU 推理或公网验收。
- [ ] 设 `RUN_BATCH=True`、`ENABLE_WEBUI=False` 后，在 Kaggle 重新运行 Cell 1–7；Cell 6 验收每图 PSD 分层、stats dtype/显存/耗时、非透明/非黑分解图；Cell 7 验收归档包含每张图的 PSD。正常日志应有 `LayerDiff3D done`、`Marigold done`、`PSD assembly done`。
- [ ] 从非 Kaggle 浏览器访问两种隧道之一，上传测试图，确认 gallery 非空并能下载分层 PSD。更换 Gradio pin 时也检查 gallery 与 PSD 在临时目录清理后仍可读取。
- [ ] 新架构实际验收连续 1536／2048 请求、CUDA OOM 后新 worker 降分辨率恢复、输出缓存仍可下载、请求超时、SIGUSR1 排空、SIGTERM 停止及父进程强杀后无孤儿 GPU worker；记录资源与退出结果，不能把启动成功算作 GPU 稳定性通过。

## 回退与平台限制

- 数值异常：Cell 2 改 `DTYPE='fp32'`、`RESOLUTION=1024`，并建议设 `NUM_WORKERS=1`，Web 同时降 `WEB_RESOLUTION`。仍 OOM 时把 `RESOLUTION_DEPTH` 降到 512。不要改成 fp16 VAE，它更容易溢出。fp32 回退的完整 T4 实跑尚未验证。
- 连续服务：历史长驻服务的 `KEEP_RESIDENT=True`／False 均未通过稳定性实测，该配置已删除。当前每请求进程隔离使成功、失败或超时的推理进程退出；OOM 后下一次请求使用新 worker，而非污染进程中的热重试。新架构有限序列验收通过，不表示单次峰值、任意输入质量或 PSD alpha／预览毛边已解决，详见[验证记录](VALIDATION.md)。
- 依赖报错：确认错误来自 Cell 4 的独立环境，而不是基础 kernel 的旧安装日志；不要通过升级/降级系统包解决。运行时版本由 Cell 2 的 `TORCH_PACKAGES` 和精简 requirements 决定，修改后重跑 Cell 4。2026-10-27 的真实 Kaggle 验收已确认隔离环境 `pip check` 通过、CUDA 可用、基础环境包清单未变化。
- 公网隧道：默认 Gradio `share=True` 使用出站 FRP 反代；失败时看 [Gradio 服务状态](https://status.gradio.app/)，改用 cloudflared quick tunnel。两种隧道都不可用时设 `TUNNEL='none'` 或关闭 Web UI，批处理不受影响。Kaggle 容器没有入站端口，不能从外部直连 7860。
- 隧道**没有鉴权**，任何获得链接的人都可用。`queue(max_size=16)` 只限制排队，不能防止滥用；勿处理隐私图。下载输出缓存按小时清理，请及时保存。
- 服务随 session 结束；常见限制为 12h 上限、交互约 20 分钟空闲断开、免费 GPU 30h/周，以 [Kaggle 当前文档与账号额度](https://www.kaggle.com/docs/notebooks) 为准。外部访客请求不保证 session 保活。
- 2026-10-26 实测环境为 Python 3.13、torch 2.11.0+cu128、Tesla T4×2：1280 / 30 steps，两图并行各生成 17 层 PSD；总耗时约 16.9 / 18.6 分钟，peak VRAM 均为 13.01 GiB，共享内存峰值约 29.04 GiB，OOM 计数 0。采用直接 fp16 加载、fp32 VAE、批处理关闭 group offload、expandable segments。这只是两张样例，不保证所有输入的耗时/峰值。
- 首次模型下载约 8GB [INFERENCE]；`/kaggle/working` 保存空间约 20GB，留意模型缓存和 PSD 占用。长期在线服务应使用已有官方 HF Space，不依赖临时 Kaggle session。

## CLI 公网服务生命周期

CLI / Save & Run All 会在 notebook 执行结束后释放 session，因此 Cell 9 除备用隧道外，也负责有上限的公网服务保活。默认 `WEB_SESSION_SECONDS=10800`（3h），避免旧 1h 默认在历史约 66.7 分钟的 2K 请求中途关闭；每次推理的 `WEB_REQUEST_TIMEOUT=5400` 映射为 `SEETHROUGH_REQUEST_TIMEOUT`。交互使用 Gradio 可只运行 Cell 8；cloudflared 还需运行 Cell 9。时长只改 Cell 2，仍受 Kaggle session 上限与 GPU 配额约束；设为 0 跳过保活。不要把 Gradio 日志中的“一周分享链接有效期”误当成 Kaggle 服务在线时长。

正常保活到期时，Cell 9 发送 **SIGUSR1**：app 停止接收新工作，当前请求沿用已经开始的请求超时；完成 Gradio 输出交接后，再保留下载端点 `WEB_DRAIN_GRACE=60` 秒。notebook 保留隧道、持续转发日志，最多再等 `WEB_REQUEST_TIMEOUT + WEB_DRAIN_GRACE + PROCESS_STOP_TIMEOUT`（默认 5475 秒），然后必要时进入强制清理。这个等待不会重置当前请求计时，也不承诺完成队列中尚未开始的请求；下载宽限也不保证大文件一定传完。

手动中断 Cell 9、重跑 Cell 8 或排空超时使用 **SIGTERM → 等待 `PROCESS_STOP_TIMEOUT=15` 秒 → 必要时 SIGKILL**。先关闭 app，再关闭 cloudflared。app 负责停止并回收当前 worker，默认 `SEETHROUGH_WORKER_STOP_TIMEOUT=5` 秒；强杀父 app 时，其持有的 stdin 管道关闭，worker 的 EOF 守卫终止自身进程组，避免正常 GPU 工作遗留为孤儿。强制停止不同于正常排空，会中止当前请求；Kaggle 平台强制回收也不保证完成请求。

## 公网实跑验收

2026-10-28 已在独立 Kaggle CPU 实例验证当前默认 Web 分支：实际执行 Cell 2、3、5–9，跳过批处理、校验和打包，生成 Gradio 公网链接，并从非 Kaggle 浏览器打开页面。此项只验证公网启动，使用最小 Web 依赖；不重跑模型、不消耗额外 GPU 配额，也不代表完整 CUDA 安装耗时或 cloudflared／fp32 验收。根因、原始日志和截图见[验证记录](VALIDATION.md)。

以下均为**旧长驻服务的历史验收**，保留当时配置，不是当前逐请求隔离架构的验收结果。

2026-10-26 已从非 Kaggle 浏览器访问真实 Kaggle T4 上的 Gradio share 链接，上传 `test_image.png`，以 1024 / 30 steps / seed 42 完成一次请求：gallery 24 项，下载并解析了 1024×1024、17 层 PSD，首次请求总耗时 1108.9 秒。当时使用现已删除的 `KEEP_RESIDENT=True`、单进程 group offload 与 expandable segments。cloudflared 备用链路及 fp32 回退仍未做完整远端实跑，不能把 Gradio 路径的验收结果外推到它们。


2026-10-27 已验证新隔离依赖环境（不是请求进程隔离）：Python 3.13、torch **2.8.0+cu128**、CUDA 12.8、Gradio 6.12.0；`pip check` 通过，基础 kernel 的包清单未变化。从公网浏览器上传同一测试图，以 **1536 / 10 steps / seed 42**、当时的 `KEEP_RESIDENT=False`、单进程 group offload 完成请求：gallery **24 项**，下载并解析了 **1536×1536、17 层 PSD**，实际渲染非黑且非透明，服务端首次请求耗时 **937.9 秒**。这证明没有在后端降回 1280；该次公网验收不覆盖其他输入或双 worker 高分辨率。后续 1536／2048 的质量诊断及正式服务稳定性失败另见[推理验证记录](VALIDATION.md)，不能把这些实验合并宣称连续 Web 服务通过。
