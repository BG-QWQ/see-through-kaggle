# Kaggle T4×2 推理与公网 Web UI

仅推理：图片 → LayerDiff 分层 → Marigold 深度 → PSD。不包含训练。

## 使用

1. 在 Kaggle 网页 Import Notebook 上传 [`see_through_kaggle.ipynb`](see_through_kaggle.ipynb)，确认 **GPU T4 x2**、**Internet On**。
2. **只改 Cell 2 配置区**。默认 `REPO_URL` 为 [`BG-QWQ/see-through-kaggle`](https://github.com/BG-QWQ/see-through-kaggle)，包含本集成；也可改成其他完整 fork，或上传完整仓库为 Kaggle Dataset 并设置 `REPO_DIR`。重跑 Cell 3 会将已有工作副本的 origin 对齐 `REPO_URL`，只允许 fast-forward 更新；分叉或本地冲突会停止，不强制覆盖。Cell 3 检查 dtype CLI、Web app、worker、进程监督、`inference_jobs.py` 和依赖文件，缺失时停止，避免旧 Dataset 静默运行旧服务。只读 Dataset 会复制到 `DATASET_COPY_DIR`，避免 editable install 写入失败。
3. 默认 `RUN_BATCH=False`、`ENABLE_WEBUI=True`，Run All 会在 Cell 1–4 安装、检查依赖后跳过 Cell 5–7，直接在 Cell 8 创建公网链接，Cell 9 保活。只做批处理时设 `RUN_BATCH=True`、`ENABLE_WEBUI=False`，运行 Cell 1–7。两者都开启时，必须等所有批处理及校验、打包结束才会出现 Web 链接；不要同时跑批处理与 Web UI。依赖安装到 `ENV_DIR` 独立 venv，不修改基础 kernel，也不需要重启 kernel。
4. CLI 提交：修改 [`kernel-metadata.json`](kernel-metadata.json) 的 `<username>`，在本目录运行 `kaggle kernels push -p .`。挂载 Dataset 时也要配置 `dataset_sources`。CLI 用 `machine_shape: "NvidiaTeslaT4"`；网页导入使用 notebook 的 `metadata.kaggle.accelerator: "nvidiaTeslaT4"`。两种路径都需确认实际分配了两张 T4。

开发修复先推送 `dev` 并验证，再通过指向 `main` 的 PR 交由维护者决定；不自动合并。测试新版时可在 Cell 2 设置 `REPO_BRANCH = 'dev'`。这不会热更新已有服务：请等待当前请求结束后再更新／重启，不要为了测试中断正在运行的任务。

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
| `ENABLE_BACKUP_TUNNEL` | 默认 True；Cell 9 在 Gradio 主入口旁启动独立 cloudflared 结果找回入口；关闭 Web 或 `TUNNEL='none'` 时不启动 |
| `WEB_REQUEST_TIMEOUT` | 每次 Web 推理上限，默认 5400 秒；传入 `SEETHROUGH_REQUEST_TIMEOUT`，超时终止该请求的 worker |
| `WEB_DRAIN_GRACE` | 正常排空并完成输出交接后保留下载端点，默认 60 秒；传入 `SEETHROUGH_DRAIN_GRACE`，0 表示不额外等待 |
| `WEB_SESSION_SECONDS` | Cell 9 为 CLI/Save & Run All 保活，默认 10800 秒（3h）；0 跳过，正常到期停止接单并排空当前请求 |
| `PROCESS_STOP_TIMEOUT` | 中断/重跑时等待进程退出，默认 15 秒；给 app 默认 5 秒的 worker teardown 留出余量 |
| `WEB_RESOLUTION` / `WEB_MAX_RESOLUTION` / `WEB_PORT` | Web 初始分辨率 / 滑块上限 / 服务端口；初始默认 1024，上限默认 2048，须 ≥512 且为 64 的倍数 |
| `WORK_DIR` / `HF_HOME` / `DATASET_COPY_DIR` | 日志、模型缓存、Dataset 可写副本；Web 回执与完成的 PSD/gallery 保存在 `WORK_DIR/see-through-jobs` |
| `ENV_DIR` | 干净的运行时 venv，默认 `/tmp/see-through-venv`；不继承系统 site-packages，不计入 notebook 输出归档 |
| `BOOTSTRAP_PACKAGES` / `TORCH_PACKAGES` / `TORCH_INDEX_URL` | Kaggle 启动钩子依赖、固定 CUDA PyTorch/torchvision wheel、安装源；不安装未使用的 torchaudio |

超时、轮询、显存/alpha 验收阈值、运行时依赖和 cloudflared 下载地址也集中在 Cell 2。修改 Web 服务配置需在当前请求结束后重跑 Cell 8；这会关闭旧服务和当前 worker。Cell 9 在已有服务旁创建 cloudflared，不需要重新启动 Cell 8，但不要中断正在保活的 Cell 9 来改隧道：其中断清理也会关闭 Web。默认 Run All 会提前创建备用入口，主地址断线时使用已经输出的备用链接。`ENABLE_WEBUI=False` 重跑 Cell 8 会关闭旧服务。

已有 Kaggle Notebook 的单元不会随 GitHub 更新自动替换。要使用新的默认 Web 流程，需重新导入更新后的 `.ipynb` 并恢复自己的 Cell 2 配置；仅在旧配置里添加 `RUN_BATCH=False` 不会跳过旧版未加开关的 Cell 5–7。正在运行的旧批处理可等待完成后再启动 Web；不要在批处理进程仍占 GPU 时另起 Web 推理。

结果找回与默认备用入口还依赖新版 Cell 8 的 `SEETHROUGH_JOB_DIR` 和新版 Cell 9；请重新导入当前 notebook 并恢复自己的配置。只拉取 Python 文件不会替换已导入的旧单元，也不会更新正在运行的服务。

批处理实际使用 `--quant_mode none --dtype fp16`，不需要 bitsandbytes。模型仓库中的 NF4 配置使用 bf16 compute，不适合直接在 T4 上跑。两张 16GB 卡不是一个 32GB 显存池；Web UI 只使用 GPU0。

Web 父进程仅负责队列、结果缓存与轻量监督；每个请求启动全新解释器，在 GPU0 分阶段加载 LayerDiff 与 Marigold，完成、报错或超时后退出，不跨请求缓存管线。请求串行执行。该隔离用于避免旧长驻推理进程的宿主空闲堆／异常张量污染下一请求，不降低单次请求的 CPU／GPU 峰值，也不修复 PSD alpha／预览毛边。连续请求及故障恢复已完成真实 T4 的有限序列验收，范围与原始证据见[验证记录](VALIDATION.md)，不代表任意输入或无限期服务保证。

### Web 进度显示

身体采样、头部采样和 Marigold 深度采样分别显示真实的 `已完成步数 / 当前阶段总步数`，由每次 scheduler step 完成后回传，不按耗时推算。切换阶段时百分比会重新开始；它不是整次请求的总耗时百分比。加载模型、输入准备、解码／保存和 PSD 组装显示阶段文字，不伪造可量化进度。单个采样步内部没有更细的进度，较高分辨率下仍可能较久才更新一次。

真实公网浏览器更新、PSD 基线一致性及验收范围见[验证记录](VALIDATION.md)。已有服务不会热更新：请等当前请求结束后重新拉取代码并重启 Web；仅刷新浏览器不会替换服务端 Python 进程。

### 断连后的结果找回

- Run 在开始长推理前创建请求 ID，显示在 `Request ID / recovery ID`，并写入当前 URL 的 `#request=...`。同一 URL 重新打开时会查询该请求；切换到另一公网域名时，复制 ID 到输入框后点 `Check status / retrieve saved result`。这个操作只读状态／提取结果，**不会重新提交推理**。计算中显示真实阶段与步数，需要稍后再次检查。
- 恢复输入框接受 ID 周围的空格、制表符和换行；不完整 ID 或误粘贴完整 URL 会显示具体英文提示，不再把输入问题作为后端异常。ID 本身仍是 32 个小写十六进制字符，`request_snapshot` 和存储层继续严格校验。网页原生 `queue=False` 找回使用有限 HTTP 请求；不能将所有备用入口报错都归因于 SSE。
- 使用 Gradio API 时，先调用 `create_request` 获取 ID，再将 ID 作为 `inference` 的第一个参数；每次新推理使用新 ID。同一 ID 不能重复开始；`recover_request` 只读取既有请求，不是重试接口。
- 完成的 PSD 与 gallery 先移入 `WORK_DIR/see-through-jobs/<ID>/artifacts`，原子提交完成回执后才清理临时目录和交付网页。原图与中间文件不进入这份结果存储；Gradio 展示缓存仍按小时清理，找回操作可从独立存储重新建立缓存。保留范围受 Kaggle session、磁盘和保存方式约束；Save & Run All 保存的版本可能保留这些工作目录输出，并非永久存储或自动删除保证。
- 最终结果落盘交接已经开始时，取消会等待该次文件交接结束，再传播原取消异常；重复取消不会提前清理源目录。交接成功的回执不会被后续取消覆盖，磁盘失败则保留真实错误，不会伪报完成。这不改变模型推理阶段的取消行为。
- 重启 app 后，同一工作目录内的已完成结果仍可按 ID 读取；旧实例的运行中记录只会提示无法确认，不会自动续算。显式取消、请求超时、服务器退出或平台回收仍会终止计算。`Cancel this page's request` 仅取消当前页面提交的事件，不是重开页面后对历史任务的跨会话取消。
- 请求 ID 和已完成结果依赖原工作目录；切换 Kaggle 账号或新建 session 不会自动迁移它们。远端验证应重新核对当前账号的 notebook、输入数据和输出目录，不沿用旧实例的运行状态。
- 请求 ID 不是用户鉴权；请勿同时公开 ID 和公网地址。没有针对敏感内容的访问控制，仍应遵守页面英文安全提示。
- **cloudflared Quick Tunnel 不支持 SSE，也没有在线时长保证**，见 [Cloudflare 官方限制](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/#limitations)。备用入口用于短 HTTP 状态查询与文件下载，不能承诺重放原来的 Gradio 进度流。不要把备用链接启动成功当成长时间实时推理验收。

### Python 客户端：上传与下载独立恢复

[`client_gradio.py`](../inference/scripts/client_gradio.py) 使用短 HTTP 提交／状态查询，不依赖 Gradio 的长 SSE 连接，也不会自动下载整个 gallery。它需要当前代码中的 `request_snapshot` 接口；旧服务需在已有请求完成后更新代码并重启，不能靠刷新网页升级。

在仓库根目录运行，`PRIMARY_URL`／`BACKUP_URL` 分别设为当前 session 输出的主／备用地址：

```bash
uv run --no-project --python python3.12 --with httpx==0.28.1 \
  python inference/scripts/client_gradio.py run "$PRIMARY_URL" common/assets/test_image.png \
  --backup "$BACKUP_URL" --resolution 1024 --steps 30 --output workspace/client-results
```

请求 ID 会在推理提交前保存到 `workspace/client-results/<ID>/request.json`。已生成 ID 后，连接中断或提示提交结果不确定时，**不要再次运行 `run`**，使用同一 ID 找回：

```bash
uv run --no-project --python python3.12 --with httpx==0.28.1 \
  python inference/scripts/client_gradio.py recover "$PRIMARY_URL" "$REQUEST_ID" \
  --backup "$BACKUP_URL" --output workspace/client-results
```

- 上传和读写默认使用 120 秒网络静默超时，连接／连接池等待各 30 秒；`--timeout` 可调整读写超时。`--attempts` 默认 3，是每个可恢复传输操作的总尝试次数，不是推理次数。
- 只有上传、只读状态查询和文件下载会对暂时网络错误、TLS EOF、429／502／503／504 进行有界重试。创建 ID 与推理提交不自动重试；提交回执丢失会保留本地 ID 并退出，不猜测是否需要重跑。证书校验失败、鉴权失败及其他永久 HTTP 错误直接报错，不关闭 TLS 校验。
- 状态查询与下载可轮换到显式配置的备用入口，两者必须指向同一个仍存活的服务。也可直接将备用地址作为 `recover` 的第一个参数；全程不需要 SSE。`--wait-timeout` 默认 7200 秒，只限制客户端轮询等待，不取消后端推理；单次网络操作仍有独立超时。
- PSD 下载先写临时文件，校验服务端报告的大小和 SHA256 后才原子替换 `<ID>/input.psd`。失败不覆盖原有完成文件；不完整或损坏的传输不会伪报成功。同一 ID 的 PSD 大小／哈希若改变则拒绝接受，允许重新建立相同内容的展示缓存。
- 回执包含服务地址和请求 ID，按私有文件保存；默认输出在被 Git 忽略的 `workspace/`。不要公开或提交这些文件。服务关闭、工作目录丢失或两条隧道均不可达时，重试不能恢复已经不存在的结果。
- CLI 的 `--output` 必须解析到当前工作目录内，不能通过 `..` 或目录符号链接写到外部；库调用者仍可显式选择自己的输出根目录。请求子目录、回执及 PSD 符号链接会被拒绝，验证发生在读写边界。



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

### 左右拆分与画质边界

左右拆分属于模型输出之后的连通域后处理。拆分子层继承原层深度排序，并保留相同深度下的原层位置，避免单侧虹膜被眼白盖住；额外孤岛及淡 alpha 不再因仅保留两大区域、掩码膨胀或再次裁切而丢失／重复。相连的部位不能由连通域可靠拆开，仍保留整层。

**2048 推理完成不代表画质合格。** 原始 RGBA 层中已经存在的背景污染、错误补全或语义分层错误，不能由左右拆分、文件校验或请求隔离修复。分辨率／精度的归因需要同一原图、检查点、seed、steps 与 depth 设置对照；不根据“没有 OOM”宣称画质正常，也不在后台偷偷降低分辨率。

## 文件归属与冲突面

集成新增文件：

- 本目录的 notebook、CLI metadata、本维护手册。
- 根目录 [`requirements-kaggle.txt`](../requirements-kaggle.txt)。
- [`inference/scripts/app_gradio.py`](../inference/scripts/app_gradio.py)：复用[作者维护的 Space 布局与流程](https://huggingface.co/spaces/24yearsold/see-through-demo/blob/main/app.py)，不依赖 `spaces.GPU`。
- [`inference/scripts/inference_process.py`](../inference/scripts/inference_process.py)：轻量请求进程监督、超时与停止管理。
- [`inference/scripts/inference_worker.py`](../inference/scripts/inference_worker.py)：每请求独立解释器入口，复用现有推理 API 并输出 PSD/gallery。
- [`inference/scripts/inference_jobs.py`](../inference/scripts/inference_jobs.py)：无模型依赖的请求回执、状态转换与完成结果存储，不重试推理。
- [`inference/scripts/client_gradio.py`](../inference/scripts/client_gradio.py)：带有界安全重试、请求 ID 恢复和 PSD 大小／SHA256 校验的独立 Python 客户端；不重试推理。
- [`common/utils/part_split.py`](../common/utils/part_split.py)：无模型依赖的连通域拆分、局部坐标与 alpha 所有权处理；由原 PSD 后处理入口调用。

上游推理修改集中在 [`inference_psd_quantized.py`](../inference/scripts/inference_psd_quantized.py) 的 dtype／混合精度边界与可选进度回调，以及 [`common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py`](../common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py)、[`common/modules/marigold/marigold_depth_pipeline.py`](../common/modules/marigold/marigold_depth_pipeline.py) 的采样步观察回调；[`inference_psd_blockswap.py`](../inference/scripts/inference_psd_blockswap.py) 的对应入口同步接受可选回调。回调只传 CPU 整数，不读取张量或主动同步 CUDA；未提供回调时维持原调用方式。默认 `bf16` 不变；已有 blockswap 调用不提供 dtype 字段时仍沿用 bf16。没有修改 `inference_psd.py` 或 annotators。

**混合精度边界不能省略**：LayerDiff 已在 VAE 编码后把 conditioning 转回 UNet dtype；当前 Marigold 原实现没有这样做。fp32 VAE conditioning 与 fp16 latent 拼接会提升到 fp32，导致 fp16 UNet 报 dtype mismatch。本集成在同一推理文件内用 `_MixedDtypeMarigoldPipeline.single_infer()` 做一次入口转换；bf16/同 dtype 仍使用原类。此边界已通过 CPU 真实小型模型验证，以及 Kaggle T4×2 完整分层、深度与 PSD 导出验收。

**加载时也要选择 dtype**：T4×2 共享主机内存，不能让两个 worker 各自先构建完整 fp32 UNet/文本编码器再转 fp16。`_load_dtype_kwargs()` 在非 bf16 模式下按组件直接加载目标 dtype；VAE 从加载到推理均保留 fp32，不经过 fp16 舍入。默认 bf16 仍不向加载器传 dtype 参数，保留原加载路径。本地真实 checkpoint 冒烟已验证 fp32 VAE 权重逐项一致。

2026-10-26 的 Kaggle 双 worker 实测还发现：group offload 会耗尽共享 30 GiB 主机内存并触发 OS OOM kill；关闭后，1280 解码需要 expandable segments 避免 CUDA 分配碎片。因此 notebook 批处理默认 `BATCH_GROUP_OFFLOAD=False`，并将 `CUDA_ALLOCATOR_CONF` 传入子进程；PyTorch 2.11+ 使用新环境变量名，旧版本使用兼容名称。Web 每次请求在独立进程中使用分阶段 group offload；进程隔离本身不改变模型推理数学、精度放置、seed、采样或 attention。PSD 左右拆分的独立修复与限制见上节。

同步上游：

```bash
# remote 已存在时不用重复 add
 git remote add upstream https://github.com/shitagaki-lab/see-through.git
 git pull upstream main
```

新文件通常不与上游冲突；推理冲突面为上列 dtype 与采样回调文件，根 README 的入口节也可能遇到文档冲突。冲突时保留上游逻辑，逐项合并 dtype 边界与可选回调，不要用整文件覆盖上游版本。上游未来若新增同名集成文件，也应逐项合并。

## 每次同步检查清单

- [ ] `inference_psd_quantized.py` 的 `REPO_MAP['none']` 模型 ID 是否改变？同步 `inference_worker.py` 的模型参数。notebook 不硬编码模型 ID。Marigold 旧地址目前会重定向到 `layerdifforg/seethroughv0.0.1_marigold`，未来若失效需同步 worker。
- [ ] `build_layerdiff_pipeline` / `build_marigold_pipeline` / `run_layerdiff` / `run_marigold` / `further_extr` 签名是否改变？同步 worker 调用。Marigold `single_infer` 签名或内部 dtype 边界改变时，检查局部子类是否仍必要、是否仍正确。
- [ ] LayerDiff／Marigold 的 scheduler loop 或 blockswap override 是否改变？进度必须在真实 step 完成后发出，保留可选 `step_callback` 接口；worker 结构化消息必须与未换行的原生 stderr 进度条正确分帧，不能退回静态三阶段百分比。
- [ ] 保留非 bf16 模式的按组件 dtype 加载；不要把 pipeline 整体先转 fp16 再把 VAE 转回 fp32，这会丢失 VAE 权重精度，也不能降低首次 fp32 加载的共享内存峰值。
- [ ] [官方 Space app.py](https://huggingface.co/spaces/24yearsold/see-through-demo/blob/main/app.py) 是否有值得复用的 UI 改进？[Space README](https://huggingface.co/spaces/24yearsold/see-through-demo/blob/main/README.md) 的 `sdk_version` 是否改变？同步 `requirements-kaggle.txt` 的 Gradio pin；当前为 `6.12.0`。
- [ ] 根 `requirements.txt` 的 transformers / diffusers / accelerate 等 pin 是否更新？同步精简依赖，避免引入 Qt、annotators 或训练包。
- [ ] 对上列推理文件和三个 Web／worker 文件运行 `python -m py_compile`，在装好精简依赖后实际运行推理 CLI `--help` 和本地 Gradio 页面。
- [ ] 运行 `python -m unittest discover -s inference/tests -v` 检查 OS 进程生命周期；它不需要模型依赖，不替代真实 GPU 推理或公网验收。
- [ ] 设 `RUN_BATCH=True`、`ENABLE_WEBUI=False` 后，在 Kaggle 重新运行 Cell 1–7；Cell 6 验收每图 PSD 分层、stats dtype/显存/耗时、非透明/非黑分解图；Cell 7 验收归档包含每张图的 PSD。正常日志应有 `LayerDiff3D done`、`Marigold done`、`PSD assembly done`。
- [ ] 从非 Kaggle 浏览器访问两种隧道之一，上传测试图，确认身体／头部／深度步数在推理返回前更新，gallery 非空且能下载分层 PSD。更换 Gradio pin 时也检查 gallery 与 PSD 在临时目录清理后仍可读取。
- [ ] 新架构实际验收连续 1536／2048 请求、CUDA OOM 后新 worker 降分辨率恢复、输出缓存仍可下载、请求超时、SIGUSR1 排空、SIGTERM 停止及父进程强杀后无孤儿 GPU worker；记录资源与退出结果，不能把启动成功算作 GPU 稳定性通过。

## 回退与平台限制

- 数值异常：Cell 2 改 `DTYPE='fp32'`、`RESOLUTION=1024`，并建议设 `NUM_WORKERS=1`，Web 同时降 `WEB_RESOLUTION`。仍 OOM 时把 `RESOLUTION_DEPTH` 降到 512。不要改成 fp16 VAE，它更容易溢出。fp32 回退的完整 T4 实跑尚未验证。
- 连续服务：历史长驻服务的 `KEEP_RESIDENT=True`／False 均未通过稳定性实测，该配置已删除。当前每请求进程隔离使成功、失败或超时的推理进程退出；OOM 后下一次请求使用新 worker，而非污染进程中的热重试。新架构有限序列验收通过，不表示单次峰值、任意输入质量或 PSD alpha／预览毛边已解决，详见[验证记录](VALIDATION.md)。
- 依赖报错：确认错误来自 Cell 4 的独立环境，而不是基础 kernel 的旧安装日志；不要通过升级/降级系统包解决。运行时版本由 Cell 2 的 `TORCH_PACKAGES` 和精简 requirements 决定，修改后重跑 Cell 4。2026-10-27 的真实 Kaggle 验收已确认隔离环境 `pip check` 通过、CUDA 可用、基础环境包清单未变化。
- 公网隧道：默认 Gradio `share=True` 使用出站 FRP；Cell 9 默认另开 cloudflared 结果找回入口。Gradio 断线不等于 worker 停止，不要重复点 Run；复制请求 ID 到备用页查询。备用创建失败会明确报错，并继续原 Gradio 的有界保活，不重启模型。Quick Tunnel 不支持 SSE，不能保证实时进度连接；两种入口都不可用时只能在 Kaggle 保存／下载工作目录中的已完成结果，或在当前请求结束后更换部署入口。Kaggle 容器不能从外部直连 7860。
- 当前为无鉴权公网服务，请勿上传敏感或未经授权的内容；Web 页的自有安全提示与进度文字统一为英文。`queue(max_size=16)` 只限制排队，不能防止滥用。Gradio 缓存按小时清理；独立完成结果仍占用 Kaggle 工作盘，请及时下载并管理保存版本。
- 服务随 session 结束；常见限制为 12h 上限、交互约 20 分钟空闲断开、免费 GPU 30h/周，以 [Kaggle 当前文档与账号额度](https://www.kaggle.com/docs/notebooks) 为准。外部访客请求不保证 session 保活。
- 2026-10-26 实测环境为 Python 3.13、torch 2.11.0+cu128、Tesla T4×2：1280 / 30 steps，两图并行各生成 17 层 PSD；总耗时约 16.9 / 18.6 分钟，peak VRAM 均为 13.01 GiB，共享内存峰值约 29.04 GiB，OOM 计数 0。采用直接 fp16 加载、fp32 VAE、批处理关闭 group offload、expandable segments。这只是两张样例，不保证所有输入的耗时/峰值。
- 首次模型下载约 8GB [INFERENCE]；`/kaggle/working` 保存空间约 20GB，留意模型缓存和 PSD 占用。长期在线服务应使用已有官方 HF Space，不依赖临时 Kaggle session。

## CLI 公网服务生命周期

CLI / Save & Run All 会在 notebook 执行结束后释放 session，因此 Cell 9 除备用隧道外，也负责有上限的公网服务保活。默认 `WEB_SESSION_SECONDS=10800`（3h），避免旧 1h 默认在历史约 66.7 分钟的 2K 请求中途关闭；每次推理的 `WEB_REQUEST_TIMEOUT=5400` 映射为 `SEETHROUGH_REQUEST_TIMEOUT`。交互 Gradio 可只运行 Cell 8；需要默认备用入口或显式 cloudflared 时再运行 Cell 9。时长只改 Cell 2，仍受 Kaggle session 上限与 GPU 配额约束；设为 0 跳过保活。不要把分享链接有效期误当成 Kaggle 在线时长，也不要为新增备用入口而中断正在运行的保活单元。

正常保活到期时，Cell 9 发送 **SIGUSR1**：app 停止接收新工作，当前请求沿用已经开始的请求超时；完成 Gradio 输出交接后，再保留下载端点 `WEB_DRAIN_GRACE=60` 秒。notebook 保留隧道、持续转发日志，最多再等 `WEB_REQUEST_TIMEOUT + WEB_DRAIN_GRACE + PROCESS_STOP_TIMEOUT`（默认 5475 秒），然后必要时进入强制清理。这个等待不会重置当前请求计时，也不承诺完成队列中尚未开始的请求；下载宽限也不保证大文件一定传完。

手动中断 Cell 9、重跑 Cell 8 或排空超时使用 **SIGTERM → 等待 `PROCESS_STOP_TIMEOUT=15` 秒 → 必要时 SIGKILL**。先关闭 app，再关闭 cloudflared。app 负责停止并回收当前 worker，默认 `SEETHROUGH_WORKER_STOP_TIMEOUT=5` 秒；强杀父 app 时，其持有的 stdin 管道关闭，worker 的 EOF 守卫终止自身进程组，避免正常 GPU 工作遗留为孤儿。强制停止不同于正常排空，会中止当前请求；Kaggle 平台强制回收也不保证完成请求。

## 公网实跑验收

2026-10-28 已在独立 Kaggle CPU 实例验证当前默认 Web 分支：实际执行 Cell 2、3、5–9，跳过批处理、校验和打包，生成 Gradio 公网链接，并从非 Kaggle 浏览器打开页面。此项只验证公网启动，使用最小 Web 依赖；不重跑模型、不消耗额外 GPU 配额，也不代表完整 CUDA 安装耗时或 cloudflared／fp32 验收。根因、原始日志和截图见[验证记录](VALIDATION.md)。

以下均为**旧长驻服务的历史验收**，保留当时配置，不是当前逐请求隔离架构的验收结果。

2026-10-26 已从非 Kaggle 浏览器访问真实 Kaggle T4 上的 Gradio share 链接，上传 `test_image.png`，以 1024 / 30 steps / seed 42 完成一次请求：gallery 24 项，下载并解析了 1024×1024、17 层 PSD，首次请求总耗时 1108.9 秒。当时使用现已删除的 `KEEP_RESIDENT=True`、单进程 group offload 与 expandable segments。cloudflared 备用链路及 fp32 回退仍未做完整远端实跑，不能把 Gradio 路径的验收结果外推到它们。


2026-10-27 已验证新隔离依赖环境（不是请求进程隔离）：Python 3.13、torch **2.8.0+cu128**、CUDA 12.8、Gradio 6.12.0；`pip check` 通过，基础 kernel 的包清单未变化。从公网浏览器上传同一测试图，以 **1536 / 10 steps / seed 42**、当时的 `KEEP_RESIDENT=False`、单进程 group offload 完成请求：gallery **24 项**，下载并解析了 **1536×1536、17 层 PSD**，实际渲染非黑且非透明，服务端首次请求耗时 **937.9 秒**。这证明没有在后端降回 1280；该次公网验收不覆盖其他输入或双 worker 高分辨率。后续 1536／2048 的质量诊断及正式服务稳定性失败另见[推理验证记录](VALIDATION.md)，不能把这些实验合并宣称连续 Web 服务通过。
