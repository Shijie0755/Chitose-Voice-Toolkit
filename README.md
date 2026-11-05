<div align="center">

# 锦木千束 Chitose Voice Toolkit

**从双语字幕到 GPT-SoVITS 角色语音模型：可追溯的数据构建、人工分类、验证与本地生成工具**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-required-007808?logo=ffmpeg&logoColor=white)
![GPT--SoVITS](https://img.shields.io/badge/GPT--SoVITS-v2-CB3A31)
![Platform](https://img.shields.io/badge/primary%20platform-Windows-0078D4?logo=windows)
![Release](https://img.shields.io/badge/model-v0.1.0-D97706)

<sub>Unofficial fan research project · 本地运行 · 不包含动画、字幕全文或训练音频</sub>

</div>

---

## 项目是什么

这个仓库记录“锦木千束 Chitose”角色语音项目从素材整理到本地推理的完整工具链，而不是只放一个模型文件。

它解决的核心问题是：如何从没有可靠说话人标签的中日双语 ASS 字幕出发，定位候选台词，从动画日语主音轨中精准切出 WAV，再经过人工试听、轻度降噪、格式验证和 GPT-SoVITS 训练，最终得到可在本地网页中调用的角色语音模型。

> 仓库展示名统一使用 **锦木千束 Chitose**。部分旧脚本、列表字段与权重文件仍保留内部标识 **Chisato**，以兼容已经完成的训练流程。

## 工作流

~~~mermaid
flowchart LR
    A[MKV + 中日双语 ASS] --> B[解析字幕与配对]
    B --> C[推断千束候选台词]
    C --> D[ffprobe 确认日语主音轨]
    D --> E[FFmpeg 切片与格式统一]
    E --> F[人工分类与试听]
    F --> G[保守降噪与清单重建]
    G --> H[自动质量验证]
    H --> I[GPT-SoVITS 训练]
    I --> J[Chitose Voice Lab]
~~~

## 当前公开内容

| 模块 | 用途 | 状态 |
|---|---|---|
| <code>pipeline/</code> | ASS 解析、候选选择、FFmpeg 切片、说话人相似度辅助排序、阶段验证与降噪准备 | 已公开 |
| <code>tools/folder_drop_router/</code> | 将试听文件拖到不同卡片后自动移动到对应分类目录 | 已公开 |
| <code>tools/voice_generator/</code> | 调用 GPT-SoVITS 权重的本地网页生成器 | 已公开 |
| <code>training/</code> | 训练配置生成与推理冒烟测试 | 已公开 |
| <code>models/</code> | v0.1.0 模型说明、文件名与校验值 | 已公开 |
| 训练音频与当前人工分类结果 | 仍在整理 | **暂不公开** |

## v0.1.0 模型

第一版模型使用 GPT-SoVITS v2 流程训练，训练清单来自 EP01–EP02：

- 154 个保留片段
- 总时长 580.332 秒（约 9.67 分钟）
- mono / 32 kHz / signed 16-bit PCM
- 使用保守 FFmpeg 稳态降噪：<code>afftdn=nr=6:nf=-48:tn=1:gs=5</code>

模型通过 GitHub **Release v0.1.0** 分发：

- <code>Chitose-v0.1.0-SoVITS.pth</code>
- <code>Chitose-v0.1.0-GPT.ckpt</code>

两份权重需要配合使用。详细校验值见 [models/README.md](models/README.md)。

## 快速开始：语音生成器

### 1. 准备环境

先按照上游项目说明安装 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)，并准备其预训练模型。

~~~powershell
python -m pip install -r requirements-tools.txt
~~~

从 Release 下载 v0.1.0 的 <code>.pth</code> 和 <code>.ckpt</code>，复制
<code>training/chitose_v0.1.0_inference.example.yaml</code> 并将其中路径改为你电脑上的真实路径。

### 2. 设置并启动

编辑 [start_chitose_webui.bat](tools/voice_generator/start_chitose_webui.bat) 顶部三个路径，然后双击运行。默认地址：

<code>http://127.0.0.1:7860</code>

公开仓库不附带版权音频，因此首次使用请上传一段你有权使用的 3–12 秒单人参考音频，并准确填写参考文本。

## 数据管线

安装 FFmpeg，并设置三个环境变量：

~~~powershell
$env:LYCORIS_SOURCE_ROOT = "D:\path\to\your\mkv-and-ass"
$env:GPT_SOVITS_ROOT = "D:\path\to\GPT-SoVITS"
$env:FFMPEG_PATH = "ffmpeg"
$env:FFPROBE_PATH = "ffprobe"
~~~

解析字幕：

~~~powershell
python pipeline\parse_lycoris_episodes.py 3 4 5 --source-root $env:LYCORIS_SOURCE_ROOT
~~~

按照 selection 配置切出候选 WAV：

~~~powershell
python pipeline\build_lycoris_chisato_batch.py 3 4 5
~~~

完成阶段 A 后进行自动验证：

~~~powershell
python pipeline\validate_lycoris_chisato_stage_a.py
~~~

候选说话人标签来自字幕内容、上下文和声音相似度的综合推断，并非 ground truth。必须人工试听，不能把所有候选直接投入训练。完整说明见 [docs/pipeline.md](docs/pipeline.md)。

## 人工分类

当前实际采用四类：

1. **能用（声音正常）**
2. **能用（声音较高）**
3. **暂留（背景音或其他）**
4. **不留（背景音很重、非千束声音或其他原因）**

双击运行：

~~~powershell
python tools\folder_drop_router\folder_drop_router.pyw
~~~

为每张卡片绑定一个目标目录，把 WAV 从桌面或资源管理器拖到卡片即可完成移动。当前人工分类结果尚未完成，因此本次不上传音频或最终分类清单。参见 [docs/classification-guide.md](docs/classification-guide.md)。

## 仓库结构

~~~text
.
├─ pipeline/                    # 字幕、切片、候选、验证、降噪
│  └─ configs/                 # EP03–EP13 候选选择配置
├─ tools/
│  ├─ folder_drop_router/      # 拖放式手动分类器
│  └─ voice_generator/         # 本地 Chitose Voice Lab
├─ training/                    # 训练/推理配置辅助脚本
├─ models/                      # v0.1.0 模型卡与 SHA-256
└─ docs/                        # 流程、分类与发布说明
~~~

## 数据与版权边界

仓库不会上传：

- 动画 MKV、字幕全文和直接提取的对白音频；
- 尚未完成的人工分类结果；
- 个人电脑绝对路径、缓存、虚拟环境或 GPT-SoVITS 整个上游仓库；
- 未经确认的最终训练清单。

本项目仅用于研究、学习和非误导性创作。请勿用生成音频冒充配音演员、欺骗他人或暗示官方授权。更多说明见 [NOTICE.md](NOTICE.md)。

## Roadmap

- [x] EP01–EP02 v0.1.0 训练与本地推理
- [x] EP03–EP13 字幕候选批处理与自动验证工具
- [x] 拖放式人工分类器
- [x] 本地网页语音生成器
- [ ] 完成四类人工试听整理
- [ ] 仅从最终保留音频重建清单
- [ ] 发布下一版全季模型与可复现训练摘要

