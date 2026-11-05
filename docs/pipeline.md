# 数据处理流程

## 1. 输入

管线预期每集包含一个 MKV 和一个中日双语 ASS。日语行通常使用 <code>TextJP</code>，中文行通常使用 <code>TextCHS</code>。中文只用于理解语境；GPT-SoVITS 的训练文本必须保留日文原文。

源文件名不需要完全一致。脚本按集数枚举实际文件，并要求每集恰好匹配一个 MKV 和一个 ASS。

## 2. 字幕解析

<code>lycoris_dataset_core.py</code>：

- 清理 ASS override tags 与换行标记；
- 将同时间范围的 TextJP / TextCHS 配对；
- 保存时间、样式、Name、日文与中文语境；
- 不把 ASS Name 字段视为可靠说话人真值。

## 3. 候选选择

<code>configs/chisato_selections_ep03_ep13.json</code> 将候选字幕 ID 分为：

- <code>high_ids</code>：高置信度千束台词；
- <code>probable_ids</code>：说话人、重叠、时长、音效或剧情上下文仍需复核。

这些标签是推断结果，不是 ground truth。

## 4. FFmpeg 切片

<code>build_lycoris_chisato_batch.py</code> 会先用 ffprobe 枚举音轨，并优先要求唯一的日语 <code>Main Audio</code>。切片输出统一为 WAV、mono、32000 Hz、signed 16-bit PCM。

相邻候选可以在间隔不超过 0.65 秒且合并后不超过约 10 秒时合并。默认 padding 为前后各 0.15 秒，并避免跨入相邻字幕。

## 5. 人工节点

阶段 A 只生成候选。试听者必须删除错误说话人、明显重叠、截断、强音乐或其他不适合训练的文件。不要绕过这个节点。

## 6. 降噪

<code>prepare_selected_and_denoise.py</code> 只对人工保留文件执行轻度稳态降噪：

~~~text
afftdn=nr=6:nf=-48:tn=1:gs=5
~~~

它不是人声/BGM 分离器，无法可靠修复强音乐、枪声、混响或多人重叠。无法安全修复的片段应排除。

## 7. 验证

<code>validate_lycoris_chisato_stage_a.py</code> 检查：

- CSV / list 指向的 WAV 存在且非空；
- mono、32 kHz、16-bit PCM；
- 路径与 clip ID 不重复；
- WAV 与切分时间差小于约 0.05 秒；
- 字幕 ID、日文、中文语境和时间区间可追溯；
- 每集统计与 summary 一致；
- 固定抽样波形边缘、响度、零值和削波指标。

脚本退出码为 0 只是结构验证通过，不能代替人工试听。

