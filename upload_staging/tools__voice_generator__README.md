# Chitose Voice Lab

基于 FastAPI 的本地 GPT-SoVITS 网页生成器，支持日文、中文、英文、参考音频上传、语速与采样参数调整，并将结果保存为 WAV。

公开版本不附带默认参考音频。请使用你有权处理的 3–12 秒干净单人参考语音，并填写准确参考文本。

启动前设置：

- <code>GPT_SOVITS_ROOT</code>
- <code>CHITOSE_CONFIG_PATH</code>
- 可选：<code>CHITOSE_OUTPUT_DIR</code>
- 可选本地默认参考：<code>CHITOSE_REFERENCE_AUDIO</code> 和 <code>CHITOSE_REFERENCE_TEXT</code>

Windows 用户可以编辑并运行 <code>start_chitose_webui.bat</code>。

