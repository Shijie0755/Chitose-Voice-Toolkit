from __future__ import annotations

import os
import random
import shutil
import sys
import threading
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

GPT_SOVITS_ROOT_TEXT = os.environ.get("GPT_SOVITS_ROOT", "")
if GPT_SOVITS_ROOT_TEXT:
    GPT_SOVITS_ROOT = Path(GPT_SOVITS_ROOT_TEXT).expanduser().resolve()
    sys.path.insert(0, str(GPT_SOVITS_ROOT))
    sys.path.insert(0, str(GPT_SOVITS_ROOT / "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config


TOOL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(
    os.environ.get("CHITOSE_CONFIG_PATH", TOOL_DIR / "chitose_v0.1.0_inference.yaml")
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get("CHITOSE_OUTPUT_DIR", TOOL_DIR / "outputs")
).expanduser()
UPLOAD_DIR = OUTPUT_DIR / "_uploads"
_reference_value = os.environ.get("CHITOSE_REFERENCE_AUDIO", "").strip()
DEFAULT_REFERENCE_PATH = Path(_reference_value).expanduser() if _reference_value else None
DEFAULT_PROMPT_TEXT = os.environ.get("CHITOSE_REFERENCE_TEXT", "").strip()
HAS_DEFAULT_REFERENCE = bool(
    DEFAULT_REFERENCE_PATH
    and DEFAULT_REFERENCE_PATH.is_file()
    and DEFAULT_PROMPT_TEXT
)

LANGUAGE_MAP = {"ja": "all_ja", "zh": "all_zh", "en": "en", "auto": "auto"}
REFERENCE_LANGUAGE_MAP = {"ja": "all_ja", "zh": "all_zh", "en": "en"}

_pipeline: TTS | None = None
_load_lock = threading.Lock()
_generation_lock = threading.Lock()


def get_pipeline() -> TTS:
    global _pipeline
    if _pipeline is None:
        with _load_lock:
            if _pipeline is None:
                if not CONFIG_PATH.exists():
                    raise FileNotFoundError(f"Inference config not found: {CONFIG_PATH}")
                _pipeline = TTS(TTS_Config(str(CONFIG_PATH)))
    return _pipeline


def audio_duration(path: str | Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception as exc:
        raise ValueError(f"无法读取参考音频：{exc}") from exc


def make_error(message: str, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=message)


app = FastAPI(title="Chitose Voice Lab", docs_url=None, redoc_url=None)


PAGE_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chitose Voice Lab</title>
  <style>
    :root { --paper:#f4eee5; --surface:rgba(255,255,255,.76); --ink:#29231f; --muted:#756b61; --line:rgba(41,35,31,.14); --red:#ce4035; --red-dark:#a72e2a; --green:#238655; --shadow:0 18px 50px rgba(70,49,28,.08); }
    * { box-sizing:border-box; }
    html { color-scheme:light; }
    body { margin:0; color:var(--ink); font-family:Inter,"Noto Sans SC","Microsoft YaHei",system-ui,sans-serif; background:radial-gradient(circle at 10% -5%,rgba(212,166,61,.24),transparent 34rem),radial-gradient(circle at 95% 12%,rgba(206,64,53,.10),transparent 30rem),linear-gradient(180deg,#faf7f0 0%,var(--paper) 100%); min-height:100vh; }
    button,input,textarea,select { font:inherit; }
    .shell { width:min(1180px,calc(100% - 32px)); margin:0 auto; padding:22px 0 48px; }
    .hero { position:relative; overflow:hidden; padding:34px 36px; border:1px solid var(--line); border-radius:26px; background:var(--surface); box-shadow:var(--shadow); }
    .hero::after { content:"千束"; position:absolute; right:28px; top:-24px; color:rgba(206,64,53,.055); font:900 170px/1 serif; letter-spacing:-.1em; pointer-events:none; }
    .kicker { color:var(--red); font:800 12px/1.2 ui-monospace,Consolas,monospace; letter-spacing:.17em; }
    h1 { margin:9px 0 7px; font-size:clamp(34px,6vw,58px); line-height:1; letter-spacing:-.05em; }
    .hero-copy { margin:0; max-width:700px; color:var(--muted); line-height:1.7; }
    .chip { display:inline-flex; align-items:center; gap:9px; margin-top:20px; padding:8px 12px; border:1px solid rgba(35,134,85,.24); border-radius:999px; color:var(--green); background:rgba(234,249,240,.85); font-size:12px; font-weight:750; }
    .chip::before { content:""; width:8px; height:8px; border-radius:50%; background:#2bac6d; box-shadow:0 0 0 4px rgba(43,172,109,.13); }
    .grid { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(320px,.85fr); gap:18px; margin-top:18px; }
    .card { padding:23px; border:1px solid var(--line); border-radius:20px; background:var(--surface); box-shadow:0 12px 34px rgba(70,49,28,.05); }
    .step { margin:0 0 17px; font-size:18px; letter-spacing:-.02em; }
    .step small { color:var(--red); font:800 11px ui-monospace,Consolas,monospace; letter-spacing:.1em; }
    label { display:block; margin:14px 0 7px; font-size:13px; font-weight:750; }
    textarea,select,input[type="number"],input[type="file"] { width:100%; border:1px solid var(--line); border-radius:12px; color:var(--ink); background:rgba(255,255,255,.76); outline:none; }
    textarea { min-height:184px; padding:14px 15px; line-height:1.65; resize:vertical; }
    select,input[type="number"],input[type="file"] { min-height:44px; padding:9px 11px; }
    textarea:focus,select:focus,input:focus { border-color:rgba(206,64,53,.65); box-shadow:0 0 0 3px rgba(206,64,53,.10); }
    .examples { display:flex; flex-wrap:wrap; gap:8px; margin-top:11px; }
    .soft-btn,.mode-btn { border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.66); color:#5f554c; cursor:pointer; padding:8px 11px; font-size:12px; font-weight:700; }
    .soft-btn:hover,.mode-btn:hover { border-color:rgba(206,64,53,.35); color:var(--red); }
    .modes { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .mode-btn { border-radius:11px; padding:11px; }
    .mode-btn.active { color:white; border-color:var(--red); background:var(--red); }
    .hint { margin:8px 0 0; color:var(--muted); font-size:12px; line-height:1.6; }
    audio { width:100%; margin-top:10px; }
    .custom-ref { display:none; }
    details { margin-top:18px; border:1px solid var(--line); border-radius:17px; background:rgba(255,255,255,.5); }
    summary { cursor:pointer; padding:16px 18px; font-weight:800; }
    .advanced { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; padding:2px 18px 18px; }
    .range-line { display:flex; justify-content:space-between; align-items:center; font-size:12px; }
    input[type="range"] { width:100%; accent-color:var(--red); }
    .generate { width:100%; min-height:58px; margin-top:18px; border:0; border-radius:15px; color:white; background:linear-gradient(135deg,var(--red),var(--red-dark)); cursor:pointer; font-weight:900; letter-spacing:.04em; box-shadow:0 12px 28px rgba(206,64,53,.23); transition:.16s ease; }
    .generate:hover { transform:translateY(-1px); filter:brightness(1.04); }
    .generate:disabled { cursor:wait; opacity:.66; transform:none; }
    .result { margin-top:18px; display:grid; grid-template-columns:1.4fr .6fr; gap:18px; }
    .output-empty { min-height:92px; display:grid; place-items:center; border:1px dashed rgba(41,35,31,.22); border-radius:14px; color:var(--muted); text-align:center; }
    #outputPlayer,#downloadLink { display:none; }
    .download { display:inline-block; margin-top:13px; color:var(--red); font-size:13px; font-weight:800; text-decoration:none; }
    .status { min-height:105px; padding:15px; border-radius:13px; background:rgba(255,255,255,.57); color:var(--muted); line-height:1.65; font-size:13px; white-space:pre-line; }
    .status.ok { color:#216e49; background:rgba(233,248,239,.78); }
    .status.error { color:#9a2f2a; background:rgba(255,234,231,.8); }
    .spinner { display:inline-block; width:12px; height:12px; margin-right:8px; border:2px solid rgba(255,255,255,.35); border-top-color:white; border-radius:50%; animation:spin .8s linear infinite; vertical-align:-2px; }
    @keyframes spin { to { transform:rotate(360deg); } }
    footer { margin-top:18px; color:#887e74; font-size:12px; line-height:1.7; text-align:center; }
    @media(max-width:820px) { .grid,.result { grid-template-columns:1fr; } .advanced { grid-template-columns:1fr 1fr; } .hero { padding:28px 24px; } .hero::after { display:none; } }
    @media(max-width:520px) { .shell { width:min(100% - 20px,1180px); } .advanced { grid-template-columns:1fr; } .card { padding:18px; } }
  </style>
</head>
<body>
<main class="shell">
  <header class="hero">
    <div class="kicker">LOCAL AI VOICE · GPT-SoVITS V2</div>
    <h1>锦木千束 Chitose Voice Lab</h1>
    <p class="hero-copy">本地千束语音生成工作台。输入文字，使用参考音频控制说话状态，然后交给 RTX 3060 完成推理。</p>
    <div class="chip">模型已绑定 · 本机运行 · 文件不会上传</div>
  </header>

  <section class="grid">
    <article class="card">
      <h2 class="step"><small>STEP 01</small><br>输入要说的内容</h2>
      <label for="text">生成文本</label>
      <textarea id="text">おはよう！今日はとってもいい天気だね。さあ、一緒に出かけよう！</textarea>
      <label for="language">文本语言</label>
      <select id="language"><option value="ja">日本語</option><option value="zh">中文</option><option value="en">English</option><option value="auto">自动识别</option></select>
      <div class="examples">
        <button class="soft-btn" data-text="おはよう！今日も元気に頑張ろうね！" data-lang="ja">元気な日本語</button>
        <button class="soft-btn" data-text="今日は何をしようか。ゆっくり考えてみよう。" data-lang="ja">自然な日本語</button>
        <button class="soft-btn" data-text="早上好！今天天气不错，我们一起出去走走吧。" data-lang="zh">中文测试</button>
      </div>
    </article>

    <aside class="card">
      <h2 class="step"><small>STEP 02</small><br>选择参考声音</h2>
      <div class="modes"><button id="builtinMode" class="mode-btn active" type="button">使用本地默认参考</button><button id="uploadMode" class="mode-btn" type="button">上传参考音频</button></div>
      <div id="builtinRef">
        <label>本地默认参考（不会随公开仓库分发）</label>
        <audio controls preload="metadata" src="/api/reference"></audio>
        <p class="hint">通过环境变量配置你有权使用的 3–12 秒干净单人语音。</p>
      </div>
      <div id="customRef" class="custom-ref">
        <label for="referenceFile">参考音频（3–12 秒）</label><input id="referenceFile" type="file" accept="audio/*">
        <label for="promptText">参考音频原文</label><textarea id="promptText" style="min-height:92px" placeholder="准确填写音频中实际说出的内容"></textarea>
        <label for="promptLanguage">参考音频语言</label><select id="promptLanguage"><option value="ja">日本語</option><option value="zh">中文</option><option value="en">English</option></select>
      </div>
    </aside>
  </section>

  <details>
    <summary>高级参数（第一次使用保持默认即可）</summary>
    <div class="advanced">
      <div><div class="range-line"><label for="speed">语速</label><b id="speedValue">1.00</b></div><input id="speed" type="range" min="0.6" max="1.4" step="0.05" value="1"></div>
      <div><label for="seed">随机种子（-1 为随机）</label><input id="seed" type="number" value="2333" step="1"></div>
      <div><div class="range-line"><label for="topK">Top K</label><b id="topKValue">15</b></div><input id="topK" type="range" min="1" max="100" step="1" value="15"></div>
      <div><div class="range-line"><label for="topP">Top P</label><b id="topPValue">1.00</b></div><input id="topP" type="range" min="0.1" max="1" step="0.05" value="1"></div>
      <div><div class="range-line"><label for="temperature">Temperature</label><b id="temperatureValue">1.00</b></div><input id="temperature" type="range" min="0.1" max="1.5" step="0.05" value="1"></div>
      <div><div class="range-line"><label for="penalty">重复惩罚</label><b id="penaltyValue">1.35</b></div><input id="penalty" type="range" min="1" max="2" step="0.05" value="1.35"></div>
    </div>
  </details>

  <button id="generate" class="generate">生成千束语音</button>
  <section class="result">
    <article class="card"><h2 class="step"><small>STEP 03</small><br>试听与下载</h2><div id="empty" class="output-empty">生成结果会出现在这里</div><audio id="outputPlayer" controls></audio><a id="downloadLink" class="download" download="chisato_generated.wav">↓ 下载 WAV 文件</a></article>
    <aside class="card"><h2 class="step"><small>STATUS</small><br>运行状态</h2><div id="status" class="status">等待生成。首次生成需要加载模型，通常会比之后慢。</div></aside>
  </section>
  <footer>参考音频会显著影响语气和节奏。请勿将生成内容用于冒充真人、误导他人或侵犯配音演员权益。关闭启动窗口即可停止服务并释放显存。</footer>
</main>
<script>
  const $ = id => document.getElementById(id);
  const hasBuiltin = __HAS_BUILTIN__;
  let referenceMode = hasBuiltin ? 'builtin' : 'upload';
  let currentUrl = null;
  function setReferenceMode(mode) { referenceMode=mode; $('builtinMode').classList.toggle('active',mode==='builtin'); $('uploadMode').classList.toggle('active',mode==='upload'); $('builtinRef').style.display=mode==='builtin'?'block':'none'; $('customRef').style.display=mode==='upload'?'block':'none'; }
  $('builtinMode').onclick=()=>setReferenceMode('builtin'); $('uploadMode').onclick=()=>setReferenceMode('upload');
  if(!hasBuiltin){ $('builtinMode').disabled=true; setReferenceMode('upload'); }
  document.querySelectorAll('.soft-btn').forEach(btn=>btn.onclick=()=>{ $('text').value=btn.dataset.text; $('language').value=btn.dataset.lang; });
  [['speed','speedValue',2],['topK','topKValue',0],['topP','topPValue',2],['temperature','temperatureValue',2],['penalty','penaltyValue',2]].forEach(([input,out,digits])=>{ $(input).oninput=()=>$(out).textContent=Number($(input).value).toFixed(digits); });
  function showStatus(message,kind='') { $('status').textContent=message; $('status').className='status '+kind; }
  $('generate').onclick=async()=>{
    const button=$('generate'); const form=new FormData();
    form.append('text',$('text').value.trim()); form.append('text_language',$('language').value); form.append('reference_mode',referenceMode); form.append('prompt_text',$('promptText').value.trim()); form.append('prompt_language',$('promptLanguage').value); form.append('speed',$('speed').value); form.append('seed',$('seed').value); form.append('top_k',$('topK').value); form.append('top_p',$('topP').value); form.append('temperature',$('temperature').value); form.append('repetition_penalty',$('penalty').value);
    if(referenceMode==='upload'&&$('referenceFile').files[0]) form.append('reference_file',$('referenceFile').files[0]);
    button.disabled=true; button.innerHTML='<span class="spinner"></span>正在生成，请保持窗口开启…'; showStatus('正在加载模型并生成音频。\n首次运行可能需要几十秒。');
    try {
      const response=await fetch('/api/generate',{method:'POST',body:form});
      if(!response.ok){ let message='生成失败'; try{message=(await response.json()).detail||message;}catch(_){} throw new Error(message); }
      const blob=await response.blob(); if(currentUrl) URL.revokeObjectURL(currentUrl); currentUrl=URL.createObjectURL(blob); $('outputPlayer').src=currentUrl; $('outputPlayer').style.display='block'; $('downloadLink').href=currentUrl; $('downloadLink').style.display='inline-block'; $('empty').style.display='none';
      showStatus(`✓ 生成完成\n时长：${response.headers.get('X-Audio-Duration')} 秒\nSeed：${response.headers.get('X-Seed')}\n文件：${response.headers.get('X-Output-Name')||''}`,'ok');
    } catch(error) { showStatus('✕ '+error.message,'error'); }
    finally { button.disabled=false; button.textContent='生成千束语音'; }
  };
</script>
</body>
</html>'''

PAGE_HTML = PAGE_HTML.replace("__HAS_BUILTIN__", "true" if HAS_DEFAULT_REFERENCE else "false")


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(PAGE_HTML)


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": CONFIG_PATH.exists(), "model_loaded": _pipeline is not None, "config": str(CONFIG_PATH), "default_reference": HAS_DEFAULT_REFERENCE})


@app.get("/api/reference")
def reference_audio() -> FileResponse:
    if not HAS_DEFAULT_REFERENCE or DEFAULT_REFERENCE_PATH is None:
        raise make_error("公开项目不附带参考音频，请上传你有权使用的参考音频。", 404)
    return FileResponse(DEFAULT_REFERENCE_PATH, media_type="audio/wav")


@app.post("/api/generate")
def generate(
    text: str = Form(...), text_language: str = Form("ja"), reference_mode: str = Form("builtin"), prompt_text: str = Form(""), prompt_language: str = Form("ja"),
    speed: float = Form(1.0), seed: int = Form(2333), top_k: int = Form(15), top_p: float = Form(1.0), temperature: float = Form(1.0), repetition_penalty: float = Form(1.35),
    reference_file: UploadFile | None = File(None),
) -> FileResponse:
    text = text.strip()
    if not text: raise make_error("请先输入要生成的文本。")
    if len(text) > 500: raise make_error("单次文本请控制在 500 个字符以内；长文本建议分段生成。")
    if text_language not in LANGUAGE_MAP: raise make_error("不支持的文本语言。")
    if not 0.6 <= speed <= 1.4: raise make_error("语速必须在 0.6–1.4 之间。")
    if not 1 <= top_k <= 100 or not 0.1 <= top_p <= 1.0: raise make_error("Top K 或 Top P 超出允许范围。")
    if not 0.1 <= temperature <= 1.5 or not 1.0 <= repetition_penalty <= 2.0: raise make_error("Temperature 或重复惩罚超出允许范围。")

    uploaded_path: Path | None = None
    if reference_mode == "upload":
        if reference_file is None or not reference_file.filename: raise make_error("请上传一段参考音频。")
        prompt_text = prompt_text.strip()
        if not prompt_text: raise make_error("使用自定义参考音频时，需要填写其中实际说出的原文。")
        if prompt_language not in REFERENCE_LANGUAGE_MAP: raise make_error("不支持的参考音频语言。")
        suffix = Path(reference_file.filename).suffix.lower() or ".wav"
        if suffix not in {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}: raise make_error("参考音频格式应为 WAV、FLAC、OGG、MP3、M4A 或 AAC。")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        uploaded_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        with uploaded_path.open("wb") as target: shutil.copyfileobj(reference_file.file, target)
        reference_path = str(uploaded_path)
        prompt_lang = REFERENCE_LANGUAGE_MAP[prompt_language]
    else:
        if not HAS_DEFAULT_REFERENCE or DEFAULT_REFERENCE_PATH is None:
            raise make_error("未配置本地默认参考音频，请切换到上传参考音频。")
        reference_path = str(DEFAULT_REFERENCE_PATH)
        prompt_text = DEFAULT_PROMPT_TEXT
        prompt_lang = "all_ja"

    try:
        duration = audio_duration(reference_path)
        if duration < 3 or duration > 12: raise make_error(f"参考音频时长为 {duration:.1f} 秒；请使用 3–12 秒的干净单人语音。")
        actual_seed = seed if seed >= 0 else random.randint(0, 2_147_483_647)
        request = {"text": text, "text_lang": LANGUAGE_MAP[text_language], "ref_audio_path": reference_path, "prompt_text": prompt_text, "prompt_lang": prompt_lang, "top_k": top_k, "top_p": top_p, "temperature": temperature, "text_split_method": "cut5", "batch_size": 1, "speed_factor": speed, "seed": actual_seed, "parallel_infer": True, "repetition_penalty": repetition_penalty}
        with _generation_lock: result = list(get_pipeline().run(request))
        if not result: raise RuntimeError("模型没有返回音频。")
        sample_rate, audio = result[-1]
        audio = np.asarray(audio)
        if audio.size == 0 or not audio.any(): raise RuntimeError("模型返回了空音频。")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"chitose_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        sf.write(output_path, audio, sample_rate, subtype="PCM_16")
        return FileResponse(output_path, media_type="audio/wav", filename=output_path.name, headers={"X-Seed": str(actual_seed), "X-Audio-Duration": f"{audio.size/sample_rate:.2f}", "X-Output-Name": output_path.name, "Access-Control-Expose-Headers": "X-Seed, X-Audio-Duration, X-Output-Name"})
    except HTTPException: raise
    except Exception as exc: raise make_error(f"生成失败：{type(exc).__name__}: {exc}", 500) from exc
    finally:
        if uploaded_path is not None:
            try: uploaded_path.unlink(missing_ok=True)
            except OSError: pass


def open_browser() -> None:
    webbrowser.open("http://127.0.0.1:7860")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threading.Timer(1.5, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
