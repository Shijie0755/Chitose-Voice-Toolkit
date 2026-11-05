from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Chitose GPT-SoVITS inference test.")
    parser.add_argument("--gpt-sovits-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--prompt-language", default="all_ja")
    parser.add_argument(
        "--text",
        default="おはよう！今日はとってもいい天気だね。さあ、一緒に出かけよう！",
    )
    parser.add_argument("--text-language", default="all_ja")
    parser.add_argument("--output", type=Path, default=Path("chitose_test.wav"))
    args = parser.parse_args()

    root = args.gpt_sovits_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "GPT_SoVITS"))
    os.chdir(root)

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    pipeline = TTS(TTS_Config(str(args.config.expanduser().resolve())))
    request = {
        "text": args.text,
        "text_lang": args.text_language,
        "ref_audio_path": str(args.reference.expanduser().resolve()),
        "prompt_text": args.prompt_text,
        "prompt_lang": args.prompt_language,
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut5",
        "batch_size": 1,
        "speed_factor": 1.0,
        "seed": 2333,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
    }
    result = list(pipeline.run(request))
    if not result:
        raise RuntimeError("Inference returned no audio.")
    sample_rate, audio = result[-1]
    if audio.size == 0 or not audio.any():
        raise RuntimeError("Inference returned silent audio.")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, sample_rate, subtype="PCM_16")
    print(f"output={output}")
    print(f"sample_rate={sample_rate}")
    print(f"samples={audio.size}")


if __name__ == "__main__":
    main()
