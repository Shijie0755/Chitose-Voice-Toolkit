from __future__ import annotations

import json
import os
from pathlib import Path

import yaml


REPO = Path(os.environ.get("GPT_SOVITS_ROOT", Path.cwd())).expanduser()
RUNTIME = Path(os.environ.get("GPT_SOVITS_RUNTIME", REPO)).expanduser()
EXP_NAME = "Chisato_v2"
EXP_DIR = RUNTIME / "experiments" / EXP_NAME
PRETRAINED = RUNTIME / "pretrained_models" / "gsv-v2final-pretrained"
CONFIG_DIR = RUNTIME / "configs"
SOVITS_WEIGHT_DIR = RUNTIME / "weights" / "SoVITS" / "v2"
GPT_WEIGHT_DIR = RUNTIME / "weights" / "GPT" / "v2"


def ensure_semantic_header() -> None:
    semantic_path = EXP_DIR / "6-name2semantic.tsv"
    content = semantic_path.read_text(encoding="utf-8")
    header = "item_name\tsemantic_audio\n"
    if not content.startswith(header):
        semantic_path.write_text(header + content, encoding="utf-8")


def build_sovits_config() -> Path:
    with (REPO / "GPT_SoVITS" / "configs" / "s2.json").open(encoding="utf-8") as source:
        config = json.load(source)

    config["train"].update(
        {
            "batch_size": 2,
            "epochs": 8,
            "fp16_run": True,
            "text_low_lr_rate": 0.4,
            "pretrained_s2G": str(PRETRAINED / "s2G2333k.pth"),
            "pretrained_s2D": str(PRETRAINED / "s2D2333k.pth"),
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": 4,
            "gpu_numbers": "0",
            "grad_ckpt": False,
        }
    )
    config["model"]["version"] = "v2"
    config["data"]["exp_dir"] = str(EXP_DIR)
    config["s2_ckpt_dir"] = str(EXP_DIR)
    config["save_weight_dir"] = str(SOVITS_WEIGHT_DIR)
    config["name"] = EXP_NAME
    config["version"] = "v2"

    output = CONFIG_DIR / "chisato_v2_sovits.json"
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def build_gpt_config() -> Path:
    with (REPO / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)

    config["train"].update(
        {
            "batch_size": 2,
            "epochs": 15,
            "precision": "16-mixed",
            "save_every_n_epoch": 5,
            "if_save_latest": True,
            "if_save_every_weights": True,
            "if_dpo": False,
            "half_weights_save_dir": str(GPT_WEIGHT_DIR),
            "exp_name": EXP_NAME,
        }
    )
    config["pretrained_s1"] = str(
        PRETRAINED / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    )
    config["train_semantic_path"] = str(EXP_DIR / "6-name2semantic.tsv")
    config["train_phoneme_path"] = str(EXP_DIR / "2-name2text.txt")
    config["output_dir"] = str(EXP_DIR / "logs_s1_v2")

    output = CONFIG_DIR / "chisato_v2_gpt.yaml"
    output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return output


def build_inference_config() -> Path:
    config = {
        "custom": {
            "bert_base_path": str(RUNTIME / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(RUNTIME / "pretrained_models" / "chinese-hubert-base"),
            "device": "cuda",
            "is_half": True,
            "t2s_weights_path": str(GPT_WEIGHT_DIR / f"{EXP_NAME}-e15.ckpt"),
            "version": "v2",
            "vits_weights_path": str(SOVITS_WEIGHT_DIR / f"{EXP_NAME}_e8_s624.pth"),
        }
    }
    output = CONFIG_DIR / "chisato_v2_inference.yaml"
    output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return output


def main() -> None:
    for directory in (CONFIG_DIR, SOVITS_WEIGHT_DIR, GPT_WEIGHT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    ensure_semantic_header()
    print(build_sovits_config())
    print(build_gpt_config())
    print(build_inference_config())


if __name__ == "__main__":
    main()
