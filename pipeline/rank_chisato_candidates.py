from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get("GPT_SOVITS_ROOT", ROOT.parent / "GPT-SoVITS"))
FFMPEG = os.environ.get("FFMPEG_PATH", "ffmpeg")
SV_MODEL = Path(
    os.environ.get(
        "SPEAKER_VERIFICATION_MODEL",
        ROOT.parent / "models" / "pretrained_eres2netv2w24s4ep4.ckpt",
    )
)
ANALYSIS_ROOT = Path(
    os.environ.get("CHITOSE_ANALYSIS_ROOT", ROOT.parent / "work" / "speaker_analysis")
)
FULL_AUDIO_ROOT = ANALYSIS_ROOT / "full_audio_16k"
SCORES_ROOT = ANALYSIS_ROOT / "scores"

sys.path.insert(0, str(REPO / "GPT_SoVITS" / "eres2net"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import kaldi as Kaldi  # noqa: E402
from ERes2NetV2 import ERes2NetV2  # noqa: E402

from lycoris_dataset_core import Dialogue, find_episode_sources, load_dialogue  # noqa: E402


@dataclass(frozen=True)
class ScoredDialogue:
    item: Dialogue
    centroid_cosine: float
    top5_mean_cosine: float
    max_reference_cosine: float


def load_pcm16_mono(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frames = source.readframes(source.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"Expected PCM16 WAV: {path}")
    samples = array("h")
    samples.frombytes(frames)
    tensor = torch.tensor(samples, dtype=torch.float32)
    if channels > 1:
        tensor = tensor.reshape(-1, channels).mean(dim=1)
    tensor = tensor / 32768.0
    return tensor, sample_rate


def extract_full_audio(episode: int) -> Path:
    FULL_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    output = FULL_AUDIO_ROOT / f"EP{episode:02d}_main_16k.wav"
    if output.is_file() and output.stat().st_size > 44:
        return output
    mkv_path, _ = find_episode_sources(episode)
    temporary = output.with_suffix(".tmp.wav")
    if temporary.exists():
        temporary.unlink()
    subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(mkv_path),
            "-map",
            "0:1",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(output)
    return output


def load_model() -> torch.nn.Module:
    state = torch.load(SV_MODEL, map_location="cpu")
    model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    model.load_state_dict(state)
    model.float().eval()
    return model


def fbank(wav: torch.Tensor) -> torch.Tensor:
    if wav.numel() < 1600:
        wav = F.pad(wav, (0, 1600 - wav.numel()))
    return Kaldi.fbank(
        wav.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0
    )


def embed_waveforms(
    model: torch.nn.Module, waveforms: list[torch.Tensor], batch_size: int = 6
) -> torch.Tensor:
    features = [fbank(wav) for wav in waveforms]
    order = sorted(range(len(features)), key=lambda index: features[index].shape[0])
    output: list[torch.Tensor | None] = [None] * len(features)
    with torch.inference_mode():
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            maximum = max(features[index].shape[0] for index in indices)
            batch = torch.stack(
                [
                    F.pad(features[index], (0, 0, 0, maximum - features[index].shape[0]))
                    for index in indices
                ]
            )
            embeddings = F.normalize(model(batch), dim=1).cpu()
            for index, embedding in zip(indices, embeddings):
                output[index] = embedding
    return torch.stack([item for item in output if item is not None])


def reference_embeddings(model: torch.nn.Module) -> tuple[torch.Tensor, list[Path]]:
    reference_paths: list[Path] = []
    for episode in (1, 2):
        directory = (
            REPO
            / "datasets"
            / f"Lycoris_EP{episode:02d}_Chisato"
            / "wavs_clean"
        )
        reference_paths.extend(sorted(directory.glob("*.wav")))
    if not reference_paths:
        raise RuntimeError("No EP01/EP02 clean reference WAV files were found")
    waveforms: list[torch.Tensor] = []
    for path in reference_paths:
        waveform, sample_rate = load_pcm16_mono(path)
        if sample_rate == 32000:
            waveform = waveform[::2]
        elif sample_rate != 16000:
            raise RuntimeError(f"Unsupported reference sample rate {sample_rate}: {path}")
        waveforms.append(waveform)
    return embed_waveforms(model, waveforms), reference_paths


def episode_embeddings(
    model: torch.nn.Module, episode: int
) -> tuple[list[Dialogue], torch.Tensor]:
    _, ass_path = find_episode_sources(episode)
    dialogue = load_dialogue(ass_path)
    full_audio, sample_rate = load_pcm16_mono(extract_full_audio(episode))
    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16 kHz analysis audio, got {sample_rate}")
    waveforms: list[torch.Tensor] = []
    for item in dialogue:
        start = max(0, math.floor(item.start * sample_rate))
        end = min(full_audio.numel(), math.ceil(item.end * sample_rate))
        waveforms.append(full_audio[start:end])
    return dialogue, embed_waveforms(model, waveforms)


def score_episode(
    dialogue: list[Dialogue],
    embeddings: torch.Tensor,
    references: torch.Tensor,
) -> list[ScoredDialogue]:
    centroid = F.normalize(references.mean(dim=0, keepdim=True), dim=1)
    centroid_scores = embeddings @ centroid.T
    all_scores = embeddings @ references.T
    top_count = min(5, references.shape[0])
    top_scores = torch.topk(all_scores, k=top_count, dim=1).values
    return [
        ScoredDialogue(
            item=item,
            centroid_cosine=float(centroid_scores[index, 0]),
            top5_mean_cosine=float(top_scores[index].mean()),
            max_reference_cosine=float(top_scores[index, 0]),
        )
        for index, item in enumerate(dialogue)
    ]


def write_scores(episode: int, rows: list[ScoredDialogue]) -> Path:
    SCORES_ROOT.mkdir(parents=True, exist_ok=True)
    output = SCORES_ROOT / f"ep{episode:02d}_speaker_scores.tsv"
    with output.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.writer(target, delimiter="\t")
        writer.writerow(
            [
                "id",
                "start",
                "end",
                "duration",
                "centroid_cosine",
                "top5_mean_cosine",
                "max_reference_cosine",
                "jp_text",
                "zh_text",
            ]
        )
        for row in rows:
            item = row.item
            writer.writerow(
                [
                    f"JP{item.id:03d}",
                    item.start_ass,
                    item.end_ass,
                    round(item.end - item.start, 3),
                    round(row.centroid_cosine, 6),
                    round(row.top5_mean_cosine, 6),
                    round(row.max_reference_cosine, 6),
                    item.jp_text,
                    item.zh_text,
                ]
            )
    return output


def known_selected_ids(episode: int) -> set[int]:
    dataset = REPO / "datasets" / f"Lycoris_EP{episode:02d}_Chisato"
    csv_path = dataset / f"ep{episode:02d}_chisato_candidates.csv"
    selected: set[int] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            selected.update(int(value[2:]) for value in row["subtitle_ids"].split())
    return selected


def calibration_summary(all_scores: dict[int, list[ScoredDialogue]]) -> dict[str, object]:
    positives: list[float] = []
    negatives: list[float] = []
    for episode in (1, 2):
        selected = known_selected_ids(episode)
        for row in all_scores[episode]:
            target = positives if row.item.id in selected else negatives
            target.append(row.top5_mean_cosine)

    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        index = round((len(ordered) - 1) * fraction)
        return round(ordered[index], 6)

    candidate_thresholds = sorted(set(positives + negatives))
    best = None
    for threshold in candidate_thresholds:
        true_positive = sum(value >= threshold for value in positives)
        false_positive = sum(value >= threshold for value in negatives)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / len(positives)
        if precision >= 0.95 and (best is None or recall > best["recall"]):
            best = {
                "threshold": round(threshold, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "true_positive": true_positive,
                "false_positive": false_positive,
            }
    return {
        "metric": "top5_mean_cosine",
        "known_positive_events": len(positives),
        "unselected_events_used_as_negative": len(negatives),
        "positive_percentiles": {
            "p05": percentile(positives, 0.05),
            "p25": percentile(positives, 0.25),
            "p50": percentile(positives, 0.50),
            "p75": percentile(positives, 0.75),
            "p95": percentile(positives, 0.95),
        },
        "negative_percentiles": {
            "p50": percentile(negatives, 0.50),
            "p90": percentile(negatives, 0.90),
            "p95": percentile(negatives, 0.95),
            "p99": percentile(negatives, 0.99),
        },
        "best_threshold_with_at_least_95pct_precision": best,
        "warning": (
            "Audio similarity is an auxiliary ranking signal only. Unselected EP01/EP02 "
            "events are not guaranteed ground-truth negatives, and music/overlap can distort scores."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank subtitle events by Chisato voice similarity")
    parser.add_argument(
        "episodes", nargs="*", type=int, default=list(range(1, 14))
    )
    args = parser.parse_args()
    if not (Path(FFMPEG).is_file() or shutil.which(FFMPEG)):
        raise FileNotFoundError(FFMPEG)
    if not SV_MODEL.is_file():
        raise FileNotFoundError(SV_MODEL)

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    print("Loading speaker verification model...", flush=True)
    model = load_model()
    references, reference_paths = reference_embeddings(model)
    print(f"Loaded {len(reference_paths)} retained EP01/EP02 clean references", flush=True)

    all_scores: dict[int, list[ScoredDialogue]] = {}
    for episode in args.episodes:
        print(f"EP{episode:02d}: extracting/loading main audio and embedding dialogue...", flush=True)
        dialogue, embeddings = episode_embeddings(model, episode)
        rows = score_episode(dialogue, embeddings, references)
        all_scores[episode] = rows
        path = write_scores(episode, rows)
        print(f"EP{episode:02d}: {len(rows)} scores -> {path}", flush=True)

    if {1, 2}.issubset(all_scores):
        summary = calibration_summary(all_scores)
        ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
        path = ANALYSIS_ROOT / "calibration_summary.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
