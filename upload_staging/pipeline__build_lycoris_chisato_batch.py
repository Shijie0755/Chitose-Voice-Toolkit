from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import wave
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path

from lycoris_dataset_core import Dialogue, find_episode_sources, load_dialogue


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("GPT_SOVITS_ROOT", ROOT.parent / "GPT-SoVITS"))
DATASETS_ROOT = PROJECT_ROOT / "datasets"
SELECTION_PATH = Path(
    os.environ.get(
        "CHITOSE_SELECTION_PATH",
        ROOT / "configs" / "chisato_selections_ep03_ep13.json",
    )
)
FFMPEG = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_PATH", "ffprobe")
STAGE_A_JSON = DATASETS_ROOT / "Lycoris_EP03_EP13_Chisato_StageA_summary.json"
STAGE_A_REPORT = DATASETS_ROOT / "Lycoris_EP03_EP13_Chisato_StageA_report.md"


def executable_exists(value: str) -> bool:
    return Path(value).is_file() or shutil.which(value) is not None


@dataclass(frozen=True)
class AudioStream:
    index: int
    codec: str
    sample_rate: int
    channels: int
    language: str
    title: str


@dataclass
class Clip:
    clip_id: str
    subtitle_ids: list[int]
    start: float
    end: float
    duration: float
    jp_text: str
    zh_text: str
    confidence: str
    conflicting_subtitle_ids: list[int]
    output_group: str
    output_path: str = ""
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None


def timecode(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def probe_audio_streams(mkv_path: Path) -> list[AudioStream]:
    result = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_name,sample_rate,channels:stream_tags=language,title",
            "-of",
            "json",
            str(mkv_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result.stdout)
    streams: list[AudioStream] = []
    for item in payload.get("streams", []):
        tags = item.get("tags", {})
        streams.append(
            AudioStream(
                index=int(item["index"]),
                codec=str(item.get("codec_name", "")),
                sample_rate=int(item.get("sample_rate", 0)),
                channels=int(item.get("channels", 0)),
                language=str(tags.get("language", "")),
                title=str(tags.get("title", "")),
            )
        )
    return streams


def select_main_audio(streams: list[AudioStream], mkv_path: Path) -> AudioStream:
    exact = [
        stream
        for stream in streams
        if stream.title.casefold() == "main audio" and stream.language.casefold() == "jpn"
    ]
    if len(exact) == 1:
        return exact[0]
    titled = [stream for stream in streams if stream.title.casefold() == "main audio"]
    if len(titled) == 1:
        return titled[0]
    raise RuntimeError(
        f"Could not identify exactly one Japanese Main Audio stream in {mkv_path}: "
        f"{[asdict(stream) for stream in streams]}"
    )


def load_selections() -> dict[int, dict[str, object]]:
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    result: dict[int, dict[str, object]] = {}
    for tag, config in payload["episodes"].items():
        episode = int(tag.removeprefix("EP"))
        result[episode] = config
    return result


def selected_ids(config: dict[str, object], key: str) -> set[int]:
    return {int(value) for value in config.get(key, [])}


def validate_selection(
    episode: int, dialogue: list[Dialogue], high: set[int], probable: set[int]
) -> None:
    overlap = high & probable
    if overlap:
        raise ValueError(f"EP{episode:02d} IDs appear in high and probable: {sorted(overlap)}")
    maximum = len(dialogue)
    invalid = sorted(value for value in high | probable if not 1 <= value <= maximum)
    if invalid:
        raise ValueError(f"EP{episode:02d} selection contains invalid IDs: {invalid}")
    if not high and not probable:
        raise ValueError(f"EP{episode:02d} has an empty selection")


def build_groups(
    dialogue: list[Dialogue], selected: set[int]
) -> list[list[Dialogue]]:
    groups: list[list[Dialogue]] = []
    current: list[Dialogue] = []
    for item in dialogue:
        if item.id not in selected:
            if current:
                groups.append(current)
                current = []
            continue
        may_join = bool(
            current
            and item.id == current[-1].id + 1
            and item.start - current[-1].end <= 0.65
            and item.end - current[0].start <= 10.0
        )
        if may_join:
            current.append(item)
        else:
            if current:
                groups.append(current)
            current = [item]
    if current:
        groups.append(current)
    return groups


def make_clips(
    episode: int,
    dialogue: list[Dialogue],
    high: set[int],
    probable: set[int],
) -> list[Clip]:
    selected = high | probable
    by_id = {item.id: item for item in dialogue}
    clips: list[Clip] = []
    for sequence, group in enumerate(build_groups(dialogue, selected), 1):
        raw_start = group[0].start
        raw_end = group[-1].end
        prior = by_id.get(group[0].id - 1)
        following = by_id.get(group[-1].id + 1)
        start_padding = 0.15
        end_padding = 0.15
        if prior is not None:
            start_padding = min(start_padding, max(0.0, (raw_start - prior.end) / 2))
        if following is not None:
            end_padding = min(end_padding, max(0.0, (following.start - raw_end) / 2))
        # Keep the preferred upper bound at 10 seconds even after boundary padding.
        # Scale both sides proportionally so a long but valid subtitle group is not
        # pushed over the limit only by the added breathing room.
        padding_budget = max(0.0, 10.0 - (raw_end - raw_start))
        padding_total = start_padding + end_padding
        if padding_total > padding_budget and padding_total > 0:
            padding_scale = padding_budget / padding_total
            start_padding *= padding_scale
            end_padding *= padding_scale
        start = max(0.0, raw_start - start_padding)
        end = raw_end + end_padding
        ids = [item.id for item in group]
        conflicts = [
            item.id
            for item in dialogue
            if item.id not in selected and item.start < end and item.end > start
        ]
        confidence = "probable" if any(item.id in probable for item in group) else "high"
        needs_review = confidence == "probable" or bool(conflicts) or (end - start) < 1.0
        clips.append(
            Clip(
                clip_id=f"EP{episode:02d}_CHISATO_{sequence:03d}",
                subtitle_ids=ids,
                start=start,
                end=end,
                duration=end - start,
                jp_text=" ".join(item.jp_text for item in group),
                zh_text=" ".join(item.zh_text for item in group if item.zh_text),
                confidence=confidence,
                conflicting_subtitle_ids=conflicts,
                output_group="review" if needs_review else "clean",
            )
        )
    return clips


def dataset_root(episode: int) -> Path:
    return DATASETS_ROOT / f"Lycoris_EP{episode:02d}_Chisato"


def clear_generated_wavs(episode: int) -> None:
    prefix = f"EP{episode:02d}_CHISATO_"
    for group in ("clean", "review"):
        output_dir = dataset_root(episode) / f"wavs_{group}"
        if not output_dir.is_dir():
            continue
        for path in output_dir.glob(f"{prefix}*.wav"):
            path.unlink()


def cut_clip(
    episode: int, mkv_path: Path, stream: AudioStream, clip: Clip
) -> None:
    output_dir = dataset_root(episode) / f"wavs_{clip.output_group}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{clip.clip_id}.wav"
    temporary = output_path.with_name(output_path.stem + ".tmp.wav")
    if temporary.exists():
        temporary.unlink()
    subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            timecode(clip.start),
            "-to",
            timecode(clip.end),
            "-i",
            str(mkv_path),
            "-map",
            f"0:{stream.index}",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            "32000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, output_path)
    clip.output_path = str(output_path.resolve())


def analyse_wav(clip: Clip) -> None:
    with wave.open(clip.output_path, "rb") as source:
        frames = source.readframes(source.getnframes())
    samples = array("h")
    samples.frombytes(frames)
    if not samples:
        clip.peak_dbfs = float("-inf")
        clip.rms_dbfs = float("-inf")
        return
    peak = max(abs(sample) for sample in samples) / 32768.0
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768.0
    clip.peak_dbfs = round(20 * math.log10(max(peak, 1e-12)), 2)
    clip.rms_dbfs = round(20 * math.log10(max(rms, 1e-12)), 2)


def write_list(path: Path, clips: list[Clip], group: str) -> None:
    lines: list[str] = []
    for clip in clips:
        if clip.output_group != group:
            continue
        text = re.sub(r"[|\r\n]+", " ", clip.jp_text).strip()
        lines.append(f"{clip.output_path}|Chisato|ja|{text}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_episode_outputs(
    episode: int,
    mkv_path: Path,
    ass_path: Path,
    streams: list[AudioStream],
    selected_stream: AudioStream,
    clips: list[Clip],
    config: dict[str, object],
) -> dict[str, object]:
    root = dataset_root(episode)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"ep{episode:02d}_chisato"
    csv_path = root / f"{prefix}_candidates.csv"
    fields = list(asdict(clips[0]).keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for clip in clips:
            row = asdict(clip)
            row["subtitle_ids"] = " ".join(f"JP{value:03d}" for value in clip.subtitle_ids)
            row["conflicting_subtitle_ids"] = " ".join(
                f"JP{value:03d}" for value in clip.conflicting_subtitle_ids
            )
            writer.writerow(row)

    for group in ("clean", "review"):
        write_list(root / f"{prefix}_{group}.list", clips, group)

    summary: dict[str, object] = {
        "episode": f"EP{episode:02d}",
        "source_mkv": str(mkv_path),
        "source_ass": str(ass_path),
        "available_audio_streams": [asdict(stream) for stream in streams],
        "selected_audio_stream": asdict(selected_stream),
        "selection_basis": (
            "Inferred from paired Japanese/Chinese subtitles, ordered dialogue and episode "
            "context; speaker labels are not ground truth. Audio similarity was used only as "
            "an auxiliary review signal."
        ),
        "selection_note": str(config.get("note", "")),
        "audio_format": "mono PCM signed 16-bit WAV, 32000 Hz",
        "total_clips": len(clips),
        "clean_clips": sum(clip.output_group == "clean" for clip in clips),
        "review_clips": sum(clip.output_group == "review" for clip in clips),
        "total_duration_seconds": round(sum(clip.duration for clip in clips), 3),
        "clean_duration_seconds": round(
            sum(clip.duration for clip in clips if clip.output_group == "clean"), 3
        ),
        "review_duration_seconds": round(
            sum(clip.duration for clip in clips if clip.output_group == "review"), 3
        ),
        "warning": (
            "Speaker labels are inferred, not ground truth. Manually listen to review clips "
            "and spot-check clean clips before any final training list is built."
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    stream_lines = "\n".join(
        f"  - stream `{stream.index}`: `{stream.codec}`, {stream.sample_rate} Hz, "
        f"{stream.channels} ch, language `{stream.language}`, title `{stream.title}`"
        for stream in streams
    )
    readme = f"""# Lycoris Recoil EP{episode:02d} - Chisato candidate dataset

This folder contains inferred Chisato Nishikigi dialogue candidates from episode {episode} only.

## Sources and audio stream

- Video/audio: `{mkv_path.name}`
- Subtitles: `{ass_path.name}`
- Available audio streams:
{stream_lines}
- Selected for extraction: stream `{selected_stream.index}` (`{selected_stream.title}`)

## Output

- `wavs_clean/`: {summary['clean_clips']} high-confidence candidates without automatically detected subtitle conflicts.
- `wavs_review/`: {summary['review_clips']} probable, overlapping, or shorter-than-one-second candidates.
- `{prefix}_clean.list`: GPT-SoVITS list for clean candidates.
- `{prefix}_review.list`: a separate manual-review list; do not merge it blindly into training.
- `{prefix}_candidates.csv`: source subtitle IDs, timing, Japanese transcript, Chinese context, confidence, conflicts, path and level metadata.
- `summary.json`: machine-readable episode summary.

All WAVs use the Japanese Main Audio stream and are mono, 32 kHz, signed 16-bit PCM. Japanese text is the training transcript; Chinese text is context metadata only.

## Important limitation and required next step

The ASS file does not provide reliable speaker names. The labels here are inferred, not ground truth. Manually listen to every file in `wavs_review/` and spot-check `wavs_clean/`. Delete wrong-speaker, overlapped, clipped, music-heavy or otherwise poor WAVs before asking for Stage B.

Do not use these candidate lists as the final all-season training list. Stage B must rebuild lists from the WAV files that remain after manual cleanup.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    return summary


def write_stage_report(summaries: list[dict[str, object]]) -> None:
    total_clips = sum(int(item["total_clips"]) for item in summaries)
    total_duration = sum(float(item["total_duration_seconds"]) for item in summaries)
    aggregate = {
        "scope": "EP03-EP13 Stage A candidate datasets",
        "episodes": summaries,
        "total_clips": total_clips,
        "clean_clips": sum(int(item["clean_clips"]) for item in summaries),
        "review_clips": sum(int(item["review_clips"]) for item in summaries),
        "total_duration_seconds": round(total_duration, 3),
        "clean_duration_seconds": round(
            sum(float(item["clean_duration_seconds"]) for item in summaries), 3
        ),
        "review_duration_seconds": round(
            sum(float(item["review_duration_seconds"]) for item in summaries), 3
        ),
        "stage_b_status": "not run; waiting for explicit confirmation after manual cleanup",
    }
    STAGE_A_JSON.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Lycoris Recoil Chisato Stage A report (EP03-EP13)",
        "",
        "Speaker assignments are inferred rather than ground truth. Stage B has not been run.",
        "",
        "| Episode | Clean clips | Review clips | Clean seconds | Review seconds | Total seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['episode']} | {item['clean_clips']} | {item['review_clips']} | "
            f"{float(item['clean_duration_seconds']):.3f} | "
            f"{float(item['review_duration_seconds']):.3f} | "
            f"{float(item['total_duration_seconds']):.3f} |"
        )
    lines.extend(
        [
            f"| **Total** | **{aggregate['clean_clips']}** | **{aggregate['review_clips']}** | "
            f"**{aggregate['clean_duration_seconds']:.3f}** | "
            f"**{aggregate['review_duration_seconds']:.3f}** | "
            f"**{aggregate['total_duration_seconds']:.3f}** |",
            "",
            "Next: manually listen to and delete wrong or poor WAVs, then explicitly request Stage B.",
        ]
    )
    STAGE_A_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_episode(episode: int, config: dict[str, object]) -> dict[str, object]:
    mkv_path, ass_path = find_episode_sources(episode)
    dialogue = load_dialogue(ass_path)
    high = selected_ids(config, "high_ids")
    probable = selected_ids(config, "probable_ids")
    validate_selection(episode, dialogue, high, probable)
    streams = probe_audio_streams(mkv_path)
    stream = select_main_audio(streams, mkv_path)
    clips = make_clips(episode, dialogue, high, probable)
    clear_generated_wavs(episode)
    print(
        f"EP{episode:02d}: {len(high)} high + {len(probable)} probable subtitle events; "
        f"{len(clips)} merged clips; stream {stream.index} {stream.title}",
        flush=True,
    )
    for index, clip in enumerate(clips, 1):
        cut_clip(episode, mkv_path, stream, clip)
        analyse_wav(clip)
        print(
            f"  [{index:03d}/{len(clips):03d}] {clip.clip_id} "
            f"{clip.output_group} {clip.duration:.2f}s",
            flush=True,
        )
    return write_episode_outputs(
        episode, mkv_path, ass_path, streams, stream, clips, config
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EP03-EP13 Chisato Stage A datasets")
    parser.add_argument(
        "episodes", nargs="*", type=int, default=list(range(3, 14))
    )
    args = parser.parse_args()
    if not executable_exists(FFMPEG):
        raise FileNotFoundError(FFMPEG)
    if not executable_exists(FFPROBE):
        raise FileNotFoundError(FFPROBE)
    if not PROJECT_ROOT.is_dir():
        raise FileNotFoundError(PROJECT_ROOT)
    selections = load_selections()
    summaries = []
    for episode in args.episodes:
        if episode not in selections:
            raise KeyError(f"No selection config for EP{episode:02d}")
        summaries.append(build_episode(episode, selections[episode]))
    if set(args.episodes) == set(range(3, 14)):
        write_stage_report(summaries)
    print(f"Stage A outputs complete for: {', '.join(item['episode'] for item in summaries)}")


if __name__ == "__main__":
    main()

