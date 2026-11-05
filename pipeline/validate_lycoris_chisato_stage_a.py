from __future__ import annotations

import csv
import json
import math
import os
import wave
from array import array
from pathlib import Path

from lycoris_dataset_core import find_episode_sources, load_dialogue


ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = Path(os.environ.get("GPT_SOVITS_ROOT", ROOT.parent / "GPT-SoVITS")) / "datasets"
VALIDATION_JSON = DATASETS_ROOT / "Lycoris_EP03_EP13_Chisato_StageA_validation.json"
SPOTCHECK_CSV = DATASETS_ROOT / "Lycoris_EP03_EP13_Chisato_StageA_spotchecks.csv"


def dbfs(value: float) -> float:
    return 20 * math.log10(max(value, 1e-12))


def wav_metrics(path: Path) -> dict[str, float | int | str]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frames = source.getnframes()
        compression = source.getcomptype()
        raw = source.readframes(frames)
    samples = array("h")
    samples.frombytes(raw)
    absolute = [abs(value) for value in samples]
    if not absolute:
        peak = rms = 0.0
    else:
        peak = max(absolute) / 32768.0
        rms = math.sqrt(sum(value * value for value in samples) / len(samples)) / 32768.0
    edge_count = min(len(samples), max(1, round(sample_rate * 0.1)))

    def window_rms(values: array | list[int]) -> float:
        if not values:
            return -240.0
        return dbfs(math.sqrt(sum(value * value for value in values) / len(values)) / 32768.0)

    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "bits": sample_width * 8,
        "compression": compression,
        "frames": frames,
        "duration": frames / sample_rate if sample_rate else 0.0,
        "peak_dbfs": round(dbfs(peak), 3),
        "rms_dbfs": round(dbfs(rms), 3),
        "leading_100ms_rms_dbfs": round(window_rms(samples[:edge_count]), 3),
        "trailing_100ms_rms_dbfs": round(window_rms(samples[-edge_count:]), 3),
        "zero_fraction": round(sum(value == 0 for value in samples) / max(1, len(samples)), 8),
        "clipped_fraction": round(
            sum(value >= 32760 for value in absolute) / max(1, len(absolute)), 8
        ),
    }


def parse_ids(value: str) -> list[int]:
    return [int(item[2:]) for item in value.split() if item]


def close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_episode(
    episode: int,
    global_clip_ids: set[str],
    global_csv_paths: set[Path],
    global_list_paths: set[Path],
) -> tuple[dict[str, object], list[dict[str, object]], list[str], list[str]]:
    tag = f"EP{episode:02d}"
    prefix = f"ep{episode:02d}_chisato"
    root = DATASETS_ROOT / f"Lycoris_{tag}_Chisato"
    csv_path = root / f"{prefix}_candidates.csv"
    summary_path = root / "summary.json"
    errors: list[str] = []
    warnings: list[str] = []
    if not csv_path.is_file():
        return ({"episode": tag, "status": "failed"}, [], [f"Missing {csv_path}"], [])

    _, ass_path = find_episode_sources(episode)
    dialogue = load_dialogue(ass_path)
    by_id = {item.id: item for item in dialogue}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    csv_paths: set[Path] = set()
    group_paths: dict[str, set[Path]] = {"clean": set(), "review": set()}
    metrics_by_path: dict[Path, dict[str, float | int | str]] = {}

    for row in rows:
        clip_id = row["clip_id"].strip()
        if clip_id in global_clip_ids:
            fail(errors, f"Duplicate global clip_id: {clip_id}")
        global_clip_ids.add(clip_id)
        if not clip_id.startswith(f"{tag}_CHISATO_"):
            fail(errors, f"Wrong episode prefix in clip_id: {clip_id}")
        output_group = row["output_group"].strip()
        if output_group not in group_paths:
            fail(errors, f"Invalid output_group for {clip_id}: {output_group}")
            continue
        path = Path(row["output_path"])
        resolved = path.resolve()
        if resolved in csv_paths or resolved in global_csv_paths:
            fail(errors, f"Duplicate CSV path: {resolved}")
        csv_paths.add(resolved)
        global_csv_paths.add(resolved)
        group_paths[output_group].add(resolved)
        if path.parent != root / f"wavs_{output_group}":
            fail(errors, f"Group/path mismatch for {clip_id}: {path}")
        if not path.is_file():
            fail(errors, f"Missing WAV for {clip_id}: {path}")
            continue
        if path.stat().st_size <= 44:
            fail(errors, f"Empty WAV for {clip_id}: {path}")
            continue
        metrics = wav_metrics(path)
        metrics_by_path[resolved] = metrics
        if (
            metrics["channels"],
            metrics["sample_rate"],
            metrics["bits"],
            metrics["compression"],
        ) != (1, 32000, 16, "NONE"):
            fail(
                errors,
                f"Wrong WAV format for {clip_id}: {metrics['channels']}ch/"
                f"{metrics['sample_rate']}Hz/{metrics['bits']}bit/{metrics['compression']}",
            )
        expected_duration = float(row["duration"])
        if not close(float(metrics["duration"]), expected_duration, 0.05):
            fail(
                errors,
                f"Duration mismatch for {clip_id}: CSV {expected_duration:.6f}s, "
                f"WAV {float(metrics['duration']):.6f}s",
            )
        if float(metrics["rms_dbfs"]) < -60:
            fail(errors, f"Likely silent WAV for {clip_id}: {metrics['rms_dbfs']} dBFS RMS")
        if float(metrics["clipped_fraction"]) > 0.01:
            warnings.append(
                f"High clipped-sample fraction for {clip_id}: {metrics['clipped_fraction']}"
            )
        if row["peak_dbfs"] and not close(
            float(row["peak_dbfs"]), float(metrics["peak_dbfs"]), 0.15
        ):
            fail(errors, f"Stored peak level mismatch for {clip_id}")
        if row["rms_dbfs"] and not close(
            float(row["rms_dbfs"]), float(metrics["rms_dbfs"]), 0.15
        ):
            fail(errors, f"Stored RMS level mismatch for {clip_id}")

        ids = parse_ids(row["subtitle_ids"])
        if not ids or any(value not in by_id for value in ids):
            fail(errors, f"Invalid subtitle_ids for {clip_id}: {row['subtitle_ids']}")
            continue
        source_rows = [by_id[value] for value in ids]
        expected_jp = " ".join(item.jp_text for item in source_rows)
        expected_zh = " ".join(item.zh_text for item in source_rows if item.zh_text)
        if row["jp_text"] != expected_jp:
            fail(errors, f"Japanese transcript/source mismatch for {clip_id}")
        if row["zh_text"] != expected_zh:
            fail(errors, f"Chinese context/source mismatch for {clip_id}")
        start = float(row["start"])
        end = float(row["end"])
        if start > source_rows[0].start + 0.001 or end < source_rows[-1].end - 0.001:
            fail(errors, f"Clip does not contain full subtitle interval for {clip_id}")
        if source_rows[0].start - start > 0.151 or end - source_rows[-1].end > 0.151:
            fail(errors, f"Padding exceeds 0.15s for {clip_id}")
        if "|" in row["jp_text"] or "\n" in row["jp_text"] or "\r" in row["jp_text"]:
            fail(errors, f"Unsafe list character in Japanese transcript for {clip_id}")

    for group in ("clean", "review"):
        list_path = root / f"{prefix}_{group}.list"
        if not list_path.is_file():
            fail(errors, f"Missing list: {list_path}")
            continue
        listed_paths: set[Path] = set()
        lines = list_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            parts = line.split("|")
            if len(parts) != 4:
                fail(errors, f"Malformed list line {list_path}:{line_number}")
                continue
            path_text, speaker, language, text = parts
            path = Path(path_text).resolve()
            if speaker != "Chisato" or language != "ja" or not text.strip():
                fail(errors, f"Bad list metadata {list_path}:{line_number}")
            if path in listed_paths or path in global_list_paths:
                fail(errors, f"Duplicate list path: {path}")
            listed_paths.add(path)
            global_list_paths.add(path)
            if not path.is_file():
                fail(errors, f"List points to missing WAV: {path}")
            if path not in group_paths[group]:
                fail(errors, f"List path absent from {group} CSV rows: {path}")
            matching = [row for row in rows if Path(row["output_path"]).resolve() == path]
            if matching and matching[0]["jp_text"] != text:
                fail(errors, f"List transcript mismatch for {path.name}")
        if listed_paths != group_paths[group]:
            missing = group_paths[group] - listed_paths
            extra = listed_paths - group_paths[group]
            fail(
                errors,
                f"CSV/list path-set mismatch for {tag} {group}: "
                f"missing={len(missing)}, extra={len(extra)}",
            )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_counts = {
        "total_clips": len(rows),
        "clean_clips": len(group_paths["clean"]),
        "review_clips": len(group_paths["review"]),
    }
    for key, expected in expected_counts.items():
        if int(summary[key]) != expected:
            fail(errors, f"Summary {key} mismatch for {tag}: {summary[key]} != {expected}")
    calculated_durations = {
        "total_duration_seconds": sum(float(row["duration"]) for row in rows),
        "clean_duration_seconds": sum(
            float(row["duration"]) for row in rows if row["output_group"] == "clean"
        ),
        "review_duration_seconds": sum(
            float(row["duration"]) for row in rows if row["output_group"] == "review"
        ),
    }
    for key, expected in calculated_durations.items():
        if not close(float(summary[key]), expected, 0.01):
            fail(errors, f"Summary {key} mismatch for {tag}")

    sample_indices: list[int] = []
    if rows:
        sample_indices = sorted(set([0, len(rows) // 2, len(rows) - 1]))
    spotchecks: list[dict[str, object]] = []
    for index in sample_indices:
        row = rows[index]
        path = Path(row["output_path"]).resolve()
        metrics = metrics_by_path[path]
        spotchecks.append(
            {
                "episode": tag,
                "clip_id": row["clip_id"],
                "subtitle_ids": row["subtitle_ids"],
                "start": row["start"],
                "end": row["end"],
                "jp_text": row["jp_text"],
                "output_group": row["output_group"],
                "duration_seconds": metrics["duration"],
                "peak_dbfs": metrics["peak_dbfs"],
                "rms_dbfs": metrics["rms_dbfs"],
                "leading_100ms_rms_dbfs": metrics["leading_100ms_rms_dbfs"],
                "trailing_100ms_rms_dbfs": metrics["trailing_100ms_rms_dbfs"],
                "zero_fraction": metrics["zero_fraction"],
                "clipped_fraction": metrics["clipped_fraction"],
                "subtitle_timing_and_text_match": True,
                "path": str(path),
            }
        )

    result = {
        "episode": tag,
        "status": "passed" if not errors else "failed",
        "clips": len(rows),
        "clean_clips": len(group_paths["clean"]),
        "review_clips": len(group_paths["review"]),
        "duration_seconds": round(calculated_durations["total_duration_seconds"], 3),
        "spotchecks": len(spotchecks),
        "errors": len(errors),
        "warnings": len(warnings),
    }
    return result, spotchecks, errors, warnings


def main() -> None:
    global_clip_ids: set[str] = set()
    global_csv_paths: set[Path] = set()
    global_list_paths: set[Path] = set()
    results: list[dict[str, object]] = []
    spotchecks: list[dict[str, object]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for episode in range(3, 14):
        result, samples, episode_errors, episode_warnings = validate_episode(
            episode, global_clip_ids, global_csv_paths, global_list_paths
        )
        results.append(result)
        spotchecks.extend(samples)
        errors.extend(episode_errors)
        warnings.extend(episode_warnings)
        print(
            f"{result['episode']}: {result['status']} - {result.get('clips', 0)} clips, "
            f"{result.get('errors', len(episode_errors))} errors, "
            f"{result.get('warnings', len(episode_warnings))} warnings",
            flush=True,
        )

    if spotchecks:
        with SPOTCHECK_CSV.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(spotchecks[0].keys()))
            writer.writeheader()
            writer.writerows(spotchecks)

    payload = {
        "status": "passed" if not errors else "failed",
        "episodes": results,
        "total_clips": sum(int(item.get("clips", 0)) for item in results),
        "total_duration_seconds": round(
            sum(float(item.get("duration_seconds", 0.0)) for item in results), 3
        ),
        "unique_clip_ids": len(global_clip_ids),
        "unique_csv_paths": len(global_csv_paths),
        "unique_list_paths": len(global_list_paths),
        "spotchecked_clips": len(spotchecks),
        "checks": [
            "CSV and list targets exist",
            "non-empty PCM WAV",
            "mono, 32000 Hz, signed 16-bit PCM",
            "unique paths and cross-episode clip IDs",
            "WAV/CSV duration within 0.05 seconds",
            "stored and recomputed peak/RMS levels",
            "subtitle IDs, Japanese text, Chinese context and source timing correspondence",
            "per-episode summary counts and duration",
            "three deterministic waveform/level/timing spot checks per episode",
        ],
        "errors": errors,
        "warnings": warnings,
    }
    VALIDATION_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
