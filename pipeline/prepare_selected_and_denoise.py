from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = Path(os.environ.get("GPT_SOVITS_ROOT", ROOT.parent / "GPT-SoVITS")) / "datasets"
COMBINED_ROOT = DATASETS_ROOT / "Lycoris_Chisato_Combined"
FFMPEG = os.environ.get("FFMPEG_PATH", "ffmpeg")

# This deliberately conservative filter reduces steady background hiss/room noise.
# It is not a vocal/BGM separator and leaves the original WAV files untouched.
DENOISE_FILTER = "afftdn=nr=6:nf=-48:tn=1:gs=5"


@dataclass(frozen=True)
class Episode:
    tag: str
    directory: Path

    @property
    def prefix(self) -> str:
        return f"{self.tag.lower()}_chisato"


EPISODES = (
    Episode("EP01", DATASETS_ROOT / "Lycoris_EP01_Chisato"),
    Episode("EP02", DATASETS_ROOT / "Lycoris_EP02_Chisato"),
)


def load_transcripts(episode: Episode) -> dict[str, str]:
    csv_path = episode.directory / f"{episode.prefix}_candidates.csv"
    transcripts: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            clip_id = row["clip_id"].strip()
            text = row["jp_text"].strip().replace("|", " ")
            if clip_id and text:
                transcripts[clip_id] = text
    return transcripts


def list_line(path: Path, text: str) -> str:
    return f"{path.resolve()}|Chisato|ja|{text}"


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def probe_wav(path: Path) -> tuple[float, int, int, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
    duration = frames / sample_rate if sample_rate else 0.0
    return duration, channels, sample_rate, sample_width * 8


def denoise(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.wav")
    if temporary.exists():
        temporary.unlink()
    command = [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-af",
        DENOISE_FILTER,
        "-ac",
        "1",
        "-ar",
        "32000",
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    os.replace(temporary, destination)


def main() -> int:
    if not (Path(FFMPEG).is_file() or shutil.which(FFMPEG)):
        raise FileNotFoundError(f"ffmpeg not found: {FFMPEG}")

    COMBINED_ROOT.mkdir(parents=True, exist_ok=True)
    combined_original: list[str] = []
    combined_denoised: list[str] = []
    manifest: list[dict[str, object]] = []
    selected_output_paths: set[Path] = set()

    for episode in EPISODES:
        transcripts = load_transcripts(episode)
        per_group: dict[str, list[str]] = {"clean": [], "review": []}
        episode_original: list[str] = []
        episode_denoised: list[str] = []
        denoised_dir = episode.directory / "wavs_denoised"
        denoised_dir.mkdir(exist_ok=True)

        selected: list[tuple[str, Path]] = []
        for group in ("clean", "review"):
            source_dir = episode.directory / f"wavs_{group}"
            for source in sorted(source_dir.glob("*.wav")):
                selected.append((group, source))

        for index, (group, source) in enumerate(selected, start=1):
            clip_id = source.stem
            if clip_id not in transcripts:
                raise KeyError(f"No Japanese transcript found for {source}")
            text = transcripts[clip_id]
            destination = denoised_dir / source.name
            selected_output_paths.add(destination.resolve())

            denoise(source, destination)
            src_duration, src_channels, src_rate, src_bits = probe_wav(source)
            out_duration, out_channels, out_rate, out_bits = probe_wav(destination)
            if abs(src_duration - out_duration) > 0.05:
                raise RuntimeError(
                    f"Duration changed unexpectedly: {source.name} "
                    f"{src_duration:.3f}s -> {out_duration:.3f}s"
                )
            if (out_channels, out_rate, out_bits) != (1, 32000, 16):
                raise RuntimeError(
                    f"Unexpected output format for {destination}: "
                    f"{out_channels}ch/{out_rate}Hz/{out_bits}bit"
                )

            original_line = list_line(source, text)
            denoised_line = list_line(destination, text)
            per_group[group].append(original_line)
            episode_original.append(original_line)
            episode_denoised.append(denoised_line)
            combined_original.append(original_line)
            combined_denoised.append(denoised_line)
            manifest.append(
                {
                    "episode": episode.tag,
                    "clip_id": clip_id,
                    "source_group": group,
                    "source_path": str(source.resolve()),
                    "denoised_path": str(destination.resolve()),
                    "duration_seconds": round(out_duration, 6),
                    "japanese_text": text,
                    "filter": DENOISE_FILTER,
                    "source_format": f"{src_channels}ch/{src_rate}Hz/{src_bits}bit",
                    "output_format": f"{out_channels}ch/{out_rate}Hz/{out_bits}bit",
                }
            )
            print(
                f"[{len(manifest):03d}] {episode.tag} {group}: "
                f"{source.name} ({out_duration:.2f}s)"
            )

        # Existing lists are rebuilt strictly from the WAV files the user retained.
        write_lines(
            episode.directory / f"{episode.prefix}_clean.list", per_group["clean"]
        )
        write_lines(
            episode.directory / f"{episode.prefix}_review.list", per_group["review"]
        )
        write_lines(
            episode.directory / f"{episode.prefix}_selected.list", episode_original
        )
        write_lines(
            episode.directory / f"{episode.prefix}_denoised.list", episode_denoised
        )

        # Remove only stale derived WAVs in this dedicated output directory.
        for stale in denoised_dir.glob("*.wav"):
            if stale.resolve() not in selected_output_paths:
                stale.unlink()

    write_lines(COMBINED_ROOT / "chisato_selected_original.list", combined_original)
    write_lines(COMBINED_ROOT / "chisato_selected_denoised.list", combined_denoised)

    manifest_path = COMBINED_ROOT / "denoise_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    total_duration = sum(float(row["duration_seconds"]) for row in manifest)
    summary = {
        "clips": len(manifest),
        "duration_seconds": round(total_duration, 3),
        "duration_minutes": round(total_duration / 60, 3),
        "episodes": {
            episode.tag: sum(1 for row in manifest if row["episode"] == episode.tag)
            for episode in EPISODES
        },
        "source_groups": {
            group: sum(1 for row in manifest if row["source_group"] == group)
            for group in ("clean", "review")
        },
        "denoise_filter": DENOISE_FILTER,
        "output_format": "mono/32000Hz/16bit PCM WAV",
        "note": (
            "Light stationary-noise reduction only; this does not reliably remove "
            "music, sound effects, reverberation, or overlapping speakers."
        ),
    }
    (COMBINED_ROOT / "denoise_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        raise
