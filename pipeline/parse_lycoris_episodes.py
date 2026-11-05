from __future__ import annotations

import argparse
from pathlib import Path

from lycoris_dataset_core import find_episode_sources, load_dialogue, write_dialogue_tsv


ROOT = Path(__file__).resolve().parent


def parse_episode(episode: int, source_root: Path, output_dir: Path) -> Path:
    _, ass_path = find_episode_sources(episode, source_root)
    dialogue = load_dialogue(ass_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ep{episode:02d}_dialogue.tsv"
    write_dialogue_tsv(dialogue, output_path)
    print(f"EP{episode:02d}: {len(dialogue)} Japanese rows -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse paired Japanese/Chinese Lycoris Recoil ASS dialogue."
    )
    parser.add_argument(
        "episodes",
        nargs="*",
        type=int,
        default=list(range(3, 14)),
        help="Episode numbers (default: 3 through 13).",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Folder containing the episode MKV and bilingual ASS files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "work" / "dialogue",
        help="Destination for generated TSV files.",
    )
    args = parser.parse_args()
    for episode in args.episodes:
        if not 1 <= episode <= 13:
            parser.error(f"Invalid episode: {episode}")
        parse_episode(episode, args.source_root, args.output_dir)


if __name__ == "__main__":
    main()
