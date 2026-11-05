from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path


SOURCE_ROOT_ENV = "LYCORIS_SOURCE_ROOT"


def configured_source_root() -> Path:
    value = os.environ.get(SOURCE_ROOT_ENV)
    if not value:
        raise RuntimeError(
            f"Set {SOURCE_ROOT_ENV} to the folder containing your MKV and ASS files."
        )
    return Path(value).expanduser()


@dataclass(frozen=True)
class AssEvent:
    event_index: int
    start_ass: str
    end_ass: str
    start: float
    end: float
    style: str
    name: str
    text: str


@dataclass(frozen=True)
class Dialogue:
    id: int
    start: float
    end: float
    start_ass: str
    end_ass: str
    name: str
    jp_text: str
    zh_text: str


def clean_ass_text(text: str) -> str:
    text = re.sub(r"\{[^}]*\}", "", text)
    text = text.replace(r"\N", " ").replace(r"\n", " ").replace(r"\h", " ")
    return re.sub(r"\s+", " ", text).strip()


def ass_time_to_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_ass_events(path: Path) -> list[AssEvent]:
    events: list[AssEvent] = []
    in_events = False
    fields: list[str] | None = None
    with path.open("r", encoding="utf-8-sig") as source:
        for raw_line in source:
            line = raw_line.rstrip("\r\n")
            if line == "[Events]":
                in_events = True
                continue
            if in_events and line.startswith("["):
                break
            if not in_events:
                continue
            if line.startswith("Format:"):
                fields = [part.strip() for part in line.split(":", 1)[1].split(",")]
                continue
            if not line.startswith("Dialogue:") or fields is None:
                continue
            values = line.split(":", 1)[1].lstrip().split(",", len(fields) - 1)
            if len(values) != len(fields):
                continue
            item = dict(zip(fields, values))
            start_ass = item["Start"].strip()
            end_ass = item["End"].strip()
            events.append(
                AssEvent(
                    event_index=len(events) + 1,
                    start_ass=start_ass,
                    end_ass=end_ass,
                    start=ass_time_to_seconds(start_ass),
                    end=ass_time_to_seconds(end_ass),
                    style=item["Style"].strip(),
                    name=item["Name"].strip(),
                    text=clean_ass_text(item["Text"]),
                )
            )
    return events


def load_dialogue(path: Path) -> list[Dialogue]:
    events = parse_ass_events(path)
    chinese_by_time: dict[tuple[str, str], list[str]] = {}
    for event in events:
        if event.style == "TextCHS" and event.text:
            chinese_by_time.setdefault((event.start_ass, event.end_ass), []).append(event.text)

    result: list[Dialogue] = []
    for event in events:
        if event.style != "TextJP" or not event.text:
            continue
        result.append(
            Dialogue(
                id=len(result) + 1,
                start=event.start,
                end=event.end,
                start_ass=event.start_ass,
                end_ass=event.end_ass,
                name=event.name,
                jp_text=event.text,
                zh_text=" / ".join(
                    chinese_by_time.get((event.start_ass, event.end_ass), [])
                ),
            )
        )
    return result


def _single_match(paths: list[Path], episode: int, kind: str) -> Path:
    if len(paths) != 1:
        rendered = "\n".join(f"  - {path}" for path in paths) or "  (none)"
        raise RuntimeError(
            f"Expected exactly one {kind} source for EP{episode:02d}; found "
            f"{len(paths)}:\n{rendered}"
        )
    return paths[0]


def find_episode_sources(
    episode: int, source_root: Path | None = None
) -> tuple[Path, Path]:
    source_root = configured_source_root() if source_root is None else source_root
    tag = f"{episode:02d}"
    mkvs = sorted(
        path
        for path in source_root.glob("*.mkv")
        if re.search(rf"\[{tag}\]", path.name)
    )
    subtitles = sorted(
        path
        for path in source_root.glob("*.ass")
        if re.search(rf"\b{tag}\b", path.name)
    )
    return (
        _single_match(mkvs, episode, "MKV"),
        _single_match(subtitles, episode, "ASS"),
    )


def write_dialogue_tsv(dialogue: list[Dialogue], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.writer(target, delimiter="\t")
        writer.writerow(["id", "start", "end", "name", "jp_text", "zh_text"])
        for item in dialogue:
            writer.writerow(
                [
                    f"JP{item.id:03d}",
                    item.start_ass,
                    item.end_ass,
                    item.name,
                    item.jp_text,
                    item.zh_text,
                ]
            )

