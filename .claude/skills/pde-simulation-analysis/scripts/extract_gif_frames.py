#!/usr/bin/env python3
"""Extract representative frames from a GIF for manual inspection."""

import argparse
from pathlib import Path

from PIL import Image, ImageSequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract first, middle, and last frames from a GIF."
    )
    parser.add_argument("gif", type=Path, help="Path to the source GIF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for extracted PNGs. Defaults to <gif stem>_frames next to the GIF.",
    )
    return parser.parse_args()


def _frame_indices(num_frames: int) -> list[tuple[str, int]]:
    if num_frames <= 0:
        raise ValueError("GIF contains no frames")

    labeled = [
        ("start", 0),
        ("middle", num_frames // 2),
        ("end", num_frames - 1),
    ]

    deduped: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, index in labeled:
        if index in seen:
            continue
        deduped.append((label, index))
        seen.add(index)
    return deduped


def main() -> None:
    args = _parse_args()
    gif_path = args.gif.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else gif_path.with_name(f"{gif_path.stem}_frames")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(gif_path) as image:
        frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(image)]

    for label, index in _frame_indices(len(frames)):
        output_path = output_dir / f"{gif_path.stem}_{label}.png"
        frames[index].save(output_path)
        print(output_path)


if __name__ == "__main__":
    main()
