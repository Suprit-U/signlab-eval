"""Shared fixtures: a synthetic YOLO dataset written to disk with real PNG bytes."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

CLASS_NAMES = ["stop", "speed-limit", "pedestrian"]


def write_png(path: Path, width: int, height: int, rgb: tuple[int, int, int] = (40, 40, 40)) -> Path:
    """Write a minimal but valid PNG without pulling in Pillow."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload))

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_label(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def yolo_box(cx: float, cy: float, w: float, h: float) -> str:
    return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def make_dataset(root: Path, *, size: tuple[int, int] = (100, 80)) -> Path:
    """A tiny but valid two-split dataset: 4 train images, 2 val images."""
    counter = 0
    for split, count in (("train", 4), ("val", 2)):
        for index in range(count):
            stem = f"{split}_{index:03d}"
            # Every image gets its own fill. Identical bytes would hash identically and the
            # leakage detector would (correctly) report every image as a duplicate.
            rgb = (10 + counter * 7, 40 + counter * 11, 200 - counter * 5)
            counter += 1
            write_png(root / "images" / split / f"{stem}.png", *size, rgb=rgb)
            if index % 2 == 0:
                write_label(
                    root / "labels" / split / f"{stem}.txt",
                    ["0 " + yolo_box(0.5, 0.5, 0.4, 0.5), "2 " + yolo_box(0.2, 0.2, 0.2, 0.2)],
                )
            else:
                write_label(root / "labels" / split / f"{stem}.txt", ["1 " + yolo_box(0.5, 0.5, 0.3, 0.3)])
    write_label(root / "classes.yaml", ["names: [" + ", ".join(CLASS_NAMES) + "]"])
    return root


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    return make_dataset(tmp_path / "ds")


@pytest.fixture()
def dataset(dataset_root: Path):
    from signlab.dataset import load_dataset

    return load_dataset(dataset_root)