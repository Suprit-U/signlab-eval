"""YOLO-format dataset reading without any imaging dependency.

Image dimensions come from the file header (PNG, JPEG, BMP, GIF, WEBP), so a dataset can be
validated on a machine with no Pillow, no torch and no OpenCV.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import Box, GroundTruth


class DatasetError(Exception):
    """Raised when a dataset cannot be read at all."""


# --------------------------------------------------------------------------- image size


def read_image_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` from the image header. Raises ``DatasetError``."""
    with open(path, "rb") as fh:
        header = fh.read(32)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
        if header[:6] in (b"GIF87a", b"GIF89a"):
            width, height = struct.unpack("<HH", header[6:10])
            return int(width), int(height)
        if header[:2] == b"BM":
            fh.seek(18)
            width, height = struct.unpack("<ii", fh.read(8))
            return int(width), abs(int(height))
        if header[:2] == b"\xff\xd8":
            return _jpeg_size(fh)
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return _webp_size(fh, header)
    raise DatasetError(f"unsupported or unreadable image header: {path}")


def _jpeg_size(fh) -> tuple[int, int]:
    fh.seek(2)
    while True:
        byte = fh.read(1)
        if not byte:
            raise DatasetError("truncated JPEG")
        if byte != b"\xff":
            continue
        marker = fh.read(1)
        while marker == b"\xff":
            marker = fh.read(1)
        code = marker[0]
        if code in (0xD8, 0x01) or 0xD0 <= code <= 0xD7:
            continue
        (length,) = struct.unpack(">H", fh.read(2))
        if code in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            fh.read(1)
            height, width = struct.unpack(">HH", fh.read(4))
            return int(width), int(height)
        fh.seek(length - 2, 1)


def _webp_size(fh, header: bytes) -> tuple[int, int]:
    chunk = header[12:16]
    if chunk == b"VP8X":
        fh.seek(24)
        data = fh.read(6)
        width = int.from_bytes(data[0:3], "little") + 1
        height = int.from_bytes(data[3:6], "little") + 1
        return width, height
    if chunk == b"VP8 ":
        fh.seek(26)
        data = fh.read(8)
        return (
            int.from_bytes(data[6:8], "little") & 0x3FFF,
            int.from_bytes(data[4:6], "little") & 0x3FFF,
        )
    raise DatasetError("unsupported WEBP variant")


# --------------------------------------------------------------------------- classes


_NAMES_INLINE = re.compile(r"^\s*names\s*:\s*\[(?P<body>.*)\]\s*$", re.MULTILINE)
_NAMES_MAPPING = re.compile(r"^\s*names\s*:\s*$", re.MULTILINE)
_NC = re.compile(r"^\s*nc\s*:\s*(?P<n>\d+)\s*$", re.MULTILINE)


def load_class_names(path: Path) -> list[str]:
    """Parse the small YAML subset Ultralytics uses for ``names``.

    Accepts ``names: [a, b]``, an indexed mapping, ``{"names": [...]}`` JSON, or a plain
    newline-separated text file. Never raises on a missing file; callers decide.
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    if text[0] in "{[":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}: invalid JSON ({exc})") from exc
        names = data.get("names", data) if isinstance(data, dict) else data
        return [str(n) for n in names]

    inline = _NAMES_INLINE.search(text)
    if inline:
        body = inline.group("body")
        return [part.strip().strip("'\"") for part in body.split(",") if part.strip()]

    if _NAMES_MAPPING.search(text):
        names: dict[int, str] = {}
        for line in text.splitlines()[1:]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if match:
                names[int(match.group(1))] = match.group(2).strip().strip("'\"")
        if names:
            return [names[i] for i in sorted(names)]
        return []

    if _NC.search(text):
        return []

    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


# --------------------------------------------------------------------------- labels


@dataclass
class LabelIssue:
    line: int
    text: str
    problem: str


@dataclass
class ImageSample:
    """One image and its labels, with everything needed to judge its quality."""

    image_id: str
    image_path: Path
    label_path: Path | None
    width: int = 0
    height: int = 0
    boxes: list[Box] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    issues: list[LabelIssue] = field(default_factory=list)
    error: str | None = None

    @property
    def area_pixels(self) -> float:
        return float(self.width * self.height)

    def ground_truth(self) -> list[GroundTruth]:
        return [GroundTruth(cid, box) for cid, box in zip(self.class_ids, self.boxes)]

    @property
    def is_background(self) -> bool:
        """A label file that exists but is empty is a valid negative sample."""
        return self.label_path is not None and not self.issues and not self.class_ids and not self.error


def parse_label(text: str, width: int, height: int) -> tuple[list[int], list[Box], list[LabelIssue]]:
    """Parse YOLO label text into class ids and pixel boxes."""
    ids: list[int] = []
    boxes: list[Box] = []
    issues: list[LabelIssue] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(LabelIssue(number, raw, f"expected 5 fields, found {len(parts)}"))
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            issues.append(LabelIssue(number, raw, "non-numeric field"))
            continue
        if cid < 0:
            issues.append(LabelIssue(number, raw, "negative class id"))
            continue
        if w <= 0 or h <= 0:
            issues.append(LabelIssue(number, raw, "zero or negative box size"))
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            issues.append(LabelIssue(number, raw, "coordinates outside [0, 1]"))
            continue
        ids.append(cid)
        boxes.append(Box.from_yolo(cx, cy, w, h, width, height))
    return ids, boxes, issues


def load_sample(image_path: Path, label_path: Path | None) -> ImageSample:
    sample = ImageSample(image_path.stem, image_path, label_path)
    try:
        sample.width, sample.height = read_image_size(image_path)
    except DatasetError as exc:
        sample.error = str(exc)
        return sample
    if label_path is None:
        return sample
    if not label_path.exists():
        sample.error = "missing label file"
        return sample
    text = label_path.read_text(encoding="utf-8", errors="replace")
    ids, boxes, issues = parse_label(text, sample.width, sample.height)
    sample.class_ids, sample.boxes, sample.issues = ids, boxes, issues
    return sample


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")


def find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


@dataclass
class Split:
    name: str
    samples: list[ImageSample] = field(default_factory=list)

    @property
    def ground_truth_count(self) -> int:
        return sum(len(s.class_ids) for s in self.samples)


@dataclass
class Dataset:
    """A YOLO project directory: ``classes.yaml`` plus one or more splits."""

    root: Path
    class_names: list[str]
    splits: dict[str, Split] = field(default_factory=dict)
    orphans: list[Path] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def split(self, name: str) -> Split:
        if name not in self.splits:
            raise DatasetError(f"unknown split '{name}' (have: {', '.join(sorted(self.splits)) or 'none'})")
        return self.splits[name]

    @property
    def class_ids(self) -> list[int]:
        return list(range(self.num_classes))


def _locate_split(root: Path, split: str) -> tuple[Path, Path]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if image_dir.is_dir():
        return image_dir, label_dir
    flat_images = root / "images"
    flat_labels = root / "labels"
    # Only assume a flat layout when no split directories exist at all. Otherwise a
    # missing split such as "test" would silently resolve to the flat directory and be
    # reported as an empty split instead of being flagged as absent.
    has_split_dirs = any((flat_images / name).is_dir() for name in ("train", "val", "test", "valid"))
    if flat_images.is_dir() and not has_split_dirs:
        return flat_images, flat_labels
    raise DatasetError(f"no images directory for split '{split}' (looked in {image_dir} and {flat_images})")


def load_dataset(root: Path, splits: list[str] | None = None) -> Dataset:
    """Load a YOLO project directory. Raises ``DatasetError`` when the root is unusable."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"dataset root does not exist: {root}")

    names_path = next((p for p in (root / "classes.yaml", root / "classes.yml", root / "classes.json") if p.exists()), None)
    class_names = load_class_names(names_path) if names_path else []
    dataset = Dataset(root=root, class_names=class_names)

    wanted = splits or ["train", "val", "test"]
    for split in wanted:
        try:
            image_dir, label_dir = _locate_split(root, split)
        except DatasetError:
            continue
        collected = Split(name=split)
        for image_path in find_images(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            collected.samples.append(load_sample(image_path, label_path if label_dir.is_dir() else None))
        dataset.splits[split] = collected

        if label_dir.is_dir():
            image_stems = {p.stem for p in find_images(image_dir)}
            dataset.orphans.extend(p for p in sorted(label_dir.glob("*.txt")) if p.stem not in image_stems)

    if not dataset.splits:
        raise DatasetError(f"no usable splits found under {root} (expected images/train or images/)")
    return dataset
