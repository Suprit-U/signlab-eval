"""Dataset quality checks.

Each check returns ``Finding`` objects with a stable code, a severity and the path it came
from, so a CI job can fail on the findings that actually matter instead of on a wall of text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .dataset import Dataset, ImageSample, Split
from .metrics import iou

ERROR = "error"
WARNING = "warning"
INFO = "info"

SEVERITY_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str
    path: str = ""

    def __str__(self) -> str:
        where = f" [{self.path}]" if self.path else ""
        return f"{self.severity.upper():7} {self.code}{where}: {self.message}"


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, float | int] = field(default_factory=dict)

    def add(self, code: str, severity: str, message: str, path: str | Path = "") -> None:
        self.findings.append(Finding(code, severity, str(message), str(path)))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    def codes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.code] = counts.get(f.code, 0) + 1
        return counts

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [str(f) for f in sorted(self.findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.code))]
        header = ", ".join(f"{code}={count}" for code, count in sorted(self.codes().items()))
        lines.append("")
        lines.append(f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        if header:
            lines.append(header)
        if self.stats:
            lines.append(", ".join(f"{k}={v}" for k, v in self.stats.items()))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "findings": [
                {"code": f.code, "severity": f.severity, "message": f.message, "path": f.path}
                for f in self.findings
            ],
            "stats": self.stats,
        }


# --------------------------------------------------------------------------- leakage


def perceptual_hash(image_path: Path, size: int = 8) -> str:
    """dHash over the image, using Pillow when available.

    Without Pillow the hash is computed from the raw bytes of the file, which is enough to
    catch byte-identical duplicates and is explicitly reported as such by the caller.
    """
    data = Path(image_path).read_bytes()
    try:
        from PIL import Image  # type: ignore
        import io

        with Image.open(io.BytesIO(data)) as img:
            small = img.convert("L").resize((size + 1, size))
            pixels = list(small.getdata())
    except Exception:
        return "raw:" + _bytes_hash(data)
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"px:{bits:0{size * size}b}"


def _bytes_hash(data: bytes) -> str:
    import hashlib

    return hashlib.blake2b(data, digest_size=16).hexdigest()


def hamming(a: str, b: str) -> int:
    """Bit distance between two hashes. Different prefixes count as maximal distance."""
    if not a or not b:
        return 1 << 30
    if a.startswith("raw:") or b.startswith("raw:"):
        return 0 if a == b else 1 << 30
    return bin(int(a.split(":")[1], 2) ^ int(b.split(":")[1], 2)).count("1")


def find_duplicates(samples: list[ImageSample], *, max_distance: int = 2) -> dict[str, str]:
    """Return ``{image_id: other_image_id}`` for near-duplicate images."""
    hashes = {s.image_id: perceptual_hash(s.image_path) for s in samples if s.error is None}
    seen: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    for image_id, digest in hashes.items():
        if digest in seen:
            duplicates[image_id] = seen[digest]
            continue
        seen[digest] = image_id
    if max_distance <= 0:
        return duplicates
    ids = list(hashes)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if b in duplicates or a in duplicates:
                continue
            if hamming(hashes[a], hashes[b]) <= max_distance:
                duplicates[b] = a
    return duplicates


# --------------------------------------------------------------------------- checks


def _sample_checks(sample: ImageSample, dataset: Dataset, report: ValidationReport, *, min_box_px: float, max_aspect: float) -> None:
    where = sample.image_path.name

    if sample.error == "missing label file":
        report.add("missing-label", ERROR, "image has no label file", where)
        return
    if sample.error and sample.error != "missing label file":
        report.add("unreadable-image", ERROR, sample.error, where)
        return
    if sample.label_path is None:
        report.add("unlabelled-image", WARNING, "image is not in a labels directory (treated as unlabelled)", where)
        return

    for issue in sample.issues:
        severity = ERROR if issue.problem != "non-numeric field" else ERROR
        report.add("malformed-label", severity, f"line {issue.line}: {issue.problem}", where)

    for cid in sample.class_ids:
        if dataset.num_classes and cid >= dataset.num_classes:
            name = dataset.class_names[cid] if cid < len(dataset.class_names) else "?"
            report.add("class-out-of-range", ERROR, f"class id {cid} is not defined in classes file", where)
            del name

    for cid, box in zip(sample.class_ids, sample.boxes):
        width, height = box.width, box.height
        if min(width, height) < min_box_px:
            report.add("tiny-box", WARNING, f"class {cid} box is {width:.1f}x{height:.1f}px", where)
        longer, shorter = max(width, height), max(min(width, height), 1e-6)
        if longer / shorter > max_aspect:
            report.add("extreme-aspect", WARNING, f"class {cid} aspect ratio {longer / shorter:.1f}", where)
        if box.x1 < -1 or box.y1 < -1 or box.x2 > sample.width + 1 or box.y2 > sample.height + 1:
            report.add("box-outside-image", WARNING, f"class {cid} box extends past the image edge", where)

    for i in range(len(sample.boxes)):
        for j in range(i + 1, len(sample.boxes)):
            if sample.class_ids[i] == sample.class_ids[j] and iou(sample.boxes[i], sample.boxes[j]) > 0.95:
                report.add("duplicate-box", WARNING, f"class {sample.class_ids[i]} box repeated with IoU > 0.95", where)


def validate_dataset(
    dataset: Dataset,
    *,
    min_box_px: float = 4.0,
    max_aspect: float = 20.0,
    check_leakage: bool = True,
    max_distance: int = 2,
    imbalance_ratio: float = 20.0,
    require_val: bool = True,
) -> ValidationReport:
    """Run every check and return a report. ``report.ok`` is False on any error."""
    report = ValidationReport()

    if not dataset.class_names:
        report.add("missing-classes", ERROR, "no classes file found (classes.yaml) at the dataset root")

    total_boxes = 0
    class_counts: dict[int, int] = {}
    for split_name, split in sorted(dataset.splits.items()):
        if not split.samples:
            report.add("empty-split", ERROR, f"split '{split_name}' has no images", split_name)
            continue
        for sample in split.samples:
            _sample_checks(sample, dataset, report, min_box_px=min_box_px, max_aspect=max_aspect)
            total_boxes += len(sample.class_ids)
            for cid in sample.class_ids:
                class_counts[cid] = class_counts.get(cid, 0) + 1

    for path in dataset.orphans:
        report.add("orphan-label", WARNING, "label file has no matching image", path.name)

    if require_val and "val" not in dataset.splits:
        report.add("missing-val-split", WARNING, "no validation split: model selection cannot be trusted")

    if class_counts and dataset.num_classes:
        missing = [cid for cid in dataset.class_ids if cid not in class_counts]
        for cid in missing:
            report.add("class-without-examples", WARNING, f"class {cid} ({dataset.class_names[cid]}) has no labels anywhere")
        counts = sorted(class_counts.values())
        if len(counts) > 1 and counts[0] > 0 and counts[-1] / counts[0] > imbalance_ratio:
            report.add(
                "class-imbalance",
                WARNING,
                f"rarest class has {counts[0]} label(s), most common has {counts[-1]} "
                f"({counts[-1] / counts[0]:.1f}x)",
            )

    if check_leakage:
        train = dataset.splits.get("train")
        val = dataset.splits.get("val")
        if train and val:
            val_ids = {s.image_id for s in val.samples}
            shared = sorted({s.image_id for s in train.samples} & val_ids)
            for image_id in shared:
                report.add("train-val-overlap", ERROR, f"image present in both train and val: {image_id}", image_id)
            duplicates = find_duplicates(train.samples + val.samples, max_distance=max_distance)
            for image_id, other in sorted(duplicates.items()):
                report.add("duplicate-image", ERROR, f"{image_id} looks like a copy of {other}", image_id)

    report.stats = {
        "splits": len(dataset.splits),
        "images": sum(len(s.samples) for s in dataset.splits.values()),
        "boxes": total_boxes,
        "classes": dataset.num_classes,
        "errors": len(report.errors),
        "warnings": len(report.warnings),
    }
    return report
