"""Generate the synthetic demo dataset used by the README.

Real YOLO exports are not redistributable, so the demo ships a generator instead:
run it to rebuild a tiny but structurally valid project (images, labels, classes.yaml)
plus a predictions file with deliberate mistakes so the mAP numbers are not 1.0.

    python examples/make_demo.py examples/demo

Everything is written with the standard library only - PNGs are assembled with zlib.
"""

from __future__ import annotations

import json
import random
import struct
import sys
import zlib
from pathlib import Path

CLASSES = ["stop", "speed-limit", "pedestrian-crossing"]
SIZE = (160, 120)
# class id -> (centre x, centre y, width, height) in normalised coordinates
LAYOUT = {
    0: (0.30, 0.55, 0.18, 0.26),
    1: (0.72, 0.40, 0.22, 0.22),
    2: (0.55, 0.80, 0.14, 0.14),
}
SPLIT_PLAN = {"train": [0, 1, 2, 0, 2, 1], "val": [0, 1, 2]}


def write_png(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload))
        )

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def to_pixels(cx: float, cy: float, w: float, h: float, width: int, height: int) -> list[float]:
    return [
        round((cx - w / 2) * width, 1),
        round((cy - h / 2) * height, 1),
        round((cx + w / 2) * width, 1),
        round((cy + h / 2) * height, 1),
    ]


def jitter(box: list[float], rng: random.Random, amount: float) -> list[float]:
    return [round(v + rng.uniform(-amount, amount), 1) for v in box]


def main(target: str) -> int:
    out = Path(target)
    rng = random.Random(7)
    width, height = SIZE
    truth: dict[str, tuple[int, list[float]]] = {}
    counter = 0

    for split, plan in SPLIT_PLAN.items():
        for index, class_id in enumerate(plan):
            stem = f"{split}_{index:03d}"
            box = to_pixels(*LAYOUT[class_id], width, height)
            # Fill advances across the whole dataset: byte-identical images in two splits
            # are train/val leakage, and the validator is right to reject them.
            write_png(
                out / "images" / split / f"{stem}.png",
                width,
                height,
                (30 + counter * 9, 60 + counter * 5, 200 - counter * 7),
            )
            counter += 1
            label = out / "labels" / split / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            cx = (box[0] + box[2]) / 2 / width
            cy = (box[1] + box[3]) / 2 / height
            bw = (box[2] - box[0]) / width
            bh = (box[3] - box[1]) / height
            label.write_text(
                f"{class_id} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}\n", encoding="utf-8"
            )
            truth[stem] = (class_id, box)

    (out / "classes.yaml").write_text(
        "nc: 3\nnames: [" + ", ".join(CLASSES) + "]\n", encoding="utf-8"
    )

    # Predictions: every true box is found with a little jitter, one object per split is
    # missed and one false positive is added, so the report shows a realistic score.
    predictions = []
    for split in SPLIT_PLAN:
        stems = sorted(s for s in truth if s.startswith(split))
        for position, stem in enumerate(stems):
            detections = []
            if position != 1:  # deliberate miss
                class_id, box = truth[stem]
                detections.append(
                    {
                        "class_id": class_id,
                        "bbox": jitter(box, rng, 3.0),
                        "score": round(0.91 - position * 0.07, 3),
                    }
                )
            if position == 0:
                detections.append(
                    {"class_id": 2, "bbox": [10.0, 10.0, 40.0, 38.0], "score": 0.42}
                )
            predictions.append({"image": f"{stem}.png", "detections": detections})

    pred_path = out / "predictions.jsonl"
    pred_path.write_text(
        "\n".join(json.dumps(p) for p in predictions) + "\n", encoding="utf-8"
    )

    print(f"wrote {out} ({len(truth)} images, {len(CLASSES)} classes) and {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "examples/demo"))