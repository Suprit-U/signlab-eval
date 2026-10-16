"""Where detections come from.

The evaluator does not care who produced a box, only what shape the answer has. That keeps
the metric code independent of any inference framework and lets CI run on pre-computed
predictions with no GPU and no torch installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .metrics import Box, Detection


@runtime_checkable
class Predictor(Protocol):
    """Anything that can turn an image into scored boxes."""

    def predict(self, image_path: Path) -> list[Detection]:  # pragma: no cover - protocol
        ...


@dataclass
class JsonlPredictor:
    """Predictions read from a JSON Lines file.

    One object per line::

        {"image": "frame_0007.jpg", "detections": [
            {"class_id": 0, "bbox": [x1, y1, x2, y2], "score": 0.91},
            {"class_id": 2, "bbox": [x1, y1, x2, y2], "score": 0.55, "confidence": 0.55}
        ]}

    Boxes are pixel coordinates. ``confidence`` is accepted as an alias for ``score`` because
    that is what most inference exports call it.
    """

    by_image: dict[str, list[Detection]]

    @classmethod
    def from_file(cls, path: Path) -> "JsonlPredictor":
        by_image: dict[str, list[Detection]] = {}
        with open(path, encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON ({exc})") from exc
                name = record.get("image") or record.get("id") or record.get("file")
                if not name:
                    raise ValueError(f"{path}:{number}: record has no 'image' key")
                boxes: list[Detection] = []
                for item in record.get("detections", []):
                    score = item.get("score", item.get("confidence", 1.0))
                    boxes.append(
                        Detection(
                            class_id=int(item["class_id"]),
                            box=Box(*[float(v) for v in item["bbox"]]),
                            score=float(score),
                        )
                    )
                by_image[Path(str(name)).stem] = boxes
        return cls(by_image=by_image)

    def predict(self, image_path: Path) -> list[Detection]:
        return self.by_image.get(image_path.stem, [])


class UltralyticsPredictor:
    """Runs a YOLO checkpoint through Ultralytics, if it is installed."""

    def __init__(self, weights: str | Path, *, device: str | None = None, conf: float = 0.001):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ultralytics is not installed. Install it with: pip install 'signlab-eval[ultralytics]', "
                "or pass --predictions with pre-computed detections."
            ) from exc
        self._model = YOLO(str(weights))
        self._device = device
        self._conf = conf

    def predict(self, image_path: Path) -> list[Detection]:  # pragma: no cover - needs a checkpoint
        result = self._model.predict(source=str(image_path), conf=self._conf, device=self._device, verbose=False)[0]
        detections: list[Detection] = []
        for xyxy, conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            detections.append(Detection(class_id=int(cls), box=Box(*xyxy), score=float(conf)))
        return detections


def load_predictions(path: Path) -> JsonlPredictor:
    return JsonlPredictor.from_file(path)
