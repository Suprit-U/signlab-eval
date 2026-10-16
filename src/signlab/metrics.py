"""Object-detection metrics, implemented from the definitions.

No numpy, no torch. If you want to check the implementation, read this file; if you want
to check the number, compare it with COCO's on a known case.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates: ``(x1, y1, x2, y2)``."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_yolo(self, img_w: float, img_h: float) -> tuple[float, float, float, float]:
        """Convert to normalised YOLO ``(cx, cy, w, h)`` for an image of this size."""
        return (
            ((self.x1 + self.x2) / 2.0) / img_w,
            ((self.y1 + self.y2) / 2.0) / img_h,
            self.width / img_w,
            self.height / img_h,
        )

    @classmethod
    def from_yolo(cls, cx: float, cy: float, w: float, h: float, img_w: float, img_h: float) -> "Box":
        half_w, half_h = w * img_w / 2.0, h * img_h / 2.0
        return cls(cx * img_w - half_w, cy * img_h - half_h, cx * img_w + half_w, cy * img_h + half_h)

    def as_iou_format(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    """One predicted box."""

    class_id: int
    box: Box
    score: float


@dataclass
class GroundTruth:
    """One labelled box."""

    class_id: int
    box: Box


@dataclass
class ClassCurve:
    class_id: int
    num_ground_truth: int
    num_detections: int
    thresholds: dict[float, float] = field(default_factory=dict)
    precision: dict[float, list[float]] = field(default_factory=dict)
    recall: dict[float, list[float]] = field(default_factory=dict)


def _match_image(
    detections: list[Detection],
    ground_truth: list[GroundTruth],
    iou_threshold: float,
) -> tuple[list[int], list[int], list[int]]:
    """Greedy score-ordered matching, as used by COCO.

    Returns ``(tp, fp, matched_gt_index)`` aligned with ``detections``.
    """
    order = sorted(range(len(detections)), key=lambda i: detections[i].score, reverse=True)
    used = [False] * len(ground_truth)
    tp = [0] * len(detections)
    fp = [0] * len(detections)
    matched = [-1] * len(detections)
    for di in order:
        det = detections[di]
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(ground_truth):
            if used[gi] or gt.class_id != det.class_id:
                continue
            value = iou(det.box, gt.box)
            if value > best_iou:
                best_iou, best_gi = value, gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            used[best_gi] = True
            matched[di] = best_gi
            tp[di] = 1
        else:
            fp[di] = 1
    return tp, fp, matched


def _pr_to_ap(precision: list[float], recall: list[float]) -> float:
    """All-point interpolated average precision (area under the PR envelope)."""
    if not recall:
        return 0.0
    mrec = [0.0] + list(recall) + [recall[-1]]
    mpre = [0.0] + list(precision) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mrec) - 1))


def average_precision(
    detections: list[Detection],
    ground_truth: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> tuple[float, list[float], list[float]]:
    """AP at one IoU threshold. Returns ``(ap, precision, recall)``."""
    if not ground_truth:
        return 0.0, [], []
    if not detections:
        return 0.0, [], [0.0]
    order = sorted(range(len(detections)), key=lambda i: detections[i].score, reverse=True)
    sorted_det = [detections[i] for i in order]
    tp, fp, _ = _match_image(sorted_det, ground_truth, iou_threshold)
    cum_tp, cum_fp = 0, 0
    precision: list[float] = []
    recall: list[float] = []
    for i in range(len(sorted_det)):
        cum_tp += tp[i]
        cum_fp += fp[i]
        precision.append(cum_tp / (cum_tp + cum_fp))
        recall.append(cum_tp / len(ground_truth))
    return _pr_to_ap(precision, recall), precision, recall


COCO_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


@dataclass
class EvalReport:
    """Aggregated metrics for one evaluation run."""

    class_ids: list[int]
    ap_per_class: dict[int, float]
    ap50_per_class: dict[int, float]
    ap50_95_per_class: dict[int, float]
    precision: dict[int, list[float]]
    recall: dict[int, list[float]]
    ground_truth_per_class: dict[int, int]
    detected_per_class: dict[int, int]
    num_images: int
    num_ground_truth: int

    @property
    def map50(self) -> float:
        values = [v for v in self.ap50_per_class.values()]
        return sum(values) / len(values) if values else 0.0

    @property
    def map50_95(self) -> float:
        values = [v for v in self.ap50_95_per_class.values()]
        return sum(values) / len(values) if values else 0.0

    @property
    def mean_precision(self) -> float:
        values = [v for v in self.ap_per_class.values()]
        return sum(values) / len(values) if values else 0.0

    def to_dict(self) -> dict:
        return {
            "num_images": self.num_images,
            "num_ground_truth": self.num_ground_truth,
            "map50": round(self.map50, 4),
            "map50_95": round(self.map50_95, 4),
            "mean_precision": round(self.mean_precision, 4),
            "per_class": {
                str(cid): {
                    "ap50": round(self.ap50_per_class[cid], 4),
                    "ap50_95": round(self.ap50_95_per_class[cid], 4),
                    "ground_truth": self.ground_truth_per_class[cid],
                    "detections": self.detected_per_class[cid],
                }
                for cid in self.class_ids
            },
        }

    def render(self) -> str:
        lines = [
            f"images={self.num_images}  ground_truth={self.num_ground_truth}",
            f"mAP@50     {self.map50:.4f}",
            f"mAP@50-95  {self.map50_95:.4f}",
            "",
            f"{'class':>6}  {'AP@50':>8}  {'AP@50-95':>9}  {'gt':>6}  {'det':>6}",
        ]
        for cid in self.class_ids:
            lines.append(
                f"{cid:>6}  {self.ap50_per_class[cid]:>8.4f}  "
                f"{self.ap50_95_per_class[cid]:>9.4f}  {self.ground_truth_per_class[cid]:>6}  "
                f"{self.detected_per_class[cid]:>6}"
            )
        return "\n".join(lines)


def evaluate(
    per_image: dict[str, tuple[list[Detection], list[GroundTruth]]],
    class_ids: list[int] | None = None,
) -> EvalReport:
    """Evaluate detections against ground truth, image by image."""
    detections: dict[int, list[Detection]] = {}
    truths: dict[int, list[GroundTruth]] = {}
    for dets, gts in per_image.values():
        for d in dets:
            detections.setdefault(d.class_id, []).append(d)
        for g in gts:
            truths.setdefault(g.class_id, []).append(g)

    ids = sorted(class_ids if class_ids is not None else set(detections) | set(truths))
    ap_per_class: dict[int, float] = {}
    ap50_per_class: dict[int, float] = {}
    ap50_95_per_class: dict[int, float] = {}
    precision: dict[int, list[float]] = {}
    recall: dict[int, list[float]] = {}

    for cid in ids:
        dets = detections.get(cid, [])
        gts = truths.get(cid, [])
        # Matching is per image, so group by image identity rather than pooling.
        by_image: dict[str, tuple[list[Detection], list[GroundTruth]]] = {}
        for name, (image_dets, image_gts) in per_image.items():
            d = [x for x in image_dets if x.class_id == cid]
            g = [x for x in image_gts if x.class_id == cid]
            if d or g:
                by_image[name] = (d, g)
        aps = []
        for threshold in COCO_IOU_THRESHOLDS:
            ap, _, _ = _pooled_ap(by_image, threshold)
            aps.append(ap)
        ap50, prec, rec = _pooled_ap(by_image, 0.5)
        ap_per_class[cid] = sum(aps) / len(aps) if aps else 0.0
        ap50_per_class[cid] = ap50
        ap50_95_per_class[cid] = ap_per_class[cid]
        precision[cid] = prec
        recall[cid] = rec

    return EvalReport(
        class_ids=ids,
        ap_per_class=ap_per_class,
        ap50_per_class=ap50_per_class,
        ap50_95_per_class=ap50_95_per_class,
        precision=precision,
        recall=recall,
        ground_truth_per_class={cid: len(truths.get(cid, [])) for cid in ids},
        detected_per_class={cid: len(detections.get(cid, [])) for cid in ids},
        num_images=len(per_image),
        num_ground_truth=sum(len(items) for items in truths.values()),
    )


def _pooled_ap(
    by_image: dict[str, tuple[list[Detection], list[GroundTruth]]],
    iou_threshold: float,
) -> tuple[float, list[float], list[float]]:
    """AP for one class across all images, with per-image matching."""
    all_dets: list[tuple[str, Detection]] = []
    total_gt = 0
    for name, (dets, gts) in by_image.items():
        total_gt += len(gts)
        all_dets.extend((name, d) for d in dets)
    if total_gt == 0:
        return 0.0, [], []
    if not all_dets:
        return 0.0, [], [0.0]

    all_dets.sort(key=lambda item: item[1].score, reverse=True)
    used = {name: [False] * len(gts) for name, (dets, gts) in by_image.items()}
    tp: list[int] = []
    fp: list[int] = []
    for name, det in all_dets:
        gts = dict(by_image)[name][1]
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gts):
            if used[name][gi] or gt.class_id != det.class_id:
                continue
            value = iou(det.box, gt.box)
            if value > best_iou:
                best_iou, best_gi = value, gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            used[name][best_gi] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    cum_tp = cum_fp = 0
    precision: list[float] = []
    recall: list[float] = []
    for i in range(len(all_dets)):
        cum_tp += tp[i]
        cum_fp += fp[i]
        precision.append(cum_tp / (cum_tp + cum_fp))
        recall.append(cum_tp / total_gt)
    return _pr_to_ap(precision, recall), precision, recall
