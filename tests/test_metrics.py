"""Metric tests. The AP expectations are hand-computed so they cannot drift."""

from __future__ import annotations

import pytest

from signlab.metrics import (
    COCO_IOU_THRESHOLDS,
    Box,
    Detection,
    GroundTruth,
    average_precision,
    evaluate,
    iou,
)


def gt(cls: int, x1: float, y1: float, x2: float, y2: float) -> GroundTruth:
    return GroundTruth(cls, Box(x1, y1, x2, y2))


def det(cls: int, score: float, x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(cls, Box(x1, y1, x2, y2), score)


def test_iou_identical_boxes_is_one():
    assert iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_half_overlap():
    # intersection 25, union 175
    assert iou(Box(0, 0, 10, 10), Box(5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_iou_disjoint_is_zero():
    assert iou(Box(0, 0, 10, 10), Box(20, 20, 30, 30)) == 0.0


def test_box_yolo_round_trip():
    box = Box.from_yolo(0.5, 0.5, 0.4, 0.2, img_w=200, img_h=100)
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((60.0, 40.0, 140.0, 60.0))
    assert box.to_yolo(200, 100) == pytest.approx((0.5, 0.5, 0.4, 0.2))


def test_perfect_detection_scores_one():
    ap, precision, recall = average_precision([det(0, 0.9, 0, 0, 10, 10)], [gt(0, 0, 0, 10, 10)], 0.5)
    assert ap == pytest.approx(1.0)
    assert precision[-1] == pytest.approx(1.0)
    assert recall[-1] == pytest.approx(1.0)


def test_highest_scoring_false_positive_halves_ap():
    """FP at 0.95, TP at 0.90 -> precision never exceeds 0.5, so AP = 0.5."""
    detections = [det(0, 0.95, 50, 50, 60, 60), det(0, 0.90, 0, 0, 10, 10)]
    ap, _, _ = average_precision(detections, [gt(0, 0, 0, 10, 10)], 0.5)
    assert ap == pytest.approx(0.5)


def test_duplicate_detection_scores_one_but_counts_as_extra():
    """The second identical box is a false positive: AP stays 1.0, detections are 2."""
    detections = [det(0, 0.9, 0, 0, 10, 10), det(0, 0.8, 0, 0, 10, 10)]
    ap, _, _ = average_precision(detections, [gt(0, 0, 0, 10, 10)], 0.5)
    assert ap == pytest.approx(1.0)


def test_wrong_class_never_matches():
    ap, _, _ = average_precision([det(1, 0.9, 0, 0, 10, 10)], [gt(0, 0, 0, 10, 10)], 0.5)
    assert ap == 0.0


def test_ap_drops_when_iou_threshold_rises():
    # IoU is 70/130 = 0.538: enough for IoU 0.50, not enough for 0.75.
    detections = [det(0, 0.9, 3, 0, 13, 10)]
    truths = [gt(0, 0, 0, 10, 10)]
    assert iou(Box(3, 0, 13, 10), Box(0, 0, 10, 10)) == pytest.approx(70 / 130)
    loose, _, _ = average_precision(detections, truths, 0.5)
    tight, _, _ = average_precision(detections, truths, 0.75)
    assert loose == pytest.approx(1.0)
    assert tight == 0.0


def test_recall_reaches_one_with_enough_detections():
    detections = [det(0, 0.9, 0, 0, 10, 10), det(0, 0.8, 100, 100, 110, 110)]
    truths = [gt(0, 0, 0, 10, 10), gt(0, 100, 100, 110, 110)]
    ap, _, recall = average_precision(detections, truths, 0.5)
    assert ap == pytest.approx(1.0)
    assert recall[-1] == pytest.approx(1.0)


def test_evaluate_reports_map_over_thresholds():
    per_image = {
        "a": ([det(0, 0.9, 0, 0, 10, 10)], [gt(0, 0, 0, 10, 10)]),
        "b": ([det(1, 0.8, 0, 0, 10, 10)], [gt(1, 0, 0, 10, 10)]),
    }
    report = evaluate(per_image)
    assert report.class_ids == [0, 1]
    assert report.num_images == 2
    assert report.num_ground_truth == 2
    assert report.map50 == pytest.approx(1.0)
    assert report.map50_95 == pytest.approx(1.0)
    assert report.mean_precision == pytest.approx(1.0)


def test_evaluate_missed_class_scores_zero():
    per_image = {"a": ([det(0, 0.9, 0, 0, 10, 10)], [gt(0, 0, 0, 10, 10), gt(1, 50, 50, 60, 60)])}
    report = evaluate(per_image)
    assert report.ap50_per_class[1] == 0.0
    assert report.map50 == pytest.approx(0.5)


def test_evaluate_can_restrict_class_ids():
    per_image = {
        "a": ([det(0, 0.9, 0, 0, 10, 10)], [gt(0, 0, 0, 10, 10)]),
        "b": ([det(1, 0.1, 0, 0, 10, 10)], [gt(1, 0, 0, 10, 10)]),
    }
    report = evaluate(per_image, class_ids=[0])
    assert report.class_ids == [0]
    assert report.map50 == pytest.approx(1.0)


def test_evaluate_on_empty_input_is_zero_not_a_crash():
    report = evaluate({})
    assert report.map50 == 0.0
    assert report.num_images == 0


def test_coco_thresholds_cover_50_to_95():
    assert COCO_IOU_THRESHOLDS[0] == pytest.approx(0.5)
    assert COCO_IOU_THRESHOLDS[-1] == pytest.approx(0.95)
    assert len(COCO_IOU_THRESHOLDS) == 10


def test_report_render_and_dict_agree():
    per_image = {"a": ([det(0, 0.9, 0, 0, 10, 10)], [gt(0, 0, 0, 10, 10)])}
    report = evaluate(per_image)
    payload = report.to_dict()
    assert payload["map50"] == pytest.approx(1.0)
    assert "mAP@50" in report.render()
    assert "class" in report.render()