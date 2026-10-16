"""Dataset loading and validation checks, exercised against real files on disk."""

from __future__ import annotations

from pathlib import Path

import pytest

from signlab.dataset import DatasetError, load_dataset, load_class_names, parse_label, read_image_size
from signlab.validate import find_duplicates, hamming, perceptual_hash, validate_dataset
from conftest import write_label, write_png, yolo_box


# --------------------------------------------------------------------- loading


def test_read_image_size_without_pillow(dataset_root: Path):
    write_png(dataset_root / "probe.png", 64, 48)
    assert read_image_size(dataset_root / "probe.png") == (64, 48)


def test_load_dataset_finds_splits_and_classes(dataset):
    assert sorted(dataset.splits) == ["train", "val"]
    assert dataset.num_classes == 3
    assert dataset.split("train").ground_truth_count == 6
    assert dataset.split("val").ground_truth_count == 3


def test_unknown_split_is_a_clear_error(dataset):
    with pytest.raises(DatasetError, match="unknown split"):
        dataset.split("holdout")


def test_missing_root_raises(tmp_path: Path):
    with pytest.raises(DatasetError, match="does not exist"):
        load_dataset(tmp_path / "nope")


def test_root_without_images_raises(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetError, match="no usable splits"):
        load_dataset(tmp_path / "empty")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("names: [stop, speed]", ["stop", "speed"]),
        ("names:\n  0: stop\n  1: speed\n", ["stop", "speed"]),
        ('{"names": ["stop", "speed"]}', ["stop", "speed"]),
        ("stop\nspeed\n", ["stop", "speed"]),
    ],
)
def test_class_name_formats(tmp_path: Path, text: str, expected: list[str]):
    path = tmp_path / "classes.yaml"
    path.write_text(text, encoding="utf-8")
    assert load_class_names(path) == expected


# ---------------------------------------------------------------------- labels


def test_parse_label_rejects_wrong_field_count():
    _, boxes, issues = parse_label("0 0.5 0.5 0.2\n", 100, 100)
    assert issues and "expected 5 fields" in issues[0].problem
    assert boxes == []


def test_parse_label_rejects_out_of_range_coordinates():
    _, _, issues = parse_label("0 1.4 0.5 0.2 0.2\n", 100, 100)
    assert issues and "outside [0, 1]" in issues[0].problem


def test_parse_label_converts_to_pixels():
    ids, boxes, issues = parse_label("1 " + yolo_box(0.5, 0.5, 0.4, 0.2) + "\n", 200, 100)
    assert ids == [1] and not issues
    assert boxes[0].x1 == pytest.approx(60.0)
    assert boxes[0].y2 == pytest.approx(60.0)


def test_empty_label_file_is_a_background_sample(dataset_root: Path):
    write_label(dataset_root / "labels" / "val" / "val_001.txt", [])
    dataset = load_dataset(dataset_root)
    sample = next(s for s in dataset.split("val").samples if s.image_id == "val_001")
    assert sample.is_background


# ------------------------------------------------------------------ validation


def test_clean_dataset_passes(dataset):
    report = validate_dataset(dataset)
    assert report.ok, report.render()
    assert report.stats["images"] == 6
    assert report.stats["boxes"] == 9
    assert "train-val-overlap" not in report.codes()


def test_missing_label_file_is_an_error(dataset_root: Path):
    (dataset_root / "labels" / "train" / "train_001.txt").unlink()
    report = validate_dataset(load_dataset(dataset_root))
    assert not report.ok
    assert report.codes()["missing-label"] == 1


def test_class_id_outside_the_class_file_is_an_error(dataset_root: Path):
    write_label(dataset_root / "labels" / "val" / "val_000.txt", ["9 " + yolo_box(0.5, 0.5, 0.2, 0.2)])
    report = validate_dataset(load_dataset(dataset_root))
    assert not report.ok
    assert report.codes()["class-out-of-range"] == 1


def test_malformed_label_line_is_an_error(dataset_root: Path):
    write_label(dataset_root / "labels" / "val" / "val_000.txt", ["0 0.5 0.5", "1 " + yolo_box(0.5, 0.5, 0.2, 0.2)])
    report = validate_dataset(load_dataset(dataset_root))
    assert report.codes()["malformed-label"] == 1


def test_tiny_box_is_a_warning(dataset_root: Path):
    write_label(dataset_root / "labels" / "val" / "val_000.txt", ["0 " + yolo_box(0.5, 0.5, 0.01, 0.01)])
    report = validate_dataset(load_dataset(dataset_root))
    assert "tiny-box" in report.codes()
    assert report.ok  # warnings do not fail the run


def test_duplicate_image_across_splits_is_an_error(dataset_root: Path):
    source = (dataset_root / "images" / "train" / "train_000.png").read_bytes()
    (dataset_root / "images" / "val" / "val_000.png").write_bytes(source)
    report = validate_dataset(load_dataset(dataset_root))
    assert report.codes().get("duplicate-image") == 1


def test_same_image_id_in_two_splits_is_an_error(dataset_root: Path):
    source = (dataset_root / "images" / "train" / "train_000.png").read_bytes()
    (dataset_root / "images" / "val" / "train_000.png").write_bytes(source)
    report = validate_dataset(load_dataset(dataset_root))
    assert report.codes().get("train-val-overlap") == 1


def test_orphan_label_file_is_a_warning(dataset_root: Path):
    write_label(dataset_root / "labels" / "val" / "ghost.txt", ["0 " + yolo_box(0.5, 0.5, 0.2, 0.2)])
    report = validate_dataset(load_dataset(dataset_root))
    assert report.codes().get("orphan-label") == 1


def test_missing_val_split_is_reported(tmp_path: Path):
    root = tmp_path / "trainonly"
    write_png(root / "images" / "train" / "a.png", 40, 40)
    write_label(root / "labels" / "train" / "a.txt", ["0 " + yolo_box(0.5, 0.5, 0.4, 0.4)])
    write_label(root / "classes.yaml", ["names: [stop]"])
    report = validate_dataset(load_dataset(root))
    assert report.codes().get("missing-val-split") == 1


def test_class_without_examples_is_reported(dataset_root: Path):
    # Rewrite *every* label in the fixture to class 0, so classes 1 and 2 have no examples.
    for split, count in (("train", 4), ("val", 2)):
        for index in range(count):
            write_label(dataset_root / "labels" / split / f"{split}_{index:03d}.txt", ["0 " + yolo_box(0.5, 0.5, 0.3, 0.3)])
    report = validate_dataset(load_dataset(dataset_root))
    assert report.codes().get("class-without-examples") == 2


def test_class_imbalance_is_reported(dataset_root: Path):
    for index in range(6, 28):
        write_png(dataset_root / "images" / "train" / f"train_{index:03d}.png", 20, 20)
        write_label(dataset_root / "labels" / "train" / f"train_{index:03d}.txt", ["0 " + yolo_box(0.5, 0.5, 0.5, 0.5)])
    report = validate_dataset(load_dataset(dataset_root), imbalance_ratio=5.0)
    assert report.codes().get("class-imbalance") == 1


def test_missing_classes_file_is_an_error(tmp_path: Path):
    root = tmp_path / "noclasses"
    write_png(root / "images" / "train" / "a.png", 20, 20)
    write_label(root / "labels" / "train" / "a.txt", ["0 " + yolo_box(0.5, 0.5, 0.5, 0.5)])
    write_png(root / "images" / "val" / "a.png", 20, 20)
    write_label(root / "labels" / "val" / "a.txt", ["0 " + yolo_box(0.5, 0.5, 0.5, 0.5)])
    report = validate_dataset(load_dataset(root))
    assert not report.ok
    assert report.codes().get("missing-classes") == 1


# ------------------------------------------------------------------- leakage


def test_perceptual_hash_is_stable_and_distinguishes(tmp_path: Path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    write_png(a, 32, 32, (10, 20, 30))
    a_bytes = a.read_bytes()
    b.write_bytes(a_bytes)
    assert perceptual_hash(a) == perceptual_hash(b)
    write_png(tmp_path / "c.png", 32, 32, (200, 10, 10))
    assert perceptual_hash(tmp_path / "c.png") != perceptual_hash(a)


def test_hamming_distance():
    # Pixel hashes are real bit vectors, so distance is meaningful.
    assert hamming("px:0000", "px:0001") == 1
    assert hamming("px:0000", "px:0000") == 0
    # Raw byte hashes carry no bit structure: equal means identical, different is maximal.
    assert hamming("raw:abcd", "raw:abcd") == 0
    assert hamming("raw:abcd", "raw:abce") == 1 << 30
    assert hamming("", "px:0000") == 1 << 30


def test_find_duplicates_maps_copy_to_original(dataset_root: Path):
    dataset = load_dataset(dataset_root)
    copy = dataset_root / "images" / "val" / "val_001.png"
    copy.write_bytes((dataset_root / "images" / "train" / "train_000.png").read_bytes())
    dataset = load_dataset(dataset_root)
    # The scan has to span both splits - a copy in val of a train image is only visible
    # when the train hashes are in the pool as well.
    samples = dataset.split("train").samples + dataset.split("val").samples
    duplicates = find_duplicates(samples)
    assert "val_001" in duplicates
    assert duplicates["val_001"] == "train_000"


def test_validate_report_serialises(dataset):
    payload = validate_dataset(dataset).to_dict()
    assert payload["ok"] is True
    assert "findings" in payload and "stats" in payload


def test_validate_report_render_sorts_errors_first(dataset_root: Path):
    (dataset_root / "labels" / "train" / "train_001.txt").unlink()
    report = validate_dataset(load_dataset(dataset_root))
    text = report.render()
    assert "missing-label" in text
    assert "1 error(s), 0 warning(s)" in text
    # Errors are listed before warnings regardless of the order findings were added in.
    severities = [f.severity for f in report.findings]
    assert severities == sorted(severities, key=lambda sev: ["error", "warning", "info"].index(sev))
    # Summary, per-code counts and stats all follow the findings block.
    body = [line for line in text.splitlines() if line.strip()]
    assert body[0].startswith("ERROR")
    assert body.index("1 error(s), 0 warning(s)") > 0
    assert body[-1].startswith("splits=")