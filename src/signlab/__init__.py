"""signlab-eval: dataset QA and reproducible mAP evaluation for YOLO-format datasets."""

from .dataset import Dataset, DatasetError, ImageSample, load_dataset, read_image_size
from .metrics import (
    COCO_IOU_THRESHOLDS,
    Box,
    Detection,
    EvalReport,
    GroundTruth,
    average_precision,
    evaluate,
    iou,
)
from .predictors import JsonlPredictor, load_predictions
from .validate import ValidationReport, validate_dataset
from .version import __version__

__all__ = [
    "__version__",
    "Box",
    "COCO_IOU_THRESHOLDS",
    "Dataset",
    "DatasetError",
    "Detection",
    "EvalReport",
    "GroundTruth",
    "ImageSample",
    "JsonlPredictor",
    "ValidationReport",
    "average_precision",
    "evaluate",
    "iou",
    "load_dataset",
    "load_predictions",
    "read_image_size",
    "validate_dataset",
]
