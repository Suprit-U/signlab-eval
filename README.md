# signlab-eval

Dataset QA and reproducible mAP for YOLO-format detection datasets.

Two commands, one gate:

- `signlab validate` — finds the problems that quietly ruin a detection run: missing labels, class drift, train/val leakage, duplicate images, degenerate boxes.
- `signlab evaluate` — computes AP@50 and AP@50-95 per class from a JSONL predictions file, with `--fail-under` so CI fails when accuracy regresses.

No required dependencies. `Pillow` is used for perceptual hashing when it is installed, and Ultralytics is an optional extra for running a model directly.

## Why this exists

Training a detector is easy. Knowing whether the *dataset* or the *model* got worse is not. Two failure modes cost teams the most time:

1. **A bad split.** The same image in `train` and `val` inflates the score until the model ships and collapses in production. Near-duplicates do the same thing more quietly.
2. **A score you cannot reproduce.** mAP printed by a notebook depends on the IoU sweep, the score threshold, the matching order and which classes are averaged. Change any of those and the number moves without the model changing.

`signlab-eval` fixes the vocabulary: one implementation of AP, an explicit class list, and JSON output that CI can diff between runs.

## Install

Not published to PyPI. Install from a checkout:

```bash
git clone https://github.com/Suprit-U/signlab-eval
cd signlab-eval
pip install -e .                              # library + CLI, no dependencies
pip install -e ".[dev]"                       # + pytest
pip install -e ".[ultralytics]"              # + run a YOLO model directly
```

## Quickstart

The repository ships a generated demo dataset, so every command below is reproducible offline.

```bash
python examples/make_demo.py examples/demo
```

### Validate a dataset

```console
$ signlab validate --dataset examples/demo
0 error(s), 0 warning(s)
splits=2, images=9, boxes=9, classes=3, errors=0, warnings=0
```

Exit code `0`. A dataset with errors exits `2`.

### Evaluate a split against predictions

`predictions.jsonl` holds one object per image:

```json
{"image": "val_000.png", "detections": [{"class_id": 0, "bbox": [10, 12, 40, 44], "score": 0.91}]}
```

```console
$ signlab evaluate --dataset examples/demo --split val \
      --predictions examples/demo/predictions.jsonl \
      --fail-under-map50 0.5 \
      --out metrics.json --html report.html
images=3  ground_truth=3
mAP@50     0.6667
mAP@50-95  0.3333
 class     AP@50   AP@50-95      gt     det
     0    1.0000     0.5000       1       1
     1    0.0000     0.0000       1       0
     2    1.0000     0.5000       1       2
wrote metrics.json
wrote report.html
```

The demo deliberately contains one missed class and one false positive, so the numbers are not a wall of `1.0000`.

With `--weights runs/best.pt` instead of `--predictions`, the Ultralytics extra is used and predictions are generated in-process.

### Gate a pull request

```yaml
- run: signlab evaluate --dataset data/traffic --split val --weights runs/best.pt --fail-under-map50 0.72 --out metrics.json
```

Exit codes: `0` passed, `1` the gate failed, `2` the dataset is invalid, `3` usage or IO error.

### Merge reports

```bash
signlab validate --dataset data/traffic --out validation.json
signlab evaluate --dataset data/traffic --split val --weights runs/best.pt --out metrics.json
signlab report --validation validation.json --metrics metrics.json --html report.html
```

## What `validate` checks

| Code | Severity | Meaning |
| :--- | :--- | :--- |
| `missing-classes` | error | no `classes.yaml` / `classes.yml` / `classes.json` |
| `empty-split` | error | a split directory exists but holds no images |
| `missing-label` | error | image has no `.txt` label file |
| `unreadable-image` | error | header could not be parsed |
| `malformed-label` | error | not 5 fields, or coordinates outside `[0, 1]` |
| `class-out-of-range` | error | class id not declared in `classes.yaml` |
| `unlabelled-image` | warning | dataset has no labels directory at all |
| `tiny-box` | warning | box smaller than `--min-box-px` |
| `extreme-aspect` | warning | aspect ratio beyond `--max-aspect` |
| `box-outside-image` | warning | coordinates fall outside the image |
| `duplicate-box` | warning | the same class and geometry twice in one file |
| `orphan-label` | warning | label with no matching image |
| `missing-val-split` | warning | no validation split found |
| `class-without-examples` | warning | declared class never appears |
| `class-imbalance` | warning | rarest class under `--imbalance-ratio` share |
| `train-val-overlap` | error | same image id in both splits |
| `duplicate-image` | error | identical or near-identical image (`--max-distance`) |

Leakage detection uses a dHash perceptual hash when Pillow is available; without it, signlab falls back to a content hash, which still catches byte-identical duplicates and says so in the finding.

## How AP is computed

- Detections are matched per image, greedily, in descending score order, and only to ground truth of the **same class**.
- A match requires IoU ≥ the threshold. Thresholds sweep `0.50 → 0.95` in steps of `0.05` (COCO); `mAP@50-95` averages all ten.
- AP uses all-point interpolation over the precision-recall curve, so there is no arbitrary 11-point subsampling.
- Unmatched detections count as false positives at every recall level; classes with no ground truth are skipped and reported rather than silently counted as zero.

## Development

```bash
pip install -e ".[dev]"
pytest -q
python examples/make_demo.py examples/demo
signlab validate --dataset examples/demo
```

## Limitations

- Predictions must already exist as JSONL, or Ultralytics must be installed to produce them. There is no ONNX or TensorRT path.
- The evaluator assumes one box per object. Overlapping crowd annotations lower recall; there is no ignore-region support.
- Image dimensions are read from PNG, JPEG, BMP, GIF and WEBP headers. Exotic formats are reported as `unreadable-image` rather than guessed.
- `duplicate-image` and `train-val-overlap` are only as good as the hash. Without Pillow the hash is content-based, so a re-encoded duplicate will not be caught.

## License

MIT