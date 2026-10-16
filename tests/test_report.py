"""Tests for the report writers.

These cover the three payload shapes the CLI can hand to ``write_html``: an
explicit table spec, a validation payload with findings, and a flat summary.
"""

from __future__ import annotations

import json

from signlab.report import metrics_section, write_html, write_json


def test_write_json_creates_parent_dirs_and_keeps_order(tmp_path):
    payload = {"tool": "signlab-eval", "map50": 0.5, "nested": {"a": 1}}
    out = write_json(payload, tmp_path / "deep" / "dir" / "metrics.json")
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_write_html_renders_explicit_table_spec(tmp_path):
    spec = {"columns": ["class", "AP@50"], "rows": [[0, "1.0000"], [1, "0.0000"]]}
    out = write_html("demo", [("Metrics", spec)], tmp_path / "r.html")
    body = out.read_text(encoding="utf-8")
    assert "<h1>demo</h1>" in body
    assert "<h2>Metrics</h2>" in body
    assert body.count("<tr") == 3  # header + two rows
    assert "AP@50" in body


def test_write_html_renders_findings_payload_with_severity_classes(tmp_path):
    payload = {
        "findings": [
            {"severity": "error", "code": "missing-label", "sample": "train_001", "message": "no label file"},
            {"severity": "warning", "code": "tiny-box", "sample": "val_002", "message": "3.0px wide"},
        ],
        "stats": {"images": 9},
    }
    out = write_html("validation", [("Validation", payload)], tmp_path / "v.html")
    body = out.read_text(encoding="utf-8")
    assert 'class="error"' in body
    assert 'class="warning"' in body
    assert "missing-label" in body


def test_write_html_renders_flat_mapping_as_field_value(tmp_path):
    out = write_html("summary", [("Metrics", {"map50": 0.6667, "classes": ["stop", "sign"]})], tmp_path / "f.html")
    body = out.read_text(encoding="utf-8")
    assert "map50" in body
    assert "0.6667" in body
    assert "stop, sign" in body


def test_write_html_handles_empty_sections_and_bad_payloads(tmp_path):
    out = write_html("empty", [("A", {"rows": []}), ("B", None), ("C", "not a dict")], tmp_path / "e.html")
    body = out.read_text(encoding="utf-8")
    assert body.count("Nothing to report.") == 3
    assert body.endswith("</body></html>")


def test_metrics_section_prefers_per_class_rows():
    payload = {
        "num_images": 3,
        "num_ground_truth": 4,
        "map50": 0.5,
        "map50_95": 0.25,
        "per_class": {"0": {"ap50": 1.0, "ap50_95": 0.5, "ground_truth": 2, "detections": 3}},
    }
    heading, spec = metrics_section(payload)
    assert heading == "Metrics"
    assert spec["columns"][0] == "class"
    assert spec["rows"] == [["0", "1.0000", "0.5000", 2, 3]]


def test_metrics_section_falls_back_to_summary_rows():
    heading, spec = metrics_section({"num_images": 3, "num_ground_truth": 4, "map50": 0.5, "map50_95": 0.25})
    assert heading == "Metrics"
    assert spec["columns"] == ["metric", "value"]
    values = dict(spec["rows"])
    assert values["images"] == 3
    assert values["mAP@50"] == "0.5000"
    assert values["mAP@50-95"] == "0.2500"