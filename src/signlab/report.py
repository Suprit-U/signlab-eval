"""Reports: JSON for machines, HTML for humans. No templates, no dependencies."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

_STYLE = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto; max-width: 60rem; padding: 2rem 1rem; }
h1 { font-size: 1.5rem; margin-bottom: .25rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; }
p.sub { opacity: .7; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; font-variant-numeric: tabular-nums; }
th, td { border: 1px solid rgba(128,128,128,.35); padding: .35rem .6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
tr.error td { background: rgba(220,60,60,.14); }
tr.warning td { background: rgba(230,160,40,.16); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
"""


def write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def _short(value: object) -> str:
    """Render a value for a cell without dumping a whole JSON tree."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _as_table(spec: object) -> tuple[list[str], list[list]]:
    """Normalise any section payload into ``(columns, rows)``.

    Accepts the explicit ``{"columns": [...], "rows": [[...]]}`` form, a report
    payload carrying ``findings``, or a flat ``{field: value}`` summary.
    """
    if not isinstance(spec, dict):
        return [], []
    rows = spec.get("rows")
    if isinstance(rows, list):
        return list(spec.get("columns") or []), list(rows)
    findings = spec.get("findings")
    if isinstance(findings, list):
        columns = ["severity", "code", "sample", "message"]
        rows = [
            [
                str(f.get("severity", "info")),
                str(f.get("code", "")),
                _short(f.get("sample") or f.get("image") or ""),
                _short(f.get("message", "")),
            ]
            for f in findings
            if isinstance(f, dict)
        ]
        return columns, rows
    return ["field", "value"], [[str(k), _short(v)] for k, v in spec.items()]


def write_html(title: str, sections: list[tuple[str, object]], path: Path) -> Path:
    """Render ``sections`` as stacked tables.

    ``sections`` is ``[(heading, payload)]`` where the payload is a table spec, a
    report payload with ``findings``, or a flat mapping of field to value.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        f"<title>{html.escape(title)}</title>",
        "<style>" + _STYLE + "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class=sub>signlab-eval &middot; generated {now}</p>",
    ]
    for heading, spec in sections:
        parts.append(f"<h2>{html.escape(str(heading))}</h2>")
        columns, rows = _as_table(spec)
        if not rows:
            parts.append("<p>Nothing to report.</p>")
            continue
        parts.append("<table><thead><tr>")
        parts.extend(f"<th>{html.escape(str(c))}</th>" for c in columns)
        parts.append("</tr></thead><tbody>")
        for row in rows:
            cells = list(row)
            cls = ""
            if cells and cells[0] in ("error", "warning", "info"):
                cls = f' class="{cells[0]}"'
                cells = cells[1:]
            parts.append(f"<tr{cls}>")
            parts.extend(f"<td>{html.escape(str(c))}</td>" for c in cells)
            parts.append("</tr>")
        parts.append("</tbody></table>")
    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def metrics_section(payload: dict) -> tuple[str, dict]:
    """Build the per-class metrics table, falling back to the headline numbers."""
    per_class = payload.get("per_class", {})
    rows = [
        [
            cid,
            f"{values['ap50']:.4f}",
            f"{values['ap50_95']:.4f}",
            values["ground_truth"],
            values["detections"],
        ]
        for cid, values in per_class.items()
    ]
    if rows:
        return "Metrics", {
            "columns": ["class", "AP@50", "AP@50-95", "gt", "det"],
            "rows": rows,
        }
    return "Metrics", {
        "columns": ["metric", "value"],
        "rows": [
            ["images", payload.get("num_images", 0)],
            ["ground truth boxes", payload.get("num_ground_truth", 0)],
            ["mAP@50", f"{payload.get('map50', 0):.4f}"],
            ["mAP@50-95", f"{payload.get('map50_95', 0):.4f}"],
        ],
    }
