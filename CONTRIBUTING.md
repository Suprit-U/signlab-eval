# Contributing

Thanks for taking the look. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/Suprit-U/signlab-eval
cd signlab-eval
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
pytest -q
python examples/make_demo.py examples/demo
signlab validate --dataset examples/demo   # must print 0 error(s), 0 warning(s)
```

The demo dataset is generated, so if `signlab validate` reports problems on it,
something in the library changed behaviour. That is a bug worth opening an issue for.

## Ground rules

- **No new required dependencies.** Pillow and Ultralytics stay optional. Anything that
  removes a dependency is always welcome; anything that adds one needs a strong reason.
- **Metrics changes need a justification.** AP is easy to "improve" by changing the
  matching order or the IoU sweep. If you touch `metrics.py`, say in the PR why the old
  behaviour was wrong and include the before/after numbers on `examples/demo`.
- **Match the surrounding style.** Type hints, `from __future__ import annotations`,
  dataclasses for value objects, no clever one-liners.
- **Add a test with the fix.** The suite is fast (under a second) and should stay that way.

## Reporting a bug

Open an issue with the command you ran, the dataset layout (a `tree` of a small
reproducer helps), and the full output. A failing dataset you can share is worth ten
descriptions.
