# Security Policy

## Supported versions

Only the latest released version receives fixes.

## Reporting a vulnerability

Please open a private security advisory on the repository rather than a public issue.
Include the affected version, a description, and a reproducer if you have one. Expect an
acknowledgement within a few days.

## What this tool does with your data

`signlab-eval` runs entirely locally. It reads images and label files you point it at and
writes reports to the paths you pass. It makes no network requests, reads no environment
variables for credentials, and has no telemetry. The optional Ultralytics predictor runs
locally too; if you use `--weights`, the model file is loaded from your disk only.
