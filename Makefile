.PHONY: help test demo validate evaluate report lint clean

help:
	@echo "test      run the test suite"
	@echo "demo      regenerate the example dataset"
	@echo "validate  validate the example dataset"
	@echo "evaluate  evaluate the example val split"
	@echo "report    build the combined HTML report"
	@echo "clean     remove build and cache artefacts"

test:
	.venv/bin/python -m pytest -q

demo:
	python examples/make_demo.py examples/demo

validate:
	.venv/bin/python -m signlab.cli validate --dataset examples/demo

evaluate:
	.venv/bin/python -m signlab.cli evaluate --dataset examples/demo --split val \
		--predictions examples/demo/predictions.jsonl --out build/metrics.json

report:
	.venv/bin/python -m signlab.cli report --metrics build/metrics.json --html build/report.html

clean:
	rm -rf build .pytest_cache **/__pycache__ src/*.egg-info
