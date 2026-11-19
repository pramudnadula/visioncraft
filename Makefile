PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
CLI := $(VENV)/bin/face-detect

.PHONY: setup test demo clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[backend,dev]"

test:
	$(PYTEST) -v

demo:
	@if [ -z "$(IMAGE)" ]; then echo "Usage: make demo IMAGE=path/to/image.jpg"; exit 1; fi
	$(CLI) --image "$(IMAGE)" --blur gaussian --output output/blurred.png --export-json output/report.json --no-show

clean:
	rm -rf output build dist *.egg-info .pytest_cache
