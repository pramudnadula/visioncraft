PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
CLI := $(VENV)/bin/face-detect

.PHONY: setup test demo clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) -v

demo:
	$(CLI) --image RDJ.png --blur gaussian --output output/RDJ_blurred.png --export-json output/rdj.json --no-show

clean:
	rm -rf output build dist *.egg-info .pytest_cache
