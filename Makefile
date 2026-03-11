# -----------------------------------------------
# Makefile
# -----------------------------------------------

# make setup       # create venv & install deps
# make format      # autoformat code with Ruff
# make lint        # run Ruff linter
# make fixlint     # auto-fix lint issues where possible
# make test        # run unit tests
# make test-debug  # run unit tests with stdout enabled
# make run         # placeholder for a future package entrypoint
# make clean       # remove caches and the virtual environment

VENV = .venv
PYTHON = $(VENV)/bin/python
UV = uv

# ---------- setup
.PHONY: setup
setup:
	$(UV) venv $(VENV) --python 3.11
	$(UV) pip install -e ".[dev]"

# ---------- formatting
.PHONY: format
format:
	$(VENV)/bin/ruff format src tests

# ---------- linting
.PHONY: lint
lint:
	$(VENV)/bin/ruff check src tests

.PHONY: fixlint
fixlint:
	$(VENV)/bin/ruff check --fix src tests

# ---------- testing
.PHONY: test
test:
	$(VENV)/bin/pytest -v --maxfail=1 --disable-warnings -q

.PHONY: test-debug
test-debug:
	$(VENV)/bin/pytest -s -v --maxfail=1 --disable-warnings

# ---------- run (placeholder for a future CLI/module entrypoint)
# .PHONY: run
# run:
# 	$(PYTHON) -m ragforge

# ---------- clean
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf $(VENV)
