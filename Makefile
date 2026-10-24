# Local development targets. Cloud GitHub Actions is optional / parked.
.PHONY: ci ci-fast test lint build help

help:
	@echo "make ci       - full local CI (compileall, ruff, pytest not live, build)"
	@echo "make ci-fast  - same without python -m build"
	@echo "make test     - pytest -m 'not live'"
	@echo "make lint     - ruff check ."
	@echo "make build    - python -m build"

ci:
	./scripts/ci.sh

ci-fast:
	./scripts/ci.sh --fast

test:
	@if [ -x .venv/bin/pytest ]; then .venv/bin/pytest -m "not live" -q --cov=link2kaggle_colab; else pytest -m "not live" -q --cov=link2kaggle_colab; fi

lint:
	@if [ -x .venv/bin/ruff ]; then .venv/bin/ruff check .; else ruff check .; fi

build:
	@if [ -x .venv/bin/python ]; then .venv/bin/python -m build; else python3 -m build; fi
