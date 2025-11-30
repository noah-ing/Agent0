SHELL := /bin/zsh
PYTHON := $(shell test -x .venv/bin/python && echo .venv/bin/python || command -v python3)
SANDFUZZ ?= sandfuzz
WORKERS ?= 4
PY_RUNTIME ?= python3.11

.PHONY: tool-server demo eval check-creds clean

tool-server:
	@echo "[tool-server] starting SandFuzz pool with $(WORKERS) workers"
	$(SANDFUZZ) pool --lang $(PY_RUNTIME) --workers $(WORKERS)

## Run toy co-evolution demo (requires live curriculum/executor endpoints)
demo:
	$(PYTHON) scripts/run_demo.py

## Run OpenCompass harness (configure endpoint + install opencompass beforehand)
eval:
	$(PYTHON) scripts/run_eval.py --suite math-lite

## Sanity-check verifier + vLLM credentials via .env
check-creds:
	$(PYTHON) scripts/check_credentials.py

clean:
	rm -rf __pycache__ */__pycache__ data/tool_runs/*.json
