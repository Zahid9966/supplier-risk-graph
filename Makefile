PY ?= .venv/bin/python
.PHONY: setup run test serve sweep docker clean
setup:
	uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
run:       ## simulate -> bronze -> silver -> gold -> evaluate (+leakage audit) -> score
	$(PY) -m srg.cli run --fresh
test:
	$(PY) -m pytest -q
serve:
	$(PY) -m srg.cli serve
sweep:     ## contagion-strength sweep incl. no-contagion control (slow: ~15 min)
	$(PY) -m srg.cli sweep
docker:
	docker build -t supplier-risk-graph . && docker run --rm -p 8000:8000 supplier-risk-graph
clean:
	rm -rf data/raw data/bronze data/silver data/gold data/sim_truth reports
