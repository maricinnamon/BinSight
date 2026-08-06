# BinSight TrashNet classifier
#
# Every target runs inside the dedicated Conda environment and returns a
# non-zero exit code on a real failure, so `make all` stops at the first problem.

ENV      ?= binsight-trashnet-metal
CONDA    ?= conda
PY        = PYTHONHASHSEED=42 $(CONDA) run -n $(ENV) --no-capture-output python
CONFIG   ?= configs/default.yaml

.DEFAULT_GOAL := help
.PHONY: help inspect prepare smoke-test train evaluate export verify benchmark test integrate all clean

help:
	@echo "BinSight TrashNet classifier"
	@echo ""
	@echo "  make inspect     record the environment (TF, Metal, TFLite checks)"
	@echo "  make prepare     validate the dataset and build deterministic splits"
	@echo "  make smoke-test  prove the whole pipeline runs before training for real"
	@echo "  make train       two-stage transfer training"
	@echo "  make evaluate    test-split metrics, reports and figures"
	@echo "  make export      float32 / float16 / int8 TFLite"
	@echo "  make verify      verify every candidate and select one"
	@echo "  make benchmark   relative latency comparison (this Mac, NOT iPhone)"
	@echo "  make test        pytest"
	@echo "  make integrate   copy the selected model into the iOS app bundle"
	@echo "  make all         prepare -> smoke-test -> train -> evaluate -> export -> verify -> benchmark"
	@echo ""
	@echo "  ENV=$(ENV)  CONFIG=$(CONFIG)"

inspect:
	$(PY) scripts/inspect_environment.py

prepare:
	$(PY) scripts/prepare_dataset.py --config $(CONFIG)

smoke-test:
	$(PY) scripts/smoke_test.py

train:
	$(PY) scripts/train.py --config $(CONFIG)

evaluate:
	$(PY) scripts/evaluate.py --config $(CONFIG)

export:
	$(PY) scripts/export_tflite.py --config $(CONFIG)

verify:
	$(PY) scripts/verify_tflite.py --config $(CONFIG)

benchmark:
	$(PY) scripts/benchmark_tflite.py --config $(CONFIG)

test:
	$(CONDA) run -n $(ENV) --no-capture-output python -m pytest tests -v

integrate:
	$(PY) scripts/integrate_ios.py --config $(CONFIG)

all: inspect prepare smoke-test train evaluate export verify benchmark
	@echo ""
	@echo "Pipeline complete. Selected model: models/tflite/binsight_trashnet.tflite"
	@echo "Reports: reports/"

clean:
	rm -rf models/keras/checkpoints reports/logs/*.csv
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
