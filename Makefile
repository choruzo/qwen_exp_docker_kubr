PYTHON ?= python

.PHONY: install validate test extract extract-dry-run clean normalize dedupe split train-preflight train-smoke train benchmark-baseline benchmark-finetuned benchmark-gguf report

install:
	$(PYTHON) -m pip install -e ".[extract,dedupe,test]"

validate:
	$(PYTHON) -m docker_k8s_finetune.cli config-validate

test:
	$(PYTHON) -m pytest

extract-dry-run:
	$(PYTHON) -m docker_k8s_finetune.cli extract --dry-run

extract:
	$(PYTHON) -m docker_k8s_finetune.cli extract

clean:
	$(PYTHON) -m docker_k8s_finetune.cli clean

normalize:
	$(PYTHON) -m docker_k8s_finetune.cli normalize

dedupe:
	$(PYTHON) -m docker_k8s_finetune.cli dedupe

split:
	$(PYTHON) -m docker_k8s_finetune.cli split

train-preflight:
	$(PYTHON) -m docker_k8s_finetune.cli train --smoke-test --preflight --no-export

train-smoke:
	docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli train --smoke-test --no-export

train:
	docker compose -f compose.train.yaml run --rm train

benchmark-baseline:
	docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli benchmark --variant baseline

benchmark-finetuned:
	docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli benchmark --variant finetuned_safetensors

benchmark-gguf:
	$(PYTHON) -m docker_k8s_finetune.cli benchmark --variant finetuned_gguf

report:
	$(PYTHON) -m docker_k8s_finetune.cli report
