PYTHON ?= python

.PHONY: install validate test status extract extract-dry-run clean normalize dedupe split train-preflight train-smoke train benchmark-baseline benchmark-finetuned benchmark-gguf benchmark-syntax-baseline benchmark-syntax-finetuned benchmark-syntax-gguf benchmark-judge-baseline benchmark-judge-finetuned benchmark-judge-gguf report

install:
	$(PYTHON) -m pip install -e ".[extract,dedupe,test]"

validate:
	$(PYTHON) -m docker_k8s_finetune.cli config-validate

test:
	$(PYTHON) -m pytest

status:
	$(PYTHON) -m docker_k8s_finetune.cli pipeline-status

extract-dry-run:
	$(PYTHON) -m docker_k8s_finetune.cli extract --dry-run

extract:
	$(PYTHON) -m docker_k8s_finetune.cli extract

clean:
	$(PYTHON) -m docker_k8s_finetune.cli clean

normalize:
	$(PYTHON) -m docker_k8s_finetune.cli normalize

dedupe:
	$(PYTHON) -m docker_k8s_finetune.cli dedupe --mode exact
	docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli dedupe --mode approximate

split:
	$(PYTHON) -m docker_k8s_finetune.cli split

train-preflight:
	$(PYTHON) -m docker_k8s_finetune.cli train --smoke-test --preflight --no-export

train-smoke:
	docker compose -f compose.train.yaml run --build --rm train python -m docker_k8s_finetune.cli train --smoke-test --no-export

train:
	docker compose -f compose.train.yaml run --build --rm train

benchmark-baseline:
	docker compose -f compose.train.yaml run --build --rm train python -m docker_k8s_finetune.cli benchmark --variant baseline --skip-judge --skip-syntax

benchmark-finetuned:
	docker compose -f compose.train.yaml run --build --rm train python -m docker_k8s_finetune.cli benchmark --variant finetuned_safetensors --skip-judge --skip-syntax

benchmark-gguf:
	docker compose -f compose.train.yaml run --build --rm train python -m docker_k8s_finetune.cli benchmark --variant finetuned_gguf --skip-judge --skip-syntax

benchmark-syntax-baseline:
	$(PYTHON) -m docker_k8s_finetune.cli benchmark-syntax --variant baseline

benchmark-syntax-finetuned:
	$(PYTHON) -m docker_k8s_finetune.cli benchmark-syntax --variant finetuned_safetensors

benchmark-syntax-gguf:
	$(PYTHON) -m docker_k8s_finetune.cli benchmark-syntax --variant finetuned_gguf

benchmark-judge-baseline:
	docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli benchmark-judge --variant baseline

benchmark-judge-finetuned:
	docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli benchmark-judge --variant finetuned_safetensors

benchmark-judge-gguf:
	docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli benchmark-judge --variant finetuned_gguf

report:
	docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli report
