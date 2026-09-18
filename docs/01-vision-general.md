# Visión general

## 1. Objetivo

Construir de extremo a extremo un pipeline reproducible que:

1. extraiga, limpie y normalice un dataset de instrucción sobre Docker/Kubernetes,
   con licencia y atribución trazables **por registro**;
2. haga fine-tuning de `unsloth/Qwen3.5-4B` con Unsloth para obtener un **experto de
   dominio estrecho** (explícitamente no un generalista);
3. compare antes/después sobre un test congelado y exporte a GGUF para servir con
   llama.cpp u Ollama.

El objetivo original está en `prompt_codex_docker_k8s_finetuning.md`. Requisitos
declarados allí y respetados en el código actual: todo vía CLI/YAML (sin notebooks
sueltos), semillas registradas en cada paso estocástico, y una única GPU de consumo
como restricción de diseño.

## 2. Restricciones de hardware

| Ruta | GPU | Estrategia |
|---|---|---|
| NVIDIA (principal) | RTX 5060 Ti 16 GB, Blackwell sm_120 | QLoRA 4-bit, `adamw_8bit` |
| ROCm (independiente) | Radeon AI PRO R9700 32 GB, gfx1201 | LoRA BF16 sin cuantizar, `adamw_torch` |

Consecuencias que atraviesan todo el diseño:

- **Inferencia y juicio son dos fases separadas.** Con una sola GPU de 16 GB no caben
  a la vez el modelo evaluado y el juez; primero se guardan las predicciones, se libera
  la GPU y luego se ejecuta `benchmark-judge`.
- **La validación sintáctica corre desde Windows**, no dentro del contenedor: necesita
  acceso directo al daemon Docker para ejecutar kubeconform y hadolint, y así se evita
  Docker-in-Docker.
- Se usa la imagen oficial de Unsloth fijada por digest para no recompilar `xformers`
  para sm_120 en Windows. La imagen derivada validada ocupa ~42 GB, por lo que un
  límite de disco Docker de 20 GB **no** es compatible con esta ruta (mínimo ~50 GB).

## 3. Paquete y punto de entrada

Paquete `docker_k8s_finetune` bajo `src/` (~7 640 líneas de Python). CLI `dkft`,
también invocable como `python -m docker_k8s_finetune.cli`.

Cada etapa del pipeline es un subpaquete con la misma forma:

- `pipeline.py` — orquestación, la invoca el CLI;
- `core.py` — lógica pura, testeable sin GPU ni red.

```
src/docker_k8s_finetune/
├── extract/    github_issues.py, stackoverflow.py, git_markdown.py,
│               huggingface.py tras base.py, driven by runner.py
├── clean/      redacción PII y limpieza
├── normalize/  core.py (pares directos) + generate.py (reverse-instruction vía LLM)
├── dedupe/     exact.py (hash) + approximate.py (embeddings + FAISS)
├── validate/   sintaxis YAML/Dockerfile de registros ChatML
├── split/      core.py (split estratificado + fuga) + audit.py (longitudes)
├── train/      runner.py (bucle Unsloth/TRL, resume, OOM) + core.py + callbacks.py
├── benchmark/  backends.py, pipeline.py, scoring.py, report.py
├── cli.py          argparse; un command_* por subcomando
├── config.py       carga y validación de YAML compartida
├── completeness.py decide provisional vs. final
├── errors.py       PipelineError, ConfigError, ExtractionError
└── io.py/schema.py JSONL atómico y esquema ChatML
```

`tests/` (18 archivos) refleja esa estructura: `test_train_core.py`,
`test_split_audit.py`, `test_benchmark_core.py`, etc.

## 4. Módulos base compartidos

**`io.py`** — `content_hash`/`file_sha256` (SHA-256 sobre string/bytes o archivo en
chunks de 1 MiB), `stable_json` (`sort_keys=True`, separadores compactos, para huellas
deterministas), `read_jsonl` (iterador que valida cada línea como objeto JSON) y
`atomic_write_jsonl`/`atomic_write_json` (temporal + `os.replace`, devuelven
`(count, sha256)`). **Toda** escritura de artefactos del pipeline pasa por aquí: la
atomicidad es lo que permite reanudar cualquier etapa tras un corte.

**`schema.py`** — dos dataclasses con `validate()`:

- `InterimRecord`: registro de extracción (`source`, `category`, `raw_content`,
  `license`, `url`, `retrieved_at` + procedencia opcional).
- `ChatRecord`: registro ChatML final `{messages, meta}`; exige roles exactos
  `["system", "user", "assistant"]`, contenido no vacío y `meta` con
  `category`/`source`/`license`/`url`.

**`config.py`** — `load_yaml` más un validador por etapa. Los más relevantes:

| Validador | Exige |
|---|---|
| `validate_sources_config` | `version == 1`, `status == approved`, `mass_extraction_allowed`, categorías exactas, licencia o `license_strategy` por fuente |
| `validate_splits_config` | ratios que suman 1.0, categorías exactas, umbral de fuga en (0,1], `status == approved` |
| `validate_auxiliary_configs` | validadores `kubeconform`/`hadolint` pinneados `@sha256:…`, revisión de embeddings de 40 hex, reglas de `benchmark.yaml` |
| `validate_cross_config` | toda fuente final tiene prioridad declarada en el dedupe aproximado |

## 5. Categorías del dominio

Seis, fijas en `config/sources.yaml` y comprobadas por los validadores:
`concepto`, `comando_cli`, `generacion_yaml`, `troubleshooting`, `dockerfile`,
`arquitectura`. A ellas se añade `out_of_domain` **solo en el benchmark** (50 preguntas
fijas) para medir olvido catastrófico.

## 6. Stack de dependencias

Python `>=3.10,<3.13`. Base: `PyYAML`, `requests`. Extras:

| Extra | Contenido |
|---|---|
| `extract` | beautifulsoup4, datasets, google-cloud-bigquery, langdetect |
| `dedupe` | faiss-cpu, numpy, sentence-transformers |
| `train` | accelerate, bitsandbytes, datasets, peft, tensorboard, `transformers==5.2.0`, trl, unsloth |
| `test` | pytest, pytest-cov |

Imágenes y versiones dentro de los contenedores, en
[02-pipeline-datos.md](02-pipeline-datos.md) § Infraestructura.

## 7. Reproducibilidad y seguridad

- **Nada se publica automáticamente**: no hay push a Hugging Face ni publicación de
  datasets en ningún punto del pipeline.
- `data/raw`, `data/interim`, `data/quarantine` y `artifacts/` están en `.gitignore`;
  solo `data/processed` se versiona (train.jsonl vía Git LFS, ~167 MB).
- `.gitattributes` fuerza `*.jsonl -text`: los hashes JSONL son parte del linaje
  congelado y **nunca** deben reescribirse los finales de línea.
- Validadores e imágenes base fijados por digest; semilla global 3407.
- Las fuentes sin licencia confirmada permanecen en `data/quarantine`.
