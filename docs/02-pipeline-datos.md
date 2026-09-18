# Pipeline de datos

> Orden obligatorio: `extract` → `clean` → `normalize` (+ `normalize-generate`) →
> `dedupe --mode exact` → `dedupe --mode approximate` → `validate-dataset` → `split`.

## 1. Resumen por etapa

| Etapa | Entrada | Salida principal | Gate |
|---|---|---|---|
| `extract` | `config/sources.yaml` | `data/interim/extracted/{fuente}.jsonl` + `.manifest.json` | `sources.yaml` aprobado; excluye `tier == quarantine` salvo `--include-quarantine` |
| `clean` | `extracted/*.jsonl` | `data/interim/cleaned/records.jsonl`, `data/quarantine/clean_rejected.jsonl`, `reports/cleaning_report.json`, `reviews/pii_redaction_sample.jsonl` | rechaza `too_short`, `too_long`, `language_not_allowed_or_uncertain` |
| `normalize` | `cleaned/records.jsonl` | `direct_chatml.jsonl`, `reverse_candidates_sample.jsonl` | — |
| `normalize-generate` | candidatos o `cleaned/records.jsonl` | `reverse_generated_{sample,full}.jsonl` + `reverse_cache/` | `scope full` exige revisión manual aprobada (§ 4) |
| `dedupe --mode exact` | `direct_chatml.jsonl` + `reverse_generated_full.jsonl` | `deduplicated/exact.jsonl`, `exact_duplicates.jsonl`, `exact_dedupe_report.json` | inputs presentes |
| `dedupe --mode approximate` | `exact.jsonl` | `chatml.jsonl`, `semantic_embeddings.npz`, reportes | **debe correr dentro de la imagen Docker validada** |
| `validate-dataset` | `chatml.jsonl` | `validated/chatml.jsonl`, `syntax_rejected.jsonl`, reporte | docker + validadores pinneados accesibles |
| `split` | `validated/chatml.jsonl` | `data/processed/*` + manifiestos de benchmark | `assess_pipeline_completeness` debe estar listo |

## 2. Fuentes y licencias (`config/sources.yaml`)

`status: approved`, `mass_extraction_allowed: true`, semilla 3407. La extracción
masiva **solo** se habilita con ese archivo aprobado.

| Fuente | Tier | Licencia | Notas |
|---|---|---|---|
| docker_docs, kubernetes_docs, helm_docs, cert_manager_docs, istio_docs, containerd_docs, argocd_docs, ingress_nginx_docs, coredns_docs | 1 | Apache-2.0 / CC-BY-4.0 (k8s) / MIT (Helm) | `git_markdown`, normalización `reverse_instruction` |
| k8s_instructions | 2 | Apache-2.0 | `direct_pair` |
| stack_yaml_k8s | 2 | `per_record` (allowlist permisiva) | muestreo determinista de 25k, `never_copy_without_instruction: true` |
| docker_nl_commands | 2 | Apache-2.0 | `direct_pair` |
| kubernetes_stackoverflow_questions | 2 | CC-BY-4.0 | hash del parquet LFS verificado |
| stackoverflow_docker | 2 | CC-BY-SA-4.0 | BigQuery, tope duro 70 GiB (dry-run validado: 63,55 GiB) |
| github_resolved_issues | 2 | `license_strategy: repository_declared` | 7 repos, máx. 75 registros/repo |
| **componentsoft_kubectl_35k**, **componentsoft_kubectl_cot_20k** | **quarantine** | unknown | `include_in_final: false` |
| **kubernetes_failure_stories** | **quarantine** | unknown | bloqueado hasta verificar licencia por página enlazada |

Allowlist del dataset final: Apache-2.0, MIT, BSD-2/3-Clause, ISC, Zlib, Unlicense,
CC0-1.0, CC-BY-4.0, CC-BY-SA-4.0.

### Credenciales necesarias

| Fuente | Variables |
|---|---|
| GitHub issues | `GITHUB_TOKEN` (p. ej. `$env:GITHUB_TOKEN = gh auth token`) |
| Stack Overflow (BigQuery) | `GOOGLE_CLOUD_PROJECT` + credenciales ADC (`gcloud auth application-default login`) |
| Reverse-instruction | `SYNTHETIC_LLM_BASE_URL`, `SYNTHETIC_LLM_MODEL` |

### Idempotencia

`extract/runner.py` despacha por `kind` (`git_markdown`, `failure_story_index`,
`huggingface_dataset`, `github_issues`, `stackoverflow_bigquery`). `should_skip` omite
una fuente si ya existen su output y manifest y no se pasa `--force`; el modo batch
tolera el fallo de una fuente sin abortar el resto.

## 3. Limpieza y PII (`config/cleaning.yaml`)

- `html_to_text` + `repair_fences` + filtrado de boilerplate.
- Longitud: `min_chars = 40`, `max_chars = 50000`.
- Idiomas permitidos `en`/`es` con `min_probability = 0.80` vía `langdetect` con semilla
  determinista; se omite la detección en fuentes marcadas como *trusted* y en
  categorías de código.
- Seis patrones de redacción PII por regex: email, IPv4 privada, AWS access key, token
  de GitHub, JWT y bearer token.

> La redacción PII es también el origen de la **excepción de procedencia de
> `val.jsonl`** descrita en [05-perfil-rocm.md](05-perfil-rocm.md): una credencial
> redactada cambió el SHA-256 del archivo respecto al manifiesto original.

Se emite `reviews/pii_redaction_sample.jsonl` para revisión humana de la redacción.

## 4. Normalización y reverse-instruction

Dos caminos:

- **Pares directos** (`normalize/core.py`): `direct_pair_to_chatml` exige
  `metadata.pair.user` y `.assistant` no vacíos.
- **Reverse-instruction** (`normalize/generate.py`): a partir de documentación se
  genera la instrucción que la respuesta contestaría, usando un LLM externo
  OpenAI-compatible.

`prepare_reverse_candidates` muestrea de forma determinista (hash SHA-256 por semilla)
por `(source, category)` y luego intercala round-robin por fuente para que ninguna
fuente domine el orden.

`config/normalization.yaml`: `sample_size = 100`, `max_reference_chars = 16000`,
`parallel_workers = 2`. El prompt fijo exige una instrucción **autosuficiente** y
respuesta JSON con clave `user`. La generación valida que el JSON tenga `user` no vacío
y **rechaza referencias no autosuficientes** (regex `_NON_SELF_CONTAINED`): frases del
tipo "como se explicó arriba" invalidarían el par.

Ejecución en paralelo con `ThreadPoolExecutor` preservando el orden, y caché por
candidato cuyo fingerprint incluye prompt, modelo, semilla y parámetros de generación:
una interrupción se reanuda sin regenerar respuestas ya validadas.

### El gate de revisión manual

Este es el **único paso manual** obligatorio del pipeline:

```
normalize-generate --scope sample      # genera 100 candidatos
# → revisar el JSONL a mano
# → copiar su SHA-256 a config/reverse_instruction_review.yaml
# → poner status: approved
normalize-generate --scope full        # solo ahora se permite
```

`_verify_full_gate` (`normalize/generate.py:190`) exige `status: approved` **y** que el
hash de la muestra generada coincida con `sample_sha256`. El estado actual del archivo
registra "95 aceptadas / 5 a cuarentena".

Endpoint local usado en este equipo: `scripts/start-synthetic-qwen3-14b.ps1` levanta
`llama-server` con `G:\models\Qwen3-14B-Q5_K_M.gguf`, dos slots de 8192 (16K total),
`--reasoning off`, `--jinja`, puerto 8080. No requiere API key.

```
$env:SYNTHETIC_LLM_BASE_URL="http://127.0.0.1:8080/v1"
$env:SYNTHETIC_LLM_MODEL="qwen3-14b"
```

Para corridas largas, `scripts/continue-after-reverse-generation.ps1` espera al PID del
generador (validando que sea `python*`), y **solo si termina con código cero** detiene
`llama-server` y encadena dedupe exacto → dedupe aproximado (Docker) →
`validate-dataset` → `split` → `pipeline-status`, parándose ante el primer error.

## 5. Deduplicación (`config/dedupe.yaml`)

**Exacta** (`dedupe/exact.py`): SHA-256 sobre contenido canónico — NFKC, normalización
de saltos de línea, `stable_json({user, assistant})`.

**Aproximada** (`dedupe/approximate.py`), obligatoriamente dentro de la imagen Docker
validada:

| Parámetro | Valor |
|---|---|
| Modelo | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| Revisión | pinneada a commit de 40 hex (`4328cf2…`) |
| Umbral | 0.92 |
| Índice | FAISS `IndexHNSWFlat`, coseno vía producto interno normalizado |
| HNSW | `m=32`, `efConstruction=200`, `efSearch=128`, `top_k=50` |

Agrupa por componentes conexas sobre el umbral. El representante de cada cluster se
elige por `quality_score`, luego `tier` de la fuente (1 = docs oficiales,
2 = datasets/SO/issues, 3 = stack_yaml_k8s) y finalmente hash como desempate
determinista. Emite `semantic_embeddings.npz`, reutilizado más tarde por la auditoría
de fuga del split.

## 6. Validación sintáctica (`config/validation.yaml`)

Extrae los bloques YAML/Dockerfile del mensaje `assistant` y los valida:

| Validador | Imagen | Configuración |
|---|---|---|
| kubeconform | `@sha256:85dbef6b…` (v0.7.0) | `kubernetes_version: 1.36.2`, `strict: true` |
| hadolint | `@sha256:9a3944b7…` (v2.15.1) | `failure_threshold: error`, `batch_size: 200` |

Ambos vía `docker run --rm --mount …readonly`. Falla si falta el ejecutable o Docker,
o si el `returncode` no es 0 ni 1 (1 = hallazgos, no error de herramienta).

También **reclasifica la categoría**: si detecta un bloque Dockerfile fuera de la
categoría `dockerfile`, o si un registro `dockerfile` no contiene bloque, corrige la
etiqueta antes del split.

`config.py` fuerza que ambos validadores estén pinneados por `@sha256:…`: la
reproducibilidad depende de no dejar flotar estas imágenes.

## 7. Split estratificado (`config/splits.yaml`)

Ratios **train/val/test = 0.90 / 0.05 / 0.05** con tolerancia 0.005.
`stratified_group_split` estratifica por `(category, source)` **agrupando por
`semantic_cluster_id`**: un cluster semántico nunca se parte entre splits, que es la
primera línea de defensa contra la fuga.

Además:

- Rebalanceo por fracción mínima de cada categoría en evaluación
  (`troubleshooting 0.20`, `generacion_yaml 0.15`, …).
- **Auditoría de fuga post-split** (`audit_and_repair`): recalcula similitud por
  embeddings con umbral 0.92, hasta 10 iteraciones, moviendo los clusters filtrados a
  train con backfill del mismo estrato.
- **Subset "hard"**: registros de `troubleshooting` con `complexity_score >= 3` según
  heurísticas léxicas; entre 100 y 500 registros y diversidad mínima de 3 fuentes.
- Verifica 0 solapamientos exactos y de cluster entre los tres splits, y que todas las
  categorías estén presentes en cada uno.

### Salidas

```
data/processed/train.jsonl, val.jsonl, test.jsonl
data/processed/split_manifest.jsonl, split_statistics.json
data/processed/leakage_report.json, hard_test_manifest.jsonl
data/processed/DATASHEET.md
benchmarks/test_manifest.jsonl, benchmarks/llm_judge_manifest.jsonl
```

El `DATASHEET.md` registra fecha UTC, número de registros y bytes/SHA-256 de cada
split, más distribuciones por fuente, licencia y categoría.

### Resultado congelado actual

| Split | Registros |
|---|---|
| train | 55 999 |
| validation | 3 111 |
| test | 3 111 |
| **total** | **62 221** |

14 fuentes, 10 licencias, 0 registros CC-BY/CC-BY-SA sin atribución. Categoría
mayoritaria: `troubleshooting` (40 030), seguida de `generacion_yaml` (12 177).
Manifiestos de benchmark: 3 111 de test y 150 para el juez.

`leakage_report.json`: `status = "clean"`, umbral 0.92, `initial_leaks = 0`,
`final_leaks = 0`, `iterations = 0`. Solapamientos exactos y semánticos = 0 en las tres
combinaciones. **Sin fuga.**

## 8. `pipeline-status` (`completeness.py`)

`assess_pipeline_completeness` es el auditor de linaje. Lo invoca
`command_pipeline_status` y también `split/pipeline.py` antes del split final. Audita,
en orden:

1. **Extracción** — por cada fuente final: output y manifest existentes, manifest
   coherente (`source` coincide, `records > 0`), sin *drift* del `source_config`
   (comparación por hash) y, si `verify_hashes`, SHA-256 real del archivo igual al del
   manifest.
2. **Limpieza** — `cleaning_report.json` debe cubrir el hash actual de cada extracción;
   si no, `cleaning_lineage_stale`.
3. **Reverse-instruction** — revisión `approved`, hash de la muestra coincidente y
   output "full" presente; si falta, añade las variables de entorno necesarias a
   `missing_environment`.
4. **Dedupe exacto** — sin `missing_inputs`, linaje de hashes al día, output con hash
   correcto.
5. **Dedupe aproximado** — su input coincide con el output exacto; output y embeddings
   con hash correcto.
6. **Validación sintáctica** — sin `input_override` provisional, input igual a la salida
   del dedupe semántico, output `accepted` con hash correcto.

Devuelve `{ready_for_final_split, expected_final_sources, extracted_final_sources,
missing_environment, issues}` y **sale con código distinto de cero** si hay cualquier
issue. `--no-verify-hashes` omite solo el recálculo de SHA-256 de archivos.

## 9. Infraestructura

### `Dockerfile.train` (NVIDIA)

Base `unsloth/unsloth@sha256:f21629b9…`. La imagen base trae Transformers 4.57.6, que
**no reconoce Qwen3.5**, así que la capa derivada instala:

| Paquete | Versión |
|---|---|
| transformers | 5.2.0 (exigida por la receta de Unsloth) |
| flash-linear-attention | 0.5.2 |
| torchao | 0.16.0 (compatible con Torch 2.10, instalado sin deps) |
| faiss-cpu | 1.15.0 |
| langdetect | 1.0.9 |
| sentence-transformers | 5.2.0 (ya en la base) |

Verifica las versiones con `assert` en runtime y anula el `ENTRYPOINT` original, que
haría `chmod` recursivo sobre todo el modelo y los datasets.

### `Dockerfile.train.rocm`

Base `unsloth/unsloth-rocm@sha256:e0fd6548…`. Añade `tensorboard==2.20.0`,
`sentence-transformers==5.2.0` y compila llama.cpp desde el commit pinneado
`44be98f057e9f9902a8ee12630e181c7f8ec2953` (target `llama-quantize`, CPU-only, para
conversión GGUF offline).

### Compose

| Archivo | Particularidades |
|---|---|
| `compose.train.yaml` | `platform: linux/amd64`, monta el repo y `.cache/huggingface`, reserva 1 GPU NVIDIA, variables de los endpoints sintético/juez/GGUF |
| `compose.train.rocm.yaml` | monta `/dev/kfd` y `/dev/dri`, `group_add` para los GID video/render, `security_opt: seccomp:unconfined`, comando por defecto con `config/training.rocm.yaml` |

La caché de Hugging Face vive en `.cache/huggingface` por bind mount, fuera de Git.

### Makefile

Equivalentes de casi todo: `install`, `validate`, `test`, `status`,
`extract[-dry-run]`, `clean`, `normalize`, `dedupe` (exacto local + aproximado en
Docker), `split`, `train-preflight` / `train-smoke` / `train`,
`benchmark-{baseline,finetuned,gguf}`, `benchmark-syntax-*`, `benchmark-judge-*`,
`report`.

### `scripts/audit-token-lengths.py`

Tokeniza cada split con `AutoTokenizer.from_pretrained(..., local_files_only=True)`
sobre `Modelo/Qwen3.5-4B`, aplica `apply_chat_template` y calcula media, p50/p90/p95/p99
y tasa sobre `--max-length` (8192 por defecto), global y por categoría.
