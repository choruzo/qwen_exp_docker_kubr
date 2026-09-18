# Benchmark y evaluación

> El orden importa: **baseline antes de entrenar**, luego variantes fine-tuned, luego
> sintaxis, luego juez, y por último `report`. Cada etapa es reanudable mediante cachés
> por registro.

## 1. Variantes

| Variante | Backend | Modelo | Requisito |
|---|---|---|---|
| `baseline` | `unsloth` | `Modelo/Qwen3.5-4B` | procedencia contra `config/training.yaml` |
| `finetuned_safetensors` | `unsloth` | `artifacts/merged` | `artifacts/export_manifest.json` |
| `finetuned_gguf` | `openai_compatible` | llama-server | `require_llama_props: true`, `expected_quantization: q4_k_m` |

`config/benchmark.rocm.yaml` es estructuralmente idéntico: solo cambia
`provenance_config` a `config/training.rocm.yaml` y las rutas a `benchmarks/rocm/*` y
`artifacts/rocm/*`.

## 2. Parámetros de generación (`config/benchmark.yaml`)

| Parámetro | Valor |
|---|---|
| seed | 3407 |
| temperature | 0.0 |
| top_p / repetition_penalty | 1.0 / 1.0 |
| `max_new_tokens` | 2048 (cachés de 1536 previos siguen siendo compatibles) |
| `batch_size` | 4 |
| `batch_order` | `prompt_length_ascending` |
| `retry_on_truncation` | activado, 1 intento, `repetition_penalty` 1.1 |
| `max_truncation_rate` | 1 % |
| `out_of_domain_count` | 50 preguntas fijas |
| `enable_thinking` | false |

Una instrucción fija de respuesta exige concisión y bloques YAML/Dockerfile completos,
en el idioma de la pregunta.

### Política de batching

- Lotes de 4 con left-padding; el conteo de tokens de cada salida **se corta en su
  primer EOS** para no contar el padding del lote.
- Los *misses* se ordenan establemente por longitud del prompt antes de agruparse, para
  reducir padding. **Nunca** se consulta la longitud de la respuesta de referencia — eso
  filtraría información del target en la planificación.
- El tamaño de lote forma parte de la huella de la caché: cambiarlo no mezcla salidas ni
  métricas de regímenes distintos.
- Escritura atómica por registro; si un lote falla, se reintenta en modo secuencial.
- Si un corte deja escrito solo parte de un lote, al reanudar se regenera el **grupo
  completo** (`deterministic_full_group_v1`) y no se desplazan los emparejamientos
  posteriores.
- Los backends HTTP sin batching nativo conservan ejecución secuencial.

### Truncamiento

Una salida se considera truncada solo si alcanza el presupuesto **sin haber emitido
EOS**: terminar exactamente en el último token permitido no es un falso positivo. Si
alcanza 2048, se reintenta **una sola vez** con `repetition_penalty` 1.10 para cortar
bucles degenerados; se registran intentos, tokens totales y latencia acumulada para no
ocultar el coste del rescate.

Una tasa de truncamiento superior al 1 % hace fallar el gate `completion.generation` y
mantiene el benchmark provisional.

## 3. Backends (`benchmark/backends.py`)

**`UnslothBackend`** — in-process, exige `torch.cuda.is_available()`, genera en lote con
left-padding y calcula tokens hasta el primer EOS por muestra. Verifica la procedencia
del modelo con `verify_local_model_provenance` (baseline) o `verify_export_artifact`
(merged).

**`OpenAICompatibleBackend`** — HTTP contra llama-server. Antes de aceptar el backend
consulta `/props` y exige que coincidan **alias, ruta normalizada, SHA-256 y
cuantización**. Que el endpoint responda no es suficiente.

`scripts/start-finetuned-gguf.ps1` cierra el círculo por el otro lado: lee
`artifacts/export_manifest.json`, localiza la entrada GGUF que corresponde a la
cuantización pedida (normalizando el nombre), exige una única coincidencia, valida que
la ruta no salga del proyecto y verifica bytes + SHA-256 antes de lanzar llama-server.

```
$env:GGUF_LLM_BASE_URL="http://host.docker.internal:8080/v1"
$env:GGUF_LLM_MODEL="qwen-docker-k8s"
$env:GGUF_LLM_API_KEY=""
```

## 4. Caché de generación

`benchmarks/work/<variante>/generation_cache/<content_hash>.json`. El fingerprint
(`_generation_cache_fingerprint`) incluye variante, backend, parámetros de generación y
contenido de los mensajes; si algo cambia, la caché se invalida.

Excepciones pensadas para no re-inferir sin motivo:

- Cambios exclusivos del umbral de auditoría **no** fuerzan nueva inferencia.
- Respuestas completas generadas con el límite anterior de 1536 se migran; solo las que
  alcanzaron ese techo se regeneran
  (`cache_compatible_prior_max_new_tokens: [1536]`).
- Cachés anteriores a la política de reintento migran las respuestas no truncadas y
  envían las truncadas directamente al reintento.

Cualquier error pendiente mantiene el benchmark como provisional.

## 5. Scoring (`benchmark/scoring.py`)

| Métrica | Cómo se calcula | Aplicación |
|---|---|---|
| exact match | comparación exacta | solo donde tiene sentido (`comando_cli`, OOD) |
| similitud semántica | coseno de embeddings normalizados, `paraphrase-multilingual-mpnet-base-v2` con revisión pinneada | todas |
| validez sintáctica | extrae bloques de la predicción y llama a kubeconform/hadolint vía `validate/pipeline.py` | solo `generacion_yaml` y `dockerfile` |
| LLM-juez | `JudgeClient` contra llama.cpp OpenAI-compatible, identidad verificada con `/props` | manifiesto de 150 + 50 OOD |

Se registran además latencia, tokens/segundo, `batch_tokens_per_second`, intentos de
generación, tasa de reintento y tasa de truncamiento, todos con media, desviación y
IC 95 %.

## 6. El juez

Gemma 4 12B Q6 (`gemma-4-12b-it-UD-Q6_K_XL.gguf`), pinneado en
`config/benchmark.yaml`:

| Campo | Valor |
|---|---|
| bytes | 10 685 011 360 |
| SHA-256 | `eb0f252863d14f7782122a4ac7e8744ed6e4a9fc132584d94686a9441f7c5d35` |
| temperature / seed | 0 / 3407 |
| `require_llama_props` | true |

Prompt fijo en `benchmarks/judge_prompt.md`. Debe arrancarse **solo después de liberar
la GPU** del modelo evaluado:

```
powershell -ExecutionPolicy Bypass -File scripts/start-judge-gemma4.ps1 -VerifyOnly
powershell -ExecutionPolicy Bypass -File scripts/start-judge-gemma4.ps1
```

El lanzador comprueba tamaño y SHA-256 antes de servir; el cliente exige además que
`/props` confirme alias, nombre de archivo y cuantización, y guarda esa identidad en
cada resultado juzgado. Cada respuesta del juez se persiste atómicamente, así que
`benchmark-judge` es reanudable.

```
$env:JUDGE_LLM_BASE_URL="http://host.docker.internal:8081/v1"
$env:JUDGE_LLM_MODEL="gemma-4-12b-judge"
$env:JUDGE_LLM_API_KEY=""
```

## 7. Las tres fases y por qué están separadas

Con una sola GPU de 16 GB no caben simultáneamente el modelo evaluado y el juez, y la
imagen de entrenamiento no tiene acceso al daemon Docker. De ahí el reparto:

| Fase | Dónde corre | Comando |
|---|---|---|
| 1. Generación | contenedor (GPU) | `benchmark --variant X --skip-judge --skip-syntax` |
| 2. Sintaxis | **Windows** (daemon Docker) | `benchmark-syntax --variant X` |
| 3. Juicio | contenedor, tras liberar la GPU y arrancar el juez | `benchmark-judge --variant X` |

Las fases 2 y 3 actualizan el mismo JSON de forma atómica y exigen
`completion.generation == True` y que los hashes de los registros coincidan exactamente
con el test congelado + OOD antes de anotar nada.

## 8. Provisional vs. final

`_is_provisional` (`benchmark/pipeline.py:139-140`): un resultado es provisional si
`split_statistics["provisional"]` es verdadero **o** si falta cualquiera de las cuatro
etapas de `completion`:

```
completion = {full_test, generation, syntax, judge}
```

`run_report` (`benchmark/report.py:410-536`) rehúsa comparar si alguna variante es
provisional o tiene `completion` incompleto, salvo `--allow-provisional`. Además valida:

- cobertura exacta de registros frente al manifiesto congelado
  (`_validate_frozen_record_coverage`);
- procedencia idéntica de los exports comparados contra el manifiesto actual
  (`_validate_export_provenance`);
- mismo hash de test entre variantes;
- misma firma CUDA entre `baseline` y `finetuned_safetensors`;
- mismos parámetros de generación, similitud, sintaxis y juez;
- identidad fija del juez servido (alias, filename, cuantización, prompt, manifiesto);
- presencia de `artifacts/metrics/training_result.json` con VRAM y filtro de longitud
  persistidos (o un `trainer_state.json` de respaldo).

Cualquier regresión se enumera explícitamente. El informe compara por categoría: exact
match, similitud semántica, validez kubeconform/hadolint, LLM-juez, latencia, tokens por
segundo, fuera de dominio y degradación por cuantización, y genera
`benchmarks/training_loss.svg`.

Los resultados de inferencia directa registran modelo de GPU, VRAM total, capacidad
CUDA y versiones de Torch/CUDA, para que latencia y rendimiento queden ligados al
hardware real.

## 9. Baseline medido (resultado final, verificado)

`benchmarks/baseline_results.json` — 17 MB, `provisional: false`,
`completion` con las cuatro etapas en `true`, **3 161 registros** (3 111 de test + 50
OOD), **0 errores**.

Runtime: RTX 5060 Ti, capacidad CUDA 12.0, Torch 2.10.0+cu128, 17 102 864 384 bytes de
VRAM total. `split_test_sha256 =
0eb2a9580bf5c1581ea542dc6068eeeb6e88443998c885aaa94ab988b3e17cc1`.

### Global

| Métrica | Media | IC 95 % | n |
|---|---|---|---|
| similitud semántica | 0,6531 | 0,6470 – 0,6592 | 3 161 |
| LLM-juez (1-5) | 3,285 | 3,059 – 3,511 | 200 |
| validez sintáctica | 0,7701 | 0,7429 – 0,7972 | 922 |
| exact match | 0,0580 | 0,0339 – 0,0821 | 362 |
| tokens/segundo | 7,59 | 7,44 – 7,74 | 3 161 |
| tasa de truncamiento | 0,57 % | — | 18 de 3 161 (umbral 1 %: **PASS**) |
| tasa de reintento | 4,05 % | — | 3 161 |

### Por categoría

| Categoría | n | Similitud | Juez | Sintaxis |
|---|---|---|---|---|
| generacion_yaml | 610 | 0,8494 | 3,80 | 0,8607 |
| out_of_domain | 50 | 0,8365 | 4,76 | — |
| arquitectura | 156 | 0,6768 | 2,48 | — |
| concepto | 269 | 0,6682 | 2,48 | — |
| comando_cli | 312 | 0,6484 | 2,46 | — |
| troubleshooting | 1 452 | 0,5779 | 3,02 | — |
| dockerfile | 312 | 0,5699 | 2,61 | 0,5929 |

Lectura útil de cara al fine-tuning: el modelo base ya es razonablemente bueno
generando YAML y respondiendo fuera de dominio, y **claramente débil en
`troubleshooting` y `dockerfile`** — que son, respectivamente, la categoría mayoritaria
del dataset (40 030 registros) y la de peor validez sintáctica (0,59). Ahí está el
margen de mejora que el entrenamiento debería capturar, y también el riesgo de
regresión en OOD (4,76 es un techo alto que conviene no perder).

Ver el estado del resto de variantes en [06-estado-actual.md](06-estado-actual.md).
