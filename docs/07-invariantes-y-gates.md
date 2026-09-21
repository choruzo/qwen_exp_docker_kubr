# Invariantes y gates

Este proyecto está construido alrededor de una idea: **cada etapa que escribe hashes o
manifiestos es un gate de corrección para otra etapa**. Aflojar una verificación de hash
sin actualizar el archivo de configuración que pinnea el valor esperado rompe silenciosamente
la cadena de reproducibilidad.

Esta página es el catálogo completo, para consultarlo antes de modificar código de
pipeline.

## 1. Mapa de gates

| Gate | Dónde | Qué exige | Cómo se desbloquea |
|---|---|---|---|
| Aprobación de fuentes | `validate_sources_config` | `status: approved`, `mass_extraction_allowed: true` | editar `config/sources.yaml` deliberadamente |
| Cuarentena de licencias | `extract/runner.py` | fuentes `tier: quarantine` fuera del dataset final | verificar licencia y cambiar el tier |
| Revisión reverse-instruction | `_verify_full_gate` | `status: approved` + `sample_sha256` coincidente | revisar la muestra a mano y copiar su hash |
| Validadores pinneados | `validate_auxiliary_configs` | kubeconform/hadolint con `@sha256:…` | actualizar el digest a conciencia |
| Revisión de embeddings | `validate_auxiliary_configs` | commit de 40 hex | — |
| Cobertura de prioridades | `validate_cross_config` | toda fuente final tiene tier en el dedupe | declarar el tier |
| Linaje completo | `assess_pipeline_completeness` | 6 etapas coherentes por hash | reejecutar la etapa desactualizada |
| Split final | `split/pipeline.py` | `ready_for_final_split` | resolver los `issues` de `pipeline-status` |
| Sin fuga | `split/core.py` | 0 solapamientos exactos y de cluster | `audit_and_repair` (hasta 10 iteraciones) |
| Identidad del modelo | `verify_local_model_provenance` | 8 SHA-256 + revisión Git | actualizar `config/training.yaml` |
| Arquitectura esperada | `inspect_local_model` | `Qwen3_5ForConditionalGeneration` | — |
| Baseline final | `_verify_baseline` | baseline no provisional, 4 etapas, hash de test e identidad del juez | completar el baseline |
| Config de entrenamiento | `validate_training_config` | ~8 reglas (§ 3) | corregir el YAML |
| Coherencia de checkpoint | `latest_checkpoint` + `checkpoint_fingerprint` | archivos completos + contrato idéntico | empezar de cero si algo cambió |
| Export completo | `finalize_gguf_export` | `q4_k_m` **y** `q8_0` presentes | reejecutar la conversión |
| Export sin smoke | `run_training` | `split_statistics` verificados | no exportar desde smoke |
| Identidad del GGUF servido | `OpenAICompatibleBackend` | `/props`: alias, ruta, SHA-256, cuantización | servir el archivo manifestado |
| Identidad del juez | `JudgeClient` | bytes, SHA-256 y `/props` | servir el Gemma pinneado |
| Truncamiento | `completion.generation` | tasa ≤ 1 % | revisar el presupuesto de tokens |
| Comparación final | `run_report` | ~7 validaciones cruzadas (§ 5) | `--allow-provisional` solo para diagnóstico |

## 2. La regla del estado "final"

Un resultado de benchmark es **provisional** si `split_statistics["provisional"]` es
verdadero o si falta cualquiera de las cuatro etapas:

```
completion = {full_test, generation, syntax, judge}
```

Y el proyecto entero no se considera terminado hasta que los JSON de benchmark son no
provisionales **y** el informe comparativo se ha generado sobre el test congelado.
`--allow-provisional` existe solo para diagnóstico explícito, nunca como vía normal.

## 3. Reglas de `validate_training_config`

Rechaza el arranque si:

1. `target_modules` no es `auto` (Qwen3.5 mezcla atención completa y lineal; solo
   `FastVisionModel` cubre ambas familias);
2. la estrategia de muestreo no es aleatoria con batch físico 1;
3. `per_device_eval_batch_size` o `eval_steps` no son positivos;
4. los perfiles de fallback OOM no reducen memoria de forma monótona;
5. `overlength_action` no es `exclude` (prohibido el truncamiento silencioso);
6. `accelerator` no es `cuda` ni `rocm`;
7. con `rocm`: hay cuantización 4-bit, `dtype` no es bfloat16, el optimizador no es
   `adamw_torch`, o `resume_from_checkpoint` no es `none`;
8. con `rocm`: alguna ruta de salida no empieza por `artifacts/rocm/`.

## 4. El contrato de checkpoint

`checkpoint_fingerprint` es el hash de un contrato que incluye:

- hash de la configuración completa;
- identidad del modelo base;
- hashes de train/val/test;
- perfil de entrenamiento (nombre, seq len, batch, acumulación, eval batch/steps,
  packing, sampling);
- clases de processor y tokenizer;
- el reporte de filtrado por longitud.

Consecuencia deliberada: **cambiar cualquiera de esas cosas invalida todos los
checkpoints existentes**. No es un inconveniente, es el punto — reanudar tras cambiar el
sampler o el contexto produciría un modelo cuya procedencia nadie puede describir.

`latest_checkpoint` exige además que existan y no estén vacíos `trainer_state.json`,
`optimizer.pt`, `scheduler.pt`, `rng_state.pth` y los pesos, y que
`global_step` coincida con el nombre del directorio. Así se ignoran los directorios
medio escritos por un corte de energía.

`oom_state.json` tiene su propio contrato (config + SHA-256 de
`split_statistics.json`): si no coincide, se ignora el estado entero.

## 5. Validaciones de `run_report`

Antes de comparar variantes exige:

1. ninguna variante provisional ni con `completion` incompleto;
2. cobertura exacta de registros frente al manifiesto congelado;
3. procedencia de los exports idéntica al manifiesto actual;
4. mismo `split_test_sha256` en todas las variantes;
5. misma firma de acelerador (CUDA o ROCm) entre `baseline` y `finetuned_safetensors`;
6. mismos parámetros de generación, similitud, sintaxis y juez;
7. identidad fija del juez servido (alias, filename, cuantización, prompt, manifiesto);
8. `training_result.json` configurado para el perfil (por defecto `artifacts/metrics/training_result.json`) con VRAM y filtro de longitud persistidos.

## 6. Invariantes de datos

- `data/raw`, `data/interim`, `data/quarantine` y `artifacts/` están gitignorados; solo
  `data/processed` se versiona (train.jsonl vía Git LFS).
- `.gitattributes` fuerza `*.jsonl -text`: **nunca** reescribir finales de línea de los
  JSONL, porque los hashes son parte del linaje congelado.
- Nada en el pipeline hace push a Hugging Face ni publica datasets automáticamente.
- Las fuentes sin licencia confirmada permanecen en `data/quarantine`.
- Licencia y atribución se conservan **por registro** en `data/processed`.

## 7. Por qué hay una única excepción documentada

La excepción de procedencia de `val.jsonl` (ver
[05-perfil-rocm.md](05-perfil-rocm.md) § 4) es la **única** desviación del contrato de
hashes en todo el proyecto, y está construida para no debilitar nada más:

- vive solo en `config/training.rocm.yaml`;
- pinnea los dos tamaños, los dos hashes de archivo y los dos hashes del registro;
- exige **exactamente un** desajuste, en la línea exacta, con los hashes exactos;
- lanza `PipelineError` si se intenta usar desde el perfil NVIDIA;
- está cubierta por tests en ambos sentidos (acepta el caso pinneado, rechaza cualquier
  variación).

Ese es el patrón a seguir si alguna vez hace falta otra excepción: declararla, acotarla
a un perfil, pinnear todo y testear que el caso incorrecto falla.

## 8. Antes de tocar código de pipeline

Lista de comprobación:

- [ ] ¿Toco un hash o manifiesto? → actualizar también el YAML que pinnea el valor.
- [ ] ¿Cambio el sampler, el contexto o el batch? → los checkpoints existentes quedan
      invalidados por diseño; archivarlos con un nombre que explique el motivo.
- [ ] ¿Relajo una verificación? → ¿qué gate aguas abajo dejará de protegerme?
- [ ] ¿Añado una fuente? → licencia en la allowlist, tier en el dedupe, categoría válida.
- [ ] ¿Cambio los parámetros de generación del benchmark? → invalida las cachés y obliga
      a regenerar todas las variantes para que sean comparables.
- [ ] `venv\Scripts\python -m pytest` y
      `python -m docker_k8s_finetune.cli config-validate`.
