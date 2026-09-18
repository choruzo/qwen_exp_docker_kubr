# Modelo y entrenamiento

> Fuentes: `config/training.yaml`, `config/training.rocm.yaml`,
> `src/docker_k8s_finetune/train/{core.py,runner.py,callbacks.py}`, `Readme.md`,
> `tests/test_train_core.py`.

## 1. Identidad del modelo base

Qwen3.5-4B **no es un LLM textual clásico**: el checkpoint declara
`Qwen3_5ForConditionalGeneration`. Por eso el runner lo carga con `FastVisionModel`
de Unsloth, congela las capas de visión y entrena solo lenguaje, atención y MLP
(`Readme.md:124-127`).

| Campo | Valor | Referencia |
|---|---|---|
| Nombre | `unsloth/Qwen3.5-4B` | `config/training.yaml:5` |
| Revisión (commit) | `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636` | `config/training.yaml:6` |
| Ruta local preferida | `Modelo/Qwen3.5-4B` | `config/training.yaml:7-8` |
| Arquitectura declarada | `qwen3_5_vlm_text_only` | `config/training.yaml:18` |
| Loader | `FastVisionModel` | `config/training.yaml:19` |
| Contexto máximo | 8192 | `config/training.yaml:21` |
| dtype | `bfloat16` | `config/training.yaml:22` |
| Semilla | 3407 | `config/training.yaml:2` |

### SHA-256 pinneados (idénticos en el perfil ROCm)

Ocho archivos quedan fijados en `config/training.yaml:9-17`. Todos son hashes
SHA-256 de 64 caracteres hexadecimales:

| Archivo | SHA-256 |
|---|---|
| `config.json` | `d19a861ea5090c9ec32542301484f78fa9948407d7889751ad26039059255bc0` |
| `model.safetensors.index.json` | `e758f61f5ffe855e274081190ed1cde2950fae8b556c5b38eeb6818ef6c1417d` |
| `model.safetensors-00001-of-00002.safetensors` | `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61` |
| `model.safetensors-00002-of-00002.safetensors` | `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188` |
| `tokenizer.json` | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |
| `tokenizer_config.json` | `b4e0223934cafd892219040bceed98fb2963e9d625c5e9c39cfbb22c039d6cf3` |
| `processor_config.json` | `bcd178c4fd1f8679d38aa195b7f2e2a907aa2f2f4462dfd45dd2d69b6423c0fc` |
| `chat_template.jinja` | `dca767ce96946decf01e87358183070ce953a06a1033f06e92295193d8a432f6` |

### Cómo se verifica la identidad

1. `resolve_model_reference` (`train/core.py:29-39`) usa la ruta local si están
   presentes `config.json`, `tokenizer_config.json`, `model.safetensors.index.json`
   y al menos un `*.safetensors`. Si falta algo y `runtime.require_local_model`
   está activo (caso ROCm), aborta en lugar de caer al repo remoto.
2. `verify_local_model_provenance` (`train/core.py:42-92`) recalcula el SHA-256 de
   cada archivo pinneado y de cada shard del `weight_map`, comparando
   case-insensitive; el primer desajuste lanza `PipelineError`. Si el directorio
   local es un repo Git, ejecuta `git rev-parse HEAD` con `safe.directory` acotado
   **al comando** (nunca global) y compara con la revisión esperada.
3. `inspect_local_model` (`train/core.py:206-235`) exige
   `architectures[0] == "Qwen3_5ForConditionalGeneration"`, comprueba que existan
   todos los shards indexados y verifica heurísticamente que la plantilla de chat
   sea compatible con el masking *response-only*.
4. El runner llama a (1) y (2) antes de `FastVisionModel.from_pretrained`
   (`train/runner.py:366-378`); para referencias remotas pasa `revision=` explícita.

## 2. Congelamiento de capas y LoRA

| Flag | Valor |
|---|---|
| `text_only` | true |
| `finetune_vision_layers` | false (visión congelada) |
| `finetune_language_layers` | true |
| `finetune_attention_modules` | true |
| `finetune_mlp_modules` | true |

(`config/training.yaml:24-29`, idéntico en ROCm.)

| Parámetro LoRA | Valor |
|---|---|
| `r` | 32 |
| `alpha` | 32 |
| `dropout` | 0.0 |
| `bias` | none |
| `target_modules` | `auto` |
| `use_gradient_checkpointing` | `unsloth` |

`target_modules: auto` es **deliberado**, no una omisión: Qwen3.5 mezcla
proyecciones de atención completa y de atención lineal, y solo `FastVisionModel`
selecciona ambas familias a partir de los flags anteriores
(`config/training.yaml:36-38`). `validate_training_config` rechaza explícitamente
cualquier lista manual de módulos (`train/core.py:467-468`).

## 3. Hiperparámetros del trainer

| Parámetro | NVIDIA | ROCm |
|---|---|---|
| epochs | 3 | 3 |
| `per_device_train_batch_size` | 1 | 1 |
| `gradient_accumulation_steps` | 16 | 16 |
| batch efectivo | 16 | 16 |
| learning rate | 2e-4 | 2e-4 |
| scheduler | cosine | cosine |
| `warmup_ratio` | 0.03 | 0.03 |
| `weight_decay` | 0.01 | 0.01 |
| optimizador | `adamw_8bit` | `adamw_torch` |
| `per_device_eval_batch_size` | 4 | 1 |
| `eval_steps` | 500 | 500 |
| `logging_steps` | 10 | 10 |
| `save_steps` | 25 | 25 |
| `save_total_limit` | 3 | 3 |
| `response_only_loss` | true | true |
| `resume_from_checkpoint` | `auto` | `none` |
| `packing` | false | false |
| `train_sampling_strategy` | random | random |
| `overlength_action` | exclude | exclude |
| `load_in_4bit` | true (QLoRA) | false (LoRA BF16) |

Delimitadores de masking: `<|im_start|>user\n` (instrucción) y
`<|im_start|>assistant\n` (respuesta).

### Por qué batch 1 × acumulación 16

Es el **único perfil validado** en la RTX 5060 Ti 16 GB (`Readme.md:138-156`):

| Perfil probado | Pico VRAM | Resultado |
|---|---|---|
| batch 4 | 15,99 GiB (59-64 MiB libres) | swap en Docker/WSL; no reproducible |
| batch 2 × acum. 8 | 15,88 GiB (173 MiB libres) | bloqueo del avance |
| **batch 1 × acum. 16** | 9,4-11,2 GiB normal; 15,72 GiB puntual | seguro, swap ~115 MiB |

### Por qué sampler aleatorio y no `group_by_length`

Con batch físico 1 no hay padding entre ejemplos que ahorrar agrupando por
longitud. Transformers 5.2 construye `LengthGroupedSampler` con el **batch
efectivo** (16), ordena megabloques de 800 ejemplos y concentra 16 secuencias
largas en un mismo update: en la prueba real los primeros 350 updates iban a ~14 s
y el siguiente megabloque largo elevó cada update a **horas sin producir OOM**.
El orden aleatorio determinista distribuye esas secuencias entre ventanas de
acumulación (`Readme.md:147-153`).

### Validación

Batch de evaluación 4 cada 500 updates → 21 puntos intermedios de `val_loss` en
las 3 épocas sobre 3109 ejemplos, en vez de repetir 104 veces un barrido completo.
El smoke conserva batch de evaluación 1 (`Readme.md:169-172`).

## 4. Fallback ante OOM

Perfiles, en orden (`config/training.yaml:66-76`; idénticos en ROCm):

| Perfil | `max_seq_length` | batch | acumulación |
|---|---|---|---|
| primary | 8192 | 1 | 16 |
| `context_reduction_4k` | 4096 | 1 | 16 |
| `context_reduction_2k` | 2048 | 1 | 16 |

- Solo se reduce **contexto**, nunca se sube el batch: `validate_training_config`
  rechaza perfiles que no reduzcan memoria de forma monótona.
- Cada perfil escribe en su propio subdirectorio de `artifacts/checkpoints/<perfil>`
  (`build_attempt`, `train/core.py:394-431`), impidiendo colisiones.
- Solo se considera recuperable un `RuntimeError` que sea OOM de CUDA
  (`_is_cuda_oom`, `train/runner.py:255-256`); cualquier otro se re-lanza. El smoke
  test nunca hace fallback.
- Tras un OOM: se registra el perfil, se persiste el estado, se vacía la caché CUDA
  y se fuerza `gc.collect()` antes del siguiente intento.

### `artifacts/metrics/oom_state.json`

Guarda `{version, updated_at, contract, failed_profiles}` con escritura atómica.
El `contract` es un hash de `{config de entrenamiento completa, SHA-256 de
split_statistics.json}` (`_oom_state_contract`, `train/runner.py:259-265`): si no
coincide, el estado se ignora como si no existiera. Al reanudar tras un corte de
energía, los perfiles ya agotados se omiten — pero **siempre se reintenta el último
perfil disponible** (`train/runner.py:644-648`). Al terminar correctamente, el
archivo se borra.

## 5. Registros por encima del contexto

`filter_formatted_by_token_length` (`train/core.py:293-346`) tokeniza en lotes de
256 sin truncar (`truncation=False`) y **excluye** —nunca trunca— los registros que
exceden el `max_seq_length` del perfil activo (8192, o 4096/2048 en fallback).
`validate_training_config` exige `overlength_action == "exclude"`
(`train/core.py:475-476`): el truncamiento silencioso está prohibido.

El reporte incluye `max_length`, `input_records`, `kept_records`,
`excluded_records`, `excluded_rate`, `maximum_observed_tokens`, `p99_tokens`, la
lista completa de excluidos (`content_hash`, `category`, `source`, `tokens`) y
`kept_content_hashes_sha256`: la **huella ordenada del subconjunto realmente
entrenado**.

En el split congelado actual: **57 registros de train (0,102 %) y 2 de validación
(0,064 %)** superan 8192 tokens (`Readme.md:178-183`). Los dos edge cases largos
del test se conservan para evaluar robustez y no intervienen en el entrenamiento.

Auditoría manual:

```
docker compose -f compose.train.yaml run --rm train python scripts/audit-token-lengths.py
```

## 6. Checkpoints y resume

`latest_checkpoint` (`train/core.py:349-391`) recorre `output_dir/checkpoint-*` y
solo acepta un checkpoint si:

- existen y no están vacíos `trainer_state.json`, `optimizer.pt`, `scheduler.pt`,
  `rng_state.pth` y al menos uno de
  `adapter_model.safetensors` / `model.safetensors` / `pytorch_model.bin`;
- `trainer_state.json["global_step"]` coincide con el número del directorio;
- si se pasa `expected_fingerprint`, el `checkpoint_provenance.json` coincide.

Devuelve el step más alto que cumpla todo, ignorando directorios incompletos.

### `checkpoint_fingerprint`

Hash de un `checkpoint_contract` (`train/runner.py:424-451`) que incluye: hash de
la config completa, identidad del modelo base, hashes de train/val/test, perfil de
entrenamiento (nombre, seq len, batch, acumulación, eval batch/steps, packing,
sampling), clases de processor/tokenizer y el reporte de filtrado por longitud. Un
checkpoint solo se reanuda si **nada** relevante cambió.

`build_checkpoint_provenance_callback` (`train/callbacks.py:48-67`) escribe
atómicamente `checkpoint_provenance.json` con `{version, fingerprint, contract}` en
cada `on_save`, y lanza `RuntimeError` si el directorio esperado no existe.

## 7. Métricas registradas

`build_vram_callback` (`train/callbacks.py:10-45`):

- `on_train_begin`: resetea el archivo de métricas (salvo `append=True` al reanudar)
  y `torch.cuda.reset_peak_memory_stats()`;
- `on_epoch_begin`: resetea picos;
- `on_epoch_end`: añade una línea JSONL con `run_id`, `epoch`, `global_step`,
  `peak_allocated_gib`, `peak_reserved_gib`, `device_used_gib`, `device_free_gib`,
  `device_total_gib`, `device`.

TensorBoard está activo (`report_to: tensorboard`); los picos de VRAM por época y
las métricas viven en `artifacts/metrics`.

## 8. Salidas: `training_result.json` y `export_manifest.json`

`artifacts/metrics/training_result.json` (o `smoke_result.json` en smoke)
— `train/runner.py:523-567`:

- `status`, `smoke_test`, `training_profile`, `model_reference`
- `runtime`: accelerator, versión de torch, HIP, CUDA, nombre de GPU, `gcn_arch`,
  memoria total
- `max_seq_length`, `batch_size`, `gradient_accumulation_steps`,
  `effective_batch_size`, `eval_batch_size`, `eval_steps`, `packing`,
  `train_sampling_strategy`
- `resumed_from`, `metrics`, `evaluation_metrics`, `log_history` completo
- `length_filter` (train y validación)
- `vram` (agregado por época) y `vram_final`
- `checkpoint_contract` y `checkpoint_fingerprint`
- En entrenamiento completo: `training_split` (hashes train/val/test),
  `validation_identity`, `base_model`
- Si hubo fallback: `oom_fallback_used`, `oom_failed_profiles`

`export_manifest.json` (`train/runner.py:594-612`), en `artifacts/` (NVIDIA) o
`artifacts/rocm/` (ROCm):

- `version`, `created_at`, `base_model`
- `training_split` (hashes exactos usados), `training_length_filter`
- `training_contract`: `{fingerprint, contract}`
- `artifacts`: `{adapter, merged, gguf}`, cada archivo con `{path, bytes, sha256}`
- `gguf_quantizations`: `[q4_k_m, q8_0]`

El export exige `split_statistics` verificados; desde un run smoke lanza
`PipelineError("Full export requires verified split statistics")`
(`train/runner.py:592-593`).

### GGUF

`model.save_pretrained_gguf` de Unsloth escribe en `<merged_dir>_gguf`.
`finalize_gguf_export` (`train/core.py:117-173`) verifica que el directorio
reportado sea exactamente el esperado, que cada `.gguf` esté dentro de él y que
existan **ambas** cuantizaciones requeridas (comparación normalizada sin guiones ni
underscores); solo entonces traslada el directorio a `artifacts/gguf`.

## 9. Gate de baseline

El entrenamiento completo se niega a arrancar hasta que
`benchmarks/baseline_results.json` sea final: sintaxis y LLM-juez completos, y hash
correspondiente exactamente al test congelado. `_verify_baseline` exige además
coincidencia estricta de `judge_provenance` e identidad del juez (alias, filename,
cuantización) y falla si cambia el prompt del juez o el split de test. **El smoke
test no requiere baseline.**

## 10. Comandos

```
venv\Scripts\python -m docker_k8s_finetune.cli train --smoke-test --preflight --no-export
docker compose -f compose.train.yaml run --build --rm train python -m docker_k8s_finetune.cli train --smoke-test --no-export
docker compose -f compose.train.yaml run --build --rm train python -m docker_k8s_finetune.cli train
```

## 11. Invariantes cubiertos por tests

`tests/test_train_core.py` cubre, entre otros:

- carga/formato ChatML con offset y límite;
- `resolve_text_tokenizer` desenvolviendo el tokenizer de un processor multimodal;
- exclusión (no truncado) por longitud y exactitud del reporte;
- detección de `hash mismatch` ante cambio de bytes del modelo local;
- `safe.directory` acotado al comando en la verificación de revisión Git;
- determinismo de `build_attempt` y unicidad de `output_dir`;
- `latest_checkpoint` con archivos incompletos y con `expected_fingerprint`;
- `_warmup_steps` en smoke y en completo;
- rechazos de `validate_training_config`: módulos manuales, sampling no aleatorio
  con batch 1, eval batch/steps no positivos, perfiles OOM no monótonos;
- lógica completa de fallback OOM, incluida la reanudación tras corte de energía;
- reglas ROCm (rechazo de 4-bit y de resume automático);
- excepción de redacción de validación (acepta exactamente el desajuste pinneado);
- `artifact_files_manifest` / `verify_export_artifact` / `finalize_gguf_export`.

Ver también [05-perfil-rocm.md](05-perfil-rocm.md) y
[07-invariantes-y-gates.md](07-invariantes-y-gates.md).
