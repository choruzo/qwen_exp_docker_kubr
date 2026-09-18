# Perfil ROCm: Radeon AI PRO R9700

> Introducido en el commit `b2185ea` (2026-09-18), el más reciente del repositorio.
> Es una **ruta independiente**: no sustituye ni reanuda la ruta NVIDIA.

## 1. Archivos propios

| Archivo | Rol |
|---|---|
| `config/training.rocm.yaml` | perfil de entrenamiento |
| `config/benchmark.rocm.yaml` | perfil de benchmark (usar con `--config`) |
| `Dockerfile.train.rocm` | imagen ROCm |
| `compose.train.rocm.yaml` | servicio con `/dev/kfd`, `/dev/dri`, `group_add`, `seccomp:unconfined` |

Salidas en `artifacts/rocm/*` y `benchmarks/rocm/*`. **Nunca** se reanudan checkpoints
de la ruta NVIDIA.

## 2. Diferencias frente al perfil NVIDIA

| Aspecto | NVIDIA | ROCm |
|---|---|---|
| Cuantización | QLoRA 4-bit | LoRA BF16 **sin cuantizar** |
| Optimizador | `adamw_8bit` (bitsandbytes) | `adamw_torch` |
| `per_device_eval_batch_size` | 4 | 1 |
| `resume_from_checkpoint` | `auto` | **`none`** |
| Rutas de salida | `artifacts/*` | `artifacts/rocm/*` |
| `runtime.accelerator` | cuda | rocm |
| `runtime.expected_gfx` | — | `gfx1201` |
| `runtime.require_local_model` | — | true |
| `smoke_test.input` | `data/interim/validated/chatml.jsonl` | `data/processed/train.jsonl` (+ `val.jsonl`) |
| `smoke_test.max_steps` | 2 | 1 |
| `smoke_test.eval_steps` / `save_steps` | defaults | 1 / 1 |
| Excepción de hash de `val.jsonl` | **no aplica** | sí, pinneada |

Todo lo demás es idéntico: mismos hashes del modelo base, mismos hashes de train y
test, LoRA r=32/α=32, 3 épocas, batch 1 × acumulación 16, lr 2e-4, cosine, semilla 3407,
pérdida solo en respuestas, texto solo, visión congelada.

El smoke ROCm toma registros directamente de los `train.jsonl` y `val.jsonl`
**congelados**, no del intermedio.

### Invariantes forzados en código

`validate_training_config` (`train/core.py:441-453`) rechaza el arranque si, con
`accelerator: rocm`:

- `load_in_4bit` no es `False` o `dtype` no es `bfloat16`
  → *"ROCm profile requires unquantized BF16 LoRA"*;
- `optimizer != "adamw_torch"`
  → *"requires the native torch AdamW optimizer"*;
- `resume_from_checkpoint != "none"`
  → *"must start without automatic checkpoint resume"*;
- cualquiera de checkpoints/adapter/merged/gguf/metrics/manifest no empieza por
  `artifacts/rocm/`.

`runtime.expected_gfx` se compara contra
`torch.cuda.get_device_properties(0).gcnArchName` (`train/runner.py:337-341`), y
`require_local_model` impide caer al repo remoto si el modelo local está incompleto.

## 3. Estado de la comprobación (18-09-2026)

### Host

| Componente | Valor |
|---|---|
| GPU | Radeon AI PRO R9700, gfx1201, 32 624 MB |
| Driver | `amdgpu 7.1.3.31500000` |
| ROCm del host | 10.0.0 |
| SO | Ubuntu 24.04.5, kernel 7.0.0-31-generic |

⚠️ La matriz oficial de ROCm 10.0.0 incluye gfx1201 pero valida Radeon sobre Ubuntu
24.04.4 con kernel HWE 6.17. **La combinación actual del host queda fuera de esa
matriz**, y por eso el smoke empírico es obligatorio. La imagen ROCm de Unsloth está
fijada por digest y usa ROCm 7.2.4, así que su interoperabilidad con el driver del host
debe verificarse antes de un entrenamiento largo.

### Stack verificado dentro de la imagen

PyTorch `2.12.1+rocm7.2`, HIP `7.2.53211`, `torch.cuda.is_available() == True`, gfx1201
detectado; Unsloth `2026.9.4`, Transformers `5.5.0`, TRL `0.24.0`, PEFT `0.20.0`,
TensorBoard `2.20.0`, sentence-transformers `5.2.0`, llama.cpp commit
`44be98f057e9f9902a8ee12630e181c7f8ec2953`. La imagen detecta `/dev/kfd` y
`/dev/dri/renderD128`.

Unsloth empleó su implementación de PyTorch al faltar `flash-linear-attention` y
`causal-conv1d`; el smoke BF16 terminó con `adamw_torch`.

### Mediciones

| Prueba | Duración | Resultado |
|---|---|---|
| Smoke de 1 paso (con evaluación y checkpoint) | 136 s | `train_loss=0.3450`, `eval_loss=1.9086`, pico 10,68 GiB |
| Calibración de 2 pasos (registro de 7 906 tokens, validación de 7 529, contexto 8192) | 168 s | pico **22,66 GiB**, 9,20 GiB libres |

La auditoría de longitudes halló los mismos 57 registros de train y 2 de validación por
encima de 8192 tokens; `overlength_action: exclude` los excluiría registrando conteos y
hashes.

**Estimación de 3 épocas completas: 2 a 5 días**, según la distribución de longitudes,
evaluaciones y guardados. Reservar al menos 40 GiB para salidas, además de imagen,
modelo y datos.

### Export del smoke

Unsloth fusionó pesos BF16, convirtió a GGUF y generó:

| Archivo | Bytes |
|---|---|
| `Q4_K_M` | 2 783 446 720 |
| `Q8_0` | 4 610 580 160 |
| `BF16-mmproj` | 675 568 864 |

El finalizador verificó las cuantizaciones y reubicó los tres archivos en
`artifacts/rocm/export_smoke`. **Son experimentales; no son el resultado de un
entrenamiento completo.**

## 4. Excepción de procedencia de `val.jsonl`

Esta es la desviación más delicada del proyecto y merece leerse completa antes de tocar
la lógica de validación.

### El hecho

| Fuente | Bytes | SHA-256 |
|---|---|---|
| `val.jsonl` real (seguido por Git, copiado desde Windows) | 9 581 643 | `bcf223584546fe573830470fcffedcffe92817034078808934bc892f6cbf90ac` |
| `split_statistics.json` (manifiesto original) | 9 581 692 | `bb5c3c04504fd7d307e4f3305ffae9c8ab5765882e85964f9bcfccbb2b25fe4a` |

### La causa

Una **credencial redactada** en el registro de validación de la línea 1659,
`source_record_id: q50271985-a50414502` (pregunta de Stack Overflow 50271985):

```
recorded_content_hash: fc685c9a4d4423bb10ceb4fbd9cca6d18af2acc6a9de0e6a09e5bee12adda42a
redacted_content_hash: 15411a00b27deb7c841006dbe1fb9e9c310ad9844f25bff6b5d870e88ff66558
```

Los 3 110 registros restantes conservan su `content_hash` original. Los 3 111 coinciden
con sus asignaciones del manifiesto y con los recuentos por categoría y fuente.

### La decisión

- **No se restaura la credencial** ni se modifican los archivos de datos o el manifiesto
  histórico.
- La excepción se declara **solo** en `config/training.rocm.yaml`
  (`data.validation_redaction`), que pinnea ambos hashes de archivo, los dos tamaños y
  las huellas anterior y actual de ese registro.
- La ruta NVIDIA **sigue exigiendo el hash original sin excepciones**.
- La identidad efectiva de validación que se guarda en los checkpoints es el SHA-256 del
  archivo redactado.

### Cómo se hace cumplir

`_verify_final_split` / `_verify_validation_redaction` (`train/runner.py:53-135`): si el
hash de `validation` no coincide con el manifiesto, se intenta la excepción; se auditan
línea a línea los `content_hash` reales contra los grabados y se exige que **exactamente
un** desajuste ocurra y coincida con los cuatro valores pinneados (`record_line`,
`source_record_id`, `recorded_content_hash`, `redacted_content_hash`). Si no,
`PipelineError("Validation record hashes do not match the pinned redaction exception")`.

Si `accelerator != "rocm"`, lanza directamente
`PipelineError("Validation redaction exception is restricted to the ROCm profile")`.

`tests/test_train_core.py` cubre ambos lados: acepta el desajuste pinneado exacto y
rechaza cualquier `redacted_content_hash` incorrecto.

## 5. Comandos

```bash
PATH=/tmp/qwen-git-lfs/usr/bin:$PATH git lfs ls-files
sha256sum data/processed/{train,val,test}.jsonl
docker compose -f compose.train.rocm.yaml config --quiet
docker compose -f compose.train.rocm.yaml build train
docker compose -f compose.train.rocm.yaml run --rm train python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
docker compose -f compose.train.rocm.yaml run --rm train python -m docker_k8s_finetune.cli train --config config/training.rocm.yaml --smoke-test --preflight --no-export
docker compose -f compose.train.rocm.yaml run --rm train python -m docker_k8s_finetune.cli train --config config/training.rocm.yaml --smoke-test --no-export
```

Preflight completo (audita los 3 111 registros de validación y acepta solo la excepción
exacta):

```bash
docker compose -f compose.train.rocm.yaml run --rm train python -m docker_k8s_finetune.cli train --config config/training.rocm.yaml --preflight --no-export
```

Entrenamiento completo, **solo tras autorización explícita**:

```bash
docker compose -f compose.train.rocm.yaml run --rm train python -m docker_k8s_finetune.cli train --config config/training.rocm.yaml
```

El baseline publicado previamente sigue siendo el requisito de entrenamiento según
`config/training.rocm.yaml`. Para evaluar las exportaciones ROCm se usa
`config/benchmark.rocm.yaml` con `--config` en `benchmark`, `benchmark-syntax`,
`benchmark-judge` y `report`.

## 6. Nota sobre reutilizar un adaptador

Es posible cargar únicamente los pesos de un adaptador LoRA compatible y validado como
**inicialización de un experimento nuevo**, pero sus estados de optimizador, scheduler y
RNG no equivalen a reanudar el entrenamiento: la procedencia de ese experimento debe
declararse aparte.
