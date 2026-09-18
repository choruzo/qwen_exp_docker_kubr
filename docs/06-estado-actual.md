# Estado actual

> Fotografía a **2026-09-18**, commit `b2185ea`. Verificada inspeccionando
> `artifacts/`, `benchmarks/`, `data/processed/` y los logs de la raíz, no solo
> leyendo documentación.

## 1. Resumen

| Etapa | Estado | Evidencia |
|---|---|---|
| Pipeline de datos completo | ✅ **FINAL** | `DATASHEET.md` marcado FINAL, 62 221 registros, `split_statistics.json` con `provisional: false` |
| Detección de fuga | ✅ limpio | `leakage_report.json`: `status="clean"`, 0 fugas iniciales y finales, 0 iteraciones |
| Benchmark `baseline` | ✅ **FINAL** | `provisional: false`, `completion` con las 4 etapas en `true`, 3 161 registros, 0 errores |
| Smoke test de entrenamiento | ✅ correcto | `artifacts/metrics/smoke_result.json`, `artifacts/smoke/checkpoint-2` |
| **Entrenamiento completo** | ❌ **nunca terminó** | mejor intento: step ~361 de 10 491 (~3,4 %) |
| Exportaciones | ❌ no existen | sin `artifacts/{adapter,merged,gguf}` ni `export_manifest.json` |
| Benchmark `finetuned_safetensors` | ❌ no ejecutable | depende de las exportaciones |
| Benchmark `finetuned_gguf` | ❌ no ejecutable | depende de las exportaciones |
| `report` (comparación final) | ❌ no ejecutable | falta `training_result.json` y ambas variantes |
| Perfil ROCm | ⚠️ solo smoke | sin `artifacts/rocm/` ni `benchmarks/rocm/` en el árbol actual |

**Cuello de botella único: completar con éxito un entrenamiento full.** Todo lo demás
aguas abajo está bloqueado por eso, y todo lo de aguas arriba está listo y verificado.

## 2. Qué hay en `artifacts/`

```
artifacts/
├── checkpoints/
│   └── archive/                  ← solo corridas archivadas, ningún primary/ activo
│       ├── primary_eval100_v2/
│       ├── primary_batch4_swap_v3/
│       ├── primary_batch2_v5/
│       └── primary_group_by_length_stalled_v6/
├── logs/
├── metrics/
│   ├── smoke_result.json         ← smoke correcto
│   ├── vram_smoke.jsonl
│   ├── vram_by_epoch.jsonl       ← 0 bytes: ninguna época completa jamás
│   └── tensorboard/
└── smoke/
    ├── checkpoint-2, train_results.json, all_results.json
```

**No existen**: `artifacts/adapter`, `artifacts/merged`, `artifacts/gguf`,
`artifacts/export_manifest.json`, `artifacts/metrics/training_result.json`,
`artifacts/metrics/oom_state.json`, `artifacts/rocm/`.

Que `vram_by_epoch.jsonl` esté vacío es el indicador más directo: el callback de VRAM
escribe una línea `on_epoch_end`, y nunca se cerró una época.

Los `README.md` de los checkpoints archivados son tarjetas de modelo autogeneradas por
TRL, sin información útil; el nombre del directorio es lo que documenta cada intento.

## 3. Qué hay en `benchmarks/`

```
benchmarks/
├── baseline_results.json      17 MB — FINAL
├── test_manifest.jsonl        3 111 líneas
├── llm_judge_manifest.jsonl   150 líneas
├── judge_prompt.md
├── fixtures/                  (incluye out_of_domain.jsonl)
└── work/baseline/             caché de generación
```

**No existen**: `finetuned_safetensors_results.json`, `finetuned_gguf_results.json`,
`comparison_report.md`, `training_loss.svg`, `benchmarks/rocm/`.

El baseline es un resultado sólido y reutilizable: satisface el gate que exige
`config/training.yaml` para permitir el entrenamiento completo. Sus métricas están en
[04-benchmark-y-evaluacion.md](04-benchmark-y-evaluacion.md) § 9.

## 4. Historia de los intentos de entrenamiento

Los logs de la raíz (gitignorados) documentan un proceso muy iterativo:

| Familia de logs | Intentos |
|---|---|
| `.baseline-full-v2` … `v13` | 12 intentos hasta estabilizar el baseline |
| `.baseline-judge-v1` | fase de juicio, completada |
| `.train-smoke-v2` … `v6` | 5 smokes |
| `.train-full-v1` … `v6` | 6 intentos de entrenamiento completo |

### El estancamiento de `train-full-v6` (el intento más avanzado)

Medido directamente sobre `.train-full-v6.err.log`:

| Update | Tiempo transcurrido | Velocidad |
|---|---|---|
| 100 / 10 491 | 21 min 51 s | 15,07 s/it |
| 200 / 10 491 | 51 min 26 s | 13,21 s/it |
| 300 / 10 491 | 1 h 19 min | 13,43 s/it |
| 350 / 10 491 | 1 h 32 min 39 s | 14,03 s/it |
| **351 / 10 491** | **37 h 17 min 31 s** | **38 617 s/it** |
| 361 / 10 491 | 77 h 45 min 42 s | 8 976 s/it (ETA: 25 258 h) |

**Un solo update pasó de ~14 segundos a ~35,7 horas**, sin lanzar OOM y sin traceback
alguno (un grep por error/traceback/oom/exception en el `.err.log` no encuentra nada).
El directorio resultante quedó archivado como `primary_group_by_length_stalled_v6`.

Esta es exactamente la patología descrita en el `Readme.md`: `LengthGroupedSampler` se
construye con el **batch efectivo** (16), ordena megabloques de 800 ejemplos y concentra
16 secuencias cercanas a 8192 tokens en un mismo update. El commit `a5ac099` cambió
`train_sampling_strategy` de `group_by_length` a `random` precisamente por esto.

### Qué falta por confirmar

**No hay evidencia en `artifacts/` de que se haya vuelto a lanzar un entrenamiento
completo después del fix del sampler.** El siguiente paso natural del proyecto es
relanzarlo con la configuración actual y comprobar que los updates se mantienen en el
orden de ~14 s a lo largo de los 10 491.

Los otros intentos archivados corresponden a las decisiones documentadas en
[08-decisiones-e-historial.md](08-decisiones-e-historial.md):

| Directorio archivado | Qué se aprendió |
|---|---|
| `primary_eval100_v2` | evaluación cada 100 updates resultaba excesiva → 500 |
| `primary_batch4_swap_v3` | batch 4 llegó a 15,99 GiB y forzó swap en Docker/WSL |
| `primary_batch2_v5` | batch 2 × acum. 8 llegó a 15,88 GiB y bloqueó el avance |
| `primary_group_by_length_stalled_v6` | `group_by_length` estanca un update durante ~36 h |

### El smoke más reciente (`.train-smoke-v6.out.log`)

Limpio: 2 steps, `train_loss = 3.32`, `eval_loss = 1.03`, pico de 3,9 GiB asignados /
5,09 GiB reservados sobre 15,93 GiB, `checkpoint_fingerprint` generado, sin errores. El
camino de código funciona; el problema es de escala temporal, no de corrección.

## 5. Estado del árbol de trabajo

```
?? .baseline-full-v9.err - copia.log     ← copia accidental, gitignorada
?? .cache - copia/                       ← copia accidental de la caché HF
?? CLAUDE.md                             ← sin commitear
```

`CLAUDE.md` es contenido de valor y está sin versionar; las dos "copias" son ruido
recuperable. Esta carpeta `docs/` también está sin commitear.

## 6. Ruta crítica hasta un resultado final

1. Relanzar `train` completo con el sampler aleatorio y verificar que la velocidad por
   update se mantenga (~14 s/it → ~41 h para 10 491 updates en la ruta NVIDIA).
   Alternativa: la ruta ROCm, con estimación de 2 a 5 días.
2. Confirmar que se generan `artifacts/{adapter,merged,gguf}` y
   `export_manifest.json`, con ambas cuantizaciones `q4_k_m` y `q8_0`.
3. `benchmark --variant finetuned_safetensors` (3 fases: generación → sintaxis → juez).
4. Arrancar llama-server con el GGUF exportado y repetir para `finetuned_gguf`.
5. `report` — exigirá `training_result.json` con VRAM y filtro de longitud, y
   coincidencia estricta de test, parámetros de generación e identidad del juez entre
   las tres variantes.

Ninguno de esos pasos admite atajos: los gates descritos en
[07-invariantes-y-gates.md](07-invariantes-y-gates.md) los bloquean explícitamente.
