# Decisiones de diseño e historial

Casi todas las decisiones no obvias de este repositorio vienen de una medición, no de una
preferencia. Esta página las recoge con su motivo empírico, para que nadie las revierta
por parecer arbitrarias.

## 1. Decisiones de entrenamiento

### Batch 1 × acumulación 16 (no batch 2 ni 4)

| Perfil | Pico VRAM | Margen libre | Resultado |
|---|---|---|---|
| batch 4 | 15,99 GiB | 59-64 MiB | swap en Docker/WSL (13,9 GiB de RAM); no reproducible |
| batch 2 × acum. 8 | 15,88 GiB | 173 MiB | pasos cortos más rápidos, pero un lote largo bloqueó el avance |
| **batch 1 × acum. 16** | 9,4-11,2 GiB habitual; 15,72 GiB puntual | amplio | seguro; swap ~115 MiB |

El batch efectivo sigue siendo 16 en los tres casos, así que no se pierde nada en
dinámica de optimización. Archivado: `primary_batch4_swap_v3`, `primary_batch2_v5`.

### Sampler aleatorio, no `group_by_length`

Con batch físico 1 **no hay padding entre ejemplos que ahorrar** agrupando por longitud:
el beneficio habitual de `group_by_length` simplemente no existe aquí. Y el coste sí:
Transformers 5.2 construye `LengthGroupedSampler` con el batch **efectivo** (16), ordena
megabloques de 800 ejemplos y concentra 16 secuencias largas en un mismo update.

Medición directa sobre `.train-full-v6.err.log`:

| Update | Transcurrido | s/it |
|---|---|---|
| 350 | 1 h 32 min | 14,03 |
| **351** | **37 h 17 min** | **38 617** |

Un único update pasó de ~14 s a ~35,7 horas, **sin OOM y sin traceback**. Archivado como
`primary_group_by_length_stalled_v6`; corregido en el commit `a5ac099`.

Es el fallo más instructivo del proyecto: no se manifestó como un error, sino como un
proceso que parecía seguir vivo.

### Evaluación cada 500 updates con batch 4

Conserva 21 puntos intermedios de `val_loss` en las 3 épocas, además del final, sin
repetir 104 veces un barrido completo de los 3 109 ejemplos de validación. Un intento
previo con evaluación cada 100 updates quedó archivado como `primary_eval100_v2`.
El smoke mantiene batch de evaluación 1 para minimizar consumo.

### Fallback OOM que solo reduce contexto

8192 → 4096 → 2048, siempre con batch 1 × acumulación 16. Subir el batch para compensar
contradiría la medición anterior, y `validate_training_config` rechaza perfiles que no
reduzcan memoria monótonamente. Cada perfil escribe en su propio subdirectorio de
checkpoints para que un fallback no contamine al principal.

### Excluir en vez de truncar los registros largos

57 registros de train (0,102 %) y 2 de validación (0,064 %) superan 8192 tokens. Se
excluyen registrando hash, categoría y longitud, más la huella ordenada del subconjunto
realmente entrenado. Truncar silenciosamente produciría ejemplos con la respuesta cortada
—exactamente el patrón que no se quiere enseñar— y además haría imposible describir qué
se entrenó. Los dos edge cases largos del test **sí** se conservan, para medir robustez.

### `target_modules: auto`

No es una omisión: Qwen3.5 mezcla proyecciones de atención completa y de atención lineal
híbrida, y solo `FastVisionModel` selecciona ambas familias a partir de los flags de
capas. Una lista manual dejaría fuera una de las dos. El validador rechaza cualquier
lista explícita para que nadie "mejore" esto por error.

### `FastVisionModel` con visión congelada

Qwen3.5-4B es `Qwen3_5ForConditionalGeneration`, no un LLM textual clásico. Cargarlo como
modelo de texto fallaría o ignoraría parte de la arquitectura. Se carga como modelo de
visión, se congelan las capas de visión y se entrena lenguaje + atención + MLP.

## 2. Decisiones de infraestructura

### Imagen oficial de Unsloth fijada por digest

Es la ruta elegida para Blackwell (sm_120): incluye el stack CUDA/PyTorch preparado y
evita recompilar `xformers` para sm_120 en Windows. Coste: la imagen derivada validada
ocupa ~42 GB, así que hace falta un disco Docker de al menos 50 GB — un límite de 20 GB
**no** es compatible con esta ruta.

La imagen base trae Transformers 4.57.6, que **no reconoce Qwen3.5**; la capa derivada
instala 5.2.0 y verifica las versiones con `assert` en runtime. También anula el
`ENTRYPOINT` original, que haría `chmod` recursivo sobre todo el modelo y los datasets.

### Dedupe semántico solo dentro de Docker

Sentence-Transformers y FAISS están pinneados en la imagen; ejecutar el dedupe con otras
versiones produciría clusters distintos y por tanto un split distinto, rompiendo la
comparabilidad con el baseline ya medido.

### Sintaxis desde Windows, no en el contenedor

kubeconform y hadolint corren como contenedores con digest fijo; hacerlo desde dentro de
la imagen de entrenamiento exigiría Docker-in-Docker. Se ejecuta desde Windows, donde el
proceso tiene el daemon directamente.

### Inferencia y juicio en fases separadas

Con una sola GPU de 16 GB no caben a la vez el modelo evaluado y el juez Gemma 4 12B Q6.
Primero se guardan todas las predicciones, se libera la GPU, se arranca el juez y se
completa el juicio de forma reanudable.

## 3. Decisiones de datos

### Reverse-instruction con gate manual

Es el único paso manual obligatorio. Generar instrucciones sintéticas a escala sin haber
mirado una muestra produce un dataset cuya calidad nadie ha verificado; el gate exige
revisar 100 candidatos, copiar el SHA-256 del JSONL revisado y marcar `approved` antes de
permitir el `--scope full`. Resultado registrado: 95 aceptadas, 5 a cuarentena.

También se rechazan automáticamente las instrucciones **no autosuficientes** (regex
`_NON_SELF_CONTAINED`): una instrucción que dice "como se explicó arriba" no es un par
válido fuera de su documento de origen.

### Agrupar por `semantic_cluster_id` en el split

Estratificar por `(category, source)` sin agrupar por cluster partiría familias de
ejemplos casi idénticos entre train y test, que es la forma más común de inflar métricas
sin darse cuenta. El cluster es la unidad indivisible; luego una auditoría por embeddings
con umbral 0,92 vuelve a comprobarlo y repara moviendo a train con backfill del mismo
estrato. Resultado actual: 0 fugas.

### Cuarentena por licencia, no por calidad

`ComponentSoft/k8s-kubectl-35k`, `…-cot-20k` y `kubernetes-failure-stories` están en
cuarentena por licencia `unknown`, no porque los datos sean malos. Se quedan fuera del
dataset final hasta que la licencia sea verificable.

### Ordenar los lotes de benchmark por longitud del **prompt**

Reduce padding sin mirar nunca la longitud de la respuesta de referencia: consultar el
target para planificar lotes filtraría información del conjunto de evaluación en el
proceso de evaluación.

### Truncamiento con reintento único

Si una salida alcanza 2048 tokens sin EOS, se reintenta **una vez** con
`repetition_penalty` 1.10 para cortar bucles degenerados, y se registran intentos, tokens
y latencia acumulada para no ocultar el coste del rescate. Terminar exactamente en el
último token permitido no cuenta como truncamiento.

### La excepción de `val.jsonl`

Una credencial redactada cambió el SHA-256 del archivo respecto al manifiesto histórico.
La decisión fue **no restaurar la credencial ni reescribir el manifiesto**, sino declarar
la excepción de forma quirúrgica en el perfil ROCm: dos tamaños, dos hashes de archivo,
dos hashes del registro, exactamente un desajuste permitido en una línea concreta, y
`PipelineError` si se intenta usar desde NVIDIA. Detalle en
[05-perfil-rocm.md](05-perfil-rocm.md) § 4.

## 4. Historial de commits

| Commit | Fecha | Contenido |
|---|---|---|
| `cd39319` | 2026-08-04 | first commit |
| `4b6477d` | 2026-08-04 | `.gitignore` para el virtualenv |
| `80cd9a4` | 2026-08-04 | documentación inicial del pipeline |
| `48fe68c` | 2026-08-05 | validación de YAML de Kubernetes y Dockerfiles |
| `b7bbc39` | 2026-08-05 | informe de benchmark y evaluación de completitud |
| `682689b` | 2026-09-18 | **el commit grande**: baseline completo (90 k líneas de JSON), split congelado, manifiestos, scripts PowerShell, reescritura de `benchmark/`, `config.py`, `normalize/generate.py` |
| `a5ac099` | 2026-09-18 | batch 2×8 → 1×16, `group_by_length` → `random`, fallbacks OOM solo de contexto |
| `b2185ea` | 2026-09-18 | perfil ROCm reproducible (Radeon AI PRO R9700) |

Lectura del historial: agosto fue construcción del pipeline; el 18 de septiembre de 2026
concentra el trabajo de puesta a punto real — congelar los datos, medir el baseline
completo, corregir el perfil de entrenamiento con lo aprendido de seis intentos fallidos,
y abrir una segunda ruta de hardware.

### Pendiente de versionar

`CLAUDE.md` y esta carpeta `docs/` están sin commitear. En el árbol también hay dos
copias accidentales gitignoradas (`.baseline-full-v9.err - copia.log`, `.cache - copia/`).

## 5. Lecciones transferibles

1. **Un proceso lento no siempre es un proceso vivo.** El estancamiento de
   `group_by_length` no produjo error alguno; solo se detectó comparando s/it entre
   updates consecutivos. Vale la pena vigilar la derivada, no solo el estado.
2. **El batch efectivo y el batch físico no son intercambiables** para las heurísticas de
   las librerías: `LengthGroupedSampler` usó el efectivo y eso cambió por completo su
   comportamiento.
3. **Archivar los intentos fallidos con nombres descriptivos** (`..._batch4_swap_v3`,
   `..._group_by_length_stalled_v6`) convirtió `artifacts/checkpoints/archive` en el
   registro más honesto del proyecto.
4. **Excluir es más honesto que truncar**, y registrar la huella del subconjunto
   entrenado hace la exclusión auditable.
5. **Una excepción bien acotada no debilita un contrato**: pinnear los dos lados de la
   diferencia y testear que el caso incorrecto falla es lo que la distingue de un parche.
