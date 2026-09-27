# Diagnóstico del piloto LR 1e-4, LoRA r16/alpha16

Fecha de ejecución: 2026-09-27 (Europe/Madrid).

## Decisión

No se promociona ni `checkpoint-250` ni `checkpoint-500`. Los dos fallan el gate congelado de estabilidad en el mismo registro de validación: alcanzan 2.048 tokens sin EOS, incluido el único reintento permitido. No procede ejecutar test, juez ni GGUF con estos checkpoints.

El piloto descarta que reducir solamente LoRA de r32/alpha32 a r16/alpha16, manteniendo LR `1e-4`, resuelva la degeneración de generación. El control de repetición consigue emitir EOS en el caso investigado, pero no recupera la factualidad; debe considerarse una barrera de seguridad, no una corrección del modelo.

## Contrato y entrenamiento

- Configuración: `config/training.rocm.lr1e4_r16_pilot_v1.yaml` (`sha256:2bb3c32bc15f3c476992a72cadc36fa706e6796ac73ee1b0777ffcb6087da424`).
- Sonda: `config/validation_probe.rocm.lr1e4_r16_pilot_v1.yaml` (`sha256:a7040a33971360ae427a6a34bb2f01ec79448837dd4cd7957b4ff45b96383d3b`).
- Benchmark congelado: `config/benchmark.rocm.patched.yaml` (`sha256:fb5ca861612637f4b370e1430e990ec06d3679e6fe04a51acd34654d8aa8e98a`).
- Validación sin modificar: 3.111 registros, `sha256:bcf223584546fe573830470fcffedcffe92817034078808934bc892f6cbf90ac`; 3.109 quedaron dentro del límite de 8.192 tokens.
- Parámetros entrenables: 32.464.896 de 4.571.730.432 (0,71 %).
- Pasos: 500; batch efectivo: 16; LR: `1e-4`; scheduler: coseno.
- Duración informada por Trainer: 4.928,8 s (82,15 min).
- `train_loss`: 1,315788.
- Pico asignado/reservado: 19,4068/20,8828 GiB.
- Fingerprint común de procedencia: `bfaa4b3c79bec25bac7f933f50b7732b3cafa11184a1813d963eb18bf735753a`.

| Checkpoint | eval_loss | EOS en caso crítico | Resultado del gate |
| --- | ---: | --- | --- |
| 250 | 1,337146 | No, 2.048 tokens tras reintento | Falla |
| 500 | 1,329143 | No, 2.048 tokens tras reintento | Falla |

El checkpoint 500 mejora `eval_loss` en 0,008004 (0,60 %) respecto a 250, pero no mejora la estabilidad generativa. Esto confirma que `eval_loss` no puede usarse como único criterio de selección.

Hashes locales relevantes:

- Adaptador 250: `222f1d91a5c00520782bc06e4e0bcbaf313581e7d3d3efc24759b9ed3f3129ec`.
- Adaptador 500: `f10d019d21a355022b3d6d73f1503d85ab9f819e0c005adbb7f86771b2bd25c8`.
- Procedencia de ambos checkpoints: `6d4d7cc2d08a09132700a1fd4a28bada675d5cf9e68773e10622f2bab3d47c29`.
- Sonda congelada 250: `48904c8bb566c658556035e9c8570f578bf5d1b3616e5a76b7639b98eda66f53`.
- Sonda congelada 500: `8f4f5230a0c19ff5637040b3630c94cc056be0f2ffd98b6dbebfe1f29e034a71`.

Los adaptadores, estados del optimizador y resultados de inferencia permanecen locales y no se versionan.

## Caso que activa la parada temprana

Registro: `ff546f400d9edfb560be73a60386e2815316bdc2c0567098440255f55ac27cd7`, categoría `comando_cli`, fuente `argocd_docs`.

Pregunta: a qué versión se actualizó Helm en Argo CD v3.3 y qué cambio importante se eliminó. La referencia indica Helm **3.19.2** y la eliminación de `--client` en las llamadas a `helm version`.

| Variante | Tokens | EOS | Resultado factual |
| --- | ---: | --- | --- |
| Baseline congelado | 92 | Sí | Incorrecto: 3.12.0 y eliminación genérica de Helm 2 |
| r32, paso 500 | 70 | Sí | Incorrecto: 3.14.0 y `--debug` |
| r32, paso 1.000 | 101 | Sí tras reintento | Incorrecto: 3.14.2 y otro cambio |
| r32, paso 2.000 | 2.048 | No | Lista repetitiva de flags inventados |
| r32, paso 3.500 | 2.048 | No | Secuencia abierta de versiones inventadas |
| r16, paso 250 | 2.048 | No | Lista abierta de flags eliminados inventados |
| r16, paso 500 | 2.048 | No | Lista abierta de flags añadidos inventados |
| r16, paso 500, repetición 1,1/1,2 | 388 | Sí tras reintento | Incorrecto: Helm 4, `--debug` y novedades inventadas |
| r16, paso 500, guard de tres frases | 2.048 | No | Ignora el límite y continúa la lista |

El resultado con control de repetición está guardado localmente con SHA-256 `bbe802573cf9807627b19ec29044830aa30a63c42d9a34f5c2dcca4060e14504`. El guard de prompt está guardado localmente con SHA-256 `bba2298aa26156114634089fff6dce0c057763397129f29abe200a881290f7a2`.

## Diagnóstico

### Verificado

1. No existe solapamiento del caso por `content_hash`, `semantic_cluster_id` ni `source_record_id`: aparece una vez en validación y cero veces en train y test.
2. Las cadenas factuales `Helm 3.19.2`, `Argo CD v3.3` y la eliminación de `--client` no aparecen en las respuestas de train. No hay evidencia de memorización directa del caso.
3. El baseline tampoco conoce el dato exacto. Este registro es una pregunta factual cerrada sobre información retenida fuera de train; sin contexto o recuperación, acertarlo exigiría conocimiento previo del modelo.
4. El token EOS sí está correctamente supervisado. La plantilla termina las respuestas con `<|im_end|>`, resuelto como token `248046`, y `train_on_responses_only` conserva el final de la respuesta en las etiquetas.
5. El control de repetición evita el truncamiento aislado, pero conserva alucinaciones graves. Un prompt más estricto no evita el bucle.
6. En los 55.999 registros de train antes del filtro de longitud, 20.986 (37,48 %) usan normalización `reverse_instruction`. Ese subconjunto aporta 253 de las 263 respuestas con diez o más viñetas (96,20 %). Es una concentración de formato material, aunque por sí sola no prueba causalidad.

### Interpretación respaldada

El patrón es compatible con degeneración distributiva: el ajuste aprende con demasiada fuerza la forma de documentos/listas reconstruidos mediante `reverse_instruction` y, ante una pregunta sobre cambios de versión cuyo dato no conoce, entra en un modo de enumeración abierta. El LR `1e-4` sigue siendo agresivo para estabilidad generativa; reducir el rango de LoRA no reduce esa presión y en este caso adelanta el fallo respecto al piloto r32.

No es sobreajuste por memorización del registro retenido. Sí es una forma de sobreajuste al estilo y a patrones del corpus: baja la pérdida media mientras empeoran obediencia, finalización y precisión factual en un caso no visto.

## Próxima iteración recomendada

1. Auditar y depurar primero `reverse_instruction`: medir alineación pregunta/respuesta, relación de longitudes y número de viñetas; excluir respuestas que sean volcados amplios de documentación para preguntas estrechas.
2. Construir ejemplos concisos y autocontenidos para preguntas factuales, incluyendo respuestas que declaren incertidumbre cuando el dato no esté en el contexto.
3. Para evaluar hechos nuevos de documentación sin contaminar splits, convertir estos casos a evaluación con contexto recuperado o adjuntar el fragmento fuente. Mantener aparte un bloque closed-book para medir conocimiento previo, sin interpretarlo como aprendizaje del fine-tuning.
4. Solo después de sanear los datos, ensayar LR `5e-5` o inferior con probes de generación más frecuentes. Seleccionar checkpoints por `eval_loss` y estabilidad, nunca solo por pérdida.
5. Mantener `repetition_control_v1` como guardrail de inferencia, no como criterio para aprobar un modelo que sigue inventando contenido.
