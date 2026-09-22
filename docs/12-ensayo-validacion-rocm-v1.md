# Ensayo diagnóstico ROCm v1 — 22-09-2026

Estado: **diagnóstico, no aprobación del modelo**. `config/benchmark.rocm.patched.yaml` y los splits permanecieron intactos. Predicciones y modelos siguen sólo en local; este informe contiene agregados.

## Contrato y alcance

`config/validation_probe.rocm.v1.yaml` fija el SHA-256 del benchmark congelado (`fb5ca861612637f4b370e1430e990ec06d3679e6fe04a51acd34654d8aa8e98a`) y de validación (`bcf223584546fe573830470fcffedcffe92817034078808934bc892f6cbf90ac`). Se seleccionaron, antes de generar respuestas, la referencia mediana y la del percentil 90 de cada una de las seis categorías: 12 registros únicos sin solapamiento con test ni con los 50 casos fuera de dominio. El mismo generador ROCm se ejecutó **individualmente**, no en lotes de cuatro como el benchmark completo; la muestra no estima tasas poblacionales.

La comparación de IDs de los 12 prompts completos entre base, adaptador y fusionado dio 12/12 coincidencias. El fusionado sigue emitiendo una advertencia de regex del tokenizer; esa advertencia no produjo una diferencia en los IDs comprobados, pero no se ha auditado todo el espacio de entradas.

## Adaptador frente a fusionado, política congelada

| Medida en los mismos 12 registros de validación | Adaptador | Fusionado |
|---|---:|---:|
| Cierres con EOS / truncamientos | 12 / 0 | 12 / 0 |
| Mediana de tokens de respuesta | 202 | 201 |
| Similitud semántica media con la referencia | 0,7519 | 0,7535 |

Dos respuestas son textualmente idénticas; las otras diez difieren, a veces bastante, pero no hay una caída agregada de similitud ni un fallo de EOS exclusivo del fusionado en esta muestra. Una revisión manual acotada encontró respuestas discutibles en ambas variantes; la similitud semántica no certifica corrección factual.

Para estudiar el mecanismo de la cola larga se eligió **un caso ya truncado** de cada categoría `comando_cli`, `concepto` y `troubleshooting` del test congelado. Es una selección condicionada por el fallo, válida sólo para diagnóstico; no se usó para escoger la política alternativa ni para estimar calidad. Los prompts tienen 125–138 tokens.

| Caso del test, inferencia individual | Adaptador | Fusionado | Fusionado en benchmark batched |
|---|---|---|---|
| comando_cli | 2.048 tokens, sin EOS | 2.048, sin EOS | 2.048, sin EOS |
| concepto | 288, con EOS | 196, con EOS | 2.048, sin EOS |
| troubleshooting | 2.048, sin EOS | 2.048, sin EOS | 2.048, sin EOS |

Por tanto, **la fusión no es causa exclusiva** de los bucles de CLI y troubleshooting: aparecen también antes de fusionar, con la misma política y el mismo modo individual. El cambio del caso `concepto` entre inferencia individual y batched exige investigar sensibilidad numérica/al lote; no permite atribuir por sí solo la diferencia a un bug concreto.

## Política alternativa, sólo en validación

El fusionado se probó en los mismos 12 casos con `repetition_penalty=1,1` y reintento a `1,2`; el resto de la generación se heredó del benchmark congelado. Frente a la política original:

| Medida | Original | Alternativa |
|---|---:|---:|
| Truncamientos | 0/12 | 0/12 |
| Mediana de tokens | 201 | 172 |
| Similitud semántica media | 0,7535 | 0,7348 |
| Casos con mayor similitud bajo la alternativa | — | 5/12 |

Esta muestra no contiene truncamientos bajo la política original, por lo que **no demuestra** que la alternativa los resuelva. Además, la similitud baja 0,0187; no hay base para promoverla al test. No se modificó el JSON congelado ni se lanzó un nuevo benchmark completo.

## Revisión del entrenamiento y decisión

La pérdida de validación fue 1,2606 en el paso 7.000 (época 2,002), saltó a 1,3011 en el paso 7.500 y acabó en 1,3012. El entrenamiento no activó `load_best_model_at_end` ni parada temprana; exportó el estado de la época 3 y sólo conserva los checkpoints 10.450, 10.475 y 10.491. El checkpoint del mínimo ya no se puede comparar.

El muestreo fue aleatorio sobre un train desequilibrado: `troubleshooting` representa 66,30 % de train frente a 46,67 % de test; `comando_cli` 1,52 % frente a 10,03 % y `dockerfile` 1,54 % frente a 10,03 %. No hay categoría fuera de dominio en train. Son factores plausibles de especialización, **no causas demostradas** de cada fallo. La regresión fuera de dominio del benchmark congelado (21/50 a 3/50 coincidencias exactas) sigue sin resolverse.

La siguiente intervención justificable es **otro entrenamiento versionado**, primero aislando la duración/selección del checkpoint: máximo dos épocas o restauración y preservación explícita del mejor estado de validación, con idénticos datos y splits. Debe superar una evaluación nueva en validación y pruebas independientes fuera de dominio antes de un test completo nuevo. Si eso no recupera generalidad, estudiar un muestreo equilibrado y datos de preservación generalista como experimento separado. No se ha iniciado dicho entrenamiento.

Mientras el modelo exportado actual falle el gate congelado de truncamiento (81/3.161, 2,56 % > 1 %) y mantenga la regresión fuera de dominio, juez, informe final y GGUF siguen bloqueados. Las salidas locales se reproducen con `scripts/probe_validation_rocm.py` y `scripts/probe_failed_test_rocm.py`.
