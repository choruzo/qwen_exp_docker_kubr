# Experimento controlado LR 1e-4 v3

Inicio: **25-09-2026 18:48 Europe/Madrid**. Fin: **27-09-2026 12:10
Europe/Madrid**. Estado: **terminado; no promocionado a test**.

## Hipótesis

El checkpoint anterior mejoró la similitud semántica, pero desarrolló bucles de
generación y falló el gate de truncamiento. Este experimento comprueba de forma
aislada si el learning rate `2e-4` fue demasiado agresivo.

## Contrato

Respecto a `training.rocm.best_epoch_v2.yaml` se mantienen modelo, hashes, splits,
semilla 3407, dos épocas, LoRA `r=32/alpha=32`, batch efectivo 16, scheduler,
optimizador, contexto y delimitadores. Sólo cambia el learning rate a `1e-4`.

También se eleva `save_total_limit` de 3 a 15 y se usan rutas nuevas bajo
`artifacts/rocm/lr1e4_v3/`. Esto no modifica el aprendizaje: permite conservar los
14 checkpoints de evaluación previstos para seleccionarlos después por estabilidad
de generación, además de por `eval_loss`.

El preflight verificó:

- 55 999 registros de entrenamiento y 3111 de validación.
- Los hashes de los dos shards y archivos del tokenizer/modelo local.
- La identidad de validación y su excepción de redacción registrada.
- El baseline final requerido.

El smoke test de un paso completó forward, backward, evaluación y checkpoint. Usó
64 929 792 parámetros entrenables (1,41 %) y alcanzó un pico de 10,68 GiB de VRAM.

## Operación

El contenedor es `qwen35-rocm-lr1e4-v3`. Dos timers de usuario restringen la
ejecución a 08:00–23:30 Europe/Madrid: el arranque/reanudación se comprueba cada
diez minutos hasta las 23:20 y la parada se ejecuta a las 23:30. La reanudación
exige un checkpoint completo con la huella exacta del contrato; si el primer
checkpoint no existe, el scheduler se niega a reiniciar desde cero.

No se exportará adaptador, modelo fusionado ni GGUF al terminar automáticamente.

## Selección posterior

1. Ordenar checkpoints por `eval_loss`, conservando la curva completa.
2. Ejecutar una sonda corta de validación sobre candidatos distribuidos a lo largo
   de la curva, sin utilizar test.
3. Ejecutar la sonda fijada de 60 casos sobre los mejores candidatos.
4. Exigir truncamiento <= 1 % y revisar similitud, repetición, reintentos y
   categorías; `eval_loss` será sólo una de las métricas.
5. Exportar únicamente el ganador y ejecutar una sola vez el test congelado.
6. Sólo si supera el gate, habilitar juez, sintaxis e inferencia GGUF real.

Si reducir sólo el LR no estabiliza la generación, el siguiente experimento podrá
reducir LoRA a `r=16/alpha=16` manteniendo `LR=1e-4`. Así no se confunden ambas
causas en una misma ejecución.

## Resultado del entrenamiento

La ejecución terminó en el paso 6994 (dos épocas) con código de salida 0. El mejor
checkpoint por pérdida fue también el final:

- `eval_loss = 1.25527822971344` en `checkpoint-6994`;
- evaluación posterior a entrenamiento: `1.2553318738937378`;
- huella del contrato/checkpoints:
  `6836bfadf6050bef8b3cbd77cde17382b32c5f6a8985f3d9317f4d43763c0030`;
- SHA-256 de `training_result.json`:
  `914c1adbb6fea04efc4483d1931334881a6ef5ba4f28a880e767a6aa5231857d`.

La pérdida descendió casi monótonamente, pero no predijo estabilidad generativa:

| Paso | Época aproximada | `eval_loss` |
|---:|---:|---:|
| 500 | 0.143 | 1.319453 |
| 1000 | 0.286 | 1.302579 |
| 2000 | 0.572 | 1.281277 |
| 3500 | 1.001 | 1.262027 |
| 6994 | 2.000 | **1.255278** |

## Exportación verificada

Se exportó el checkpoint final sin GGUF. Una segunda lectura independiente recalculó
todos los tamaños y hashes del manifiesto:

| Artefacto local | Archivos | Bytes | Verificación |
|---|---:|---:|---|
| adaptador | 7 | 149 947 691 | correcta |
| fusionado BF16 | 9 | 9 339 922 543 | correcta |

El manifiesto tiene SHA-256
`9bc7cca9dcaaff59423fe3d58e6916fd66f80709d24772264f9f867180f4fcef`,
conserva la revisión base `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636` y el hash de test
`0eb2a9580bf5c1581ea542dc6068eeeb6e88443998c885aaa94ab988b3e17cc1`.

## Selección por estabilidad sobre validación

La sonda usa 60 cuantiles de longitud, diez por categoría, orden y lotes
deterministas, y el contrato congelado cuyo SHA-256 es
`fb5ca861612637f4b370e1430e990ec06d3679e6fe04a51acd34654d8aa8e98a`.
No hay solapamiento con test ni con los casos fuera de dominio. Los candidatos que
ya no podían lograr 0/60 se detuvieron al primer lote fallido; por ello sus tasas
parciales no son estimaciones poblacionales.

| Variante | Casos observados | Truncados | Tasa observada | Decisión |
|---|---:|---:|---:|---|
| baseline | 60 | 1 | 1.67 % | control completo |
| checkpoint 500 | 8 | 2 | 25.0 % | descartado |
| checkpoint 1000 | 8 | 2 | 25.0 % | descartado |
| checkpoint 2000 | 4 | 1 | 25.0 % | descartado |
| checkpoint 3500 | 4 | 1 | 25.0 % | descartado |
| fusionado final | 16 | 2 | 12.5 % | descartado |

Con 60 casos, el gate `<= 1 %` exige cero truncamientos. El baseline tampoco lo
supera por un único caso final de Dockerfile, pero sí termina con EOS los dos casos
que fallan de forma sistemática desde el checkpoint 500. En esos dos casos el
baseline usa 161 y 401 tokens; los modelos ajustados agotan 2048 incluso después
del reintento con penalización 1.1.

Las salidas fallidas son falsas notas de versión de Docker. La proporción de
8-gramas repetidos fue 40--47 % en el checkpoint 500, 92--93 % en el 1000 y 58--80 %
en el modelo final, frente a 0--0.3 % en el baseline. Los encabezados genéricos de
notas de versión aparecen con frecuencia en train, pero las frases y enlaces
concretos generados no aparecen literalmente: el fallo combina patrón aprendido,
alucinación y bucle, no copia textual exacta.

## Calidad pareada antes del gate

En los 16 casos comunes entre baseline y fusionado final, la similitud semántica
media sube de `0.6839` a `0.7887` (`+0.1048`), con 10 victorias y 6 derrotas. El
modelo ajustado mejora en las cuatro categorías presentes en ese prefijo:

| Categoría | Baseline | Ajustado |
|---|---:|---:|
| arquitectura | 0.6502 | 0.8832 |
| comando CLI | 0.6525 | 0.7705 |
| concepto | 0.6956 | 0.7433 |
| generación YAML | 0.9008 | 1.0000 |

El resultado es por tanto mixto: especializa y acorta las respuestas que terminan,
pero introduce una regresión severa de terminación. `eval_loss` por sí sola habría
seleccionado precisamente el candidato inestable.

## Decisión y siguiente experimento

No se ejecuta test, sintaxis, juez ni GGUF para esta corrida: el gate de validación
falla y usar test para seguir seleccionando checkpoints contaminaría el protocolo.

El siguiente experimento debe cambiar una sola causa: mantener `LR=1e-4` y reducir
LoRA a `r=16/alpha=16`. Conviene hacerlo primero como piloto de 500 pasos, guardar
en pasos 250 y 500 y ejecutar esta misma sonda con parada al primer truncamiento.
Sólo si un candidato logra 0/60 se prolongará el entrenamiento y se consumirá test.

## Cierre operativo

Los timers `qwen35-rocm-lr1e4-v3-{start,stop}.timer` figuran como `not-found` e
`inactive`; no queda ningún contenedor v3 en ejecución. Los artefactos, checkpoints,
datos y resultados crudos permanecen locales y no deben añadirse a Git.
