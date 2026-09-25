# Diagnóstico del checkpoint 6994 y ensayos de estabilización

Fecha de cierre: **25-09-2026**. Este documento registra resultados agregados; los
modelos, datos y JSON con predicciones permanecen locales y no se versionan.

## Conclusión ejecutiva

El checkpoint 6994 mejora la similitud semántica global frente al modelo base,
pero **no se promueve**: termina sin EOS en 131 de 3161 casos (4,14 %), por encima
del gate congelado de 1 %. La regresión está presente dentro del dominio
(126/3111, 4,05 %), de modo que no puede justificarse únicamente por la pérdida
fuera de dominio aceptada al especializar el modelo.

Los ensayos en validación muestran que una penalización global de repetición
reduce mucho los reintentos, pero pierde calidad y aún deja 1/60 truncamientos.
Aplicarla sólo al reintento empeora el resultado. No se ha ejecutado otro barrido
sobre `test`: se preserva el split congelado y se evita optimizar contra él.

## Comparación congelada sobre test

Ambas variantes cubren los mismos 3161 casos y usan la configuración congelada
`config/benchmark.rocm.best_epoch_v2.yaml`.

| Variante | Similitud semántica | Truncamientos | Tasa | Reintentos |
|---|---:|---:|---:|---:|
| Base | 0,654252 | 18 | 0,569 % | 131 |
| Checkpoint 6994 fusionado | 0,679152 | 131 | 4,144 % | 902 |

La ganancia semántica es +0,024900, pero el incremento de truncamientos es de
113 casos. El export final anterior (paso 10491) obtuvo 0,681636 y 81/3161
truncamientos (2,56 %): el checkpoint elegido por menor `eval_loss` es menos
estable durante generación. Esto demuestra que `eval_loss` por sí sola no es un
criterio suficiente para seleccionar el checkpoint.

El resultado base local tiene SHA-256
`75fbc758970e01cae0b5bf0399d3a72d3046390e00e923a58faeb3133d329ba9`.
El resultado legible del checkpoint 6994 tiene SHA-256
`017432d62c696a03c09ec378e37795ad5283fec61bac943c8f7d1c36ee7a2305`.

## Sonda de validación sin tocar test

Se fijaron 60 ejemplos de `val.jsonl`: diez cuantiles deterministas de longitud de
referencia por cada una de las seis categorías. La sonda usa el mismo orden por
longitud, batching y generación que el benchmark. También verifica que ninguno de
los hashes seleccionados pertenezca al manifiesto de test.

| Política | Semántica | Truncamientos | Reintentos | Mediana tokens |
|---|---:|---:|---:|---:|
| Congelada: 1,0; retry 1,1 | 0,700015 | 3/60 (5,00 %) | 19/60 | 146,5 |
| Global: 1,1; retry 1,2 | 0,687148 | 1/60 (1,67 %) | 2/60 | 136,5 |
| Sólo retry 1,2 | 0,686705 | 7/60 (11,67 %) | 19/60 | 160,5 |

La política global pierde 0,012867 puntos semánticos. Gana 25 comparaciones
pareadas, pierde 31 y empata 4. La mayor regresión por categoría aparece en
`dockerfile` (-0,055050), seguida de `comando_cli` (-0,024656). La política
selectiva no es monótona: de los tres truncamientos originales sólo recupera uno,
mantiene dos y crea cinco nuevos.

Huellas de los resultados locales:

- Congelada: `c172683698e67f49d81f083fda09a95156f2a91a9a3e1640d96e0718f7b42cdb`.
- Penalización global: `5c4aa83b9c4c6b9df4b7149b98fa3d6a61a6657db6d88f8a77466ca7216f57e5`.
- Sólo reintento: `a12a191707b467a27d87953ad9deaaf654af67ab946a646d0c5354b3c2ff0d86`.

## Mecanismo de la regresión

Los casos truncados del checkpoint muestran bucles repetitivos claros. En el test
completo, la mediana de repetición de 8-gramas es 0,4033 entre los truncados y 0
entre los no truncados; 83 de 131 truncados superan una proporción de 0,25. En el
modelo base sólo 4 de 18 truncados superan ese umbral.

No es un efecto de prompts excesivamente largos: la mediana de longitud del prompt
es 148 en los casos truncados y 221 en el resto. Las fuentes con mayor tasa son
`docker_docs` (47/317, 14,8 %) y `cert_manager` (9/62, 14,5 %), pero el fenómeno
atraviesa varias fuentes y categorías.

La auditoría de las 55 999 respuestas de entrenamiento tampoco encuentra una masa
equivalente de bucles: hay 239 duplicados exactos excedentes en 228 grupos (máximo
5 copias), y 1337 respuestas (2,39 %) superan una repetición de 8-gramas de 0,25.
Por tanto, los datos pueden contribuir por distribución y ponderación, pero no hay
evidencia de que el modelo se limite a copiar una gran colección de respuestas
repetitivas.

## EOS y plantilla

Se reprodujo el formateo real con el tokenizer local. La respuesta termina en
`<|im_end|>\n`; `<|im_end|>` es el EOS con id 248046. Al aplicar la misma función
`train_on_responses_only`, los EOS de sistema y usuario quedan en `-100`, mientras
que el EOS final del asistente conserva la etiqueta 248046. La plantilla y el
enmascarado no explican la falta de terminación.

## Decisión y siguiente experimento

Se rechazan las dos políticas de decodificación probadas y se mantienen bloqueados
el juez LLM, el informe final y GGUF. Ejecutarlos ahora produciría un artefacto que
ya incumple un gate anterior.

El siguiente entrenamiento debe tratar la estabilidad como criterio de selección:

1. Mantener los splits y la sonda de validación fijados en este ensayo.
2. Reducir primero sólo el LR a `1e-4`, manteniendo LoRA `r=32/alpha=32`, y
   comparar contra el checkpoint actual con la misma sonda.
3. Seleccionar checkpoints con una métrica compuesta que exija truncamiento <= 1 %
   además de `eval_loss`; no elegir sólo por pérdida teacher-forced.
4. Revisar el peso de fuentes documentales con alta tasa de bucle si el problema
   persiste, sin modificar validación ni test.
5. Sólo si se supera el gate, ejecutar juez/sintaxis y después una cuantización GGUF
   con inferencia real.

Este experimento controlado quedó iniciado el 25-09-2026 y se documenta en
[14-experimento-lr1e4-v3.md](14-experimento-lr1e4-v3.md).

Todas las inferencias de diagnóstico se ejecutaron dentro de la ventana autorizada
08:00–23:30 Europe/Madrid. Los temporizadores de entrenamiento y benchmark quedan
desactivados.
