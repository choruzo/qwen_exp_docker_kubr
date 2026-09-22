# Diagnóstico del benchmark ROCm congelado — 22-09-2026

Estado: **necesita revisión; no es un benchmark final**. Se usó `config/benchmark.rocm.patched.yaml` sin modificar el split de test ni los parámetros de generación. Los JSON de predicciones permanecen locales.

## Comparabilidad y gates

Las dos variantes cubren los mismos 3.161 registros (3.111 del test y 50 fuera de dominio), en idéntico orden, con el mismo SHA-256 del test `0eb2a9580bf5c1581ea542dc6068eeeb6e88443998c885aaa94ab988b3e17cc1`, igual configuración de generación y la misma firma ROCm (`gfx1201`, Torch `2.12.1+rocm7.2`, HIP `7.2.53211`). No hubo errores de generación.

| Gate o medida | Baseline | Ajustado |
|---|---:|---:|
| Truncamientos / 3.161 | 18 (0,57 %) | **81 (2,56 %)** |
| Límite congelado de truncamiento | ≤ 1 %: pasa | ≤ 1 %: **falla** |
| Similitud semántica media, provisional | 0,6543 | 0,6816 |
| Coincidencia exacta fuera de dominio | 21/50 | **3/50** |
| Coincidencia exacta en comandos CLI | 0/312 | 4/312 |
| Sintaxis YAML/Dockerfile | 717/922 válidos (77,8 %) | No ejecutable con el gate fallido |
| Juez LLM | Pendiente | Bloqueado por el gate fallido |

La mejora agregada de similitud (+0,0274) **no compensa** el exceso de truncamientos ni demuestra una mejora global de calidad. El pipeline prohíbe finalizar sintaxis y juez para `finetuned_safetensors` cuando `completion.generation=false`; el informe final y la prueba GGUF quedan pendientes.

## Por categoría

Los conteos se recalcularon directamente de los registros pareados; la similitud es la media provisional de cada categoría.

| Categoría | Casos | Trunc. base → ajustado | Similitud base → ajustado |
|---|---:|---:|---:|
| arquitectura | 156 | 0 → 6 | 0,67 → 0,77 |
| comando_cli | 312 | 0 → 16 | 0,65 → 0,74 |
| concepto | 269 | 2 → 17 | 0,66 → 0,77 |
| dockerfile | 312 | 1 → 1 | 0,57 → 0,58 |
| generacion_yaml | 610 | 5 → 17 | 0,85 → 0,92 |
| out_of_domain | 50 | 0 → 0 | **0,85 → 0,57** |
| troubleshooting | 1.452 | 10 → 24 | 0,58 → 0,57 |

En muestras truncadas, el modelo ajustado continúa listas o tablas hasta 2.048 tokens y termina sin EOS; por tanto, no parece una simple marca errónea de truncamiento. En fuera de dominio aparecen respuestas más verbosas, cambios de idioma y material ajeno a la pregunta. Son observaciones de muestra, no una causa demostrada; requieren diagnóstico del entrenamiento o de una configuración de generación **nueva y versionada**. El benchmark congelado no debe editarse para declarar aprobado este resultado.

## Estado operativo

La caché del scorer semántico se reparó y verificó antes de cerrar ambos JSON. La validación sintáctica baseline terminó; la del ajustado y el juez no se ejecutaron porque el gate de generación falló. No se inició GGUF. Los timers de inferencia y entrenamiento están desactivados. Los modelos, datos, cachés y predicciones individuales siguen fuera de Git.
