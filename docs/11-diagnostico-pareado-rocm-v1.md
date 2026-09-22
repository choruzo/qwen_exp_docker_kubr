# Diagnóstico pareado ROCm v1 — 22-09-2026

Este análisis es **exploratorio** sobre las predicciones locales de `config/benchmark.rocm.patched.yaml` (SHA-256 `fb5ca861612637f4b370e1430e990ec06d3679e6fe04a51acd34654d8aa8e98a`). No altera el test, sus manifiestos ni los parámetros congelados; tampoco convierte el resultado ajustado en final.

## Evidencia comprobada

`scripts/diagnose_rocm_benchmark.py` valida la identidad y el orden de los 3.161 pares, el SHA-256 del test `0eb2a9580bf5c1581ea542dc6068eeeb6e88443998c885aaa94ab988b3e17cc1` y la igualdad de configuración de generación. Sólo emite agregados, sin preguntas ni respuestas.

| Medida | Baseline | Ajustado |
|---|---:|---:|
| Casos con reintento | 131/3.161 (4,14 %) | 480/3.161 (15,19 %) |
| Truncamientos tras reintento | 18/3.161 (0,57 %) | 81/3.161 (2,56 %) |
| Fallo entre reintentados | 18/131 (13,74 %) | 81/480 (16,88 %) |
| Mediana de tokens de respuesta | 567 | 135 |
| P95 de tokens de respuesta | 1.536 | 801 |

Hay 78 truncamientos nuevos del ajustado, 15 que desaparecen y sólo 3 compartidos. Los 81 casos ajustados truncados se reintentaron y volvieron a llegar a 2.048 tokens sin EOS. La combinación de mediana mucho menor y cola truncada indica una distribución bimodal: la regresión no se explica por una simple mayor longitud media.

Fuera de dominio (50 casos), la coincidencia exacta cae de 21 a 3, la similitud semántica de 0,847 a 0,570 y la mediana de respuesta sube de 4 a 26 tokens. No hay ejemplos `out_of_domain` en los 55.999 registros de entrenamiento declarados en `split_statistics.json`; esto es compatible con pérdida de generalidad por especialización, pero **no prueba** causalidad. La categoría `troubleshooting`, dominante en train (37.126/55.999), también baja levemente en similitud (0,581 a 0,569).

El archivo `tokenizer.json` base y exportado tiene el mismo SHA-256, y las propiedades `chat_template`, `eos_token` y `added_tokens_decoder` de `tokenizer_config.json` son iguales. Hay otros cambios de metadatos introducidos al exportar; no basta para declarar que el tokenizer sea la causa ni para descartarlo de modo absoluto.

## Experimento siguiente, separado del benchmark congelado

1. Tomar una muestra fija y versionada **sólo de validación** con respuestas largas y cortas, seleccionada por hashes; no usar el test ni las 50 pruebas fuera de dominio para elegir parámetros.
2. En la misma versión de ROCm, comparar baseline, adaptador antes de fusionar y modelo fusionado con la misma plantilla y dos políticas de generación predeclaradas: la congelada y una alternativa con control de repetición más fuerte. Guardar predicciones sólo en local.
3. Medir EOS, longitud, reintentos y validez de artefactos; inspeccionar manualmente una muestra ciega de bucles/listas para distinguir defecto del modelo de defecto del decodificador o la exportación.
4. Elegir una política **antes** de ejecutar un nuevo test completo. Ese resultado debe etiquetarse como experimento nuevo, nunca como aprobado bajo el contrato congelado. Si el modelo fusionado difiere del adaptador, investigar la exportación antes de valorar GGUF.

Mientras el gate congelado siga fallando, juez, comparación final y cuantización GGUF continúan bloqueados. El script de reproducción es:

```bash
python3 scripts/diagnose_rocm_benchmark.py \
  benchmarks/rocm/hipblaslt_patched/baseline_results.json \
  benchmarks/rocm/hipblaslt_patched/finetuned_safetensors_results.json
```
