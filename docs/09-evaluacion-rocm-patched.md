# Evaluación ROCm parcheada: estado del 21-09-2026

La evaluación congelada usa `config/benchmark.rocm.patched.yaml`. El split de test y los manifiestos de benchmark no se han modificado. Este documento es un estado **parcial**, no una comparación final de calidad.

## Comprobaciones cerradas

- Entrenamiento completado: 3 épocas; pérdida final de entrenamiento 0,37787; pérdida final de validación 1,30121; mejor pérdida de validación 1,26056 en el paso 7000.
- Hash SHA-256 del test usado por el entrenamiento: `0eb2a9580bf5c1581ea542dc6068eeeb6e88443998c885aaa94ab988b3e17cc1`.
- Manifiesto de exportación verificado recalculando los SHA-256 de los 19 archivos: adaptador (7 archivos, 149.947.691 bytes), merged (9, 9.339.922.030 bytes) y GGUF (3, 8.069.595.744 bytes). Todos coinciden.
- Juez congelado `gemma-4-12b-it-UD-Q6_K_XL.gguf` recuperado de la revisión `63238a33fe662634aea31a417f8c9e9385028726`: 10.685.011.360 bytes y SHA-256 `eb0f252863d14f7782122a4ac7e8744ed6e4a9fc132584d94686a9441f7c5d35`, idénticos a la configuración.
- Los temporizadores `qwen35-rocm-start.timer` y `qwen35-rocm-stop.timer` están desactivados y no aparecen en `systemctl --user list-timers --all`.

## Evaluación en curso

El primer intento de generar las respuestas del baseline terminó antes de completar el test. El 21-09-2026 se reanudó en un contenedor independiente de la sesión, `qwen35-rocm-benchmark-baseline`, reutilizando su caché determinista. Se ejecuta con `--skip-judge --skip-syntax` para separar la inferencia principal de las fases posteriores. Su progreso puede consultarse con:

```sh
docker logs --tail 20 qwen35-rocm-benchmark-baseline
docker ps --filter name=qwen35-rocm-benchmark-baseline
```

El juez todavía no ha producido métricas: necesita que existan las respuestas del baseline. Tampoco hay aún una comparación válida baseline/ajustado sobre el test ni análisis de regresiones, categorías o calidad; por tanto, no se puede decidir todavía si la cuantización GGUF supera los gates. Un benchmark CUDA anterior no sustituye al baseline ROCm congelado.

## Secuencia pendiente

1. Terminar baseline ROCm y generar el resultado local.
2. Ejecutar la variante `finetuned_safetensors` con la misma configuración y el mismo test.
3. Ejecutar validación sintáctica y juez con la identidad congelada para ambas variantes.
4. Generar el informe comparativo y analizar resultados por categoría y regresiones.
5. Solo si los gates de calidad se cumplen, ejecutar inferencia real de `finetuned_gguf`, validar sintaxis y pasar el mismo juez.

## Horario automático de inferencia

Los timers `qwen35-rocm-benchmark-start.timer` y `qwen35-rocm-benchmark-stop.timer` limitan la generación de `baseline` y `finetuned_safetensors` a 08:00–23:30, hora de Madrid. El inicio comprueba cada 10 minutos si debe reanudar una variante desde su caché; la parada diaria detiene el contenedor activo a las 23:30. Un resultado incompleto o una salida inesperada requieren diagnóstico manual y no se reintentan indefinidamente. El juez y GGUF no se arrancan automáticamente.

```sh
systemctl --user list-timers --all | rg 'qwen35-rocm-benchmark'
bash scripts/schedule_rocm_benchmark.sh status
```
