# Documentación del proyecto

Pipeline reproducible para convertir Qwen3.5-4B en un **experto de dominio estrecho**
en Docker y Kubernetes: extracción y limpieza de datos, fine-tuning con QLoRA/LoRA y
comparación base vs. fusionado vs. GGUF sobre un test congelado.

Esta carpeta documenta el proyecto tal y como está en el repositorio a fecha
**2026-09-18** (commit `b2185ea`). El `Readme.md` de la raíz sigue siendo la guía
operativa de comandos; estos documentos explican el **por qué**, los contratos
internos y el estado real de avance.

## Índice

| Documento | Contenido |
|---|---|
| [01-vision-general.md](01-vision-general.md) | Objetivo, arquitectura de módulos, restricciones de hardware, entorno |
| [02-pipeline-datos.md](02-pipeline-datos.md) | extract → clean → normalize → dedupe → validate → split, fuentes y licencias |
| [03-modelo-y-entrenamiento.md](03-modelo-y-entrenamiento.md) | Identidad del modelo, LoRA, hiperparámetros, OOM fallback, checkpoints |
| [04-benchmark-y-evaluacion.md](04-benchmark-y-evaluacion.md) | Variantes, backends, juez LLM, provisional vs. final, informe |
| [05-perfil-rocm.md](05-perfil-rocm.md) | Ruta independiente Radeon AI PRO R9700 y excepción de `val.jsonl` |
| [06-estado-actual.md](06-estado-actual.md) | **Qué está hecho y qué falta**, con evidencia |
| [07-invariantes-y-gates.md](07-invariantes-y-gates.md) | Todos los gates de corrección y qué rompe cada uno |
| [08-decisiones-e-historial.md](08-decisiones-e-historial.md) | Decisiones de diseño y su motivo empírico, historial de commits |
| [09-evaluacion-rocm-patched.md](09-evaluacion-rocm-patched.md) | Estado de la exportación y evaluación ROCm parcheada (21-09-2026) |

## Regla central del proyecto

> Un resultado **no es final** hasta que el JSON de benchmark cubra el test congelado
> completo, la validación sintáctica y el juicio LLM, y el informe comparativo se haya
> generado. `--allow-provisional` existe solo para diagnóstico explícito.

## Atajo: estado en una línea

El entrenamiento ROCm parcheado terminó y sus exports se verificaron; el benchmark
ROCm sobre test sigue en curso y aún no hay comparación final ni juicio LLM.
Estado reciente en [09-evaluacion-rocm-patched.md](09-evaluacion-rocm-patched.md);
el [06-estado-actual.md](06-estado-actual.md) conserva el estado histórico anterior.
