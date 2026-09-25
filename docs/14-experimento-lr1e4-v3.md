# Experimento controlado LR 1e-4 v3

Inicio: **25-09-2026 18:48 Europe/Madrid**. Estado inicial: **en ejecución**.

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
