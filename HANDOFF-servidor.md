# Prompt de continuación — servidor ROCm (preparado el 27-09-2026)

> Pega todo lo que hay bajo la línea en una sesión nueva de Claude Code abierta en la raíz
> del repositorio del servidor ROCm (Linux, Radeon AI PRO R9700).

---

Continúas una sesión anterior sobre este repositorio (fine-tuning de Qwen3.5-4B para
Docker/Kubernetes). Lee primero `CLAUDE.md`, `docs/README.md`, `docs/15-diagnostico-piloto-lr1e4-r16-v1.md`
y `docs/16-auditoria-reverse-instruction-v1.md`. Responde en español.

## Estado al que llegamos

- Ninguna corrida ha superado el gate de estabilidad: 0 truncamientos en la sonda de 60 casos
  de validación, o ≤ 1 % en test. Todas quedan en bucle hasta 2048 tokens sin EOS:
  v1/v2 (LR 2e-4), v3 (LR 1e-4, r32) y el piloto r16 (LR 1e-4, 500 pasos).
  Juez, informe final y GGUF siguen bloqueados.
- El piloto r16 descartó que el LR o el rango de LoRA expliquen por sí solos los bucles.
  La hipótesis actual es que la causa está en los datos: `reverse_instruction` usa
  fragmentos crudos de documentación como respuesta. Destaca `docker_docs / troubleshooting`:
  41 % preguntas estrechas respondidas con listas, 30 % secciones de changelog y 51 % que
  terminan a mitad de una lista.
- Caso centinela de validación: `ff546f400d9edfb560be73a60386e2815316bdc2c0567098440255f55ac27cd7`
  (argocd_docs, comando_cli, pregunta por la versión de Helm en Argo CD v3.3). Los checkpoints
  r16 250 y 500 entran en bucle ahí. Los dos casos que fallan de forma sistemática en v3 son
  notas de versión falsas de Docker.

## Commits de la sesión anterior (hechos en Windows)

- `702bc04` Audit reverse-instruction format risks: `scripts/audit_reverse_instruction.py`,
  sus tests y `docs/16`.
- `967276b` Derive train split without reverse-instruction tier A:
  `scripts/build_filtered_train.py`, el contrato `data.train_derivation` en
  `src/docker_k8s_finetune/train/runner.py` (`_verify_train_derivation`), tests, y el
  manifiesto y la lista de hashes en `data/derived/`.
- `badcb8f` Keep derived split files byte-exact across platforms (`.gitattributes`).

Train derivado (nivel A excluido: `release_notes`, `narrow_question_list_answer` o `long_list`
en registros RI):

| Campo | Valor esperado |
|---|---|
| Padre `data/processed/train.jsonl` | SHA-256 `0881e496673e276849c617e7ec08627727d0e83618c54a2763045f2e012a36c3`, 55 999 registros |
| Derivado `data/derived/train.ri_tier_a_filtered.v1.jsonl` | SHA-256 `717a65890ccb5052ff9dc0fe8812d2293353f2ad00f91809a101a2773cade2d5`, 169 387 281 bytes, 54 227 registros |
| Hashes excluidos `data/derived/train.ri_tier_a_filtered.v1.excluded_hashes.txt` | SHA-256 `c075b3525d7428a32164e62ad3853ac9999c2c7b8c6002a9f25616ad47f810f6`, 1 772 registros |

El JSONL derivado está en `.gitignore`: hay que regenerarlo en el servidor.

## Tareas, en orden

1. **Sincronizar y verificar.** Ejecuta `git pull --ff-only` y confirma que `HEAD` contiene
   `badcb8f`. Si no está, detente y pregúntame: los commits se hicieron en Windows y hay
   que empujarlos. Ejecuta los tests: `python -m pytest tests/test_train_core.py
   tests/test_audit_reverse_instruction.py -q`.
2. **Regenerar el train derivado.** Ejecuta `python scripts/build_filtered_train.py`.
   El resultado debe dar exactamente los hashes de la tabla. Compara también
   `git diff --exit-code data/derived/`: el manifiesto regenerado debe coincidir con el
   versionado. Si algo difiere, no sigas y averigua por qué (fin de línea, versión del
   padre, cambio en el script de auditoría).
3. **Cruzar las flags con los truncamientos reales.** Es el contraste pendiente para pasar
   de correlación a evidencia. Ejecuta `python scripts/audit_reverse_instruction.py
   data/processed/test.jsonl --flagged-output artifacts/audits/reverse_instruction_flags_test.v1.jsonl`
   y cruza por `content_hash` con los registros truncados de
   `benchmarks/rocm/best_epoch_v2/finetuned_safetensors_results.json` (checkpoint 6994,
   131 truncados) y de `benchmarks/rocm/hipblaslt_patched/baseline_results.json` (18).
   Revisa primero la estructura del JSON; no supongas los nombres de campo. Reporta solo
   agregados:
   - tasa de truncamiento con y sin flag de nivel A;
   - tasa por fuente y categoría;
   - número de truncados en registros RI frente a `direct_pair`.
   Añade el resultado a `docs/16`.
4. **Proponer la config del piloto con datos filtrados.** Copia
   `config/training.rocm.lr1e4_r16_pilot_v1.yaml` a
   `config/training.rocm.lr1e4_r16_ri_filtered_pilot_v1.yaml` y cambia solo lo necesario:
   - `data.train` apunta al derivado;
   - añade el bloque `data.train_derivation` con `parent`, `excluded_hashes`,
     `excluded_sha256`, `excluded_records`, `sha256`, `bytes` y `count`;
   - rutas `output.*` nuevas bajo `artifacts/rocm/lr1e4_r16_ri_filtered_pilot_v1/`;
   - `smoke_test.output_dir` nuevo;
   - `smoke_test.input` puede seguir siendo el padre.

   Crea la sonda equivalente `config/validation_probe.rocm.lr1e4_r16_ri_filtered_pilot_v1.yaml`
   (checkpoints 250 y 500; `outputs_dir` nuevo). Así el único cambio frente al piloto r16 son
   los datos. Enséñame el diff antes de continuar.
5. **Preflight y smoke**, dentro del contenedor ROCm con los mismos compose que el piloto
   (`compose.train.rocm.yaml` + `compose.train.rocm.patched.yaml`):
   `train --config <nueva> --preflight` y después `--smoke-test --no-export`.
   El preflight debe mostrar 54 227 registros de train y aceptar la derivación.
6. **No lances el entrenamiento real ni actives timers de systemd sin mi confirmación
   explícita.** Si lo apruebo:
   - crea el scheduler y los timers copiando el patrón de
     `scripts/schedule_rocm_training_lr1e4_v3.sh` y `systemd/user/qwen35-rocm-lr1e4-v3-*`,
     con nombres nuevos;
   - respeta la ventana 08:00–23:30 Europe/Madrid;
   - usa `--no-export`.
   Después, ejecuta la sonda con `--stop-on-truncation`, pero **ejecuta primero los casos
   centinela** (el de Argo CD y los dos de notas de versión de v3). La sonda ordena los
   prompts de más corto a más largo, así que el prefijo que se evalúa está sesgado hacia los
   casos que más fallan.

## Reglas que no se negocian

- No modificar `data/processed/*`, `split_statistics.json`, `benchmarks/test_manifest.jsonl`
  ni ninguna config congelada (`config/benchmark.rocm.patched.yaml`, SHA-256
  `fb5ca861612637f4b370e1430e990ec06d3679e6fe04a51acd34654d8aa8e98a`).
- No usar test para seleccionar checkpoints. El test completo se ejecuta una sola vez, y solo
  si un candidato logra 0/60 en la sonda de validación.
- No aflojar ningún hash check sin fijar su valor esperado en la config.
- En `docs/` y en los commits solo van agregados; nunca texto de preguntas o respuestas del
  dataset. Modelos, checkpoints, resultados crudos y el JSONL derivado se quedan en local.
- No hacer push ni publicar nada sin preguntarme.
- Mensajes de commit en inglés, con el estilo del repo.

## Contexto útil de la sesión anterior

- `eval_loss` no predice la estabilidad: el mejor checkpoint por pérdida ha sido siempre el
  más inestable.
- La plantilla y el EOS están verificados: `<|im_end|>` (id 248046) se supervisa bien.
- `repetition_penalty` es un guardarraíl, no una solución: pierde calidad y el modelo sigue
  inventando.
- Al filtrar, `comando_cli` pierde el 13,7 % de sus registros y ya estaba infrarrepresentada
  (1,5 % de train frente a 10 % de test).
- Test y validación tienen la misma proporción de RI que train (≈ 47 %). La similitud
  semántica premia reproducir fragmentos crudos, así que un modelo más estable puede
  puntuar menos en esos casos. Tenlo en cuenta al interpretar los resultados.
