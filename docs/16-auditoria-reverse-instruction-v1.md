# Auditoría de `reverse_instruction` v1

Fecha: **27-09-2026**. Diagnóstico sobre los splits congelados; no modifica datos,
splits, benchmark ni configuraciones. Sólo se versionan agregados; la lista de hashes
marcados queda local en `artifacts/audits/` (ignorado por Git).

## Motivo

[15-diagnostico-piloto-lr1e4-r16-v1.md](15-diagnostico-piloto-lr1e4-r16-v1.md) descartó
que LR o el rango de LoRA expliquen por sí solos los bucles de enumeración, y señaló
que `reverse_instruction` concentra el 96 % de las respuestas con diez o más viñetas.
Esta auditoría mide qué forma de respuesta enseña ese subconjunto.

## Método

`scripts/audit_reverse_instruction.py` calcula, por normalización, fuente y categoría:

| Señal | Definición |
|---|---|
| `release_notes` | Sección de changelog (encabezado tipo `Bug fixes`, `For all platforms`, versión…) con ≥ 3 viñetas, o ≥ 3 viñetas con referencias a issues |
| `narrow_question_list_answer` | Pregunta sin marcador de enumeración (`enumera`, `lista`, `todos`, `qué pasos`…) y respuesta con ≥ 5 viñetas |
| `long_list` | ≥ 10 viñetas markdown (no se cuentan secuencias YAML) |
| `template_residue` | Shortcodes Hugo (`{{< >}}`, `{{% %}}`) o enlaces relativos `/docs/…` |
| `ends_in_structure` | La respuesta termina dentro de una lista, tabla, encabezado, separador o plantilla |
| `language_mismatch` | Pregunta en español y respuesta en inglés (por stopwords; no se aplica a YAML) |
| `context_reference` | La pregunta cita "la información proporcionada" u otro contexto que no está en el prompt |
| `length_imbalance` | Palabras de respuesta / pregunta ≥ 15 |

El solapamiento léxico pregunta/respuesta se reporta pero **no** se usa como flag: en
`direct_pair` la mediana es 0,14, así que no discrimina pares mal alineados.

```bash
python scripts/audit_reverse_instruction.py data/processed/train.jsonl \
  --flagged-output artifacts/audits/reverse_instruction_flags_train.v1.jsonl
```

## Resultados en train

| Señal | `direct_pair` (35 013) | `reverse_instruction` (20 986) |
|---|---:|---:|
| Empieza con encabezado markdown | 0,0 % | 51,7 % |
| `language_mismatch` | 0,0 % | 44,7 % |
| `ends_in_structure` | 0,6 % | 13,8 % |
| `template_residue` | 1,3 % | 13,1 % |
| `narrow_question_list_answer` | 3,4 % | 6,8 % |
| `release_notes` | 0,0 % | 2,9 % |
| Dos o más señales | 1,1 % | 23,1 % |

Las cifras de `reverse_instruction` incluyen 10 587 registros de `stack_yaml_k8s`, que
salen limpios. Los 10 399 registros restantes proceden de documentación y son,
en la práctica, **fragmentos crudos de la web de documentación usados como respuesta**:
entre el 83 % y el 99 % están en inglés para preguntas en español, y entre el 76 % y el
100 % empiezan con un encabezado de sección.

Grupos con más riesgo de enseñar enumeración abierta:

| Fuente / categoría | n | Pregunta estrecha + lista | Changelog | Termina en estructura | Plantilla |
|---|---:|---:|---:|---:|---:|
| docker_docs / troubleshooting | 1 053 | 40,7 % | 29,6 % | 50,7 % | 6,6 % |
| docker_docs / comando_cli | 249 | 24,5 % | 13,7 % | 32,5 % | 7,6 % |
| argocd_docs / troubleshooting | 282 | 23,4 % | 0,4 % | 19,9 % | 16,3 % |
| istio_docs / troubleshooting | 198 | 22,7 % | 0,0 % | 53,5 % | 88,9 % |
| cert_manager_docs / troubleshooting | 231 | 21,6 % | 12,6 % | 30,3 % | 0,4 % |
| docker_docs / arquitectura | 263 | 19,8 % | 13,7 % | 36,1 % | 4,6 % |

Ejemplo típico, descrito sin texto literal: una pregunta sobre un único cambio en una
herramienta concreta tiene como respuesta la sección entera `For all platforms` de unas
notas de versión de Docker Desktop, con decenas de correcciones sin relación entre sí.
Es el mismo modo de fallo que muestran los bucles de v2, v3 y r16: falsas notas de
versión y listas abiertas de flags.

## Candidatos a filtrar

| Nivel | Criterio | Train | Val | Test |
|---|---|---:|---:|---:|
| A (forma de enumeración) | `release_notes` o `narrow_question_list_answer` o `long_list` en RI | 1 772 | 122 | 137 |
| A, sólo `docker_docs` | idem | 1 098 | 73 | 80 |
| A o B (+ plantilla, corte, contexto ausente, desproporción) | RI | 5 303 | 441 | 436 |

## Limitaciones

- Heurísticas basadas en regex; no miden la alineación semántica. El siguiente paso
  es puntuar pregunta/respuesta con embeddings dentro de la imagen Docker validada.
- El caso centinela de Argo CD (`ff546f40…`) sólo activa `language_mismatch`: su
  fallo no se debe a su propio formato, sino al modo changelog aprendido de otros
  registros. Esto es compatible con la hipótesis, pero no la demuestra.
- Los resultados de benchmark ROCm no están en esta máquina, así que no se han
  cruzado los 131 truncamientos del checkpoint 6994 con estas señales. Es el siguiente
  contraste necesario para convertir la correlación en evidencia.
- **Test y validación tienen la misma composición** (≈ 47 % RI, mismas tasas). Filtrar
  sólo train cambia la distribución respecto al test congelado, y la similitud semántica
  premia reproducir fragmentos crudos. Una mejora en estabilidad podría verse como
  pérdida de similitud en esos casos; hay que decidirlo antes de reentrenar.

## Train derivado sin nivel A

`scripts/build_filtered_train.py` genera `data/derived/train.ri_tier_a_filtered.v1.jsonl`.
Primero verifica el `train.jsonl` congelado contra `split_statistics.json`, recalcula las
flags de nivel A con los umbrales fijados y copia byte a byte las líneas conservadas.
El padre no se modifica: SHA-256 `0881e496673e276849c617e7ec08627727d0e83618c54a2763045f2e012a36c3` antes y después.

| Identidad | Valor |
|---|---|
| Registros | 54 227 (55 999 − 1 772; 1 772 hashes únicos) |
| SHA-256 derivado | `717a65890ccb5052ff9dc0fe8812d2293353f2ad00f91809a101a2773cade2d5` |
| Bytes | 169 387 281 |
| Lista de exclusión | `data/derived/train.ri_tier_a_filtered.v1.excluded_hashes.txt`, SHA-256 `c075b3525d7428a32164e62ad3853ac9999c2c7b8c6002a9f25616ad47f810f6` |

Por fuente se excluyen `docker_docs` 1 098, `kubernetes_docs` 284, `cert_manager_docs` 108,
`argocd_docs` 104 y 178 del resto. Por categoría: troubleshooting 918, concepto 560,
arquitectura 167, comando_cli 116 y dockerfile 11. **`comando_cli` pierde el 13,7 % de sus
849 registros** y ya estaba infrarrepresentada (1,5 % de train frente a 10 % de test).

El JSONL derivado se ignora en Git porque se reproduce desde el padre. Sí se versionan el
manifiesto y la lista de hashes excluidos.

### Contrato en el runner

Una configuración puede usarlo con `data.train_derivation`, en la que fija `parent`,
`excluded_hashes`, `excluded_sha256`, `excluded_records`, `sha256`, `bytes` y `count`. Antes
de entrenar, `_verify_train_derivation` exige cuatro cosas: que el padre coincida con
`split_statistics.json`, que la lista de exclusión coincida con su hash, que cada hash
excluido exista en el padre y que el derivado sea exactamente el padre sin esas líneas.
El hash derivado pasa al contrato de checkpoints y al manifiesto de exportación junto con
`parent_sha256`. Validación y test no cambian.
