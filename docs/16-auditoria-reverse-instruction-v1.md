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
