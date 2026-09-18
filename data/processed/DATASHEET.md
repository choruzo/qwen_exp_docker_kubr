# Datasheet: Docker/Kubernetes instruction tuning

## Resumen

- Registros totales: 62221
- Fecha de generacion (UTC): 2026-08-18T08:42:12.549463Z
- Tamano JSONL total: 193760175 bytes
- Semilla de split: 3407
- Estrategia: stratified_group_split
- Fugas exactas y semanticas: verificadas en data/processed/leakage_report.json
- Registros CC BY/CC BY-SA sin atribucion: 0
- Estado: FINAL

## Splits

| Split | Registros | Bytes | SHA-256 |
|---|---:|---:|---|
| train | 55999 | 174710516 | 0881e496673e276849c617e7ec08627727d0e83618c54a2763045f2e012a36c3 |
| validation | 3111 | 9581692 | bb5c3c04504fd7d307e4f3305ffae9c8ab5765882e85964f9bcfccbb2b25fe4a |
| test | 3111 | 9467967 | 0eb2a9580bf5c1581ea542dc6068eeeb6e88443998c885aaa94ab988b3e17cc1 |

## Fuentes

| Fuente | Registros |
|---|---:|
| argocd_docs | 837 |
| cert_manager_docs | 773 |
| containerd_docs | 337 |
| coredns_docs | 345 |
| docker_docs | 4628 |
| docker_nl_commands | 118 |
| helm_docs | 281 |
| ingress_nginx_docs | 151 |
| istio_docs | 518 |
| k8s_instructions | 183 |
| kubernetes_docs | 4290 |
| kubernetes_stackoverflow_questions | 18952 |
| stack_yaml_k8s | 11763 |
| stackoverflow_docker | 19045 |

## Licencias

| Licencia | URL canonica | Registros |
|---|---|---:|
| Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | 15321 |
| BSD-2-Clause | https://opensource.org/license/bsd-2-clause | 43 |
| BSD-3-Clause | https://opensource.org/license/bsd-3-clause | 207 |
| CC-BY-4.0 | https://creativecommons.org/licenses/by/4.0/ | 23242 |
| CC-BY-SA-4.0 | https://creativecommons.org/licenses/by-sa/4.0/ | 19045 |
| CC0-1.0 | https://creativecommons.org/publicdomain/zero/1.0/ | 44 |
| ISC | https://opensource.org/license/isc-license-txt | 6 |
| MIT | https://opensource.org/license/mit | 4190 |
| Unlicense | https://unlicense.org/ | 116 |
| Zlib | https://opensource.org/license/zlib | 7 |

## Categorias

| Categoria | Registros |
|---|---:|
| arquitectura | 1684 |
| comando_cli | 1473 |
| concepto | 5371 |
| dockerfile | 1486 |
| generacion_yaml | 12177 |
| troubleshooting | 40030 |

## Transformaciones

Extraccion con revision de fuente, limpieza y redaccion de PII, normalizacion ChatML,
deduplicacion exacta y semantica, validacion kubeconform/hadolint, split estratificado
por categoria y fuente y auditoria exacta de similitud entre splits.

Los datasets sin licencia confirmada permanecen en data/quarantine y no se incluyen.
Las preguntas de Stack Overflow conservan URL, autor y atribucion CC BY-SA por registro.

## Limitaciones

- El contenido tecnico puede quedar obsoleto; la revision de fuente y fecha se conserva en meta.
- La similitud semantica no demuestra equivalencia tecnica perfecta.
- El set hard se construye con indicadores reproducibles de complejidad y debe revisarse junto al manifiesto.
