# Qwen3.5-4B experto en Docker y Kubernetes

Pipeline reproducible para extraer, limpiar, validar y estratificar datos de
Docker/Kubernetes; entrenar Qwen3.5-4B con QLoRA en una RTX 5060 Ti de 16 GB; y
comparar el modelo base, el modelo fusionado y sus exportaciones GGUF.

Los resultados y modelos no se consideran terminados hasta que los JSON de
benchmark no sean completos (sin la marca provisional) y el informe comparativo
haya sido generado sobre el test congelado.

## Requisitos

- Python 3.10-3.12 para ETL y pruebas.
- Docker Desktop con backend WSL2 y acceso a la GPU NVIDIA.
- Git LFS para versionar `data/processed/train.jsonl` (aprox. 167 MB).
- Pesos originales en Modelo/Qwen3.5-4B.
- Espacio para checkpoints, modelo fusionado y GGUF en el directorio artifacts.
- Al menos 50 GB para el disco de Docker si se usa la imagen oficial de
  Unsloth. La imagen derivada validada ocupa aproximadamente 42 GB por si sola,
  por lo que un limite de 20 GB no es compatible con esta ruta.

Se usa la imagen oficial de Unsloth fijada por digest como base de
Dockerfile.train. Es la ruta elegida para Blackwell porque incluye el stack
CUDA/PyTorch preparado y evita recompilar xformers para sm_120 en Windows. La
imagen base trae Transformers 4.57.6, que no reconoce Qwen3.5; la capa derivada
instala la version 5.2.0 exigida por la receta oficial de Unsloth, junto con
flash-linear-attention 0.5.2, torchao 0.16.0 (compatible con Torch 2.10),
sentence-transformers 5.2.0 y FAISS 1.15.0. Tambien anula el entrypoint que
haria chmod recursivo sobre todo el modelo y los datasets. La cache persistente
se guarda en .cache/huggingface mediante bind mount y queda fuera de Git.

## Instalacion y comprobaciones

    git lfs install
    python -m venv venv
    venv\Scripts\python -m pip install -e ".[extract,dedupe,test]"
    venv\Scripts\python -m docker_k8s_finetune.cli config-validate
    venv\Scripts\python -m pytest
    docker compose -f compose.train.yaml config --quiet
    docker compose -f compose.train.yaml build train
    venv\Scripts\python -m docker_k8s_finetune.cli pipeline-status

`pipeline-status` verifica hashes y linaje de todas las etapas. Devuelve un codigo
distinto de cero y enumera los bloqueos mientras falte una fuente, la revision
manual de reverse-instruction o un artefacto intermedio. El split final y el
entrenamiento completo rechazan esas entradas incompletas; la excepcion
`--allow-provisional` queda reservada a diagnosticos explicitos.

## Pipeline de datos

Las fuentes, licencias y cuarentenas estan declaradas en config/sources.yaml.
La extraccion masiva solo se habilita cuando ese archivo esta aprobado.

    venv\Scripts\python -m docker_k8s_finetune.cli extract
    venv\Scripts\python -m docker_k8s_finetune.cli clean
    venv\Scripts\python -m docker_k8s_finetune.cli normalize
    venv\Scripts\python -m docker_k8s_finetune.cli dedupe --mode exact
    docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli dedupe --mode approximate

La deduplicacion semantica se realiza en la imagen Docker validada con Sentence
Transformers y FAISS fijados.

La normalizacion inversa requiere un endpoint OpenAI-compatible:

    .\scripts\start-synthetic-qwen3-14b.ps1

El lanzador habilita dos slots locales con 16K de contexto total (8192 por
slot) y la normalizacion usa dos workers
ordenados. El cache por candidato hace que una interrupcion pueda reanudarse
sin regenerar respuestas ya validadas.
    $env:SYNTHETIC_LLM_BASE_URL="http://127.0.0.1:8080/v1"
    $env:SYNTHETIC_LLM_MODEL="qwen3-14b"

El perfil local usa `G:\models\Qwen3-14B-Q5_K_M.gguf`, contexto total 16K y
llama-server sobre la GPU. No requiere `SYNTHETIC_LLM_API_KEY`. Para otro
proveedor OpenAI-compatible se pueden sustituir URL, modelo y clave.

Primero se genera y revisa una muestra:

    venv\Scripts\python -m docker_k8s_finetune.cli normalize-generate --scope sample

Despues de revisar el JSONL, se copia su SHA-256 a
config/reverse_instruction_review.yaml y se cambia status a approved. Solo
entonces se permite:

    venv\Scripts\python -m docker_k8s_finetune.cli normalize-generate --scope full
    venv\Scripts\python -m docker_k8s_finetune.cli dedupe --mode exact
    venv\Scripts\python -m docker_k8s_finetune.cli dedupe --mode approximate
    venv\Scripts\python -m docker_k8s_finetune.cli validate-dataset
    venv\Scripts\python -m docker_k8s_finetune.cli split

Para ejecuciones locales largas, `scripts/continue-after-reverse-generation.ps1`
puede esperar al PID del generador y, solo si termina con codigo cero, liberar
Qwen y encadenar deduplicacion exacta/semantica, validacion y split. Cada paso
se detiene ante el primer error y `pipeline-status` valida el linaje final.

La deduplicacion aproximada usa un modelo de embeddings fijado por commit. El
split vuelve a calcular similitudes exactas entre train/validation/test, mueve
fugas completas a train y hace backfill del mismo estrato. Tambien genera:

- data/processed/train.jsonl, val.jsonl y test.jsonl;
- data/processed/DATASHEET.md;
- benchmarks/test_manifest.jsonl y llm_judge_manifest.jsonl;
- informes de sintaxis, deduplicacion y fugas.

La ficha registra fecha UTC, numero de registros y bytes/SHA-256 de cada split,
ademas de distribuciones por fuente, licencia y categoria.

Fuentes que requieren credenciales:

- GitHub issues: `GITHUB_TOKEN`; puede obtenerse temporalmente con
  `$env:GITHUB_TOKEN = gh auth token` despues de `gh auth login`.
- Stack Overflow Docker via BigQuery: `GOOGLE_CLOUD_PROJECT` y credenciales ADC
  creadas con `gcloud auth application-default login`. La consulta tiene un
  limite duro de 70 GiB; el dry-run validado proceso 63,55 GiB.

La generacion reverse-instruction requiere ademas SYNTHETIC_LLM_BASE_URL y
SYNTHETIC_LLM_MODEL, seguida de la aprobacion manual descrita arriba.

Las fuentes sin licencia confirmada permanecen en data/quarantine.

## Entrenamiento

Qwen3.5-4B es Qwen3_5ForConditionalGeneration, no un LLM textual clasico. El
runner usa FastVisionModel, congela vision y entrena solo lenguaje, atencion y
MLP. La seleccion de modulos LoRA queda a Unsloth para cubrir tanto la atencion
completa como la atencion lineal hibrida.

La revision del modelo base y los SHA-256 de ambos shards, indice, configuracion,
tokenizer, plantilla de chat y processor estan fijados en
`config/training.yaml`. El runner vuelve a calcularlos antes de cargar el modelo
y rechaza cualquier diferencia.

    venv\Scripts\python -m docker_k8s_finetune.cli train --smoke-test --preflight --no-export
    docker compose -f compose.train.yaml run --build --rm train python -m docker_k8s_finetune.cli train --smoke-test --no-export
    docker compose -f compose.train.yaml run --build --rm train python -m docker_k8s_finetune.cli train

El perfil principal usa contexto 8192, batch por dispositivo 1 y acumulacion 16.
El batch efectivo sigue siendo 16. La prueba real con batch 4 alcanzo 15,99 GiB
de VRAM, dejo solo 59-64 MiB libres y llevo Docker/WSL (13,9 GiB de RAM) a
usar swap, por lo que no era un perfil reproducible ni dejaba margen seguro.
Batch 2 x acumulacion 8 mejoro los pasos cortos, pero un lote largo alcanzo
15,88 GiB, dejo 173 MiB libres y volvio a bloquear el avance. Batch 1 x
acumulacion 16 es el perfil seguro: usa unos 9,4-11,2 GiB en los lotes normales
y llego puntualmente a 15,72 GiB en un lote cercano a 8192 tokens, pero siguio
avanzando y mantuvo el swap alrededor de 115 MiB durante la prueba real.
El sampler usa orden aleatorio determinista. Con batch fisico 1 no existe padding
entre ejemplos que ahorrar agrupandolos por longitud. Transformers 5.2 construye
`LengthGroupedSampler` con el batch efectivo (16), ordena megabloques de 800
ejemplos y concentra 16 secuencias largas en un mismo update. En la prueba real
esto mantuvo los primeros 350 updates en unos 14 segundos, pero el siguiente
megabloque largo elevo cada update a horas sin producir OOM. El orden aleatorio
distribuye esas secuencias entre ventanas de acumulacion y evita ese bloqueo.
Si CUDA devuelve OOM, reduce primero el contexto a 4096 y despues a 2048,
manteniendo batch 1 x acumulacion 16. El resultado deja constancia del
perfil que termino el entrenamiento y de todos los perfiles que fallaron.
Los checkpoints se guardan cada 25 updates (conservando los tres mas recientes)
y se reanudan automaticamente. Tras un corte, el runner ignora
directorios de checkpoint incompletos y elige el step mas alto que conserve
estado, pesos, optimizador, scheduler y RNG coherentes. TensorBoard y la VRAM
pico por epoca se guardan en artifacts/metrics. Cada perfil escribe bajo su
propio subdirectorio de `artifacts/checkpoints` para impedir colisiones entre
el perfil principal y los fallbacks OOM. Si se corta la energia durante un
fallback, `artifacts/metrics/oom_state.json` recuerda atomicamente los perfiles
que ya agotaron CUDA para la misma configuracion y split; al reanudar los omite,
vuelve a intentar siempre el ultimo perfil disponible y elimina el estado al
terminar correctamente.

La validacion usa batch 4 y se ejecuta cada 500 updates. Esto conserva 21 puntos
intermedios de `val_loss` durante las tres epocas, ademas del resultado final,
sin repetir 104 veces un barrido completo de los 3109 ejemplos de validacion.
El smoke conserva batch de evaluacion 1 para mantener su consumo minimo.

La longitud renderizada puede auditarse con:

    docker compose -f compose.train.yaml run --rm train python scripts/audit-token-lengths.py

En el split congelado actual, 57 registros de train (0,102 %) y 2 de validacion
(0,064 %) superan 8192 tokens. El runner los excluye antes de tokenizar para SFT
en vez de truncarlos silenciosamente; registra cada hash, categoria y longitud,
ademas de la huella ordenada del subconjunto realmente entrenado, tanto en
`training_result.json` como en `export_manifest.json`. Los dos edge cases largos
del test se conservan para evaluar robustez y no intervienen en el entrenamiento.

El entrenamiento completo se niega a arrancar hasta que
`benchmarks/baseline_results.json` sea final, incluya sintaxis y LLM-juez y su
hash corresponda exactamente al test congelado. El smoke test no requiere ese
baseline.

Al finalizar se exportan:

- artifacts/adapter;
- artifacts/merged en safetensors;
- artifacts/gguf con q4_k_m y q8_0.
- artifacts/export_manifest.json con SHA-256 de cada archivo exportado y los
  hashes exactos de train/validation/test usados para entrenar.

El runner usa la ruta devuelta por la version instalada de Unsloth (que genera
`<merged>_gguf`), la traslada a `artifacts/gguf` y comprueba que existan tanto
q4_k_m como q8_0 antes de permitir la creacion del manifiesto de exportacion.

Los benchmarks fine-tuned verifican ese manifiesto antes de cargar el modelo
merged o conectarse al servidor GGUF. Para GGUF tambien consultan `/props` de
llama-server y exigen que alias, ruta, SHA-256 y cuantizacion `q4_k_m`
correspondan al archivo manifestado; no basta con que el endpoint responda.

## Benchmark e informe

El baseline debe ejecutarse antes del entrenamiento:

    docker compose -f compose.train.yaml run --build --rm train python -m docker_k8s_finetune.cli benchmark --variant baseline --skip-judge --skip-syntax

Las variantes Unsloth usan lotes de inferencia de 4 solicitudes y el mismo
presupuesto maximo de 2048 tokens. El tamano del lote forma parte de la huella
del cache; cambiarlo no mezcla salidas ni metricas de regimenes distintos. La
generacion por lotes conserva escritura atomica por registro y vuelve a modo
secuencial para cualquier lote que falle. Los backends HTTP que no implementan
batching nativo conservan ejecucion secuencial. Cada resultado registra la latencia
concurrente y el throughput agregado del lote; el recuento de cada salida se
corta en su primer EOS para excluir el padding que completa el lote. Los misses
se ordenan establemente por longitud del prompt antes de agruparse para reducir
padding; nunca se consulta la longitud de la respuesta de referencia. La
pertenencia al grupo queda identificada en cada cache. Si un corte deja solo
parte de un lote escrita, al reanudar se regenera ese grupo completo y no se
desplazan los emparejamientos posteriores. Una
salida solo se considera truncada si alcanza el presupuesto sin haber emitido
EOS; terminar exactamente en el ultimo token permitido no produce un falso
positivo. Una instruccion fija que exige respuestas concisas y bloques YAML/Dockerfile
completos. Cada resultado registra `truncated`; una tasa superior al 1 % hace
fallar el gate `completion.generation` y mantiene el benchmark provisional. La
cache es reanudable y los cambios exclusivos del umbral de auditoria no fuerzan
una nueva inferencia. Las respuestas completas creadas con el limite anterior
de 1536 se migran de forma segura; las que alcanzaron ese techo se regeneran.
Si una salida alcanza 2048 tokens, se reintenta una sola vez con penalizacion de
repeticion 1.10 para cortar bucles degenerados. Las caches anteriores a esta
politica migran las respuestas no truncadas sin regenerarlas y envian las
truncadas directamente al reintento. Se registran intentos, tokens totales y
latencia acumulada para no ocultar el coste del rescate.

Despues del entrenamiento:

    docker compose -f compose.train.yaml run --build --rm train python -m docker_k8s_finetune.cli benchmark --variant finetuned_safetensors --skip-judge --skip-syntax

Para GGUF se inicia llama.cpp u Ollama con el GGUF exportado y se configuran:

    powershell -ExecutionPolicy Bypass -File scripts/start-finetuned-gguf.ps1

    $env:GGUF_LLM_BASE_URL="http://host.docker.internal:8080/v1"
    $env:GGUF_LLM_MODEL="qwen-docker-k8s"
    $env:GGUF_LLM_API_KEY=""

    docker compose -f compose.train.yaml run --build --rm train python -m docker_k8s_finetune.cli benchmark --variant finetuned_gguf --skip-judge --skip-syntax

El lanzador selecciona el q4_k_m desde `artifacts/export_manifest.json`, exige
una unica coincidencia y verifica bytes y SHA-256 antes de iniciar llama.cpp.

El juez externo usa un prompt fijo en benchmarks/judge_prompt.md:

En este equipo se usa Gemma 4 12B Q6 como juez independiente. Debe iniciarse
solo despues de liberar de la GPU el modelo que se esta evaluando:

    powershell -ExecutionPolicy Bypass -File scripts/start-judge-gemma4.ps1 -VerifyOnly
    powershell -ExecutionPolicy Bypass -File scripts/start-judge-gemma4.ps1

El lanzador comprueba el tamano (10.685.011.360 bytes) y el SHA-256
`eb0f252863d14f7782122a4ac7e8744ed6e4a9fc132584d94686a9441f7c5d35`.
El cliente exige ademas que `/props` confirme el alias, nombre de archivo y
cuantizacion fijados en `config/benchmark.yaml`; esa identidad queda guardada
en cada resultado juzgado.

En otra consola se configuran las variables y se completa el juicio guardado:

    $env:JUDGE_LLM_BASE_URL="http://host.docker.internal:8081/v1"
    $env:JUDGE_LLM_MODEL="gemma-4-12b-judge"
    $env:JUDGE_LLM_API_KEY=""

La sintaxis se completa despues desde Windows, donde el proceso tiene acceso al
daemon Docker que ejecuta las imagenes fijadas de kubeconform y hadolint. Esto
evita depender de Docker-in-Docker dentro de la imagen de entrenamiento:

    venv\Scripts\python -m docker_k8s_finetune.cli benchmark-syntax --variant baseline
    venv\Scripts\python -m docker_k8s_finetune.cli benchmark-syntax --variant finetuned_safetensors
    venv\Scripts\python -m docker_k8s_finetune.cli benchmark-syntax --variant finetuned_gguf

Con una sola GPU de 16 GB, la inferencia y el juicio se ejecutan en dos fases.
Los objetivos `benchmark-baseline`, `benchmark-finetuned` y `benchmark-gguf`
guardan primero todas las predicciones. La fase de sintaxis actualiza el mismo
JSON de forma atomica. Tras liberar el modelo evaluado e iniciar el juez, se
completa tambien el juicio de forma reanudable:

    docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli benchmark-judge --variant baseline
    docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli benchmark-judge --variant finetuned_safetensors
    docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli benchmark-judge --variant finetuned_gguf

Cada respuesta del juez se persiste atomicamente. Un resultado solo deja de
ser provisional cuando cubre el test congelado completo, la sintaxis y el
juicio.

La inferencia tambien usa un cache por registro en
`benchmarks/work/<variante>/generation_cache`. Su huella incluye mensajes,
backend y parametros de generacion. Una interrupcion reanuda predicciones ya
guardadas y cualquier error pendiente mantiene el benchmark como provisional.

Finalmente:

    docker compose -f compose.train.yaml run --rm train python -m docker_k8s_finetune.cli report

El informe compara por categoria exact match, similitud semantica, validez
kubeconform/hadolint, LLM-juez, latencia, tokens por segundo, fuera de dominio
y degradacion por cuantizacion. Cualquier regresion se enumera explicitamente.
El historial completo de train/eval se persiste en
`artifacts/metrics/training_result.json`; el informe final exige esos puntos (o
un `trainer_state.json` de respaldo) y genera `benchmarks/training_loss.svg`.
Tambien rechaza variantes incompletas o comparaciones que difieran en test,
parametros de generacion, prompt/manifiesto del juez o identidad del GGUF juez.
Los resultados de inferencia directa registran tambien el modelo de GPU, VRAM
total, capacidad CUDA y versiones de Torch/CUDA usadas para que las medidas de
latencia y rendimiento queden ligadas al hardware real.
La generacion evaluada y el juez usan temperatura cero y la semilla 3407,
registrada en `config/benchmark.yaml` y en cada resultado.

## Reproducibilidad y seguridad

- No se hace push a Hugging Face ni se publican datasets automaticamente.
- data/raw, data/interim, data/quarantine y artifacts estan ignorados.
- data/processed queda versionable y conserva licencia/atribucion por registro.
- Los validadores se ejecutan con imagenes fijadas por digest.
- Ningun benchmark provisional puede generar el informe final salvo que se use
  explicitamente --allow-provisional.

## Perfil ROCm: Radeon AI PRO R9700

La ruta ROCm usa `config/training.rocm.yaml`, `Dockerfile.train.rocm` y
`compose.train.rocm.yaml`; la ruta NVIDIA anterior permanece independiente. El
perfil usa LoRA BF16 sin cuantización, `adamw_torch`, entrenamiento solo de texto,
pérdida solo en respuestas, semilla 3407, los mismos hashes del modelo,
de train y de test, y la excepción de validación descrita abajo.
El smoke toma registros directamente de `train.jsonl` y `val.jsonl` congelados.
Todas las salidas nuevas se escriben en `artifacts/rocm`; no se reanudan
checkpoints archivados de NVIDIA. Es posible cargar únicamente pesos de un
adaptador LoRA compatible y validado como inicialización de un experimento nuevo,
pero sus estados de optimizador, scheduler y RNG no equivalen a reanudar el
entrenamiento y la procedencia de ese experimento debe declararse aparte.

En el host examinado el 18-09-2026, `amd-smi` informa Radeon AI PRO R9700
(gfx1201, 32624 MB), `amdgpu 7.1.3.31500000` y ROCm 10.0.0. Ubuntu es
24.04.5 y el kernel es 7.0.0-31-generic. La matriz oficial de ROCm 10.0.0
incluye gfx1201, pero valida Radeon sobre Ubuntu 24.04.4 con kernel HWE 6.17;
la combinación actual del host queda fuera de esa matriz y requiere el smoke
empírico. La imagen ROCm de Unsloth está fijada por digest y usa ROCm 7.2.4,
por lo que debe verificarse su interoperabilidad con el driver del host antes
de usarla para un entrenamiento largo.

```bash
PATH=/tmp/qwen-git-lfs/usr/bin:$PATH git lfs ls-files
sha256sum data/processed/{train,val,test}.jsonl
docker compose -f compose.train.rocm.yaml config --quiet
docker compose -f compose.train.rocm.yaml build train
docker compose -f compose.train.rocm.yaml run --rm train python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
docker compose -f compose.train.rocm.yaml run --rm train python -m docker_k8s_finetune.cli train --config config/training.rocm.yaml --smoke-test --preflight --no-export
docker compose -f compose.train.rocm.yaml run --rm train python -m docker_k8s_finetune.cli train --config config/training.rocm.yaml --smoke-test --no-export
```

El entrenamiento completo se ejecuta solo tras autorización explícita:

```bash
docker compose -f compose.train.rocm.yaml run --rm train python -m docker_k8s_finetune.cli train --config config/training.rocm.yaml
```

Para evaluar las exportaciones ROCm, use `config/benchmark.rocm.yaml` con
`--config` en los comandos `benchmark`, `benchmark-syntax`, `benchmark-judge`
y `report`. Conserva los manifiestos y el test congelados; escribe resultados
en `benchmarks/rocm` y lee `artifacts/rocm/export_manifest.json`. El baseline
publicado previamente sigue siendo el requisito de entrenamiento según
`config/training.rocm.yaml`.

### Estado de la comprobación ROCm (18-09-2026)

La imagen construida detecta `/dev/kfd` y `/dev/dri/renderD128`; PyTorch
`2.12.1+rocm7.2` informa HIP `7.2.53211`, `torch.cuda.is_available() == True`
y `gfx1201`. La imagen incluye Unsloth `2026.9.4`, Transformers `5.5.0`, TRL
`0.24.0`, PEFT `0.20.0`, TensorBoard `2.20.0`, sentence-transformers `5.2.0`
y llama.cpp en el commit `44be98f057e9f9902a8ee12630e181c7f8ec2953`.
Unsloth empleó su implementación de PyTorch al faltar flash-linear-attention
y causal-conv1d; el smoke BF16 terminó con `adamw_torch`.

El smoke de un paso, con evaluación y checkpoint, terminó en 136 s:
`train_loss=0.3450`, `eval_loss=1.9086` y máximo observado de 10.68 GiB de
VRAM. Una calibración de dos pasos sobre un registro de 7906 tokens y
validación de 7529 tokens con contexto 8192 terminó en 168 s; el máximo
observado fue 22.66 GiB, con 9.20 GiB libres. La auditoría de longitudes
halló 57 registros train y 2 val por encima de 8192 tokens; la regla
`overlength_action: exclude` los excluiría y registraría los conteos y hashes.
Con batch 1 y acumulación 16, la estimación preliminar de tres épocas es de
2 a 5 días, según la distribución de longitudes, evaluaciones y guardados.
Reservar al menos 40 GiB para salidas además de la imagen, modelo y datos.

**Excepción de procedencia de validación:** el `val.jsonl` seguido por Git y
copiado desde la máquina Windows mide 9 581 643 bytes y tiene SHA-256
`bcf223584546fe573830470fcffedcffe92817034078808934bc892f6cbf90ac`.
`split_statistics.json` conserva 9 581 692 bytes y SHA-256
`bb5c3c04504fd7d307e4f3305ffae9c8ab5765882e85964f9bcfccbb2b25fe4a`.
La diferencia corresponde a una credencial redactada en el registro de
validación 1659, de la pregunta Stack Overflow 50271985. Los 3111 registros
coinciden con sus asignaciones del manifiesto y los recuentos por categoría y
fuente. Todos los `content_hash` de validación coinciden salvo el del registro
redactado. No se restaura la credencial ni se cambian los archivos de datos o el
manifiesto histórico. `config/training.rocm.yaml` fija ambos hashes de archivo,
los dos tamaños y las huellas anterior y actual de ese registro. El preflight
ROCm audita los 3111 registros y acepta solo esa excepción exacta; la identidad
efectiva de validación que se guarda en checkpoints es el SHA-256 del archivo
redactado. Esta excepción altera de forma explícita el contrato de validación
respecto al manifiesto original; la ruta NVIDIA sigue exigiendo el hash original.

```bash
docker compose -f compose.train.rocm.yaml run --rm train python -m docker_k8s_finetune.cli train --config config/training.rocm.yaml --preflight --no-export
```

La exportación del adaptador del smoke también terminó: Unsloth fusionó pesos BF16,
convirtió a GGUF y generó `Q4_K_M` (2 783 446 720 bytes), `Q8_0`
(4 610 580 160 bytes) y `BF16-mmproj` (675 568 864 bytes). El finalizador del
pipeline verificó las cuantizaciones y reubicó los tres archivos. La prueba
creó `artifacts/rocm/export_smoke`; sus archivos son experimentales y no son
el resultado del entrenamiento completo.
