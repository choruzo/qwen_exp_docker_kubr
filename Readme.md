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
- Pesos originales en Modelo/Qwen3.5-4B.
- Espacio para checkpoints, modelo fusionado y GGUF en el directorio artifacts.

Se usa la imagen oficial de Unsloth fijada por digest en compose.train.yaml. Es
la ruta elegida para Blackwell porque incluye el stack CUDA/PyTorch preparado y
evita recompilar xformers para sm_120 en Windows.

## Instalacion y comprobaciones

    python -m venv venv
    venv\Scripts\python -m pip install -e ".[extract,dedupe,test]"
    venv\Scripts\python -m docker_k8s_finetune.cli config-validate
    venv\Scripts\python -m pytest
    docker compose -f compose.train.yaml config --quiet

## Pipeline de datos

Las fuentes, licencias y cuarentenas estan declaradas en config/sources.yaml.
La extraccion masiva solo se habilita cuando ese archivo esta aprobado.

    make extract
    make clean
    make normalize

La normalizacion inversa requiere un endpoint OpenAI-compatible:

    SYNTHETIC_LLM_BASE_URL=http://host.docker.internal:8080/v1
    SYNTHETIC_LLM_MODEL=modelo-juez
    SYNTHETIC_LLM_API_KEY=opcional

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

La deduplicacion aproximada usa un modelo de embeddings fijado por commit. El
split vuelve a calcular similitudes exactas entre train/validation/test, mueve
fugas completas a train y hace backfill del mismo estrato. Tambien genera:

- data/processed/train.jsonl, val.jsonl y test.jsonl;
- data/processed/DATASHEET.md;
- benchmarks/test_manifest.jsonl y llm_judge_manifest.jsonl;
- informes de sintaxis, deduplicacion y fugas.

Fuentes que requieren credenciales:

- GitHub issues: GITHUB_TOKEN.
- Stack Overflow Docker via BigQuery: GOOGLE_CLOUD_PROJECT y credenciales ADC.

Las fuentes sin licencia confirmada permanecen en data/quarantine.

## Entrenamiento

Qwen3.5-4B es Qwen3_5ForConditionalGeneration, no un LLM textual clasico. El
runner usa FastVisionModel, congela vision y entrena solo lenguaje, atencion y
MLP. La seleccion de modulos LoRA queda a Unsloth para cubrir tanto la atencion
completa como la atencion lineal hibrida.

    make train-preflight
    make train-smoke
    make train

El perfil principal usa contexto 8192, batch por dispositivo 4 y acumulacion 4.
Si CUDA devuelve OOM, reintenta con contexto 4096 y batch 1 x acumulacion 16.
Los checkpoints se reanudan automaticamente. TensorBoard y la VRAM pico por
epoca se guardan en artifacts/metrics.

Al finalizar se exportan:

- artifacts/adapter;
- artifacts/merged en safetensors;
- artifacts/gguf con q4_k_m y q8_0.

## Benchmark e informe

El baseline debe ejecutarse antes del entrenamiento:

    make benchmark-baseline

Despues del entrenamiento:

    make benchmark-finetuned

Para GGUF se inicia llama.cpp u Ollama con el GGUF exportado y se configuran:

    GGUF_LLM_BASE_URL=http://localhost:8080/v1
    GGUF_LLM_MODEL=qwen-docker-k8s
    GGUF_LLM_API_KEY=opcional

    make benchmark-gguf

El juez externo usa un prompt fijo en benchmarks/judge_prompt.md:

    JUDGE_LLM_BASE_URL=http://localhost:8081/v1
    JUDGE_LLM_MODEL=modelo-juez
    JUDGE_LLM_API_KEY=opcional

Finalmente:

    make report

El informe compara por categoria exact match, similitud semantica, validez
kubeconform/hadolint, LLM-juez, latencia, tokens por segundo, fuera de dominio
y degradacion por cuantizacion. Cualquier regresion se enumera explicitamente.

## Reproducibilidad y seguridad

- No se hace push a Hugging Face ni se publican datasets automaticamente.
- data/raw, data/interim, data/quarantine y artifacts estan ignorados.
- data/processed queda versionable y conserva licencia/atribucion por registro.
- Los validadores se ejecutan con imagenes fijadas por digest.
- Ningun benchmark provisional puede generar el informe final salvo que se use
  explicitamente --allow-provisional.
