## ROL Y OBJETIVO

Actúa como un ingeniero de ML senior especializado en fine-tuning de LLMs. Vas a construir, de principio a fin, un pipeline reproducible para:

1. Extraer, limpiar y normalizar un dataset de instrucción especializado en **Docker y Kubernetes** (documentación, Q&A de troubleshooting, generación de manifiestos YAML, comandos CLI).
2. Fine-tunear `unsloth/Qwen3.5-4B` con Unsloth (QLoRA) para convertirlo en un modelo **experto de dominio estrecho**, no un generalista — prioriza precisión y consistencia en Docker/K8s sobre capacidad general.
3. Evaluar el modelo **antes y después** del entrenamiento con un benchmark reproducible, y producir un informe comparativo.

El objetivo final es un modelo pequeño (4B) pero muy sólido en un ámbito muy concreto, exportado a GGUF para servir con llama.cpp/Ollama.

## RESTRICCIONES DE ENTORNO

- GPU disponible: NVIDIA GeForce RTX 5060 Ti 16GB (arquitectura Blackwell, chip GB206, bus de memoria de 128-bit, 448 GB/s de ancho de banda). Es la única GPU del entorno; diseña todo para caber ahí cómodamente con QLoRA 4-bit sobre un modelo de 4B (consumo esperado ~6-8GB, así que hay margen para subir `max_seq_length` o batch size frente al mínimo necesario).
- **Nota de instalación específica de Blackwell/RTX 50-series**: antes de instalar Unsloth, comprueba si el entorno necesita recompilar `xformers` para `sm_120` (arquitectura Blackwell consumer):
  ```bash
  pip uninstall xformers -y
  pip install ninja
  export TORCH_CUDA_ARCH_LIST="12.0"
  git clone --depth=1 https://github.com/facebookresearch/xformers --recursive
  cd xformers && python setup.py install && cd ..
  ```
  Alternativa más simple y recomendada si Docker está disponible: usar la imagen Docker oficial de Unsloth (`unsloth/unsloth`), que ya trae el stack compilado para Blackwell sin pasos manuales. Decide cuál usar según lo que detectes disponible en el entorno, y documenta la decisión en el README.
- Modelo base: `unsloth/Qwen3.5-4B` (NO uses la versión `-GGUF`, esa es solo para inferencia; necesitas los pesos originales para entrenar).
- Librerías: `unsloth`, `trl`, `transformers`, `peft`, `datasets`, `bitsandbytes`, `accelerate`. Usa las versiones más recientes compatibles entre sí y con soporte confirmado para Blackwell (CUDA 12.8+); resuelve conflictos si aparecen.
- Todo el código debe ser reproducible: scripts parametrizados por CLI o config YAML, no notebooks sueltos sin equivalente en script.
- Registra semillas aleatorias en todos los pasos con componente estocástico (splits, sampling, shuffling).

## FUENTES DE DATOS A UTILIZAR

Organiza la extracción en módulos independientes, uno por fuente, todos con salida a un formato intermedio común (JSONL con campos `source`, `category`, `raw_content`, `license`, `url`).

### Documentación oficial (Tier 1 — licencia permisiva, citar fuente)
- `docker/docs` (Apache-2.0): https://github.com/docker/docs — clonar y trocear por sección/página en Markdown.
- `kubernetes/website` (CC BY 4.0): https://github.com/kubernetes/website — usar solo `content/en/docs/`.
- Docs de ecosistema adyacente (Helm, cert-manager, Istio, containerd, Argo CD, ingress-nginx, CoreDNS) — mismo tratamiento, útiles para ampliar cobertura de casos de uso reales combinados con K8s.

### Datasets ya construidos (verificar licencia antes de mezclar)
- `substratusai/k8s-instructions` (Apache-2.0) — pares instrucción→YAML, listos para usar.
- `substratusai/the-stack-yaml-k8s` (licencia por archivo, `other`) — 276k manifiestos YAML reales; usar solo como fuente para generación sintética de instrucciones (método reverse-instruction, ver más abajo), no reproducir tal cual sin comprobar licencia por archivo.
- `MattCoddity/dockerNLcommands` (Apache-2.0) — NL→comando Docker.
- `genaidevops/kubernetes-stackoverflow-questions` (CC BY 4.0) — Q&A de Stack Overflow ya extraído.
- `ComponentSoft/k8s-kubectl-35k` y `ComponentSoft/k8s-kubectl-cot-20k` — **sin licencia especificada en el momento de escribir esto**. Inclúyelos en una carpeta `quarantine/` separada del dataset final hasta que se confirme la licencia; no los mezcles en el split de entrenamiento por defecto.

### Extracción propia (requiere pipeline de scraping/consulta)
- Stack Overflow para preguntas etiquetadas `docker`, `docker-compose`, `dockerfile` (no existe dataset listo específico de Docker): usar BigQuery público `bigquery-public-data.stackoverflow` o el Stack Exchange Data Dump. Filtrar por tag, quedarte con la respuesta aceptada o de mayor score, limpiar HTML. Licencia CC BY-SA 4.0 — mantener atribución si el dataset se redistribuye.
- Issues cerrados con label de bug/resuelto de: `moby/moby`, `kubernetes/kubernetes`, `containerd/containerd`, `moby/buildkit`, `helm/helm`, `cert-manager/cert-manager`, `kubernetes/ingress-nginx`. Extraer pares (título+descripción del problema) → (comentario que lo resolvió, si está marcado o tiene mayor reacción positiva).
- Repo `kubernetes-failure-stories` (postmortems reales etiquetados por componente: CoreDNS, OOMKill, etcd, CPU throttling, etc.) — muy valioso para el bloque de troubleshooting avanzado.

## PIPELINE — ETAPAS A IMPLEMENTAR

Crea un repo con esta estructura (o similar, justifica si cambias algo):

```
docker-k8s-finetune/
├── config/
│   ├── sources.yaml          # config de todas las fuentes, URLs, licencias
│   ├── training.yaml         # hiperparámetros de entrenamiento
│   └── splits.yaml           # proporciones y estrategia de split
├── src/
│   ├── extract/               # un módulo por fuente
│   ├── clean/
│   ├── normalize/
│   ├── dedupe/
│   ├── split/
│   ├── validate/              # validación sintáctica de YAML/comandos generados
│   ├── train/
│   └── benchmark/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── quarantine/            # datasets con licencia sin confirmar
│   └── processed/             # train.jsonl, val.jsonl, test.jsonl
├── benchmarks/
│   ├── baseline_results.json
│   ├── finetuned_results.json
│   └── comparison_report.md
└── README.md
```

### 1. Extracción (`src/extract/`)

Un script por fuente. Cada uno debe:
- Ser idempotente (si ya existe el output, no re-descargar salvo `--force`).
- Guardar metadata de procedencia (`source`, `url`, `license`, `retrieved_at`).
- Loguear cuántos registros extrajo.

### 2. Limpieza (`src/clean/`)

- Eliminar HTML residual, boilerplate de navegación, bloques de código rotos.
- Filtrar por longitud mínima/máxima (descarta ruido y contenido excesivamente largo para el contexto de entrenamiento).
- Eliminar contenido en idiomas distintos a inglés/español si aparece mezclado inconsistentemente (decide un criterio y documéntalo).
- Detectar y eliminar PII básica si aparece en issues/SO (emails, tokens, IPs privadas expuestas en logs de error) con una lista de patrones regex + revisión de muestra.

### 3. Normalización (`src/normalize/`)

Convertir todo a formato ChatML uniforme:

```json
{
  "messages": [
    {"role": "system", "content": "Eres un asistente experto en Docker y Kubernetes..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {"source": "...", "category": "...", "license": "..."}
}
```

Define y etiqueta categorías (`category`) al menos: `concepto`, `comando_cli`, `generacion_yaml`, `troubleshooting`, `dockerfile`, `arquitectura`. Esto es clave para poder estratificar el split y el benchmark por tipo de tarea, no solo en global.

Para contenido de documentación (que no viene en formato pregunta-respuesta), genera pares instrucción→respuesta usando un modelo grande vía API (self-instruct / reverse-instruction), siguiendo el método de substratusai: dado un fragmento de doc o un YAML real, pide al modelo que genere la pregunta/instrucción natural que ese contenido respondería. Valida una muestra manualmente antes de escalar.

### 4. Deduplicación y control de fugas (`src/dedupe/`)

- Deduplicación exacta (hash del contenido normalizado).
- Deduplicación aproximada por embeddings (ej. `sentence-transformers`, similitud coseno > 0.92 se considera duplicado) — esto es crítico para evitar que el mismo caso aparezca en train y en test con wording distinto.
- Genera un reporte de cuántos duplicados se eliminaron y de qué fuente.

### 5. Split train/val/test (`src/split/`)

- Proporción por defecto: 90% train / 5% val / 5% test — parametrizable en `config/splits.yaml`.
- Split **estratificado por `category` y por `source`**, no puramente aleatorio, para que ninguna categoría quede infrarrepresentada en val/test.
- Verifica explícitamente que no haya fugas: tras el split, vuelve a correr la comprobación de similitud por embeddings entre train y test; si algo supera el umbral, muévelo a train y repórtalo.
- El test set debe incluir, a propósito, una submuestra "hard" con preguntas de troubleshooting complejas y edge cases (para que el benchmark no sea solo sobre lo fácil).

### 6. Validación sintáctica del dataset (`src/validate/`)

- Todo YAML generado (sintético o extraído) debe pasar `kubeconform`/`kubeval` antes de entrar al dataset final.
- Todo Dockerfile generado debe pasar `hadolint`.
- Descarta o corrige automáticamente los que fallen; reporta el % de descarte por fuente (una fuente con alto % de fallo es señal de mala calidad, decide si excluirla).

### 7. Entrenamiento (`src/train/`)

Script de fine-tuning con Unsloth, QLoRA 4-bit, ejemplo de arranque (ajusta hiperparámetros tras experimentar). Dado que un 4B en QLoRA deja mucho margen libre en los 16GB de la 5060 Ti, aprovecha ese margen para contexto más largo o batch efectivo mayor en vez de ir al mínimo:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-4B",
    max_seq_length=8192,   # margen de sobra en 16GB para un 4B en QLoRA; baja a 4096 si ves OOM
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)

# SFTConfig orientativo para 16GB:
# per_device_train_batch_size=4, gradient_accumulation_steps=4 (batch efectivo 16)
# monitoriza VRAM con nvidia-smi durante los primeros steps y ajusta si hay holgura o OOM
# entrenar con SFTTrainer sobre data/processed/train.jsonl,
# eval_dataset sobre val.jsonl, 2-3 épocas, lr ~2e-4, cosine schedule
```

Requisitos del script:
- Guarda checkpoints intermedios y permite reanudar.
- Loguea loss de train y val por step/época (usa `wandb` o `tensorboard`, lo que sea más simple de levantar en local).
- Loguea uso de VRAM pico por época (útil para documentar cuánto margen real quedó libre en los 16GB y decidir si merece la pena subir batch/contexto en la siguiente iteración).
- Al final, exporta:
  - El adapter LoRA por separado.
  - El modelo fusionado en safetensors.
  - El modelo fusionado exportado a GGUF (`q4_k_m` y `q8_0` como mínimo) vía `model.save_pretrained_gguf`.

### 8. Benchmark — ANTES del entrenamiento (`src/benchmark/`)

Antes de tocar el modelo, corre el benchmark sobre `unsloth/Qwen3.5-4B` base (sin fine-tune) y guarda resultados en `benchmarks/baseline_results.json`. El benchmark debe incluir, como mínimo:

- **Test set held-out** (del split del punto 5): mide exact match / similitud semántica contra la respuesta de referencia donde aplique (comandos CLI), y validez sintáctica (kubeconform/hadolint) donde aplique (YAML/Dockerfiles).
- **Evaluación cualitativa asistida por LLM-juez**: usa un modelo grande externo como juez para puntuar de 1-5 corrección técnica y completitud sobre una muestra de ~100-200 preguntas del test set, con un prompt de rúbrica fijo y reproducible (guarda el prompt de juicio en el repo).
- **Preguntas trampa / fuera de dominio**: incluye un pequeño set de preguntas no relacionadas con Docker/K8s para verificar que el modelo no ha perdido capacidad general de forma catastrófica (catastrophic forgetting) — esto importa porque el objetivo es especialización, no un modelo inútil fuera de su nicho.
- Latencia/tokens por segundo en tu hardware real (RTX 5060 Ti 16GB), para tener referencia de coste de inferencia.

### 9. Benchmark — DESPUÉS del entrenamiento

Repite exactamente el mismo benchmark (mismo test set, mismo prompt de juez, mismas métricas) sobre el modelo fine-tuneado (tanto la versión safetensors como la GGUF exportada, para detectar si la cuantización degrada resultados). Guarda en `benchmarks/finetuned_results.json`.

### 10. Informe comparativo (`benchmarks/comparison_report.md`)

Genera automáticamente un informe que incluya:
- Tabla comparativa baseline vs fine-tuned por categoría (`concepto`, `comando_cli`, `generacion_yaml`, `troubleshooting`, `dockerfile`, `arquitectura`).
- % de mejora en validez sintáctica de YAML/Dockerfile generado.
- Puntuación del LLM-juez antes/después, con intervalo o desviación si es posible.
- Resultado del set fuera de dominio (para detectar catastrophic forgetting).
- Curvas de loss de entrenamiento (train/val) como gráfico.
- Conclusión honesta: si alguna categoría empeoró, decirlo explícitamente y proponer hipótesis (dataset insuficiente en esa categoría, overfitting, etc.).

## ENTREGABLES ESPERADOS

1. Repo completo y ejecutable con la estructura anterior.
2. `data/processed/{train,val,test}.jsonl` versionados con su ficha de procedencia (`data/processed/DATASHEET.md`: tamaño, fuentes, licencias, fecha).
3. Modelo fine-tuneado exportado en GGUF, listo para servir.
4. `benchmarks/comparison_report.md` con resultados antes/después.
5. `README.md` con instrucciones de reproducción end-to-end (`make extract`, `make clean`, `make train`, `make benchmark` o equivalente).

## CRITERIOS DE ÉXITO

- El pipeline completo corre de extremo a extremo sin intervención manual salvo la validación de muestra puntual del punto 3.
- El modelo fine-tuneado mejora de forma medible sobre el baseline en las categorías de dominio (comando_cli, generacion_yaml, troubleshooting, dockerfile) según el benchmark del punto 10.
- La degradación en el set fuera de dominio es aceptable y está cuantificada (no se ignora).
- Todo el dataset final tiene trazabilidad de licencia por registro.

Antes de empezar a picar código, propón primero el `config/sources.yaml` completo y el esquema exacto de `config/splits.yaml`, y espera confirmación antes de lanzar la extracción masiva.
