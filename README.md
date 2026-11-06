# FunnelESM

ESM-2 congelado con una cabeza de atención enmascarada para DeepLoc y CARE.
Las variantes reducen la secuencia mediante mean-pool o pool-query-only.

## Instalación

Instala un PyTorch con CUDA compatible con tu servidor y ejecuta:

```bash
pip install -e .
```

Se mantiene `transformers==5.21.0`, porque los modelos utilizan APIs internas de ESM.
Los entrenamientos y las mediciones reales requieren CUDA y Zeus con acceso a NVML.

## Estructura

| Función | DeepLoc | CARE |
|---|---|---|
| Lanzar piloto | `run_deeploc_convergence_pilot.py` | `run_care.py --phase pilot` |
| Lanzar finales | `run_deeploc_final_experiments.py` | `run_care.py --phase final` |
| Resumir curvas | `summarize_convergence_pilot.py` | `summarize_care_convergence_pilot.py` |
| Entrenar baseline | `train_deeploc_baseline.py` | `train_care_baseline.py` |
| Entrenar mean-pool | `train_funnel.py` | El mismo script con `--task care` |
| Entrenar pool-query | `train_funnel_pool_query.py` | El mismo script con `--task care` |
| Cargar datos | `deeploc.py` | `care.py` |
| Métricas | `metrics.py` | El mismo módulo |
| Evaluar checkpoints | `run_hpa_evaluation.py`, `evaluate_hpa.py` | `evaluate_care_inference.py` |

Los ejecutables están en `scripts/` y los módulos en `src/funnel_esm/`.
`modeling_classification.py`, `modeling_funnel_esm.py` y `pooling.py` se comparten.
Los baselines se separan porque DeepLoc hace CV y selección de checkpoint interna,
mientras que CARE termina con un ajuste sobre todo train y épocas fijas.

No hay `architectures.py`, `care_experiments.py`, `care_training.py`,
`train_care.py`, `train_care_final.py` ni un resumidor de finales.
Las arquitecturas se indican en el diccionario **EXPERIMENTS** del lanzador:

```python
EXPERIMENTS = {
    "baseline": ("train_care_baseline.py", []),
    "mean_layer_16": ("train_funnel.py", [16]),
    "mean_layers_11_22": ("train_funnel.py", [11, 22]),
    "pool_query_layer_16": ("train_funnel_pool_query.py", [16]),
    "pool_query_layers_11_22": ("train_funnel_pool_query.py", [11, 22]),
}
```

Los números son capas de pooling explícitas, contando desde 1. Los nombres son
etiquetas de experimento; no se interpretan para decidir capas. Los lanzadores
DeepLoc usan el mismo formato de diccionario, conservando sus capas 6 y 4/8.

## CARE: piloto

Por defecto se comparan `baseline` y `mean_layer_16` con ESM-2 650M.
Los LR del siguiente ejemplo son una búsqueda inicial, no un óptimo establecido:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_care.py \
  --phase pilot \
  --train_csv /ruta/protein_train50.csv \
  --output_dir outputs/care650_pilot \
  --model_name facebook/esm2_t33_650M_UR50D \
  --epochs 30 --learning_rates 1e-5 5e-5 1e-4 \
  --batch_size 8
```

El piloto crea una validación interna fija dentro de train: 10% de grupos,
semilla 0. Por defecto un grupo es una secuencia única. Si dispones del mapa
`Sequence,clusterRes50`, añade `--cluster_metadata_csv /ruta/metadatos.csv` para
separar clústeres completos; debe cubrir todas las secuencias de train. Sin ese mapa
se impide compartir secuencias idénticas, pero no se garantiza separación por homología.
El piloto no acepta ni carga los tests oficiales.

Se guardan `epoch_history.json` y `run_result.json` por ejecución, y al terminar
se generan `learning_curves.csv`, `pilot_summary.csv` y `pilot_summary.json`.
Las curvas incluyen pérdida, F1, MCC, Jaccard, exactitud multilabel y top-1 EC1.
La energía del piloto incluye entrenamiento y las validaciones por época.

Puedes volver a compactar la información con:

```bash
python scripts/summarize_care_convergence_pilot.py --pilot_dir outputs/care650_pilot
```

`best_epoch` describe el mínimo observado de pérdida de validación. El resumidor
no fija hiperparámetros, no guarda una selección automática y no elige checkpoints.

## CARE: finales

Fija LR y épocas mirando únicamente el piloto. Los valores siguientes son un ejemplo:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_care.py \
  --phase final \
  --train_csv /ruta/protein_train50.csv \
  --test_csvs /ruta/30_protein_test.csv /ruta/30-50_protein_test.csv \
  --output_dir outputs/care650_final \
  --model_name facebook/esm2_t33_650M_UR50D \
  --learning_rate 5e-5 --epochs 30 \
  --batch_size 8 --seeds 42 43 44 45 46
```

`--learning_rate`, `--epochs` y los tests son obligatorios en finales. Cada run
carga los pesos preentrenados, inicializa la cabeza, usa **todo train** y evalúa
los pesos de la última época. No hay selección automática de checkpoint, early
stopping ni validación interna durante ese ajuste final. El LR usa el scheduler
lineal habitual de Trainer, igual que DeepLoc. Se entrena únicamente la cabeza.

Cada run guarda `final_model/`, `run_result.json`, `training_protocol.json`,
`epoch_history.json` y predicciones por proteína y test. El lanzador escribe
**care_results.json**, tanto para la campaña como para cada arquitectura:

```json
{
  "config": {},
  "runs": [],
  "summary": {
    "baseline": {
      "training": {"train_time_s": {"mean": null, "std": null}},
      "tests": {"30_protein_test": {"metrics": {}, "resources": {}}}
    }
  }
}
```

El fragmento solo ilustra la estructura de los resultados.
La forma `config/runs/summary` y las entradas `mean/std` siguen la organización de
los resultados CV de DeepLoc. En CARE se agregan **semillas**, no folds. Con una
sola semilla la desviación es `null`. No se necesita un script posterior de resumen.
Los tiempos/energías de los tests mediante Trainer son diagnósticos; para comparar
inferencia con CLEAN utiliza el benchmark del siguiente apartado.

## CARE: inferencia

Usa también el proyecto CLEAN_CARE_INFERENCE actualizado que acompaña esta revisión.
Su cargador ya admite nombres libres y capas manuales guardadas en los checkpoints.
Se conserva el mismo bucle de medición para CLEAN y FunnelESM:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_care_inference.py \
  --models_dir outputs/care650_final \
  --clean_project /ruta/CLEAN_CARE_INFERENCE \
  --train_csv /ruta/protein_train50.csv \
  --test_csvs /ruta/30_protein_test.csv /ruta/30-50_protein_test.csv \
  --output_dir outputs/care650_inference \
  --batch_size 1 --warmup 1 --repeats 5
```

El script evalúa una campaña o un único run sin entrenar. Conserva
`evaluation_result.json` por checkpoint y escribe automáticamente
**inference_results.json** con `config`, `runs` y `summary`, usando `mean/std`.
La métrica principal se llama `top1_ec1_accuracy`. La variación entre checkpoints
se separa de la variación entre repeticiones de tiempo/energía, conservada en los runs.
El comparador común verifica datos, orden, GPU y configuración antes de agregar.

La configuración compartida sigue siendo FP32, recorte central a 1022 residuos,
atención eager y una GPU. La ventana incluye tokenización, PLM, clasificación,
transferencia a CPU y cálculo de accuracy. Excluye lectura del CSV, carga de pesos,
preparación de centros CLEAN y escritura de resultados. No hay acumulación en inferencia.
Para la tabla conjunta CLEAN/FunnelESM se conserva `compare_results.py` del proyecto CLEAN.

## Otro ESM-2 preentrenado y varias GPU

`--model_name` acepta un identificador Hugging Face o una carpeta local de pesos ESM-2.
La profundidad, dimensión de embeddings y número de cabezas proceden de AutoConfig.
La cabeza de clasificación se construye con esa dimensión; no se fija a 35M o 650M.

| Modelo | Capas | Dimensión | Ejemplos de capas de pooling manuales |
|---|---:|---:|---|
| `facebook/esm2_t6_8M_UR50D` | 6 | 320 | `[3]` o `[2, 4]` |
| `facebook/esm2_t12_35M_UR50D` | 12 | 480 | `[6]` o `[4, 8]` |
| `facebook/esm2_t33_650M_UR50D` | 33 | 1280 | `[16]` o `[11, 22]` |

Actualiza también EXPERIMENTS con las capas y nombres que quieras para ese modelo.
Las capas no se reescalan automáticamente: si son inválidas para la profundidad
elegida, el lanzador falla antes de entrenar. Estas posiciones son ejemplos,
no se afirma que sean óptimas. Repite el piloto al cambiar de backbone.

También puedes lanzar directamente un entrenador compartido, sin editar un lanzador:

```bash
torchrun --standalone --nproc_per_node=1 scripts/train_funnel.py \
  --task care --train_csv /ruta/protein_train50.csv \
  --output_dir outputs/care35_mean_pilot \
  --model_name facebook/esm2_t12_35M_UR50D \
  --pool_after_layers 4 8 --architecture mean_layers_4_8 \
  --convergence_pilot --epochs 30 --learning_rate 5e-5
```

Los lanzadores usan torchrun. Para dos GPU visibles, añade `--nproc_per_node 2`.
El batch es por GPU. Batch efectivo = batch por GPU × procesos × acumulación.
La acumulación por defecto es 1; 650M no obliga a aumentarla, porque el encoder está
congelado. Ajusta el batch a la VRAM disponible y mantén la configuración elegida
entre piloto y finales. Los ejemplos de batch no implican una medición de VRAM previa.
La precisión de entrenamiento puede ser BF16 aunque el benchmark posterior sea FP32.

`--architectures` permite seleccionar entradas de EXPERIMENTS. Para LR o épocas
diferentes por arquitectura, ejecuta el mismo lanzador con una arquitectura y
carpetas separadas. `--dry_run` escribe el plan sin entrenar. `--skip_existing`
omite runs completos únicamente cuando coinciden configuración y contenido de los CSV.

## Cambios y comprobaciones de DeepLoc

El antiguo `train_baseline.py` se llama `train_deeploc_baseline.py`; sus lanzadores
están actualizados. Los entrenadores de pooling eligen el entrenador de tarea con
`--task`, usando DeepLoc por defecto. La carga de pesos y el cálculo de métricas
comparten funciones, y los checkpoints guardan las claves RoPE actuales.
HPA lee la clase del modelo del checkpoint, sin una lista cerrada de nombres de
experimento. Se mantienen las particiones oficiales, la validación interna de 2048,
la selección por pérdida de validación y las predicciones/etiquetas de HPA.

Se corrige además un error previo en la calibración MCC: los candidatos se contaban
con `>=`, pero las predicciones finales usan `>`. El optimizador ahora maximiza MCC
para la comparación estricta que se aplica realmente. Esto cambia **las nuevas
calibraciones** de DeepLoc. Los umbrales ya guardados se leen como están y no se
modifican automáticamente. Para resultados con el criterio corregido hay que repetir
la calibración y las evaluaciones correspondientes; el cambio no exige por sí mismo
volver a entrenar los pesos. CARE no calibra umbrales: usa 0.5 para métricas multilabel.

## Datos y validación

CARE admite CSV/TSV con `Entry`, `Sequence`, `EC number`. Une etiquetas de secuencias
idénticas y de EC separados por `;`, conserva EC1–EC7 y rechaza filas inválidas y
solapamiento exacto train/test. Top-1 EC1 considera acierto si la clase predicha está
en la unión de etiquetas válidas; la métrica adicional `accuracy` es exactitud multilabel.
Los tests oficiales de CARE no se utilizan para escoger hiperparámetros.

Las pruebas ejecutan entrenamiento real pequeño en CPU, validación, guardado,
recarga, predicción y agregación JSON; también comprueban pesos oficiales de ESM-2
8M y configuraciones de varias profundidades/dimensiones. Las mediciones de dispositivo
se sustituyen en estas pruebas: no se presentan como energía de GPU real.
No se ha ejecutado entrenamiento 650M ni medición CUDA/Zeus en este entorno.

```bash
python -m unittest discover -s tests -v
```

Las pruebas opcionales usan `ESM2_8M` (pesos locales), `CARE_ASSETS` (CSV oficiales)
y `CLEAN_PROJECT` (carpeta del evaluador común). Extrae esta revisión en un directorio
nuevo para que no permanezcan los scripts retirados. Los checkpoints anteriores
pueden conservarse en sus ubicaciones originales.
