# Funnel-ESM

Primer baseline de clasificación multilabel de DeepLoc 2.0 con ESM-2 y las
APIs de Hugging Face.

Todavía no contiene ninguna modificación interna de ESM-2.

## Archivos

```text
scripts/train_baseline.py
src/funnel_esm/data.py
src/funnel_esm/metrics.py
```

- `data.py` carga el CSV mediante `datasets.load_dataset`, separa una de las
  cinco particiones para validación y tokeniza las secuencias.
- `metrics.py` calcula macro-F1 y micro-F1.
- `train_baseline.py` crea el modelo, `TrainingArguments` y `Trainer`.

## Instalación

```bash
pip install -e .
```

## Entrenamiento

```bash
python scripts/train_baseline.py \
    --dataset_path /ruta/multisub_5_partitions_unique.csv \
    --output_dir outputs/baseline
```

El identificador `facebook/esm2_t12_35M_UR50D` hace referencia al checkpoint
publicado por Meta en Hugging Face Hub. El modelo se carga con
`AutoModelForSequenceClassification`, es decir, con la implementación de
Transformers.

Por defecto:

- se utiliza la partición 0 para validación;
- se excluyen secuencias de más de 1024 residuos;
- se congela el encoder ESM-2;
- se entrena solamente la cabeza de clasificación;
- se usa el token `<cls>` como representación de la proteína.
