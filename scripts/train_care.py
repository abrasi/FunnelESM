#!/usr/bin/env python3
"""CARE EC1 baseline/Funnel experiments using Hugging Face Trainer."""
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
from zeus.monitor import ZeusMonitor
from sklearn.metrics import f1_score, jaccard_score, matthews_corrcoef

from funnel_esm.care import CARE_LABELS, load_care
from funnel_esm.metrics import sigmoid, inverse_frequency_label_weights, optimize_mcc_thresholds, apply_thresholds
from funnel_esm.modeling_classification import MaskedAttentionEsmForSequenceClassification
from funnel_esm.modeling_funnel_esm import FunnelEsmForSequenceClassification, FunnelEsmPoolQueryForSequenceClassification
from train_baseline import ROPE_KEY_MAPPING

ARCHITECTURES = ('baseline', 'mean_layer_6', 'mean_layers_4_8', 'pool_query_layer_6', 'pool_query_layers_4_8')


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True)
    p.add_argument('--test_csv')
    p.add_argument('--cluster_metadata_csv')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--architecture', choices=ARCHITECTURES, required=True)
    p.add_argument('--model_name', default='facebook/esm2_t12_35M_UR50D')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validation_seed', type=int, default=0)
    p.add_argument('--validation_size', type=float, default=0.1)
    p.add_argument('--mixed_precision', choices=('no', 'fp16', 'bf16'), default='bf16')
    p.add_argument('--focal_gamma', type=float, default=1.0)
    p.add_argument('--convergence_pilot', action='store_true')
    return p


def build_model(a):
    config = AutoConfig.from_pretrained(a.model_name)
    config.num_labels = 7
    config.problem_type = 'multi_label_classification'
    config.id2label = dict(enumerate(CARE_LABELS))
    config.label2id = {name: i for i, name in enumerate(CARE_LABELS)}
    config.deeploc_focal_gamma = a.focal_gamma  # Shared head currently names this setting deeploc_*.
    classes = {'baseline': MaskedAttentionEsmForSequenceClassification}
    if a.architecture.startswith('mean_'):
        cls = FunnelEsmForSequenceClassification
        config.funnel_pooling_type = 'pairwise_masked_mean'
    elif a.architecture.startswith('pool_query_'):
        cls = FunnelEsmPoolQueryForSequenceClassification
        config.funnel_pooling_type = 'pool_query_only_mean'
    else:
        cls = classes['baseline']
    if a.architecture != 'baseline':
        config.funnel_pool_after_layers = ([4, 8] if 'layers_4_8' in a.architecture else [6])
        config.funnel_pool_stride = 2
    return cls.from_pretrained(a.model_name, config=config, key_mapping=ROPE_KEY_MAPPING)


def metrics_at(logits, labels, thresholds=0.5):
    probabilities = sigmoid(logits)
    pred = (probabilities > thresholds).astype(int)
    true = np.asarray(labels, dtype=int)
    values = {
        'macro_f1': float(f1_score(true, pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(true, pred, average='micro', zero_division=0)),
        'jaccard': float(jaccard_score(true, pred, average='samples', zero_division=0)),
        'accuracy': float(np.mean(np.all(true == pred, axis=1))),
        'num_labels_predicted': float(pred.sum(axis=1).mean()),
    }
    for i, label in enumerate(CARE_LABELS):
        values[f'mcc_{label.lower()}'] = float(matthews_corrcoef(true[:, i], pred[:, i]))
    values['macro_mcc'] = float(np.mean([values[f'mcc_{label.lower()}'] for label in CARE_LABELS]))
    values['top1_ec1_accuracy'] = float(np.mean(true[np.arange(len(true)), probabilities.argmax(axis=1)] == 1))
    return values


def main():
    a = parser().parse_args()
    if a.epochs < 1 or a.learning_rate <= 0 or a.batch_size < 1:
        raise ValueError('epochs, learning rate and batch size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for Zeus measurements')
    if a.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 not supported by this GPU')
    set_seed(a.seed)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    data = load_care(a.train_csv, tok, test_csv=a.test_csv,
                     validation_size=a.validation_size, validation_seed=a.validation_seed,
                     cluster_metadata_csv=a.cluster_metadata_csv)
    model = build_model(a)
    model.set_label_weights(torch.tensor(inverse_frequency_label_weights(data['train']['labels']), dtype=torch.float32))
    for param in model.base_model.parameters():
        param.requires_grad = False
    args = TrainingArguments(output_dir=str(out), learning_rate=a.learning_rate,
                            per_device_train_batch_size=a.batch_size,
                            per_device_eval_batch_size=a.batch_size, num_train_epochs=a.epochs,
                            eval_strategy='epoch', logging_strategy='epoch',
                            save_strategy='no' if a.convergence_pilot else 'epoch',
                            load_best_model_at_end=not a.convergence_pilot,
                            metric_for_best_model='eval_loss' if not a.convergence_pilot else None,
                            greater_is_better=False, save_total_limit=1,
                            fp16=a.mixed_precision == 'fp16', bf16=a.mixed_precision == 'bf16',
                            report_to='none', seed=a.seed, data_seed=a.seed)
    collator = DataCollatorWithPadding(tok)
    trainer = Trainer(model=model, args=args, train_dataset=data['train'],
                      eval_dataset=data['validation'], data_collator=collator,
                      compute_metrics=lambda p: metrics_at(p.predictions, p.label_ids), processing_class=tok)
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()], cpu_indices=[])
    torch.cuda.reset_peak_memory_stats()
    monitor.begin_window('train')
    trainer.train()
    measured = monitor.end_window('train')
    energy = float(measured.total_energy)
    resource = {'train_gpu_energy_j': energy, 'train_gpu_energy_kwh': energy / 3600000,
                'train_time_s': float(measured.time),
                'peak_gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9}
    (out / 'epoch_history.json').write_text(json.dumps(trainer.state.log_history, indent=2))
    (out / 'training_protocol.json').write_text(json.dumps({**vars(a), 'early_stopping': False,
       'checkpoint_selection': 'lowest validation eval_loss', 'label_scheme': 'direct_ec1_multilabel',
       'validation_split': 'fixed group split of training set',
       'test_used_in_selection': False}, indent=2))
    if a.convergence_pilot:
        result = {'architecture': a.architecture, 'seed': a.seed, 'resources': resource,
                  'epoch_history': 'epoch_history.json'}
    else:
        if trainer.state.best_model_checkpoint is None:
            raise RuntimeError('No best checkpoint selected')
        # Threshold calibration uses held-out validation, never the test set.
        trainer.compute_metrics = None
        monitor.begin_window('calibration')
        calibrate = trainer.predict(data['validation'])
        thresholds = optimize_mcc_thresholds(sigmoid(calibrate.predictions), calibrate.label_ids)
        calibrated = monitor.end_window('calibration')
        calibration_energy = float(calibrated.total_energy)
        trainer.compute_metrics = lambda p: metrics_at(p.predictions, p.label_ids, thresholds)
        result = {'architecture': a.architecture, 'seed': a.seed,
                  'best_checkpoint': trainer.state.best_model_checkpoint,
                  'best_validation_loss': trainer.state.best_metric,
                  'thresholds': dict(zip(CARE_LABELS, thresholds.tolist())),
                  'resources': {**resource, 'threshold_calibration_gpu_energy_j': calibration_energy,
                     'threshold_calibration_gpu_energy_kwh': calibration_energy / 3600000,
                     'threshold_calibration_time_s': float(calibrated.time),
                     'pipeline_gpu_energy_kwh': (energy + calibration_energy) / 3600000,
                     'pipeline_time_s': float(measured.time + calibrated.time)}}
        if 'test' in data:
            pred = trainer.predict(data['test'], metric_key_prefix='test')
            result['test_metrics'] = {key: float(value) for key, value in pred.metrics.items()
                                      if isinstance(value, (float, int, np.number))}
            probs = sigmoid(pred.predictions)
            binary = apply_thresholds(probs, thresholds, force_at_least_one=False, inclusive=False)
            with (out / 'test_predictions.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sequence_index'] + [f'prob_{n}' for n in CARE_LABELS] +
                                [f'true_{n}' for n in CARE_LABELS] + [f'pred_{n}' for n in CARE_LABELS])
                for i, (pr, truth, b) in enumerate(zip(probs, pred.label_ids, binary)):
                    writer.writerow([i, *pr.tolist(), *truth.tolist(), *b.tolist()])
        trainer.save_model()
    (out / 'run_result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
