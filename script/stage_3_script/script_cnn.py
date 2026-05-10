from pathlib import Path
import csv
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


dough_dir = Path(__file__).resolve().parents[2]

# Davis apartment PC absolute path reference.
# Keep this commented on other devices. Uncomment only when back on that PC
# if automatic path resolution is not enough for the local setup there.
# dough_dir = Path(r'C:\Users\ocaro\OneDrive - University of California, Davis\Spring 2026\ECS 170 - Artificial Intelligence\Project\dough')

if str(dough_dir) not in sys.path:
    sys.path.insert(0, str(dough_dir))

from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_3_code.Evaluate_Precision import Evaluate_Precision
from code.stage_3_code.Evaluate_Recall import Evaluate_Recall
from code.stage_3_code.Evaluate_F1 import Evaluate_F1


def configure_torch():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')


def reset_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_all_metrics(prediction_result):
    evaluators = [
        ('accuracy', Evaluate_Accuracy('accuracy', '')),
        ('precision', Evaluate_Precision('precision', '')),
        ('recall', Evaluate_Recall('recall', '')),
        ('f1', Evaluate_F1('f1', '')),
    ]
    scores = {}
    for metric_name, evaluator in evaluators:
        evaluator.data = prediction_result
        scores[metric_name] = evaluator.evaluate()
    return scores


def save_learning_curve(method_obj, plot_path, title):
    if not method_obj.loss_history:
        return

    epochs = range(1, len(method_obj.loss_history) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, method_obj.loss_history, linewidth=2, label='train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(epochs, method_obj.train_accuracy_history, linewidth=1.5, label='train accuracy', color='tab:green')
    if method_obj.test_accuracy_history:
        eval_epochs = np.linspace(
            method_obj.eval_every,
            len(method_obj.loss_history),
            num=len(method_obj.test_accuracy_history),
            dtype=int,
        )
        ax2.plot(eval_epochs, method_obj.test_accuracy_history, linewidth=1.5, label='test accuracy', color='tab:orange')
    ax2.set_ylabel('Accuracy')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')
    plt.title(title)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)


def run_experiment(experiment, data_dir, result_dir, seed):
    experiment_name = experiment['name']
    dataset_key = experiment['dataset_key']
    config = dict(experiment['config'])
    config['dataset_key'] = dataset_key

    print()
    print('============ Experiment:', experiment_name, '============')
    reset_random_seeds(seed)

    data_obj = Dataset_Loader('stage 3 ' + dataset_key, '', dataset_key=dataset_key)
    data_obj.dataset_source_folder_path = str(data_dir)

    method_obj = Method_CNN('convolutional neural network', '', config=config)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir / (experiment_name + '_'))
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('provided train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

    start_time = time.perf_counter()
    accuracy, _ = setting_obj.load_run_save_evaluate()
    total_time_seconds = time.perf_counter() - start_time

    prediction_result = result_obj.data
    metrics = evaluate_all_metrics(prediction_result)
    metrics['accuracy'] = accuracy

    plot_path = result_dir / (experiment_name + '_training_convergence.png')
    save_learning_curve(method_obj, plot_path, experiment_name + ' Training Convergence')

    summary = {
        'experiment_name': experiment_name,
        'dataset': dataset_key,
        'device': str(method_obj.device),
        'architecture': config.get('architecture', 'auto'),
        'epochs': method_obj.max_epoch,
        'batch_size': method_obj.batch_size,
        'optimizer': method_obj.optimizer_name,
        'learning_rate': method_obj.learning_rate,
        'weight_decay': method_obj.weight_decay,
        'dropout': config.get('dropout', ''),
        'depth': config.get('depth', ''),
        'widen_factor': config.get('widen_factor', ''),
        'augment': method_obj.augment,
        'mixup_alpha': method_obj.mixup_alpha,
        'label_smoothing': method_obj.label_smoothing,
        'use_ema': method_obj.use_ema,
        'test_time_augmentation': method_obj.test_time_augmentation,
        'best_epoch': method_obj.best_epoch,
        'best_accuracy_seen_during_training': method_obj.best_accuracy,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'training_time_seconds': method_obj.training_time_seconds,
        'total_time_seconds': total_time_seconds,
        'plot_path': str(plot_path),
    }

    print('Summary:', summary)
    return summary


def experiment_definitions(profile):
    # DV: You guys can add MNIST/ORL experiment configs here later.
    allowed_profiles = ('cifar_main', 'cifar_wide', 'cifar_ablation', 'cifar_all', 'orl_all')
    if profile not in allowed_profiles:
        raise ValueError('This CIFAR-only branch supports these profiles: ' + ', '.join(allowed_profiles))

    main_cnn_experiment = {
        'name': 'cifar_residual_bf16',
        'dataset_key': 'cifar',
        'config': {
            'architecture': 'residual',
            'widths': [80, 160, 320],
            'blocks_per_stage': 3,
            'dropout': 0.08,
            'max_epoch': 140,
            'batch_size': 512,
            'optimizer': 'adamw',
            'learning_rate': 1.5e-3,
            'weight_decay': 5e-4,
            'label_smoothing': 0.08,
            'mixup_alpha': 0.15,
            'augment': True,
            'use_bf16_autocast': True,
        },
    }

    wideresnet_ablation = {
        'name': 'cifar_wrn28_8_adamw_ema_tta_bf16',
        'dataset_key': 'cifar',
        'config': {
            'architecture': 'wide_residual',
            'depth': 28,
            'widen_factor': 8,
            'dropout': 0.1,
            'max_epoch': 180,
            'batch_size': 384,
            'optimizer': 'adamw',
            'learning_rate': 0.0013,
            'min_learning_rate': 2e-5,
            'weight_decay': 5e-4,
            'label_smoothing': 0.08,
            'mixup_alpha': 0.1,
            'augment': True,
            'cifar_padding': 4,
            'cutout_size': 12,
            'cutout_probability': 0.5,
            'color_jitter_probability': 0.15,
            'color_jitter_strength': 0.08,
            'use_bf16_autocast': True,
            'use_ema': True,
            'ema_decay': 0.999,
            'ema_start_epoch': 20,
            'test_time_augmentation': True,
            'tta_padding': 4,
        },
    }

    orl_experiment = {
      'name': 'orl_cnn',
      'dataset_key': 'orl',
      'config': {
          'architecture': 'residual',
          'widths': [32, 64, 128],
          'blocks_per_stage': 2,
          'dropout': 0.1,
          'max_epoch': 40,
          'batch_size': 16,
          'optimizer': 'adamw',
          'learning_rate': 1e-3,
          'weight_decay': 1e-4,
          'label_smoothing': 0.0,
          'mixup_alpha': 0.0,
          'augment': False,
          'use_bf16_autocast': False,
          'input_channels': 1,
          'class_count': 40,
      },
    }

    orl_small = {
      'name': 'orl_small',
      'dataset_key': 'orl',
      'config': {
          'architecture': 'residual',
          'widths': [16, 32, 64],
          'blocks_per_stage': 2,
          'dropout': 0.1,
          'max_epoch': 40,
          'batch_size': 16,
          'optimizer': 'adamw',
          'learning_rate': 1e-3,
          'weight_decay': 1e-4,
          'augment': False,
          'input_channels': 1,
          'class_count': 40,
      },
    }

    orl_no_dropout = {
      'name': 'orl_no_dropout',
      'dataset_key': 'orl',
      'config': {
          'architecture': 'residual',
          'widths': [32, 64, 128],
          'blocks_per_stage': 2,
          'dropout': 0.0,
          'max_epoch': 40,
          'batch_size': 16,
          'optimizer': 'adamw',
          'learning_rate': 1e-3,
          'weight_decay': 1e-4,
          'augment': False,
          'input_channels': 1,
          'class_count': 40,
      },
    }
    

    # DV: These epoch counts were for my GPU, so lower them if your laptop is slow.
    if profile == 'cifar_main':
        return [main_cnn_experiment]

    if profile in ('cifar_wide', 'cifar_ablation'):
        return [wideresnet_ablation]
    
    if profile == 'orl_all':
      return [orl_experiment, orl_small, orl_no_dropout]

    if profile == 'cifar_all':
      return [main_cnn_experiment, wideresnet_ablation]


if __name__ == '__main__':
    configure_torch()
    profile = sys.argv[1] if len(sys.argv) > 1 else 'cifar_main'
    seed = 170
    data_dir = dough_dir / 'data' / 'stage_3_data'
    result_dir = dough_dir / 'result' / 'stage_3_result'
    result_dir.mkdir(parents=True, exist_ok=True)

    print('************ Stage 3 CNN Experiments ************')
    print('Profile:', profile)
    print('Torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA device:', torch.cuda.get_device_name(0))
        print('CUDA capability:', torch.cuda.get_device_capability(0))
        print('BF16 supported:', torch.cuda.is_bf16_supported())

    summaries = []
    for experiment in experiment_definitions(profile):
        summaries.append(run_experiment(experiment, data_dir, result_dir, seed))

    summary_path = result_dir / ('experiment_summary_' + profile + '.csv')
    with open(summary_path, 'w', newline='') as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print()
    print('Saved summary:', summary_path)
    print('************ Finish ************')


