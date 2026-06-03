'''
Stage 5 experiment runner for GCN node classification.

Useful commands:
    python script_gcn.py smoke
    python script_gcn.py full
    python script_gcn.py ablation
    python script_gcn.py all
'''

from pathlib import Path
import argparse
import csv
from datetime import datetime
import pickle
import random
import sys
import time
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


dough_dir = Path(__file__).resolve().parents[2]
sys.modules.pop('code', None)
if str(dough_dir) not in sys.path:
    sys.path.insert(0, str(dough_dir))

from code.stage_5_code.Dataset_Loader import Dataset_Loader
from code.stage_5_code.Method_GCN import Method_GCN


def configure_torch():
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True


def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timestamp_for_name():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def clean_name(value):
    cleaned = ''.join(
        c if c.isalnum() or c in ('-', '_') else '_'
        for c in value
    )
    cleaned = cleaned.strip('_')
    return cleaned or 'run'


def safe_float(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def default_data_dir():
    return dough_dir / 'data' / 'stage_5_data'


def ensure_data(data_dir):
    needed = ['cora', 'citeseer', 'pubmed']
    if all((data_dir / name / 'node').exists() for name in needed):
        return data_dir

    zip_candidates = [
        dough_dir.parent / 'Stage 5 Files' / 'stage_5_data.zip',
        dough_dir / 'data' / 'stage_5_data.zip',
        data_dir.parent / 'stage_5_data.zip',
    ]
    for zip_path in zip_candidates:
        if zip_path.exists():
            print(f'Extracting Stage 5 data from {zip_path}')
            data_dir.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir.parent)
            if all((data_dir / name / 'node').exists() for name in needed):
                return data_dir

    raise FileNotFoundError(
        'Stage 5 data was not found. Expected data/stage_5_data with '
        'cora, citeseer, and pubmed folders.'
    )


def metric_summary(true_y, pred_y):
    return {
        'accuracy': accuracy_score(true_y, pred_y),
        'precision_weighted': precision_score(
            true_y, pred_y, average='weighted', zero_division=0
        ),
        'recall_weighted': recall_score(
            true_y, pred_y, average='weighted', zero_division=0
        ),
        'f1_weighted': f1_score(
            true_y, pred_y, average='weighted', zero_division=0
        ),
        'precision_macro': precision_score(
            true_y, pred_y, average='macro', zero_division=0
        ),
        'recall_macro': recall_score(
            true_y, pred_y, average='macro', zero_division=0
        ),
        'f1_macro': f1_score(
            true_y, pred_y, average='macro', zero_division=0
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_pickle(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def save_learning_curve(result, path, title):
    loss = list(result.get('loss_history', []))
    train_acc = list(result.get('train_accuracy_history', []))
    test_acc = list(result.get('test_accuracy_history', []))
    if not loss:
        return

    epochs = range(1, len(loss) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, loss, label='train loss', color='tab:blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_acc, label='train accuracy', color='tab:green')
    ax2.plot(epochs, test_acc, label='test accuracy', color='tab:orange')
    ax2.set_ylabel('Accuracy')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax1.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


def save_ablation_plot(rows, path):
    ablation_rows = [r for r in rows if r.get('group') == 'ablation']
    if not ablation_rows:
        return

    labels = [r['experiment'] for r in ablation_rows]
    scores = [float(r['accuracy']) for r in ablation_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, scores, color='tab:blue')
    ax.set_ylabel('Test accuracy')
    ax.set_ylim(0, 1)
    ax.set_title('Stage 5 Cora Ablation Accuracy')
    ax.tick_params(axis='x', rotation=35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


def profile_epochs(profile):
    if profile == 'smoke':
        return 3
    if profile == 'quick':
        return 40
    return 200


def baseline_experiments(datasets, epochs):
    rows = []
    for dataset_name in datasets:
        rows.append({
            'group': 'baseline',
            'experiment': dataset_name + '_gcn_base',
            'dataset': dataset_name,
            'max_epoch': epochs,
            'hidden_dim': 16,
            'num_layers': 2,
            'dropout': 0.5,
            'learning_rate': 0.01,
            'weight_decay': 5e-4,
        })
    return rows


def ablation_experiments(dataset_name, epochs):
    base = {
        'dataset': dataset_name,
        'max_epoch': epochs,
        'hidden_dim': 16,
        'num_layers': 2,
        'dropout': 0.5,
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
    }
    variants = [
        ('ab_base', {}),
        ('hidden32', {'hidden_dim': 32}),
        ('dropout02', {'dropout': 0.2}),
        ('dropout07', {'dropout': 0.7}),
        ('one_layer', {'num_layers': 1}),
        ('three_layers', {'num_layers': 3}),
        ('lr005', {'learning_rate': 0.005}),
        ('no_weight_decay', {'weight_decay': 0.0}),
    ]

    rows = []
    for name, changes in variants:
        config = {**base, **changes}
        config['group'] = 'ablation'
        config['experiment'] = dataset_name + '_' + name
        rows.append(config)
    return rows


def run_one_experiment(exp, args, data_dir, result_dir):
    reset_seeds(args.seed)
    dataset_name = exp['dataset']
    print()
    print('============ ' + exp['experiment'] + ' ============')

    loader = Dataset_Loader(
        seed=args.seed,
        dName=dataset_name,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        val_per_class=args.val_per_class,
        normalize_features=not args.no_feature_norm,
    )
    loader.dataset_source_folder_path = str(data_dir)
    data = loader.load()
    data['dataset_name'] = dataset_name

    config = {
        'seed': args.seed,
        'device': args.device,
        'max_epoch': exp['max_epoch'],
        'hidden_dim': exp['hidden_dim'],
        'num_layers': exp['num_layers'],
        'dropout': exp['dropout'],
        'learning_rate': exp['learning_rate'],
        'weight_decay': exp['weight_decay'],
        'print_every': args.print_every,
    }

    method_obj = Method_GCN('gcn', '', config=config)
    method_obj.data = data

    start = time.perf_counter()
    result = method_obj.run()
    total_time = time.perf_counter() - start

    metrics = metric_summary(result['true_y'], result['pred_y'])
    split = data['train_test_val']['split_summary']
    graph = data['graph']

    plot_path = result_dir / (exp['experiment'] + '_curve.png')
    save_learning_curve(result, plot_path, exp['experiment'])

    result_path = result_dir / (exp['experiment'] + '_result.pkl')
    save_pickle(result_path, result)

    summary = {
        'group': exp['group'],
        'experiment': exp['experiment'],
        'dataset': dataset_name,
        'nodes': int(graph['X'].shape[0]),
        'features': int(graph['X'].shape[1]),
        'classes': len(graph['utility']['class_names']),
        'train_nodes': split['total_train'],
        'test_nodes': split['total_test'],
        'val_nodes': split['total_val'],
        'epochs': exp['max_epoch'],
        'best_epoch': result['best_epoch'],
        'hidden_dim': exp['hidden_dim'],
        'num_layers': exp['num_layers'],
        'dropout': exp['dropout'],
        'learning_rate': exp['learning_rate'],
        'weight_decay': exp['weight_decay'],
        'accuracy': safe_float(metrics['accuracy']),
        'precision_weighted': safe_float(metrics['precision_weighted']),
        'recall_weighted': safe_float(metrics['recall_weighted']),
        'f1_weighted': safe_float(metrics['f1_weighted']),
        'precision_macro': safe_float(metrics['precision_macro']),
        'recall_macro': safe_float(metrics['recall_macro']),
        'f1_macro': safe_float(metrics['f1_macro']),
        'final_loss': safe_float(result['loss_history'][-1]),
        'training_time': round(method_obj.training_time_seconds, 1),
        'total_time': round(total_time, 1),
        'device': str(method_obj.device),
        'curve_file': str(plot_path),
        'result_file': str(result_path),
    }

    print('  Summary:', summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Stage 5 GCN node classification experiments.'
    )
    parser.add_argument(
        'profile',
        nargs='?',
        default='all',
        choices=['smoke', 'quick', 'full', 'ablation', 'all'],
    )
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--result-root', default=None)
    parser.add_argument('--seed', type=int, default=170)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--ablation-dataset', default='cora')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--print-every', type=int, default=20)
    parser.add_argument('--only', action='append', default=None)
    parser.add_argument('--train-per-class', type=int, default=None)
    parser.add_argument('--test-per-class', type=int, default=None)
    parser.add_argument('--val-per-class', type=int, default=0)
    parser.add_argument('--no-feature-norm', action='store_true')
    return parser.parse_args()


def choose_experiments(args):
    epochs = args.epochs or profile_epochs(args.profile)
    datasets = args.datasets
    if datasets is None:
        datasets = ['cora'] if args.profile in ('smoke', 'quick') else [
            'cora',
            'citeseer',
            'pubmed',
        ]

    experiments = []
    if args.profile in ('smoke', 'quick', 'full', 'all'):
        experiments.extend(baseline_experiments(datasets, epochs))
    if args.profile in ('ablation', 'all'):
        experiments.extend(ablation_experiments(args.ablation_dataset, epochs))

    if args.only:
        wanted = set(args.only)
        experiments = [exp for exp in experiments if exp['experiment'] in wanted]

    return experiments


def main():
    configure_torch()
    args = parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    data_dir = ensure_data(data_dir)

    result_root = (
        Path(args.result_root)
        if args.result_root
        else dough_dir / 'result' / 'stage_5_result'
    )
    run_name = args.run_name or (args.profile + '_' + timestamp_for_name())
    result_dir = result_root / clean_name(run_name)
    result_dir.mkdir(parents=True, exist_ok=True)

    experiments = choose_experiments(args)

    print('************ Stage 5 GCN Experiments ************')
    print('Profile:', args.profile)
    print('Data dir:', data_dir)
    print('Result dir:', result_dir)
    print('Python:', sys.executable)
    print('Torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA device:', torch.cuda.get_device_name(0))
    print('Device used:', args.device)
    print('Experiment count:', len(experiments))

    rows = []
    for exp in experiments:
        rows.append(run_one_experiment(exp, args, data_dir, result_dir))

    summary_path = result_dir / ('summary_' + args.profile + '.csv')
    write_csv(summary_path, rows)
    save_ablation_plot(rows, result_dir / 'ablation_accuracy.png')

    print()
    print('Saved summary:', summary_path)
    print('************ Stage 5 Complete ************')


if __name__ == '__main__':
    main()
