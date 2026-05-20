'''
Stage 4 experiment runner - text classification and generation with RNN/LSTM/GRU.

Useful commands:
    python script_rnn.py smoke
    python script_rnn.py quick
    python script_rnn.py classification
    python script_rnn.py generation
    python script_rnn.py all
    python script_rnn.py ablation_classification
    python script_rnn.py ablation_generation
'''

from pathlib import Path
import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# path setup (same approach as stage 3)
dough_dir = Path(__file__).resolve().parents[2]
if str(dough_dir) not in sys.path:
    sys.path.insert(0, str(dough_dir))

from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Evaluate_Precision import Evaluate_Precision
from code.stage_4_code.Evaluate_Recall import Evaluate_Recall
from code.stage_4_code.Evaluate_F1 import Evaluate_F1


# -------------------------------------------------
# helpers
# -------------------------------------------------

def configure_torch():
    '''Set up CUDA optimizations.'''
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')


def reset_seeds(seed=170):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def clean_run_name(value):
    cleaned = ''.join(
        c if c.isalnum() or c in ('-', '_') else '_'
        for c in value
    )
    cleaned = cleaned.strip('_')
    return cleaned or 'run'


def timestamp_for_name():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def timestamp_for_log():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def evaluate_all_metrics(result_data):
    '''Run all four metrics on the classification result.'''
    evaluators = [
        ('accuracy', Evaluate_Accuracy('accuracy', '')),
        ('precision', Evaluate_Precision('precision', '')),
        ('recall', Evaluate_Recall('recall', '')),
        ('f1', Evaluate_F1('f1', '')),
    ]
    scores = {}
    for name, evaluator in evaluators:
        evaluator.data = result_data
        scores[name] = evaluator.evaluate()
    return scores


def save_learning_curve(loss_history, plot_path, title,
                        train_acc_history=None, test_acc_history=None):
    '''Plot training loss and, when available, train/test accuracy.'''
    if not loss_history:
        return

    epochs = range(1, len(loss_history) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epochs, loss_history, linewidth=2, label='train loss',
             color='tab:blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    has_train_acc = train_acc_history is not None and len(train_acc_history) > 0
    has_test_acc = test_acc_history is not None and len(test_acc_history) > 0
    if has_train_acc or has_test_acc:
        ax2 = ax1.twinx()
        if has_train_acc:
            ax2.plot(epochs, train_acc_history, linewidth=1.5,
                     label='train accuracy', color='tab:green')
        if has_test_acc:
            ax2.plot(epochs, test_acc_history, linewidth=1.5,
                     label='test accuracy', color='tab:orange')
        ax2.set_ylabel('Accuracy')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    else:
        ax1.legend()

    plt.title(title)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)


def write_csv(csv_path, rows):
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_log_header(log_path):
    if log_path.exists():
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('# Stage 4 Experiment Log\n\n')
        f.write('Persistent record for RNN/LSTM/GRU classification and ')
        f.write('generation runs. Automatic entries are useful raw notes; ')
        f.write('human observations are added as results are reviewed.\n')


def append_run_header(log_path, run_id, profile, command, data_limits, result_dir):
    ensure_log_header(log_path)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n\n## Run Batch: ' + run_id + '\n\n')
        f.write('- Timestamp: ' + timestamp_for_log() + '\n')
        f.write('- Profile: `' + profile + '`\n')
        f.write('- Command: `' + command + '`\n')
        f.write('- Result directory: `' + str(result_dir) + '`\n')
        f.write('- Data limits: `' + json.dumps(data_limits, sort_keys=True) + '`\n')


def append_experiment_log(log_path, title, command, config, runtime,
                          metrics, output_files, observations, next_step):
    ensure_log_header(log_path)
    config_text = json.dumps(config, indent=2, sort_keys=True, default=str)

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n### ' + title + '\n\n')
        f.write('- Timestamp: ' + timestamp_for_log() + '\n')
        f.write('- Command: `' + command + '`\n')
        f.write('- Runtime: ' + str(round(runtime, 1)) + ' seconds\n')
        f.write('- Config:\n\n')
        f.write('```json\n' + config_text + '\n```\n\n')
        f.write('- Metrics:\n')
        for key, value in metrics.items():
            f.write('  - ' + str(key) + ': ' + str(value) + '\n')
        f.write('- Output files:\n')
        for path in output_files:
            f.write('  - `' + str(path) + '`\n')
        f.write('- Observations: ' + observations + '\n')
        f.write('- Next: ' + next_step + '\n')


def apply_data_limits(data_obj, data_limits):
    data_obj.max_train_per_class = data_limits.get('max_train_per_class')
    data_obj.max_test_per_class = data_limits.get('max_test_per_class')
    data_obj.max_jokes = data_limits.get('max_jokes')


def default_data_limits(args, profile):
    limits = {
        'max_train_per_class': args.max_train_per_class,
        'max_test_per_class': args.max_test_per_class,
        'max_jokes': args.max_jokes,
    }

    # These defaults keep exploratory runs quick without affecting full runs.
    if profile == 'smoke':
        limits['max_train_per_class'] = limits['max_train_per_class'] or 300
        limits['max_test_per_class'] = limits['max_test_per_class'] or 100
        limits['max_jokes'] = limits['max_jokes'] or 300
    elif profile == 'quick':
        limits['max_train_per_class'] = limits['max_train_per_class'] or 1500
        limits['max_test_per_class'] = limits['max_test_per_class'] or 500
        limits['max_jokes'] = limits['max_jokes'] or 800
    elif profile == 'classification_pilot':
        limits['max_train_per_class'] = limits['max_train_per_class'] or 2500
        limits['max_test_per_class'] = limits['max_test_per_class'] or 1000
    elif profile == 'generation_pilot':
        limits['max_jokes'] = limits['max_jokes'] or 1200
    elif profile == 'ablation_classification':
        limits['max_train_per_class'] = limits['max_train_per_class'] or 4000
        limits['max_test_per_class'] = limits['max_test_per_class'] or 1500
    elif profile == 'ablation_generation':
        limits['max_jokes'] = limits['max_jokes'] or 1200

    return limits


# -------------------------------------------------
# classification experiments
# -------------------------------------------------

def classification_experiments(profile):
    '''Define classification experiment configs for different cell types.'''

    base = {
        'max_epoch': 10,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'embedding_dim': 128,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True,
        'max_seq_length': 300,
    }

    lstm_exp = {'name': 'cls_lstm', 'cell_type': 'lstm', **base}
    gru_exp = {'name': 'cls_gru', 'cell_type': 'gru', **base}
    rnn_exp = {'name': 'cls_rnn', 'cell_type': 'rnn', **base}

    if profile == 'smoke':
        smoke = {
            'max_epoch': 1,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'embedding_dim': 64,
            'hidden_dim': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'bidirectional': False,
            'max_seq_length': 120,
        }
        return [{'name': 'cls_lstm_smoke', 'cell_type': 'lstm', **smoke}]

    if profile == 'quick':
        quick = {**lstm_exp, 'name': 'cls_lstm_quick',
                 'max_epoch': 3, 'max_seq_length': 200}
        return [quick]

    if profile == 'classification_pilot':
        pilot = {**base, 'max_epoch': 4, 'max_seq_length': 200}
        return [
            {'name': 'cls_lstm_pilot', 'cell_type': 'lstm', **pilot},
            {'name': 'cls_gru_pilot', 'cell_type': 'gru', **pilot},
            {'name': 'cls_rnn_pilot', 'cell_type': 'rnn', **pilot},
        ]

    if profile == 'ablation_classification':
        ab_base = {**base, 'max_epoch': 5, 'max_seq_length': 250}
        return [
            {'name': 'cls_gru_ab_base', 'cell_type': 'gru', **ab_base},
            {'name': 'cls_gru_hidden256', 'cell_type': 'gru',
             **{**ab_base, 'hidden_dim': 256}},
            {'name': 'cls_gru_embed64', 'cell_type': 'gru',
             **{**ab_base, 'embedding_dim': 64}},
            {'name': 'cls_gru_dropout01', 'cell_type': 'gru',
             **{**ab_base, 'dropout': 0.1}},
            {'name': 'cls_gru_no_bidir', 'cell_type': 'gru',
             **{**ab_base, 'bidirectional': False}},
            {'name': 'cls_gru_lr0005', 'cell_type': 'gru',
             **{**ab_base, 'learning_rate': 5e-4}},
        ]

    if profile == 'lstm_only':
        return [lstm_exp]

    return [lstm_exp, gru_exp, rnn_exp]


def run_classification(experiment, data_dir, result_dir, seed,
                       data_limits, log_path, command):
    '''Run one classification experiment.'''
    name = experiment['name']
    cell_type = experiment['cell_type']

    print()
    print(f'============ Classification: {name} ============')
    reset_seeds(seed)

    data_obj = Dataset_Loader('imdb', '', task='classification')
    data_obj.dataset_source_folder_path = str(data_dir)
    apply_data_limits(data_obj, data_limits)

    config = {k: v for k, v in experiment.items() if k not in ('name',)}
    method_obj = Method_RNN_Classification('rnn classifier', '', config=config)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir / (name + '_'))
    result_obj.result_destination_file_name = 'prediction_result'

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    setting_obj = Setting_Train_Test_Split('provided split', '')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

    start = time.perf_counter()
    accuracy, learned_result = setting_obj.load_run_save_evaluate()
    total_time = time.perf_counter() - start

    metrics = evaluate_all_metrics(learned_result)
    metrics['accuracy'] = accuracy

    plot_path = result_dir / (name + '_training_convergence.png')
    save_learning_curve(
        method_obj.loss_history,
        plot_path,
        f'{name} Training Convergence',
        train_acc_history=method_obj.train_accuracy_history,
        test_acc_history=method_obj.test_accuracy_history,
    )

    result_path = result_dir / (name + '_prediction_result_1')
    summary = {
        'experiment': name,
        'cell_type': cell_type,
        'epochs': method_obj.max_epoch,
        'best_epoch': method_obj.best_epoch,
        'accuracy': safe_float(metrics['accuracy']),
        'precision': safe_float(metrics['precision']),
        'recall': safe_float(metrics['recall']),
        'f1': safe_float(metrics['f1']),
        'best_accuracy': safe_float(method_obj.best_accuracy),
        'training_time': round(method_obj.training_time_seconds, 1),
        'total_time': round(total_time, 1),
        'embedding_dim': method_obj.embedding_dim,
        'hidden_dim': method_obj.hidden_dim,
        'num_layers': method_obj.num_layers,
        'dropout': method_obj.dropout,
        'bidirectional': method_obj.bidirectional,
        'max_seq_length': method_obj.max_seq_length,
        'learning_rate': method_obj.learning_rate,
        'max_train_per_class': data_limits.get('max_train_per_class'),
        'max_test_per_class': data_limits.get('max_test_per_class'),
    }

    log_config = {**experiment, **data_limits, 'seed': seed}
    append_experiment_log(
        log_path,
        'Classification: ' + name,
        command,
        log_config,
        total_time,
        summary,
        [result_path, plot_path],
        'Automatic run entry. Learning curve and prediction pickle saved.',
        'Compare metrics and curve shape against the other recurrent cells.',
    )

    print(f'  Results: {summary}')
    return summary


# -------------------------------------------------
# generation experiments
# -------------------------------------------------

def generation_experiments(profile):
    '''Define generation experiment configs.'''

    base = {
        'max_epoch': 50,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'max_seq_length': 50,
        'generation_length': 30,
        'temperature': 0.8,
        'sample_temperatures': [0.6, 0.8, 1.0, 1.2],
    }

    lstm_exp = {'name': 'gen_lstm', 'cell_type': 'lstm', **base}
    gru_exp = {'name': 'gen_gru', 'cell_type': 'gru', **base}
    rnn_exp = {'name': 'gen_rnn', 'cell_type': 'rnn', **base}

    if profile == 'smoke':
        smoke = {
            'max_epoch': 2,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'embedding_dim': 64,
            'hidden_dim': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'max_seq_length': 30,
            'generation_length': 20,
            'temperature': 0.8,
            'sample_temperatures': [0.8],
        }
        return [{'name': 'gen_lstm_smoke', 'cell_type': 'lstm', **smoke}]

    if profile == 'quick':
        quick = {**lstm_exp, 'name': 'gen_lstm_quick',
                 'max_epoch': 10, 'max_seq_length': 40,
                 'generation_length': 25}
        return [quick]

    if profile == 'generation_pilot':
        pilot = {**base, 'max_epoch': 20, 'generation_length': 30}
        return [
            {'name': 'gen_lstm_pilot', 'cell_type': 'lstm', **pilot},
            {'name': 'gen_gru_pilot', 'cell_type': 'gru', **pilot},
            {'name': 'gen_rnn_pilot', 'cell_type': 'rnn', **pilot},
        ]

    if profile == 'ablation_generation':
        ab_base = {**base, 'max_epoch': 25}
        return [
            {'name': 'gen_gru_ab_base', 'cell_type': 'gru', **ab_base},
            {'name': 'gen_gru_hidden128', 'cell_type': 'gru',
             **{**ab_base, 'hidden_dim': 128}},
            {'name': 'gen_gru_embed64', 'cell_type': 'gru',
             **{**ab_base, 'embedding_dim': 64}},
            {'name': 'gen_gru_seq30', 'cell_type': 'gru',
             **{**ab_base, 'max_seq_length': 30}},
            {'name': 'gen_gru_len60', 'cell_type': 'gru',
             **{**ab_base, 'generation_length': 60}},
        ]

    if profile == 'lstm_only':
        return [lstm_exp]

    return [lstm_exp, gru_exp, rnn_exp]


def run_generation(experiment, data_dir, result_dir, seed,
                   data_limits, log_path, command):
    '''Run one generation experiment.'''
    name = experiment['name']
    cell_type = experiment['cell_type']

    print()
    print(f'============ Generation: {name} ============')
    reset_seeds(seed)

    data_obj = Dataset_Loader('jokes', '', task='generation')
    data_obj.dataset_source_folder_path = str(data_dir)
    apply_data_limits(data_obj, data_limits)

    config = {k: v for k, v in experiment.items() if k not in ('name',)}
    method_obj = Method_RNN_Generation('rnn generator', '', config=config)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir / (name + '_'))
    result_obj.result_destination_file_name = 'generation_result'

    setting_obj = Setting_Train_Test_Split('generation', '')
    setting_obj.prepare(data_obj, method_obj, result_obj, None)

    start = time.perf_counter()
    _, learned_result = setting_obj.load_run_save_evaluate()
    total_time = time.perf_counter() - start

    plot_path = result_dir / (name + '_training_convergence.png')
    save_learning_curve(
        method_obj.loss_history,
        plot_path,
        f'{name} Training Convergence',
    )

    result_path = result_dir / (name + '_generation_result_1')
    final_loss = (
        method_obj.loss_history[-1]
        if method_obj.loss_history
        else None
    )
    summary = {
        'experiment': name,
        'cell_type': cell_type,
        'epochs': method_obj.max_epoch,
        'final_loss': safe_float(final_loss),
        'training_time': round(method_obj.training_time_seconds, 1),
        'total_time': round(total_time, 1),
        'embedding_dim': method_obj.embedding_dim,
        'hidden_dim': method_obj.hidden_dim,
        'num_layers': method_obj.num_layers,
        'dropout': method_obj.dropout,
        'max_seq_length': method_obj.max_seq_length,
        'learning_rate': method_obj.learning_rate,
        'generation_length': method_obj.generation_length,
        'temperature': method_obj.temperature,
        'sample_temperatures': ';'.join(str(t) for t in method_obj.sample_temperatures),
        'max_jokes': data_limits.get('max_jokes'),
        'sample_generations': learned_result.get('generated_texts', []),
    }

    log_config = {**experiment, **data_limits, 'seed': seed}
    log_metrics = {k: v for k, v in summary.items() if k != 'sample_generations'}
    append_experiment_log(
        log_path,
        'Generation: ' + name,
        command,
        log_config,
        total_time,
        log_metrics,
        [result_path, plot_path],
        'Automatic run entry. Generated text samples saved in the batch file.',
        'Review repetition, coherence, and whether samples resemble short jokes.',
    )

    print(f'  Results: {summary}')
    return summary


# -------------------------------------------------
# main
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Stage 4 RNN/LSTM/GRU experiments.'
    )
    parser.add_argument(
        'profile',
        nargs='?',
        default='all',
        choices=[
            'all',
            'classification',
            'generation',
            'quick',
            'smoke',
            'lstm_only',
            'classification_pilot',
            'generation_pilot',
            'ablation_classification',
            'ablation_generation',
        ],
        help='Experiment profile to run.',
    )
    parser.add_argument('--run-name', default=None,
                        help='Readable name for this output folder.')
    parser.add_argument('--data-dir', default=None,
                        help='Override the Stage 4 data directory.')
    parser.add_argument('--result-root', default=None,
                        help='Override result/stage_4_result.')
    parser.add_argument('--seed', type=int, default=170)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--max-train-per-class', type=int, default=None)
    parser.add_argument('--max-test-per-class', type=int, default=None)
    parser.add_argument('--max-jokes', type=int, default=None)
    parser.add_argument('--only', action='append', default=None,
                        help='Run only an experiment name. Can be repeated.')
    return parser.parse_args()


def find_data_dir(args):
    if args.data_dir:
        return Path(args.data_dir)

    # First check the zip-style folder some Stage 4 handouts use.
    data_dir = dough_dir.parent / 'stage_4_data' / 'stage_4_data'

    # Fallback: the current project keeps the data inside dough/data.
    if not data_dir.exists():
        alt = dough_dir / 'data' / 'stage_4_data'
        if (alt / 'stage_4_data').exists():
            data_dir = alt / 'stage_4_data'
        elif alt.exists():
            data_dir = alt

    return data_dir


def filter_experiments(experiments, only_names):
    if not only_names:
        return experiments
    wanted = set(only_names)
    return [exp for exp in experiments if exp['name'] in wanted]


def add_device_to_experiments(experiments, device):
    if device is None:
        return experiments
    for exp in experiments:
        exp['device'] = device
    return experiments


def command_for_log():
    parts = [Path(sys.executable).name, str(Path(__file__))]
    parts.extend(sys.argv[1:])
    return ' '.join(parts)


if __name__ == '__main__':
    configure_torch()
    args = parse_args()

    profile = args.profile
    seed = args.seed
    data_dir = find_data_dir(args)

    result_root = (
        Path(args.result_root)
        if args.result_root
        else dough_dir / 'result' / 'stage_4_result'
    )
    result_root.mkdir(parents=True, exist_ok=True)

    run_id = (
        clean_run_name(args.run_name)
        if args.run_name
        else clean_run_name(profile + '_' + timestamp_for_name())
    )
    result_dir = result_root / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    log_path = result_root / 'experiment_log.md'
    command = command_for_log()
    data_limits = default_data_limits(args, profile)

    append_run_header(log_path, run_id, profile, command, data_limits, result_dir)

    print('************ Stage 4 RNN Experiments ************')
    print(f'Profile: {profile}')
    print(f'Run id: {run_id}')
    print(f'Data dir: {data_dir}')
    print(f'Result dir: {result_dir}')
    print(f'Data limits: {data_limits}')
    print(f'Torch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')

    classification_profiles = (
        'all',
        'classification',
        'quick',
        'smoke',
        'lstm_only',
        'classification_pilot',
        'ablation_classification',
    )
    generation_profiles = (
        'all',
        'generation',
        'quick',
        'smoke',
        'lstm_only',
        'generation_pilot',
        'ablation_generation',
    )

    cls_summaries = []
    if profile in classification_profiles:
        cls_experiments = classification_experiments(profile)
        cls_experiments = add_device_to_experiments(cls_experiments, args.device)
        cls_experiments = filter_experiments(cls_experiments, args.only)

        for exp in cls_experiments:
            cls_summaries.append(
                run_classification(exp, data_dir, result_dir, seed,
                                   data_limits, log_path, command)
            )

        if cls_summaries:
            csv_path = result_dir / f'classification_summary_{profile}.csv'
            write_csv(csv_path, cls_summaries)
            print(f'\nSaved classification summary: {csv_path}')

    gen_summaries = []
    if profile in generation_profiles:
        gen_experiments = generation_experiments(profile)
        gen_experiments = add_device_to_experiments(gen_experiments, args.device)
        gen_experiments = filter_experiments(gen_experiments, args.only)

        for exp in gen_experiments:
            gen_summaries.append(
                run_generation(exp, data_dir, result_dir, seed,
                               data_limits, log_path, command)
            )

        if gen_summaries:
            csv_path = result_dir / f'generation_summary_{profile}.csv'
            csv_rows = []
            for summary in gen_summaries:
                row = {
                    k: v
                    for k, v in summary.items()
                    if k != 'sample_generations'
                }
                csv_rows.append(row)
            write_csv(csv_path, csv_rows)
            print(f'Saved generation summary: {csv_path}')

            txt_path = result_dir / f'generated_texts_{profile}.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                for summary in gen_summaries:
                    f.write(f'=== {summary["experiment"]} ({summary["cell_type"]}) ===\n')
                    for text in summary.get('sample_generations', []):
                        f.write(f'  {text}\n')
                    f.write('\n')
            print(f'Saved generated texts: {txt_path}')

    print('\n************ Stage 4 Complete ************')
