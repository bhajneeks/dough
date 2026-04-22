from pathlib import Path
import csv
import sys
import time


dough_dir = Path(__file__).resolve().parents[2]

if str(dough_dir) not in sys.path:
    sys.path.insert(0, str(dough_dir))

from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_2_code.Evaluate_Precision import Evaluate_Precision
from code.stage_2_code.Evaluate_Recall import Evaluate_Recall
from code.stage_2_code.Evaluate_F1 import Evaluate_F1
import matplotlib.pyplot as plt
import numpy as np
import torch


def reset_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(experiment_name, experiment_config, data_dir, result_dir, seed):
    print()
    print('============ Experiment:', experiment_name, '============')
    reset_random_seeds(seed)

    data_obj = Dataset_Loader('stage 2 dataset', '')
    data_obj.dataset_source_folder_path = str(data_dir)
    data_obj.train_file_name = 'train.csv'
    data_obj.test_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '', config=experiment_config)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir / f'{experiment_name}_')
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    experiment_start_time = time.perf_counter()
    accuracy_score, _ = setting_obj.load_run_save_evaluate()
    total_time_seconds = time.perf_counter() - experiment_start_time
    prediction_result = result_obj.data

    precision_obj = Evaluate_Precision('precision', '')
    precision_obj.data = prediction_result
    precision_score = precision_obj.evaluate()

    recall_obj = Evaluate_Recall('recall', '')
    recall_obj.data = prediction_result
    recall_score = recall_obj.evaluate()

    f1_obj = Evaluate_F1('f1', '')
    f1_obj.data = prediction_result
    f1_score = f1_obj.evaluate()

    saved_result_path = (
        result_obj.result_destination_folder_path
        + result_obj.result_destination_file_name
        + '_1'
    )

    plot_path = result_dir / f'{experiment_name}_training_convergence.png'
    if method_obj.loss_history:
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, len(method_obj.loss_history) + 1),
            method_obj.loss_history,
            linewidth=2
        )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{experiment_name} Training Convergence')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

    print('Device:', method_obj.device)
    print('Accuracy:', accuracy_score)
    print('Precision:', precision_score)
    print('Recall:', recall_score)
    print('F1:', f1_score)
    print('Training time (s):', method_obj.training_time_seconds)
    print('Total experiment time (s):', total_time_seconds)
    print('Saved predictions:', saved_result_path)
    print('Saved convergence plot:', plot_path)

    return {
        'experiment_name': experiment_name,
        'device': str(method_obj.device),
        'use_bf16_autocast': method_obj.use_bf16_autocast,
        'hidden_dims': '-'.join(str(hidden_dim) for hidden_dim in method_obj.hidden_dims),
        'dropout_rate': method_obj.dropout_rate,
        'batch_size': method_obj.batch_size,
        'max_epoch': method_obj.max_epoch,
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'training_time_seconds': method_obj.training_time_seconds,
        'total_time_seconds': total_time_seconds,
        'prediction_path': saved_result_path,
        'plot_path': str(plot_path),
    }


# Stage 2 script to do list
# 1.) Read the training data and testing data from the csv files.
# 2.) Hook up all the pieces so they know about each other (data, model, saver, evaluator).
# 3.) Train the MLP so it learns patterns from the training data.
# 4.) Use the trained MLP to make predictions on the testing data.
# 5.) Save those predictions to a file so we can look at them later.
# 6.) Compare the predictions to the real answers using accuracy, precision, recall, and F1.
# 7.) Print the final scores so we can actually read them.
# 8.) Make a training convergence plot (epoch on x axis, loss on y axis) for the report.
# 9.) Later, try different MLP setups (more layers, different learning rate, etc) and see if the scores improve.

# Simple metric notes

# Accuracy = correct predictions / total predictions.
# This is the overall percent the model got right, but it can hide weak performance on some classes.

# Precision = correct predicted items in a class / total predicted items in that class.
# This tells us how trustworthy the model's predictions are for a class, but it can still miss many true cases.

# Recall = correct predicted items in a class / total true items in that class.
# This tells us how many real cases the model caught, but it can drop when the model is too careful and predicts less.

# F1 = 2 * precision * recall / (precision + recall).
# This gives one balanced score for precision and recall, but it can hide which of the two is causing the problem.


#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    experiment_seed = 2
    reset_random_seeds(experiment_seed)
    #------------------------------------------------------

    #---- path section ------------------------------------
    data_dir = dough_dir / 'data' / 'stage_2_data'
    result_dir = dough_dir / 'result' / 'stage_2_result'
    result_dir.mkdir(parents=True, exist_ok=True)
    #------------------------------------------------------

    # 1.) Check that the data files can be loaded.
    # This is just a quick sanity check before the full run.
    # ---- quick loading check section ---------------------
    print('Checking if data can be loaded successfully...')
    print()

    data_obj = Dataset_Loader('stage 2 dataset', '')
    data_obj.dataset_source_folder_path = str(data_dir)
    data_obj.train_file_name = 'train.csv'
    data_obj.test_file_name = 'test.csv'

    print('Start ==================================')
    print('Data folder:', data_obj.dataset_source_folder_path)
    print('Train file:', data_obj.train_file_name)
    print('Test file:', data_obj.test_file_name)

    loaded_data = data_obj.load()


    print('Train instances:', len(loaded_data['train']['X']))
    print('Test instances:', len(loaded_data['test']['X']))
    print('Feature count:', len(loaded_data['train']['X'][0]))
    print('First label:', loaded_data['train']['y'][0])
    print('End ==================================')

    experiment_definitions = [
        {
            'name': 'baseline_cuda',
            'config': {}
        },
        {
            'name': 'ablation_a_wider_dropout',
            'config': {
                'hidden_dims': [512, 256, 128],
                'dropout_rate': 0.2,
            }
        },
        {
            'name': 'ablation_c_bf16_autocast',
            'config': {
                'use_bf16_autocast': True,
            }
        },
        {
            'name': 'baseline_cpu',
            'config': {
                'device': 'cpu',
            }
        },
    ]

    if not torch.cuda.is_available():
        experiment_definitions = [
            experiment_definition
            for experiment_definition in experiment_definitions
            if experiment_definition['name'] != 'ablation_c_bf16_autocast'
        ]

    experiment_summaries = []
    print('************ Start ************')
    for experiment_definition in experiment_definitions:
        experiment_summary = run_experiment(
            experiment_definition['name'],
            experiment_definition['config'],
            data_dir,
            result_dir,
            experiment_seed,
        )
        experiment_summaries.append(experiment_summary)

    summary_csv_path = result_dir / 'experiment_summary.csv'
    with open(summary_csv_path, 'w', newline='') as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(experiment_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(experiment_summaries)

    print()
    print('************ Experiment Summary ************')
    for experiment_summary in experiment_summaries:
        print(
            experiment_summary['experiment_name'],
            '| device =', experiment_summary['device'],
            '| accuracy =', experiment_summary['accuracy'],
            '| f1 =', experiment_summary['f1'],
            '| training time (s) =', experiment_summary['training_time_seconds']
        )
    print('Saved experiment summary:', summary_csv_path)
    print('************ Finish ************')