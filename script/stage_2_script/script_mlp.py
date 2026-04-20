from pathlib import Path
import sys


dough_dir = Path(__file__).resolve().parents[2]

if str(dough_dir) not in sys.path:
    sys.path.insert(0, str(dough_dir))

from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch


#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    #---- path section ------------------------------------
    data_dir = dough_dir / 'data' / 'stage_2_data'
    result_dir = dough_dir / 'result' / 'stage_2_result'
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('stage 2 dataset', '')
    data_obj.dataset_source_folder_path = str(data_dir)
    data_obj.train_file_name = 'train.csv'
    data_obj.test_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir / 'MLP_')
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- quick loading check section ---------------------
    print('Checking if data can be loaded successfully...')
    print()

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