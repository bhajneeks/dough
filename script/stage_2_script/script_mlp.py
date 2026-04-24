from pathlib import Path
import sys
import matplotlib.pyplot as plt


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

    # 1.) Check that the data files can be loaded.
    # This is just a quick sanity check before the full run.
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

    

    # 2.) Hook everything together.
    # The setting object is basically a manager.
    # It needs to know about the data, the model, the saver, and the evaluator.
    # Call (Uncomment): setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # prepare() is defined in code/base_class/setting.py

    # 3.) Tell the manager to do the full run.
    # This one call trains the model, tests it, saves the predictions, and computes accuracy.
    # Call (Uncomment): setting_obj.load_run_save_evaluate()
    # load_run_save_evaluate() is defined in code/stage_2_code/Setting_Train_Test_Split.py
    # Inside that function it calls method_obj.run() which is defined in code/stage_2_code/Method_MLP.py
    # It gives back the accuracy score (and a second value we can ignore for now).

    # 4.) The save step happens automatically inside the run above.
    # The saver writes the predictions and the true answers into the result folder.
    # save() is defined in code/stage_2_code/Result_Saver.py
    # We do not need to call save ourselves here.

    # 5.) Compute the other metrics (precision, recall, F1).
    # The accuracy was already computed in step 3.
    # For the other three, take the same prediction result and pass it into our other evaluators.
    # Evaluate_Precision is in code/stage_2_code/Evaluate_Precision.py
    # Evaluate_Recall is in code/stage_2_code/Evaluate_Recall.py
    # Evaluate_F1 is in code/stage_2_code/Evaluate_F1.py
    # Each one has an evaluate() method that returns one number.

    # 6.) Print the final scores.
    # Just print accuracy, precision, recall, and F1 on plain lines.
    # Optionally print where the result file was saved.

    # 7.) After the basic run works, try changing the MLP.
    # All the layer sizes, learning rate, and optimizer are set in code/stage_2_code/Method_MLP.py
    # Change those values there, re-run this script, and compare the new scores to the first run.

    # 8.) Make the training convergence plot for the report.
    # The report wants epoch on the x axis and loss on the y axis.
    # First, Method_MLP (code/stage_2_code/Method_MLP.py) needs to keep a list of loss values during training.
    # Then here in the script, take that list and plot it with matplotlib.
    # Save the plot as a png in the result folder so we can drop it into the report.
   
    """
    loss = method_obj.loss_history
    epochs = range(1, len(loss)+1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, loss, label='Training Loss', color='steelblue')

    plt.title('Model Training Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('convergence_plot.png')
    plt.show()
    """