from pathlib import Path
import sys

import torch


# Current-device default: resolve the repo from this script location.
dough_dir = Path(__file__).resolve().parents[2]

# Davis apartment PC absolute path reference.
# Keep this commented on other devices. Uncomment only when back on that PC
# if automatic path resolution is not enough for the local setup there.
# dough_dir = Path(r'C:\Users\ocaro\OneDrive - University of California, Davis\Spring 2026\ECS 170 - Artificial Intelligence\Project\dough')

if str(dough_dir) not in sys.path:
    sys.path.insert(0, str(dough_dir))

from code.stage_3_code.Dataset_Loader import Dataset_Loader


def print_environment_summary():
    print('Python executable:', sys.executable)
    print('Torch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA version:', torch.version.cuda)
        print('CUDA device:', torch.cuda.get_device_name(0))
        print('CUDA capability:', torch.cuda.get_device_capability(0))


def print_dataset_summary(dataset_key, data_dir):
    data_obj = Dataset_Loader('stage 3 ' + dataset_key, '', dataset_key=dataset_key)
    data_obj.dataset_source_folder_path = str(data_dir)
    loaded_data = data_obj.load()
    summary = data_obj.summarize(loaded_data)

    print()
    print('Dataset:', dataset_key.upper())
    print('Data folder:', data_dir)
    print('Train instances:', summary['train_count'])
    print('Test instances:', summary['test_count'])
    print('First image shape:', summary['first_image_shape'])
    print('First label:', summary['first_label'])


if __name__ == '__main__':
    data_dir = dough_dir / 'data' / 'stage_3_data'
    result_dir = dough_dir / 'result' / 'stage_3_result'
    result_dir.mkdir(parents=True, exist_ok=True)

    print('************ Stage 3 Starter Sanity Check ************')
    print_environment_summary()

    for dataset_key in ['mnist', 'orl', 'cifar']:
        print_dataset_summary(dataset_key, data_dir)

    print()
    print('No CNN training is run by this starter script.')
    print('Next implementation target: fill in code.stage_3_code.Method_CNN.')
    print('************ Finish ************')
