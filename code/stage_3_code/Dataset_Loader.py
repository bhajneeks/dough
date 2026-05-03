'''
Starter IO class for the Stage 3 image datasets.
'''

from pathlib import Path
import pickle

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    dataset_key = None

    dataset_files = {
        'mnist': 'MNIST',
        'orl': 'ORL',
        'cifar': 'CIFAR',
    }

    def __init__(self, dName=None, dDescription=None, dataset_key=None):
        super().__init__(dName, dDescription)
        self.dataset_key = dataset_key

    def _default_data_folder(self):
        return Path(__file__).resolve().parents[2] / 'data' / 'stage_3_data'

    def _resolve_file_name(self):
        if self.dataset_source_file_name is not None:
            return self.dataset_source_file_name

        if self.dataset_key is None:
            raise ValueError('Set dataset_key to one of: mnist, orl, cifar.')

        normalized_key = self.dataset_key.lower()
        if normalized_key not in self.dataset_files:
            raise ValueError('Unknown Stage 3 dataset_key: ' + str(self.dataset_key))

        return self.dataset_files[normalized_key]

    def load(self):
        print('loading stage 3 data...')

        data_folder = Path(self.dataset_source_folder_path) if self.dataset_source_folder_path else self._default_data_folder()
        data_file = data_folder / self._resolve_file_name()

        with open(data_file, 'rb') as data_handle:
            self.data = pickle.load(data_handle)

        return self.data

    def summarize(self, loaded_data=None):
        loaded_data = loaded_data if loaded_data is not None else self.data
        if loaded_data is None:
            loaded_data = self.load()

        train_instances = loaded_data.get('train', [])
        test_instances = loaded_data.get('test', [])
        first_instance = train_instances[0] if train_instances else None
        first_image = first_instance.get('image') if first_instance else None
        first_label = first_instance.get('label') if first_instance else None

        image_shape = None
        if first_image is not None:
            image_shape = getattr(first_image, 'shape', None)

        return {
            'train_count': len(train_instances),
            'test_count': len(test_instances),
            'first_image_shape': tuple(image_shape) if image_shape is not None else None,
            'first_label': first_label,
        }
