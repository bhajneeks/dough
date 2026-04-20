'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from pathlib import Path

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    train_file_name = None
    test_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load_one_file(self, file_path):
        X = []
        y = []
        f = open(file_path, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X.append(elements[1:])
            y.append(elements[0])
        f.close()
        return {'X': X, 'y': y}

    def load(self):
        print('loading data...')

        if self.dataset_source_folder_path is None:
            self.dataset_source_folder_path = str(
                Path(__file__).resolve().parents[2] / 'data' / 'stage_2_data'
            )

        train_file_path = Path(self.dataset_source_folder_path) / self.train_file_name
        test_file_path = Path(self.dataset_source_folder_path) / self.test_file_name

        train_data = self.load_one_file(train_file_path)
        test_data = self.load_one_file(test_file_path)
        return {'train': train_data, 'test': test_data}