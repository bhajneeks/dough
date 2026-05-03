'''
Starter MethodModule for Stage 3 CNN work.

This file intentionally does not implement the CNN yet. It only reserves the
project-template surface that the Stage 3 implementation should fill in.
'''

from code.base_class.method import method
import torch
from torch import nn


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 128

    def __init__(self, mName, mDescription, config=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.config = config or {}
        self.max_epoch = self.config.get('max_epoch', self.max_epoch)
        self.learning_rate = self.config.get('learning_rate', self.learning_rate)
        self.batch_size = self.config.get('batch_size', self.batch_size)
        self.dataset_key = self.config.get('dataset_key')
        self.loss_history = []
        self.training_time_seconds = 0.0

        configured_device = self.config.get('device')
        if configured_device is None:
            configured_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if configured_device == 'cuda' and not torch.cuda.is_available():
            configured_device = 'cpu'
        self.device = torch.device(configured_device)

        # TODO Stage 3:
        # Build dataset-specific CNN layers here after inspecting image shapes:
        # MNIST: 1 x 28 x 28, ORL: 1 x 112 x 92, CIFAR: 3 x 32 x 32.
        self.network = None

    def forward(self, x):
        raise NotImplementedError('Stage 3 CNN forward pass has not been implemented yet.')

    def fit(self, train_instances):
        raise NotImplementedError('Stage 3 CNN training loop has not been implemented yet.')

    def test(self, test_instances):
        raise NotImplementedError('Stage 3 CNN test loop has not been implemented yet.')

    def run(self):
        print('method running...')
        raise NotImplementedError('Stage 3 CNN run() has not been implemented yet.')
