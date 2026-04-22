'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from contextlib import nullcontext
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 256
    input_feature_count = 784
    class_count = 10

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, config=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.config = config or {}
        self.max_epoch = self.config.get('max_epoch', self.max_epoch)
        self.learning_rate = self.config.get('learning_rate', self.learning_rate)
        self.batch_size = self.config.get('batch_size', self.batch_size)
        self.hidden_dims = self.config.get('hidden_dims', [256, 128])
        self.dropout_rate = self.config.get('dropout_rate', 0.0)
        self.use_bf16_autocast = bool(self.config.get('use_bf16_autocast', False))
        configured_device = self.config.get('device')
        if configured_device is None:
            configured_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if configured_device == 'cuda' and not torch.cuda.is_available():
            configured_device = 'cpu'
        self.device = torch.device(configured_device)
        self.loss_history = []
        self.training_time_seconds = 0.0

        layers = []
        current_input_size = self.input_feature_count
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(current_input_size, hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            current_input_size = hidden_dim
        layers.append(nn.Linear(current_input_size, self.class_count))
        self.network = nn.Sequential(*layers)
        self.to(self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        '''Forward propagation'''
        # CrossEntropyLoss expects raw logits instead of softmax probabilities.
        logits = self.network(x)
        return logits

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here
    def fit(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        train_tensor_x = torch.tensor(np.array(X), dtype=torch.float32) / 255.0
        train_tensor_y = torch.tensor(np.array(y), dtype=torch.long)
        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(self.device.type == 'cuda')
        )
        self.loss_history = []
        self.training_time_seconds = 0.0
        nn.Module.train(self, True)
        training_start_time = time.perf_counter()

        # it will be an iterative gradient updating process
        # mini-batch training is much faster and more stable on this dataset size.
        for epoch in range(self.max_epoch):
            total_loss = 0.0
            total_correct = 0
            total_examples = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=(self.device.type == 'cuda'))
                batch_y = batch_y.to(self.device, non_blocking=(self.device.type == 'cuda'))

                autocast_context = nullcontext()
                if self.use_bf16_autocast and self.device.type == 'cuda':
                    autocast_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)

                with autocast_context:
                    logits = self.forward(batch_x)
                    train_loss = loss_function(logits, batch_y)

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad(set_to_none=True)
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

                batch_size = batch_x.size(0)
                total_loss += train_loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total_examples += batch_size

            average_loss = total_loss / total_examples
            accuracy = total_correct / total_examples
            self.loss_history.append(average_loss)

            if epoch % 5 == 0 or epoch == self.max_epoch - 1:
                print('Epoch:', epoch, 'Accuracy:', accuracy, 'Loss:', average_loss)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.training_time_seconds = time.perf_counter() - training_start_time

    def test(self, X):
        # do the testing, and result the result
        nn.Module.train(self, False)
        test_tensor_x = torch.tensor(np.array(X), dtype=torch.float32, device=self.device) / 255.0
        with torch.no_grad():
            y_pred = self.forward(test_tensor_x)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.argmax(dim=1).cpu().numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': np.array(self.data['test']['y'])}