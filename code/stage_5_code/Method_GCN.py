'''
Stage 5 GCN method.

The model does node classification on one whole graph at a time.
Only the sampled train node labels are used for the loss.
'''

import copy
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from code.base_class.method import method


class GraphConvolution(nn.Module):
    '''One graph convolution layer: A X W.'''

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        x = self.linear(x)
        if adj.is_sparse:
            return torch.sparse.mm(adj, x)
        return torch.mm(adj, x)


class GCNNetwork(nn.Module):
    '''Small GCN with configurable depth.'''

    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(GraphConvolution(dims[i], dims[i + 1]))
        self.dropout = dropout

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, adj)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x


class Method_GCN(method, nn.Module):
    '''Project method wrapper for GCN node classification.'''

    data = None

    def __init__(self, mName='gcn', mDescription='', config=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.config = config or {}
        self.max_epoch = self.config.get('max_epoch', 200)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.weight_decay = self.config.get('weight_decay', 5e-4)
        self.hidden_dim = self.config.get('hidden_dim', 16)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.5)
        self.print_every = self.config.get('print_every', 20)
        self.seed = self.config.get('seed', 170)

        hidden_dims = self.config.get('hidden_dims')
        if hidden_dims is None:
            hidden_dims = [self.hidden_dim] * max(0, self.num_layers - 1)
        self.hidden_dims = hidden_dims

        device_name = self.config.get('device')
        if device_name is None:
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_name == 'cuda' and not torch.cuda.is_available():
            device_name = 'cpu'
        self.device = torch.device(device_name)

        self.network = None
        self.loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.test_accuracy_history = []
        self.best_epoch = None
        self.best_monitor = None
        self.training_time_seconds = 0.0

    def _build_network(self, input_dim, output_dim):
        self.network = GCNNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout=self.dropout,
        ).to(self.device)

    def forward(self, x, adj):
        return self.network(x, adj)

    def _move_graph_to_device(self):
        graph = self.data['graph']
        splits = self.data['train_test_val']

        x = graph['X'].to(self.device)
        y = graph['y'].to(self.device)
        adj = graph['utility']['A'].coalesce().to(self.device)
        idx_train = splits['idx_train'].to(self.device)
        idx_test = splits['idx_test'].to(self.device)
        idx_val = splits['idx_val'].to(self.device)

        return x, y, adj, idx_train, idx_val, idx_test

    def _accuracy(self, logits, labels, idx):
        if idx.numel() == 0:
            return None
        pred = logits[idx].argmax(dim=1)
        return (pred == labels[idx]).float().mean().item()

    def fit(self):
        '''Train the GCN and keep learning curve values.'''
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        x, y, adj, idx_train, idx_val, idx_test = self._move_graph_to_device()
        input_dim = x.shape[1]
        output_dim = int(y.max().item()) + 1
        self._build_network(input_dim, output_dim)

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        self.loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.test_accuracy_history = []
        self.best_epoch = None
        self.best_monitor = None
        best_state = None
        use_val_checkpoint = idx_val.numel() > 0

        start_time = time.perf_counter()

        for epoch in range(1, self.max_epoch + 1):
            self.network.train()
            optimizer.zero_grad()
            logits = self.network(x, adj)
            loss = loss_fn(logits[idx_train], y[idx_train])
            loss.backward()
            optimizer.step()

            self.network.eval()
            with torch.no_grad():
                logits = self.network(x, adj)
                train_acc = self._accuracy(logits, y, idx_train)
                val_acc = self._accuracy(logits, y, idx_val)
                test_acc = self._accuracy(logits, y, idx_test)

            train_loss = loss.item()
            self.loss_history.append(train_loss)
            self.train_accuracy_history.append(train_acc)
            self.val_accuracy_history.append(val_acc)
            self.test_accuracy_history.append(test_acc)

            # With a val split, keep the best val checkpoint.
            if use_val_checkpoint:
                monitor = val_acc
                if self.best_monitor is None or monitor > self.best_monitor:
                    self.best_monitor = monitor
                    self.best_epoch = epoch
                    best_state = copy.deepcopy(self.network.state_dict())

            should_print = (
                epoch == 1
                or epoch == self.max_epoch
                or epoch % self.print_every == 0
            )
            if should_print:
                val_text = 'none' if val_acc is None else f'{val_acc:.4f}'
                print(
                    f'  Epoch {epoch:03d}/{self.max_epoch} '
                    f'loss={train_loss:.4f} '
                    f'train={train_acc:.4f} '
                    f'val={val_text} '
                    f'test={test_acc:.4f}'
                )

        self.training_time_seconds = time.perf_counter() - start_time
        if use_val_checkpoint and best_state is not None:
            self.network.load_state_dict(best_state)
        if not use_val_checkpoint:
            self.best_epoch = self.max_epoch

        return x, y, adj, idx_test

    def test(self, x, y, adj, idx_test):
        '''Return predictions and true labels for test nodes.'''
        self.network.eval()
        with torch.no_grad():
            logits = self.network(x, adj)
            pred = logits.argmax(dim=1)
        pred_y = pred[idx_test].detach().cpu().numpy()
        true_y = y[idx_test].detach().cpu().numpy()
        return pred_y, true_y

    def run(self, trainData=None, trainLabel=None, testData=None):
        '''Train the model and return the test predictions.'''
        dataset_name = self.data.get('dataset_name', 'graph')
        print(f'[Stage 5] dataset={dataset_name}, device={self.device}')
        print(
            f'  layers={self.num_layers}, hidden={self.hidden_dims}, '
            f'dropout={self.dropout}, lr={self.learning_rate}'
        )

        x, y, adj, idx_test = self.fit()
        pred_y, true_y = self.test(x, y, adj, idx_test)

        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'loss_history': np.array(self.loss_history),
            'train_accuracy_history': np.array(self.train_accuracy_history),
            'val_accuracy_history': np.array(self.val_accuracy_history, dtype=object),
            'test_accuracy_history': np.array(self.test_accuracy_history),
            'best_epoch': self.best_epoch,
            'training_time_seconds': self.training_time_seconds,
            'config': dict(self.config),
        }
