'''
RNN model for text classification (IMDB sentiment).

Supports swapping between RNN, LSTM, and GRU cells via config.
Uses word embeddings -> recurrent layer -> fully connected output.
Follows the same pattern as Method_CNN from Stage 3.
'''

import time
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from code.base_class.method import method


# ──────────────────────────────────────────────
# dataset wrapper for batching
# ──────────────────────────────────────────────

class TextClassificationDataset(Dataset):
    '''Wraps tokenized sequences + labels for DataLoader.'''

    def __init__(self, instances, max_seq_length=300):
        self.sequences = []
        self.labels = []
        self.lengths = []

        for inst in instances:
            seq = inst['sequence'][:max_seq_length]  # truncate long reviews
            self.sequences.append(seq)
            self.labels.append(inst['label'])
            self.lengths.append(len(seq))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index], self.lengths[index]


def collate_batch(batch):
    '''Pad sequences to the same length within each batch.'''
    sequences, labels, lengths = zip(*batch)

    max_len = max(lengths)
    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded, labels, lengths


# ──────────────────────────────────────────────
# the actual RNN classification model
# ──────────────────────────────────────────────

class RNNClassifier(nn.Module):
    '''
    Embedding -> RNN/LSTM/GRU -> take last hidden state -> FC -> 2 classes.
    The cell_type is swappable so we can easily do the ablation study.
    '''

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 num_layers, cell_type, dropout, bidirectional):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type

        # embedding: turns word indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # pick the recurrent cell — this is the key swappable part
        rnn_class = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell_type]

        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # if bidirectional, hidden is doubled
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_dim, output_dim)

    def forward(self, x, lengths):
        # x shape: (batch, seq_len) — integer token ids

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        # pack padded sequences so the RNN ignores padding
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, hidden = self.rnn(packed)

        # grab the last hidden state
        if self.cell_type == 'lstm':
            hidden = hidden[0]  # lstm returns (h_n, c_n), we want h_n

        # hidden shape: (num_layers * directions, batch, hidden_dim)
        # grab the last layer's hidden state
        if self.bidirectional:
            # concat forward and backward last layer
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)  # (batch, output_dim)

        return output


# ──────────────────────────────────────────────
# method wrapper (follows the stage 3 pattern)
# ──────────────────────────────────────────────

class Method_RNN_Classification(method, nn.Module):
    '''Wraps the RNN classifier with the project framework's method interface.'''

    data = None
    max_epoch = 10
    learning_rate = 1e-3
    batch_size = 64

    def __init__(self, mName, mDescription, config=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.config = config or {}

        # ── training hyperparams ──
        self.max_epoch = self.config.get('max_epoch', self.max_epoch)
        self.learning_rate = self.config.get('learning_rate', self.learning_rate)
        self.batch_size = self.config.get('batch_size', self.batch_size)
        self.weight_decay = self.config.get('weight_decay', 1e-5)
        self.max_seq_length = self.config.get('max_seq_length', 300)

        # ── model architecture ──
        self.cell_type = self.config.get('cell_type', 'lstm')  # 'rnn', 'lstm', or 'gru'
        self.embedding_dim = self.config.get('embedding_dim', 128)
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.3)
        self.bidirectional = self.config.get('bidirectional', True)

        # ── tracking ──
        self.loss_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.training_time_seconds = 0.0
        self.best_epoch = None
        self.best_accuracy = -1.0
        self.best_state_dict = None

        # ── device (use GPU if available) ──
        device_name = self.config.get('device')
        if device_name is None:
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name)

        # model gets built later in run() once we know vocab size
        self.network = None

    def _build_network(self, vocab_size):
        '''Create the RNN model with the right vocab size.'''
        self.network = RNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=2,  # pos vs neg
            num_layers=self.num_layers,
            cell_type=self.cell_type,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self.device)

    def forward(self, x, lengths):
        return self.network(x, lengths)

    def _make_loader(self, instances, shuffle):
        ds = TextClassificationDataset(instances, self.max_seq_length)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_batch,
            num_workers=0,
            pin_memory=(self.device.type == 'cuda'),
        )

    def fit(self, train_instances, test_instances):
        '''Train the model, track loss and accuracy each epoch.'''
        train_loader = self._make_loader(train_instances, shuffle=True)
        test_loader = self._make_loader(test_instances, shuffle=False)

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        self.loss_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.best_epoch = None
        self.best_accuracy = -1.0
        self.best_state_dict = None

        self.network.train()
        start_time = time.perf_counter()

        for epoch in range(self.max_epoch):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch_x, batch_y, batch_lengths in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_lengths = batch_lengths.to(self.device)

                optimizer.zero_grad()
                logits = self.network(batch_x, batch_lengths)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                # clip gradients to avoid exploding gradients (common with RNNs)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item() * batch_y.size(0)
                total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total_samples += batch_y.size(0)

            avg_loss = total_loss / total_samples
            train_acc = total_correct / total_samples
            self.loss_history.append(avg_loss)
            self.train_accuracy_history.append(train_acc)

            # evaluate on test set
            test_acc = self._evaluate_accuracy(test_loader)
            self.test_accuracy_history.append(test_acc)

            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self.best_epoch = epoch + 1
                self.best_state_dict = copy.deepcopy(self.network.state_dict())

            print(f'  Epoch {epoch + 1}/{self.max_epoch}  '
                  f'Loss: {avg_loss:.4f}  '
                  f'Train Acc: {train_acc:.4f}  '
                  f'Test Acc: {test_acc:.4f}')

        self.training_time_seconds = time.perf_counter() - start_time
        if self.best_state_dict is not None:
            self.network.load_state_dict(self.best_state_dict)
            print(f'  restored best epoch {self.best_epoch} for final testing')
        return train_loader, test_loader

    def _evaluate_accuracy(self, loader):
        '''Quick accuracy check on a data loader.'''
        self.network.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_lengths in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_lengths = batch_lengths.to(self.device)
                logits = self.network(batch_x, batch_lengths)
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total += batch_y.size(0)
        self.network.train()
        return correct / total

    def test(self, test_instances):
        '''Run predictions on test set, return pred_y and true_y arrays.'''
        test_loader = self._make_loader(test_instances, shuffle=False)
        self.network.eval()

        all_preds = []
        all_trues = []

        with torch.no_grad():
            for batch_x, batch_y, batch_lengths in test_loader:
                batch_x = batch_x.to(self.device)
                batch_lengths = batch_lengths.to(self.device)
                logits = self.network(batch_x, batch_lengths)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_trues.append(batch_y.numpy())

        return np.concatenate(all_preds), np.concatenate(all_trues)

    def run(self):
        '''Main entry point — build model, train, test, return results.'''
        print(f'[Classification] cell_type={self.cell_type}, device={self.device}')

        vocab_size = len(self.data['word_to_index'])
        self._build_network(vocab_size)

        print(f'  vocab_size={vocab_size}, params={sum(p.numel() for p in self.network.parameters()):,}')
        print('  training...')
        self.fit(self.data['train'], self.data['test'])

        print('  testing...')
        pred_y, true_y = self.test(self.data['test'])

        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'loss_history': self.loss_history,
            'train_accuracy_history': self.train_accuracy_history,
            'test_accuracy_history': self.test_accuracy_history,
            'best_epoch': self.best_epoch,
            'best_accuracy': self.best_accuracy,
        }
