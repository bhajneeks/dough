'''
RNN model for text generation (jokes dataset).

Trains a language model: given previous words, predict the next word.
Then generates text by feeding predictions back in autoregressively.
Supports RNN, LSTM, and GRU cells (swappable via config).
'''

import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from code.base_class.method import method
from code.stage_4_code.Dataset_Loader import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN


# ──────────────────────────────────────────────
# dataset: each joke becomes (input_seq, target_seq) pairs
# ──────────────────────────────────────────────

class TextGenerationDataset(Dataset):
    '''
    For language modeling, input is the sequence shifted by one:
      input:  [SOS, w1, w2, w3, ...]
      target: [w1,  w2, w3, ..., EOS]
    '''

    def __init__(self, sequences, max_seq_length=50):
        self.inputs = []
        self.targets = []

        for seq in sequences:
            seq = seq[:max_seq_length]  # truncate if needed
            self.inputs.append(seq[:-1])    # everything except last token
            self.targets.append(seq[1:])    # everything except first token

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


def collate_generation(batch):
    '''Pad input and target sequences to the same length in a batch.'''
    inputs, targets = zip(*batch)

    max_len = max(len(s) for s in inputs)

    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long)
    lengths = []

    for i in range(len(inputs)):
        seq_len = len(inputs[i])
        padded_inputs[i, :seq_len] = torch.tensor(inputs[i], dtype=torch.long)
        padded_targets[i, :seq_len] = torch.tensor(targets[i], dtype=torch.long)
        lengths.append(seq_len)

    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded_inputs, padded_targets, lengths


# ──────────────────────────────────────────────
# the language model
# ──────────────────────────────────────────────

class RNNLanguageModel(nn.Module):
    '''
    Embedding -> RNN/LSTM/GRU -> FC -> vocab_size logits per timestep.
    '''

    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 num_layers, cell_type, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        rnn_class = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell_type]

        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        '''
        x: (batch, seq_len)
        returns: logits (batch, seq_len, vocab_size), hidden state
        '''
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        '''Zero-initialize hidden state.'''
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        if self.cell_type == 'lstm':
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h, c)
        return h


# ──────────────────────────────────────────────
# method wrapper
# ──────────────────────────────────────────────

class Method_RNN_Generation(method, nn.Module):
    '''Wraps the language model with the project framework's method interface.'''

    data = None
    max_epoch = 30
    learning_rate = 1e-3
    batch_size = 32

    def __init__(self, mName, mDescription, config=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.config = config or {}

        # ── training hyperparams ──
        self.max_epoch = self.config.get('max_epoch', self.max_epoch)
        self.learning_rate = self.config.get('learning_rate', self.learning_rate)
        self.batch_size = self.config.get('batch_size', self.batch_size)
        self.max_seq_length = self.config.get('max_seq_length', 50)

        # ── model architecture ──
        self.cell_type = self.config.get('cell_type', 'lstm')
        self.embedding_dim = self.config.get('embedding_dim', 128)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.3)

        # ── generation settings ──
        self.generation_length = self.config.get('generation_length', 30)
        self.temperature = self.config.get('temperature', 0.8)
        self.sample_temperatures = self.config.get('sample_temperatures', [self.temperature])

        # ── tracking ──
        self.loss_history = []
        self.training_time_seconds = 0.0

        # ── device ──
        device_name = self.config.get('device')
        if device_name is None:
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name)

        self.network = None

    def _build_network(self, vocab_size):
        self.network = RNNLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            cell_type=self.cell_type,
            dropout=self.dropout,
        ).to(self.device)

    def fit(self, sequences):
        '''Train the language model on joke sequences.'''
        dataset = TextGenerationDataset(sequences, self.max_seq_length)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_generation,
            num_workers=0,
            pin_memory=(self.device.type == 'cuda'),
        )

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
        )
        # ignore padding (index 0) when computing loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        self.loss_history = []
        self.network.train()
        start_time = time.perf_counter()

        for epoch in range(self.max_epoch):
            total_loss = 0.0
            total_tokens = 0

            for batch_in, batch_target, lengths in loader:
                batch_in = batch_in.to(self.device)
                batch_target = batch_target.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.network(batch_in)

                # reshape for cross entropy: (batch*seq_len, vocab) vs (batch*seq_len)
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    batch_target.reshape(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
                optimizer.step()

                # count non-padding tokens
                non_pad = (batch_target != 0).sum().item()
                total_loss += loss.item() * non_pad
                total_tokens += non_pad

            avg_loss = total_loss / total_tokens
            self.loss_history.append(avg_loss)
            print(f'  Epoch {epoch + 1}/{self.max_epoch}  Loss: {avg_loss:.4f}')

        self.training_time_seconds = time.perf_counter() - start_time

    def generate(self, seed_words, word_to_index, index_to_word, temperature=None):
        '''
        Generate text starting from seed_words (a list of strings).
        Uses temperature sampling for variety.
        '''
        self.network.eval()

        sos_idx = word_to_index[SOS_TOKEN]
        eos_idx = word_to_index[EOS_TOKEN]
        unk_idx = word_to_index.get('<UNK>', 1)

        # start with SOS + the seed words
        input_indices = [sos_idx]
        for w in seed_words:
            input_indices.append(word_to_index.get(w.lower(), unk_idx))

        sample_temperature = self.temperature if temperature is None else temperature
        sample_temperature = max(sample_temperature, 0.05)

        generated_words = list(seed_words)
        hidden = None

        with torch.no_grad():
            # feed the seed sequence through the model first
            input_tensor = torch.tensor([input_indices], dtype=torch.long, device=self.device)
            logits, hidden = self.network(input_tensor, hidden)

            # start generating from the last position
            next_logits = logits[0, -1, :]  # logits for the next word

            for _ in range(self.generation_length):
                # temperature sampling
                probs = torch.softmax(next_logits / sample_temperature, dim=0)
                next_idx = torch.multinomial(probs, 1).item()

                # stop if we hit EOS or PAD
                if next_idx == eos_idx or next_idx == 0:
                    break

                word = index_to_word.get(next_idx, '<UNK>')
                generated_words.append(word)

                # feed the predicted word back in
                next_input = torch.tensor([[next_idx]], dtype=torch.long, device=self.device)
                next_logits_out, hidden = self.network(next_input, hidden)
                next_logits = next_logits_out[0, -1, :]

        self.network.train()
        return ' '.join(generated_words)

    def run(self):
        '''Main entry point — train and generate sample text.'''
        print(f'[Generation] cell_type={self.cell_type}, device={self.device}')

        word_to_index = self.data['word_to_index']
        index_to_word = self.data['index_to_word']
        vocab_size = len(word_to_index)
        self._build_network(vocab_size)

        print(f'  vocab_size={vocab_size}, params={sum(p.numel() for p in self.network.parameters()):,}')
        print('  training...')
        self.fit(self.data['sequences'])

        # generate several samples with different seed words
        seed_sets = [
            ['what', 'did', 'the'],
            ['why', 'did', 'the'],
            ['how', 'do', 'you'],
            ['i', 'told', 'my'],
        ]

        print('  generating samples...')
        generated_texts = []
        for temperature in self.sample_temperatures:
            for seeds in seed_sets:
                text = self.generate(seeds, word_to_index, index_to_word,
                                     temperature=temperature)
                labeled_text = f'temp={temperature:.1f}, seeds={seeds} -> {text}'
                generated_texts.append(labeled_text)
                print(f'    {labeled_text}')

        return {
            'generated_texts': generated_texts,
            'seed_sets': seed_sets,
            'loss_history': self.loss_history,
        }
