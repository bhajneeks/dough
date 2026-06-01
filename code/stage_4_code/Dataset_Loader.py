'''
Dataset Loader for Stage 4 — text classification (IMDB) and text generation (jokes).

Handles loading raw text files, basic cleaning, vocabulary building,
and converting words to integer sequences for the RNN.
'''

import csv
import os
import re
import string
from pathlib import Path
from collections import Counter

from code.base_class.dataset import dataset


# ── special tokens ──
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'  # start of sequence (used in generation)
EOS_TOKEN = '<EOS>'  # end of sequence (used in generation)


class Dataset_Loader(dataset):
    '''Loads either the classification or generation dataset depending on task.'''

    data = None
    dataset_source_folder_path = None
    task = None  # 'classification' or 'generation'
    max_train_per_class = None
    max_test_per_class = None
    max_jokes = None

    def __init__(self, dName=None, dDescription=None, task='classification'):
        super().__init__(dName, dDescription)
        self.task = task

    # ──────────────────────────────────────────────
    # text cleaning 
    # ──────────────────────────────────────────────

    def clean_text(self, text):
        '''Basic text cleaning: lowercase, strip html, remove punctuation.'''
        text = text.lower()

        # strip html tags like <br /> that show up in IMDB reviews
        text = re.sub(r'<[^>]+>', ' ', text)

        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # ──────────────────────────────────────────────
    # vocabulary builder
    # ──────────────────────────────────────────────

    def build_vocab(self, texts, max_vocab_size=20000):
        '''Count all words, keep the most frequent ones, return word-to-index mappings.'''
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())

        # grab the most common words
        most_common = word_counts.most_common(max_vocab_size)

        # build the lookup dicts — pad=0, unk=1, then real words start at 2
        word_to_index = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for word, _count in most_common:
            word_to_index[word] = len(word_to_index)

        index_to_word = {idx: word for word, idx in word_to_index.items()}

        return word_to_index, index_to_word

    def build_generation_vocab(self, texts):
        '''Vocab for generation — includes SOS/EOS tokens, keeps ALL words (small dataset).'''
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())

        word_to_index = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
        for word, _count in word_counts.most_common():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)

        index_to_word = {idx: word for word, idx in word_to_index.items()}

        return word_to_index, index_to_word

    # ──────────────────────────────────────────────
    # convert words to integer sequences
    # ──────────────────────────────────────────────

    def text_to_indices(self, text, word_to_index):
        '''Turn a string into a list of integer indices.'''
        unk_idx = word_to_index[UNK_TOKEN]
        indices = [word_to_index.get(w, unk_idx) for w in text.split()]
        if not indices:
            indices = [unk_idx]
        return indices

    # ──────────────────────────────────────────────
    # classification data loading (IMDB reviews)
    # ──────────────────────────────────────────────

    def _load_classification_split(self, split_dir, max_per_label=None):
        '''Load all .txt reviews from a train/ or test/ folder.'''
        instances = []

        for label_name, label_int in [('pos', 1), ('neg', 0)]:
            folder = os.path.join(split_dir, label_name)
            if not os.path.isdir(folder):
                print(f'  warning: folder not found: {folder}')
                continue

            loaded_for_label = 0
            for filename in sorted(os.listdir(folder)):
                if not filename.endswith('.txt'):
                    continue
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()

                cleaned = self.clean_text(raw_text)
                instances.append({'text': cleaned, 'label': label_int})
                loaded_for_label += 1

                if max_per_label is not None and loaded_for_label >= max_per_label:
                    break

        return instances

    def load_classification(self):
        '''Load IMDB data, build vocab, convert text to index sequences.'''
        print('loading classification data (IMDB)...')

        data_root = Path(self.dataset_source_folder_path) / 'text_classification'

        train_instances = self._load_classification_split(
            str(data_root / 'train'),
            max_per_label=self.max_train_per_class,
        )
        test_instances = self._load_classification_split(
            str(data_root / 'test'),
            max_per_label=self.max_test_per_class,
        )

        print(f'  train: {len(train_instances)}, test: {len(test_instances)}')

        # build vocab from training data only
        train_texts = [inst['text'] for inst in train_instances]
        word_to_index, index_to_word = self.build_vocab(train_texts, max_vocab_size=20000)
        print(f'  vocab size: {len(word_to_index)}')

        # convert text to sequences of indices
        for inst in train_instances:
            inst['sequence'] = self.text_to_indices(inst['text'], word_to_index)
        for inst in test_instances:
            inst['sequence'] = self.text_to_indices(inst['text'], word_to_index)

        self.data = {
            'train': train_instances,
            'test': test_instances,
            'word_to_index': word_to_index,
            'index_to_word': index_to_word,
        }
        return self.data

    # ──────────────────────────────────────────────
    # generation data loading (jokes)
    # ──────────────────────────────────────────────

    def _load_jokes(self, filepath):
        '''Read the jokes CSV file and return cleaned texts.'''
        jokes = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip the header row
            for row in reader:
                if len(row) < 2:
                    continue
                raw_joke = row[1]
                cleaned = self.clean_text(raw_joke)
                if len(cleaned.split()) >= 3:  # skip very short ones
                    jokes.append(cleaned)
        return jokes

    def load_generation(self):
        '''Load jokes data, build vocab, convert to sequences with SOS/EOS.'''
        print('loading generation data (jokes)...')

        data_root = Path(self.dataset_source_folder_path) / 'text_generation'
        jokes = self._load_jokes(str(data_root / 'data'))
        total_jokes = len(jokes)
        if self.max_jokes is not None:
            jokes = jokes[:self.max_jokes]

        if self.max_jokes is None:
            print(f'  jokes loaded: {len(jokes)}')
        else:
            print(f'  jokes loaded: {len(jokes)} of {total_jokes}')

        # build vocab from all jokes (small dataset, keep everything)
        word_to_index, index_to_word = self.build_generation_vocab(jokes)
        print(f'  vocab size: {len(word_to_index)}')

        # convert each joke to a sequence: [SOS, word1, word2, ..., EOS]
        sos_idx = word_to_index[SOS_TOKEN]
        eos_idx = word_to_index[EOS_TOKEN]

        sequences = []
        for joke in jokes:
            indices = self.text_to_indices(joke, word_to_index)
            sequences.append([sos_idx] + indices + [eos_idx])

        self.data = {
            'sequences': sequences,
            'raw_jokes': jokes,
            'word_to_index': word_to_index,
            'index_to_word': index_to_word,
        }
        return self.data

    # ──────────────────────────────────────────────
    # main load dispatcher
    # ──────────────────────────────────────────────

    def load(self):
        '''Load the right dataset based on self.task.'''
        if self.task == 'classification':
            return self.load_classification()
        elif self.task == 'generation':
            return self.load_generation()
        else:
            raise ValueError(f'Unknown task: {self.task}. Use "classification" or "generation".')
