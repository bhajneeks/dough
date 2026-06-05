'''
Dataset loader for Stage 5 graph node classification.

This prepares Cora, Citeseer, and Pubmed for a GCN.
'''

from pathlib import Path
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from code.base_class.dataset import dataset


DEFAULT_SPLITS = {
    'cora': {'train_per_class': 20, 'test_per_class': 150, 'val_per_class': 0},
    'citeseer': {'train_per_class': 20, 'test_per_class': 200, 'val_per_class': 0},
    'pubmed': {'train_per_class': 20, 'test_per_class': 200, 'val_per_class': 0},
}


class Dataset_Loader(dataset):
    '''
    Load one Stage 5 citation graph and create class-balanced node splits.

    Expected source layout:
        data/stage_5_data/<dataset_name>/node
        data/stage_5_data/<dataset_name>/link

    The returned dictionary follows the original project template shape:
        {
            'graph': {
                'node': node_id_to_row_index,
                'edge': edge_index_array,
                'X': feature_tensor,
                'y': label_tensor,
                'utility': {'A': normalized_sparse_adjacency, ...}
            },
            'train_test_val': {
                'idx_train': ...,
                'idx_test': ...,
                'idx_val': ...,
                'split_summary': ...
            }
        }
    '''

    data = None
    dataset_name = None
    dataset_source_folder_path = None

    def __init__(
        self,
        seed=170,
        dName=None,
        dDescription=None,
        train_per_class=None,
        test_per_class=None,
        val_per_class=None,
        normalize_features=True,
    ):
        super().__init__(dName, dDescription)
        self.seed = 170 if seed is None else seed
        self.train_per_class = train_per_class
        self.test_per_class = test_per_class
        self.val_per_class = val_per_class
        self.normalize_features = normalize_features

    def _resolve_dataset_path(self):
        if self.dataset_name not in DEFAULT_SPLITS:
            supported = ', '.join(sorted(DEFAULT_SPLITS))
            raise ValueError(f'Unknown dataset_name {self.dataset_name!r}; expected one of: {supported}')

        if self.dataset_source_folder_path is None:
            dough_dir = Path(__file__).resolve().parents[2]
            return dough_dir / 'data' / 'stage_5_data' / self.dataset_name

        source_path = Path(self.dataset_source_folder_path)
        if (source_path / 'node').exists() and (source_path / 'link').exists():
            return source_path

        dataset_path = source_path / self.dataset_name
        if (dataset_path / 'node').exists() and (dataset_path / 'link').exists():
            return dataset_path

        raise FileNotFoundError(
            'Could not find node/link files. Set dataset_source_folder_path to '
            'either data/stage_5_data or data/stage_5_data/<dataset_name>.'
        )

    def _split_config(self):
        defaults = DEFAULT_SPLITS[self.dataset_name]
        return {
            'train_per_class': (
                defaults['train_per_class']
                if self.train_per_class is None
                else self.train_per_class
            ),
            'test_per_class': (
                defaults['test_per_class']
                if self.test_per_class is None
                else self.test_per_class
            ),
            'val_per_class': (
                defaults['val_per_class']
                if self.val_per_class is None
                else self.val_per_class
            ),
        }

    def row_normalize(self, mx):
        '''Normalize each feature row to sum to 1.'''
        rowsum = np.array(mx.sum(1)).flatten()
        r_inv = np.power(rowsum, -1.0)
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(mx)

    def adj_normalize(self, mx):
        '''Symmetric GCN normalization: D^-1/2 A D^-1/2.'''
        rowsum = np.array(mx.sum(1)).flatten()
        r_inv = np.power(rowsum, -0.5)
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(mx).dot(r_mat_inv)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        '''Convert a scipy sparse matrix to a torch sparse COO tensor.'''
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='Sparse invariant checks are implicitly disabled.*',
            )
            try:
                return torch.sparse_coo_tensor(
                    indices,
                    values,
                    shape,
                    check_invariants=False,
                ).coalesce()
            except TypeError:
                return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def encode_labels(self, labels):
        '''Map raw dataset labels to deterministic integer class ids.'''
        class_names = sorted(set(labels), key=str)
        label_to_index = {label: idx for idx, label in enumerate(class_names)}
        encoded = np.array([label_to_index[label] for label in labels], dtype=np.int64)
        return encoded, class_names, label_to_index

    def class_balanced_split(self, labels, class_names):
        '''Create reproducible class-balanced train/test/optional-val splits.'''
        config = self._split_config()
        rng = np.random.default_rng(self.seed)

        idx_train = []
        idx_val = []
        idx_test = []
        per_class_counts = {}

        for class_idx, class_name in enumerate(class_names):
            class_indices = np.where(labels == class_idx)[0]
            rng.shuffle(class_indices)

            train_n = config['train_per_class']
            val_n = config['val_per_class']
            test_n = config['test_per_class']
            needed = train_n + val_n + test_n
            if len(class_indices) < needed:
                raise ValueError(
                    f'Class {class_name!r} has {len(class_indices)} nodes, '
                    f'but the requested split needs {needed}.'
                )

            train_end = train_n
            val_end = train_end + val_n
            test_end = val_end + test_n

            idx_train.extend(class_indices[:train_end])
            idx_val.extend(class_indices[train_end:val_end])
            idx_test.extend(class_indices[val_end:test_end])

            per_class_counts[class_name] = {
                'available': int(len(class_indices)),
                'train': int(train_n),
                'val': int(val_n),
                'test': int(test_n),
            }

        idx_train = np.array(idx_train, dtype=np.int64)
        idx_val = np.array(idx_val, dtype=np.int64)
        idx_test = np.array(idx_test, dtype=np.int64)

        rng.shuffle(idx_train)
        rng.shuffle(idx_val)
        rng.shuffle(idx_test)

        split_summary = {
            'seed': self.seed,
            'train_per_class': int(config['train_per_class']),
            'val_per_class': int(config['val_per_class']),
            'test_per_class': int(config['test_per_class']),
            'total_train': int(len(idx_train)),
            'total_val': int(len(idx_val)),
            'total_test': int(len(idx_test)),
            'per_class': per_class_counts,
        }

        return (
            torch.LongTensor(idx_train),
            torch.LongTensor(idx_val),
            torch.LongTensor(idx_test),
            split_summary,
        )

    def load(self):
        '''Load citation graph features, labels, normalized adjacency, and splits.'''
        dataset_path = self._resolve_dataset_path()
        print(f'Loading {self.dataset_name} dataset from {dataset_path}...')

        idx_features_labels = np.genfromtxt(dataset_path / 'node', dtype=np.dtype(str))
        raw_node_ids = np.array(idx_features_labels[:, 0], dtype=np.int64)
        raw_labels = idx_features_labels[:, -1]

        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        if self.normalize_features:
            features = self.row_normalize(features)

        encoded_labels, class_names, label_to_index = self.encode_labels(raw_labels)

        idx_map = {node_id: row_idx for row_idx, node_id in enumerate(raw_node_ids)}
        reverse_idx_map = {row_idx: node_id for node_id, row_idx in idx_map.items()}

        edges_unordered = np.genfromtxt(dataset_path / 'link', dtype=np.int64)
        if edges_unordered.ndim == 1:
            edges_unordered = edges_unordered.reshape(1, -1)
        if edges_unordered.shape[1] != 2:
            raise ValueError('The link file must have two columns.')

        mapped_edges_flat = [idx_map.get(node_id) for node_id in edges_unordered.flatten()]
        if any(edge_idx is None for edge_idx in mapped_edges_flat):
            raise ValueError('At least one edge endpoint does not appear in the node file.')

        edges = np.array(mapped_edges_flat, dtype=np.int64).reshape(edges_unordered.shape)
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0], dtype=np.float32), (edges[:, 0], edges[:, 1])),
            shape=(encoded_labels.shape[0], encoded_labels.shape[0]),
            dtype=np.float32,
        ).tocsr()

        # GCN usually treats citation links as undirected.
        adj = adj.maximum(adj.T)
        adj.data[:] = 1.0
        adj.setdiag(1.0)
        adj.eliminate_zeros()

        norm_adj = self.adj_normalize(adj)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(encoded_labels)
        adj_tensor = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        idx_train, idx_val, idx_test, split_summary = self.class_balanced_split(
            encoded_labels,
            class_names,
        )

        train_test_val = {
            'idx_train': idx_train,
            'idx_test': idx_test,
            'idx_val': idx_val,
            'split_summary': split_summary,
        }

        graph = {
            'node': idx_map,
            'edge': edges,
            'X': features,
            'y': labels,
            'utility': {
                'A': adj_tensor,
                'reverse_idx': reverse_idx_map,
                'class_names': class_names,
                'label_to_index': label_to_index,
                'normalize_features': self.normalize_features,
            },
        }

        self.data = {'graph': graph, 'train_test_val': train_test_val}
        return self.data
