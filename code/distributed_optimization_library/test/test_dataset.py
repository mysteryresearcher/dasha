import pytest

import numpy as np
import scipy.sparse

from distributed_optimization_library.dataset import Dataset, SparseDataset


def test_split():
    data = np.array([[2, 3], [4, 5], [-1, 0]], dtype=np.float32)
    labels = np.array([1, 0, 1])
    dataset = Dataset(data, labels)
    datasets = dataset.split(2)
    data, labels = datasets[0].get_data_and_labels()
    assert data.tolist() == [[2, 3], [4, 5]]
    assert labels.tolist() == [1, 0]
    data, labels = datasets[1].get_data_and_labels()
    assert data.tolist() == [[-1, 0]]
    assert labels.tolist() == [1]
    
    
def test_split_without_remainder():
    data = np.array([[2, 3], [4, 5], [-1, 0], [4, 0]], dtype=np.float32)
    labels = np.array([1, 0, 1, 0])
    dataset = Dataset(data, labels)
    datasets = dataset.split(3, ignore_remainder=True)
    data, labels = datasets[0].get_data_and_labels()
    assert data.tolist() == [[2, 3]]
    assert labels.tolist() == [1]
    data, labels = datasets[1].get_data_and_labels()
    assert data.tolist() == [[4, 5]]
    assert labels.tolist() == [0]
    data, labels = datasets[2].get_data_and_labels()
    assert data.tolist() == [[-1, 0]]
    assert labels.tolist() == [1]


def test_sparse_split_without_remainder():
    data = scipy.sparse.csr_matrix(np.array([[2, 3], [4, 5], [-1, 0], [4, 0]], dtype=np.float32))
    labels = np.array([1, 0, 1, 0])
    dataset = SparseDataset(data, labels)
    datasets = dataset.split(3, ignore_remainder=True)
    data, labels = datasets[0].get_data_and_labels()
    assert data.tolist() == [[2, 3]]
    assert labels.tolist() == [1]
    data, labels = datasets[1].get_data_and_labels()
    assert data.tolist() == [[4, 5]]
    assert labels.tolist() == [0]
    data, labels = datasets[2].get_data_and_labels()
    assert data.tolist() == [[-1, 0]]
    assert labels.tolist() == [1]


def test_combine():
    data = np.array([[2, 3], [4, 5]], dtype=np.float32)
    labels = np.array([1, 0])
    dataset_first = Dataset(data, labels)
    data = np.array([[-1, 0]], dtype=np.float32)
    labels = np.array([1])
    dataset_second = Dataset(data, labels)
    data, labels = Dataset.combine([dataset_first, dataset_second]).get_data_and_labels()
    assert data.tolist() == [[2, 3], [4, 5], [-1, 0]]
    assert labels.tolist() == [1, 0, 1]


def test_split_into_groups_by_labels():
    num_samples = 5
    data = np.array([[2, 3], [4, 5], [-1, 0], [6, 2], [1, -3]], dtype=np.float32)
    labels = np.array([1, 0, 1, 1, 0])
    dataset = Dataset(data, labels)
    datasets, nodes_indices_splits = dataset.split_into_groups_by_labels(4)
    assert nodes_indices_splits == [0, 2, 4]
    data, labels = datasets[0].get_data_and_labels()
    assert data.tolist() == [[4, 5]]
    assert labels.tolist() == [0]
    data, labels = datasets[1].get_data_and_labels()
    assert data.tolist() == [[1, -3]]
    assert labels.tolist() == [0]
    data, labels = datasets[2].get_data_and_labels()
    assert data.tolist() == [[2, 3], [-1, 0]]
    assert labels.tolist() == [1, 1]
    data, labels = datasets[3].get_data_and_labels()
    assert data.tolist() == [[6, 2]]
    assert labels.tolist() == [1]


def test_split_with_controling_homogeneity():
    num_samples = 3
    data = np.array([[2, 3], [4, 5], [-1, 0]], dtype=np.float32)
    labels = np.array([1, 0, 1])
    dataset = Dataset(data, labels)
    for prob in [0, 1]:
        datasets = dataset.split_with_controling_homogeneity(2, prob_taking_from_hold_out=prob, seed=42)
        if prob == 1.0:
            data, labels = datasets[0].get_data_and_labels()
            assert data.tolist() == [[2, 3]]
            assert labels.tolist() == [1]
            data, labels = datasets[1].get_data_and_labels()
            assert data.tolist() == [[2, 3]]
            assert labels.tolist() == [1]
        if prob == 0.0:
            data, labels = datasets[0].get_data_and_labels()
            assert data.tolist() == [[4, 5]]
            assert labels.tolist() == [0]
            data, labels = datasets[1].get_data_and_labels()
            assert data.tolist() == [[-1, 0]]
            assert labels.tolist() == [1]
