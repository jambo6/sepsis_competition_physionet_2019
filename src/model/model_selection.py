"""
model_selection.py
===========================
"""
from copy import deepcopy
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold


def stratified_kfold_cv(dataset, labels, n_splits=5, seed=3):
    """Custom cv selection algorithm.

    The imbalance of sepsis to non-sepsis cases, along with the variability in the lengths of the series makes choosing
    representative CV folds difficult. Here we provide a custom method that stratifies by overall group, that is, by
    sepsis of non sepsis, and outputs the cv to include all indexes corresponding to those time-series.

    Args:
        dataset (TimeSeriesDataset): A TimeSeriesDataset instance.
        labels (torch.Tensor): The full label list, there must be a label for each time-point in the dataset.
        n_splits (int): Number of CV splits.
        return_as_list (bool): Set true to return the indexes as a list where each entry corresponds to the indexes for
            a given id.
        seed (int): Random seed.

    """
    # Set seed
    np.random.seed(seed)

    # Get the ids that end up with sepsis
    all_idxs = torch.arange(labels.shape[0])
    all_ids = torch.arange(dataset.size(0))

    # Find which ids contain a one and their corresponding indexes
    labels_ = deepcopy(labels)
    id_labels, id_idxs = [], []
    idx_start = 0
    for l in dataset.lengths:
        id_labels.append(labels[:l])
        id_idxs.append(all_idxs[idx_start:idx_start + l])
        idx_start += l
        labels_ = labels_[l:]
    contains_one = np.array([labels.max().item() > 0 for labels in id_labels])

    # Split the one ids into n pieces
    one_ids = all_ids[contains_one].numpy()
    np.random.shuffle(one_ids)
    val_ones = np.array_split(one_ids, n_splits)

    # Train ones
    train_ones = [list(set(one_ids) - set(x)) for x in val_ones]

    # Now cv the training set
    zero_ids = [x.item() for x in all_ids if x.item() not in one_ids]
    np.random.shuffle(zero_ids)
    val_zeros = np.array_split(np.array(zero_ids), n_splits)
    train_zeros = [list(set(zero_ids) - set(x)) for x in val_zeros]

    # Compile
    id_cv = [
        (list(train_zeros[i]) + list(train_ones[i]), list(val_zeros[i]) + list(val_ones[i])) for i in range(n_splits)
    ]

    # Finally get the indexes
    cv = []
    for train_ids, test_ids in id_cv:
        train_idxs = torch.cat([id_idxs[i] for i in train_ids]).numpy()
        test_idxs = torch.cat([id_idxs[i] for i in test_ids]).numpy()
        cv.append([train_idxs, test_idxs])

    return cv, id_cv


