"""
model_selection.py
===========================
"""
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold


def stratified_kfold_cv(dataset, labels, n_splits=5, return_as_list=False, seed=1):
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
    # Set a seed so the same cv is used everytime
    np.random.seed(seed)

    # List of all indexes in the dataset
    all_idxs = torch.arange(labels.shape[0])

    # Find which ids contain a one and their corresponding indexes
    id_labels, id_idxs = [], []
    idx_start = 0
    for l in dataset.lengths:
        id_labels.append(labels[:l])
        id_idxs.append(all_idxs[idx_start:idx_start + l])
        idx_start += l
        labels = labels[l:]
    contains_one = np.array([labels.max().item() > 0 for labels in id_labels])

    # Now perform a group K fold on the ids so we have stratified sepsis/non-sepsis split
    id_cv = list(StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
                 .split(contains_one, contains_one, groups=contains_one))

    # Now convert this back onto the actual time series ids
    cv = []
    for i, fold in enumerate(id_cv):
        train_idxs = [id_idxs[i] for i in fold[0]]
        test_idxs = [id_idxs[i] for i in fold[1]]

        if not return_as_list:
            train_idxs = np.concatenate([id_idxs[i] for i in fold[0]])
            test_idxs = np.concatenate([id_idxs[i] for i in fold[1]])

        cv.append([train_idxs, test_idxs])

    return cv, id_cv

