from definitions import *
import numpy as np
import torch
from lightgbm import LGBMClassifier
from skorch import NeuralNetClassifier
from torch.utils.data import DataLoader, Dataset
from src.data.dataset import TimeSeriesDataset
from src.data.functions import torch_ffill
from src.model.model_selection import stratified_kfold_cv
from src.model.optimizer import CVThresholdOptimizer, compute_utility_from_indexes
from src.model._validation import cross_val_predict_custom
from src.model.nets import GRU

# Lets do a gru

# Load the dataset
dataset = TimeSeriesDataset().load(DATA_DIR + '/raw/data.tsd')

# Load the training labels
utility = load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle')
labels = utility.clone()
labels[utility > -0.05] = 1.
labels[utility == -0.05] = 0.
label_weights = np.abs(utility.numpy())

# Make GRU ready data
gru_data = torch.nn.utils.rnn.pad_sequence(dataset.to_list(), padding_value=0, batch_first=True)
gru_data[gru_data != gru_data] = 0

# Apply a forward fill
dataset.data = torch_ffill(dataset.data)

# Extract machine learning data
X = dataset.to_ml()
assert len(X) == len(labels)    # Sanity check

# Train a model
cv, _ = stratified_kfold_cv(dataset, labels, n_splits=5, seed=1)

# LGBM model
clf = LGBMClassifier(n_estimators=100)
predictions = cross_val_predict_custom(clf, X, labels, cv=cv, n_jobs=-1, sample_weights=label_weights)
utility = compute_utility_from_indexes(predictions, 0.1)
print('LGBM Utility: {:.3f}'.format(utility))

# GRU
