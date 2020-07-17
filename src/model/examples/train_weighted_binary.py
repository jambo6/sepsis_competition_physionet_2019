from definitions import *
import torch
import numpy as np
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMRegressor, LGBMClassifier
from src.model._validation import cross_val_predict_custom
from src.data.dataset import TimeSeriesDataset
from src.data import dicts
from src.data.functions import torch_ffill
from src.features.derived_features import shock_index, partial_sofa, bun_cr
from src.features.rolling import RollingStatistic
from src.features.signatures.compute import RollingSignature
from src.model.model_selection import stratified_kfold_cv
from src.model.optimizer import CVThresholdOptimizer

# Load the dataset
dataset = TimeSeriesDataset().load(DATA_DIR + '/raw/data.tsd')

# Load the training labels
labels = load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle')

# First get counts of the laboratory values
count_variables = dicts.feature_types['laboratory'] + ['Temp']
counts = RollingStatistic(statistic='count', window_length=8).transform(dataset[count_variables])
dataset.add_features(counts)

# Apply a forward fill
dataset.data = torch_ffill(dataset.data)

# Add on some additional features
dataset['ShockIndex'] = shock_index(dataset)
dataset['PartialSOFA'] = partial_sofa(dataset)
dataset['BUN/CR'] = bun_cr(dataset)

# Now moments
changing_vars = dicts.feature_types['vitals']
dataset.add_features(RollingStatistic(statistic='moments', window_length=7).transform(dataset[changing_vars]))

# # Now generate some rolling window features
max_vals = RollingStatistic(statistic='max', window_length=6).transform(dataset[changing_vars])
min_vals = RollingStatistic(statistic='min', window_length=6).transform(dataset[changing_vars])
dataset.add_features(torch.cat((max_vals, min_vals), 2))

# Now some rolling signatures
roller = RollingSignature(window=7, depth=3, aug_list=['leadlag'], logsig=True)
for vbl in ['BUN/CR', 'PartialSOFA', 'MAP']:
    signatures = roller.transform(dataset[vbl])
    dataset.add_features(signatures)

# Extract machine learning data
X = dataset.to_ml()
assert len(X) == len(labels)    # Sanity check

# Train a model
cv, _ = stratified_kfold_cv(dataset, labels, n_splits=5, seed=6)

# Regressor
print('Regressor')
clf = LGBMRegressor(n_estimators=100)
predictions = cross_val_predict(clf, X, labels, cv=cv, n_jobs=-1)
scores = CVThresholdOptimizer(labels, predictions).optimize(cv, parallel=True)
print('Average: {:.3f}'.format(np.mean(scores)))

# Binary model
print('Binary')
labels_binary = labels.clone()
labels_binary[labels > -0.05] = 1.
labels_binary[labels == -0.05] = 0.
clf = LGBMClassifier(n_estimators=100)
predictions = cross_val_predict_custom(clf, X, labels_binary, cv=cv, n_jobs=-1, sample_weights=np.abs(labels.numpy()))
scores = CVThresholdOptimizer(labels, predictions).optimize(cv, parallel=True)
print('Average: {:.3f}'.format(np.mean(scores)))
