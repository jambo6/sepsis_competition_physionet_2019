from definitions import *
import numpy as np
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMRegressor, LGBMClassifier
from src.data.dataset import TimeSeriesDataset
from src.data.functions import torch_ffill
from src.model.model_selection import stratified_kfold_cv
from src.model.optimizer import CVThresholdOptimizer
from src.model._validation import cross_val_predict_custom

# Load the dataset
dataset = TimeSeriesDataset().load(DATA_DIR + '/raw/data.tsd')

# Load the training labels
labels = load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle')

# Apply a forward fill
dataset.data = torch_ffill(dataset.data)

# Extract machine learning data
X = dataset.to_ml()
assert len(X) == len(labels)    # Sanity check

# Train a model
cv, _ = stratified_kfold_cv(dataset, labels, n_splits=5, seed=1)

# Regressor
print('Regressor')
clf = LGBMRegressor(n_estimators=100)
predictions = cross_val_predict(clf, X, labels, cv=cv, n_jobs=-1)
scores = CVThresholdOptimizer(labels, predictions).optimize(cv, parallel=True)
print('Average: {:.3f}'.format(np.mean(scores)))


# Binary model
print('Binary p/m 1')
labels_binary = labels.clone()
labels_binary[labels > -0.05] = 1.
labels_binary[labels == -0.05] = -1.
clf = LGBMClassifier(n_estimators=100)
predictions = cross_val_predict_custom(clf, X, labels_binary, cv=cv, n_jobs=-1, sample_weights=np.abs(labels.numpy()))
scores = CVThresholdOptimizer(labels, predictions).optimize(cv, parallel=True)
print('Average: {:.3f}'.format(np.mean(scores)))

# Binary model
print('Binary zero')
labels_binary = labels.clone()
labels_binary[labels > -0.05] = 1.
labels_binary[labels == -0.05] = 0.
clf = LGBMClassifier(n_estimators=100)
predictions = cross_val_predict_custom(clf, X, labels_binary, cv=cv, n_jobs=-1, sample_weights=np.abs(labels.numpy()))
scores = CVThresholdOptimizer(labels, predictions).optimize(cv, parallel=True)
print('Average: {:.3f}'.format(np.mean(scores)))

# # Newer
# print('The wcustom loss thing with u')
# clf = LGBMClassifier(n_estimators=100, objective=weighted_log_likelihood)
# predictions = cross_val_predict(clf, X, labels_binary, cv=cv, n_jobs=-1, sample_weights=labels.numpy(), objective=weighted_log_likelihood)
# scores = CVThresholdOptimizer(labels, predictions).optimize(cv, parallel=False)
# print('Average: {:.3f}'.format(np.mean(scores)))

