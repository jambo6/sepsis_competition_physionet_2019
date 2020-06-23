from definitions import *
import numpy as np
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMRegressor
import torch
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
dataset['BUN/CR'] = partial_sofa(dataset)

# Now moments
changing_vars = dicts.feature_types['vitals']
dataset.add_features(RollingStatistic(statistic='moments', window_length=7).transform(dataset[changing_vars]))

# # Now generate some rolling window features
max_vals = RollingStatistic(statistic='max', window_length=6).transform(dataset[dicts.feature_types['vitals']])
min_vals = RollingStatistic(statistic='min', window_length=6).transform(dataset[dicts.feature_types['vitals']])
dataset.add_features(torch.cat((max_vals, min_vals), 2))

# Now some rolling signatures
roller = RollingSignature(window=7, depth=3, aug_list=['leadlag'], logsig=True)
for vbl in ['BUN/CR', 'PartialSOFA', 'MAP', 'HR', 'SBP']:
    signatures = roller.transform(dataset[vbl])
    dataset.add_features(signatures)

# Extract machine learning data
X = dataset.to_ml()
assert len(X) == len(labels)    # Sanity check

# Setup cv
# cv, cv_id = stratified_kfold_cv(dataset, labels, n_splits=5, seed=5)
cv = load_pickle(MODELS_DIR + '/cross_validation/cv_folds.pkl')

# Load in the lgbm gridsearch
lgbm_params = load_pickle(MODELS_DIR + '/official/lgb_fast_params.pkl')

# Regressor
print('Training model...')
clf = LGBMRegressor().set_params(**lgbm_params)
predictions = cross_val_predict(clf, X, labels, cv=cv, n_jobs=-1)

# Evaluation
print('Thresholding...')
scores = CVThresholdOptimizer(labels, predictions).optimize(cv, parallel=True)
print('Average: {:.3f}'.format(np.mean(scores)))