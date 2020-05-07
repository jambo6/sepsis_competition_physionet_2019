from definitions import *
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from lightgbm import LGBMRegressor, LGBMClassifier
from src.data.dataset import TimeSeriesDataset
from src.data.functions import torch_ffill
from src.features.derived_features import shock_index, partial_sofa
from src.features.rolling import RollingStatistic
from src.features.signatures.augmentations import apply_augmentation_list
from src.features.signatures.compute import RollingSignature
from src.model.model_selection import stratified_kfold_cv
from src.model.optimizer import ThresholdOptimizer

# Load the dataset
dataset = TimeSeriesDataset().load(DATA_DIR + '/raw/data.tsd')

# Load the training labels
labels = load_pickle(DATA_DIR + '/processed/labels/binary.pickle')

# Apply a forward fill
dataset.data = torch_ffill(dataset.data)

# Add on some additional features
dataset['ShockIndex'] = shock_index(dataset)
dataset['PartialSOFA'] = partial_sofa(dataset)

# Now generate some rolling window features
max_shock = RollingStatistic(statistic='max', window_length=5).transform(dataset['ShockIndex'])
dataset['MaxShockIndex'] = max_shock

# Now some rolling signatures
augmented_path = apply_augmentation_list(dataset[['ShockIndex', 'HR']], aug_list=['addtime'])
rolling_signature = RollingSignature(window=6, depth=3, logsig=True)
signatures = rolling_signature.transform(augmented_path)
dataset.add_features(signatures)

# Extract machine learning data
X = dataset.to_ml()
assert len(X) == len(labels)    # Sanity check

# Train a model
cv = stratified_kfold_cv(dataset, labels, n_splits=5, seed=1)

# Regressor
print('Training model...')
clf = LGBMClassifier()
predictions = cross_val_predict(clf, X, labels, cv=cv, n_jobs=-1)

# Evaluation
print('Thresholding...')
scores = ThresholdOptimizer(labels, predictions).optimize_cv(cv, parallel=True)
print('Average: {:.3f}'.format(np.mean(scores)))