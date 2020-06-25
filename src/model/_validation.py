"""
Contains a custom cross_val_predict method as the sklearn one does not allow us to specify our custom label weights.
"""
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.model_selection._validation import _fit_and_predict, _check_is_permutation, _num_samples, check_cv, \
    indexable, is_classifier, _check_fit_params, _safe_split, _enforce_prediction_order


def cross_val_predict_custom(estimator, X, y=None, groups=None, cv=None,
                      n_jobs=None, verbose=0, fit_params=None,
                      pre_dispatch='2*n_jobs', method='predict',
                      # New stuff
                      sample_weights=None, objective=None):
    """ A copy of the sklearn function but allows for different sample weights to be put in for each run. """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # If classification methods produce multiple columns of output,
    # we need to manually encode classes to ensure consistent column ordering.
    encode = method in ['decision_function', 'predict_proba',
                        'predict_log_proba']
    if encode:
        y = np.asarray(y)
        if y.ndim == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.ndim == 2:
            y_enc = np.zeros_like(y, dtype=np.int)
            for i_label in range(y.shape[1]):
                y_enc[:, i_label] = LabelEncoder().fit_transform(y[:, i_label])
            y = y_enc

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method, sample_weights, objective)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    elif encode and isinstance(predictions[0], list):
        # `predictions` is a list of method outputs from each fold.
        # If each of those is also a list, then treat this as a
        # multioutput-multiclass task. We need to separately concatenate
        # the method outputs for each label into an `n_labels` long list.
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = np.concatenate(predictions)

    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]


def _fit_and_predict(estimator, X, y, train, test, verbose, fit_params,
                     method, sample_weights, objective):
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)

    if y_train is None:
        estimator.fit(X_train, **fit_params)
    elif objective is None:
        estimator.fit(X_train, y_train, sample_weight=sample_weights[train], **fit_params)
    else:
        estimator.set_params(**{'objective': lambda l, p: objective(l, p, sample_weights[train])})
        estimator.fit(X_train, y_train, **fit_params)
    func = getattr(estimator, method)
    predictions = func(X_test)
    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        if isinstance(predictions, list):
            predictions = [_enforce_prediction_order(
                estimator.classes_[i_label], predictions[i_label],
                n_classes=len(set(y[:, i_label])), method=method)
                for i_label in range(len(predictions))]
        else:
            # A 2D y array should be a binary label indicator matrix
            n_classes = len(set(y)) if y.ndim == 1 else y.shape[1]
            predictions = _enforce_prediction_order(
                estimator.classes_, predictions, n_classes, method)
    return predictions, test
