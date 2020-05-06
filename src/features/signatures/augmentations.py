"""
augmentations.py
================================
Contains sklearn transformers for path augmentations to be applied before computation of signatures.

This file includes
    - AddTime
    - PenOff
    - LeadLag
    - Cumulative Sum
    - Basepoint
"""
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class AddTime(BaseEstimator, TransformerMixin):
    """Add time component to each path.

    For a path of shape [N, L, C] this adds a time channel to be placed at the first index. The time channel will be of
    length L and scaled to exist in [0, 1].
    """
    def fit(self, X, y=None):
        return self

    def transform(self, data):
        # Batch and length dim
        B, L = data.shape[0], data.shape[1]

        # Time scaled to 0, 1
        time_scaled = torch.linspace(0, 1, L).repeat(B, 1).view(B, L, 1)

        return torch.cat((time_scaled, data), 2)


class PenOff(BaseEstimator, TransformerMixin):
    """Adds a 'penoff' dimension to each path. """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Batch, length, channels
        B, L, C = X.shape[0], X.shape[1], X.shape[2]

        # Add in a dimension of ones
        X_pendim = torch.cat((torch.ones(B, L, 1), X), 2)

        # Add pen down to 0
        pen_down = X_pendim[:, [-1], :]
        pen_down[:, :, 0] = 0
        X_pendown = torch.cat((X_pendim, pen_down), 1)

        # Add home
        home = torch.zeros(B, 1, C + 1)
        X_penoff = torch.cat((X_pendown, home), 1)

        return X_penoff


class LeadLag(BaseEstimator, TransformerMixin):
    """Applies the leadlag transformation to each path.

    Example:
        [1, 2, 3] -> [[1, 1], [2, 1], [2, 2], [3, 2], [3, 3]]
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Interleave
        X_repeat = X.repeat_interleave(2, dim=1)

        # Split out lead and lag
        lead = X_repeat[:, 1:, :]
        lag = X_repeat[:, :-1, :]

        # Combine
        X_leadlag = torch.cat((lead, lag), 2)

        return X_leadlag


class CumulativeSum(BaseEstimator, TransformerMixin):
    """Cumulative sum transform. """
    def __init__(self, append_zero=False):
        self.append_zero = append_zero

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.append_zero:
            X = Basepoint().transform(X)
        return torch.cumsum(X, 1)


class Basepoint(BaseEstimator, TransformerMixin):
    """ This will append a zero starting vector to every path. """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        zero_vec = torch.zeros(size=(X.size(0), 1, X.size(2)))
        return torch.cat((zero_vec, X), dim=1)


def build_augmentation_pipeline(aug_list):
    """ Builds a pipeline from a string list of augmentations.

    Args:
        aug_list (list): List of augmentation strings.

    Example:
        >>> aug_list = ['leadlag', 'penoff']
        >>> build_augmentation_pipeline(aug_list)
        will return
        Pipeline([
            ('leadlag', LeadLag()),
            ('penoff', PenOff())
        ])
    """
    AUGMENTATIONS = {
        'leadlag': LeadLag(),
        'penoff': PenOff(),
        'addtime': AddTime(),
        'cumsum': CumulativeSum(),
        'basepoint': Basepoint()
    }

    pipeline = Pipeline([
        (tfm_str, AUGMENTATIONS[tfm_str]) for tfm_str in aug_list
    ])

    return pipeline


def apply_augmentation_list(data, aug_list):
    """Applies augmentations to the data if specified in list format with keys corresponding to AUGMENTATIONS.keys().

    This will build a sklearn pipeline from the augmentation list, as such, each augmentation must operate a fit and
    a transform method.

    Example:
        >>> out_data = apply_augmentation_list(data, ['addtime', 'leadlag'])
        # Is equivalent to
        >>> out_data = LeadLag().transform(AddTime().transform(data))

    Args:
        data (torch.Tensor): [N, L, C] shaped data.
        aug_list (list): A list of augmentation strings that correspond to an element of AUGMENTATIONS.

    Returns:
        torch.Tensor: Data with augmentations applied in order.
    """
    pipeline = build_augmentation_pipeline(aug_list)

    # Transform
    data_tfmd = pipeline.fit_transform(data)

    return data_tfmd

