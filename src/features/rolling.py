"""
rolling.py
=============================
For calculating features over rolling windows.
"""
import numpy as np
import torch
import warnings
from src.omni.decorators import timeit
from src.features.helpers import pytorch_rolling


class RollingStatistic:
    """Applies statistics to a rolling window along the time dimension of a tensor.

    Given an input tensor of shape [N, L, C] and a specified window size, W, this function first expands the tensor to
    one of shape [N, L, C, W] where W has expanded out the time dimension (this here is the L-dimension). The final
    dimension contains the most recent W time-steps (with nans if not filled). The specified statistic is then computed
    along this W dimension to give the statistic over the rolling window.

    Example:
        >>> means = RollingStatistic(statistic='mean', window_length=5).transform(data)
    """
    def __init__(self, statistic, window_length, step_size=1, func_kwargs={}):
        """
        # TODO implement a method that removes statistics that contained insufficient data.
        Args:
            statistic (str): The statistic to compute.
            window_length (int): Length of the window.
            step_size (int): Window step size.
        """
        self.statistic = statistic
        self.window_length = window_length
        self.step_size = step_size
        self.func_kwargs = func_kwargs

    @staticmethod
    def count(data):
        counts = (~torch.isnan(data)).sum(axis=-1)
        return counts.to(data.dtype)

    @staticmethod
    def max(data):
        return data.max(axis=3)[0]
        # return torch.Tensor(np.nanmax(data, axis=3))

    @staticmethod
    def min(data):
        return data.min(axis=3)[0]
        # return torch.Tensor(np.nanmin(data, axis=3))

    @staticmethod
    def mean(data):
        return torch.Tensor(np.nanmean(data, axis=3))

    @staticmethod
    def var(data):
        return torch.Tensor(np.nanvar(data, axis=3))

    @staticmethod
    def change(data):
        """ Notes the change in the variable over the interval. """
        return data[:, :, :, -1] - data[:, :, :, 0]

    @staticmethod
    def moments(data, n=3):
        """Gets statistical moments from the data.

        Args:
            data (torch.Tensor): Pytorch rolling window data.
            n (int): Moments to compute up to. Must be >=2 computes moments [2, 3, ..., n].
        """
        # Removes the mean of empty slice warning
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        assert n >= 2, "Number of moments is {}, must be >= 2.".format(n)

        # Pre computation
        nanmean = torch.Tensor(np.nanmean(data, axis=3)).unsqueeze(-1)
        # frac = torch.Tensor(1 / (data.size(3) - np.isnan(data.numpy()).sum(axis=3)))
        frac = torch.Tensor(1 / (data.size(3) - np.isnan(data.numpy()).sum(axis=3) - 1))
        frac[(frac == float("Inf")) | (frac < 0)] = float('nan')
        mean_reduced = data - nanmean

        # Compute each moment individually
        moments = []
        for i in range(2, n+1):
            moment = torch.mul(frac, torch.Tensor((mean_reduced ** i).sum(axis=3)))
            moments.append(moment)
        moments = np.concatenate(moments, axis=2)
        moments = torch.Tensor(moments)

        return moments

    @timeit
    def transform(self, data):
        # Remove mean of empty slice warning
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Error handling
        assert self.statistic in dir(self), 'Statistic {} is not implemented via this method.'.format(self.statistic)

        # Setup function
        func = eval('self.{}'.format(self.statistic))

        # Make rolling
        rolling = pytorch_rolling(data, 1, self.window_length, self.step_size)

        # Apply and output
        output = func(rolling, **self.func_kwargs)

        return output
