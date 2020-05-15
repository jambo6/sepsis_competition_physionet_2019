import numpy as np
import torch


def torch_ffill(data):
    """ Forward fill for a torch tensor.

    This (currently) assumes a torch tensor input of shape [N, L, C] and will forward will along the 2nd (L)
    dimension.

    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """
    def ffill2d(arr):
        """ 2d ffill. """
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]
        return out

    data_ffilled = torch.Tensor([ffill2d(x.numpy().T) for x in data]).transpose(1, 2)
    return data_ffilled


def pytorch_rolling(x, dimension, window_size, step_size=1, return_same_size=True):
    """ Outputs an expanded tensor to perform rolling window operations on a pytorch tensor.

    Given an input tensor of shape [N, L, C] and a window length W, computes an output tensor of shape [N, L-W, C, W]
    where the final dimension contains the values from the current timestep to timestep - W + 1.

    Args:
        x (torch.Tensor): Tensor of shape [N, L, C].
        dimension (int): Dimension to open.
        window_size (int): Length of the rolling window.
        step_size (int): Window step, defaults to 1.
        return_same_size (bool): Set True to return a tensor of the same size as the input tensor with nan values filled
                                 where insufficient prior window lengths existed. Otherwise returns a reduced size
                                 tensor from the paths that had sufficient data.

    Returns:
        torch.Tensor: Tensor of shape [N, L, C, W] where the window values are opened into the fourth W dimension.
    """
    if return_same_size:
        x_dims = list(x.size())
        x_dims[dimension] = window_size - 1
        nans = np.nan * torch.zeros(x_dims)
        x = torch.cat((nans, x), dim=dimension)

    # Unfold ready for mean calculations
    unfolded = x.unfold(dimension, window_size, step_size)

    return unfolded


def torch_ffill_3d(data):
    """ Forward fill a 3d tensor in one swift movement. """
    pass


if __name__ == '__main__':
    a = torch.randn(3, 4, 2)
    a[0, [2, 3], 0] = np.nan
    a[0, [1, 2, 3], 1] = np.nan
    a[1, [1, 3], 0] = np.nan
    a[1, 3, 1] = np.nan
    torch_ffill_3d(data)
