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


