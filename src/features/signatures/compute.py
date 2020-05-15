"""
compute.py
================================
Methods for computing signatures.
"""
import torch
from torch.utils.data import TensorDataset, DataLoader
import signatory
from src.omni.decorators import timeit
from src.data.functions import pytorch_rolling
from src.features.signatures.augmentations import apply_augmentation_list


class RollingSignature:
    """Computes signatures from rolling windows.

    Given input paths of shape [N, L, C] and a window size W. This computes the signatures over every possible interval
    of length W in the path length dimension (dim=1).

    Note:
        The way this is computed is to do the following:
            1. Convert the [N, L, C] tensor onto [N, L, C, W] where W is the size of the rolling window so that now each
            index corresponds to a rolling window.
            2. Reshape to get a tensor [N * L, W, C].
            3. Apply augmentations to get shape [N * L, W_new, C_new] (as the augmentations can change the shape of the
            {W, C} dimensions.
            4. Compute the signatures to get shape [N * L, Sig_dim].
            5. Reshape to get [N, L, Sig_dim.
        One can see that if the matrix is already large, the expansion to shape [N, L, C, W] makes the matrix `W` times
        larger. When augmentations are applied (especially 'leadlag') it will become larger still. So one *must* be
        careful of memory errors here. To alleviate this we have added an argument `n_batches` that will compute the
        signatures in batches to help with memory issues. If `n_batches=1` fails, consider increasing this argument.

    Example:
        If the path is of shape [1, 5, 2] and W = 3 then we compute the signatures of each of:
            path[1, 0:3, 2], path[1, 1:4, 2], path[1, 2:5, 2]
        and then stack together. We also stack an additional two nan rows on top so the output size is also of shape
        [1, 5, *], the first two rows corresponding to the fact we cannot have a length 3 look-back window over the
        first two indexes.
    """
    def __init__(self, window, depth, logsig=False, aug_list=None, nanfill=True, n_batches=2, device=None):
        """
        Args:
            window (int): Length of the rolling window.
            depth (int): Signature depth.
            logsig (bool): Set True for a logsignature computation.
            aug_list (list): List of augmentations to apply to the signature before computation e.g. ['addtime', 'lead
                lag'] will apply the 'addtime' transform followed by 'leadlag' followed by signature computation. For
                a full list of augmentations please check src.augmentations.signatures.augmentations
            nanfill (bool): Set True to fill nan values with 0. If this is not set, nan values effectively prevent the
                signature from being computed due to limitations of signatory.
            n_batches (int): If specified, will put the data onto a dataloader with the specified number of batches and
                compute each sequentially. This is to help alleviate memory errors that can be cause by the increases in
                dimension.
            device (torch.device): A device to pass the torch tensors over to. If this is specified it will first
                extract the device the tensor originally lived on, pass over to the new device, and pass back once
                computation has completed.
        """
        self.window = window
        self.depth = depth
        self.logsig = logsig
        self.aug_list = aug_list
        self.nanfill = nanfill
        self.n_batches = n_batches
        self.device = device

    @timeit
    def transform(self, data):
        # Handle device
        original_device = data.device
        if self.device is not None:
            data = data.to(self.device)

        # Nanfill
        if self.nanfill:
            data[torch.isnan(data)] = 0

        # Path info
        N, L, C = data.shape[0], data.shape[1], data.shape[2]

        # Get signature function
        sig_func = signatory.logsignature if self.logsig else signatory.signature

        # Setup a dataloader and iterate
        signatures = []
        dataloader = DataLoader(TensorDataset(data), batch_size=int(N / self.n_batches), shuffle=False, drop_last=False)
        for batch in dataloader:
            batch = batch[0]

            # Use a rolling window and restack to be of shape [N_batch * L, W, C]
            rolled = pytorch_rolling(batch, 1, self.window).reshape(batch.size(0) * L, C, self.window).transpose(1, 2)

            # Apply a list of augmentations to each window
            augmented = apply_augmentation_list(rolled, self.aug_list)

            # Compute the signatures and reshape back to [N, L, Num sig channels]
            signatures.append(sig_func(augmented, self.depth))

        signatures = torch.cat(signatures).view(N, L, -1)

        # Revert to original device
        if self.device is not None:
            signatures.to(original_device)

        return signatures
