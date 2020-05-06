"""
compute.py
================================
Methods for computing signatures.
"""
import numpy as np
import torch
import signatory


class RollingSignature:
    """Computes signatures from rolling windows.

    Given input paths of shape [N, L, C] and a window size W. This computes the signatures over every possible interval
    of length W in the path length dimension (dim=1).

    Example:
        If the path is of shape [1, 5, 2] and W = 3 then we compute the signatures of each of:
            path[1, 0:3, 2], path[1, 1:4, 2], path[1, 2:5, 2]
        and then stack together and add some initial nan rows to illustrate a window of sufficient did not exist yet (if
        specified).
    """
    def __init__(self, window, depth, logsig=False, return_same_size=True, nanfill=True):
        """
        Args:
            window (int): Length of the rolling window.
            depth (int): Signature depth.
            logsig (bool): Set True for a logsignature computation.
            return_same_size (bool): Set True to return a signature of the same size as the input path. This is achieved
                by adding a tensor of nans of size [N, window, signature_channels]
            nanfill (bool): Set True to fill nan values with 0. If this is not set, nan values effectively prevent the
                signature from being computed due to limitations of signatory.
        """
        self.window = window
        self.depth = depth
        self.logsig = logsig
        self.return_same_size = return_same_size
        self.nanfill = nanfill

    @staticmethod
    def get_windows(path_len, window_len):
        """Gets the start and end indexes for the sliding windows.

        Args:
            path_len (int): Total length of the path.
            window_len (int): Desired window length.

        Returns:
            (list, list): List of start points and a list of end points.
        """
        end_points = np.arange(1, path_len) + 1     # +1 for zero indexing
        start_points = np.arange(1, path_len) - window_len
        start_points[start_points < 0] = 0
        return start_points, end_points

    def transform(self, data):
        # Nanfill
        if self.nanfill:
            data[torch.isnan(data)] = 0

        # Path info
        N, L, C = data.shape[0], data.shape[1], data.shape[2]

        # Full signatures
        path_class = signatory.Path(data, self.depth)

        # Logsig logic
        sig_func = path_class.logsignature if self.logsig else path_class.signature

        # Get indexes of the windows, apply the path_class signature function to each index.
        start_idxs, end_idxs = self.get_windows(L, self.window)
        signatures = torch.stack([sig_func(start, end) for start, end in zip(start_idxs, end_idxs)], dim=1)

        # Add a nan row since we cannot compute the signature of the first point
        signatures = torch.cat((float('nan') * torch.ones(N, 1, signatures.size(2)), signatures), dim=1)

        return signatures
