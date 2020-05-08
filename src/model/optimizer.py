"""
optimizer.py
=================================
Optimization functions for the sepsis utility score.

The sepsis challenge returned scores according to a pre-defined utility function. Functions here are to help in choosing
prediction cutoff thresholds so as to best optimise our score on this utility function.
"""
from definitions import *
from multiprocessing.pool import Pool
from nevergrad.optimization import optimizerlib
from nevergrad import instrumentation as inst
import torch


class CVThresholdOptimizer:
    """Optimises the cutoff threshold for each cross validation fold.

    Given cross validation indexes, and predictions made over the whole dataset. This `optimize` method of this class
    takes the CV folds in turn, optimizes the threshold on the training fold and returns the score achieved with this
    threshold on the test set of that fold.
    """
    def __init__(self, labels, predictions, verbose=1, budget=200, num_workers=1):
        """
        Args:
            labels (torch.Tensor): The True labels.
            predictions (torch.Tensor): Full predictions array.
            verbose (int): Verbosity level.
            budget (int): Budget of the optimizer.
            num_workers (int): Num parallel optimization jobs.
        """
        assert len(labels) == len(predictions), 'predictions and labels are not the same length. They must both ' \
                                                'correspond to the full dataset with aligned indexes.'
        self.labels = labels
        self.predictions = predictions
        self.scores = load_pickle(DATA_DIR + '/processed/labels/full_scores.pickle').values

        # Nevergrad options
        self.budget = budget
        self.num_workers = num_workers

        # Printing
        self.verbose = verbose

    def optimize_fold(self, train_idx, test_idx):
        """ Performs optimization for a single fold in the cross validation. """
        thresh = optimize_utility_threshold(
            self.predictions[train_idx], self.scores[train_idx], budget=self.budget, num_workers=self.num_workers
        )

        # Apply the threshold to the test set
        test_score = compute_utility_from_indexes(self.predictions[test_idx], thresh, scores=self.scores[test_idx])

        if self.verbose > 0:
            print('\tScore on cv fold: {:.3f}'.format(test_score))

        return test_score

    def optimize(self, cv, parallel=False):
        """ Optimizes each CV fold and returns the scores achieved on each test fold. """
        scores = parallel_cv_loop(self.optimize_fold, cv, parallel=parallel)
        return scores


def optimize_utility_threshold(predictions, scores=None, idxs=None, budget=200, num_workers=1):
    """Optimizes the cutoff threshold to maximise the utility score.

    Sepsis predictions must be binary labels. Dependent on where these are in a patients time-series, we achieve a
    different utility score for that prediction. Our current methodology involves regressing against the utility
    function such that our output predictions are number in $\mathbb{R}$ with the expectation that larger values
    correspond to a higher likelihood of sepsis. To convert onto binary predictions we must choose some cutoff value,
    `thresh` with which to predict 1 (sepsis) if `pred > thresh` else 0. Given a set of predictions, this function
    optimizes that `thresh` value such that we would achieve the maximum utility score. This `thresh` value can now be
    used as the final step in the full model.

    This function would take a huge amount of time to compute if we did not compute for every patient, the utility
    of a zero or of a one prediction at each time-point in their series. The downside of this is that we must always
    specify the indexes from the full dataset from which the predictions correspond to, as we load in this precomputed
    scores tensor and query at the indexes to get the score.

    Args:
        predictions (torch.Tensor): The predictions (or a subset of the predictions) on the data. NOTE: if idxs is
            specified, predictions must be specified as the subset of predictions corresponding to those indexes. That
            is predictions[idxs].
        idxs (torch.Tensor): The index locations of the predictions in the full dataset.
        budget (int): The number optimizer iterations.
        num_workers (int): The number of parallel workers in the optimization.

    Returns:
        float: The estimation of the optimal cutoff threshold for which to predict sepsis.
    """
    # Load the full version of the scores
    scores = load_pickle(DATA_DIR + '/processed/labels/full_scores.pickle').values

    if idxs is not None:
        scores = scores[idxs]

    # Set optimizer and instrumentation (bounds)
    instrum = inst.Instrumentation(*[inst.var.Array(1).asscalar().bounded(-0.2, 0.2)])
    optimizer = optimizerlib.TwoPointsDE(instrumentation=instrum, budget=budget, num_workers=num_workers)

    # Optimize
    recommendation = optimizer.optimize(
        lambda thresh: -compute_utility(scores, predictions, thresh)
    )

    # Get the threshold and return the score
    threshold = recommendation.args[0]

    return threshold


def compute_utility_from_indexes(predictions, thresh, scores=None, idxs=None):
    """Given a prediction subset of the full data, and its index locations in the full data, returns the utility score.

    Args:
        predictions (torch.Tensor): The predictions made at the idxs.
        thresh (float): Threshold cutoff.
        scores (torch.Tensor): Can specify as the full scores array if already loaded from disk. Otherwise will reload
            from disk.
        idxs (torch.Tensor): Tensor/array of index locations of predictions in the full scores array.

    Returns:
        float: The normalised utility score.
    """
    # Load the full version of the scores
    if scores is None:
        scores = load_pickle(DATA_DIR + '/processed/labels/full_scores.pickle').values

    # Check correct size
    assert len(idxs) == len(predictions), "idxs and predictions have a different size. idxs must correspond to the " \
                                          "location of the predictions in the full dataset, and predictons must " \
                                          "be that subset of predictions. You may need to have predictions[idxs] as the" \
                                          "predictions input of this function."

    # Get the corresponding indexes
    scores = scores[idxs] if idxs is not None else scores

    # Get normalised utility
    normalised_utility = compute_utility(scores, predictions, thresh)

    return normalised_utility


def compute_utility(scores, predictions, thresh):
    """Computes the utility score of the predictions given the scores for predicting 0 or 1 at each timepoint.

    Args:
        scores (torch.Tensor): The scores tensor that gives the score of a 0 or 1 at each timepoint.
        predictions (torch.Tensor): The predictions with indexed aligned with the score indexes.
        thresh (float): The threshold to cut the predictions off at.

    Returns:
        float: The normalised score.
    """
    # Precompute the inaction and perfect scores
    inaction_score = scores[:, 0].sum()
    perfect_score = scores[:, [0, 1]].max(axis=1).sum()

    # Apply the threshold
    predictions = (predictions > thresh).astype(int)

    # Get the actual score
    actual_score = scores[:, 1][predictions == 1].sum() + scores[:, 0][predictions == 0].sum()

    # Get the normalized score
    normalized_score = (actual_score - inaction_score) / (perfect_score - inaction_score)

    return normalized_score.item()


def parallel_cv_loop(func, cv, parallel=True):
    """
    Performs a parallel training loop over the cv train_idx and test_idxs.

    Example:
        - func will usually be a class that contains df, labels info but __call__ method will run a single training loop
        given train_idx, test_idx
        - This will run func.__call__(train_idx, test_idx) for each idx pair in cv and return results

    Args:
        func (object): Class that has information relating to data, labels and takes a __call__(train_idx, test_idx) to
                       run loop.
        cv (list): List of [(train_idx, test_idx), ...] pairs.
        give_cv_num (bool): Gives the cv num to the underlying function, used when using the full dataset and loading
                            precomputed arrays for a specific cv_num
        parallel (bool): Set to false for a for loop (allows for debugging)

    Return:
        (list): A list of whatever func outputs for each cv idxs.
    """
    if parallel:
        pool = Pool(len(cv))
        results = pool.starmap(
            func, cv
        )
        pool.close()
    else:
        results = []
        for args in cv:
            results.append(func(*args))

    return results