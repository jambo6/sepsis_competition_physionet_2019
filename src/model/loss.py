import numpy as np


def sigmoid(x):
    """ Numerically stable implementation of the sigmoid function. """
    answer = np.where(x >= 0,
                      1. / (1. + np.exp(-x)),
                      np.exp(x) / (1. + np.exp(x)))
    return answer


def weighted_log_likelihood(labels, preds, weights):
    """A weighted version of the log likelihood for lgbm.

    We wish to optimise the following loss:
        Loss = argmin \sum_{s \in S} \sum_{t=1}^T_s (UD_{s, t} y \hat{y} - UD_{s, t} (1 - y) (1 - \hat{y}))
    where UD_{s, t} is the utility difference for patient s at time t. That is:
        UD_{s, t} = U^p_{s, t} - U^n_{s, t}.

    Args:
        labels (np.array): True labels.
        preds (np.array): Predictions.
        weights (np.array): Weights to be applied to each label.

    Returns:
        tuple: A tuple containing the gradient and the hessian.
    """
    preds = sigmoid(preds)
    grad = 2 * weights * preds * labels - weights * (labels + preds)
    hess = (2 * weights * labels - weights) * preds * (1 - preds)
    return grad, hess
