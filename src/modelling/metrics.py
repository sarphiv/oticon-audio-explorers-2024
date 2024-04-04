import torch as th


def binary_stat_scores(y_pred: th.Tensor, y_true: th.Tensor):
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    y_true = y_true.reshape(y_true.shape[0], -1)

    tp = ((y_true == y_pred) & (y_true == 1)).sum(dim=1)
    fn = ((y_true != y_pred) & (y_true == 1)).sum(dim=1)
    fp = ((y_true != y_pred) & (y_true == 0)).sum(dim=1)
    tn = ((y_true == y_pred) & (y_true == 0)).sum(dim=1)

    return tp, fp, tn, fn


def intersection_over_union(tp: th.Tensor, fp: th.Tensor, fn: th.Tensor) -> th.Tensor:
    return tp / (tp + fp + fn)

def recall(tp: th.Tensor, fn: th.Tensor) -> th.Tensor:
    return tp / (tp + fn)

def precision(tp: th.Tensor, fp: th.Tensor) -> th.Tensor:
    return tp / (tp + fp)

def dice(tp: th.Tensor, fp: th.Tensor, fn: th.Tensor) -> th.Tensor:
    return 2 * tp / (2 * tp + fp + fn)

def accuracy(tp: th.Tensor, fp: th.Tensor, tn: th.Tensor, fn: th.Tensor) -> th.Tensor:
    return (tp + tn) / (tp + fp + fn + tn)
