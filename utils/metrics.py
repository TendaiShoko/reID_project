import numpy as np
import torch
from sklearn.metrics import average_precision_score

def compute_accuracy(outputs, labels):
    """
    Compute the accuracy of the model predictions.

    Args:
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): True labels.

    Returns:
        accuracy (float): Accuracy of the model predictions.
    """
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels.data)
    accuracy = corrects.double() / len(labels)
    return accuracy.item()

def compute_mAP(distmat, query_ids, gallery_ids, query_cams, gallery_cams):
    """
    Compute mean Average Precision (mAP) for re-identification.

    Args:
        distmat (numpy.ndarray): Distance matrix of shape (num_queries, num_gallery).
        query_ids (numpy.ndarray): Query identities.
        gallery_ids (numpy.ndarray): Gallery identities.
        query_cams (numpy.ndarray): Query camera ids.
        gallery_cams (numpy.ndarray): Gallery camera ids.

    Returns:
        mAP (float): Mean Average Precision.
    """
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis]).astype(np.int32)

    # Compute AP for each query
    aps = []
    for i in range(m):
        # Remove gallery samples from the same camera as the query
        valid = gallery_cams[indices[i]] != query_cams[i]
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        aps.append(average_precision_score(y_true, y_score))

    if len(aps) == 0:
        return 0.0
    return np.mean(aps)

def compute_cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams, topk=100):
    """
    Compute Cumulative Matching Characteristic (CMC) for re-identification.

    Args:
        distmat (numpy.ndarray): Distance matrix of shape (num_queries, num_gallery).
        query_ids (numpy.ndarray): Query identities.
        gallery_ids (numpy.ndarray): Gallery identities.
        query_cams (numpy.ndarray): Query camera ids.
        gallery_cams (numpy.ndarray): Gallery camera ids.
        topk (int): Top-k ranks to compute.

    Returns:
        cmc_scores (numpy.ndarray): CMC scores.
    """
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis]).astype(np.int32)

    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Remove gallery samples from the same camera as the query
        valid = gallery_cams[indices[i]] != query_cams[i]
        if not np.any(matches[i, valid]):
            continue
        index = np.nonzero(matches[i, valid])[0]
        delta = 1. / len(index)
        for j, k in enumerate(index):
            if k < topk:
                ret[k:] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        return None
    return ret / num_valid_queries
