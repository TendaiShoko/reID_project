import torch
import numpy as np
from sklearn.metrics import average_precision_score


def compute_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).float().mean()


def compute_cmc(distmat, query_ids, gallery_ids, topk=50):
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis]).astype(np.int32)

    cmc = matches[:, :topk].cumsum(axis=1)
    cmc[cmc > 1] = 1
    return cmc.mean(axis=0)


def compute_map(distmat, query_ids, gallery_ids):
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis]).astype(np.int32)

    aps = []
    for i in range(m):
        y_true = matches[i]
        y_score = -distmat[i][indices[i]]
        ap = average_precision_score(y_true, y_score)
        aps.append(ap)

    return np.mean(aps)
