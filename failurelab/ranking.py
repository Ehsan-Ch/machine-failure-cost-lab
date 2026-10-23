"""Binary ranking metrics with threshold groups (tied scores stay together)."""
import numpy as np

def curve(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    labels, values = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(values) != 0), len(values)-1]
    tp = np.cumsum(labels)[ends]
    fp = ends+1-tp
    positives, negatives = y.sum(), len(y)-y.sum()
    recall = tp/positives if positives else np.zeros(len(tp))
    precision = tp/(ends+1)
    fpr = fp/negatives if negatives else np.zeros(len(fp))
    return np.r_[0.0, recall], np.r_[1.0, precision], np.r_[0.0, fpr]

def ranking_metrics(y, scores):
    recall, precision, fpr = curve(y, scores)
    both = len(np.unique(y)) == 2
    return {"average_precision": float(np.sum(np.diff(recall)*precision[1:])) if np.sum(y) else None,
            "roc_auc": float(np.trapezoid(recall, fpr)) if both else None,
            "brier_score": float(np.mean((np.asarray(y)-scores)**2))}
