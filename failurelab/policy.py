import numpy as np
from .ranking import ranking_metrics

def validate_scores(y, scores):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("Labels and scores must be nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Scores must be finite and between zero and one")
    return y.astype(int), scores

def operating_point(y, scores, threshold, miss_cost=25.0, false_alarm_cost=1.0):
    y, scores = validate_scores(y, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1.0, 2.0):
        raise ValueError("Invalid threshold")
    if not np.isfinite([miss_cost, false_alarm_cost]).all() or min(miss_cost, false_alarm_cost) <= 0:
        raise ValueError("Costs must be finite and positive")
    alert = scores >= threshold
    tp = int(((y == 1) & alert).sum())
    fp = int(((y == 0) & alert).sum())
    fn = int(((y == 1) & ~alert).sum())
    tn = int(((y == 0) & ~alert).sum())
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "alert_rate": float(alert.mean()),
            "cost": float(miss_cost * fn + false_alarm_cost * fp),
            "cost_per_1000": float((miss_cost * fn + false_alarm_cost * fp) * 1000 / len(y))}

def choose_threshold(y, scores, miss_cost=25.0, false_alarm_cost=1.0):
    y, scores = validate_scores(y, scores)
    candidates = np.r_[0.0, np.unique(scores), np.nextafter(1.0, 2.0)]
    points = [operating_point(y, scores, t, miss_cost, false_alarm_cost) for t in candidates]
    # Equal costs: use fewer alerts, then the higher threshold; ties are never split.
    return min(points, key=lambda p: (p["cost"], p["alert_rate"], -p["threshold"]))

def metrics(y, scores, threshold=0.5, miss_cost=25.0):
    y, scores = validate_scores(y, scores)
    return {**ranking_metrics(y, scores),
            **operating_point(y, scores, threshold, miss_cost)}

def block_bootstrap(y, scores, threshold, miss_cost=25.0, repeats=500, block_size=100, seed=123):
    y, scores = validate_scores(y, scores)
    if repeats < 1 or block_size < 1:
        raise ValueError("Invalid bootstrap configuration")
    blocks = [np.arange(i, min(i + block_size, len(y))) for i in range(0, len(y), block_size)]
    rng = np.random.default_rng(seed)
    draws = {"average_precision": [], "recall": [], "cost_per_1000": []}
    for _ in range(repeats):
        ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if len(np.unique(y[ix])) < 2:
            continue
        m = metrics(y[ix], scores[ix], threshold, miss_cost)
        for key in draws:
            draws[key].append(m[key])
    if not draws["average_precision"]:
        raise ValueError("No bootstrap sample contains both classes")
    return {"method": "non-overlapping ordered-block percentile bootstrap",
            "block_size": block_size, "requested_repeats": repeats,
            "valid_repeats": len(draws["average_precision"]),
            "scope": "Test-sample variation conditional on fitted model and selected threshold",
            "intervals_95": {key: np.quantile(vals, [0.025, 0.975]).tolist()
                             for key, vals in draws.items()}}
