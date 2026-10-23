"""Small NumPy reference estimators; not wrappers around XGBoost or scikit-learn."""
import numpy as np
from .data import SensorFeatures, NUMERIC, ENGINEERED

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))

def tree_predict(tree, bins):
    out = np.empty(len(bins), dtype=float)
    def visit(node, ix):
        if "value" in node:
            out[ix] = node["value"]
        else:
            left = bins[ix, node["feature"]] <= node["bin"]
            visit(node["left"], ix[left])
            visit(node["right"], ix[~left])
    visit(tree, np.arange(len(bins)))
    return out

def fit_tree(bins, gradient, hessian, depth=4, min_leaf=20, l2=1.0):
    """Greedy Newton tree; candidates are bins fitted on training data."""
    def grow(ix, level):
        g, h = gradient[ix].sum(), hessian[ix].sum()
        leaf = {"value": float(np.clip(-g / (h + l2), -5.0, 5.0))}
        if level >= depth or len(ix) < 2 * min_leaf:
            return leaf
        best = None
        best_gain = 1e-12
        for feature in range(bins.shape[1]):
            b = bins[ix, feature]
            count = np.bincount(b)
            if len(count) < 2:
                continue
            nl = np.cumsum(count)[:-1]
            gl = np.cumsum(np.bincount(b, weights=gradient[ix], minlength=len(count)))[:-1]
            hl = np.cumsum(np.bincount(b, weights=hessian[ix], minlength=len(count)))[:-1]
            gain = 0.5 * (gl**2 / (hl + l2) + (g-gl)**2 / (h-hl+l2) - g*g/(h+l2))
            gain[(nl < min_leaf) | (len(ix)-nl < min_leaf)] = -np.inf
            cut = int(np.argmax(gain))
            if gain[cut] > best_gain:
                best_gain = float(gain[cut])
                best = feature, cut
        if best is None:
            return leaf
        feature, cut = best
        mask = bins[ix, feature] <= cut
        return {"feature": feature, "bin": cut,
                "left": grow(ix[mask], level+1), "right": grow(ix[~mask], level+1)}
    return grow(np.arange(len(bins)), 0)

class Model:
    def __init__(self, name):
        if name not in ("prevalence", "logistic", "boosting_raw", "boosting_physics"):
            raise ValueError(f"Unknown model: {name}")
        self.name = name
        self.engineered = name == "boosting_physics"

    def matrix(self, frame):
        x = SensorFeatures(self.engineered).transform(frame)
        columns = NUMERIC + (ENGINEERED if self.engineered else [])
        return np.column_stack([x[columns].to_numpy(float),
                                (x["Type"] == "H").to_numpy(float),
                                (x["Type"] == "M").to_numpy(float)])

    def fit(self, frame, y):
        x = self.matrix(frame)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(x),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("Training requires aligned binary labels containing both classes")
        self.prevalence = float(y.mean())
        self.base = float(np.log(self.prevalence / (1-self.prevalence)))
        if self.name == "logistic":
            self.mean = x.mean(axis=0)
            self.scale = x.std(axis=0)
            self.scale[self.scale == 0] = 1
            design = np.column_stack([np.ones(len(x)), (x-self.mean)/self.scale])
            w = np.zeros(design.shape[1])
            w[0] = self.base
            penalty = np.ones(len(w))  # sum-log-loss + 0.5 * ||weights||^2
            penalty[0] = 0
            def objective(weights):
                z = design @ weights
                return float(np.sum(np.logaddexp(0, z) - y*z) + 0.5*np.sum(penalty*weights**2))
            for iteration in range(100):
                p = sigmoid(design @ w)
                gradient = design.T @ (p-y) + penalty*w
                hessian = design.T @ ((p*(1-p))[:, None]*design) + np.diag(penalty + 1e-9)
                step = np.linalg.solve(hessian, gradient)
                rate = 1.0
                old = objective(w)
                while rate > 1e-8 and objective(w-rate*step) > old:
                    rate *= 0.5
                w -= rate*step
                if np.max(np.abs(rate*step)) < 1e-8:
                    break
            self.weights = w
            self.iterations = iteration+1
        elif self.name.startswith("boosting"):
            self.cuts = [np.unique(np.quantile(x[:, j], np.linspace(0, 1, 65)[1:-1]))
                         for j in range(x.shape[1])]
            bins = self.bin_matrix(x)
            raw = np.full(len(x), self.base)
            self.rate = 0.08
            self.trees = []
            for _ in range(200):
                p = sigmoid(raw)
                tree = fit_tree(bins, p-y, np.maximum(p*(1-p), 1e-8))
                raw += self.rate*tree_predict(tree, bins)
                self.trees.append(tree)
        return self

    def bin_matrix(self, x):
        return np.column_stack([np.searchsorted(cuts, x[:, j], side="right")
                                for j, cuts in enumerate(self.cuts)])

    def predict_proba(self, frame):
        x = self.matrix(frame)
        if self.name == "prevalence":
            p = np.full(len(x), self.prevalence)
        elif self.name == "logistic":
            design = np.column_stack([np.ones(len(x)), (x-self.mean)/self.scale])
            p = sigmoid(design @ self.weights)
        else:
            bins = self.bin_matrix(x)
            raw = np.full(len(x), self.base)
            for tree in self.trees:
                raw += self.rate*tree_predict(tree, bins)
            p = sigmoid(raw)
        return np.column_stack([1-p, p])

    def to_dict(self):
        def plain(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, list):
                return [plain(v) for v in value]
            return value
        return {"format": "failurelab-numpy-v1",
                "state": {key: plain(value) for key, value in self.__dict__.items()}}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("format") != "failurelab-numpy-v1":
            raise ValueError("Unknown model format")
        state = payload["state"]
        model = cls(state["name"])
        model.__dict__.update(state)
        for key in ("mean", "scale", "weights"):
            if key in state:
                setattr(model, key, np.asarray(state[key], dtype=float))
        if "cuts" in state:
            model.cuts = [np.asarray(c, dtype=float) for c in state["cuts"]]
        return model

def make_model(name, seed=42):
    # All estimators are deterministic; seed is retained in the benchmark protocol.
    return Model(name)
