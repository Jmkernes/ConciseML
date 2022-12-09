import scipy.stats
import numpy as np


def entropy(p):
    if p.size > 0:
        p = p[np.where(p)]
        return -(p * np.log(p)).sum()
    return 0


def discrete_entropy(arr):
    return entropy(np.bincount(arr) / len(arr))


def get_split(y, n=None, left_arr=None, right_arr=None):
    n = len(y) - 1 if n is None else n
    if n <= 0:
        return float("inf"), 0
    avg_entropy = (
        len(y[n:]) * discrete_entropy(y[n:]) + n * discrete_entropy(y[:n])
    ) / len(y)
    next_entropy, idx = get_split(y, n - 1, left_arr, right_arr)
    return (next_entropy, idx) if next_entropy < avg_entropy else (avg_entropy, n)


class DecisionTree:
    def __init__(self, max_depth: int = 3):
        self.left = self.right = self.feature = self.value = self.label = None
        self.max_depth = max_depth

    def fit(self, X, y, depth: int = 0):
        self.feature = np.random.randint(X.shape[-1])
        ids = np.argsort(X[:, self.feature])
        new_entropy, n = get_split(y[ids])
        if depth < self.max_depth and new_entropy < discrete_entropy(y):
            # take the midpoint between value and value - 1
            self.value = X[ids[max(n - 1, 0) : n + 1], self.feature].mean()
            self.left = DecisionTree(max_depth=self.max_depth).fit(
                X[ids[:n]], y[ids[:n]], depth=depth + 1
            )
            self.right = DecisionTree(max_depth=self.max_depth).fit(
                X[ids[n:]], y[ids[n:]], depth=depth + 1
            )
        else:
            self.label = np.argmax(np.bincount(y))
        return self

    def predict(self, X):
        if len(X.shape) == 2:
            return np.apply_along_axis(self.predict, 1, X)
        if self.label is not None:
            return self.label
        if X[self.feature] >= self.value:
            return self.right.predict(X)
        return self.left.predict(X)

    def __repr__(self):
        return f"Tree(feature={self.feature}, value={self.value}, label={self.label})"


class RandomForest:
    def fit(self, X, y, n_trees: int = 20, max_depth: int = 6):
        self.trees = [
            DecisionTree(max_depth=max_depth).fit(X, y) for _ in range(n_trees)
        ]
        self.classes_ = y.max() + 1
        return self

    def predict(self, X):
        return scipy.stats.mode(
            np.vstack([tree.predict(X) for tree in self.trees]), 0, keepdims=False
        ).mode
