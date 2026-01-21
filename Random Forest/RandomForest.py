import numpy as np
from collections import Counter
from Decisiontree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_sample_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth = self.max_depth,
                min_sample_split = self.min_sample_split,
                n_features = self.n_features
            )

            X_sample, y_sample = self.bootstrap_samples(X, y)

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        tree_predictions = np.swapaxes(tree_predictions, 0, 1)

        y_preds = [self.most_common_label(sample_preds) for sample_preds in tree_predictions]
        return np.array(y_preds)