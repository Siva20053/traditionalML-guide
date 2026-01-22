import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples , n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.prior_ = np.zeros(n_classes)

        for idx,c in enumerate(self.classes):
            X_c = X[y == c]

            self.mean_[idx,:] = X_c.mean(axis=0)
            self.var_[idx,:] = X_c.var(axis=0)
            self.prior_[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []

        for idx,c in enumerate(self.classes):
            prior = np.log(self.prior_[idx])
            posterior = np.sum(np.log(self.gaussian(idx, x)))

            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
    
    def gaussian(self, class_idx, x):
        mean = self.mean_[class_idx]
        var = self.var_[class_idx]

        numerator = np.exp(-(x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator