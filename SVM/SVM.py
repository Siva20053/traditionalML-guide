import numpy as np

class SVM:
    def __init__(self, lr=0.001, lamda=0.01, n_iters=1000):
        self.lr = lr
        self.lamda = lamda
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0


        for _ in range(self.n_iters):
          idx = np.arange(n_samples)
          np.random.shuffle(idx)

          for i in idx:
                x = X[i]
                y_i = y[i]
                condition = y_i * (np.dot(x, self.w) - self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lamda * self.w)

                else:
                    self.w -= self.lr * (2 * self.lamda * self.w - (y_i*x))
                    self.b -= self.lr * y[i]

    def predict(self,X):
        y_pred = np.dot(X,self.w) - self.b
        return np.sign(y_pred)