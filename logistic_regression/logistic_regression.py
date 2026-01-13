import numpy as np

class LogisticRegressionScratch:
    def __init__(self,lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self,x,y):
        m,n = x.shape
        self.w = np.zeros(n)
        self.b = 0
        for _ in range(self.n_iters):
            z = np.dot(x, self.w) + self.b
            y_pred = self.sigmoid(z)

            dw = (1/m) * np.dot(x.T, (y_pred-y))
            db = (1/m) * np.sum(y_pred - y)

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db


    def predict_proba(self,x):
        z = np.dot(x, self.w) + self.b
        return self.sigmoid(z)

    def predict(self,x):
        y_hat = self.predict_proba(x)
        return (y_hat >= 0.5).astype(int)
    



