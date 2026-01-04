import numpy as np

class LogisticRegressionScratch:
    def __init__(self,lr = 0.01, epochs = 1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self,x,y):
        m,n = x.shape
        self.w = np.zeros(n)
        self.b = 0
        for epoch in range(self.epochs):
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
    

X_binary = np.array([
    [1, 1],
    [2, 1],
    [2, 2],
    [3, 2],
    [3, 3],
    [4, 3],
    [4, 4],
    [5, 4]
])

y_binary = np.array([0, 0, 0, 0, 1, 1, 1, 1])



model = LogisticRegressionScratch(lr=0.01 , epochs=3000)
model.fit(X_binary, y_binary)
predictions = model.predict(X_binary)
print("Predicted labels:", predictions)

