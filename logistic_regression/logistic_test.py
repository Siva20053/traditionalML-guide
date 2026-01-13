import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from logistic_regression import LogisticRegressionScratch

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=69)

model = LogisticRegressionScratch(lr=0.001,n_iters=3000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

def accuracy(y_test,y_pred):
    return np.sum(y_pred == y_test)/len(y_test)

print('Accuracy-->',accuracy(y_test,y_pred))

