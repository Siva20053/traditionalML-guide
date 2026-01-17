import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from SVM import SVM



data = load_breast_cancer()
X = data.data
y = data.target
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=72)



model = SVM(lr=0.001, lamda=0.001, n_iters=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

accuracy = np.sum(y_pred == y_test) / len(y_test)

print("Accuracy of model-->",accuracy * 100)