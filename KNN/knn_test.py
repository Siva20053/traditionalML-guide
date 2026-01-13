import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from knn_scratch import KNN


data = load_iris()
X = data.data
y = data.target

colors = ['blue' if val == 0 else 'red' if val == 1 else 'green' for val in y]

plt.figure()
plt.scatter(X[:,2],X[:,3],c=colors,edgecolors='k',s=20)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

model = KNN()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = np.sum(y_pred == y_test)/len(y_test)

print('Accuracy--> ',accuracy)

