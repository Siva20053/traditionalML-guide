import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest

data = load_digits()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

model = RandomForest(n_trees=14, max_depth=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = np.sum(y_pred == y_test)/len(y_test)

print("Accuracy-->",accuracy*100)