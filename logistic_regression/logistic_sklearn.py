import numpy as np
from sklearn.linear_model import LogisticRegression

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

model = LogisticRegression()
model.fit(X_binary, y_binary)
predictions = model.predict(X_binary)
print("Predicted labels:", predictions)


import matplotlib.pyplot as plt

# Scratch model boundary
x_vals = np.linspace(0, 6, 100)
y_vals_scratch = -(model.coef_[0]*x_vals + model.intercept_[0]) / model.coef_[1]

# Sklearn boundary
w_sk = model.coef_[0]
b_sk = model.intercept_[0]
y_vals_sk = -(w_sk[0]*x_vals + b_sk) / w_sk[1]

plt.scatter(X_binary[y_binary==0][:,0], X_binary[y_binary==0][:,1], color="red", label="Class 0")
plt.scatter(X_binary[y_binary==1][:,0], X_binary[y_binary==1][:,1], color="blue", label="Class 1")

plt.plot(x_vals, y_vals_scratch, "k--", label="Scratch Boundary")
plt.plot(x_vals, y_vals_sk, "g-", label="Sklearn Boundary")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Scratch vs Sklearn Logistic Regression")
plt.show()
