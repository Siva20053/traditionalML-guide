import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split

X,y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=69)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=69)


model = LinearRegression(lr=0.001,n_iters=2000)
model.fit(X_train,y_train)

predictions = model.predict(X_test)

def rmse(y_test,predictions):
    return np.sqrt(np.mean((y_test-predictions)**2))




rmse = rmse(y_test,predictions)

print('rmse --> ',rmse)

regression_line = model.predict(X)

plt.scatter(X,y)
plt.plot(X,regression_line,color='red')
plt.show()


