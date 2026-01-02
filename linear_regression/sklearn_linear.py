from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('linear_data.csv')
X = data[['Study_Time']]
y = data['Score']

model = LinearRegression()
model.fit(X,y)
print(model.coef_, model.intercept_)

plt.scatter(data.Study_Time, data.Score, color='blue')
regression_line = model.predict(X)
plt.plot(data.Study_Time, regression_line, color='red')
plt.show()