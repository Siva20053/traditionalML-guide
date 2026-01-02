import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('linear_data.csv')

def loss_function(m,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Study_Time
        y = points.iloc[i].Score
        total_error += (y - (m*x + b))**2
    return  total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    for i in range(len(points)):
        x = points.iloc[i].Study_Time
        y = points.iloc[i].Score
        m_gradient = -(2/len(points)) * x * (y -(m_now*x +b_now))
        b_gradient = -(2/len(points)) * (y -(m_now*x +b_now))

    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return m,b

m = 0
b= 0
L = 0.01
epochs = 3000
for i in range(epochs):
    m,b = gradient_descent(m,b,data,L)
    if i % 50 == 0:
        print(f"Epoch {i}: Loss = {loss_function(m,b,data)}")


plt.scatter(data.Study_Time, data.Score, color='blue')
regression_line = [m*x + b for x in data.Study_Time]
plt.plot(data.Study_Time, regression_line, color='red')
plt.show()
print(m,b)