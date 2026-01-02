import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('linear_data.csv')
x = data[['Study_Time']].values
y = data['Score'].values
n=len(x)

x_mean = x.mean()
x_std = x.std()

x = (x-x_mean) / x_std

def loss_function(m,b,x,y):
    total_loss = 0
    for i in range(n):
        total_loss += (y[i]-(m*x[i]+b))**2
    return total_loss/n

def gradient_descent(m,b,x,y,lr):
    m_gradient = 0
    b_gradient = 0
    for i in range(n):
        m_gradient += -(2/n) * x *(y[i]-m*x[i]-b)
        b_gradient += -(2/n) * (y[i]-m*x[i]-b)

    m = m - lr*m_gradient
    b = b - lr*b_gradient
    return m,b
m = 0
b = 0
lr = 0.001
epochs = 1000

for epoch in range(epochs):
    m,b = gradient_descent(m,b,x,y,lr)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Loss = {loss_function(m,b,x,y)}')



plt.scatter(data.Study_Time, data.Score, color='blue')
regression_line = [m*x + b for x in data.Study_Time]
plt.plot(data.Study_Time, regression_line, color='red')
plt.show()
print(m,b)
