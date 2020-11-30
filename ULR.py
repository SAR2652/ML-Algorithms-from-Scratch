import random
import numpy as np
import matplotlib.pyplot as plt

X = np.array([x + random.randint(-10, 10) for x in range(100)])
y = np.array([x + random.randint(-2000, 2000) for x in range(0, 10000, 100)])
print("X:")
print(X)
print("y:")
print(y)

plt.scatter(X, y)
plt.show()

def linear_equation(theta0, theta1, X):
    return theta0 + theta1 * X

def tune_theta(theta0, theta1, X, learning_rate, m, term = 'bias'):
    linear_term = linear_equation(theta0, theta1, X) - y
    theta = theta0
    if term == 'weight':
        theta = theta1
        linear_term *= X
    return theta - (learning_rate * np.sum(linear_term) / m)

def cost_function(theta0, theta1, X, y, m):
    return np.sum(np.square(linear_equation(theta0, theta1, X) - y)) / (2 * m)

def gradient_descent(theta0, theta1, learning_rate, X, m):
    temp0 = tune_theta(theta0, theta1, X, learning_rate, m)
    temp1 = tune_theta(theta0, theta1, X, learning_rate, m, term = 'weight')
    return temp0, temp1

def train_regression_model(X, y, learning_rate = 0.0001, n_iter = 10000):
    # Initialise with temporary values
    theta0 = random.randint(-100, 100)
    theta1 = random.randint(-100, 100)
    print(theta0, theta1)
    m = len(X)
    cost = cost_function(theta0, theta1, X, y, m)
    print("Initial Loss = {}".format(cost))
    for i in range(n_iter):
        theta0, theta1 = gradient_descent(theta0, theta1, learning_rate, X, m)
        new_cost = cost_function(theta0, theta1, X, y, m)
        print("Loss at Iteration {} = {}".format(i + 1, new_cost))
    return theta0, theta1

bias, weight = train_regression_model(X, y)
print("Bias = {}".format(bias))
print("Weight = {}".format(weight))

