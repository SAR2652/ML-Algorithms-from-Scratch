import random
import numpy as np
import matplotlib.pyplot as plt

X = 3 * np.random.randn(100, 1)
y = 7 + 40 * np.random.randn(100, 1)
print("X:")
print(X)
print("y:")
print(y)

plt.scatter(X, y)
plt.show()


def hypothesis(theta0, theta1, X):
    """h(X) = theta0 + theta1 * X"""
    return theta0 + theta1 * X


def tune_theta(theta0, theta1, X, learning_rate, m, term='bias'):

    # calculate linear term
    hypothesis_term = hypothesis(theta0, theta1, X) - y

    # initialize theta parameter to be changed
    theta = theta0

    # Change in Theta1 value
    if term == 'weight':
        theta = theta1
        hypothesis_term *= X

    return theta - (learning_rate * np.sum(hypothesis_term) / m)


def cost_function(theta0, theta1, X, y, m):
    """Define the Cost Function as:
    J(Theta0, Theta1) = Sum([h(X) - y]^2) / 2m"""
    return np.sum(np.square(hypothesis(theta0, theta1, X) - y)) / (2 * m)


def gradient_descent(theta0, theta1, learning_rate, X, m):
    """Assign new values for each iteration"""
    """of the Gradient Descent Algorithm"""
    temp0 = tune_theta(theta0, theta1, X, learning_rate, m)
    temp1 = tune_theta(theta0, theta1, X, learning_rate, m, term='weight')
    return temp0, temp1


# Set to default values
n_iter = 1000
learning_rate = 0.001

# Store these values for plotting after computation
cost_history = []
theta0_history = []
theta1_history = []


def train_regression_model(X, y, learning_rate=0.0001, n_iter=10000):
    """This algorithm requires a learning rate of 0.1 or lesser due to"""
    """the values of the data taken as input. A larger learning rate can be"""
    """used for different input values"""

    # Initialise with temporary values
    theta0 = random.randint(-100, 100)
    theta1 = random.randint(-100, 100)
    print(theta0, theta1)
    m = len(X)
    cost = cost_function(theta0, theta1, X, y, m)
    print("Initial Loss = {}".format(cost))
    for i in range(n_iter):
        theta0, theta1 = gradient_descent(theta0, theta1, learning_rate, X, m)
        cost = cost_function(theta0, theta1, X, y, m)
        cost_history.append(cost)
        theta0_history.append(theta0)
        theta1_history.append(theta1)
        print("Loss at Iteration {} = {}".format(i + 1, cost))
    return theta0, theta1


bias, weight = train_regression_model(X, y, learning_rate, n_iter)
print("Bias = {}".format(bias))
print("Weight = {}".format(weight))

plt.title('Change in Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.plot([i for i in range(1, n_iter + 1)], cost_history)
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Bias')
plt.title('Change in Bias')
plt.plot([i for i in range(1, n_iter + 1)], theta0_history)
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Feature Weight')
plt.title('Change in Feature Weight')
plt.plot([i for i in range(1, n_iter + 1)], theta1_history)
plt.show()
