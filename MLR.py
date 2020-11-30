import numpy as np
import matplotlib.pyplot as plt

# Generate an input matrix of 1000 rows and 4 columns/features
N = 4
X_ip = 3 * np.random.randn(1000, N)
y_ip = 100 + 101 * np.random.randn(1000, 1)

# Take transpose of feature and target matrices for ease of computation
X = np.transpose(X_ip)
y = np.transpose(y_ip)

fig, axes = plt.subplots(2, 2)

# Check the plots of the 4 columns vs target variable
for i in range(X.shape[0]):
    r = int(i / (X.shape[0] / 2))
    c = int(i % (X.shape[0] / 2))
    axes[r][c].scatter(X[i, :], y)

plt.show()

# Declare an extra column of ones to deal with weights for bias \
# and stack it horizontally with the feature matrix  
ones = np.ones((1, 1000))
X = np.vstack((ones, X))
print("X: {}".format(X))

# Store these values for plotting after computation
cost_history = []
weights_history = []

def hypothesis(X, y, weights):
    """Returns an array of dimensions features 1 x m"""
    return np.dot(np.transpose(weights), X) - y

def cost_function(X, y, m, weights):
    return np.sum(np.square(hypothesis(X, y, weights))) / (2 * m)

n_iter = 10000
learning_rate = 0.0001

def gradient_descent(X, y, weights, learning_rate, m):
    """The output of the hypothesis function must be multiplied with a transpose of 
    the feature matrix X in order to restore dimensions to (1 x no. of features).
    However, the weights array has a dimension of (no. of features x 1).
    Hence, we calculate differnce of transpose of weights and the product of
    the result of the hypothesis function and transpose of feature matrix X.
    The transpose of the difference is returned to restore priginal dimensions of weights array, i.e. (features x 1)"""
    new_weights = np.transpose(weights) - learning_rate * np.dot(hypothesis(X, y, weights), np.transpose(X)) / m
    return np.transpose(new_weights)

def train_multivariate_regression_model(X, y, n_iter = 1000, learning_rate = 0.001):
    m = X.shape[1]
    # Declare and initialize random
    weights = np.random.randn(N + 1, 1)
    cost = cost_function(X, y, m, weights)
    print("Initial Cost = {}".format(cost))
    for i in range(n_iter):
        weights = gradient_descent(X, y, weights, learning_rate, m)
        weights_history.append(np.transpose(weights).ravel())
        cost = cost_function(X, y, m, weights)
        cost_history.append(cost)
        print("Cost for Iteration {} = {}".format(i + 1, cost))

    return weights

weights = train_multivariate_regression_model(X, y, n_iter, learning_rate)

weights_history = np.array(weights_history)

plt.title('Change in Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.plot([i for i in range(1, n_iter + 1)], cost_history)
plt.show()

plt.title('Change in Bias')
plt.xlabel('Iterations')
plt.ylabel('Bias')
plt.plot([i for i in range(1, n_iter + 1)], weights_history[:, 0])
plt.show()

fig, axes = plt.subplots(2, 2)
plt.subplots_adjust(hspace = 0.5)
for i in range(1, 5):
    r = int((i - 1) / (N / 2))
    c = int((i - 1) % (N / 2))
    axes[r][c].set_title('Feature {}'.format(i))
    axes[r][c].set_xlabel('Iterations')
    axes[r][c].set_ylabel('Value')
    axes[r][c].plot([i for i in range(1, n_iter + 1)], weights_history[:, i])
    
plt.show()


