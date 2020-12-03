import numpy as np
import matplotlib.pyplot as plt

# Generate an input matrix of 1000 rows and 4 columns/features
N = 4
X_ip = 0.07 * np.random.randn(1000, N)             # dimensions = 1000 x N
y_ip = np.random.randint(0, 2, (1000, 1))     # dimensions = 1000 x 1

# Take transpose of feature and target matrices for ease of computation
X = np.transpose(X_ip)      # dimensions = N x 1000
y = np.transpose(y_ip)      # dimensions = 1 x 1000

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
print("X: \n{}".format(X))    # dimensions = (N + 1) x 1000
print("y: \n{}".format(y))    # dimensions = 1 x 1000

# Store these values for plotting after computation
cost_history = []
weights_history = []
accuracy_history = []

def sigmoid(z):
    return 1 / (1 + np.exp((-1) * z))

def hypothesis(X, weights):
    """
    Dimensionality Analysis:
    Dot Product of Transpose(Weights) & X = (1 x (N + 1)) . ((N + 1) x m) = (1 x m)
    Returns an array with dimensions (1 x m)"""
    return sigmoid(np.dot(np.transpose(weights), X))

def cost_function(X, y, m, weights):
    """
    Dot Product of y & Transpose(log(hypothesis)) = (1 x m) . (m x 1) = 1 x 1 <- Scalar
    Dot product of (1 - y) & Transpose(log(1 - hypothesis)) = (1 x m) . (m x 1) = 1 x 1 <- Scalar
    Returns Scalar
    """
    return (-1) * np.asscalar(np.dot(y, np.transpose(np.log(hypothesis(X, weights)))) - np.dot((1 - y), np.transpose(np.log(1 - hypothesis(X, weights))))) / m

def cost_function_derivative(X, y, m, weights):
    """
    Dot Product of (hypothesis - y) & Transpose(X) = (1 x m) . (m x (N + 1)) = (1 x (N + 1))
    """
    return np.dot(hypothesis(X, weights) - y, np.transpose(X)) / m

def accuracy(X, y, m, weights):
    """Formula for Accuracy:
    Accuracy = (True Positives + True Negatives) / (No. of Training Samples)
    """
    y_pred_raw = hypothesis(X, weights)
    y_pred_raw = np.where(y_pred_raw >= 0.5, y_pred_raw, 0) # Apply opposite condition
    y_pred = np.where(y_pred_raw < 0.5, y_pred_raw, 1)
    unique, counts = np.unique(y == y_pred, return_counts = True)
    count_dict = dict(zip(unique, counts))
    acc = 0
    if True in count_dict.keys():
        acc = count_dict[True] / m
    return acc


def gradient_descent(X, y, m, learning_rate, weights):
    """Returns Array with dimensions ((N + 1) x 1)"""
    new_weights = np.transpose(weights) - learning_rate * cost_function_derivative(X, y, m, weights)
    return np.transpose(new_weights)

n_iter = 1000
learning_rate = 0.001

def train_binary_classifier(X, y, learning_rate = 0.001, n_iter = 1000):
    m = X.shape[1]      # Number of training samples
    # Declare and initialize random
    weights = np.random.randn(N + 1, 1)     # dimensions = (N + 1) x 1
    cost = cost_function(X, y, m, weights)
    print("Initial Cost = {}".format(cost))
    for i in range(n_iter):
        weights = gradient_descent(X, y, m, learning_rate, weights)
        weights_history.append(np.transpose(weights).ravel())
        cost = cost_function(X, y, m, weights)
        cost_history.append(cost)
        print("Cost for Iteration {} = {}".format(i + 1, cost))
        score = accuracy(X, y, m, weights)
        accuracy_history.append(score)
        print("Accuracy at Iteration {} = {}".format(i + 1, score))

    return weights

weights = train_binary_classifier(X, y ,learning_rate, n_iter)

weights_history = np.array(weights_history)

plt.title('Change in Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.plot([i for i in range(1, n_iter + 1)], cost_history)
plt.show()

plt.title('Change in Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.plot([i for i in range(1, n_iter + 1)], accuracy_history)
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