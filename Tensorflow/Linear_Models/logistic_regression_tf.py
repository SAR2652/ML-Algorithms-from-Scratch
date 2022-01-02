import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, n_iter = 100, learning_rate = 0.003, verbose = False):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_samples = None
        self.n_features = None
        self.verbose = verbose
        self.cost_history = []
        self.weight_history = []
        self.bias_history = []
        self.accuracy_history = []
        self.w = None
        self.b = None

    def plot_cost_values(self):
        if not self.cost_history:
            sys.exit('Model has not been trained on any input yet!')
        plt.title('Change in Cost Function')
        plt.xlabel('Iterations')
        plt.ylabel('Cost Function')
        plt.plot([i for i in range(1, self.n_iter + 1)], self.cost_history)
        plt.show()

    def plot_weights(self):
        fig, axes = plt.subplots(self.n_features // 2, 2, figsize = (10, 10))
        plt.subplots_adjust(wspace = 0.5, hspace = 0.4)
        for i in range(1, self.n_features + 1):
            r = (i - 1) // (self.n_features // 2)
            c = (i - 1) % (self.n_features // 2)
            axes[r][c].set_title('Feature {}'.format(i))
            axes[r][c].set_xlabel('Iterations')
            axes[r][c].set_ylabel('Value')
            axes[r][c].plot([i for i in range(1, self.n_iter + 1)], self.weight_history[:, r * 2 + c])
        plt.show()

    def plot_bias(self):
        if not self.bias_history:
            sys.exit('Model has not been trained on any input yet!')
        plt.title('Change in Bias')
        plt.xlabel('Iterations')
        plt.ylabel('Bias')
        plt.plot([i for i in range(1, self.n_iter + 1)], self.bias_history)
        plt.show()

    def plot_accuracy(self):
        if not self.accuracy_history:
            sys.exit('Model has not been trained on any input yet!')
        plt.title('Change in Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.plot([i for i in range(1, self.n_iter + 1)], self.accuracy_history)
        plt.show()

    def sigmoid(self, x):
        return tf.divide(1, 1 + tf.exp(-x))

    def predict(self, X):
        X_tensor = tf.convert_to_tensor(X, dtype = tf.float32)
        if X.shape[1] != self.n_features:
            sys.exit('Expected {} features but got {}'.format(self.features, X.shape[1]))
        X_tensor_transpose = tf.transpose(X_tensor)
        b_np = [self.b.numpy().tolist()[0][0]] * X.shape[0]
        b_reshaped = tf.Variable(b_np)
        return self.sigmoid(tf.add(tf.matmul(tf.transpose(self.w), X_tensor_transpose), b_reshaped))

    def score(self, X_test, y_test):
        """Formula for Accuracy:
        Accuracy = (True Positives + True Negatives) / (No. of Training Samples)"""
        if X_test.shape[0] != y_test.shape[0]:
            sys.exit('Number of samples for X and y does not match!')
        y_pred_raw = self.predict(X_test)
        y_pred = tf.where(y_pred_raw > 0.5, 1, 0)
        y_pred_int = tf.cast(y_pred, tf.int32)
        y_test_tensor = tf.cast(tf.convert_to_tensor(y_test), tf.int32)
        bools = tf.equal(y_pred_int, y_test_tensor)
        int_bools = tf.where(bools == True, 1, 0)
        accuracy = tf.reduce_sum(int_bools) / X_test.shape[0]
        return accuracy

    def fit(self, X, y):
        del self.cost_history[:]
        if isinstance(self.weight_history, np.ndarray):
            self.weight_history = self.weight_history.tolist()
        del self.weight_history[:]
        del self.bias_history[:]
        del self.accuracy_history[:]
        X_tensor = tf.convert_to_tensor(X, dtype = tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype = tf.float32)
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.w = tf.Variable(tf.random.normal((self.n_features, 1)))
        self.b = tf.Variable(tf.random.normal((1, self.n_samples)))
        X_tensor_transpose = tf.transpose(X_tensor)
        y_tensor_transpose = tf.transpose(y_tensor)

        for i in range(self.n_iter):
            self.weight_history.append(self.w.numpy().ravel())
            self.bias_history.append(self.b.numpy()[0][0])

            with tf.GradientTape(persistent = True) as tape:
                z = tf.add(tf.matmul(tf.transpose(self.w), X_tensor_transpose), self.b)
                a = self.sigmoid(z)
                J = tf.reduce_sum(-1 * (y_tensor_transpose * tf.math.log(a) + y_tensor_transpose * tf.math.log(1 - a))) / self.n_samples
                self.cost_history.append(J)
                accuracy = self.score(X_tensor, y_tensor)
                self.accuracy_history.append(accuracy)
                if self.verbose:
                    print("Cost for Iteration {} = {}".format(i + 1, J))
                    print("Accuracy at Iteration {} = {}".format(i + 1, accuracy))
            
            [dw, db] = tape.gradient(J, [self.w, self.b])
            self.w.assign_sub(self.learning_rate * dw)
            self.b.assign_sub(self.learning_rate * db)

        self.weight_history = np.array(self.weight_history)
