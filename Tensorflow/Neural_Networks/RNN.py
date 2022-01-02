import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class RecurrentLayer:
    def __init__(self):
        self.U = None
        self.W = None
        self.n_samples = None
        self.n_features = None

    def ReLU(self, x):
        return 0 if x == 0 else x

    def fit(self, X, y, batch_size = 32):
        X_tensor = tf.convert_to_tensor(X)
        X_tensor_transpose = tf.transpose(X_tensor)
        y_tensor = tf.convert_to_tensor(y)
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        U = tf.random.normal((X.shape[1], 1))
        W = tf.random.normal((X.shape[1], 1))
        h_old = tf.zeros()
        h_new = self.ReLU(tf.add(tf.matmul(U, h_old), tf.matmul(W, X_tensor_transpose)))
        
