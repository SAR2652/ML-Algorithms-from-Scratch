import tensorflow as tf

class Layer:
    @staticmethod
    def ReLU(x):
        return tf.math.maximum(0, x)

    @staticmethod
    def Sigmoid(x):
        return tf.divide(1, 1 + tf.exp(-x))

    @staticmethod
    def Softmax(x):
        return tf.divide(tf.exp(x) / tf.reduce_sum(tf.exp(x)))

    @staticmethod
    def tanh(x):
        return tf.divide(tf.exp(tf.math.scalar_mul(2, x)) - 1, tf.exp(tf.math.scalar_mul(2, x)) + 1)

    activation_dict = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'softmax': Softmax
    }

    def __init__(self, N, activation = 'relu', use_bias = True, kernel_initializer = 'normal', bias_initializer = 'normal'):
        self.prev = None
        self.next = None
        self.W = None
        self.b = None
        self.activation = activation_dict[activation]

class Dense(Layer):
    def __init__(self):
        super(Layer, self).__init__(self)

    def forward(self, x):
        z = tf.matmul(tf.transpose(self.W), x)
        return self.activation(z)

class SequentialModel:
    def __init__(self):
        self.start = None

    def add(layer):
        if self.start == None:
            self.start = layer
        else:
            ptr = self.start
            while ptr.next != None:
                ptr = ptr.next
            ptr.next = layer
            layer.prev = ptr

    def compile(self, optimizer: str, loss: str, metrics: list):
        pass

    def fit(self, X, y):

        ptr = self.start
        with tf.GradientTape(persistent = True) as tape:
            while ptr != None:
                
            
