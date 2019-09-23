import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, session, input_height, input_width, output_size, name="main"):
        self.session = session
        self.input_width = input_width
        self.input_height = input_height
        # self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=200, l_rate=0.01):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_height, self.input_width], name="input_x")

            # W1 = tf.get_variable("W1", shape=[self.input_height, self.input_width, h_size],
            #                      initializer=tf.contrib.layers.xavier_initializer())
            # # b = tf.get_variable("b1", shape=[h_size],initializer=tf.contrib.layers.xavier_initializer())
            # layer1 = tf.matmul(self._X, W1)
            # flatten = tf.layers.flatten(layer1)

            nn = tf.layers.dense(self._X, h_size, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            flatten = tf.layers.flatten(nn)
            #
            # W2 = tf.get_variable("W2", shape=[h_size, 1024],
            #                      initializer=tf.contrib.layers.xavier_initializer())
            # layer2 = tf.nn.tanh(tf.matmul(layer1, W2))
            # #
            # W3 = tf.get_variable("W3", shape=[1024, 512],
            #                      initializer=tf.contrib.layers.xavier_initializer())
            # layer3 = tf.nn.tanh(tf.matmul(layer2, W3))

            # W4 = tf.get_variable("W4", shape=[256, self.output_size],
            #                      initializer=tf.contrib.layers.xavier_initializer())
            # W4 = tf.get_variable("W4", shape=[h_size, self.output_size],
            #                      initializer=tf.contrib.layers.xavier_initializer())
            self._Qpred = tf.layers.dense(flatten, self.output_size, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.hypothesis = tf.nn.softmax(self._Qpred)

            # self._Qpred = tf.matmul(layer1, W4)
            # self._Qpred = tf.matmul(layer3, W4)
        self.N = self._X.shape[0]

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        self.actions = []
        self.values = []

        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._Qpred,labels=tf.stop_gradient([self._Y])))
        # self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

        # self._loss = tf.log(self._Qpred[tf.range(self.N), self.actions],name='loss').dot(self.values) / self.N
        # self._train = tf.train.AdamOptimizer(learning_rate=l_rate,).minimize(self._loss)
        self._train = tf.compat.v1.train.AdamOptimizer(learning_rate=l_rate,).minimize(self._loss)
        # self._train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=l_rate, ).minimize(self._loss)

    def predict(self, state):
        # print(state)
        # x = np.reshape(state, [1, self.input_size])
        return self.session.run(self.hypothesis, feed_dict={self._X: state})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={
            self._X: x_stack, self._Y: y_stack
        })