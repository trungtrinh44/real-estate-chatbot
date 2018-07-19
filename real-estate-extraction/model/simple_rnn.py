import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops, rnn_cell_impl


class SRUCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, activation=None, reuse=False):
        super(SRUCell, self).__init__(_reuse=reuse)
        self.num_units = num_units
        self.activation = activation

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, state):
        f = tf.slice(inputs, [0, self.num_units], [-1, self.num_units])
        x = tf.slice(inputs, [0, 0], [-1, self.num_units])
        c = math_ops.multiply(f, state) + math_ops.multiply(1-f, x)
        h = self.activation(c) if self.activation else c
        return h, c