# import tensorflow as tf


# def cosine(a, b):
#     d = tf.nn.l2_normalize(a, -1) * tf.nn.l2_normalize(b, -1)
#     output = tf.layers.dense(inputs=d,
#                              units=1,
#                              activation=None,
#                              kernel_initializer=tf.contrib.layers.xavier_initializer())
#     return tf.squeeze(output, -1)


# def attention(inputs,  align_fn, sequence_length, units, name="attention"):
#     with tf.variable_scope(name):
#         inputs *= tf.stop_gradient(
#             tf.tile(tf.expand_dims(tf.sequence_mask(sequence_length, dtype=tf.float32),
#                                    axis=-1), multiples=[1, 1, tf.shape(inputs)[-1]])
#         )
#         temp = tf.transpose(inputs, [1, 0, 2])
#         score = tf.map_fn(lambda a: tf.map_fn(
#             lambda b: align_fn(a, b), temp), temp)
#         score = tf.transpose(score, [2, 0, 1])
#         # t = score
#         score = tf.nn.softmax(score, -1)
#         # s = score
#         g = tf.reduce_sum(tf.expand_dims(score, -1) *
#                           tf.expand_dims(inputs, 2), 1)
#         z = tf.layers.dense(inputs=tf.concat([inputs, g], -1),
#                             units=units,
#                             activation=tf.tanh,
#                             kernel_initializer=tf.contrib.layers.xavier_initializer())
#         return z


# # if __name__ == '__main__':
# #     import numpy as np
# #     a = np.random.randint(0, 10, [2, 3, 4])
# #     print(a)
# #     with tf.Session() as sess:
# #         a = tf.constant(a, tf.float32)
# #         z,g,t,s = attention(a,lambda x,y:tf.reduce_sum(tf.square(x-y),-1),[3,3],5)
# #         print(sess.run(g))
# #         print(sess.run(t))
# #         print(sess.run(s))
#         # print(sess.run()
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops, rnn_cell_impl


class AttentionCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, prev_inputs, sequence_length, align_fn='cosine', reuse=None, name=None):
        super(AttentionCell, self).__init__(_reuse=reuse, name=name)
        self._num_units = num_units
        self._align_fn = align_fn
        self._prev_inputs = prev_inputs
        self._seq_len = sequence_length

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        self._prev_inputs *= tf.stop_gradient(
            tf.tile(tf.expand_dims(tf.sequence_mask(
                self._seq_len, dtype=tf.float32), -1), [1, 1, tf.shape(self._prev_inputs)[-1]]))
        self._prev_inputs = tf.transpose(self._prev_inputs, [1, 0, 2])
        input_depth = input_shape[1].value
        self._input_shape = input_shape
        if self._align_fn != 'luong':
            self._W_score = tf.get_variable(
                'W_score', shape=[input_depth, 1],
                initializer=tf.contrib.layers.xavier_initializer()
            )
        self._W1_outputs = tf.get_variable(
            'W1_output', shape=[input_depth, self._num_units],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self._W2_outputs = tf.get_variable(
            'W2_output', shape=[input_depth, self._num_units],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        if self._align_fn == 'cosine':
            self._prev_input_norm = tf.nn.l2_normalize(self._prev_inputs, -1)
        elif self._align_fn == 'luong':
            self._W_luong = tf.get_variable(
                'W_luong', shape=[input_depth, input_depth],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self._prev_input_luong = tf.tensordot(
                self._prev_inputs, self._W_luong, [2, 0])
        self.built = True

    def call(self, inputs, state):
        """Remember to transpose the inputs to [time, batch, channel] before using this cell"""
        if self._align_fn == 'cosine':
            score = self._prev_input_norm * tf.nn.l2_normalize(inputs, -1)
        elif self._align_fn == 'manhattan':
            score = tf.abs(self._prev_inputs - inputs)
            score = tf.reduce_max(score, 0)-score
        elif self._align_fn == 'euclidean':
            score = tf.square(self._prev_inputs - inputs)
            score = tf.reduce_max(score, 0)-score
        elif self._align_fn == 'luong':
            score = self._prev_input_luong*inputs
            score = tf.reduce_sum(score, -1)
            score = tf.expand_dims(score, -1)/tf.sqrt(tf.constant(self._input_shape.as_list()[1],tf.float32))
        # score_shape = tf.shape(score)
        if self._align_fn != 'luong':
            score = tf.tensordot(score, self._W_score, [2, 0])
        # score = tf.reshape(score, [score_shape[0], score_shape[1], 1])
        alpha = tf.nn.softmax(score, 0)
        g = tf.reduce_sum(self._prev_inputs * alpha, 0)
        z = tf.nn.relu(tf.matmul(g, self._W1_outputs) +
                       tf.matmul(inputs, self._W2_outputs))
        return z, z
