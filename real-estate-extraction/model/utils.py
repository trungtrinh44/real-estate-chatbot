import time
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs  # pylint: disable=no-name-in-module
from tensorflow.python.ops import array_ops, rnn  # pylint: disable=no-name-in-module
from model.rnn_cell import GRUCell
def build_gru_cell(hidden):
    return GRUCell(
        hidden,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.constant_initializer(0.0, tf.float32)
    )


def stack_bidirectional_dynamic_rnn_cnn(cells_fw,
                                        cells_bw,
                                        cnn_filters,
                                        cnn_sizes,
                                        dropout,
                                        inputs,
                                        initial_states_fw=None,
                                        initial_states_bw=None,
                                        dtype=None,
                                        sequence_length=None,
                                        parallel_iterations=64,
                                        time_major=False,
                                        swap_memory=True,
                                        scope=None):
    states_fw = []
    states_bw = []
    prev_layer = inputs

    with vs.variable_scope(scope or "stack_bidirectional_rnn"):
        for i, (cell_fw, cell_bw, filters, size) in enumerate(zip(cells_fw, cells_bw, cnn_filters, cnn_sizes)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]

            with vs.variable_scope("cell_%d" % i):
                outputs, (state_fw, state_bw) = rnn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    prev_layer,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw,
                    sequence_length=sequence_length,
                    parallel_iterations=parallel_iterations,
                    dtype=dtype,
                    swap_memory=swap_memory,
                    time_major=time_major)
                # Concat the outputs to create the new input.
                prev_layer = array_ops.concat(outputs, 2)
            with vs.variable_scope("conv_%d" % i):
                prev_layer = tf.layers.conv1d(
                    inputs=prev_layer,
                    filters=filters,
                    kernel_size=size,
                    padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )
                if dropout is not None:
                    prev_layer = tf.nn.dropout(x=prev_layer, keep_prob=dropout, noise_shape=[
                                               tf.shape(prev_layer)[0], 1, prev_layer.get_shape()[-1]])
            states_fw.append(state_fw)
            states_bw.append(state_bw)

    return prev_layer, tuple(states_fw), tuple(states_bw)


def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    concat_residual=False,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    dtype=None,
                                    sequence_length=None,
                                    parallel_iterations=64,
                                    time_major=False,
                                    swap_memory=True,
                                    scope=None):
    states_fw = []
    states_bw = []
    prev_layer = inputs

    with vs.variable_scope(scope or "stack_bidirectional_rnn"):
        for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]
            with vs.variable_scope("cell_%d" % i):
                outputs, (state_fw, state_bw) = rnn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    prev_layer,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw,
                    sequence_length=sequence_length,
                    parallel_iterations=parallel_iterations,
                    dtype=dtype,
                    swap_memory=swap_memory,
                    time_major=time_major)
                if concat_residual:
                    prev_layer = array_ops.concat(outputs+(prev_layer,), 2)
                else:
                    res_con = tf.layers.conv1d(inputs=array_ops.concat(outputs, 2),
                                               filters=prev_layer.get_shape(
                    )[-1],
                        kernel_size=1,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    )
                    res_con *= tf.stop_gradient(
                        tf.tile(tf.expand_dims(tf.sequence_mask(sequence_length, dtype=tf.float32),
                                               axis=-1), multiples=[1, 1, tf.shape(res_con)[-1]])
                    )
                    prev_layer += res_con
            states_fw.append(state_fw)
            states_bw.append(state_bw)

    return prev_layer, tuple(states_fw), tuple(states_bw)


def build_lstm_layer_norm(hidden, keep_prob):
    return tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden,
                                                 forget_bias=1.0,
                                                 activation=tf.tanh,
                                                 layer_norm=True,
                                                 norm_gain=1.0,
                                                 norm_shift=0.0,
                                                 dropout_keep_prob=keep_prob,
                                                 dropout_prob_seed=None,
                                                 reuse=None
                                                 )


def build_gru_cell_with_dropout(hidden, keep_prob):
    return GRUCell(
        hidden,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.constant_initializer(0.0, tf.float32),
        use_dropout = True,
        dropout = keep_prob
    )


def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)
