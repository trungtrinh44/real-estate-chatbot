import datetime
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs  # pylint: disable=no-name-in-module
from tensorflow.python.ops import array_ops, rnn  # pylint: disable=no-name-in-module

from data_utils.get_chunks import get_chunks
from model.base_model import BaseModel
from model.configs import from_json, Config
from model.causal_conv import bi_causal_conv
from model.QRNN import QRNN
from model.simple_rnn import SRUCell
from model.attention import AttentionCell
from model.utils import build_gru_cell, build_lstm_layer_norm, build_gru_cell_with_dropout, stack_bidirectional_dynamic_rnn_cnn, stack_bidirectional_dynamic_rnn


def load_model(path):
    model = SequenceTaggingModel(Config())
    model.sess = tf.Session()
    saver = tf.train.import_meta_graph("{}.meta".format(path))
    saver.restore(model.sess, path)
    graph = tf.get_default_graph()
    model.word_ids = graph.get_operation_by_name('word_ids').outputs[0]
    model.char_ids = graph.get_operation_by_name('char_ids').outputs[0]
    model.sequence_length = graph.get_operation_by_name(
        'sequence_length').outputs[0]
    model.word_length = graph.get_operation_by_name('word_length').outputs[0]
    model.dropout = graph.get_operation_by_name('dropout').outputs[0]
    model.state_dropout = graph.get_operation_by_name(
        'state_dropout').outputs[0]
    try:
        model.transition_params = graph.get_operation_by_name(
            'transitions').outputs[0]
        model.decode_tags = graph.get_operation_by_name(
            'decode_tags').outputs[0]
        model.best_scores = graph.get_operation_by_name(
            'best_scores').outputs[0]
        # print(model.sess.run(graph.get_operation_by_name(
        #     'word_embedding/mul_3/x').outputs[0]))
    except KeyError:
        model.label_preds = graph.get_operation_by_name(
            'label_preds').outputs[0]
    return model


def build_with_params(path, mode='train'):
    # with open(path, 'r') as file:
    #     hp = json.load(file)
    # configs = MockConfigs()
    # configs.__dict__.update(**hp)
    config = from_json(path)
    model = SequenceTaggingModel(config, mode=mode)
    if not mode == 'train':
        mock_embedding = np.zeros([config.nwords, config.wdims], dtype=float)
        model.build_model(mock_embedding)
    return model


def load_and_save_model(hp_path, weight_path, out_path, version):
    model = build_with_params(hp_path, 'infer')
    model.sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(model.sess, weight_path)
    export_path = os.path.join(
        tf.compat.as_bytes(out_path),
        tf.compat.as_bytes(str(version))
    )
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    word_ids_info = tf.saved_model.utils.build_tensor_info(model.word_ids)
    char_ids_info = tf.saved_model.utils.build_tensor_info(model.char_ids)
    sequence_length_info = tf.saved_model.utils.build_tensor_info(
        model.sequence_length)
    word_length_info = tf.saved_model.utils.build_tensor_info(
        model.word_length)
    decode_tags_info = tf.saved_model.utils.build_tensor_info(
        model.decode_tags)
    best_scores_info = tf.saved_model.utils.build_tensor_info(
        model.best_scores)
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            'word_ids': word_ids_info,
            'char_ids': char_ids_info,
            'sequence_length': sequence_length_info,
            'word_length': word_length_info
        },
        outputs={
            'decode_tags': decode_tags_info,
            'best_scores': best_scores_info
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    builder.add_meta_graph_and_variables(
        model.sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "sequence_tags": prediction_signature
        }
    )
    builder.save(as_text=False)


class SequenceTaggingModel(BaseModel):
    def __init__(self, configs, mode='train'):
        super().__init__(configs)
        self.mode = mode

    def _add_placeholders(self):
        # Word input sequence [batch_size x max_sequence_length]
        self.word_ids = tf.placeholder(tf.int32, [None, None], name='word_ids')

        # Character input sequence [batch_size x max_sequence_length x max_word_length]
        self.char_ids = tf.placeholder(
            tf.int32, [None, None, None],
            name='char_ids'
        )

        # Word input sequence length [batch_size]
        self.sequence_length = tf.placeholder(
            tf.int32, [None],
            name='sequence_length'
        )

        # Word lengths [batch_size x max_sequence_length]
        self.word_length = tf.placeholder(
            tf.int32, [None, None],
            name='word_length'
        )

        # Sequence labels for training [batch_size x max_sequence_length]
        self.labels = tf.placeholder(
            tf.int32, [None, None],
            name='labels'
        )
        self.loss = 0
        if self.mode == 'train':
            self.dropout = tf.placeholder(tf.float32, [], name='dropout')
            if self.configs.use_rnn or self.configs.char_embedding == 'rnn' or self.configs.use_gcnn:
                self.state_dropout = tf.placeholder(
                    tf.float32, [], name='state_dropout')
            # self.learning_rate = tf.placeholder(
            # tf.float32, [], name='learning_rate')
                self.rnn_cell = (lambda hidden: build_gru_cell_with_dropout(
                    hidden, self.state_dropout))  # (lambda hidden: build_lstm_layer_norm(hidden, self.dropout)) if self.configs.lstm_layer_norm else

        else:
            self.dropout = None
            self.state_dropout = None
            # build_lstm_layer_norm if self.configs.lstm_layer_norm else
            if self.configs.use_rnn or self.configs.char_embedding == 'rnn':
                self.rnn_cell = build_gru_cell
        if self.configs.batch_norm:
            self.training = tf.placeholder(tf.bool, [], name='training')

    def build_model(self, pretrain_word_embedding):
        self._add_placeholders()
        self._build_word_embedding(pretrain_word_embedding)
        self._add_logits_op()
        self._add_loss_op()
        self._add_pred_op()
        if self.mode == 'train':
            self._add_train_op(method=self.configs.training_method, loss=self.loss,
                               learning_rate=self.configs.learning_rate, momentum=self.configs.momentum, use_nesterov=self.configs.use_nesterov,
                               clip=self.configs.clip_grad)
            self._initialize_session()

    def save_hyperparams(self):
        self.configs.to_json(os.path.join(
            self.configs.out_dir, 'hyperparams.json'))

    def _add_pred_op(self):
        if self.configs.use_crf:
            self.decode_tags, self.best_scores = tf.contrib.crf.crf_decode(
                potentials=self.logits,
                transition_params=self.transition_params,
                sequence_length=self.sequence_length
            )
            self.decode_tags = tf.identity(self.decode_tags, 'decode_tags')
            self.best_scores = tf.identity(self.best_scores, 'best_scores')
        else:
            self.label_preds = tf.argmax(
                self.logits, axis=-1, output_type=tf.int32, name='label_preds')

    def _build_word_embedding(self, pretrain_word_embedding):
        with tf.variable_scope('word_embedding'):
            with tf.device('/cpu:0'):
                if pretrain_word_embedding is None:
                    word_embedding = tf.Variable(
                        tf.random_uniform(
                            [self.configs.nwords, self.configs.wdims],
                            -0.1, 0.1,
                            tf.float32
                        )
                    )
                else:
                    self.configs.nwords, self.configs.wdims = pretrain_word_embedding.shape
                    word_embedding = tf.Variable(
                        pretrain_word_embedding,
                        dtype=tf.float32,
                        trainable=self.configs.train_word_embedding
                    )
                word_embedding = tf.nn.embedding_lookup(
                    word_embedding,
                    self.word_ids,
                    name='word_embedding'
                )
                char_embedding = tf.Variable(
                    tf.random_uniform(
                        [self.configs.nchars, self.configs.cdims],
                        -0.1, 0.1,
                        tf.float32
                    )
                )
                char_embedding = tf.nn.embedding_lookup(
                    char_embedding, self.char_ids,
                    name='char_embedding'
                )
            s = tf.shape(char_embedding)
            char_embedding = tf.reshape(
                char_embedding,
                [s[0]*s[1], s[2], self.configs.cdims]
            )
            if self.configs.char_embedding == 'rnn':
                word_length = tf.reshape(self.word_length, [s[0]*s[1]])
                _, fs_fw, fs_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    [self.rnn_cell(x) for x in self.configs.num_hidden_char],
                    [self.rnn_cell(x) for x in self.configs.num_hidden_char],
                    char_embedding,
                    sequence_length=word_length,
                    parallel_iterations=32,
                    dtype=tf.float32
                )
                output = tf.concat([fs_fw[-1], fs_bw[-1]], axis=-1)
                output = tf.nn.relu(output)
            else:
                word_length = tf.reshape(self.word_length, [s[0]*s[1]])
                mask = tf.expand_dims(tf.sequence_mask(
                    word_length, dtype=tf.float32), axis=-1)
                output = char_embedding
                for i, (num_filters, kernel_size) in enumerate(zip(self.configs.num_hidden_char, self.configs.char_embedding_kernel_size)):
                    with tf.variable_scope('char_conv_%d' % i):
                        output *= tf.stop_gradient(
                            tf.tile(mask, multiples=[
                                    1, 1, tf.shape(output)[-1]])
                        )
                        output = tf.layers.conv1d(
                            inputs=output,
                            filters=num_filters,
                            kernel_size=kernel_size,
                            padding='same',
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer()
                        )
                output = tf.reduce_max(output, axis=-2)
            if self.configs.sum_vector:
                W0 = tf.get_variable(
                    'W0', shape=[2*self.configs.num_hidden_char[-1], pretrain_word_embedding.shape[-1]],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                output = tf.tanh(tf.matmul(output, W0))
                word_embedding = tf.reshape(
                    word_embedding, [s[0]*s[1], pretrain_word_embedding.shape[-1]])
                g = np.ones(pretrain_word_embedding.shape[0])
                g[-1] = 0
                g = tf.nn.embedding_lookup(
                    tf.constant(g, dtype=tf.float32),
                    self.word_ids,
                    name='g'
                )
                self.loss += self.configs.word_char_cosine * tf.reduce_mean(g*tf.losses.cosine_distance(
                    tf.stop_gradient(tf.nn.l2_normalize(
                        word_embedding, -1)),
                    tf.nn.l2_normalize(output, -1),
                    -1
                ))
                W1 = tf.get_variable(
                    'W1', shape=[pretrain_word_embedding.shape[-1], pretrain_word_embedding.shape[-1]],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                W2 = tf.get_variable(
                    'W2', shape=[pretrain_word_embedding.shape[-1], pretrain_word_embedding.shape[-1]],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                W3 = tf.get_variable(
                    'W3', shape=[pretrain_word_embedding.shape[-1], pretrain_word_embedding.shape[-1]],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                z = tf.sigmoid(tf.matmul(tf.tanh(tf.add(tf.matmul(word_embedding, W1),
                                                        tf.matmul(output, W2))), W3))
                word_embedding = tf.add(tf.multiply(z, word_embedding),
                                        tf.multiply((1-z), output))
                word_embedding = tf.reshape(
                    word_embedding, [s[0], s[1],
                                     pretrain_word_embedding.shape[-1]]
                )
            else:
                output = tf.reshape(
                    output, [s[0], s[1], output.get_shape()[-1]]
                )
                word_embedding = tf.concat([word_embedding, output], axis=-1)
            if self.mode == 'train':
                self.word_embedding = tf.nn.dropout(
                    x=word_embedding, keep_prob=self.dropout)  # , noise_shape=[tf.shape(word_embedding)[0], 1, word_embedding.get_shape()[2]]) if self.mode == 'train' else word_embedding
            else:
                self.word_embedding = word_embedding

    def _add_logits_op(self):
        if self.configs.use_rnn:
            if self.configs.use_conv and not self.configs.stack_rnn_cnn:
                with tf.variable_scope('mask_cnn'):
                    mask = tf.tile(tf.expand_dims(tf.sequence_mask(self.sequence_length, dtype=tf.float32),
                                                  axis=-1),
                                   multiples=[
                        1, 1, self.word_embedding.get_shape()[-1]]
                    )
                    inputs = self.word_embedding * tf.stop_gradient(mask)
                if type(self.configs.kernel_size) is list:
                    rnn_inputs = []
                    for kernel_size, filters in zip(self.configs.kernel_size, self.configs.num_filters):
                        rnn_inputs.append(tf.layers.conv1d(inputs,
                                                           filters=filters,
                                                           kernel_size=kernel_size,
                                                           strides=1,
                                                           padding='same',
                                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                           activation=tf.nn.relu))
                    rnn_input = tf.concat(rnn_inputs, axis=-1)
                else:
                    rnn_input = tf.layers.conv1d(inputs,
                                                 filters=self.configs.num_filters,
                                                 kernel_size=self.configs.kernel_size,
                                                 strides=1,
                                                 padding='same',
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 activation=tf.nn.relu)
                if self.mode == 'train':
                    # s = tf.shape(rnn_input)
                    rnn_input = tf.nn.dropout(
                        rnn_input, keep_prob=self.dropout)  # , noise_shape=[s[0], 1, s[-1]])
                # if self.configs.concat_residual and self.configs.use_residual:
                #     rnn_input = tf.concat([rnn_input, inputs], axis=-1)
            else:
                rnn_input = self.word_embedding
            if self.configs.bi_gru:
                if self.configs.stack_gru:
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=tf.nn.rnn_cell.MultiRNNCell(
                            cells=[self.rnn_cell(x) for x in self.configs.num_hidden_word]),
                        cell_bw=tf.nn.rnn_cell.MultiRNNCell(
                            cells=[self.rnn_cell(x) for x in self.configs.num_hidden_word]),
                        inputs=rnn_input,
                        sequence_length=self.sequence_length,
                        dtype=tf.float32
                    )
                else:
                    if self.configs.stack_rnn_cnn:
                        if self.configs.use_conv:
                            rnn_input = tf.layers.conv1d(self.word_embedding,
                                                         filters=self.configs.num_filters[0],
                                                         kernel_size=self.configs.kernel_size[0],
                                                         strides=1,
                                                         padding='same',
                                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                         activation=tf.nn.relu)
                            filters = self.configs.num_filters[1:]
                            kernel_sizes = self.configs.kernel_size[1:]
                        else:
                            filters = self.configs.num_filters
                            kernel_sizes = self.configs.kernel_size
                        outputs, _, _ = stack_bidirectional_dynamic_rnn_cnn(
                            cells_fw=[self.rnn_cell(x)
                                      for x in self.configs.num_hidden_word],
                            cells_bw=[self.rnn_cell(x)
                                      for x in self.configs.num_hidden_word],
                            cnn_filters=filters,
                            cnn_sizes=kernel_sizes,
                            dropout=self.dropout,
                            inputs=rnn_input,
                            dtype=tf.float32
                        )
                        final_size = self.configs.num_filters[-1]
                    else:
                        if self.configs.block_rnn:
                            dynamic_rnn = tf.contrib.rnn.stack_bidirectional_dynamic_rnn
                            outputs = rnn_input
                            for i in range(self.configs.num_block_rnn):
                                temp = outputs
                                with tf.variable_scope('block_rnn_%d' % i):
                                    outputs, _, _ = dynamic_rnn(
                                        [self.rnn_cell(x)
                                         for x in self.configs.num_hidden_word],
                                        [self.rnn_cell(x)
                                         for x in self.configs.num_hidden_word],
                                        rnn_input,
                                        sequence_length=self.sequence_length,
                                        dtype=tf.float32
                                    )
                                    if self.configs.use_residual:
                                        if self.configs.concat_residual:
                                            outputs = array_ops.concat(
                                                [outputs, temp], 2)
                                        else:
                                            outputs = tf.layers.conv1d(inputs=array_ops.concat(outputs, 2),
                                                                       filters=temp.get_shape(
                                            )[-1],
                                                kernel_size=1,
                                                activation=tf.nn.relu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            ) + temp
                                    final_size = outputs.get_shape()[-1]
                        else:
                            dynamic_rnn = (lambda *args, **kwargs: stack_bidirectional_dynamic_rnn(
                                concat_residual=self.configs.concat_residual, *args, **kwargs)) if self.configs.use_residual else tf.contrib.rnn.stack_bidirectional_dynamic_rnn
                            outputs, _, _ = dynamic_rnn(
                                [self.rnn_cell(x)
                                 for x in self.configs.num_hidden_word],
                                [self.rnn_cell(x)
                                 for x in self.configs.num_hidden_word],
                                rnn_input,
                                sequence_length=self.sequence_length,
                                dtype=tf.float32
                                # swap_memory=True
                            )

            else:
                outputs, _ = tf.nn.dynamic_rnn(
                    cell=tf.nn.rnn_cell.MultiRNNCell(
                        cells=[self.rnn_cell(x) for x in self.configs.num_hidden_word]),
                    inputs=rnn_input,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32
                )
                final_size = self.configs.num_hidden_word[-1]
        else:
            with tf.name_scope('sequence_mask'):
                seq_mask = tf.expand_dims(tf.sequence_mask(self.sequence_length, dtype=tf.float32),
                                          axis=-1)
            with tf.variable_scope('mask_cnn'):
                mask = tf.tile(seq_mask, multiples=[
                    1, 1, self.word_embedding.get_shape()[-1]]
                )
                inputs = self.word_embedding * tf.stop_gradient(mask)
            if self.configs.use_latent_dim:
                outputs = tf.layers.conv1d(
                    inputs=inputs,
                    filters=self.configs.latent_dim,
                    kernel_size=1,
                    activation=tf.nn.relu,
                    strides=1
                )
            else:
                outputs = inputs
            if self.configs.use_bi_causal_conv:
                # outputs = inputs
                if self.configs.bi_causal_conv_block:
                    for j in range(self.configs.bi_causal_conv_block):
                        with tf.variable_scope('bi_causal_conv_block' if self.configs.reuse else ('bi_causal_conv_block_%d' % j), reuse=self.configs.reuse):
                            temp = outputs
                            i = 0
                            gate_filter_width = self.configs.gate_filter_width if self.configs.gate_filter_width else self.configs.kernel_size
                            for filters, kernel_size, dilation_rate, gate_kernel_size in zip(self.configs.num_hidden_word, self.configs.kernel_size, self.configs.dilation_rate, gate_filter_width):
                                outputs = bi_causal_conv(
                                    outputs, kernel_size, filters, dilation_rate, self.sequence_length,
                                    self.configs.use_gcnn, kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        self.configs.l2_regs),
                                    gate_filter_width=gate_kernel_size, combine_gated=self.configs.bcc_combine_gated, name='bi_causal_conv_%d' % i)
                                i += 1
                            output_shape = tf.shape(outputs)
                            if self.mode == 'train':
                                outputs = tf.nn.dropout(x=outputs, keep_prob=self.dropout, noise_shape=[
                                    output_shape[0], 1, output_shape[-1]])
                            if self.configs.use_residual:
                                if not self.configs.concat_residual:
                                    outputs = tf.layers.conv1d(
                                        outputs,
                                        filters=temp.get_shape()[-1],
                                        kernel_size=1,
                                        strides=1,
                                        padding='valid',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation=tf.nn.relu,
                                        name='conv1d_res_{}'.format(i),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            self.configs.l2_regs)
                                    ) + temp
                                else:
                                    outputs = tf.concat(
                                        [outputs, temp], axis=-1)
                            final_size = output_shape[-1]
                            if self.configs.layer_norm:
                                outputs = tf.contrib.layers.layer_norm(inputs=outputs,
                                                                       center=True,
                                                                       scale=True,
                                                                       activation_fn=None)
                            if self.configs.batch_norm:
                                outputs = tf.contrib.layers.batch_norm(
                                    inputs=outputs, is_training=self.training)

                else:
                    i = 0
                    gate_filter_width = self.configs.gate_filter_width if self.configs.gate_filter_width else self.configs.kernel_size
                    for filters, kernel_size, dilation_rate, gate_kernel_size in zip(self.configs.num_hidden_word, self.configs.kernel_size, self.configs.dilation_rate, gate_filter_width):
                        temp = outputs
                        if self.configs.use_residual and not self.configs.concat_residual:
                            outputs = tf.layers.conv1d(
                                outputs,
                                filters=temp.get_shape()[-1]//2,
                                kernel_size=1,
                                strides=1,
                                padding='valid',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                activation=tf.nn.relu,
                                name='conv1d_res_{}_in'.format(i),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        self.configs.l2_regs)
                            )
                        outputs = bi_causal_conv(
                            outputs, kernel_size, filters, dilation_rate,
                            self.sequence_length, self.configs.use_gcnn,
                            tf.contrib.layers.l2_regularizer(
                                self.configs.l2_regs),
                            gate_kernel_size, combine_gated=self.configs.bcc_combine_gated, name='bi_causal_conv_%d' % i)
                        output_shape = tf.shape(outputs)
                        if self.mode == 'train':
                            outputs = tf.nn.dropout(x=outputs, keep_prob=self.dropout, noise_shape=[
                                output_shape[0], 1, output_shape[-1]])
                        if self.configs.use_residual:
                            if not self.configs.concat_residual:
                                outputs = tf.layers.conv1d(
                                    outputs,
                                    filters=temp.get_shape()[-1],
                                    kernel_size=1,
                                    strides=1,
                                    padding='valid',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.relu,
                                    name='conv1d_res_{}_out'.format(i),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        self.configs.l2_regs)
                                ) + temp
                            else:
                                outputs = tf.concat(
                                    [outputs, temp], axis=-1)
                        i += 1
                        if self.configs.layer_norm:
                            outputs = tf.contrib.layers.layer_norm(inputs=outputs,
                                                                   center=True,
                                                                   scale=True,
                                                                   activation_fn=None)
                        if self.configs.batch_norm:
                            outputs = tf.contrib.layers.batch_norm(
                                inputs=outputs, is_training=self.training)

                        final_size = output_shape[-1]

            else:
                if self.configs.bi_causal_conv_block:
                    for j in range(self.configs.bi_causal_conv_block):
                        with tf.variable_scope('conv_block_%d' % j):
                            temp = outputs
                            for i, num_filters in enumerate(self.configs.num_hidden_word):
                                prev_layer = outputs
                                outputs = tf.layers.conv1d(
                                    outputs,
                                    filters=num_filters,
                                    kernel_size=self.configs.kernel_size[i],
                                    strides=1,
                                    padding='same',
                                    dilation_rate=self.configs.dilation_rate[i],
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=None if self.configs.use_gcnn else tf.nn.relu,
                                    name='conv1d_word_{}'.format(i),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        self.configs.l2_regs)
                                )
                                if self.configs.use_gcnn:
                                    if self.configs.gate_filter_width:
                                        kernel_size = self.configs.gate_filter_width[i]
                                    else:
                                        kernel_size = self.configs.kernel_size[i]
                                    gate = tf.layers.conv1d(
                                        prev_layer,
                                        filters=num_filters,
                                        kernel_size=kernel_size,
                                        strides=1,
                                        padding='same',
                                        dilation_rate=self.configs.dilation_rate[i],
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation=tf.sigmoid,
                                        name='conv1d_gate_{}'.format(i),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            self.configs.l2_regs)
                                    )
                                    if self.mode == 'train':
                                        gate = tf.nn.dropout(
                                            x=gate, keep_prob=self.state_dropout, name='drop_out_gate_{}'.format(i))
                                    outputs = tf.multiply(outputs, gate)
                                output_shape = tf.shape(outputs)
                                if self.mode == 'train' and not self.configs.use_gcnn:
                                    outputs = tf.nn.dropout(
                                        x=outputs, keep_prob=self.dropout, name='drop_out_{}'.format(i))
                                outputs *= tf.stop_gradient(
                                    tf.tile(seq_mask,
                                            multiples=[
                                                1, 1, outputs.get_shape()[-1]]
                                            ), name='mask_output_{}'.format(i)
                                )
                            if self.configs.use_residual:
                                if self.configs.concat_residual:
                                    outputs = array_ops.concat(
                                        [outputs, temp], axis=-1)
                                else:
                                    outputs = tf.layers.conv1d(
                                        outputs,
                                        filters=temp.get_shape()[-1],
                                        kernel_size=1,
                                        strides=1,
                                        padding='valid',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation=tf.nn.relu,
                                        name='conv1d_res_{}'.format(i),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            self.configs.l2_regs)
                                    ) + temp
                            if self.configs.layer_norm:
                                outputs = tf.contrib.layers.layer_norm(inputs=outputs,
                                                                       center=True,
                                                                       scale=True,
                                                                       activation_fn=None)
                            if self.configs.batch_norm:
                                outputs = tf.contrib.layers.batch_norm(
                                    inputs=outputs, is_training=self.training)

                            final_size = output_shape[-1]

                else:
                    for i, (num_filters, ksz, dr) in enumerate(zip(self.configs.num_hidden_word, self.configs.kernel_size, self.configs.dilation_rate)):
                        outputs *= tf.stop_gradient(
                            tf.tile(tf.expand_dims(tf.sequence_mask(self.sequence_length, dtype=tf.float32),
                                                   axis=-1),
                                    multiples=[
                                1, 1, outputs.get_shape()[-1]]
                            ),
                            name='mask_word_{}'.format(i)
                        )
                        temp = outputs
                        outputs = tf.layers.conv1d(
                            outputs,
                            filters=num_filters,
                            kernel_size=ksz,
                            strides=1,
                            padding='same',
                            dilation_rate=dr,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            activation=None if self.configs.use_gcnn else tf.nn.relu,
                            name='conv1d_word_{}'.format(i),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                self.configs.l2_regs)
                        )
                        if self.configs.use_gcnn:
                            if self.configs.gate_filter_width:
                                kernel_size = self.configs.gate_filter_width[i]
                            else:
                                kernel_size = ksz
                            gate = tf.layers.conv1d(
                                temp,
                                filters=num_filters,
                                kernel_size=kernel_size,
                                strides=1,
                                padding='same',
                                dilation_rate=dr,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                activation=tf.sigmoid,
                                name='conv1d_gate_{}'.format(i),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                    self.configs.l2_regs)
                            )
                            if self.mode == 'train':
                                gate = tf.nn.dropout(
                                    x=gate, keep_prob=self.state_dropout)
                            outputs = tf.multiply(outputs, gate)
                        output_shape = tf.shape(outputs)
                        if self.mode == 'train' and not self.configs.use_gcnn:
                            outputs = tf.nn.dropout(
                                x=outputs, keep_prob=self.dropout)
                        if self.configs.use_residual:
                            if self.configs.concat_residual:
                                outputs = array_ops.concat(
                                    [outputs, temp], axis=-1)
                            else:
                                outputs *= tf.stop_gradient(
                                    tf.tile(tf.expand_dims(tf.sequence_mask(self.sequence_length, dtype=tf.float32),
                                                           axis=-1),
                                            multiples=[
                                        1, 1, outputs.get_shape()[-1]]
                                    ),
                                    name='mask_res_{}'.format(i)
                                )
                                outputs = tf.layers.conv1d(
                                    outputs,
                                    filters=temp.get_shape()[-1],
                                    kernel_size=1,
                                    strides=1,
                                    padding='valid',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=None,
                                    name='conv1d_res_{}'.format(i),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        self.configs.l2_regs)
                                ) + temp
                        # if self.configs.layer_norm:
                        #     outputs = tf.contrib.layers.layer_norm(inputs=outputs,
                        #                                            center=True,
                        #                                            scale=True,
                        #                                            activation_fn=None)
                        # if self.configs.batch_norm:
                        #     outputs = tf.contrib.layers.batch_norm(
                        #         inputs=outputs, is_training=self.training)

                        final_size = output_shape[-1]
        if self.configs.attention:
            luong_att = tf.contrib.seq2seq.LuongAttention(
                num_units=self.configs.attention_units,
                memory=outputs,
                memory_sequence_length=self.sequence_length,
                scale=True
            )
            outputs, _ = rnn.dynamic_rnn(
                cell=tf.contrib.seq2seq.AttentionWrapper(
                    cell=self.rnn_cell(
                        self.configs.attention_units),
                    attention_mechanism=luong_att,
                    attention_layer_size=self.configs.attention_units
                ),
                swap_memory=True,
                parallel_iterations=64,
                inputs=outputs, sequence_length=self.sequence_length,
                dtype=tf.float32
            )
        final_size = outputs.get_shape()[-1]
        if self.configs.final_layer == 'cnn':
            self.logits = outputs
            self.configs.final_layer_filters[-1] = self.configs.num_classes
            for filters, size in zip(self.configs.final_layer_filters, self.configs.final_layer_kernel):
                self.logits *= tf.stop_gradient(
                    tf.tile(tf.expand_dims(tf.sequence_mask(
                        self.sequence_length, dtype=tf.float32), -1), [1, 1, tf.shape(self.logits)[-1]]))
                self.logits = tf.layers.conv1d(
                    inputs=self.logits,
                    filters=filters,
                    kernel_size=size,
                    padding='valid',
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    activation=tf.nn.relu,
                    name='conv_final_%d_%d' % (filters, size),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                        self.configs.l2_regs)
                )
            self.logits *= tf.stop_gradient(
                tf.tile(tf.expand_dims(tf.sequence_mask(
                        self.sequence_length, dtype=tf.float32), -1), [1, 1, tf.shape(self.logits)[-1]]))
        else:
            W0 = tf.get_variable(
                'W0', shape=[final_size, self.configs.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b0 = tf.Variable(
                tf.zeros(self.configs.num_classes, dtype=tf.float32), name='b0'
            )
            l2_loss = tf.nn.l2_loss(W0) + tf.nn.l2_loss(b0)
            self.loss += l2_loss * self.configs.l2_regs
            nsteps = tf.shape(outputs)[1]
            outputs = tf.reshape(outputs, [-1, final_size])
            scores = tf.matmul(outputs, W0) + b0
            self.logits = tf.reshape(
                scores, [-1, nsteps, self.configs.num_classes], name='logits'
            )
            self.logits = tf.nn.relu(self.logits)

    def _add_loss_op(self):
        if self.configs.use_crf:
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs=self.logits,
                tag_indices=self.labels,
                sequence_lengths=self.sequence_length
            )
            self.transition_params = transition_params
            self.loss += tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_length)
            losses = tf.boolean_mask(losses, mask)
            self.loss += tf.reduce_mean(losses)
        with tf.name_scope('train'):
            self.loss += tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', self.loss)

    def _get_feed_dict(self,
                       sentences,
                       sentence_length,
                       words,
                       word_length,
                       #    lr=None,
                       labels=None,
                       is_training=False,
                       dropout=1.0,
                       state_dropout=1.0):
        feed_dict = {}
        feed_dict[self.word_ids] = sentences
        feed_dict[self.sequence_length] = sentence_length
        feed_dict[self.char_ids] = words
        feed_dict[self.word_length] = word_length
        feed_dict[self.state_dropout] = state_dropout
        if self.configs.batch_norm:
            feed_dict[self.training] = is_training
        # feed_dict[self.learning_rate] = lr
        feed_dict[self.dropout] = dropout
        if labels is not None:
            feed_dict[self.labels] = labels
        return feed_dict

    def predict_batch(self,
                      sentences,
                      sentence_lengths,
                      words,
                      word_lengths,
                      labels=None,
                      with_summary=False):
        if self.configs.use_crf:
            if with_summary:
                fd = self._get_feed_dict(
                    sentences, sentence_lengths, words, word_lengths, labels
                )
                decode_tags, best_scores, train_summaries = self.sess.run([
                    self.decode_tags, self.best_scores, self.train_summaries
                ], feed_dict=fd)
                return decode_tags, best_scores, train_summaries
            fd = self._get_feed_dict(
                sentences, sentence_lengths, words, word_lengths
            )
            decode_tags, best_scores = self.sess.run([
                self.decode_tags, self.best_scores
            ], feed_dict=fd)
            return decode_tags, best_scores
        else:
            labels_pred = self.sess.run(self.label_preds, feed_dict=fd)
            return labels_pred, [0]

    def train_dev_loop(self,
                       train_iter, dev_iter, eval_freq, num_epochs, early_stopping,
                       test_iter=None):
        self._add_summary()
        next_batch = train_iter.get_next()
        # current_lr = self.configs.learning_rate
        if early_stopping > 0:
            smaller_count = 0
            best = 0
        for i in range(num_epochs):
            self.sess.run(train_iter.initializer)
            run = []
            try:
                while True:
                    sentences, sentence_lengths, words, word_lengths, labels = self.sess.run(
                        next_batch
                    )
                    fd = self._get_feed_dict(
                        sentences, sentence_lengths, words, word_lengths, labels, True, self.configs.dropout, self.configs.rnn_state_dropout)
                    run = self.sess.run(
                        [self.train_op, self.train_summaries,
                            self.loss, self.global_step], fd
                    )
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {}".format(
                        time_str, run[3], run[2]))
                    self.logger.info("Step {}, loss {}".format(run[3], run[2]))
                    self.train_summaries_writer.add_summary(run[1], run[3])
            except tf.errors.OutOfRangeError:
                if (i + 1) % eval_freq == 0:
                    self._save_model()
                    metrics, summaries = self.evaluate_step(
                        dev_iter, with_summary=True)
                    self.dev_summaries_writer.add_summary(
                        summaries, self.sess.run(self.global_step))
                    self.logger.info("Evaluation {}".format(str(metrics)))
                    if test_iter:
                        metrics_test = self.evaluate_step(test_iter)
                        self.logger.info(
                            "Evaluation on test {}".format(str(metrics_test)))
                        # self.dev_summaries_writer.add_summary(
                        #     metrics_test['loss'])
                    if early_stopping > 0:
                        if metrics['total']['f1'] < best:
                            smaller_count += 1
                            # current_lr = current_lr / 10
                        else:
                            smaller_count = 0
                            best = metrics['total']['f1']
                        if smaller_count >= early_stopping:
                            break
        self._save_model()
        self._close_session()

    def evaluate_step(self,
                      dev_iter, with_summary=False):
        self.sess.run(dev_iter.initializer)
        next_dev_batch = dev_iter.get_next()
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        main_tag = [x.split('-')[-1] for x in self.configs.vocab_tags]
        class_metrics = {
            x: {
                'correct_preds': 0.,
                'total_preds': 0.,
                'total_correct': 0.
            } for x in main_tag
        }
        try:
            while True:
                sentences, sentence_lengths, words, word_lengths, labels = self.sess.run(
                    next_dev_batch
                )
                if with_summary:
                    pred_labels, _, summaries = self.predict_batch(
                        sentences, sentence_lengths, words, word_lengths, labels, with_summary=with_summary
                    )
                else:
                    pred_labels, _ = self.predict_batch(
                        sentences, sentence_lengths, words, word_lengths, None, with_summary=False
                    )
                for lab, lab_pred, length in zip(labels, pred_labels,
                                                 sentence_lengths):
                    lab = lab[:length]
                    lab_pred = lab_pred[:length]
                    accs += [a == b for (a, b) in zip(lab, lab_pred)]

                    lab_chunks = set(get_chunks(lab, self.configs.vocab_tags))
                    lab_pred_chunks = set(get_chunks(lab_pred,
                                                     self.configs.vocab_tags))

                    correct_preds += len(lab_chunks & lab_pred_chunks)
                    total_preds += len(lab_pred_chunks)
                    total_correct += len(lab_chunks)
                    for tag in main_tag:
                        tag_lab_chunks = set(
                            chunk for chunk in lab_chunks if chunk[0] == tag)
                        tag_lab_pred_chunks = set(
                            chunk for chunk in lab_pred_chunks if chunk[0] == tag)
                        class_metrics[tag]['correct_preds'] += len(
                            tag_lab_chunks & tag_lab_pred_chunks)
                        class_metrics[tag]['total_preds'] += len(
                            tag_lab_pred_chunks)
                        class_metrics[tag]['total_correct'] += len(
                            tag_lab_chunks)
        except tf.errors.OutOfRangeError:
            pass
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs),
        res = {}
        res['total'] = {"f1": 100 * f1,
                        "precision": 100 * p, "recall": 100 * r, "acc": acc[0]}
        for tag, value in class_metrics.items():
            p = value['correct_preds'] / \
                value['total_preds'] if value['correct_preds'] > 0 else 0
            r = value['correct_preds'] / \
                value['total_correct'] if value['correct_preds'] > 0 else 0
            f1 = 2 * p * r / (p + r) if value['correct_preds'] > 0 else 0
            res[tag] = {"f1": 100*f1, "precision": 100*p, "recall": 100*r}
        return (res, summaries) if with_summary else res
