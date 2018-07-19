import tensorflow as tf


def causal_conv(value, filter_width, filter_num, dilation, seq_length, use_gcnn,  kernel_regularizer, activation=None, gate_filter_width=None,
                name='causal_conv', kernel_initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name):
        value *= tf.stop_gradient(
            tf.tile(tf.expand_dims(tf.sequence_mask(
                seq_length, dtype=tf.float32), -1), [1, 1, tf.shape(value)[-1]], name="seq_mask")
        )
        input1 = tf.pad(value, [[0, 0], [dilation*(filter_width-1), 0],
                                [0, 0]], mode='CONSTANT', constant_values=0.0)
        result = tf.layers.conv1d(inputs=input1, filters=filter_num,
                                  kernel_size=filter_width, dilation_rate=dilation,
                                  activation=activation,
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  padding='valid')
        result *= tf.stop_gradient(
            tf.tile(tf.expand_dims(tf.sequence_mask(
                seq_length, dtype=tf.float32), -1), [1, 1, tf.shape(result)[-1]], name="seq_mask_result")
        )
        # result = tf.contrib.layers.layer_norm(inputs=result,
        #                                       center=True,
        #                                       scale=True,
        #                                       activation_fn=tf.tanh)
        if use_gcnn:
            input2 = tf.pad(value, [[0, 0], [dilation*(gate_filter_width-1), 0],
                                    [0, 0]], mode='CONSTANT', constant_values=0.0) if gate_filter_width else input1
            gate = tf.layers.conv1d(inputs=input2, filters=filter_num,
                                    kernel_size=gate_filter_width if gate_filter_width else filter_width, dilation_rate=dilation,
                                    activation=tf.sigmoid,
                                    kernel_regularizer=kernel_regularizer,
                                    kernel_initializer=kernel_initializer,
                                    padding='valid',
                                    name='gcnn')
            # gate = tf.contrib.layers.layer_norm(inputs=gate,
            #                                     center=True,
            #                                     scale=True,
            #                                     activation_fn=tf.sigmoid)
            result = tf.multiply(result, gate)
        return result


def bi_causal_conv(value, filter_width, filter_num, dilation, seq_length, use_gcnn, kernel_regularizer, gate_filter_width=None, combine_gated=False,
                   name='bi_causal_conv', kernel_initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name):
        fw = causal_conv(value, filter_width, filter_num,
                         dilation, seq_length, use_gcnn and not combine_gated, kernel_regularizer, None if combine_gated else tf.nn.relu, gate_filter_width, name='fw', kernel_initializer=kernel_initializer)
        bw = causal_conv(tf.reverse_sequence(
            value, seq_length, 1, 0), filter_width, filter_num, dilation, seq_length, use_gcnn and not combine_gated, kernel_regularizer, None if combine_gated else tf.nn.relu, gate_filter_width, name='bw', kernel_initializer=kernel_initializer)
        bw = tf.reverse_sequence(bw, seq_length, 1, 0)
        if combine_gated:
            z = tf.sigmoid(tf.layers.conv1d(inputs=fw, filters=filter_num, kernel_size=1, padding='valid', activation=tf.nn.tanh, name='fw_tanh') +
                           tf.layers.conv1d(inputs=bw, filters=filter_num, kernel_size=1, padding='valid', activation=tf.nn.tanh, name="bw_tanh"), name='gate')
            output = tf.multiply(z, tf.nn.relu(fw)) + \
                                 tf.multiply((1-z), tf.nn.relu(bw))
        else:
            output = tf.concat([fw, bw], axis=-1)
        return output * tf.stop_gradient(
            tf.tile(tf.expand_dims(tf.sequence_mask(
                seq_length, dtype=tf.float32), -1), [1, 1, tf.shape(output)[-1]], name="seq_mask_output"))


if __name__ == '__main__':
    import numpy as np
    a=1.0*np.random.randint(0, 10, [2, 7, 2])
    a[1][-3:][:]=0.0
    seq_length=np.array([7, 4])
    print('a', a)
    with tf.Session() as sess:
        a=tf.constant(a, tf.float32)
        # a = tf.pad(a, [[0, 0], [2, 0], [0, 0]])
        b=causal_conv(a, 3, 2, 1, seq_length, False, name='b',
                        kernel_initializer=tf.constant_initializer(1.0), kernel_regularizer=None)
        c=causal_conv(a, 3, 2, 2, seq_length, False, name='c',
                        kernel_initializer=tf.constant_initializer(1.0), kernel_regularizer=None)
        d=bi_causal_conv(a, 3, 2, 2, seq_length, False, name='d',
                           kernel_initializer=tf.constant_initializer(1.0), kernel_regularizer=None, combine_gated=True)
        sess.run(tf.global_variables_initializer())
        # print(sess.run(b))
        # # print(sess.run(batch_to_time(time_to_batch(a, 2), 2)))
        # print(sess.run(c))
        # print(tf.reverse_sequence(a, seq_length, 1, 0).eval())
        # print(tf.reverse_sequence(a, seq_length, 1, 0).eval())
        print(sess.run(b))
        print(sess.run(c))
        print(sess.run(d))
