import tensorflow as tf
import os


class BaseModel(object):
    def __init__(self, configs):
        self.configs = configs
        self.logger = configs.logger
        self.sess = None
        self.saver = None

    def _add_train_op(self, method, loss, learning_rate, momentum=None, use_nesterov=True, clip='global'):
        with tf.name_scope('train'):
            self.global_step = tf.Variable(0, trainable=False)
            if method == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(
                    learning_rate=learning_rate)
            elif method == 'sgd':
                # learning_rate = tf.train.exponential_decay(
                #     learning_rate,                # Base learning rate.
                #     self.global_step,  # Current index into the dataset.
                #     100,          # Decay step.
                #     0.95,                # Decay rate.
                #     staircase=True)
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=learning_rate)
                # clip = 'global'
            elif method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=learning_rate)
            elif method == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate=learning_rate)
            elif method == 'momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate,
                    momentum=momentum,
                    use_nesterov=use_nesterov
                )
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            if clip == 'global':
                gradients, _ = tf.clip_by_global_norm(
                    gradients, 1.0, name='global_clip')
            elif type(clip) is float:
                gradients = [tf.clip_by_norm(g, clip_norm=clip)
                             for g in gradients]
            self.train_op = optimizer.apply_gradients(
                zip(gradients, variables), global_step=self.global_step
            )
            # for g, v in zip(gradients, variables):
            #     if g is not None:
            #         tf.summary.histogram(
            #             "{}/grad/hist".format('_'.join(v.name.split(':'))),
            #             g
            #         )
            #         tf.summary.scalar(
            #             "{}/grad/sparsity".format('_'.join(v.name.split(':'))),
            #             tf.nn.zero_fraction(g)
            #         )

    def _initialize_session(self):
        self.logger.info("Initialize TF session")
        self.sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True)
        )
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=self.configs.num_checkpoints
        )
        self.sess.run(tf.global_variables_initializer())

    def _save_model(self):
        self.saver.save(
            self.sess, self.configs.checkpoint_prefix, self.global_step
        )

    def _close_session(self):
        self.sess.close()

    def _add_summary(self):
        self.train_summaries = tf.summary.merge_all()
        self.train_summaries_writer = tf.summary.FileWriter(
            self.configs.train_summary_dir,
            self.sess.graph
        )
        self.dev_summaries = tf.summary.merge_all()
        self.dev_summaries_writer = tf.summary.FileWriter(
            self.configs.dev_summary_dir,
            self.sess.graph
        )

    def _load_model(self, model_dir):
        self.saver.restore(self.sess, model_dir)
