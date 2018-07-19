import tensorflow as tf
import numpy as np

table = tf.contrib.lookup.index_table_from_file(
    vocabulary_file='all_words.txt',
    num_oov_buckets=0,
    vocab_size=None,
    default_value=-1,
    hasher_spec=tf.contrib.lookup.FastHashSpec,
    key_dtype=tf.string,
    name='word2id',
    key_column_index=0,
    value_column_index=1,
    delimiter='\t'
)
words = tf.contrib.lookup.index_table_from_file(
    vocabulary_file='all_chars.txt',
    num_oov_buckets=0,
    vocab_size=None,
    default_value=-1,
    hasher_spec=tf.contrib.lookup.FastHashSpec,
    key_dtype=tf.string,
    name='char2id',
    key_column_index=0,
    value_column_index=1,
    delimiter='\t'
)
s = 'hello\nword'
s = tf.string_split([s],' \n')
features = tf.constant(["0", "nhà", "and", "palmer",""])
ids = table.lookup(features)
cids = words.lookup(features)
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.tables_initializer().run()
    print(sess.run([ids,cids,s]))
