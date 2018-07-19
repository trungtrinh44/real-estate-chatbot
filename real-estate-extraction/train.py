import argparse
import pickle

import numpy as np
import tensorflow as tf

from data_utils import constants, read_data, read_word_vec
from model import configs, ner_model

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='./log_dir')
parser.add_argument('--word_tokenizer', type=str,
                    default='./output/word_tokenizer.pkl')
parser.add_argument('--char_tokenizer', type=str,
                    default='./output/char_tokenizer.pkl')
parser.add_argument('--data', type=str, default='./output')
parser.add_argument('--word_vec', type=str, default='./data/word_vec.pkl')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epoch', type=int, default=25)
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--version', type=str)
parser.add_argument('--hp_path', type=str)
parser.add_argument('--checkpoints', type=str)
args = parser.parse_args()
with open(args.word_tokenizer, 'rb') as file:
    word_tokenizer = pickle.load(file)
with open(args.char_tokenizer, 'rb') as file:
    char_tokenizer = pickle.load(file)

configs = configs.Config()
configs.version = args.version
configs.set_log_dir(args.log_dir)
if args.hp_path:
    model = ner_model.build_with_params(args.hp_path)
    # model.configs.training_method = 'adam'
    # model.configs.learning_rate = 1e-3
    model.configs.num_checkpoints = 20
    model.logger = configs.logger
    model.configs.train_summary_dir = configs.train_summary_dir
    model.configs.log_dir = configs.log_dir
    model.configs.dev_summary_dir = configs.dev_summary_dir
    model.configs.out_dir = configs.out_dir
    model.configs.saved_model_dir = configs.saved_model_dir
    model.configs.checkpoint_prefix = configs.checkpoint_prefix
    model.configs.version = configs.version
    model.configs.num_classes = constants.NUM_CLASSES
    model.configs.vocab_tags = constants.CLASSES
    mock_embedding = np.zeros(
        [model.configs.nwords, model.configs.wdims], dtype=float)
    model.build_model(mock_embedding)
else:
    embedding = read_word_vec.read_word_vec(
        args.word_vec, word_tokenizer)
    configs.nwords = len(word_tokenizer.word_index)+1
    configs.wdims = 164
    configs.nchars = len(char_tokenizer.word_index)+1
    configs.cdims = 30
    configs.use_conv = False
    configs.num_filters = 200
    configs.kernel_size = 3
    configs.dilation_rate = [1]*2
    configs.num_hidden_word = [128]*2
    configs.bcc_combine_gated = True
    configs.use_gcnn = True
    configs.reuse = False
    configs.latent_dim = 200
    configs.use_latent_dim = False
    configs.layer_norm = False
    configs.gate_filter_width = [None, None, None]
    configs.num_hidden_char = [30]
    configs.logit_hidden = 100
    configs.num_classes = constants.NUM_CLASSES
    configs.vocab_tags = constants.CLASSES
    configs.train_word_embedding = True
    configs.use_crf = True
    configs.sum_vector = False
    configs.char_embedding = 'cnn'
    configs.char_embedding_kernel_size = 3
    configs.word_char_cosine = 1.0
    configs.learning_rate = 1e-3
    configs.use_nesterov = True
    configs.training_method = 'adam'
    configs.momentum = 0.9
    configs.num_checkpoints = 20
    configs.use_rnn = True
    configs.block_rnn = False
    configs.num_block_rnn = 1
    configs.clip_grad = 5.0
    configs.use_residual = True
    configs.use_bi_causal_conv = True
    configs.final_layer = 'cnn'
    configs.final_layer_kernel = [1]
    configs.final_layer_filters = [configs.num_classes]
    configs.l2_regs = 1e-5
    configs.lstm_layer_norm = False
    configs.bi_gru = True
    configs.stack_gru = False
    configs.stack_rnn_cnn = False
    configs.concat_residual = True
    configs.attention_units = 200
    configs.attention = False
    configs.attention_align_fn = 'cosine'
    configs.bi_causal_conv_block = 2
    configs.use_sru = True
    configs.sru_units = 100
    model = ner_model.SequenceTaggingModel(configs)
    model.build_model(embedding)
    model.save_hyperparams()
train_iter, dev_iter = read_data.read_folder(
    args.data, args.batch_size, args.num_epoch, 0.1, random_seed=1, buffer_size=10000)
if args.checkpoints:
    model.saver.restore(
        model.sess, tf.train.latest_checkpoint(args.checkpoints))
model.train_dev_loop(train_iter, dev_iter,
                     args.eval_freq, args.num_epoch, 10, 0.5)
