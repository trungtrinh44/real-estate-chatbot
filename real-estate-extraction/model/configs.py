import json
import os
import time
from datetime import datetime
from functools import singledispatch

from model.utils import get_logger


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(datetime)
def ts_datetime(val):
    """Used if *val* is an instance of datetime."""
    return val.isoformat() + "Z"


def from_json(path):
    with open(path, 'r') as target:
        target = json.load(target)
    config = Config()
    config.__dict__.update(**target)
    return config


class Config(object):
    def __init__(self):
        self.log_dir = None
        self.num_checkpoints = 25
        self.learning_rate = 1e-3
        self.eval_freq = 50
        self.nwords = None
        self.wdims = None
        self.num_classes = None
        self.num_hidden_word = 128
        self.num_hidden_char = 32
        self.nchars = None
        self.cdims = None
        self.train_word_embedding = True
        self.use_crf = True
        self.sum_vector = True
        self.version = 1
        self.use_conv = True
        self.num_filters = 128
        self.kernel_size = 5
        self.char_embedding = 'rnn'
        self.char_embedding_kernel_size = 3
        self.use_rnn = True
        self.clip_grad = 'global'
        self.use_residual = False
        self.l2_regs = 1e-2
        self.training_method = 'adam'
        self.dilation_rate = None
        self.vocab_tags = None
        self.logger = None
        self.saved_model_dir = None
        self.final_layer = 'logits'
        self.final_layer_kernel = 3
        self.latent_dim = 150
        self.layer_norm = False
        self.bi_gru = True
        self.stack_gru = False
        self.stack_rnn_cnn = False
        self.concat_residual = False
        self.use_bi_causal_conv = False
        self.bi_causal_conv_block = 0
        self.block_rnn = False
        self.num_block_rnn = 0
        self.use_latent_dim = False
        self.use_gcnn = False
        self.gate_filter_width = None
        self.use_sru = False
        self.sru_units = None
        self.momentum = 0.9
        self.use_nesterov = True
        self.reuse = False
        self.bcc_combine_gated = False
        self.batch_norm = False
        self.attention_units = 200
        self.attention = False
        self.attention_align_fn = 'cosine'
        self.attn_length = 5
        self.dropout = 0.5
        self.rnn_state_dropout = 0.8

    def to_json(self, path):
        with open(path, 'w') as target:
            json.dump(self.__dict__, target, default=to_serializable)

    def set_log_dir(self, new_dir):
        self.log_dir = new_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.out_dir = os.path.abspath(os.path.join(
            self.log_dir, "runs", str(self.version)))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.saved_model_dir = os.path.abspath(
            os.path.join(self.out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.saved_model_dir, "model")
        self.train_summary_dir = os.path.join(
            self.out_dir, "summaries", "train")
        self.dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")
        self.logger = get_logger(os.path.join(self.out_dir, 'log'))


if __name__ == '__main__':
    config = Config()
    config.to_json('path')
