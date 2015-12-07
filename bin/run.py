
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tf_seq2seq_chatbot.configs.config import FLAGS
from tf_seq2seq_chatbot.lib.train import train
from tf_seq2seq_chatbot.lib.test import test
from tf_seq2seq_chatbot.lib.decode import decode

def main(_):
    if FLAGS.decode:
        decode()
    elif FLAGS.test:
        test()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()