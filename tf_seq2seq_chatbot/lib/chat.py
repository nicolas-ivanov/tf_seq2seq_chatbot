import os
import sys

import tensorflow as tf

from tf_seq2seq_chatbot.configs.config import FLAGS
from tf_seq2seq_chatbot.lib import data_utils
from tf_seq2seq_chatbot.lib.seq2seq_model_utils import create_model, get_predicted_sentence


def chat():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, forward_only=True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()

    while sentence:
        predicted_sentence = get_predicted_sentence(sentence, vocab, rev_vocab, model, sess)
        print(predicted_sentence)
        print("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()