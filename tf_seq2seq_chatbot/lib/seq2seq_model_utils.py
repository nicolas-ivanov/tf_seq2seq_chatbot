from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from tf_seq2seq_chatbot.configs.config import FLAGS, BUCKETS
from tf_seq2seq_chatbot.lib import data_utils
from tf_seq2seq_chatbot.lib import seq2seq_model


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      source_vocab_size=FLAGS.vocab_size,
      target_vocab_size=FLAGS.vocab_size,
      buckets=BUCKETS,
      size=FLAGS.size,
      num_layers=FLAGS.num_layers,
      max_gradient_norm=FLAGS.max_gradient_norm,
      batch_size=FLAGS.batch_size,
      learning_rate=FLAGS.learning_rate,
      learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
      use_lstm=False,
      forward_only=forward_only)

  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def get_predicted_sentence(input_sentence, vocab, rev_vocab, model, sess):
    input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    outputs = []

    feed_data = {bucket_id: [(input_token_ids, outputs)]}
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)

    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

    outputs = []
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    for logit in output_logits:
        selected_token_id = int(np.argmax(logit, axis=1))

        if selected_token_id == data_utils.EOS_ID:
            break
        else:
            outputs.append(selected_token_id)

    # Forming output sentence on natural language
    output_sentence = ' '.join([rev_vocab[output] for output in outputs])

    return output_sentence