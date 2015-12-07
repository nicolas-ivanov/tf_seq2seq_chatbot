from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import gfile
from seq2seq_upgrade import decoding_enhanced

from tf_seq2seq_chatbot.configs.config import FLAGS, buckets
from tf_seq2seq_chatbot.lib import data_utils
from tf_seq2seq_chatbot.lib import seq2seq_model

__author__ = 'nicolas'


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.vocab_size, FLAGS.vocab_size, buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.variables.initialize_all_variables())
  return model


def _get_predicted_sentence(input_sentence, vocab, rev_vocab, model, sess):
    token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)
    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(buckets)) if buckets[b][0] > len(token_ids)])
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

    TEMPERATURE = 0.7
    outputs = []
    for logit in output_logits:
      select_word_number = int(decoding_enhanced.sample_with_temperature(logit[0], TEMPERATURE))
      outputs.append(select_word_number)

    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    # outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]

    # Forming output sentence on natural language
    output_sentence = ' '.join([rev_vocab[output] for output in outputs])
    return output_sentence