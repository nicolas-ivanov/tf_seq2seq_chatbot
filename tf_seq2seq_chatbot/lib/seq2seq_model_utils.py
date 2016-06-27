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
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def _sample(probs, temperature=1.0):
        """
        helper function to sample an index from a probability array
        """
        strethced_probs = np.log(probs) / temperature
        vocab_size = len(strethced_probs)
        strethced_probs = np.exp(strethced_probs) / np.sum(np.exp(strethced_probs))
        idx = np.random.choice(vocab_size, p=strethced_probs)
        idx_prob = strethced_probs[idx]
        return idx, idx_prob

    input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)
    temperature = 0.5

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    output_token_ids = []
    max_answer_len_for_bucket = BUCKETS[bucket_id][1]

    for i in xrange(max_answer_len_for_bucket):
        feed_data = {bucket_id: [(input_token_ids, output_token_ids)]}
        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)

        # Get output logits for the sentence.
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

        # only interested in i-th logit on i-th step
        curr_logit = output_logits[i][0]
        curr_probs = softmax(curr_logit)

        # time to sample with temperature:
        curr_token_id, curr_prob = _sample(curr_probs, temperature)
        output_token_ids += [curr_token_id]

    # Forming output sentence on natural language
    output_sentence = ' '.join([rev_vocab[output] for output in output_token_ids])

    return output_sentence