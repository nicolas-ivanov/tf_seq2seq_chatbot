import tensorflow as tf


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")

tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 80000, "DIalog vocabulary size.")

tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
# step ?
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("test", False, "Print out for test dataset if this is set to True.")

FLAGS = tf.app.flags.FLAGS

TEST_DATASET_PATH = 'data/test/test_set.txt'

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]