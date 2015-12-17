import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from tf_seq2seq_chatbot.lib.train importin()

if __name__ == "__main__":
    tf.app.run()