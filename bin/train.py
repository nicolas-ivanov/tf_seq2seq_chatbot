import tensorflow as tf

from tf_seq2seq_chatbot.lib.train import train


def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()