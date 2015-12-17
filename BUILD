# Description:
# Example conversation chatbot model

package(default_visibility = ["//visibility:public"])


py_library(
    name = "rnn_enhanced",
    srcs = [
        "tf_seq2seq_chatbot/rnn_enhancement/cf.py",
        "tf_seq2seq_chatbot/rnn_enhancement/decoding_enhanced.py",
        "tf_seq2seq_chatbot/rnn_enhancement/linear_enhanced.py",
        "tf_seq2seq_chatbot/rnn_enhancement/linear_functions_enhanced.py",
        "tf_seq2seq_chatbot/rnn_enhancement/rnn_cell_enhanced.py",
        "tf_seq2seq_chatbot/rnn_enhancement/rnn_enhanced.py",
        "tf_seq2seq_chatbot/rnn_enhancement/seq2seq_enhanced.py",
        "tf_seq2seq_chatbot/rnn_enhancement/unitary_linear.py",
        "tf_seq2seq_chatbot/rnn_enhancement/unitary_rnn_library.py"
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "tf_seq2seq",
    srcs = [
        "tf_seq2seq_chatbot/configs/config.py",
        "tf_seq2seq_chatbot/lib/data_utils.py",
        "tf_seq2seq_chatbot/lib/chat.py",
        "tf_seq2seq_chatbot/lib/predict.py",
        "tf_seq2seq_chatbot/lib/seq2seq_model.py",
        "tf_seq2seq_chatbot/lib/seq2seq_model_utils.py",
        "tf_seq2seq_chatbot/lib/train.py"
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        ":rnn_enhanced",
        ":Project-RNN-Enhancement"
    ],
)


py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        ":tf_seq2seq"
    ],
)

py_binary(
    name = "test",
    srcs = [
        "test.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        ":tf_seq2seq"
    ],
)

py_binary(
    name = "chat",
    srcs = [
        "chat.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        ":tf_seq2seq"
    ],
)


filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
