"""
Contains various helper functions to create/train a BRNN
"""

import tensorflow as tf
from keras import Input, Model, backend as K
from keras.activations import relu
from keras.initializers import random_normal
from keras.layers import TimeDistributed, Dense, Activation, Dropout, Bidirectional, SimpleRNN, Lambda
from keras.utils import get_custom_objects


def deep_speech_model(num_features, num_hidden=2048, rnn_size=512, dropout=0.1, num_classes=28):
    """
    Deep Speech model with architecture as described in the paper:
        5 Layers (3xFC + 1xBRNN + 1xFC) with Dropout applied to FC layer
    The output contains 28 classes: a..z, space, blank

    Differences to the original setup:
        * We are not translating the raw audio files by 5 ms (Sec 2.1 in [1])
        * We are not striding the RNN to halve the timesteps (Sec 3.3 in [1])
        * We are not using frames of context

    Reference: [1] https://arxiv.org/abs/1412.5567
    """
    get_custom_objects().update({"clipped_relu": clipped_relu})
    K.set_learning_phase(1)

    input_data = Input(name='the_input', shape=(None, num_features))

    # 3 FC layers with clipped ReLU followed by a Dropout
    init = random_normal(stddev=0.046875)
    x = TimeDistributed(
        Dense(num_hidden, name='fc1', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(
        input_data)
    # o = TimeDistributed(Activation(clipped_relu))(o)
    x = TimeDistributed(Dropout(dropout))(x)
    x = TimeDistributed(
        Dense(num_hidden, name='fc2', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)
    # o = TimeDistributed(Activation(clipped_relu))(o)
    x = TimeDistributed(Dropout(dropout))(x)
    x = TimeDistributed(
        Dense(num_hidden, name='fc3', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)
    # o = TimeDistributed(Activation(clipped_relu))(o)
    x = TimeDistributed(Dropout(dropout))(x)

    # BRNN layer
    x = Bidirectional(SimpleRNN(rnn_size, return_sequences=True, activation=clipped_relu, dropout=dropout,
                                kernel_initializer='he_normal'), merge_mode='sum')(x)
    # o = TimeDistributed(Dropout(dropout))(o)

    # another FC layer with dropout
    x = TimeDistributed(Dense(num_hidden))(x)
    x = TimeDistributed(Activation(clipped_relu))(x)
    x = TimeDistributed(Dropout(dropout))(x)

    # Output layer
    y_pred = TimeDistributed(
        Dense(num_classes, name='y_pred', kernel_initializer=init, bias_initializer=init, activation='softmax'))(x)

    # Input placeholders for true labels and input sequence lengths (need to be fed during training/validation)
    labels = Input(name='the_labels', shape=(None,), dtype='int32', sparse=True)
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Output layer for decoded label (values are integers)
    # dec = Lambda(decoder_lambda_func, arguments={'is_greedy': False}, name='decoder')
    # y_pred = dec([output, inputs_length])

    # CTC-loss
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])

    return Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)


def decoder_lambda_func(args, is_greedy=False, beam_width=100, top_paths=1, merge_repeated=True):
    """
    CTC-Decoder function. Decodes a batch by evaluating sequences of probabilities either using best-path
    (greedy) or beam-search, depending on the input arguments.

    Since this is used in a lambda layer and Keras calls functions with an arguments tuple (a,b,c,...)
    and not *(a,b,c,...) the function's parameters must be unpacked inside the function.

    The parameters are as follows:
    :param y_pred: (batch_size, T_x, num_classes)
        tensor containing the probabilities of each character for each time step of each sequence in the batch
        - batch_size: number of sequences in batch
        - T_x: number of time steps in the current batch (maximum number of time steps over all batch sequences)
        - num_classes: number of characters (i.e. number of probabilities per time step)
    :param seq_len: (batch_size,)
        tensor containing the lenghts of each sequence in the batch (they have been padded)
    :param is_greedy: Flag for best-path or beam search decoding
        If True decoding will be done following the character with the highest probability in each
        time step (best-path decoding).
        If False beam search decoding will be done. The beam width, number of top paths
        and whether to merge repeated prefixes are specified in their respective kwargs
        --> see https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder
    :param beam_width: number of paths to follow simultaneously. Only used if is_greedy=False
    :param top_paths: number of paths to return. Only used if is_greedy=False
    :param merge_repeated: whether to merge/sum up partial paths that lead to the same prefix
    :return A sparse tensor with the {top_paths} decoded sequences
    """
    # hack: we need to import tensorflow here again to make keras.engine.saving.load_model work...
    import tensorflow as tf

    y_pred, seq_len = args

    # using K.ctc_decode does not work because it returns a dense Tensor and we need a sparse tensor to
    # calculate LER with tf.edit_distance !
    # decoded = K.ctc_decode(y_pred, seq_len[:, 0], greedy=is_greedy, beam_width=beam_width, top_paths=top_paths)
    # prediction = decoded[0][0]
    # prediction_sparse = to_sparse(prediction)
    # return prediction_sparse

    seq_len = tf.cast(seq_len[:, 0], tf.int32)
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])  # time major

    if is_greedy:
        return tf.nn.ctc_greedy_decoder(y_pred, seq_len)[0][0]
    return tf.nn.ctc_beam_search_decoder(y_pred, seq_len, beam_width, top_paths, merge_repeated)[0][0]


def to_sparse(x):
    """
    https://stackoverflow.com/questions/42127505/tensorflow-dense-to-sparse
    :param x:
    :return:
    """
    idx = tf.where(tf.not_equal(x, 0))
    sparse = tf.SparseTensor(idx, tf.gather_nd(x, idx), x.get_shape())
    return sparse


def ctc_lambda_func(args):
    """
    CTC cost function. Calculates the CTC cost over a whole batch.

    Since this is used in a lambda layer and Keras calls functions with an arguments tuple (a,b,c,...)
    and not *(a,b,c,...) the function's parameters must be unpacked inside the function.

    The parameters are as follows:
    :param y_pred: (batch_size, T_x, num_classes)
        tensor containing the probabilities of each character for each time step of each sequence in the batch
        - batch_size: number of sequences in batch
        - T_x: number of time steps in the current batch (maximum number of time steps over all batch sequences)
        - num_classes: number of characters (i.e. number of probabilities per time step)
    :param labels: (batch_size, T_x)
        tensor containing the true labels (encoded as integers)
    :param inputs_length: (batch_size,)
        tensor containing the lenghts of each sequence in the batch (they have been padded)
    :return: tensor for the CTC-loss
    """

    # hack: we need to import tensorflow here again to make keras.engine.saving.load_model work...
    import tensorflow as tf
    y_pred, labels, input_length, label_length = args
    labels = tf.sparse_tensor_to_dense(labels)
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def clipped_relu(x):
    return relu(x, max_value=20)


def ctc_dummy_loss(y_true, y_pred):
    """
    CTC-Loss for Keras. Because Keras has no CTC-loss built-in, we create this dummy.
    We simply use the output of the RNN, which corresponds to the CTC loss.
    """
    return y_pred


def decoder_dummy_loss(y_true, y_pred):
    """
    Loss of the decoded sequence. Since this is not optimized, we simply return a zero-Tensor
    """
    return K.zeros((1,))


def ler(y_true, y_pred, **kwargs):
    """
    LER-Loss (see https://www.tensorflow.org/api_docs/python/tf/edit_distance)
    """
    return tf.reduce_mean(tf.edit_distance(y_pred, y_true, normalize=True, **kwargs))
