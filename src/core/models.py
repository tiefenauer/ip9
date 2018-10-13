from keras import backend as K
from keras.activations import relu
from keras.initializers import random_normal
from keras.layers import Dense, Bidirectional, Lambda, Input
from keras.layers import LSTM
from keras.layers import TimeDistributed, Dropout
from keras.models import Model
from keras.utils import get_custom_objects


def deep_speech_lstm(n_features=26, n_fc=1024, n_recurrent=1024, n_labels=29):
    """
    Simplified DeepSpeech model using LSTM cells in the recurrent layer. Simplifications/change sto original model:
    - no Dropout
    - LSTM instead of SimpleRNN
    - assuming MFCC instead of spectrograms
    - no translation of raw audio
    - no striding

    DeepSpeech paper:
        https://arxiv.org/abs/1412.5567

    :param n_features: number of input features (typically 26 for MFCC)
    :param n_fc: number of hidden units in FC layers
    :param n_recurrent: number of hidden units in recurrent layer
    :param n_labels: number of labels (for decoding)
    :return:
    """
    get_custom_objects().update({"clipped_relu": clipped_relu})

    init = random_normal(stddev=0.046875)

    # input layer
    features = Input(name='the_input', shape=(None, n_features))

    # First 3 FC layers
    x = TimeDistributed(Dense(n_fc, activation=clipped_relu, kernel_initializer=init, bias_initializer=init),
                        name='FC_1')(features)
    x = TimeDistributed(Dense(n_fc, activation=clipped_relu, kernel_initializer=init, bias_initializer=init),
                        name='FC_2')(x)
    x = TimeDistributed(Dense(n_fc, activation=clipped_relu, kernel_initializer=init, bias_initializer=init),
                        name='FC_3')(x)

    # recurrent layer: BiDirectional LSTM
    x = Bidirectional(
        LSTM(n_recurrent, activation=clipped_relu, return_sequences=True, kernel_initializer='glorot_uniform'),
        merge_mode='sum', name='BRNN')(x)

    # output layer
    y_pred = TimeDistributed(
        Dense(n_labels, activation="softmax", name="y_pred", kernel_initializer=init, bias_initializer=init),
        name="out")(x)

    # Input for CTC
    labels = Input(name='the_labels', shape=[None, ], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Implement CTC loss in lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[features, labels, input_length, label_length], outputs=[loss_out])

    return model


def deep_speech_dropout(n_features=26, n_fc=1024, n_recurrent=1024, n_labels=29):
    """
    Simplified DeepSpeech model with LSTM-cells in recurrent layer. Like deep_speech_lstm, but with dropouts.
    """
    get_custom_objects().update({"clipped_relu": clipped_relu})

    init = random_normal(stddev=0.046875)

    # input layer
    features = Input(name='the_input', shape=(None, n_features))

    # First 3 FC layers
    x = TimeDistributed(Dense(n_fc, activation=clipped_relu, kernel_initializer=init, bias_initializer=init),
                        name='FC_1')(features)
    x = TimeDistributed(Dropout(0.1))(x)
    x = TimeDistributed(Dense(n_fc, activation=clipped_relu, kernel_initializer=init, bias_initializer=init),
                              name='FC_2')(x)
    x = TimeDistributed(Dropout(0.1))(x)
    x = TimeDistributed(Dense(n_fc, activation=clipped_relu, kernel_initializer=init, bias_initializer=init),
                              name='FC_3')(x)
    x = TimeDistributed(Dropout(0.1))(x)

    # recurrent layer: BiDirectional LSTM
    x = Bidirectional(
        LSTM(n_recurrent, activation=clipped_relu, return_sequences=True, dropout=0.1, kernel_initializer='he_normal'),
        merge_mode='sum', name='BRNN')(x)
    x = TimeDistributed(Dropout(0.1))(x)

    # output layer
    y_pred = TimeDistributed(
        Dense(n_labels, activation="softmax", name="y_pred", kernel_initializer=init, bias_initializer=init),
        name="out")(x)

    # Input for CTC
    labels = Input(name='the_labels', shape=[None, ], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Implement CTC loss in lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[features, labels, input_length, label_length], outputs=[loss_out])

    return model


def clipped_relu(x):
    return relu(x, max_value=20)


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
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc(y_true, y_pred):
    return y_pred
