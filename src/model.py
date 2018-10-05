from keras import backend as K
from keras.activations import relu
from keras.initializers import random_normal
from keras.layers import Dense, Bidirectional, Lambda, Input
from keras.layers import LSTM
from keras.layers import TimeDistributed, Dropout
from keras.models import Model
from keras.utils import get_custom_objects


def selu(x):
    # from Keras 2.0.6 - does not exist in 2.0.4
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
       x: A tensor or variable to compute the activation function for.
    # References
       - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * K.elu(x, alpha)


def clipped_relu(x):
    return relu(x, max_value=20)


# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    # hack for load_model

    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''

    # print("CTC lambda inputs / shape")
    # print("y_pred:",y_pred.shape)  # (?, 778, 30)
    # print("labels:",labels.shape)  # (?, 80)
    # print("input_length:",input_length.shape)  # (?, 1)
    # print("label_length:",label_length.shape)  # (?, 1)

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc(y_true, y_pred):
    return y_pred


def ds1(input_dim=26, fc_size=1024, rnn_size=1024, output_dim=29):
    """ DeepSpeech 1 Implementation without dropout

    Architecture:
        Input MFCC TIMEx26
        3 Fully Connected using Clipped Relu activation function
        1 BiDirectional LSTM
        1 Fully connected Softmax

    Details:
        - Removed Dropout on this implementation
        - Uses MFCC's rather paper's 80 linear spaced log filterbanks
        - Uses LSTM's rather than SimpleRNN
        - No translation of raw audio by 5ms
        - No stride the RNN

    References:
        https://arxiv.org/abs/1412.5567
    """
    # hack to get clipped_relu to work on bidir layer
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"clipped_relu": clipped_relu})

    input_data = Input(name='the_input', shape=(None, input_dim))  # >>(?, 778, 26)

    init = random_normal(stddev=0.046875)

    # First 3 FC layers
    x = TimeDistributed(
        Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(
        input_data)  # >>(?, 778, 2048)
    x = TimeDistributed(
        Dense(fc_size, name='fc2', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(
        x)  # >>(?, 778, 2048)
    x = TimeDistributed(
        Dense(fc_size, name='fc3', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(
        x)  # >>(?, 778, 2048)

    # # Layer 4 BiDirectional RNN - note coreml only supports LSTM BIDIR
    x = Bidirectional(LSTM(rnn_size, return_sequences=True, activation=clipped_relu,
                           kernel_initializer='glorot_uniform', name='birnn'), merge_mode='sum')(x)  #

    # Layer 5+6 Time Dist Layer & Softmax

    # x = TimeDistributed(Dense(fc_size, activation=clipped_relu))(x)
    y_pred = TimeDistributed(
        Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax"),
        name="out")(x)
    # y_pred = Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax")(x)

    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None, ], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model


def deep_speech_dropout(input_dim=26, fc_size=2048, rnn_size=512, output_dim=29):
    """ DeepSpeech 1 Implementation with Dropout

    Architecture:
        Input MFCC TIMEx26
        3 Fully Connected using Clipped Relu activation function
        3 Dropout layers between each FC
        1 BiDirectional LSTM
        1 Dropout applied to BLSTM
        1 Dropout applied to FC dense
        1 Fully connected Softmax

    Details:
        - Uses MFCC's rather paper's 80 linear spaced log filterbanks
        - Uses LSTM's rather than SimpleRNN
        - No translation of raw audio by 5ms
        - No stride the RNN

    Reference:
        https://arxiv.org/abs/1412.5567
    """
    get_custom_objects().update({"clipped_relu": clipped_relu})
    K.set_learning_phase(1)

    # input layer
    input_data = Input(name='the_input', shape=(None, input_dim))

    # 3 x FC layer
    init = random_normal(stddev=0.046875)
    x = TimeDistributed(
        Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(input_data)
    x = TimeDistributed(Dropout(0.1))(x)
    x = TimeDistributed(
        Dense(fc_size, name='fc2', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)
    x = TimeDistributed(Dropout(0.1))(x)
    x = TimeDistributed(
        Dense(fc_size, name='fc3', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)
    x = TimeDistributed(Dropout(0.1))(x)

    # RNN layer
    x = Bidirectional(LSTM(rnn_size, return_sequences=True, activation=clipped_relu, dropout=0.1,
                           kernel_initializer='he_normal', name='birnn'), merge_mode='sum')(x)
    x = TimeDistributed(Dropout(0.1))(x)
    y_pred = TimeDistributed(
        Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax"),
        name="out")(x)

    # Change shape
    labels = Input(name='the_labels', shape=[None, ], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model
