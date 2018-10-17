from keras import backend as K

from util.ctc_util import decode, get_tokens


class Decoder(object):
    def __init__(self, model, language, greedy):
        self.ctc_input = model.get_layer('ctc').input[0]
        self.input_data = model.get_layer('the_input').input
        self.test_func = K.function([self.input_data, K.learning_phase()], [self.ctc_input])
        self.greedy = greedy
        self.strategy = 'best-path' if greedy else 'beam search'
        self.tokens = get_tokens(language)

    def decode(self, batch_input, batch_input_lengths):
        # https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/
        y_pred = self.test_func([batch_input])[0]
        decoded_int = K.get_value(K.ctc_decode(y_pred, input_length=batch_input_lengths, greedy=self.greedy)[0][0])
        return [decode(int_seq, self.tokens) for int_seq in decoded_int]


class BestPathDecoder(Decoder):

    def __init__(self, model, language):
        super().__init__(model, language, True)


class BeamSearchDecoder(Decoder):

    def __init__(self, model, language):
        super().__init__(model, language, False)
