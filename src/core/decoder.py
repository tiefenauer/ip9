from keras import backend as K

from util.ctc_util import decode, get_tokens


class Decoder(object):
    """
    Base class for CTC-Decoders
    """
    def __init__(self, model, language, greedy):
        """
        Initialize the decoded
        :param model: The trained Keras model that made the inferences
        :param language: language to use for decoding. This will affect the alphabet used for decoding
        :param greedy: whether a best-path (True) or a beam search approch (False) shall be used
        """
        self.ctc_input = model.get_layer('ctc').input[0]
        self.input_data = model.get_layer('the_input').input
        self.test_func = K.function([self.input_data, K.learning_phase()], [self.ctc_input])
        self.greedy = greedy
        self.strategy = 'best-path' if greedy else 'beam search'
        self.tokens = get_tokens(language)

    def decode(self, batch_input, batch_input_lengths):
        y_pred = self.test_func([batch_input])[0]
        decoded_int = K.get_value(K.ctc_decode(y_pred, input_length=batch_input_lengths, greedy=self.greedy)[0][0])
        return [decode(int_seq, self.tokens) for int_seq in decoded_int]


class BestPathDecoder(Decoder):
    """
    Decodes a batch of encoded labels greedily (best-path)
    """

    def __init__(self, model, language):
        super().__init__(model, language, True)


class BeamSearchDecoder(Decoder):
    """
    Decodes a batch of encoded labels approximately optimal (beam search)
    """

    def __init__(self, model, language):
        super().__init__(model, language, False)
