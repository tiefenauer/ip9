import string

import numpy as np

SPACE_TOKEN = '<space>'


def get_alphabet(language):
    if language == 'de':
        return string.ascii_lowercase + 'äöü'
    return string.ascii_lowercase + '\''


def get_tokens(language):
    # English has 31 target labels  (+ blank token):
    #   <space>=0, a=1, b=2, ..., z=26, ä=27, ö=28, ü=29, '=30, _ (padding token)=31
    # German has 28 target labels (+ blank token):
    #   <space>=0, a=1, b=2, ..., z=26, '=27, _ (padding token)=28
    return ' ' + get_alphabet(language)


def tokenize(text):
    """Splits a text into tokens.
    The text must only contain the lowercase characters a-z and digits. This must be ensured prior to calling this
    method for performance reasons. The tokens are the characters in the text. A special <space> token is added between
    the words. Since numbers are a special case (e.g. '1' and '3' are 'one' and 'three' if pronounced separately, but
    'thirteen' if the text is '13'), digits are mapped to the special '<unk>' token.
    """

    text = text.replace(' ', '  ')
    words = text.split(' ')

    tokens = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in words])
    return tokens


def encode(text, tokens):
    return [encode_token(token, tokens) for token in tokenize(text.strip())]


def encode_token(token, tokens):
    return 0 if token == SPACE_TOKEN else tokens.index(token)


def decode(int_seq, tokens):
    return ''.join([decode_token(x, tokens) for x in int_seq]).strip()


def decode_token(ind, tokens):
    return '' if ind in [-1, len(tokens)] else tokens[ind]
