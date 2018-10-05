import string

import numpy as np

# 29 target classes
# <space> = 0, a=1, b=2, ..., z=26, '=27, _ (padding token) = 28
SPACE_TOKEN = '<space>'
ALLOWED_CHARS = string.ascii_lowercase  # add umlauts here
CHAR_TOKENS = ' ' + ALLOWED_CHARS + '\''


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


def encode(text):
    return [encode_token(token) for token in tokenize(text)]


def encode_token(token):
    return 0 if token == SPACE_TOKEN else CHAR_TOKENS.index(token)


def decode(tokens):
    return ''.join([decode_token(x) for x in tokens])


def decode_token(ind):
    return '' if ind in [-1, len(CHAR_TOKENS)] else CHAR_TOKENS[ind]
