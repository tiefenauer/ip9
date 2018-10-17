from unittest import TestCase

from hamcrest import assert_that, is_

from util.ctc_util import encode, decode, get_tokens, get_alphabet


class TestEncodingDecoding(TestCase):

    def test_get_alphabet(self):
        alphabet_en = get_alphabet('en')
        alphabet_de = get_alphabet('de')
        assert_that(len(alphabet_en), is_(26))
        assert_that(len(alphabet_de), is_(29))

    def test_get_tokens(self):
        tokens_en = get_tokens('en')
        tokens_de = get_tokens('de')
        assert_that(len(tokens_en), is_(28))
        assert_that(len(tokens_de), is_(31))

    def test_encoding_english(self):
        text = 'foo bar  '
        tokens = get_tokens('en')
        encoded = encode(text, tokens)
        assert_that(encoded, is_([6, 15, 15, 0, 2, 1, 18]), 'leading/trailing spaces should be stripped')

    def test_decoding_english(self):
        int_sequence = [6, 15, 15, 0, 2, 1, 18, 28, 28, 0, 0]
        tokens = get_tokens('en')
        decoded = decode(int_sequence, tokens)
        assert_that(decoded, is_('foo bar'), 'leading/trailing spaces should be stripped')

    def test_encoding_german(self):
        text = 'färöer ü  '
        tokens = get_tokens('de')
        encoded = encode(text, tokens)
        assert_that(encoded, is_([6, 27, 18, 28, 5, 18, 0, 29]), 'leading/trailing spaces should be stripped')

    def test_decoding_german(self):
        int_sequence = [6, 27, 18, 28, 5, 18, 0, 29, 0, 0]
        tokens = get_tokens('de')
        decoded = decode(int_sequence, tokens)
        assert_that(decoded, is_('färöer ü'), 'leading/trailing spaces should be stripped')
