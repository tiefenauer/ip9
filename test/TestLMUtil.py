from unittest import TestCase

from hamcrest import is_, not_, assert_that

from util import lm_util
from util.lm_util import load_LM


class TestLMUtil(TestCase):

    def test_load_LM(self):
        lm, vocab = load_LM('../lm/timit_en/libri-timit-lm.klm')
        assert_that(lm, is_(not_(None)))
        assert_that(vocab, is_(not_(None)))

    def test_correction(self):
        lm, lm_vocab = load_LM('../lm/timit_en/libri-timit-lm.klm')
        # 2 insertions (buut has >1 candidate!), 1 deltion, 1 substitution
        text = 'buut langage modelang is aweesome'
        text_corrected = lm_util.correction(text, lm=lm, lm_vocab=lm_vocab)
        assert_that(text_corrected, is_('but language modeling is awesome'))
