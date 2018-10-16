from unittest import TestCase

from hamcrest import assert_that, is_

from util import lm_util
from util.lm_util import load_lm, load_lm_and_vocab


class TestLMUtil(TestCase):

    def test_splits(self):
        result = lm_util.splits('abc')
        # 4 splits: /abc, a/bc, ab/c, abc/
        assert_that(len(list(result)), is_(4))

    def test_deletes(self):
        result = lm_util.deletes('abc')
        # 3 deletes: ab, ac, bc
        assert_that(len(list(result)), is_(3))

    def test_swaps(self):
        result = lm_util.swaps('abc')
        # 2 transposes: bac, acb
        assert_that(len(list(result)), is_(2))

    def test_replaces(self):
        result = lm_util.replaces('abc')
        # 3*26 = 78 replaces: bbc, cbc, dbc, ..., aac, acc, adc, ... aba, abb, abd, ...
        assert_that(len(list(result)), is_(78))

    def test_inserts(self):
        result = lm_util.inserts('abc')
        # 4*26 = 104 inserts: aabc, babc, ..., aabc, abbc, acbc, ..., abac, abbc, abcc, abdc, ..., abca, abcb, abcc, ...
        assert_that(len(list(result)), is_(104))

    def test_edits_1(self):
        result = lm_util.edits_1('abc')
        # 3 deletes + 2 swaps + 78 replaces + 104 insert
        assert_that(len(list(result)), is_(3 + 2 + 78 + 104))

    def test_load_LM(self):
        lm, vocab = load_lm_and_vocab('../lm/timit_en/libri-timit-lm.klm')
        assert_that(lm, is_(not_(None)))
        assert_that(vocab, is_(not_(None)))

    def test_correction(self):
        lm, lm_vocab = load_lm_and_vocab('../lm/timit_en/libri-timit-lm.klm')
        # 2 insertions (buut has >1 candidate!), 1 deltion, 1 substitution
        text = 'buut langage modelang is aweesome'
        text_corrected = lm_util.correction(text, lm=lm, lm_vocab=lm_vocab)
        assert_that(text_corrected, is_('but language modeling is awesome'))