from unittest import TestCase

from hamcrest import assert_that, is_, not_

from util import lm_util
from util.ctc_util import get_alphabet
from util.lm_util import ler, ler_norm, wer, wer_norm, load_lm_and_vocab


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
        alphabet = get_alphabet('en')
        result = lm_util.replaces('abc', alphabet)
        # 3*26 = 78 replaces: bbc, cbc, dbc, ..., aac, acc, adc, ... aba, abb, abd, ...
        assert_that(len(list(result)), is_(81))

    def test_inserts(self):
        alphabet = get_alphabet('en')
        result = lm_util.inserts('abc', alphabet)
        # 4*26 = 104 inserts: aabc, babc, ..., aabc, abbc, acbc, ..., abac, abbc, abcc, abdc, ..., abca, abcb, abcc, ...
        assert_that(len(list(result)), is_(108))

    def test_edits_1(self):
        alphabet = get_alphabet('en')
        result = lm_util.edits_1('abc', alphabet)
        # 3 deletes + 2 swaps + 3*27 replaces + 4*27- insert
        assert_that(len(list(result)), is_(3 + 2 + 81 + 108))

    def test_load_lm(self):
        lm, vocab = load_lm_and_vocab('/media/daniel/IP9//lm/timit_en/libri-timit-lm.klm')
        assert_that(lm, is_(not_(None)))
        assert_that(vocab, is_(not_(None)))

    def test_correction(self):
        lm, lm_vocab = load_lm_and_vocab('/media/daniel/IP9/lm/timit_en/libri-timit-lm.klm')
        # 2 insertions (buut has >1 candidate!), 1 deltion, 1 substitution
        text = 'buut langage modelang is aweesome'
        text_corrected = lm_util.correction(text, 'en', lm=lm, lm_vocab=lm_vocab)
        assert_that(text_corrected, is_('but language modeling is awesome'))

    def test_ler(self):
        ground_truth = 'and i put the vice president in charge of mission control'
        inference_no_lm = 'ii put he bice president in charge of mission control'
        inference_lm = 'i put the vice president in charge of mission control'
        assert_that(ler(ground_truth, inference_no_lm), is_(6))
        assert_that(ler(ground_truth, inference_lm), is_(4))

    def test_ler_norm(self):
        ground_truth = 'and i put the vice president in charge of mission control'
        inference_no_lm = 'ii put he bice president in charge of mission control'
        inference_lm = 'i put the vice president in charge of mission control'
        assert_that(ler_norm(ground_truth, inference_no_lm), is_(6 / len(ground_truth)))
        assert_that(ler_norm(ground_truth, inference_lm), is_(4 / len(ground_truth)))

    def test_wer(self):
        ground_truth = 'and i put the vice president in charge of mission control'
        inference_no_lm = 'ii put he bice president in charge of mission control'
        inference_lm = 'i put the vice president in charge of mission control'
        assert_that(wer(ground_truth, inference_no_lm), is_(4))
        assert_that(wer(ground_truth, inference_lm), is_(1))

    def test_wer_norm(self):
        ground_truth = 'and i put the vice president in charge of mission control'
        inference_no_lm = 'ii put he bice president in charge of mission control'
        inference_lm = 'i put the vice president in charge of mission control'
        assert_that(wer_norm(ground_truth, inference_no_lm), is_(4 / len(ground_truth.split())))
        assert_that(wer_norm(ground_truth, inference_lm), is_(1 / len(ground_truth.split())))
