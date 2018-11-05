from unittest import TestCase

from hamcrest import assert_that, is_, greater_than

from util.corpus_util import get_corpus


class TestCorpus(TestCase):

    def test_ls_corpus(self):
        corpus = get_corpus('ls')
        corpus.summary()
        assert_that(corpus.corpus_id, is_('ls'))
        assert_that(len(corpus), is_(greater_than(0)))
        entry = corpus[0]
        print(entry.id)
        print(entry.audio_path)
        print(entry.corpus.root_path)
        print(entry.transcript)

    def test_rl_corpus(self):
        corpus = get_corpus('rl')
        corpus.summary()
        assert_that(corpus.corpus_id, is_('rl'))
        assert_that(len(corpus), is_(greater_than(0)))
        entry = corpus[0]
        print(entry.summary())

    def test_cv_corpus(self):
        corpus = get_corpus('cv')
        corpus.summary()
        assert_that(corpus.corpus_id, is_('cv'))
        assert_that(len(corpus), is_(greater_than(0)))
