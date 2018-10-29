from unittest import TestCase

from hamcrest import assert_that, is_, greater_than

from corpus.corpus import LibriSpeechCorpus


class TestCorpus(TestCase):

    def test_ls_corpus(self):
        corpus = LibriSpeechCorpus('/media/daniel/IP9/corpora/librispeech/index.csv')
        corpus.summary()
        assert_that(corpus.corpus_id, is_('ls'))
        assert_that(corpus.name, is_('LibriSpeech'))
        assert_that(len(corpus), is_(greater_than(0)))
        assert_that(len(corpus.entries), is_(greater_than(0)))
        assert_that(len(corpus.segments()), is_(greater_than(0)))
