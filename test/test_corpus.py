from unittest import TestCase

from hamcrest import assert_that, is_, greater_than

from corpus.corpus import LibriSpeechCorpus


class TestCorpus(TestCase):

    def test_corpus_entries(self):
        corpus = LibriSpeechCorpus('/media/daniel/IP9/corpora/librispeech/index.csv')
        corpus.summary()
        assert_that(corpus.corpus_id, is_('ls'))
        assert_that(corpus.corpus_id, is_('LibriSpeech'))
        assert_that(len(corpus), is_(greater_than()))
