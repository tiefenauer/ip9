from unittest import TestCase

from hamcrest import greater_than, is_, assert_that

from core.lm_vocabulary import Vocabulary


class TestVocabulary(TestCase):

    def test_load(self):
        file = '/media/daniel/IP9/lm/wiki_de/tmp/wiki_de.counts'
        vocab = Vocabulary(file)
        assert_that(len(vocab), is_(greater_than(0)))
        assert_that(len(vocab.words), is_(len(vocab)))
        assert_that(len(vocab.counts), is_(len(vocab)))