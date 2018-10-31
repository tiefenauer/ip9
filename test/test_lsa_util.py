from unittest import TestCase

from hamcrest import assert_that, is_

from util.lsa_util import needle_wunsch, snap_to_closest_word_boundary


class TestLsaUtil(TestCase):

    def test_needle_wunsch(self):
        needle_wunsch("GCATGCU", "GATTACA", [0])
        needle_wunsch("ich bin ein berliner", "ach bene ain berlinör", [0, 4, 9, 13])

    def test_snap_to_closest_word_boundary(self):
        text = 'foo bar foobar'
        assert_that(snap_to_closest_word_boundary(1, text), is_(0), 'f|oo bar foobar')
        assert_that(snap_to_closest_word_boundary(0, text), is_(0), '|foo bar foobar')

        assert_that(snap_to_closest_word_boundary(5, text), is_(4), 'foo b|ar foobar')
        assert_that(snap_to_closest_word_boundary(6, text), is_(8), 'foo ba|r foobar')
        assert_that(snap_to_closest_word_boundary(11, text), is_(8), 'foo bar foo|bar')

        assert_that(snap_to_closest_word_boundary(13, text), is_(14), 'foo bar fooba|r')
        assert_that(snap_to_closest_word_boundary(14, text), is_(14), 'foo bar foobar|')