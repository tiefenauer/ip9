from unittest import TestCase

from hamcrest import assert_that, is_
from util.gsa_util import needle_wunsch, snap_to_closest_word_boundary, snap_left, snap_right


class TestLsaUtil(TestCase):

    def test_needle_wunsch(self):
        alignments = needle_wunsch("GCATGCU", "GATTACA", [(0, 7)])
        print(alignments)
        alignments = needle_wunsch("ich bin ein berliner", "ach bene ain berlinör", [(0, 3), (4, 7), (8, 11), (12, 20)])
        print(alignments)

    def test_snap_left(self):
        text = 'foo    bar'
        assert_that(snap_left(0, text), is_(0), '|foo    bar --> no change')
        assert_that(snap_left(1, text), is_(0), 'f|oo    bar --> snap to closest word boundary')
        assert_that(snap_left(2, text), is_(7), 'fo|o    bar --> snap to closest word boundary + skip spaces')
        assert_that(snap_left(3, text), is_(7), 'foo|    bar --> skip spaces')
        assert_that(snap_left(4, text), is_(7), 'foo |   bar --> skip spaces')
        assert_that(snap_left(7, text), is_(7), 'foo    |bar --> no change')
        assert_that(snap_left(8, text), is_(7), 'foo    b|ar --> snap to closest word boundary')
        assert_that(snap_left(9, text), is_(10), 'foo    ba|r --> snap to closest word boundary')

        # verify that special chars are skipped
        text = 'foo  ***    bar'
        for i in range(3,14):
            left = snap_left(i, text)
            assert_that(text[left:], is_('bar'))

    def test_snap_right(self):
        text = 'foo    bar'
        assert_that(snap_right(0, text), is_(0), '|foo    bar --> no change')
        assert_that(snap_right(1, text), is_(0), 'f|oo    bar --> snap to closest word boundary')
        assert_that(snap_right(2, text), is_(3), 'fo|o    bar --> snap to closest word boundary')
        assert_that(snap_right(3, text), is_(3), 'foo|    bar --> no change')
        assert_that(snap_right(4, text), is_(3), 'foo |   bar --> skip spaces')
        assert_that(snap_right(7, text), is_(3), 'foo    |bar --> skip spaces')
        assert_that(snap_right(8, text), is_(3), 'foo    b|ar --> snap to closest word boundary + skip spaces')
        assert_that(snap_right(9, text), is_(10), 'foo    ba|r --> snap to closest word boundary')

        # verify that special chars are skipped
        text = 'foo  ***    bar'
        for i in range(3, 14):
            right = snap_right(i, text)
            assert_that(text[:right], is_('foo'))

    def test_snap_to_closest_word_boundary(self):
        text = 'foo bar foobar'
        assert_that(snap_to_closest_word_boundary(1, text), is_(0), 'f|oo bar foobar')
        assert_that(snap_to_closest_word_boundary(0, text), is_(0), '|foo bar foobar')

        assert_that(snap_to_closest_word_boundary(5, text), is_(4), 'foo b|ar foobar')
        assert_that(snap_to_closest_word_boundary(6, text), is_(7), 'foo ba|r foobar')
        assert_that(snap_to_closest_word_boundary(11, text), is_(8), 'foo bar foo|bar')

        assert_that(snap_to_closest_word_boundary(13, text), is_(14), 'foo bar fooba|r')
        assert_that(snap_to_closest_word_boundary(14, text), is_(14), 'foo bar foobar|')
