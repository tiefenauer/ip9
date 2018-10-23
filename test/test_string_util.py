import string
from unittest import TestCase

from hamcrest import assert_that, is_

from util.ctc_util import get_alphabet
from util.string_util import unidecode_with_alphabet, replace_not_allowed, normalize


class TestStringUtil(TestCase):

    def test_normalize(self):
        assert_that(normalize('Foo', 'en'), is_('foo'), 'normalization should make text lowercase')
        assert_that(normalize('Foo, bar', 'en'), is_('foo bar'), 'normalization should remove punctuation')
        assert_that(normalize('$Foo, bar!', 'en'), is_('foo bar'), 'normalization should strip leading/trailing spaces')

        assert_that(normalize('Färöer Straße', 'en'), is_('faroer strasse'), 'English should only consider ASCII')
        assert_that(normalize('Färöer Straße', 'de'), is_('färöer strasse'), 'German should also consider umlauts')

        assert_that(normalize("won't doesn't", 'en'), is_("won't doesn't"), 'English should keep apostrophe')
        assert_that(normalize("won't doesn't", 'de'), is_("won t doesn t"), 'German should remove apostrophe')

    def test_unidecode_with_alphabet(self):
        ascii_chars = string.ascii_lowercase
        ascii_chars_with_umlauts = string.ascii_lowercase + 'ÄäÖöÜü'
        assert_that(unidecode_with_alphabet('Färöer Straße', ascii_chars), is_('Faroer Strasse'))
        assert_that(unidecode_with_alphabet('Färöer Straße', ascii_chars_with_umlauts), is_('Färöer Strasse'))
        assert_that(unidecode_with_alphabet("won't doesn't Straße", get_alphabet('en')), is_("won't doesn't Strasse"))
        assert_that(unidecode_with_alphabet("won't doesn't Straße", get_alphabet('de')), is_("won't doesn't Strasse"))

    def test_replace_not_allowed(self):
        alphabet_en_space_nums = get_alphabet('en') + ' 0123456789'
        alphabet_de_space_nums = get_alphabet('de') + ' 0123456789'

        assert_that(replace_not_allowed('färöer strasse', alphabet_en_space_nums), is_('f r er strasse'))
        assert_that(replace_not_allowed('färöer strasse', alphabet_en_space_nums, '#'), is_('f#r#er strasse'))
        assert_that(replace_not_allowed('färöer strasse', alphabet_de_space_nums), is_('färöer strasse'))
        assert_that(replace_not_allowed('färöer strasse', alphabet_de_space_nums, '#'), is_('färöer strasse'))