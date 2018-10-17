from unittest import TestCase

from util.lsa_util import needle_wunsch


class TestLsaUtil(TestCase):

    def test_needle_wunsch(self):
        needle_wunsch("GCATGCU", "GATTACA")
        needle_wunsch("ich bin ein berliner", "ach bene ain berlinör")
