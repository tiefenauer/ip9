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

# my fellow americans
# we prosecutors lawnfarsmenofficers homeland security official
# a law makers joyed me at the white house to address
# a very vicious threat to arcumatities
# the savage gang an as thirteent
# waring loopholes and our laws of alowedcriminalsandgangmembers to break into our country
# example
# under current law unaccompanied alien miners at the botter
# released into american communities no matter wher
# no matter howp
# easy for them because the laws are bad and they have to be changed
# loopallis easily exploded by emstheirteen
# which now operate in at least forty stapes
# in addition to amesthergeen
# many other gangs are breaking into our country routinely
# because a large so wek
# sadr
# t
# this was round table we learned the story of one family right here in washington dc
# hoste a person an their home who then began to recruit their youngson
# as thertene
# when the boys mother tried to stop it to gang member shot her the head
# lining her for alife
# she's lucky she lived but she's pang a very big pric
# in my state of the union i called on congress to immediately close dangerous loopoles and federal law
# in danger our communitie
# and imposed enormous burdens on us taxbear
# my administration as identified three major pririties
# we creating a tafe
# modern and lawful immigration system fully securing the boder
# and being can mygraton
# and can link the resalottery
# same mygrationhisadesesser and very unfair to our country
# as a lottery is something that should have never been allowed in the first place
# people had drew lottery to come into our countary
# kind of a sistem is thate
# time for congress to act and to protect american
# member of congress should choose the side of lawdforcement
# he side of the american peopl
# thats the way it has thobe
# hhu
