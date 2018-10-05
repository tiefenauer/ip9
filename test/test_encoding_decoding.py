from unittest import TestCase

from hamcrest import assert_that, is_

from util.ctc_util import encode, decode


class TestEncodingDecoding(TestCase):

    def test_encoding(self):
        text = 'foo bar'
        encoded = encode(text)
        encoded_ref = text_to_int_sequence(text)
        assert_that(encoded, is_(encoded_ref))

    def test_decoding(self):
        int_sequence = [6, 15, 15, 0, 2, 1, 18, 28, 28]
        decoded = decode(int_sequence)
        decoded_ref = ''.join(int_to_text_sequence(int_sequence))
        assert_that(decoded, is_(decoded_ref))


def text_to_int_sequence(text):
    """
    LEGACY CODE
    Use a character map and convert text to an integer sequence
    :param text:
    :return:
    """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence


def int_to_text_sequence(seq):
    """
    LEGACY CODE
     Use a index map and convert int to a text sequence
        >>> from utils import int_to_text_sequence
        >>> a = [2,22,10,11,21,2,13,11,6,1,21,2,8,20,17]
        >>> b = int_to_text_sequence(a)
    :param seq:
    :return:
    """
    text_sequence = []
    for c in seq:
        if c == 28:  # ctc/pad char
            ch = ''
        else:
            ch = index_map[c]
        text_sequence.append(ch)
    return text_sequence


char_map_str = """
<SPACE> 0
a 1
b 2
c 3
d 4
e 5
f 6
g 7
h 8
i 9
j 10
k 11
l 12
m 13
n 14
o 15
p 16
q 17
r 18
s 19
t 20
u 21
v 22
w 23
x 24
y 25
z 26
' 27

"""
char_map = {}
index_map = {}

for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch

index_map[0] = ' '
