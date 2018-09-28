# trains a n-Gram LM
# http://victor.chahuneau.fr/notes/2012/07/03/kenlm.html
import sys

from lm.lm_util import process_line

LANGUAGES = {'de': 'german', 'en': 'english'}

if __name__ == '__main__':
    lang = LANGUAGES[sys.argv[1]]
    for line in sys.stdin:
        for sentence in process_line(line, language=lang):
            print(sentence)
