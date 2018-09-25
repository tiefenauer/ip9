import argparse
import string

import nltk
from tqdm import tqdm
from unidecode import unidecode

from util.string_util import remove_multi_spaces


def main(input, output, min_length, num_lines=None):
    with open(input, 'r') as f_in, open(output, 'w') as f_out:
        # process non-empty lines that don't start with tabulator (oftern literature and links, not sentences)
        lines = (line for line in f_in if not line.startswith('\t') and not line.startswith('\n'))
        for line in tqdm(lines, total=num_lines, unit=' lines'):
            sentences = nltk.sent_tokenize(line.strip(), language='german')
            for sentence in sentences:
                words = [w.lower() for w in nltk.word_tokenize(sentence, language='german') if w not in string.punctuation]
                if len(words) >= min_length:
                    sentence_normalized = unidecode(remove_multi_spaces(' '.join(words).lower()))
                    # print(sentence_normalized)
                    f_out.write(sentence_normalized + '.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Tokenize inputfile to list of sentences""")
    parser.add_argument('--input', type=str, help='path to input file to read from')
    parser.add_argument('--output', type=str, help='path to output file to read to')
    parser.add_argument('--min_length', type=int, default=4,
                        help='(optional) minimum number of words in a sentence. (Default: 4)')
    parser.add_argument('--num_lines', type=int, default=None,
                        help='(optional) number of lines in the input file. If set a progress bar will be shown.')
    args = parser.parse_args()
    main(args.input, args.output, args.min_length, args.num_lines)
