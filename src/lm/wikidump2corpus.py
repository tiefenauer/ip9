"""
Creates a corpus from Wikipedia dump file.
from: https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
Inspired by:
https://github.com/panyang/Wikipedia_Word2vec/blob/master/v1/process_wiki.py
"""
import argparse

import nltk
from gensim.corpora import WikiCorpus
from tqdm import tqdm

from lm.lm_corpus_util import normalize_word


def main(input_file, output_file):
    """Convert Wikipedia xml dump file to text corpus"""
    print(f'creating wiki text corpus from {input_file} and saving to {output_file}')
    print(f'opening {input_file} to create create word->word_id mapping. This will take ~8h on full wiki...', end='')
    wiki = WikiCorpus(input_file)
    print(f'...done!')

    with open(output_file, 'w') as f_out:
        print(f'starting text extraction..')
        for i, tokens in tqdm(enumerate(wiki.get_texts()), unit=' articles'):
            f_out.write(tokens2string(tokens) + '\n')
        print(f'Processing complete! Processed {i+1} articles')


def tokenize_sentences(text: str, token_min_len: int, token_max_len: int, lower: bool):
    return nltk.sent_tokenize(text, 'german')


def tokens2string(tokens):
    return ' '.join(t for t in (normalize_word(token) for token in tokens) if t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""convert Wikipedia dump to text corpus file""")
    parser.add_argument('--input', type=str, help='path to input file to read from')
    parser.add_argument('--output', type=str, help='path to output file to read to')
    args = parser.parse_args()
    main(args.input, args.output)
