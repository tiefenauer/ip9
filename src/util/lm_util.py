# this file is an adaptation from the work at mozilla deepspeech github.com/mozilla/DeepSpeech
import itertools
import kenlm
from heapq import heapify
from os.path import abspath, exists, dirname, join, splitext

import numpy as np
from pattern3.metrics import levenshtein

from util.ctc_util import get_alphabet

# the LER is just the Levenshtein/edit distance
ler = levenshtein


def wer(ground_truth, prediction):
    """
    Calculates the Word Error Rate (WER) between two strings as the edit distance on word level
    """
    return ler(ground_truth.split(), prediction.split())


def ler_norm(ground_truth, prediction):
    """
    Calculates the normalized LER by dividing the LER by the length of the longer string. The result will be in [0,1]
    """
    return ler(ground_truth, prediction) / float(max(len(ground_truth), 1))


def wer_norm(ground_truth, prediction):
    """
    Calculates the normalized WER by dividing the WER by the length of the ground truth
    """
    return wer(ground_truth, prediction) / float(len(ground_truth.split()))


def wers(ground_truths, predictions):
    assert len(ground_truths) > 0, f'ERROR: no ground truths provided!'
    assert len(ground_truths) == len(predictions), f'ERROR: # of ground truths does not match # of predictions!'
    return np.array([wer_norm(ground_truth, pred) for (ground_truth, pred) in zip(ground_truths, predictions)])


def lers(ground_truths, predictions):
    assert len(ground_truths) > 0, f'ERROR: no ground truths provided!'
    assert len(ground_truths) == len(predictions), f'ERROR: # of ground truths does not match # of predictions!'
    return np.array([ler(ground_truth, pred) for (ground_truth, pred) in zip(ground_truths, predictions)])


def lers_norm(ground_truths, predictions):
    assert len(ground_truths) > 0, f'ERROR: no ground truths provided!'
    assert len(ground_truths) == len(predictions), f'ERROR: # of ground truths does not match # of predictions!'
    return np.array([ler_norm(ground_truth, pred) for (ground_truth, pred) in zip(ground_truths, predictions)])


def load_lm_and_vocab(lm_path, vocab_path=None):
    lm = load_lm(lm_path)
    lm_abs_path = abspath(lm_path)
    if vocab_path:
        lm_vocab_abs_path = abspath(vocab_path)
    else:
        lm_vocab_abs_path = abspath(join(dirname(lm_abs_path), splitext(lm_abs_path)[0] + '.vocab'))
        if not exists(lm_vocab_abs_path):
            raise ValueError(f'ERROR: LM vocabulary not set and no vocabulary found at {lm_vocab_abs_path}')
    lm_vocab = load_vocab(lm_vocab_abs_path)
    return lm, lm_vocab


def load_lm(lm_path):
    global LANGUAGE_MODELS
    if lm_path in LANGUAGE_MODELS:
        lm = LANGUAGE_MODELS[lm_path]
        print(f'using cached LM ({lm.order}-gram):')
        return lm

    lm_abs_path = abspath(lm_path)
    if not exists(lm_abs_path):
        raise ValueError(f'ERROR: LM not found at {lm_abs_path}')

    print(f'loading LM from {lm_abs_path}...', end='')
    lm = kenlm.Model(lm_abs_path)
    print(f'done! Loaded {lm.order}-gram model.')
    LANGUAGE_MODELS[lm_path] = lm
    return lm


def load_vocab(vocab_path):
    global LM_VOCABS
    if vocab_path in LM_VOCABS:
        lm_vocab = LM_VOCABS[vocab_path]
        print(f'using cached LM vocab ({len(lm_vocab)} words)')
        return lm_vocab

    lm_vocab_abs_path = abspath(vocab_path)
    if not exists(lm_vocab_abs_path):
        raise ValueError(f'ERROR: LM vocabulary not found at {lm_vocab_abs_path}')

    with open(lm_vocab_abs_path) as vocab_f:
        print(f'loading LM vocab from {lm_vocab_abs_path}...', end='')
        lm_vocab = set(vocab_f.read().split())
        print(f'done! Loaded {len(lm_vocab)} words.')
        LM_VOCABS[vocab_path] = lm_vocab
    return lm_vocab


def score(word_list, lm):
    """
    Use LM to calculate a log10-based probability for a given sentence (as a list of words)
    :param word_list:
    :return:
    """
    return lm.score(' '.join(word_list), bos=False, eos=False)


def correction(sentence, language, lm=None, vocab=None):
    """
    Get most probable spelling correction for a given sentence.
    :param sentence:
    :return:
    """
    assert language in ['en', 'de', 'fr', 'it', 'es'], 'language must be one of [\'en\', \'de\']'
    if not lm or not vocab:
        return sentence
    alphabet = get_alphabet(language)
    beam_width = 1024
    layer = [(0, [])]  # list of (score, 2-gram)-pairs
    for word in sentence.split():
        layer = [(-score(node + [word_c], lm), node + [word_c])
                 for word_c in candidate_words(word, vocab, alphabet)
                 for sc, node in layer]
        heapify(layer)
        layer = layer[:beam_width]
    return ' '.join(layer[0][1])


def candidate_words(word, lm_vocab, alphabet):
    """
    Generate possible spelling corrections for a given word.
    :param word: single word as a string
    :return: list of possible spelling corrections for each word
    """
    return known_words([word], lm_vocab) \
           or known_words(edits_1(word, alphabet), lm_vocab) \
           or known_words(edits_2(word, alphabet), lm_vocab) \
           or [word]


def known_words(word_list, lm_vocab):
    """
    Filters out from a list of words the subset of words that appear in the vocabulary of KNOWN_WORDS.
    :param word_list: list of word-strings
    :return: set of unique words that appear in vocabulary
    """
    return set(filter(lambda word: word in lm_vocab, word_list))


def edits_1(word_str, alphabet):
    """
    generates a list of all words with edit distance 1 for a given word
    Note that generators must be used for performance reasons.
    :param word_str: single word as a string
    :return: generator for all possible words with edit distance 1
    """
    return itertools.chain(deletes(word_str),
                           swaps(word_str),
                           replaces(word_str, alphabet),
                           inserts(word_str, alphabet))


def splits(word_str):
    """
    generates all possible splits of a word (e.g. 'abc' -> ('', 'abc'), ('a', 'bc'), ('ab', 'c') ('abc', '') )
    :param word_str: the word as string
    :return: generator for all possible splits for the word
    """
    return ((word_str[:i], word_str[i:]) for i in range(len(word_str) + 1))


def deletes(word_str):
    """
    generates all possible variants of a word with 1 character deleted (e.g. 'abc' -> 'ab', 'ac', 'bc')
    :param word_str: the word as string
    :return: generator for all variants of the word where 1 character is deleted
    """
    return (L + R[1:] for L, R in splits(word_str) if R)


def swaps(word_str):
    """
    generates all possible variants of a word where 2 adjacent characters are swapped (e.g. 'abc' -> 'bac', 'acb'
    :param word_str: the word as string
    :return: generator for all variants of the word where 2 adjacent characters are swapped
    """
    return (L + R[1] + R[0] + R[2:] for L, R in splits(word_str) if len(R) > 1)


def replaces(word_str, alphabet):
    """
    generates all possible variants of a word where 1 character is replaces
    (e.g. 'abc' -> 'bbc', 'cbc', ..., 'aac', 'acc', ..., 'aba', 'abb', ... )
    :param word_str: the word as string
    :param alphabet: the alphabet to use for the language
    :return: generator for all variants of the word where 1 character is replaced
    """
    return (L + c + R[1:] for L, R in splits(word_str) if R for c in alphabet)


def inserts(words_str, alphabet):
    """
    generates all possible variants of a word where 1 character is inserted. Identical elements may be present
    (e.g. 'abc' -> 'aabc', 'babc', 'cabc', ..., 'aabc', 'abbc', 'acbc', ..., 'abac', 'abbc', 'abcc', ..., 'abca', 'abcb', 'abcc', ...
    :param words_str: the word as string
    :param alphabet: the alphabet to use for the language
    :return: generator for all variants of the word where 1 character is inserted
    """
    return (L + c + R for L, R in splits(words_str) for c in alphabet)


def edits_2(word, alphabet):
    """
    generates a list of all words with edit distance 2 for a list of words
    :param word: list of word-strings
    :param alphabet: the alphabet to use for the language
    :return:
    """
    return (e2 for e1 in edits_1(word, alphabet) for e2 in edits_1(e1, alphabet))


# globals
LANGUAGE_MODELS = {}
LM_VOCABS = {}
