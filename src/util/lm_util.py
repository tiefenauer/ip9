# this file is an adaptation from the work at mozilla deepspeech github.com/mozilla/DeepSpeech
import itertools
import kenlm
import re
from heapq import heapify
from os.path import abspath, exists

import numpy as np
from pattern3.metrics import levenshtein

from util.ctc_util import ALLOWED_CHARS

# the LER is just the Levenshtein/edit distance
ler = levenshtein


def ler_norm(ground_truth, prediction):
    """
    Calculates the normalized LER by dividing the LER by the length of the longer string. The result will be in [0,1]
    """
    return levenshtein(ground_truth, prediction) / float(max(len(ground_truth), len(prediction), 1.0))


def wer(ground_truth, prediction):
    """
    The WER is defined as the editing/Levenshtein distance on word level (not on character-level!).
    """
    ground_truth = ground_truth.split()
    prediction = prediction.split()
    return levenshtein(ground_truth, prediction) / float(len(ground_truth))


def wers(ground_truths, predictions):
    assert len(ground_truths) > 0, f'ERROR: no ground truths provided!'
    assert len(ground_truths) == len(predictions), f'ERROR: # of ground truths does not match # of predictions!'
    rates = [wer(ground_truth, prediction) for (ground_truth, prediction) in zip(ground_truths, predictions)]
    return rates, np.mean(rates)


def lers(ground_truths, predictions):
    assert len(ground_truths) > 0, f'ERROR: no ground truths provided!'
    assert len(ground_truths) == len(predictions), f'ERROR: # of ground truths does not match # of predictions!'
    lers_raw = [ler(ground_truth, prediction) for (ground_truth, prediction) in zip(ground_truths, predictions)]
    lers_norm = [ler_norm(ground_truth, prediction) for (ground_truth, prediction) in zip(ground_truths, predictions)]
    return lers_norm, np.mean(lers_norm), lers_raw, np.mean(lers_raw)


def load_lm(lm_path, vocab_path):
    global LM_MODELS
    if lm_path in LM_MODELS:
        print(f'using cached LM and vocab:')
        return LM_MODELS[lm_path]

    lm_abs_path = abspath(lm_path)
    lm_vocab_abs_path = abspath(vocab_path)
    if not exists(lm_abs_path):
        raise ValueError(f'ERROR: LM not found at {lm_abs_path}')
    if not exists(lm_vocab_abs_path):
        raise ValueError(f'ERROR: LM vocabulary not found at {lm_vocab_abs_path}')

    with open(lm_vocab_abs_path) as vocab_f:
        print(f'loading LM from {lm_abs_path}...', end='')
        lm = kenlm.Model(lm_abs_path)
        print(f'done! Loaded {lm.order}-gram model.')
        print(f'loading LM vocab from {lm_vocab_abs_path}...', end='')
        vocab = set(words(vocab_f.read()))
        print(f'done! Loaded {len(vocab)} words.')
        LM_MODELS[lm_path] = (lm, vocab)
    return lm, vocab


def words(text):
    """
    splits a text into a list of words
    :param text: a text-string
    :return: list of word-strings
    """
    return re.findall(r'\w+', text.lower())


def score(word_list, lm):
    """
    Use LM to calculate a log10-based probability for a given sentence (as a list of words)
    :param word_list:
    :return:
    """
    return lm.score(' '.join(word_list), bos=False, eos=False)


def correction(sentence, lm=None, lm_vocab=None):
    """
    Get most probable spelling correction for a given sentence.
    :param sentence:
    :return:
    """
    if not lm or not lm_vocab:
        return sentence
    beam_width = 1024
    layer = [(0, [])]  # list of (score, 2-gram)-pairs
    for word in words(sentence):
        layer = [(-score(node + [word_c], lm), node + [word_c]) for word_c in candidate_words(word, lm_vocab) for
                 sc, node in layer]
        heapify(layer)
        layer = layer[:beam_width]
    return ' '.join(layer[0][1])


def candidate_words(word, lm_vocab):
    """
    Generate possible spelling corrections for a given word.
    :param word: single word as a string
    :return: list of possible spelling corrections for each word
    """
    return known_words([word], lm_vocab) \
           or known_words(edits_1(word), lm_vocab) \
           or known_words(edits_2(word), lm_vocab) \
           or [word]  # fallback: the original word as a list


def known_words(word_list, lm_vocab):
    """
    Filters out from a list of words the subset of words that appear in the vocabulary of KNOWN_WORDS.
    :param word_list: list of word-strings
    :return: set of unique words that appear in vocabulary
    """
    return set(filter(lambda word: word in lm_vocab, word_list))


def edits_1(word_str):
    """
    generates a list of all words with edit distance 1 for a given word
    Note that generators must be used for performance reasons.
    :param word_str: single word as a string
    :return: generator for all possible words with edit distance 1
    """
    return itertools.chain(deletes(word_str), swaps(word_str), replaces(word_str), inserts(word_str))


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


def replaces(word_str):
    """
    generates all possible variants of a word where 1 character is replaces
    (e.g. 'abc' -> 'bbc', 'cbc', ..., 'aac', 'acc', ..., 'aba', 'abb', ... )
    :param word_str: the word as string
    :return: generator for all variants of the word where 1 character is replaced
    """
    return (L + c + R[1:] for L, R in splits(word_str) if R for c in ALLOWED_CHARS)


def inserts(words_str):
    """
    generates all possible variants of a word where 1 character is inserted. Identical elements may be present
    (e.g. 'abc' -> 'aabc', 'babc', 'cabc', ..., 'aabc', 'abbc', 'acbc', ..., 'abac', 'abbc', 'abcc', ..., 'abca', 'abcb', 'abcc', ...
    :param words_str: the word as string
    :return: generator for all variants of the word where 1 character is inserted
    """
    return (L + c + R for L, R in splits(words_str) for c in ALLOWED_CHARS)


def edits_2(word):
    """
    generates a list of all words with edit distance 2 for a list of words
    :param word: list of word-strings
    :return:
    """
    return (e2 for e1 in edits_1(word) for e2 in edits_1(e1))


# globals
LM_MODELS = {}
