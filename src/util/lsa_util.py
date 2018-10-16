"""
Utility functions for LSA stage
"""
import itertools

import numpy as np
from pattern3.metrics import levenshtein_similarity
from tqdm import tqdm

from corpus.alignment import Alignment
from util.string_util import normalize


def align(voice_segments, transcript, printout=False):
    a = transcript.upper()  # transcript[end:]  # transcript[:len(transcript)//2]
    alignments, lines = [], []

    progress = tqdm([voice for voice in voice_segments if len(voice.transcript.strip()) > 0], unit='voice activities')
    for voice in progress:
        b = voice.transcript.upper()
        text_start, text_end, b_ = smith_waterman(a, b)
        alignment_text = transcript[text_start:text_end]
        edit_distance = levenshtein_similarity(normalize(b), normalize(alignment_text))

        line = f'edit distance: {edit_distance:.2f}, transcript: {voice.transcript}, alignment: {alignment_text}'
        progress.set_description(line)
        if printout and type(printout) is bool:
            print(line)
        lines.append(line)

        if edit_distance > 0.5:
            alignments.append(Alignment(voice, text_start, text_end))
            # prevent identical transcripts to become aligned with the same part of the audio
            # a = transcript.replace(alignment_text, '-' * len(alignment_text), 1)

    if printout and type(printout) is str:
        with open(printout, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(lines))
    return alignments


def align_globally(transcript, partial_transcripts):
    inference = '#' + '#'.join(partial_transcripts)
    return needle_wunsch(transcript, inference)


def smith_waterman(a, b, match_score=3, gap_cost=2):
    """
    Find some variant b' of b in a
    Implementation of Smith-Waterman algorithm for Local Sequence Alignment (LSA) according to Wikipedia.
    (http://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm)

    :param a:
    :param b:
    :param match_score:
    :param gap_cost:
    :return: (text_start, text_end, b_)
        - text_start: start position of aligned sequence
        - text_end: end position of aligned sequence
        - b_: variant b' of string b including deletions and insertions marked with dashes
    """
    a, b = a.upper(), b.upper()
    H = matrix(a, b, match_score, gap_cost)
    b_, pos = traceback(H, b)
    return pos, pos + len(b_), b_


def matrix(a, b, match_score=3, gap_cost=2):
    """
    Create scoring matrix H to find b in a by applying the Smith-Waterman algorithm
    :param a: string a (reference string)
    :param b: string b to be aligned with a
    :param match_score: (optional) score to use if a[i] == b[j]
    :param gap_cost: (optional) cost to use for inserts in a or deletions in b
    :return: the scoring matrix H
    """
    H = np.zeros((len(a) + 1, len(b) + 1), np.int)

    for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
        match = H[i - 1, j - 1] + (match_score if a[i - 1] == b[j - 1] else - match_score)
        delete = H[i - 1, j] - gap_cost
        insert = H[i, j - 1] - gap_cost
        H[i, j] = max(match, delete, insert, 0)
    return H


def traceback(H, b, b_='', old_i=0):
    """
    Finds the start position of string b in string a by recursively backtracing a scoring matrix H.
    Note: string a is not needed for this because only the position is searched.
    :param H: scoring matrix (numpy 2D-array)
    :param b: string b to find in string a
    :param b_: (optional) string b_ from previous recursion
    :param old_i: (optional) index of row containing the last maximum value from previous iteration
    :return: (b_, pos)
        b_: string b after applying gaps and deletions
        pos: local alignment (starting position of b in a)
    """
    # flip H to get index of last occurrence of H.max()
    H_flip = np.flip(np.flip(H, 0), 1)
    i_, j_ = np.unravel_index(H_flip.argmax(), H_flip.shape)
    i, j = np.subtract(H.shape, (i_ + 1, j_ + 1))  # (i, j) are **last** indexes of H.max()
    # print(H, i,j)
    if H[i, j] == 0:
        return b_, i

    # represent characters that are skipped in string A (i.e. insertions in B) with a dash
    rows_skipped = max(old_i - i - 1, 0)
    infix = '-' * rows_skipped

    b_ = b[j - 1] + infix + b_
    return traceback(H[0:i, 0:j], b, b_, i)


def needle_wunsch(str_1, str_2, match_score=10, mismatch_score=-5, gap_score=-5):
    """
    Needle-Wunsch algorithm for global sequence alignemnt. Performs a global alignment to match a string with a
    reference string.
    Code (with changes) from https://github.com/alevchuk/pairwise-alignment-in-python/blob/master/alignment.py
    :param str_1: the reference string
    :param str_2: the string to align with the reference string
    :param match_score:
    :param mismatch_score:
    :param gap_score:
    :return:
    """

    def match_score(a, b):
        if a == b:
            return 10
        if a == '-' or b == '-':
            return -5
        return -5

    # reference string on axis 0, other string on axis 1
    m, n = len(str_1) + 1, len(str_2) + 1

    # Generate DP table and traceback path pointer matrix
    scores = np.zeros((m, n))
    scores[:, 0] = np.arange(m) * gap_score
    scores[0, :] = np.arange(n) * gap_score

    for i, j in itertools.product(range(1, m), range(1, n)):
        match = scores[i - 1][j - 1] + match_score(str_1[i - 1], str_2[j - 1])
        delete = scores[i - 1][j] + gap_score
        insert = scores[i][j - 1] + gap_score
        scores[i][j] = max(match, delete, insert)

    alignments = []
    source_str, target_str = '', ''

    # Traceback: start from the bottom right cell
    i, j = m - 1, n - 1
    end = i
    while i > 0 and j > 0:
        if str_2[j - 1] == '#':  # beginnings of speech segments are marked with '#'
            start = snap_to_closest_word_boundary(i, str_1)
            end = snap_to_closest_word_boundary(end, str_1)
            alignments.insert(0, {'start': start - 1, 'end': end - 2, 'text': str_1[start - 1:end - 2]})
            end = start

        score_current = scores[i][j]
        score_diagonal = scores[i - 1][j - 1]
        score_up = scores[i][j - 1]
        score_left = scores[i - 1][j]

        if score_current == score_diagonal + match_score(str_1[i - 1], str_2[j - 1]):
            source_str = str_1[i - 1] + source_str
            target_str = str_2[j - 1] + target_str
            i -= 1
            j -= 1
        elif score_current == score_left + gap_score:
            source_str = str_1[i - 1] + source_str
            target_str = '-' + target_str
            i -= 1
        elif score_current == score_up + gap_score:
            source_str = '-' + source_str
            target_str = str_2[j - 1] + target_str
            j -= 1

    # add one additional alignment (at max!) if segment beginnings that are not yet processed are encountered
    # The first unprocessed segment is aligned with the remaining text at the beginning of the reference string
    if j > 0:
        alignments.insert(0, {'start': 0, 'end': end, 'text': str_1[0:end]})
    # Finish tracing up to the top left cell
    while j > 0:
        source_str = '-' + source_str
        target_str = str_2[j - 1] + target_str
        j -= 1
    while i > 0:
        source_str = str_1[i - 1] + source_str
        target_str = '-' + target_str
        i -= 1

    alignment_str = ''.join(a if a == b else ' ' for a, b in zip(source_str, target_str))
    score = sum(match_score(a, b) for a, b in zip(source_str, target_str))
    match = float(100) * sum(1 if a == b else 0 for a, b in zip(source_str, target_str)) / len(target_str)

    # print(f'Match: {match:3.3f}%')
    # print(f'Score: {score}', )
    # print(f'Source   : {source_str}')
    # print(f'Alignment: {alignment_str}')
    # print(f'Target   : {target_str}')
    return alignments


def snap_to_closest_word_boundary(ix, text):
    left, right = 0, 0
    while ix - left - 1 > 0 and text[ix - left - 1] != ' ':
        left += 1
    while ix + right + 1 < len(text) and text[ix + right + 1] != ' ':
        right += 1
    return ix - left + 1 if left <= right else ix + right - 1
