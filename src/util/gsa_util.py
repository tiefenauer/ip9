"""
Utility functions for LSA stage
"""
import itertools

import numpy as np


def needle_wunsch(str_1, str_2, boundaries, match_score=10, mismatch_score=-5, gap_score=-5, align_endings=True):
    """
    Needle-Wunsch algorithm for global sequence alignemnt. Performs a global alignment to match a string with a
    reference string.
    Code (with changes) from https://github.com/alevchuk/pairwise-alignment-in-python/blob/master/alignment.py
    :param str_1: the reference string (the original transcript)
    :param str_2: the string to align with the reference string (the space-separated concatenated partial transcripts
    :param boundaries: array of int-tuples [(ix_start, ix_end)] containing the start-/end-indices of the concatenated partial transcripts
    :param match_score: score to assign in the alignment-matrix if two characters match
    :param mismatch_score: score to assign in the alignment-matrix if two characters do not match
    :param gap_score: score to assign in the alignment-matrix if one of the characters marks a gap
    :param align_endings: by default the algorithm will consider the end boundaries given in the boundaries array. This
                          makes it possible that there are unaligned parts between transcripts because each alignment
                          relates to the underlying partial transcript. If this flag is set to False, each alignment
                          ends one character before the next one starts. This prevents unaligned parts between
                          alignments, but the alignments might contain parts that do actually not have any relation to
                          the corresponding transcript.
    :return:
    """

    def calculate_score(a, b):
        if a == b:
            return match_score
        if a == '-' or b == '-':
            return gap_score
        return mismatch_score

    # reference string on axis 0, other string on axis 1
    m, n = len(str_1) + 1, len(str_2) + 1

    # Generate DP table and traceback path pointer matrix
    scores = np.zeros((m, n))
    scores[:, 0] = np.arange(m) * gap_score
    scores[0, :] = np.arange(n) * gap_score

    for i, j in itertools.product(range(1, m), range(1, n)):
        match = scores[i - 1][j - 1] + calculate_score(str_1[i - 1], str_2[j - 1])
        delete = scores[i - 1][j] + gap_score
        insert = scores[i][j - 1] + gap_score
        scores[i][j] = max(match, delete, insert)

    alignments = []
    source_str, target_str = '', ''

    # Traceback: start from the bottom right cell
    i, j = m - 1, n - 1  # i/j point to the row/column in the alignment matrix
    i_start, i_end = None if align_endings else len(str_1), i  # markers for the alignment in str_1
    j_start, j_end = boundaries[-1]  # markers for the aligned text in str_2
    while i > 0 and j > 0:
        score_current = scores[i][j]
        score_diagonal = scores[i - 1][j - 1]
        score_up = scores[i][j - 1]
        score_left = scores[i - 1][j]

        if score_current == score_diagonal + calculate_score(str_1[i - 1], str_2[j - 1]):
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

        if align_endings and j == j_end:
            i_end = i
        if j == j_start:
            if not align_endings:
                i_end = i_start - 1
            i_start = i

        if i_start is not None and i_end is not None:
            i_start = snap_left(i_start, str_1)
            i_end = snap_right(i_end, str_1)
            alignments.insert(0, {'start': i_start, 'end': i_end, 'text': str_1[i_start:i_end]})

            i_end = None
            i_start = None if align_endings else i_start
            boundaries.remove((j_start, j_end))
            j_start, j_end = boundaries[-1] if boundaries else (None, None)

    while j > 0:
        source_str = '-' + source_str
        target_str = str_2[j - 1] + target_str
        j -= 1
    while i > 0:
        source_str = str_1[i - 1] + source_str
        target_str = '-' + target_str
        i -= 1

    return alignments, source_str, target_str


def snap_left(ix, text):
    ix = snap_to_closest_word_boundary(ix, text)
    while ix < len(text) and text[ix] in [' ', '\n']:
        ix += 1
    return ix


def snap_right(ix, text):
    ix = snap_to_closest_word_boundary(ix, text)
    while ix > 0 and text[ix - 1] in [' ', '\n']:
        ix -= 1
    return ix


def snap_to_closest_word_boundary(ix, text):
    left, right = 0, 0
    while ix - left > 0 and text[ix - left - 1] not in [' ', '\n']:
        left += 1
    while ix + right < len(text) and text[ix + right] not in [' ', '\n']:
        right += 1

    return max(ix - left, 0) if left <= right else min(ix + right, len(text))
