import kenlm
import re
from heapq import heapify
from operator import itemgetter
from os.path import abspath, basename, splitext, dirname, join, exists

import nltk

from util.rnn_util import ALLOWED_CHARS
from util.string_util import replace_numeric, remove_punctuation, unidecode_keep_umlauts

LM_MODELS = {}


def load_LM(lm_path):
    global LM_MODELS
    lm_abs_path = abspath(lm_path)
    if not exists(lm_abs_path):
        raise ValueError(f'ERROR: LM not found at {lm_abs_path}')

    lm_dir = dirname(lm_abs_path)
    lm_name, _ = splitext(basename(lm_abs_path))
    lm_vocab_abs_path = abspath(join(lm_dir, lm_name + '.vocab'))
    if not exists(lm_vocab_abs_path):
        raise ValueError(f'ERROR: LM vocabulary not found at {lm_vocab_abs_path}')

    if lm_abs_path not in LM_MODELS:
        with open(lm_vocab_abs_path) as vocab_f:
            lm = kenlm.Model(lm_abs_path)
            vocab = words(vocab_f.read())
            LM_MODELS[lm_abs_path] = (lm, vocab)
    else:
        lm, vocab = LM_MODELS[lm_abs_path]
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


def correction(sentence, lm_path=None, lm=None, lm_vocab=None):
    """
    Get most probable spelling correction for a given sentence.
    :param sentence:
    :return:
    """
    if not lm_path and not lm:
        raise ValueError("ERROR: either lm_path or lm must be set!")

    if lm and not lm_vocab:
        raise ValueError("ERROR: if lm is set, then LM vocabulary must also be set!")

    lm, vocab = lm, lm_vocab if lm  and lm_vocab else load_LM(lm_path)
    beam_width = 1024
    layer = [(0, [])]  # list of (score, 2-gram)-pairs
    for word in words(sentence):
        layer = [(-score(node + [word_c], lm), node + [word_c]) for word_c in candidate_words(word, lm_vocab) for sc, node in
                 layer]
        heapify(layer)
        layer = layer[:beam_width]
    return ' '.join(layer[0][1])


def candidate_words(word, vocab):
    """
    Generate possible spelling corrections for a given word.
    :param word: single word as a string
    :return: list of possible spelling corrections for each word
    """
    return known_words([word], vocab) \
           or known_words(edits_1(word), vocab) \
           or known_words(edits_2(word), vocab) \
           or [word]  # fallback: the original word as a list


def known_words(word_list, vocab):
    """
    Filters out from a list of words the subset of words that appear in the vocabulary of KNOWN_WORDS.
    :param word_list: list of word-strings
    :return: set of unique words that appear in vocabulary
    """
    return set(w for w in word_list if w in vocab)


def edits_1(word_list):
    """
    generates a list of all words with edit distance 1 for a list of words
    :param word_list: list of word-strings
    :return:
    """
    splits = [(word_list[:i], word_list[i:]) for i in range(len(word_list) + 1)]  # all possible splits

    deletes = [L + R[1:] for L, R in splits if R]  # all words with one character removed
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]  # all words with two swapped characters
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALLOWED_CHARS]  # all words with one character replaced
    inserts = [L + c + R for L, R in splits for c in ALLOWED_CHARS]  # all words with one character inserted
    return set(deletes + transposes + replaces + inserts)


def edits_2(word):
    """
    generates a list of all words with edit distance 2 for a list of words
    :param word: list of word-strings
    :return:
    """
    return (e2 for e1 in edits_1(word) for e2 in edits_1(e1))


def process_line(line, min_words=4, language='german'):
    sentences = []
    sents = nltk.sent_tokenize(line.strip(), language=language)
    for sentence in sents:
        sentence_processed = process_sentence(sentence, min_words)
        if sentence_processed:
            sentences.append(sentence_processed)

    return sentences


def process_sentence(sent, min_words=4):
    words = [normalize_word(word) for word in nltk.word_tokenize(sent, language='german')]
    if len(words) >= min_words:
        return ' '.join(w for w in words if w).strip()  # prevent multiple spaces
    return ''


def normalize_word(token):
    _token = unidecode_keep_umlauts(token)
    _token = remove_punctuation(_token)  # remove any special chars
    _token = replace_numeric(_token, by_single_digit=True)
    _token = '<num>' if _token == '#' else _token  # if token was a number, replace it with <unk> token
    return _token.strip().lower()


def check_lm(lm_path, vocab_path, sentence):
    import kenlm
    model = kenlm.LanguageModel(lm_path)
    print(f'loaded {model.order}-gram model from {lm_path}')
    print(f'sentence: {sentence}')
    print(f'score: {model.score(sentence)}')

    words = ['<s>'] + sentence.split() + ['</s>']
    for i, (prob, length, oov) in enumerate(model.full_scores(sentence)):
        two_gram = ' '.join(words[i + 2 - length:i + 2])
        print(f'{prob} {length}: {two_gram}')
        if oov:
            print(f'\t\"{words[i+1]}" is an OOV!')

    vocab = set(word for line in open(vocab_path) for word in line.strip().split())
    print(f'loaded vocab with {len(vocab)} unique words')
    print()
    word = input('Your turn now! Start a sentence by writing a word: (enter nothing to abort)\n')
    sentence = ''
    state_in, state_out = kenlm.State(), kenlm.State()
    total_score = 0.0
    model.BeginSentenceWrite(state_in)

    while word:
        sentence += ' ' + word
        sentence = sentence.strip()
        print(f'sentence: {sentence}')
        total_score += model.BaseScore(state_in, word, state_out)

        candidates = list((model.score(sentence + ' ' + next_word), next_word) for next_word in vocab)
        bad_words = sorted(candidates, key=itemgetter(0), reverse=False)
        top_words = sorted(candidates, key=itemgetter(0), reverse=True)
        worst_5 = bad_words[:5]
        print()
        print(f'least probable 5 next words:')
        for w, s in worst_5:
            print(f'\t{w}\t\t{s}')

        best_5 = top_words[:5]
        print()
        print(f'most probable 5 next words:')
        for w, s in best_5:
            print(f'\t{w}\t\t{s}')

        if '.' in word:
            print(f'score for sentence \"{sentence}\":\t {total_score}"')  # same as model.score(sentence)!
            sentence = ''
            state_in, state_out = kenlm.State(), kenlm.State()
            model.BeginSentenceWrite(state_in)
            total_score = 0.0
            print(f'Start a new sentence!')
        else:
            state_in, state_out = state_out, state_in

        word = input('Enter next word: ')

    print(f'That\'s all folks. Thanks for watching.')
