# trains a n-Gram LM
# http://victor.chahuneau.fr/notes/2012/07/03/kenlm.html
import string
import sys
from operator import itemgetter

import nltk

LANGUAGES = {'de': 'german', 'en': 'english'}

if __name__ == '__main__':
    lang = LANGUAGES[sys.argv[1]]
    vocab = set()
    for line in sys.stdin:
        words = [w for w in nltk.word_tokenize(line, language=lang) if not any(p in w for p in string.punctuation)]
        print(' '.join(words).lower())
        vocab.update(words)


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
