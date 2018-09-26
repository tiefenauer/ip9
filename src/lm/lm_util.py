import nltk

from util.string_util import replace_numeric, remove_punctuation, unidecode_keep_umlauts


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
    _token = '<unk>' if _token == '#' else _token  # if token was a number, replace it with <unk> token
    return _token.strip().lower()
