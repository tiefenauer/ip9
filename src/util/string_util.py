import re
import string
import warnings

from unidecode import _warn_if_not_unicode, Cache

from util.ctc_util import get_alphabet

punctiation_extended = string.punctuation + """"„“‚‘"""
umlauts = re.compile('[äöü]]')
number = re.compile('[0-9]+')
digit = re.compile('[0-9]')
not_alphanumeric = re.compile('[^0-9a-zA-Z ]+')
not_alphanumeric_with_umlauts = re.compile('[^0-9a-zA-Zäöü ]+')
alphabet_with_umlauts = string.ascii_letters + 'äöüÄÖÜ'


def normalize(text, language):
    text = text.strip().lower()
    alphabet = get_alphabet(language)
    text = unidecode_with_alphabet(text, alphabet)
    text = replace_not_allowed(text, alphabet + ' 0123456789')
    return remove_multi_spaces(text)


def remove_multi_spaces(text):
    return ' '.join(text.split())


def create_filename(text):
    return replace_not_alphanumeric(text).replace(' ', '_').lower()


def remove_punctuation(text):
    return ''.join(c for c in text if c not in punctiation_extended)


def replace_not_allowed(text, allowed_chars, repl=' '):
    return ''.join(char if char in allowed_chars else repl for char in text)


def replace_not_alphanumeric(text, repl=' ', keep_umlauts=False):
    if keep_umlauts:
        return re.sub(not_alphanumeric_with_umlauts, repl, text)
    return re.sub(not_alphanumeric, repl, text)


def replace_numeric(text, repl='#', by_single_digit=False):
    return re.sub(number, repl, text) if by_single_digit else re.sub(digit, repl, text)


def contains_numeric(text):
    return any(char.isdigit() for char in text)


def unidecode_with_alphabet(text, alphabet):
    """
    Modified version from unidecode.unidecode_expect_ascii that does not replace characters of a given alphabet
    :param text: the text to unidecode
    :param alphabet: the alphabet as a string of the allowed characters
    :return: the unidecoded text
    """
    _warn_if_not_unicode(text)
    try:
        text_ascii = text.encode('ASCII')
    except UnicodeEncodeError:
        return _unidecode_excluding_allowed_chars(text, set(alphabet))
    return text_ascii.decode('utf-8')


def _unidecode_excluding_allowed_chars(text, allowed_chars):
    # modified version from unidecode._unidecode that does not replace allowed characters
    retval = []
    allowed_codepoints = [ord(char) for char in allowed_chars]

    for char in text:
        codepoint = ord(char)

        # Basic ASCII, ä/Ä, ö/Ö, ü/Ü
        if codepoint < 0x80 or codepoint in allowed_codepoints:
            retval.append(str(char))
            continue

        if codepoint > 0xeffff:
            continue  # Characters in Private Use Area and above are ignored

        if 0xd800 <= codepoint <= 0xdfff:
            warnings.warn("Surrogate character %r will be ignored. "
                          "You might be using a narrow Python build." % (char,),
                          RuntimeWarning, 2)

        section = codepoint >> 8  # Chop off the last two hex digits
        position = codepoint % 256  # Last two hex digits

        try:
            table = Cache[section]
        except KeyError:
            try:
                mod = __import__('unidecode.x%03x' % (section), globals(), locals(), ['data'])
            except ImportError:
                Cache[section] = None
                continue  # No match: ignore this character and carry on.

            Cache[section] = table = mod.data

        if table and len(table) > position:
            retval.append(table[position])

    return ''.join(retval)
