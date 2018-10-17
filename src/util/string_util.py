import re
import string
import warnings
from sys import version_info

from unidecode import unidecode, _warn_if_not_unicode, Cache

punctiation_extended = string.punctuation + """"„“‚‘"""
umlauts = re.compile('[äöü]]')
number = re.compile('[0-9]+')
digit = re.compile('[0-9]')
not_alphanumeric = re.compile('[^0-9a-zA-Z ]+')


def normalize(text):
    return remove_multi_spaces(replace_not_alphanumeric(unidecode(text.strip().lower())))


def remove_multi_spaces(text):
    return ' '.join(text.split())


def create_filename(text):
    return replace_not_alphanumeric(text).replace(' ', '_').lower()


def remove_punctuation(text):
    return ''.join(c for c in text if c not in punctiation_extended)


def replace_not_alphanumeric(text, repl=' '):
    return re.sub(not_alphanumeric, repl, text)


def replace_numeric(text, repl='#', by_single_digit=False):
    return re.sub(number, repl, text) if by_single_digit else re.sub(digit, repl, text)


def contains_numeric(text):
    return any(char.isdigit() for char in text)


def unidecode_keep_umlauts(text):
    # modified version from unidecode.unidecode_expect_ascii that does not replace umlauts
    _warn_if_not_unicode(text)
    try:
        bytestring = text.encode('ASCII')
    except UnicodeEncodeError:
        return _unidecode_keep_umlauts(text)
    if version_info[0] >= 3:
        return text
    else:
        return bytestring


def _unidecode_keep_umlauts(text):
    # modified version from unidecode._unidecode that keeps umlauts
    retval = []

    for char in text:
        codepoint = ord(char)

        # Basic ASCII, ä/Ä, ö/Ö, ü/Ü
        if codepoint < 0x80 or codepoint in [0xe4, 0xc4, 0xf6, 0xd6, 0xfc, 0xdc]:
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
