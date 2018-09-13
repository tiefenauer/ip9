"""
Utility functions to work with corpora
"""
import gzip
import pickle
from datetime import datetime
from os import listdir
from os.path import join

from constants import LS_TARGET, RL_TARGET


def get_corpus(corpus_id, lang=None):
    corpus_root = get_corpus_root(corpus_id)
    corpus = load_corpus(corpus_root)
    corpus.root_path = corpus_root
    if lang:
        corpus = corpus(languages=[lang])
    return corpus


def get_corpus_root(corpus_id):
    if corpus_id == 'ls':
        return LS_TARGET
    if corpus_id == 'rl':
        return RL_TARGET
    raise ValueError(f'unknown corpus id: {corpus_id}')


def load_corpus(corpus_root):
    corpus_file = join(corpus_root, 'index')
    print(f'loading {corpus_file} ...')
    if corpus_file.endswith('.gz'):
        with gzip.open(corpus_file, 'rb') as corpus_f:
            corpus = pickle.loads(corpus_f.read())
    else:
        with open(corpus_file, 'rb') as corpus_f:
            corpus = pickle.load(corpus_f)
    print(f'...done! Loaded {corpus.name}: {len(corpus)} corpus entries')
    return corpus


def save_corpus(corpus, target_root, gzip=False):
    corpus.root_path = target_root
    corpus.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    corpus_file = join(target_root, 'index')
    if gzip:
        corpus_file += '.gz'
        with gzip.open(corpus_file, 'wb') as f:
            f.write(pickle.dumps(corpus))
    else:
        with open(corpus_file, 'wb') as f:
            pickle.dump(corpus, f)
    return corpus_file


def find_file_by_suffix(dir, suffix):
    """
    Find first file inside a directory ending with a given suffix or None if no file is found
    :param dir: path to directory to walk
    :param suffix: suffix to use
    :return: name of first found file or None if directory does not contain any file with given suffix
    """
    return next(iter(fn for fn in listdir(dir) if fn.lower().endswith(suffix.lower())), None)


def filter_corpus_entry_by_subset_prefix(corpus_entries, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [corpus_entry for corpus_entry in corpus_entries
            if corpus_entry.subset
            and any(corpus_entry.subset.startswith(prefix) for prefix in prefixes)]
