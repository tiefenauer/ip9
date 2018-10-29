"""
Utility functions to work with corpora
"""
import gzip
import pickle
from datetime import datetime
from os import listdir
from os.path import join, isdir, abspath, exists, isfile, pardir

from constants import LS_TARGET, RL_TARGET


def get_corpus(corpus_id_or_path, lang=None):
    corpus_root = get_corpus_root(corpus_id_or_path)
    corpus = load_corpus(corpus_root)
    corpus.root_path = corpus_root
    if lang:
        corpus = corpus(languages=[lang])
    return corpus


def get_corpus_root(corpus_id_or_path):
    if corpus_id_or_path in ['ls', 'rl']:
        return get_corpus_root_by_id(corpus_id_or_path)
    return get_corpus_root_by_path(corpus_id_or_path)


def get_corpus_root_by_id(corpus_id):
    env_mapping = {'rl': {'name': 'RL_TARGET', 'value': RL_TARGET}, 'ls': {'name': 'LS_TARGET', 'value': LS_TARGET}}
    if corpus_id not in env_mapping.keys():
        raise ValueError(f'unknown corpus id: {corpus_id}')
    env_var = env_mapping[corpus_id]
    env_var_name = env_var['name']
    corpus_root = env_var['value']
    if not corpus_root:
        raise ValueError(f'corpus with id {corpus_id} requested but environment variable {env_var_name} not set')
    if not isdir(corpus_root) and not isfile(corpus_root):
        raise ValueError(
            f'corpus with id {corpus_id} requested but entironment variable {env_var_name} points to an invalid location at {env_var_name}')
    return get_corpus_root_by_path(corpus_root)


def get_corpus_root_by_path(corpus_path):
    if isdir(corpus_path) and not exists(corpus_path):
        raise ValueError(f'corpus from directory path requested but directory {corpus_path} does not exist!')
    if isfile(corpus_path):
        if not exists(corpus_path):
            raise ValueError(f'corpus from file path requested but file {corpus_path} does not exist!')
        corpus_path = pardir(corpus_path)
    return abspath(corpus_path)


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
    return next(iter(join(dir, file) for file in listdir(dir) if file.lower().endswith(suffix.lower())), None)


def filter_corpus_entry_by_subset_prefix(corpus_entries, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [corpus_entry for corpus_entry in corpus_entries
            if corpus_entry.subset
            and any(corpus_entry.subset.startswith(prefix) for prefix in prefixes)]
