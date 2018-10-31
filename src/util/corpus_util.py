"""
Utility functions to work with corpora
"""
from os import listdir
from os.path import join, isdir, abspath, exists, isfile, pardir

from constants import LS_TARGET, RL_TARGET
from corpus.corpus import ReadyLinguaCorpus, LibriSpeechCorpus


def get_corpus(corpus_id_or_file, language=None):
    corpus_root = get_corpus_root(corpus_id_or_file)
    df_path = join(corpus_root, 'index.csv')
    if corpus_id_or_file == 'rl' or 'readylingua' in corpus_id_or_file:
        corpus = ReadyLinguaCorpus(df_path)
    elif corpus_id_or_file == 'ls' or 'librispeech' in corpus_id_or_file:
        corpus = LibriSpeechCorpus(df_path)
    else:
        raise ValueError(f'ERROR: could not determine corpus id from {corpus_id_or_file}')
    if language:
        return corpus(languages=[language])
    return corpus


def get_corpus_root(corpus_id_or_path):
    if corpus_id_or_path in ['ls', 'rl']:
        return get_corpus_file_by_id(corpus_id_or_path)
    return get_corpus_path(corpus_id_or_path)


def get_corpus_file_by_id(corpus_id):
    id_map = {'rl': {'name': 'RL_TARGET', 'value': RL_TARGET}, 'ls': {'name': 'LS_TARGET', 'value': LS_TARGET}}
    if corpus_id not in id_map.keys():
        raise ValueError(f'unknown corpus id: {corpus_id}')
    var_name = id_map[corpus_id]['name']
    var_value = id_map[corpus_id]['value']
    if not var_value:
        raise ValueError(f'corpus with id {corpus_id} requested but environment variable {var_name} not set')
    if not isdir(var_value) and not isfile(var_value):
        raise ValueError(
            f'corpus with id {corpus_id} requested but variable {var_name} points to an invalid location at {var_name}')
    return get_corpus_path(var_value)


def get_corpus_path(corpus_path):
    if isdir(corpus_path) and not exists(corpus_path):
        raise ValueError(f'corpus from directory path requested but directory {corpus_path} does not exist!')
    if isfile(corpus_path):
        if not exists(corpus_path):
            raise ValueError(f'corpus from file path requested but file {corpus_path} does not exist!')
        corpus_path = pardir(corpus_path)
    return abspath(corpus_path)


def find_file_by_suffix(dir, suffix):
    """
    Find first file inside a directory ending with a given suffix or None if no file is found
    :param dir: path to directory to walk
    :param suffix: suffix to use
    :return: name of first found file or None if directory does not contain any file with given suffix
    """
    return next(iter(join(dir, file) for file in listdir(dir) if file.lower().endswith(suffix.lower())), None)
