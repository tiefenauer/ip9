from abc import ABC
from copy import deepcopy
from datetime import timedelta
from itertools import product
from os.path import getmtime
from time import ctime

import pandas as pd
from tqdm import tqdm

from corpus.corpus_entry import CorpusEntry


class Corpus(ABC):
    """
    Base class for corpora
    """

    def __init__(self, corpus_id, name, df_path):
        """
        Create a new corpus holding a list of corpus entries
        :param corpus_id: unique ID
        :param name: unique corpus name
        :param corpus_entries: list of CorpusEntry instances
        """
        self.corpus_id = corpus_id
        self._name = name
        self.df_path = df_path
        self.df = pd.read_csv(df_path)
        self.creation_date = getmtime(df_path)
        self.entries = self.create_entries_from_dataframe(self.df)

    def create_entries_from_dataframe(self, df):
        entries = []
        for (entry_id, subset, lang, wav), df_segments in df.groupby(['entry_id', 'subset', 'language', 'audio_file']):
            entry = CorpusEntry(self, entry_id, subset, lang, wav, df_segments)
            entries.append(entry)
        return entries

    def segments(self, numeric=None):
        if numeric is True:
            return [seg for entry in self.entries for seg in entry.segments_numeric]
        elif numeric is False:
            return [seg for entry in self.entries for seg in entry.segments_not_numeric]
        return [seg for entry in self.entries for seg in entry]

    def __iter__(self):
        for corpus_entry in self.entries:
            yield corpus_entry

    def __getitem__(self, val):
        # access by index
        if isinstance(val, int) or isinstance(val, slice):
            return self.entries[val]
        # access by id
        if isinstance(val, str):
            return next(iter([entry for entry in self.entries if entry.id == val]), None)
        return None

    def __len__(self):
        return len(self.entries)

    def __call__(self, *args, **kwargs):
        languages = kwargs['languages'] if 'languages' in kwargs else self.languages
        include_numeric = kwargs['include_numeric'] if 'include_numeric' in kwargs else True
        print(f'filtering languages={languages}')
        entries = [entry for entry in self.entries if entry.language in languages]
        print(f'found {len(entries)} entries for languages {languages}')

        if not include_numeric:
            print(f'filtering out speech segments with numbers in transcript')
            entries = [entry(include_numeric=include_numeric) for entry in tqdm(entries, unit=' entries')]

        _copy = deepcopy(self)
        _copy.entries = entries
        return _copy

    @property
    def name(self):
        languages = ', '.join(self.languages)
        return self._name + f' (languages: {languages})'

    @property
    def languages(self):
        return sorted(set(lang for lang in (corpus_entry.language for corpus_entry in self.entries)))

    @property
    def keys(self):
        return sorted([corpus_entry.id for corpus_entry in self.entries])

    def train_set(self, numeric=None):
        return self._filter_segments('train', numeric)

    def dev_set(self, numeric=None):
        return self._filter_segments('dev', numeric)

    def test_set(self, numeric=None):
        return self._filter_segments('test', numeric)

    def _filter_segments(self, subset_prefix, numeric=None):
        df_subset = filter_df(self.df, subset_prefix, numeric)
        return self.create_entries_from_dataframe(df_subset)

    def summary(self):
        print(f"""
Corpus:        {self.name}
File:          {self.df_path}
Creation date: {ctime(self.creation_date)}
# entries:     {len(self.entries)}    
        """)

        def abs_perc_string(value, total, unit=None):
            percent = 100 * value / total if total > 0 else 0
            value = timedelta(seconds=value) if unit == 's' else value
            return f'{value} ({percent:.2f}%)'

        def create_rows(df_total):
            df_train = df_total[df_total['subset'] == 'train']
            df_dev = df_total[df_total['subset'] == 'dev']
            df_test = df_total[df_total['subset'] == 'test']

            n_all = abs_perc_string(len(df_total), len(df_total))
            n_train = abs_perc_string(len(df_train), len(df_total))
            n_dev = abs_perc_string(len(df_dev), len(df_total))
            n_test = abs_perc_string(len(df_test), len(df_total))

            s_all_sum = df_total['duration'].sum()
            s_train_sum = df_train['duration'].sum()
            s_dev_sum = df_dev['duration'].sum()
            s_test_sum = df_test['duration'].sum()

            audio_all = abs_perc_string(s_all_sum, s_all_sum, unit='s')
            audio_train = abs_perc_string(s_train_sum, s_all_sum, unit='s')
            audio_dev = abs_perc_string(s_dev_sum, s_all_sum, unit='s')
            audio_test = abs_perc_string(s_test_sum, s_all_sum, unit='s')

            s_tot_mean = df_total['duration'].mean() if df_total['duration'].any() else 0
            s_train_mean = df_train['duration'].mean() if df_train['duration'].any() else 0
            s_dev_mean = df_dev['duration'].mean() if df_dev['duration'].any() else 0
            s_test_mean = df_test['duration'].mean() if df_test['duration'].any() else 0

            audio_all_av = timedelta(seconds=s_tot_mean)
            audio_train_av = timedelta(seconds=s_train_mean)
            audio_dev_av = timedelta(seconds=s_dev_mean)
            audio_test_av = timedelta(seconds=s_test_mean)

            return [
                [n_train, n_dev, n_test, n_all],
                [audio_train, audio_dev, audio_test, audio_all],
                [audio_train_av, audio_dev_av, audio_test_av, audio_all_av]
            ]

        num_map = {True: 'numeric', False: 'non-numeric', None: 'all'}

        data = []
        for lang, num, in product(self.languages + [None], num_map.keys()):
            df_tot = self.df
            if lang:
                df_tot = df_tot[df_tot['language'] == lang]
            if num:
                df_tot = df_tot[df_tot['numeric'] == num]
            data += create_rows(df_tot)

        index = pd.MultiIndex.from_product([
            self.languages + ['all'],
            ['numeric', 'non-numeric', 'all'],
            ['samples', 'audio', 'Ø audio']
        ])
        columns = ['train', 'dev', 'test', 'total']
        df_stats = pd.DataFrame(data=data, index=index, columns=columns)
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', 1000,
                               'colheader_justify', 'center',
                               'display.max_colwidth', -1):
            print(df_stats)


class ReadyLinguaCorpus(Corpus, ABC):

    def __init__(self, df_path):
        super().__init__('rl', 'ReadyLingua', df_path)


class LibriSpeechCorpus(Corpus):

    def __init__(self, df):
        super().__init__('ls', 'LibriSpeech', df)


def filter_corpus_entry_by_subset_prefix(corpus_entries, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [corpus_entry for corpus_entry in corpus_entries
            if corpus_entry.subset
            and any(corpus_entry.subset.startswith(prefix) for prefix in prefixes)]


def filter_df(df, lang=None, subset=None, numeric=None):
    result = df
    result = result[result.apply(lambda s: subset in ['all', None] or s['subset'].startswith(subset), axis=1)]
    if not result.empty:
        result = result[result.apply(create_filter('language', lang), axis=1)]
    if not result.empty:
        result = result[result.apply(create_filter('numeric', numeric), axis=1)]
    return result


def create_filter(key, value):
    return lambda s: value in ['all', None] or s[key] is value
