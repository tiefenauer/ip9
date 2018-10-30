from abc import ABC
from datetime import timedelta
from os.path import getmtime, dirname

import pandas as pd
from tqdm import tqdm

from corpus.corpus_entry import CorpusEntry


class Corpus(ABC):
    """
    Base class for corpora
    """

    def __init__(self, corpus_id, name, language, df_path):
        """
        Create a new corpus holding a list of corpus entries
        :param corpus_id: unique ID
        :param df_path: path to CSV file
        """
        self.language = language
        self.corpus_id = corpus_id
        self.name = name
        self.df_path = df_path
        self.root_path = dirname(self.df_path)
        self.entries = self.create_entries(pd.read_csv(df_path))
        pass

    def create_entries(self, df):
        grouped_segments = df.groupby(['subset', 'audio_file'])
        entries = []
        for (subset, audio_file), segments in tqdm(grouped_segments, unit='segments'):
            entries.append(CorpusEntry(self, subset, audio_file, segments))
        return entries

    def segments(self, numeric=None):
        if numeric is True:
            return [seg for entry in self.entries for seg in entry.segments_numeric]
        elif numeric is False:
            return [seg for entry in self.entries for seg in entry.segments_not_numeric]
        return [seg for entry in self.entries for seg in entry]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, val):
        # access by index
        if isinstance(val, int) or isinstance(val, slice):
            return self.entries[val]
        # access by id
        if isinstance(val, str):
            return next(iter([entry for entry in self.entries if entry.id == val]), None)
        return None

    @property
    def keys(self):
        return sorted([corpus_entry.id for corpus_entry in self.entries])

    def train_dev_test_split(self, numeric=False):
        """
        return training -, validation - and test - set
        """
        train_entries = filter_corpus_entry_by_subset_prefix(self.entries, 'train')
        dev_entries = filter_corpus_entry_by_subset_prefix(self.entries, 'dev')
        test_entries = filter_corpus_entry_by_subset_prefix(self.entries, ['test', 'unknown'])

        if numeric is None:
            train_set = [seg for corpus_entry in train_entries for seg in corpus_entry.segments]
            dev_set = [seg for corpus_entry in dev_entries for seg in corpus_entry.segments]
            test_set = [seg for corpus_entry in test_entries for seg in corpus_entry.segments]
        elif numeric:
            train_set = [seg for corpus_entry in train_entries for seg in corpus_entry.segments_numeric]
            dev_set = [seg for corpus_entry in dev_entries for seg in corpus_entry.segments_numeric]
            test_set = [seg for corpus_entry in test_entries for seg in corpus_entry.segments_numeric]
        else:
            train_set = [seg for corpus_entry in train_entries for seg in corpus_entry.segments_not_numeric]
            dev_set = [seg for corpus_entry in dev_entries for seg in corpus_entry.segments_not_numeric]
            test_set = [seg for corpus_entry in test_entries for seg in corpus_entry.segments_not_numeric]

        return train_set, dev_set, test_set

    def summary(self):
        creation_date = getmtime(self.df_path)
        print(f"""
        Corpus: {self.name} (id={self.corpus_id})
        File: {self.df_path}
        Creation date: {creation_date}
        # entries: {len(self.entries)}
        """)
        print('-------------------------------------')

        index = pd.MultiIndex.from_product([
            ['numeric', 'non-numeric', 'total'],
            ['#', 'audio length', 'avg. audio length']
        ])
        columns = ['train', 'dev', 'test', 'total']
        df = pd.DataFrame(index=index, columns=columns)

        def add_subsets(ix, train, dev, test):
            total = train + dev + test
            n_train, n_dev, n_test = len(train), len(dev), len(test)
            n_total = len(total)
            len_trn = sum(s.audio_length for s in train)
            len_dev = sum(s.audio_length for s in dev)
            len_tst = sum(s.audio_length for s in test)
            len_tot = sum(s.audio_length for s in total)
            len_trn_avg = len_trn / n_total if n_total > 0 else 0
            len_dev_avg = len_dev / n_total if n_total > 0 else 0
            len_tst_avg = len_tst / n_total if n_total > 0 else 0
            len_tot_avg = len_tot / n_total if n_total > 0 else 0
            add_row((ix, '#'), n_train, n_dev, n_test, n_total)
            add_row((ix, 'audio length'), len_trn, len_dev, len_tst, len_tot, unit='s')
            add_row((ix, 'avg. audio length'), len_trn_avg, len_dev_avg, len_tst_avg, len_tot_avg, unit='s')

        def add_row(index, train, dev, test, total, unit=None):
            train_perc = 100 * train / total if total > 0 else 0
            dev_perc = 100 * dev / total if total > 0 else 0
            test_perc = 100 * test / total if total > 0 else 0
            if unit == 's':
                train = timedelta(seconds=train)
                dev = timedelta(seconds=dev)
                test = timedelta(seconds=test)
                total = timedelta(seconds=total)
            df.loc[index, :] = [
                f'{train} ({train_perc:.2f}%)',
                f'{dev} ({dev_perc:.2f}%)',
                f'{test} ({test_perc:.2f}%)',
                f'{total}',
            ]

        train_set, dev_set, test_set = self.train_dev_test_split(numeric=True)
        add_subsets('numeric', train_set, dev_set, test_set)

        train_set, dev_set, test_set = self.train_dev_test_split(numeric=False)
        add_subsets('non-numeric', train_set, dev_set, test_set)

        train_set, dev_set, test_set = self.train_dev_test_split(numeric=None)
        add_subsets('total', train_set, dev_set, test_set)

        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', 1000,
                               'colheader_justify', 'center',
                               'display.max_colwidth', -1):
            print(df)


class ReadyLinguaCorpus(Corpus, ABC):

    def __init__(self, language, df_path):
        super().__init__('rl', 'ReadyLingua', language, df_path)


class LibriSpeechCorpus(Corpus):

    def __init__(self, df_path):
        super().__init__('ls', 'LibriSpeech', 'en', df_path)


def filter_corpus_entry_by_subset_prefix(corpus_entries, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [corpus_entry for corpus_entry in corpus_entries
            if corpus_entry.subset
            and any(corpus_entry.subset.startswith(prefix) for prefix in prefixes)]
