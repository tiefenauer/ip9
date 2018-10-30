from abc import ABC
from copy import deepcopy
from datetime import timedelta

from tabulate import tabulate
from tqdm import tqdm


class Corpus(ABC):
    """
    Base class for corpora
    """

    def __init__(self, corpus_id, name, entries):
        """
        Create a new corpus holding a list of corpus entries
        :param corpus_id: unique ID
        :param name: unique corpus name
        :param corpus_entries: list of CorpusEntry instances
        """
        self.corpus_id = corpus_id
        self._name = name
        for entry in entries:
            entry.corpus = self
        self.entries = entries
        self.root_path = None  # must be set when saving/loading
        self.creation_date = None  # must be set when saving/loading

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
        print('')
        print(f'Corpus: {self.name}')
        print(self.root_path)
        print(f'Creation date: {self.creation_date}')
        print()
        table = {}
        t_entries = n_speeches_total = length_speeches_total = duration_total = 0

        # stats per language
        for lang in self.languages:
            entries = [entry for entry in self.entries if entry.language == lang]
            n_entries = len(entries)
            t_entries += n_entries

            speech_segments = [segment for entry in entries for segment in entry]
            n_speeches = len(speech_segments)
            length_speeches = sum(segment.audio_length for segment in speech_segments)
            n_speeches_total += n_speeches
            length_speeches_total += length_speeches

            duration = sum(entry.audio_length for entry in self.entries if entry.language == lang)
            duration_total += duration

            table[lang] = (n_entries,
                           n_speeches, timedelta(seconds=int(length_speeches)),
                           timedelta(seconds=int(duration)))

        # total over all languages
        table['total'] = (t_entries,
                          n_speeches_total, timedelta(seconds=int(length_speeches_total)),
                          timedelta(seconds=int(duration_total))
                          )
        headers = ['lang', '#entries',
                   '#speech segments', 'speech segments length',
                   'audio length']
        print(tabulate([(k,) + v for k, v in table.items()], headers=headers))


class ReadyLinguaCorpus(Corpus, ABC):

    def __init__(self, entries):
        super().__init__('rl', 'ReadyLingua', entries)


class LibriSpeechCorpus(Corpus):

    def __init__(self, entries):
        super().__init__('ls', 'LibriSpeech', entries)


def filter_corpus_entry_by_subset_prefix(corpus_entries, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [corpus_entry for corpus_entry in corpus_entries
            if corpus_entry.subset
            and any(corpus_entry.subset.startswith(prefix) for prefix in prefixes)]
