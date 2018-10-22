from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta

from tabulate import tabulate
from tqdm import tqdm

from util.corpus_util import filter_corpus_entry_by_subset_prefix


class Corpus(ABC):
    """
    Base class for corpora
    """

    def __init__(self, corpus_entries):
        """
        Create a new corpus holding a list of corpus entries
        :param name: unique corpus name
        :param corpus_entries: list of CorpusEntry instances
        :param root_path: path to directory holding the audio files for the corpus entries
        """
        for corpus_entry in corpus_entries:
            corpus_entry.corpus = self
        self.corpus_entries = corpus_entries
        self.root_path = None  # must be set when saving/loading
        self.creation_date = None  # must be set when saving/loading

    @property
    @abstractmethod
    def _name(self):
        raise NotImplementedError

    def __iter__(self):
        for corpus_entry in self.corpus_entries:
            yield corpus_entry

    def __getitem__(self, val):
        # access by index
        if isinstance(val, int) or isinstance(val, slice):
            return self.corpus_entries[val]
        # access by id
        if isinstance(val, str):
            return next(iter([corpus_entry for corpus_entry in self.corpus_entries if corpus_entry.id == val]), None)
        return None

    def __len__(self):
        return len(self.corpus_entries)

    def __call__(self, *args, **kwargs):
        languages = kwargs['languages'] if 'languages' in kwargs else self.languages
        include_numeric = kwargs['include_numeric'] if 'include_numeric' in kwargs else True
        print(f'filtering languages={languages}')
        entries = [entry for entry in self.corpus_entries if entry.language in languages]
        print(f'found {len(entries)} entries for languages {languages}')

        if not include_numeric:
            print(f'filtering out speech segments with numbers in transcript')
            entries = [entry(include_numeric=include_numeric) for entry in tqdm(entries, unit=' entries')]

        _copy = deepcopy(self)
        _copy.corpus_entries = entries
        return _copy

    @property
    def name(self):
        languages = ', '.join(self.languages)
        return self._name + f' (languages: {languages})'

    @property
    def languages(self):
        return sorted(set(lang for lang in (corpus_entry.language for corpus_entry in self.corpus_entries)))

    @property
    def keys(self):
        return sorted([corpus_entry.id for corpus_entry in self.corpus_entries])

    @abstractmethod
    def train_dev_test_split(self, include_numeric=False):
        """return training-, validation- and test-set
        Since these sets are constructed
        """
        pass

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
            entries = [entry for entry in self.corpus_entries if entry.language == lang]
            n_entries = len(entries)
            t_entries += n_entries

            speech_segments = [segment for entry in entries for segment in entry]
            n_speeches = len(speech_segments)
            length_speeches = sum(segment.audio_length for segment in speech_segments)
            n_speeches_total += n_speeches
            length_speeches_total += length_speeches

            duration = sum(entry.audio_length for entry in self.corpus_entries if entry.language == lang)
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

    @property
    def _name(self):
        return 'ReadyLingua'

    def train_dev_test_split(self, include_numeric=False):
        if include_numeric:
            speech_segments = [seg for corpus_entry in self.corpus_entries for seg in corpus_entry.speech_segments]
        else:
            speech_segments = [seg for corpus_entry in self.corpus_entries
                               for seg in corpus_entry.speech_segments_not_numeric]

        total_length = sum(segment.audio_length for segment in speech_segments)
        train_split = self.get_index_for_audio_length(speech_segments, total_length * 0.8)
        test_split = self.get_index_for_audio_length(speech_segments, total_length * 0.9)

        return speech_segments[:train_split], speech_segments[train_split:test_split], speech_segments[test_split:]

    @staticmethod
    def get_index_for_audio_length(segments, min_length):
        """
        get index to split speech segments at a minimum audio length. Index will not split segments of same corpus entry
        :param segments: list of speech segments
        :param min_length: minimum audio length to split
        :return: firs index where total length of speech segments is equal or greater to minimum legnth
        """
        audio_length = 0
        prev_corpus_entry_id = None
        for ix, segment in enumerate(segments):
            audio_length += segment.audio_length
            if audio_length > min_length and segment.corpus_entry.id is not prev_corpus_entry_id:
                return ix
            prev_corpus_entry_id = segment.corpus_entry.id


class LibriSpeechCorpus(Corpus):

    @property
    def _name(self):
        return 'LibriSpeech'

    def train_dev_test_split(self, include_numeric=False):
        train_entries = filter_corpus_entry_by_subset_prefix(self.corpus_entries, 'train-')
        dev_entries = filter_corpus_entry_by_subset_prefix(self.corpus_entries, 'dev-')
        test_entries = filter_corpus_entry_by_subset_prefix(self.corpus_entries, ['test-', 'unknown'])

        if include_numeric:
            train_set = [seg for corpus_entry in train_entries for seg in corpus_entry.speech_segments]
            dev_set = [seg for corpus_entry in dev_entries for seg in corpus_entry.speech_segments]
            test_set = [seg for corpus_entry in test_entries for seg in corpus_entry.speech_segments]
        else:
            train_set = [seg for corpus_entry in train_entries for seg in corpus_entry.speech_segments_not_numeric]
            dev_set = [seg for corpus_entry in dev_entries for seg in corpus_entry.speech_segments_not_numeric]
            test_set = [seg for corpus_entry in test_entries for seg in corpus_entry.speech_segments_not_numeric]

        return train_set, dev_set, test_set
