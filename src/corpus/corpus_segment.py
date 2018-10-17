import librosa
import numpy as np
from python_speech_features import mfcc

from corpus.audible import Audible
from util.audio_util import ms_to_frames
from util.string_util import normalize, contains_numeric


class SpeechSegment(Audible):
    """
    Base class for audio segments
    """
    text = ''
    _transcript = ''

    # cache features
    _mag_specgram = None
    _pow_specgram = None
    _mel_specgram = None
    _mfcc = None

    def __init__(self, start_frame, end_frame, transcript, language):
        self.language = language
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.transcript = transcript.strip() if transcript else ''
        self.corpus_entry = None  # must be set by enclosing corpus entry

    @property
    def contains_numeric(self):
        return contains_numeric(self.text)

    @property
    def audio(self):
        return self.corpus_entry.audio[self.start_frame:self.end_frame]

    @audio.setter
    def audio(self, audio):
        self._mag_specgram = None
        self._pow_specgram = None
        self._mel_specgram = None
        self._mfcc = None

    @property
    def rate(self):
        return self.corpus_entry.rate

    @property
    def transcript(self):
        return self._transcript

    @transcript.setter
    def transcript(self, transcript):
        self._transcript = transcript
        keep_umlauts = self.language == 'de'
        self.text = normalize(transcript, keep_umlauts=keep_umlauts)

    @property
    def audio_length(self):
        sample_rate = int(float(self.corpus_entry.media_info['sample_rate']))
        return (self.end_frame - self.start_frame) / sample_rate

    def audio_features(self, feature_type):
        if feature_type == 'mfcc':
            return self.mfcc()
        elif feature_type == 'mel':
            return self.mel_specgram().T
        elif feature_type == 'pow':
            return self.pow_specgram().T
        elif feature_type == 'log':
            return np.log(self.mel_specgram().T + 1e-10)

        raise ValueError(f'Unknown feature type: {feature_type}')

    def mag_specgram(self, window_size=20, step_size=10, unit='ms'):
        if self._mag_specgram is not None:
            return self._mag_specgram

        if unit == 'ms':
            window_size = ms_to_frames(window_size, self.rate)
            step_size = ms_to_frames(step_size, self.rate)

        D = librosa.stft(self.audio, n_fft=window_size, hop_length=step_size)
        self._mag_specgram, phase = librosa.magphase(D)

        return self._mag_specgram

    def pow_specgram(self, window_size=20, step_size=10, unit='ms'):
        """
        Power-Spectrogram
        :param window_size: size of sliding window in frames or milliseconds
        :param step_size: step size for sliding window in frames or milliseconds
        :param unit: unit of window size ('ms' for milliseconds or None for frames)
        :return: (T_x, num_freqs) whereas num_freqs will be calculated from sample rate
        """
        if self._pow_specgram is not None:
            return self._pow_specgram

        self._pow_specgram = self.mag_specgram(window_size, step_size, unit) ** 2
        return self._pow_specgram

    def mel_specgram(self, num_mels=40, window_size=20, step_size=10, unit='ms'):
        """
        Mel-Spectrogram
        :param num_mels: number of mels to produce
        :param window_size: size of sliding window in frames or milliseconds
        :param step_size: step size for sliding window in frames or milliseconds
        :param unit: unit of window size ('ms' for milliseconds or None for frames)
        :return: (T_x, n_mels) matrix
        """
        if self._mel_specgram is not None:
            return self._mel_specgram

        if unit == 'ms':
            window_size = ms_to_frames(window_size, self.rate)
            step_size = ms_to_frames(step_size, self.rate)

        self._mel_specgram = librosa.feature.melspectrogram(y=self.audio, sr=self.rate,
                                                            n_fft=window_size, hop_length=step_size, n_mels=num_mels)
        return self._mel_specgram

    def mfcc(self, num_ceps=13):
        """
        MFCC coefficients
        :param num_ceps: number of coefficients to produce
        :return: (T_x, num_ceps) matrix
        """
        if self._mfcc is not None:
            return self._mfcc

        self._mfcc = mfcc(self.audio, self.rate, numcep=num_ceps)
        return self._mfcc
