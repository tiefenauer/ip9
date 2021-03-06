class Voice(object):
    """
    Class representing voiced segment inside an audio signal with a transcript.
    The transcript does not need to be 100% correct (i.e. not necessarily a substring of the real transcript)
    and can be generated by an ASR system
    """

    def __init__(self, audio, rate, start_frame, end_frame):
        """
        :param audio: audio signal (numpy array)
        :param rate: sampling rate
        :param start_frame: start frame of voiced segment
        :param end_frame: end frame of voiced segment
        """
        self.audio = audio
        self.rate = rate
        self.start_frame = start_frame
        self.end_frame = end_frame


class Alignment(Voice):
    """
    Class representing a voiced segment whose transcript has been aligned with some other text. In contrast to a Voice
    object, the aligned text must be a substring of the real transcript.
    """

    def __init__(self, voice, text_start, text_end):
        """

        :param voice: voiced segment that is aligned
        :param text_start: start index of aligned text inside full transcript
        :param text_end: end index of aligned text inside full transcript
        """
        super().__init__(voice.audio, voice.rate, voice.start_frame, voice.end_frame)
        self.text_start = text_start
        self.tex_end = text_end
