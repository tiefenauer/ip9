from abc import ABC, abstractmethod


class Audible(ABC):
    """
    Base class for corpus objects that contain an audio signal
    """

    @property
    @abstractmethod
    def audio(self):
        return None

    @property
    @abstractmethod
    def rate(self):
        return None
