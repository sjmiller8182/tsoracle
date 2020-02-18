"""API Abstracts
"""

from abc import ABC, abstractmethod, abstractproperty

class Generator(ABC):
    """Abstract class for signal generators
    """

    @abstractmethod
    def gen(self, size: int):
        """Abstract method for generating a signal
        """
        # overwrite this method
        raise NotImplementedError
