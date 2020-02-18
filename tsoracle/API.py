"""API Abstracts
"""

from abc import ABC, abstractmethod, abstractproperty

class Generator(ABC):
    """Abstract class for signal generators
    """

    @abstractproperty
    def coef(self):
        """Return coefficients of the generator
        """
        # overwrite this method
        raise NotImplementedError

    @abstractmethod
    def gen(self, size: int):
        """Abstract method for generating a signal
        """
        # overwrite this method
        raise NotImplementedError
