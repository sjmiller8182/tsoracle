"""API Abstracts
"""

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, List, Union

# libs
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
# custom types
from numpy import ndarray
from pandas import Series
from matplotlib.axes import Axes

class Generator(ABC):
    """Abstract class for signal generators
    """

    @abstractmethod
    def gen(self, size: int):
        """Abstract method for generating a signal
        """
        # overwrite this method
        raise NotImplementedError

class Model(ABC):
    """Abstract class for models
    """

    def __init__(self, model, realization, scorer, score):
        self.interal_model = model
        self.realization = None
        self.scorer = scorer
        self.score = score

    @abstractproperty
    def coef(self):
        """float, float: Get the fit model coefficients (slope, intercept)
        """
        raise NotImplementedError

    @property
    def fit_score(self) -> float:
        """float: Get the score from fitting the model to the realization
        """
        return self.score

    @abstractmethod
    def fit(self):
        """Fit model to data
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_noise(self):
        """Remove fitted linear model from realization
        """
        raise NotImplementedError

    @abstractmethod
    def plot_fit(self):
        """Plot the fit model and the realization together
        """
        raise NotImplementedError

class Filter(ABC):
    """Abstract class for filters
    """

    def __init__(self, b, a):
        self.b = b
        self.a = a

    def apply(self, realization: Union[ndarray, Series, List]) -> ndarray: 
        """Apply a filter to a realization

        Parameters
        ----------
        realization: list-like
            Apply the filter to the given realization.

        Returns
        -------
        filtered_signal: np.ndarray
            Result of filtered realization.
        """
        return lfilter(self.b, self.a, realization)

    def plot_response(self, 
                      ax: Axes = None, 
                      figsize: Tuple[float, float] = None,
                      points: int = 2000):
        """Plot the frequency response of the filter

        Parameters
        ----------
        ax: ax, optional
            An axes object to plot on. 
            If `None` is given, a new figure will be generated.
        figsize: Tuple[float, float], optional
            Size of the figure to plot.
            Only applies if `ax` is not supplied.
        points: integer, optional
            Number of points to use in generating the plot.
        """

        if not isinstance(points, int):
            raise TypeError('Input for size must an integer.')

        # setup plotting
        if ax is None:
            if figsize is None:
                figsize = (10, 10)
            fig, ax = plt.subplots(figsize = figsize)
            fig.suptitle('Filter Frequency Response')
        
        # get frequency response from filter transfer function
        w, h = freqz(self.b, self.a, worN=points)
        # make the plot
        # also need to scale w to frequency i.e. w = 2*pi*f
        ax.plot(w / (2 * np.pi), np.abs(h), color = 'black')
        ax.set_xlim(0, 0.5)
        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Frequency (Hz)')
