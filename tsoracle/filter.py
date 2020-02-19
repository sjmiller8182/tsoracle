"""Filtering
"""

from typing import Tuple, List, Union
# this package
from tsoracle.API import Filter
# libs
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
# custom types
from numpy import ndarray
from matplotlib.axes import Axes

class MovingAverage(Filter):
    """A moving average filter

    Methods
    -------
    apply(realization)
        Apply filter to realization
    plot_response(ax = None, figsize = (10, 10), points = 2000)
        Plot the frequency response of the filter
    """

    def __init__(self,
                 order: int) -> None:
        """
        Parameters
        ----------
        order: integer
            Order of the filter.
        """
        if not isinstance(order, int):
            raise TypeError('Input for order must an integer.')
        if order < 2:
            raise ValueError('Order of a moving average filter must be at least 2.')
        super().__init__(np.ones(order), order)

class Butterworth(Filter):
    """A Butterworth filter

    **Sample input for a lowpass filter**

    * cutoff: [0.01] - cutoff of 0.01
    * type: 'lowpass' - lowpass filtering
    * order: 5 - 5th order filter

    **Sample input for a bandpass filter**

    * cutoff: [0.01, 0.25] - cutoff of 0.01 and 0.25
    * type: 'band' - bandpass filtering
    * order: 5 - 5th order filter

    Methods
    -------
    apply(realization)
        Apply filter to realization
    plot_response(ax = None, figsize = (10, 10), points = 2000)
        Plot the frequency response of the filter

    """

    def __init__(self,
                 cutoff: Union[List[float]],
                 type: str,
                 order: int = 5) -> None:
        """
        Parameters
        ----------
        order: integer
            Order of the filter.
        """
        super().__init__(*butter(order, cutoff, btype=type))
