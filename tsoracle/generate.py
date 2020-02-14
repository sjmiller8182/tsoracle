
from typing import Callable, Tuple, List, Union

# anaconda API
import numpy as np

# custom types
from numpy import ndarray
from pandas import Series

def noise(var: float, size: int) -> ndarray:
    """ Generate sequential noise from a random normal .

    Parameters
    ----------
    var: scalar float
    	Nosie variance level.
    size: scalar int
        Number of samples to generate.
    
    Returns
    -------
    noise: np.ndarray
        Sequential noise.

    """
    return np.random.normal(scale = np.sqrt(var),
                            size = size)

def linear(intercept: float,
           slope: float, 
           size: int, 
           var: float = 0.01):
    """ Generate linear signal plus noise.

    Parameters
    ----------
    intercept: scalar float
    	Intercept of linear signal.
    slope: scalar float
        Slope of linear signal.
    size: scalar int
        Number of samples to generate.
    var: scalar float, optional
    	Nosie variance level.

    Returns
    -------
    signal: np.ndarray
        Sequential linear signal.

    """
    # generate time samples
    time_index = np.arange(size)
    # get noise
    sig_noise = noise(var = var, size = time_index.size)
    # calculate signal
    signal = slope * time_index + intercept

    return signal + sig_noise

def sinusoidal(mag: Union[float, ndarray, Series, List],
              freq: Union[float, ndarray, Series, List],
              shift: Union[float, ndarray, Series, List], 
              size: int, 
              var: float = 0.01):
    """ Generate sinusoidal signal plus noise.

    Parameters
    ----------
    mag: scalar or list-like
    	Signal magnitudes(ies).
    freq: scalar or list-like
        Signal frequency(ies).
    shift: scalar or list-like
        Phase shift(s).
    size: scalar int
        Number of samples to generate.
    var: scalar float, optional
    	Nosie variance level.

    Returns
    -------
    signal: np.ndarray
        Sequential sinusoidal signal.

    """
    mag = np.array(mag).reshape(np.array(mag).size, 1)
    freq = np.array(freq).reshape(np.array(freq).size, 1)
    shift = np.array(shift).reshape(np.array(shift).size, 1)

    # generate time samples
    time_index = np.empty((mag.size, size))
    for i, _ in enumerate(time_index):
        time_index[i] = np.linspace(-np.pi, np.pi, size)
    # calculate signal
    signal = np.sum(mag * np.sin(2 * np.pi * freq * time_index + shift), axis = 0)
    # get noise
    sig_noise = noise(var = var, size = size)

    return signal + sig_noise
