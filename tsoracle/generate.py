
from typing import Callable, Tuple, List, Union

# anaconda API
import numpy as np

# custom types
from numpy import ndarray
from pandas import Series

# API
from tsoracle.API import Generator

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

def arima_with_seasonality(size: int = 100,
                           phi: Union[float, ndarray] = 0,
                           theta: Union[float, ndarray] = 0,
                           d: int = 0,
                           s: int = 0,
                           var: float = 0.01) -> ndarray:
    """Simulate a realization from an ARIMA with seasonality characteristic.

    Parameters
    ----------
    size: scalar int
        Number of samples to generate.
    phi: scalar float or list-like
        AR process order
    theta: scalar float or list-like
        MA process order
    d: scalar int
        ARIMA process difference order
    s: scalar int
        Seasonality process order
    var: scalar float, optional
    	Nosie variance level.

    Returns
    -------
    signal: np.ndarray
        Simulated ARIMA with seasonality.
    """
    raise NotImplementedError

def arima(size: int = 100,
          phi: Union[float, ndarray] = 0,
          theta: Union[float, ndarray] = 0,
          d: int = 0,
          var: float = 0.01) -> ndarray:
    # inherit from arima_with_seasonality
    """Simulate a realization from an ARIMA characteristic.

    Parameters
    ----------
    size: scalar int
        Number of samples to generate.
    phi: scalar float or list-like
        AR process order
    theta: scalar float or list-like
        MA process order
    d: scalar int
        ARIMA process difference order
    var: scalar float, optional
    	Nosie variance level.

    Returns
    -------
    signal: np.ndarray
        Simulated ARIMA.
    """
    raise NotImplementedError

def arma(size: int = 100,
         phi: Union[float, ndarray] = 0,
         theta: Union[float, ndarray] = 0,
         var: float = 0.01) -> ndarray:
    # inherit from arima_with_seasonality
    """Simulate a realization from an ARMA characteristic.

    Parameters
    ----------
    size: scalar int
        Number of samples to generate.
    phi: scalar float or list-like
        AR process order
    theta: scalar float or list-like
        MA process order
    var: scalar float, optional
    	Nosie variance level.

    Returns
    -------
    signal: np.ndarray
        Simulated ARMA.
    """
    raise NotImplementedError

# Object-O API

class ARIMASignal(Generator):
    """Generator for ARUMA (ARIMA with seasonality) class signals.
    
    Attributes
    ----------
    phi: scalar float or list-like
        AR process order
    theta: scalar float or list-like
        MA process order
    d: scalar int
        ARIMA process difference order
    s: scalar int
        Seasonality process order
    var: scalar float, optional
        Nosie variance level.

    Methods
    -------
    gen(size)
        Generates a signal
    """

    def __init__(self, 
                 phi: Union[float, ndarray] = 0,
                 theta: Union[float, ndarray] = 0,
                 d: int = 0,
                 s: int = 0,
                 var: float = 0.01) -> None:
        """

        Parameters
        ----------
        phi: scalar float or list-like
            AR process order
        theta: scalar float or list-like
            MA process order
        d: scalar int
            ARIMA process difference order
        s: scalar int
            Seasonality process order
        var: scalar float, optional
            Nosie variance level.

        """

        self.phi = phi
        self.theta = theta
        self.d = d
        self.s = s
        self.var = var

    def gen(self, size: int) -> ndarray:
        """Generate a realization of given size.

        Parameters
        ----------
        size: scalar int
            Number of samples to generate.
            Must be strictly positive.

        Returns
        -------
        signal: np.ndarray
            Simulated ARMA.
        """

        return arima_with_seasonality(size, 
                                      self.phi, 
                                      self.theta, 
                                      self.d, 
                                      self.s, 
                                      self.var)

class LinearSignal(Generator):
    """Generator for linearly deterministic signals.
    
    Attributes
    ----------
    b0: scalar float or list-like
        AR process order
    b1: scalar float or list-like
        MA process order
    var: scalar float, optional
        Nosie variance level.

    Methods
    -------
    gen(size)
        Generates a signal
    """

    def __init__(self, 
                 intercept: float,
                 slope: float,  
                 var: float = 0.01) -> None:
        """

        Parameters
        ----------
        intercept: scalar float
            Intercept of linear signal.
        slope: scalar float
            Slope of linear signal.
        var: scalar float, optional
            Nosie variance level.

        """

        self.intercept = intercept
        self.slope = slope
        self.var = var

    def gen(self, size: int) -> ndarray:
        """Generate a realization of given size.

        Parameters
        ----------
        size: scalar int
            Number of samples to generate.
            Must be strictly positive.

        Returns
        -------
        signal: np.ndarray
            Simulated linear.
        """

        return linear(self.intercept,
                      self.slope,
                      size,
                      self.var)

class SinusoidalSignal(Generator):
    """Generator for sinusoidal deterministic signals.
    
    Attributes
    ----------
    mag: scalar or list-like
    	Signal magnitudes(ies).
    freq: scalar or list-like
        Signal frequency(ies).
    shift: scalar or list-like
        Phase shift(s).
    var: scalar float, optional
    	Nosie variance level.

    Methods
    -------
    gen(size)
        Generates a signal
    """

    def __init__(self, 
                 mag: Union[float, ndarray, Series, List],
                 freq: Union[float, ndarray, Series, List],
                 shift: Union[float, ndarray, Series, List],
                 var: float = 0.01) -> None:
        """
        Parameters
        ----------
        mag: scalar or list-like
            Signal magnitudes(ies).
        freq: scalar or list-like
            Signal frequency(ies).
        shift: scalar or list-like
            Phase shift(s).
        var: scalar float, optional
            Nosie variance level.
        """

        self.mag = mag
        self.freq = freq
        self.shift = shift
        self.var = var

    def gen(self, size: int) -> ndarray:
        """Generate a realization of given size.

        Parameters
        ----------
        size: scalar int
            Number of samples to generate.
            Must be strictly positive.

        Returns
        -------
        signal: np.ndarray
            Simulated linear.
        """

        return sinusoidal(self.mag,
                          self.freq,
                          self.shift, 
                          size,
                          self.var)
