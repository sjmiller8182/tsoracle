
from typing import Callable, Tuple, List, Union

# anaconda API
import numpy as np
from numpy.random import RandomState
from scipy.signal import lfilter

# custom types
from numpy import ndarray
from pandas import Series

# API
from tsoracle.API import Generator
from tsoracle.factor import roots_in_unit_circle

# functional API

def noise(var: Union[float, int], 
          size: int, 
          random_state: float = None) -> ndarray:
    """ Generate sequential noise from a random normal .

    Parameters
    ----------
    var: scalar float
    	Nosie variance level.
    size: scalar int
        Number of samples to generate, strictly positive.
    random_state: scalar int, optional
        Seed the random number generator
    
    Returns
    -------
    noise: np.ndarray
        Sequential noise.

    """

    if size < 1:
        raise ValueError('The value for size must be strictly positive')
    
    if var == 0:
        noise_signal = np.zeros(size)
    else:
        noise_signal = RandomState(random_state).normal(scale = np.sqrt(var),
                                                        size = size)

    return noise_signal

def linear(intercept: float,
           slope: float, 
           size: int, 
           var: float = 0.01, 
           random_state: float = None):
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
    random_state: scalar int, optional
        Seed the random number generator

    Returns
    -------
    signal: np.ndarray
        Sequential linear signal.

    """
    # check for input errors
    if size < 1:
        raise ValueError('The value for size must be strictly positive')

    # generate time samples
    time_index = np.arange(size)
    # get noise
    sig_noise = noise(var = var, 
                      size = time_index.size, 
                      random_state = random_state)
    # calculate signal
    signal = slope * time_index + intercept

    return signal + sig_noise

def sinusoidal(mag: Union[float, ndarray, Series, List],
              freq: Union[float, ndarray, Series, List],
              shift: Union[float, ndarray, Series, List], 
              size: int, 
              var: float = 0.01, 
              random_state: float = None):
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
    random_state: scalar int, optional
        Seed the random number generator.

    Returns
    -------
    signal: np.ndarray
        Sequential sinusoidal signal.

    """

    mag = np.array(mag).reshape(np.array(mag).size, 1)
    freq = np.array(freq).reshape(np.array(freq).size, 1)
    shift = np.array(shift).reshape(np.array(shift).size, 1)

    # check for input errors
    if size < 1:
        raise ValueError('The value for size must be strictly positive')

    # generate time samples
    time_index = np.empty((mag.size, size))
    for i, _ in enumerate(time_index):
        time_index[i] = np.linspace(-np.pi, np.pi, size)
    # calculate signal
    signal = np.sum(mag * np.sin(2 * np.pi * freq * time_index + shift),
                    axis = 0)
    # get noise
    sig_noise = noise(var = var,
                      size = size,
                      random_state = random_state)

    return signal + sig_noise

def arima_with_seasonality(size: int = 100,
                           phi: Union[float, ndarray] = 0,
                           theta: Union[float, ndarray] = 0,
                           d: int = 0,
                           s: int = 0,
                           var: float = 0.01, 
                           random_state: float = None) -> ndarray:
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
    random_state: scalar int, optional
        Seed the random number generator.

    Returns
    -------
    signal: np.ndarray
        Simulated ARIMA with seasonality.
    """

    # check for input errors
    if size < 1:
        raise ValueError('The value for size must be strictly positive')

    raise NotImplementedError

def arima(size: int = 100,
          phi: Union[float, ndarray] = 0,
          theta: Union[float, ndarray] = 0,
          d: int = 0,
          var: float = 0.01, 
          random_state: float = None) -> ndarray:
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
    random_state: scalar int, optional
        Seed the random number generator.

    Returns
    -------
    signal: np.ndarray
        Simulated ARIMA.
    """

    raise NotImplementedError

def arma(size: int = 100,
         phi: Union[float, ndarray] = 0,
         theta: Union[float, ndarray] = 0,
         var: float = 0.01, 
         random_state: float = None) -> ndarray:
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
    random_state: scalar int, optional
        Seed the random number generator.

    Returns
    -------
    signal: np.ndarray
        Simulated ARMA.
    """

    # check for input errors
    if size < 1:
        raise ValueError('The value for size must be strictly positive')

    if roots_in_unit_circle(phi, theta):
        raise ValueError('The input polynomials have roots in the unit circle. \
                          This is not an ARMA process.')

    arparams = np.array(phi)
    maparams = np.array(theta)
    arparams = np.r_[1, np.negative(arparams)]
    maparams = np.r_[1, maparams]

    sig_noise = noise(var,size,random_state = random_state)

    signal = lfilter(maparams, arparams, sig_noise)

    return signal

# Object-O API

class Noise(Generator):
    """Generator for noise.
    
    Attributes
    ----------
    var: scalar float, optional
        Nosie variance level.

    Methods
    -------
    gen(size)
        Generates a signal
    """

    def __init__(self, 
                 var: float = 0.01) -> None:
        """
        Parameters
        ----------
        var: scalar float, optional
            Nosie variance level.
        """
        self.var = var

    def gen(self, 
            size: int, 
            random_state: float = None) -> ndarray:
        """Generate a realization of given size.

        Parameters
        ----------
        size: scalar int
            Number of samples to generate.
            Must be strictly positive.
        random_state: scalar int, optional
            Seed the random number generator.

        Returns
        -------
        signal: np.ndarray
            Simulated noise.
        """

        return noise(self.var, size, random_state)

class ARIMA(Generator):
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
    factor_table(table_type)
        Get a factor table for the generator
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

    def gen(self, 
            size: int, 
            random_state: float = None) -> ndarray:
        """Generate a realization of given size.

        Parameters
        ----------
        size: scalar int
            Number of samples to generate.
            Must be strictly positive.
        random_state: scalar int, optional
            Seed the random number generator.

        Returns
        -------
        signal: np.ndarray
            Simulated ARIMA.
        """

        return arima_with_seasonality(size, 
                                      self.phi, 
                                      self.theta, 
                                      self.d, 
                                      self.s, 
                                      self.var,
                                      random_state)

    def factor_table(self, table_type: str = 'both'):
        """Create a factor table from the factors in the generator

        Parameters
        ----------
        table_type: str
            Options are 'both' | 'AR' | 'MA'
        
        Returns
        -------
        table: np.ndarray
            Formatted string containing the factor table.
            Use `print()`.
        """
        raise NotImplementedError

class Linear(Generator):
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

    def gen(self, 
            size: int, 
            random_state: float = None) -> ndarray:
        """Generate a realization of given size.

        Parameters
        ----------
        size: scalar int
            Number of samples to generate.
            Must be strictly positive.
        random_state: scalar int, optional
            Seed the random number generator.

        Returns
        -------
        signal: np.ndarray
            Simulated linear.
        """

        return linear(self.intercept,
                      self.slope,
                      size,
                      self.var,
                      random_state)

class Sinusoidal(Generator):
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

    def gen(self, 
            size: int, 
            random_state: float = None) -> ndarray:
        """Generate a realization of given size.

        Parameters
        ----------
        size: scalar int
            Number of samples to generate.
            Must be strictly positive.
        random_state: scalar int, optional
            Seed the random number generator.

        Returns
        -------
        signal: np.ndarray
            Simulated linear.
        """

        return sinusoidal(self.mag,
                          self.freq,
                          self.shift, 
                          size,
                          self.var,
                          random_state)
