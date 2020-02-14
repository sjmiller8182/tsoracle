
from typing import Callable, Tuple, List, Union

# anaconda api
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import periodogram, parzen
from statsmodels.tsa.stattools import acf

# custom types
from numpy import ndarray
from pandas import Series

def sample(x: Union[ndarray, Series, List],
           figsize: Tuple[float, float] = None,
           window: Callable = None) -> None:
    """ Show the typical sample plots.

    The following plots are generated in a grid

    * Realization
    * Sample autocorrelations
    * Periodogram
    * Windowed Spectral Density

    Acts like `tswge::plotts.sample.wge()`.

    Parameters
    ----------
    x: list-like
    	A time series realization.
    figsize: Tuple[float, float], optional
        Size of the plotting grid.
        If `None` is provided, `(10, 10)` will be used.
    window: Callable, optional
        A callable windowing function.
        Must work like window functions from `scipy.signal`.
        If `None` is provided, `scipy.signal.parzen()` will be used.

    """
    
    # attempt coercion to ndarray
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    # calculate periodogram
    per_x, per_y = periodogram(x)
    
    # calculate acf
    acf_mag = acf(x, nlags = np.sqrt(x.size).astype(int))
    lags = np.arange(acf_mag.size)
    
    # calculate window
    if window is None:
        window = parzen
    windowed_sd = x * window(x.size)
    window_name = window.__name__.capitalize()
    
    # set figsize to default if none is given
    if figsize is None:
        figsize = (10, 10)
        
    # plotting
    fig, ax = plt.subplots(2,2, figsize = figsize)
    
    fig.suptitle('Sample Plots')
    
    # plot the realization
    ax[0,0].plot(x, color = 'black')
    ax[0,0].set_xlabel('Time')
    ax[0,0].set_title('Realization')
    
    # plot the sample acf
    for lag, mag in zip(lags, acf_mag):
        ax[0,1].vlines(lag, 0, mag)
    ax[0,1].set_title('Sample Autocorrelations')
    ax[0,1].set_xlabel('lag')
    
    # plot periodogram
    for pos, mag in zip(per_x, per_y):
        ax[1,0].vlines(pos, 0, mag)
    ax[1,0].set_title('Periodogram')
    ax[1,0].set_xlabel('Frequency')
    
    # plot windowed periodogram
    ax[1,1].plot(*periodogram(windowed_sd), color = 'black')
    ax[1,1].set_title('Spectral Density (' + window_name + ')')
    ax[1,1].set_xlabel('Frequency')
