
from typing import Callable, Tuple, List, Union

# anaconda api
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.signal.windows import parzen
from statsmodels.tsa.stattools import acf

# custom types
from numpy import ndarray
from pandas import Series

def realization(x: Union[ndarray, Series, List],
                figsize: Tuple[float, float] = None) -> None:
    """Plot a given realization

    Acts like `tswge::plotts.wge()`.

    Parameters
    ----------
    x: list-like
    	A time series realization.
    figsize: Tuple[float, float], optional
        Size of the plotting grid.
        If `None` is provided, `(10, 10)` will be used.

    """

    # attempt coercion to ndarray
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    fig, ax = plt.subplots(figsize = figsize)
    
    # set figsize to default if none is given
    if figsize is None:
        figsize = (10, 10)

    fig.suptitle('Sample Plots')
    
    # plot the realization
    ax.plot(x, color = 'black')
    ax.set_xlabel('Time')
    ax.set_title('Realization')


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
        Must work like window functions from `scipy.signal.windows`.
        If `None` is provided, `scipy.signal.windows.parzen()` will be used.

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

def constellation(zeros: np.ndarray,
                  poles: np.ndarray = None,
                  scale_magnitude = False,
                  scale_factor: float = 1000,
                  ax = None,
                  figsize = None) -> None:
    """Plots roots on the complex plane with reference to the unit circle
    
    Parameters
    ----------
    zeros: list-like
        A set of roots.
    poles: list-like
        A set of roots.
    scale_magnitude: bool
        Option to scale the marker based on the 
        absolute magnitude of the root.
    scale_factor: float
        Marker scaling factor.
        Only used if `scale_magnitude` is set to `True`.
    ax: float
        The forecast confidence level as the multiplier
        (default is 95% confidence).
    figsize: Tuple[float, float]
        Size of the figure to generate.
        Only used if `None` provided to `ax`.
    """
    
    # create unit circle corrds
    x_corr = np.array([x for x in range(-1000, 1001)]) / 1000
    y_corr = np.sqrt(1 - x_corr ** 2)
    
    # create plot
    if ax is None:
        if figsize is None:
            # set a default for figsize
            figsize = (10, 10)
        # generate a new plotting axis if none is given
        fig, ax = plt.subplots(figsize = (10, 10))
    ax.grid()
    ax.set_ylabel('Imaginary')
    ax.set_xlabel('Real')
    ax.set_title('Constallation Plot')
    
    # plot the unit circle
    ax.plot(x_corr, y_corr, color = 'blue')
    ax.plot(x_corr, np.negative(y_corr), color = 'blue')
    
    # concat to consider all roots
    # to set plot limits correctly
    if poles is not None:
        roots = np.concatenate((zeros, poles))
    else: 
        roots = zeros
    
    # set plotting scales to preserve squareness
    max_abs = np.array(
        [
            np.abs(roots.imag.max()),
            np.abs(roots.imag.min()),
            np.abs(roots.real.max()),
            np.abs(roots.real.min())
        ]
    ).max()
    # make limits square
    ax.set_ylim(-max_abs - 0.5, max_abs + 0.5)
    ax.set_xlim(max_abs + 0.5, -max_abs - 0.5)

    # check for roots in the unit circle
    zeros_in_circle_idx = np.where(np.abs(zeros) < 1.0)
    zeros_idx = np.where(np.abs(zeros) >= 1.0)
    
    # create size scaling from root sizes
    size = None
    if scale_magnitude:
        size = np.reciprocal(np.exp(np.abs(zeros[zeros_idx]))) * scale_factor
    # plot the zeros
    ax.scatter(zeros[zeros_idx].real,
               zeros[zeros_idx].imag,
               s = size, label = 'Zeros')
    if zeros[zeros_in_circle_idx].size != 0:
        ax.scatter(zeros[zeros_in_circle_idx].real,
                   zeros[zeros_in_circle_idx].imag,
                   s = 100, color = 'red',
                   label = 'Unstable Zeros')
    
    # plot the poles
    if poles is not None:
        
        # check for roots in the unit circle
        poles_in_circle_idx = np.where(np.abs(poles) < 1.0)
        poles_idx = np.where(np.abs(poles) >= 1.0)
        
        size = None
        if scale_magnitude:
            size = np.reciprocal(np.exp(np.abs(poles))) * scale_factor
        ax.scatter(poles.real[poles_idx], poles.imag[poles_idx],
                   s = size, marker = 'x', label = 'Poles')
        
        if poles[poles_in_circle_idx].size != 0:
            ax.scatter(poles[poles_in_circle_idx].real,
                       poles[poles_in_circle_idx].imag,
                       s = 100, marker = 'x',
                       color = 'red', label = 'Unstable Poles')
        # only create the legend when both sets of roots are provided
        ax.legend()

def cross_lag_plot(series,
                   lagged_series,
                   max_lag,
                   names = None,
                   ncols = 2,
                   display_corr = True,
                   figsize = None):
    """Create scatter plots of `series` vs lags of `lagged_series`.
    `series` and `lagged_series` must be the same size.
    
    Parameters
    ----------
    series: list-like
        A time series.
    lagged_series: list-like
        A time series to lag against `series`.
    max_lag: int
        The number of lag plots to produce.
    names: list-like
        Names of the two series that are plotted:
        `(series, lagged_series)`.
    ncols: int
        number of columns to use in plot.
    display_corr: bool
        Whether to display the correlation coefficient on the lag plots.
    figsize: Tuple
        Size of the figure.
    """
    # figure setup
    fig = plt.figure(figsize = figsize)
    fig.tight_layout()
    if names is not None:
        fig.suptitle(f'{names[0]} vs Lags of {names[1]}')
    # create lag plots
    for i in range(max_lag):
        ax = plt.subplot2grid((max_lag // ncols, ncols),
                              [i // ncols, i % ncols],
                              fig = fig)
        ax.scatter(series[:-i-1], lagged_series[i + 1:])
        ax.set_title(f'Lag {i+1}')
        # display the correlation coefficients between the series and
        # the lagged series
        if display_corr:
            correlation = np.corrcoef(series[:-(i + 1)],
                                      lagged_series[i + 1:])
            y = ax.get_ylim()[1] * 0.8
            if correlation[0][1] > 0:
                x = ax.get_xlim()[0] * 0.8
            else:
                x = ax.get_xlim()[1] * 0.3
            ax.text(x, y, 'Corr {:.3f}'.format(correlation[0][1]))
