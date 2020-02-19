"""Fit models to data
"""

from typing import Tuple, List, Union, Callable
# this package
from tsoracle.API import Model
# libs
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# custom types
from numpy import ndarray
from pandas import Series
from matplotlib.axes import Axes

class Linear(Model):
    """Linear model

    Methods
    -------
    fit(size)
        Generates a signal
    get_noise(size)
        Generates a signal
    """
    def __init__(self, scorer: callable = None) -> None:

        
        # use MSE if no scorer is provided
        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = mean_squared_error
        
        model = LinearRegression()

        # init super class
        super().__init__(model,
                         None,
                         scorer,
                         None)
    
    # properites
    @property
    def coef(self) -> Tuple[float, float]:
        """float, float: Get the fit model coefficients (slope, intercept)
        """
        return self.interal_model.coef_[0], self.interal_model.intercept_

    # methods
    def fit(self, realization: Union[ndarray, Series, List]):
        """Fit model to data

        Parameters
        ----------
        realization: list-like
            Fit model to given realization.
        """
        # attempt coercion to ndarray
        # and save realization
        if not (isinstance(realization, ndarray) or isinstance(realization, Series)):
            self.realization = np.array(realization)
        else:
            self.realization = realization
        # make a time index
        time_index = np.arange(self.realization.size).reshape(-1, 1)
        # fit the internal model to the data
        self.interal_model.fit(time_index, self.realization)
        self.score = self.scorer(self.realization, 
                                 self.interal_model.predict(time_index))

    def get_noise(self, realization: Union[ndarray, Series, List] = None):
        """Remove fitted linear model from realization

        Parameters
        ----------
        realization: list-like
            A realization to compare with the model.
            If `None` is supplied,
            the realization used to fit the model will be used.

        Returns
        -------
        filtered_signal: np.ndarray
            Result of subtracting model from realization.
        """
        if realization is not None:
            # attempt coercion to ndarray
            if not (isinstance(realization, ndarray) or isinstance(realization, Series)):
                realization = np.array(realization)
        else:
            realization = self.realization
        # build a time index array
        time_index = np.arange(realization.size).reshape(-1, 1)
        # make the prediction for the indexes
        prediction = self.interal_model.predict(time_index)
        # substract the model from the realization
        return realization - prediction

    def plot_fit(self, 
                 realization: Union[ndarray, Series, List] = None,
                 ax: Axes = None, 
                 figsize: Tuple[float, float] = None,
                 xticks: List[str] = None):
        """Plot the fit model and the realization together

        Parameters
        ----------
        realization: list-like
            Plot the model together with a realization.
            If `None` is supplied,
            the realization used to fit the model will be used.
        ax: ax, optional
            An axes object to plot on. 
            If `None` is given, a new figure will be generated.
        figsize: Tuple[float, float], optional
            Size of the figure to plot.
            Only applies if `ax` is not supplied.
        xticks: list-like
            Labels for the x-axis
        """

        if realization is not None:
            # attempt coercion to ndarray
            if not (isinstance(realization, ndarray) or isinstance(realization, Series)):
                realization = np.array(realization)
        else:
            realization = self.realization

        # setup plotting
        if ax is None:
            if figsize is None:
                figsize = (10, 10)
            fig, ax = plt.subplots(figsize = figsize)
            fig.suptitle('Fit to Realization')

        time_index = np.arange(realization.size)
        model_points = self.interal_model.predict(time_index.reshape(-1, 1))

        ax.plot(time_index, realization, color = 'black', label = 'Realization')
        ax.plot(time_index, model_points, color = 'blue', label = 'Model')
        ax.legend()
        ax.set_ylabel('Realization')
        ax.set_xlabel('Time')
        if xticks is not None:
            ax.set_xticks(xticks)