"""Forecasting
"""

from typing import Union, List
from numpy import ndarray
from pandas import Series

import numpy as np

from tsoracle.factor import to_glp
    
def get_glp_multipliers(psi_values: Union[List, ndarray, Series]):
    """Calculate the psi coefficients for each forecast step from 
    the model GLP form.
    
    Parameters
    ----------
    psi_values: list-like
        The GLP coefficients of a model.
    
    Returns
    -------
    forecast_glp_multipliers: ndarray
        The half width of the forecast limit for each step.
    """
    psi_terms = np.empty(psi_values.shape[0])
    for x in range(psi_values.shape[0]):
        psi_terms[x] = np.sqrt((psi_values[:x + 1] ** 2).sum())
    return psi_terms

def get_half_limits(phi = None,
                    theta = None,
                    steps: int = 0,
                    wnv: float = 1.0,
                    conf_level = 1.96):
    """Calculate the GLP coefficients
    
    The inputs are defined as model phi and theta coefficients
    
    As an example, the following model
    
    (1 - 0.9 B) X_t = (1 + 0.3 B + 0.4 B^2) a_t
    
    should be input as
    
    phi = [ 0.9 ]
    theta = [ -0.3 , -0.4 ]
    
    Parameters
    ----------
    phi: list-like
        A set of AR coefficients i.e. the phis.
    theta: list-like
        A set of MA coefficients i.e. the thetas.
    steps: int
        The number of forecast steps.
    wnv: float
        The white noise variance
    conf_level: float
        The forecast confidence level as the multiplier
        (default is 95% confidence).
    
    Returns
    -------
    limit_half_widths: ndarray
        The half width of the forecast limit for each step.
    """
    # convert the process into glp form 
    glp_form = to_glp(phi, theta, steps)
    # calculate the half width intervals from the glp form
    return conf_level * np.sqrt(wnv) *  get_glp_multipliers(glp_form)
