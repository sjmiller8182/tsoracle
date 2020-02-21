
from typing import List, Union

# numpy
import numpy as np
# polynomials
from numpy.polynomial.polynomial import polyroots, polyfromroots, polymul
from scipy.signal import lfilter
# types
from numpy import poly1d, ndarray

# constants
RECIPROCAL_2PI = 0.15915494309189535
"""Constant for system frequency calculation: 1 / (2 * pi)
"""

def get_roots(coef: Union[ndarray, List]) -> ndarray:
    """Calculate roots from an AR or MA polynomial

    Parameters
    ----------
    coef: list-like
        A set of AR or MA coefficients i.e. the phis and thetas.

        An example AR polynomial is
        
        (1 - 1.2 B + 0.4 B^2) X_t = 0
    
        This would be entered as 
        
        get_roots( [ [1.2, -0.4] ] )

        and the output would be 

        [1.5-0.5j, 1.5+0.5j]

    """
    polynomial = np.insert(np.negative(coef), 0, 1)
    return polyroots(polynomial)

def roots_in_unit_circle(phi:Union[List, ndarray] = 0,
                         theta:Union[List, ndarray] = None) -> bool:
    """Check if any roots from an AR or MA polynomial are inside the unit circle
    """

    # check for roots in phi section
    if phi is not None:
        phi_roots = get_roots(phi)
        roots_in_phi = np.any(np.absolute(phi_roots) < 1)
    else:
        roots_in_phi = False

    # check for roots in theta section
    if theta is not None:
        theta_roots = get_roots(theta)
        roots_in_theta = np.any(np.absolute(theta_roots) < 1)
    else:
        roots_in_theta = False

    return roots_in_theta or roots_in_phi

def get_system_freq(first_order_coef: float, sec_order_coef: float) -> float:
    """Calculates the system frequency for a 2nd degree polynomial
    with complex roots.

    Parameters
    ----------
    first_order_coef: float
        The coefficient of the first order term.
    sec_order_coef: float
        The coefficient of the second order term.

    Returns
    -------
    frequency: float
        The system frequency (Hz)

    """
    # calculate the radial angle from coefficients
    radial_angle = first_order_coef \
                    / (2 * np.sqrt(np.negative(sec_order_coef)))

    return (RECIPROCAL_2PI * np.arccos(radial_angle))

def table(polynomial: Union[ndarray, List]) -> str:
    """Create a factor table from a polynomial

    An example AR polynomial is
        
    (1 - 1.2 B + 0.4 B^2) X_t = 0

    This would be entered as 
    
    table( [1.2, -0.4] )

    Acts like `tswge::factor.wge`
    """

    roots = polyroots(polynomial)

    raise NotImplementedError

def to_glp(phi:Union[List, ndarray] = None,
           theta:Union[List, ndarray] = None,
           lags: int = 0) -> ndarray:
    """Calculate the GLP coefficients

    The inputs are defined as model phi and theta coefficients

    As an example, the following model

    (1 - 0.9 B) X_t = (1 + 0.3 B + 0.4 B^2) = a_t

    should be input as

    phi = [ 0.9 ]
    theta = [ -0.3 , -0.4 ]

    Parameters
    ----------
    phi: list-like
        A set of AR coefficients i.e. the phis.
    theta: list-like
        A set of MA coefficients i.e. the thetas.
    lags:
        The number of coefficients to calculate.
    
    Returns
    -------
    glp_coef: ndarray
        The coefficients representing the input model.
        The first element is always 1.

    Acts like `tswge::psi.weights.wge`
    """
    if theta is None:
        theta_poly = [1.0]
    else: 
        # negate poly and insert leading one
        theta_poly = np.insert(np.negative(theta),0 , 1.0)
    if phi is None:
        phi_poly = [1.0]
    else:
        phi_poly = np.insert(np.negative(phi),0 , 1.0)

    # psi values can be calculated as the impulse response to
    # the filter
    input_array = np.zeros(lags, dtype = float)
    input_array[0] = 1
    # return the filter response
    return lfilter(theta_poly, phi_poly, input_array)

def multiply(factors: List[List[float]]) -> ndarray:
    """Multiply together time series factors
    
    Parameters
    ----------
    factors: List of Lists
        List of factors to multiply together.
        Factors should be entered as phis and thetas
        
        An example AR polynomial is
        
        (1 + 0.8 B)(1 - 1.2 B + 0.4 B^2) X_t = 0
    
        This would be entered as 
        
        multiply( [ [-0.8] , [1.2, -0.4] ] )
    
    Acts like `tswge::mult.wge`
    """
    
    if not isinstance(factors, ndarray):
        nd_factors = np.array(factors)
    else:
        nd_factors = factors
    
    if nd_factors.shape[0] == 0:
        # raise error here, not enough factors
        pass
    elif nd_factors.shape[0] == 1:
        # only one factor, just return it
        return nd_factors
    else:
        # list of factors
        for i, factor in enumerate(nd_factors):
            # seed with the first polynomial
            factor = np.negative(factor)[::-1]
            if i == 0:
                accum_poly = np.append(factor, 1)
            else:
                accum_poly = polymul(accum_poly, np.append(factor, 1))
        accum_poly = np.negative(accum_poly[::-1])[1:]
        return accum_poly

def poly_to_string(poly: ndarray) -> str:
    """Convert a polynomial to a string representation of the full ploynomial

        An example AR polynomial is

        (1 + 0.1 B - 0.2 B^2) X_t = 0

        This would be entered as

        [-0.1,  0.2]

        And would produce

        1.0 + 0.1*x - 0.2*x^2
    """
    char_poly = np.append(np.negative(poly[::-1]),1)
    char =''
    for i, element in enumerate(char_poly[::-1]):
        if i == 0:
            char += str(np.round(element, 6))
        elif i == 1:
            if np.round(element, 6) >= 0:
                char += ' + '
            else:
                char += ' - '
            char += str(np.round(element, 6))
            char += '*x'
        else:
            if np.round(element, 6) >= 0:
                char += ' + '
            else:
                char += ' - '
            char += str(np.round(element, 6))[1:]
            char += ('*x^' + str(i))
    return char

class TSPolynomial:
    """Container for a time series polynomial.
    
    Attributes
    ----------
    model_coef: scalar or list-like
    	Signal magnitudes(ies).
    poly: str
        Signal frequency(ies).
    """

    def __init__(self, model_coef: Union[float, List]):
        self.model_coef = model_coef
        self.poly = None
    
    @property
    def coef(self) -> ndarray:
        """The model coefficients
        """
        return self.model_coef
    
    @property
    def poly_str(self) -> str:
        """A string representation of the input polynomial
        """
        # only create this if requested, probably don't need it most of the time
        if self.poly is None:
            self.poly = poly_to_string(self.model_coef)
        return self.poly
