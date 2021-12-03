from jax._src.numpy.lax_numpy import _arraylike

import jax.numpy as np

from tensorflow_probability.substrates.jax.util import ParameterProperties
from tensorflow_probability.substrates.jax.bijectors import softplus as softplus_bijector

# ============================================================================
# TFP distribution helper functions and classes
# ============================================================================

class PositiveScalarProperties(ParameterProperties):
    """Alias to assist in defining properties of positive scalar Tensor parameters."""

    def __new__(cls, softplus_low:float, is_preferred:bool=True):
        return super(PositiveScalarProperties, cls).__new__(
            cls=cls,
            event_ndims=0,                                      # default
            event_ndims_tensor=None,                            # default
            shape_fn=lambda sample_shape: sample_shape[:-1],    # modified: scalar parameter for a vector-support distr
            default_constraining_bijector_fn=                   # modified: ensure positive
                lambda: softplus_bijector.Softplus(low=softplus_low),
            is_preferred=is_preferred,                          # user specifiable
            is_tensor=True,                                     # default
            specifies_shape=False,                              # default
        )


# ============================================================================
# Safe math
# ============================================================================

def log_bessel_iv_asymptotic(x:_arraylike) -> _arraylike:
    """Logarithm of the asymptotic value of the modified Bessel function of
    the first kind :math:`I_nu`, for any order :math:`nu`.
    The asymptotic representation is given by Equation B.49 of the source as
    .. math:
        I_\nu(x) = \frac{1}{\sqrt{2\pi}} x^{-1/2} e^{x}
    for :math:`x\rightarrow\infty` with :math:`|\arg(x)| < \pi/2`.

    Source:
        Mainardi, F. "Appendix B: The Bessel Functions" in
        "Fractional Calculus and Waves in Linear Viscoelasticity,"
        World Scientific, 2010.
    """
    return x - 0.5*np.log(2*np.pi*x)

def log_sinh(x:_arraylike) -> _arraylike:
    """Calculate the log of the hyperbolic sine function in a numerically stable manner.
    
    The sinh function is defined as
    .. math:
        \sinh(x) = \frac{e^x - e^{-x}}{2} = \frac{e^x * (1-e^{-2x}}{2}
    which yields the equation
    .. math:
        \log \sinh(x) = x + \log (1-e^{-2x}) - \log 2
    """
    return x + np.log(1 - np.exp(-2*x)) - np.log(2)

def coth(x:_arraylike) -> _arraylike:
    """Calculate the hyperbolic cotangent function and catch non-finite cases.
    The hyperbolic cotangent, which is the inverse of the hyperbolic tangent,
    is defined as 
    .. math:
         \coth(x) = \frac{e^{2x} + 1}{e^{2x} - 1}
    
    The asymptotic values of the hyperbolic cotangent, are given by
    .. math:
         \lim_{x \rightarrow \infty} \coth(x) = 1
    and
    .. math:
         \lim_{x \rightarrow 0} \coth(x) = 1/x = \infty
    """

    out = (np.exp(2*x) + 1) / (np.exp(2*x) - 1)
    
    # Replace nan values (which occur when dividing inf by inf) with 1's
    # Replace posinf values with large finite number (via posinf=None)
    return np.nan_to_num(out, nan=1.0, posinf=None, neginf=-np.inf)

def log_vmf_d_normalizer(concentration:_arraylike, d:int) -> _arraylike:
    """Log normalizer of d-dimensional von Mises-Fisher distribution.

    Note: A simplified form exists when d=3; see: `log_vmf3_normalizer`.

    Parameters
        concentration: ([...],)
            A positive value indicating the concentration of the distribution.
        d: positive int, >2
            Dimension of the space in which the d-1 hypersphere is a subset.
    """

    lz = -0.5 * d * np.log(2*np.pi)
    lz -= (1 - 0.5*d) * np.log(concentration)
    lz -= log_bessel_iv_asymptotic(concentration)
    return lz

def log_vmf_3_normalizer(concentration:_arraylike) -> _arraylike:
    """Log normalizer of 3-dimensional von Mises-Fisher distribution.

    Note: For the more general d-dimensional case, see `log_vmf_normalizer`.
    
    Parameters
        concentration: ([...],)
            A positive value indicating the concentration of the distribution.
    """
    lz = np.log(concentration)
    lz -= np.log(4*np.pi)
    lz -= log_sinh(concentration)
    return lz