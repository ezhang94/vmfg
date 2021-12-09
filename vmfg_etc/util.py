from jax._src.numpy.lax_numpy import _arraylike
from typing import Sequence

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

def SHAPE_FN_NOT_IMPLEMENTED(sample_shape):  # pylint: disable=invalid-name
    """Raise NotImplementedError if shape_fn is called.

    Function defined in tfp substrate but not the jax substrate. Source:
    https://github.com/tensorflow/probability/blob/f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python/internal/parameter_properties.py

    """
    del sample_shape  # Unused
    raise NotImplementedError('No shape function is implemented for this parameter.')

class BatchedComponentProperties(ParameterProperties):
    """Alias to assist in defining properties of non-Tensor parameters.

    It is commonly used with definining properties of a distribution.
    Function defined in tfp substrate but not the jax substrate. Source:
    https://github.com/tensorflow/probability/blob/f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python/internal/parameter_properties.py    
    """

    def __new__(cls,
                event_ndims=0,
                event_ndims_tensor=None,
                default_constraining_bijector_fn=None,
                is_preferred=True):
        
        return super(BatchedComponentProperties, cls).__new__(
            cls=cls,
            event_ndims=event_ndims,
            event_ndims_tensor=event_ndims_tensor,
            shape_fn=SHAPE_FN_NOT_IMPLEMENTED,
            default_constraining_bijector_fn=default_constraining_bijector_fn,
            is_preferred=is_preferred,
            is_tensor=False,
            specifies_shape=False)

# ============================================================================
# Tree-structured graph functions
# ============================================================================

def parents_to_adjacency(par_list: Sequence[int]) -> _arraylike:
    """Converts condensed parent node encoding of tree-structured graphs into
    square adjacency matrices.

    Parameters
        par_list : sequence, length N
            Sequence of integers indicating the index of each parent, i.e.
            `par[i]` is the parent node of node `i`. All nodes are ordered, wlog,
            such that `par[i] <= i`. The parent of the root node is itself,
            i.e. `par_list[0] = 0` is always true.

    Returns
        UA : array_like, shape (N,N)
            Boolean upper triangular adjacency matrix, where `UA[j,i] == True`
            if nodes `i` and `j` are connected, and more specifically, if `j` is
            the parent node of `i`. The full adjacency matrix is constructed as
            `UA + UA.T`
    """

    N  = len(par_list)
    UA = np.zeros((N,N), dtype=bool)
    
    for ii, j in enumerate(par_list[1:]):
        i = ii+1
        UA = UA.at[j,i].set(True)

    return UA

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