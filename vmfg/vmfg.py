"""
Implementation of the von Mises-Fisher-Gaussian distribution,
as presented in

[1] Mukhopadhyay M, Li D, and Dunson, DB. "Estimating densities with nonlinear
support using Fisher-Gaussian kernels." Journal of the Royal Statistical
Society: Series B (Statistical Methodology). 2020; 82: 1249–1271.
"""


from vmfg.util import *

import jax.numpy as np
import jax.random as jr

import tensorflow_probability.substrates.jax.distributions as tfd

from tensorflow_probability.python.internal.reparameterization \
                            import FULLY_REPARAMETERIZED, NOT_REPARAMETERIZED
from tensorflow_probability.substrates.jax.util import ParameterProperties
from vmfg.util import PositiveScalarProperties
from tensorflow_probability.substrates.jax.internal.parameter_properties \
                                                import BIJECTOR_NOT_IMPLEMENTED

from typing import Callable, Optional
from jax._src.numpy.lax_numpy import _arraylike

CONC_REGULARIZEDR = 1e-8
VARIANCE_REGULARIZER = 1e-8

class VonMisesFisherGaussian(tfd.Distribution):
    """The von Mises-Fisher-Gaussian distribution over data near the surface
    of a sphere.

    Parameters
        mean_direction: shape ([B1,...Bb],D)
            A unit vector indicating the mode of the distribution.
            Note: dimensionality `D` is restricted to {2,3}.
        concentration: shape ([B1,...,Bb],)
            A positive value indicating the concentration of samples around
            `mean_direction` on the sphere. A value of 0 corresponds to
            the uniform distribution on the hypersphere, and INF corresponds to
            a delta function at `mean_direction`.
        scale: shape ([B1,...,Bb],)
            A positive value indicating spread of samples off the surface of
            the hypersphere.
        center: shape ([B1,...,Bb], D), optional. default: origin
            Location of the center of the hypersphere.
        radius: shape ([B1,...,Bb]), optional. default: 1
            Radius of the hypersphere.
    """
    def __init__(self,
                 mean_direction: _arraylike,
                 concentration: _arraylike,
                 scale: _arraylike,
                 center: Optional[_arraylike]=None, 
                 radius: Optional[_arraylike]=None,
                ):

        parameters = dict(locals())
        
        self._mean_direction = mean_direction
        self._concentration = concentration
        self._scale = scale
        self._center = center if center is not None \
                       else np.zeros_like(self._mean_direction)
        self._radius = radius if radius is not None \
                       else np.ones_like(self._concentration)

        # From tfp.VonMisesFisher source code:
        #   "mean_direction is always reparameterized. concentration is only
        #   reparameterized for event_dim==3, via an inversion sampler."
        _event_dim = self._event_shape()
        reparameterization_type = (
            FULLY_REPARAMETERIZED
            if _event_dim == 3 \
            else NOT_REPARAMETERIZED
        )


        super(VonMisesFisherGaussian, self).__init__(
            self._mean_direction.dtype,
            reparameterization_type,
            validate_args=False,
            allow_nan_stats=False,
            parameters=parameters,
            name='VonMisesFisherGaussian',
        )

        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None) -> dict:
        """Return a mapping of constructor arguments to expected properties."""
        # Set low value of softplus bijector to machine precision of given dtype
        softplus_low = np.finfo(dtype).eps

        return dict(
            mean_direction=ParameterProperties(
                event_ndims=1,
                default_constraining_bijector_fn=BIJECTOR_NOT_IMPLEMENTED,
            ),
            concentration=PositiveScalarProperties(softplus_low),
            scale=PositiveScalarProperties(softplus_low),
            center=ParameterProperties(event_ndims=1,),
            radius=PositiveScalarProperties(softplus_low),
        )

    def _event_shape(self):
        return self.mean_direction.shape[-1:]

    def _event_shape_tensor(self, mean_direction=None):
        return self.mean_direction.shape[-1:] \
               if mean_direction is None else mean_direction.shape[-1:]

    @property
    def mean_direction(self,):
        return self._mean_direction

    @property
    def concentration(self,):
        return self._concentration

    @property
    def scale(self,):
        return self._scale

    @property
    def center(self,):
        return self._center

    @property
    def radius(self,):
        return self._radius

    # -------------------------------------------------------------------------

    def _sample_n(self, n, seed):
        seed_ = jr.split(seed)
        dirs = tfd.VonMisesFisher(
                self._mean_direction, self._concentration
                )._sample_n(n, seed_[0])
        
        pos = jr.normal(seed_[1], shape=dirs.shape)
        pos *= self._scale[...,None]
        pos += self._radius[...,None] * dirs
        pos += self._center

        return pos
    
    def _log_unnormalized_prob(self, x: _arraylike):
        """Argument of exponential, in Eqn. 4 of [1]."""

        x0 = x - self._center                      # Translate samples to origin

        lp = np.einsum('...d, ...d -> ...', x0, x0)
        lp += self._radius[...,None] ** 2
        lp /= (-2 * self._scale**2)

        return lp

    def _log_prob(self, x: _arraylike) -> _arraylike:
        return self._log_unnormalized_prob(x, concentration) - self.log_normalization(concentration)
    
    def _mean(self,
             ):
        raise NotImplementedError

    def _mode(self,
             ):
        raise NotImplementedError
    
    def _variance(self,
                 ):
        raise NotImplementedError
    
    def _entropy(self,
                ):
        raise NotImplementedError
    
    def _sample_and_log_prob(self,
                ):
        raise NotImplementedError

    def _maximum_likelihood_parameters(self,
                                      ):
        raise NotImplementedError