"""
Implementation of the hierarchical von Mises-Fisher-Gaussian distribution,
with a tree-structured hierarchy. This distribution was presented in
    [2]
"""


from functools import partial

import jax.numpy as np
import jax.random as jr

from vmfg_etc import util

# TFP and helpers
import tensorflow_probability.substrates.jax.distributions as tfd

from tensorflow_probability.python.internal.reparameterization \
                            import FULLY_REPARAMETERIZED, NOT_REPARAMETERIZED
from vmfg_etc.util import BatchedComponentProperties

# Typing
from typing import Callable, Optional, Sequence
from jax._src.numpy.lax_numpy import _arraylike

# -----------------------------------------------------------------------------
CONC_REGULARIZEDR = 1e-8
VARIANCE_REGULARIZER = 1e-8

# INIT COMPLETED, NEXT: PARAMETER_PROPERTIES
class HierarchicalVonMisesFisherGaussian(tfd.Distribution):
    """The hierarchial von Mises-Fisher-Gaussian distribution over a collection
    data near the surface of a mixture of spheres. In particular, each sphere
    is centered at the mean direction, and the given radius, of its parent node.
    The generative model is given by

    .. math::
        x_0 \sim N(\mu_0, \Sigma_0)
        \textrm{for } i = 1,...K-1
        \quad u_i \sim vMF(\nu_i, \kappa_i) \qquad \textrm{for }
        \quad x_i \sim N(x_{\textrm{par[i]}} + \rho_i*u_i, \sigma_i^2 I)

    NOTE [**]: The root value (`K=0`) for parameters `mean_directions`,
    `concentrations`, `scale`, and `radius` are technically undefined, and thus
    ignored. They are retained so that indexing remains consistent.

    Parameters
        parents: sequence of ints, length K
            Parent list encoding of tree-structured hierarchy, such that
            `parents[i]` is the parent node of node `i`. The parent of the root
            node is itself, `parents[0]==0`.
        root_distribution: tfd.Distribution, optional.
            Multivariate normal distribution specifying location of root node.
            Event shape (D,). If none specified, defaults to N(0,I) * 1e3
        leaf_distribution: tfd.Distribution, optional.
            Batch of vMFG distributions specifying location of leaf nodes wrt
            their respective parent. Event shape (D,), rightmost batchshape (L,).
    """
    def __init__(self,
                 parents:Sequence[int],
                 root_distribution: tfd.Distribution=None,
                 leaf_distribution: tfd.Distribution=None,
                ):

        parameters = dict(locals())
        
        self._parents = parents
        self._root_distribution = root_distribution

        # Make sure that leaf distribution is represented as a set of
        # non-independent of vMFGs (for consistent log probability calculation later)
        assert len(leaf_distribution.event_shape) == 1
        assert leaf_distribution.batch_shape[-1] == len(parents)-1
        self._leaf_distribution = leaf_distribution
        
        super(HierarchicalVonMisesFisherGaussian, self).__init__(
            self._root_distribution.dtype,
            NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=False,
            parameters=parameters,
            name='HierarchicalVonMisesFisherGaussian',
        )

        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None) -> dict:
        """Return a mapping of constructor arguments to expected properties."""
        
        # TODO parents??
        return dict(
            root_distribution=BatchedComponentProperties(event_ndims=1),
            leaf_distribution=BatchedComponentProperties(event_ndims=1),
        )

    def _event_shape(self):
        num_leaves = self._leaf_distribution.batch_shape[-1]
        dim = self._root_distribution.event_shape[-1]
        return (num_leaves+1, dim)

    @property
    def parents(self,):
        return self._parents

    @property
    def root_distribution(self,):
        return self._root_distribution

    @property
    def leaf_distribution(self,):
        return self._leaf_distribution

    # -------------------------------------------------------------------------

    def _reconstruct_pos(self, root_pos: _arraylike, dleaf_pos:_arraylike) -> _arraylike:
        """Return absolute positions from root position and leaf positions
        relative to their parent.

        Params
            root_pos: array-like, shape (...,D)
            dleaf_pos: array-like, shape (...,K-1,D)
        
        Return
            abs_pos: array-like, shape (...,K,D)

        TODO This is not jittable...how to get a node's parents and grandparents
        (all the way up to root node), so that we might be able to jit
        """
        
        abs_pos = np.concatenate([root_pos[...,None,:], dleaf_pos], axis=-2)
        
        for k in range(1, len(self.parents)):
            par = self.parents[k]
            abs_pos = abs_pos.at[...,k,:].add(abs_pos[...,par,:])

        return abs_pos


    def _sample_n(self, n, seed):
        seed_ = jr.split(seed)

        root_samples = self.root_distribution.sample(n, seed_[0])
        dleaf_samples = self.leaf_distribution.sample(n, seed_[1])
        
        return self._reconstruct_pos(root_samples, dleaf_samples)

    def _log_prob(self, samples: _arraylike) -> _arraylike:
        """Marginal log probability of p(x) = int p(x | u) du
        
        Params
            samples: 
        """

        # Leaf positions relative to parent positions
        dx = samples - samples[...,self._parents,:]
        dx = dx[...,1:,:]

        lp = self._leaf_distribution.log_prob(dx)

        raise NotImplementedError

        # # Log of exp term
        # lp = np.einsum('...d, ...d -> ...', x0, x0)
        # lp += self._radius ** 2
        # lp /= (-2 * self._scale**2)

        # # Log normalization of posterior
        # cond_concentration = x0 * (self._radius/self._scale**2)[...,None]
        # cond_concentration += self._mean_direction * self._concentration[...,None]
        # cond_concentration = np.linalg.norm(cond_concentration, axis=-1)

        # lp -= self._log_vmf_normalizer(cond_concentration)
        
        # # Log normalization of priors
        # lp += self._log_vmf_normalizer(self._concentration)
        # lp -= 0.5 * self.dim * np.log(2*np.pi*(self._scale**2))

        # return lp

    def _mean(self):
        raise NotImplementedError

    def _mode(self,):
        root_mode = self.root_distribution.mode()
        dleaf_mode = self.leaf_distribution.mode()

        return self._reconstruct_pos(root_mode, dleaf_mode)
    
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