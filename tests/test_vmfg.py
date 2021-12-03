import pytest

import jax.numpy as np
import jax.random as jr

from vmfg import VonMisesFisherGaussian

SEED = jr.PRNGKey(1325)

@pytest.fixture
def sample_shape():
    return (4,5,3) # (B1, B2, D)

@pytest.fixture
def vmfg(sample_shape):
    """Randomly instantiate a VonMisesFisherGuassian distribution object and
    use default `center` and `radius` values.
    """

    seed_ = iter(jr.split(SEED, 5))

    # Unit direrctional vectors
    mean_direction = jr.normal(next(seed_), sample_shape)
    mean_direction /= np.linalg.norm(mean_direction, axis=-1, keepdims=True)
    
    # Non-negative values
    concentration = jr.normal(next(seed_,), sample_shape[:-1]) * 10 + 50
    concentration = np.maximum(concentration, 5)

    scale = 1 / jr.gamma(next(seed_), 1, sample_shape[:-1])

    return VonMisesFisherGaussian(mean_direction,
                                  concentration,
                                  scale)

def test_instantiation(sample_shape, vmfg):
    batch_shape = sample_shape[:-1]
    event_shape = sample_shape[-1:]

    # Check shapes
    assert vmfg.batch_shape == batch_shape
    assert vmfg.event_shape == event_shape

    # Check default values for center and radius parameters
    assert np.all(vmfg.center == 0)
    assert np.all(vmfg.radius == 1)

def test_sample(sample_shape, vmfg):
    n = 500

    seed = jr.fold_in(SEED, 100)

    xs = vmfg.sample(seed=seed, sample_shape=(n,))

    assert xs.shape == (n, *sample_shape)

    # Pick arbitrary number that lps should be greater than
    lps = vmfg.log_prob(xs).mean(axis=0)
    assert np.all(lps > -20)