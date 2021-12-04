import pytest

from jax.experimental import enable_x64 as enable_x64_context

import jax.numpy as np
import scipy.special
import jax.random as jr

from vmfg import util

def test_log_bessel_iv_asymptotic():
    nu = 5.

    # JAX defaults to float32's, so choose a small `z` to avoid inf-ing out
    z = 50.
    test_val = util.log_bessel_iv_asymptotic(z)
    refr_val = np.log(scipy.special.iv(nu, z)) 
    
    # This is an approximation, so just ensure that the  test value is within
    # 1e0 of the reference value
    assert np.isclose(test_val, refr_val, atol=1., rtol=0.)

    # When `z`` too large, np.log in reference value returns `inf` due to
    # float32 limits, while direct log calculation is okay. Use float64's to
    # ensure that approximation is still okay at large z's
    # Technically, approximation should be better for large z's than small z's
    with enable_x64_context(True):
        z = 500.
        test_val = util.log_bessel_iv_asymptotic(z)
        refr_val = np.log(scipy.special.iv(nu, z)) 

        assert np.isclose(test_val, refr_val, atol=1., rtol=0.)

def test_log_sinh():
    x = 20.

    # This is an exact expression, so ensure that values are very close
    test_val = util.log_sinh(x)
    refr_val = np.log(np.sinh(x))

    assert np.isclose(test_val, refr_val)
    assert test_val == refr_val

def test_coth():
    x = 50.

    # This is an exact expression, so ensure that values are very close
    test_val = util.coth(x)
    refr_val = 1/np.tanh(x)

    assert np.isclose(test_val, refr_val)
    assert test_val == refr_val

def test_coth_asymptotic():
    x = np.array([1e-32, 1e32])

    test_val = util.coth(x)
    refr_val = np.array([np.nan_to_num(np.inf), 1.])

    assert np.all(test_val == refr_val)

def test_log_vmf_normalizer():
    """Test that the (approximate) general calculation of the log vMF normalizer
    is close to the exact calculation that exists for D=3.
    """
    # Low concentration, nearly a uniform distribution over the sphere
    concentration = 1e-3
    exact_val = util.log_vmf_3_normalizer(concentration)
    approx_val = util.log_vmf_d_normalizer(concentration, 3)
    err_small = np.abs(exact_val - approx_val)
    assert np.isclose(approx_val, exact_val, rtol=1e1)

    # Midling concentration
    concentration = 20
    exact_val = util.log_vmf_3_normalizer(concentration)
    approx_val = util.log_vmf_d_normalizer(concentration, 3)
    assert np.isclose(approx_val, exact_val, rtol=1e-5)
    assert np.abs(exact_val - approx_val) < err_small
    
    # High concentration
    concentration = 100
    exact_val = util.log_vmf_3_normalizer(concentration)
    approx_val = util.log_vmf_d_normalizer(concentration, 3)

    assert np.isclose(approx_val, exact_val, rtol=1e-5)
    assert np.abs(exact_val - approx_val) < err_small

def test_parents_to_adjacency():
    seed = jr.PRNGKey(1020)
    N = 10                                          # Number of nodes in tree

    parents = np.array(
        [jr.randint(jr.fold_in(seed, i), (), 0, i, int)
        for i in range(N)]
    )

    # Parent of root node is itself
    assert parents[0] == 0

    # Nodes should be ordered such that parent nodes always precede given node
    assert np.all(parents <= np.arange(N))

    # ------------------------------------------------------------------------
    # Convert condensed list into (N,N) square upper adjacency matrix
    upper_adj_mat = util.parents_list_to_adjacency_mat(parents)

    # Sum over rows: Each node (except root) should have exactly 1 parent
    assert np.all(upper_adj_mat[:,1:].sum(axis=0) == 1)
    
    # Sum over columns: Indicates the number of children each node has
    nodes_with_children, ref_num_children = \
                                    np.unique(parents[1:], return_counts=True)
    num_children_per_node = upper_adj_mat.sum(axis=-1)

    assert np.all(np.nonzero(num_children_per_node)[0] == nodes_with_children)
    assert np.all(num_children_per_node[nodes_with_children] == ref_num_children)
