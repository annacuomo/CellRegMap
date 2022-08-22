import unittest

from numpy import logical_and
from numpy.random import RandomState
from numpy.testing import assert_, assert_allclose, assert_equal

from cellregmap._simulate import (
    # column_normalize,
    # create_environment_matrix,
    # create_variances,
    # sample_covariance_matrix,
    sample_genotype,
    # sample_gxe_effects,
    sample_maf,
    # sample_noise_effects,
    # sample_persistent_effsizes,
    # sample_phenotype,
)

# define parameters
random = RandomState(0)
n_snps = 30
n_samples = 3
maf_min = 0.2
maf_max = 0.3

class TestSimulations(unittest.TestCase):

    def test_sample_maf(self):
        mafs = sample_maf(n_snps, maf_min, maf_max, random)
        assert_(all(logical_and(maf_min <= mafs, mafs <= maf_max)))
        assert_(len(mafs) == n_snps)

    def test_sample_genotype(self):
        random = RandomState(0)
        mafs = sample_maf(n_snps, maf_min, maf_max, random)
        G = sample_genotype(n_samples, mafs, random)
        assert_(G.shape == (n_samples, n_snps)) 

        A = set(list(G.ravel()))
        B = set([0.0, 1.0, 2.0]) # check this is biallelic SNPs
        assert_(A - B == set())