import unittest

from numpy import ones
from numpy.random import RandomState
from numpy_sugar import ddot
from numpy_sugar.linalg import economic_svd

from cellregmap import CellRegMap
from cellregmap import run_interaction, run_association, run_association_fast, run_gene_set_association

random = RandomState(10)
n = 100                              # number of samples (cells)
p = 10                               # number of individuals
k = 5                                # number of contexts
y = random.randn(n, 1)               # outcome vector (expression phenotype)
C = random.randn(n, k)               # context matrix  
W = ones((n, 1))                     # intercept (covariate matrix)
hK = random.rand(n, p)               # decomposition of kinship matrix (K = hK @ hK.T)
G = 1.0 * (random.rand(n, 5) < 0.2)  # SNP vector

delta = 0.02

class TestRunners(unittest.TestCase):

    def test_interaction_runner(self):
        #### old approach
        [U, S, _] = economic_svd(C)                          # get eigendecomposition of C
        us = U * S                                           # and derive that of CCt
        Ls = [ddot(us[:,i], hK) for i in range(us.shape[1])] # get decomposition of K \odot CCt
        crm = CellRegMap(y, W, C, Ls)                        # fit null model (Ls -> background is K*EEt + EEt)
        pv_old = crm.scan_interaction(g)[0]                  # test
        #### new approach
        pv_new = run_interaction(y, W, C, g, hK=hK)[0]
        self.assertEqual(pv_old, pv_new)

    def test_association_runner(self):
        #### old approach
        crm = CellRegMap(y, W, C, hK=hK) # fit null model (hK -> background is K + EEt)
        pv_old = crm.scan_association(g)[0]
        #### new approach
        pv_new = run_association(y, W, C, g, hK=hK)[0]
        self.assertEqual(pv_old, pv_new)

    def test_fast_association_runner(self):
        #### old approach
        crm = CellRegMap(y, W, C, hK=hK) # fit null model (hK -> background is K + EEt)
        pv_old = crm.scan_association_fast(g)[0]
        #### new approach
        pv_new = run_association_fast(y, W, C, g, hK=hK)[0] # faster but less accurate method
        self.assertEqual(pv_old, pv_new)

    def test_slow_fast_association_runner(self, delta=delta):
        #### regular approach
        pv_slow = run_association(y, W, C, g, hK=hK)[0] 
        #### fast approach
        pv_fast = run_association_fast(y, W, C, g, hK=hK)[0] # faster but less accurate 
        message = "first and second are not almost equal."
        self.assertAlmostEqual(pv_slow, pv_fast, None, message, delta) # allow tolerance

    def test_gene_set_association_runner(self):
        #### old approach
        crm = CellRegMap(y, W, C, hK=hK) # fit null model (hK -> background is K + EEt)
        pv_old = crm.scan_gene_set_association(G)[0]
        #### new approach
        pv_new = run_gene_set_association(y, W, C, G, hK=hK)[0]
        self.assertEqual(pv_old, pv_new)


