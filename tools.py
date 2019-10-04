# coding: utf-8
""" Functions mostly for debugging purposes """
from scipy.stats import ortho_group
import numpy as np

def get_sv_matrix(size, num_SV):
    """ Returns a matrix of size `size` with `num_SV` singular values. """
    assert num_SV <= min(size)
    U = ortho_group.rvs(size[0])
    V = ortho_group.rvs(size[1])
    S = np.zeros(size)
    row, col = np.diag_indices(num_SV)
    S[row, col] = np.random.rand(num_SV)
    return(U @ S @ V)
