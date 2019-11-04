# coding: utf-8
""" Functions mostly for debugging purposes """
from scipy.stats import ortho_group
import time
import numpy as np
import pickle

def get_sv_matrix(size, num_SV):
    """ Returns a matrix of size `size` with `num_SV` singular values. """
    assert num_SV <= min(size)
    U = ortho_group.rvs(size[0])
    V = ortho_group.rvs(size[1])
    S = np.zeros(size)
    row, col = np.diag_indices(num_SV)
    S[row, col] = np.random.rand(num_SV)
    return(U @ S @ V)

def breakpoint(start):
    count = 0
    while True:
        print("\t\t\tBP {0} at time T = {1}".format(count, time.time() - start))
        count += 1
        yield

def check_bond_dim(T, chi = 32):
    for i in T.shape:
        if i > chi:
            return True
    return(False)

def save_tmpfile(obj):
    with open("tmpfile.pkl", "wb+") as f:
        pickle.dump(obj, f)

def load_tmpfile():
    with open("tmpfile.pkl", "rb") as f:
        return(pickle.load(f))

def contract_tri(a1, s1, b1):
    """ Contracts the result of split_psi """
    return(np.tensordot(np.tensordot(a1, b1, [2,0]), s1, [[1,2],[0, 2]]).transpose([0,2,1]))

def compare(T1, T2):
    """ Compares two tensors """
    return(np.all(np.abs(T1 - T2) < 1.e-10))
