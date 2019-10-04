""" This file contains functions to find the U minimizing the 2nd Renyi entropy.
"""

import numpy as np
from misc import *
import scipy.linalg as la

def get_E2(theta, U):
    Utheta = np.tensordot(theta, U, [[1,2],[0,1]]).transpose([0,2,3,1])
    E2 = np.tensordot(Utheta, Utheta, [[2,3],[2,3]])
    E2 = np.tensordot(E2, Utheta, [[2,3],[0,1]])
    E2 = np.tensordot(E2, theta, [[0,3],[0,3]])
    return(E2)

def S2_disentangler(theta, Ui, num_iter):
    """ Finds the optimal U given an initial environment tensor excluding the 
    Us (this is the output of get_psi_env) """
    Ss = [] 
    for i in range(num_iter):
        E2 = get_E2(theta, Ui)
        E2, pipeE = group_legs(E2, [[0,1],[2,3]])
        W, S, V = la.svd(E2)
        U_next = ungroup_legs(W @ V, pipeE)
        E2 = ungroup_legs(E2, pipeE)
        Ui = U_next
    return(U_next)

