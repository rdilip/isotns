""" This module provides methods to split and disentangle a tensor, which is
necessary for the Moses move. In particular, the essential idea behind the
disentangler is the same regardless of the tensor shape. The primary function
one should import from this module is split; it calls a different variation of
the same function depending on the shape of the input tensor T.

Lovely updated version as of 2019/10/11: Let's do what the DMRG toycodes did
and have the PEPs be an array of five index tensors -- we just need to be 
careful about the loops, but don't need to worry about figuring out what kind
of tensors are on the boundary. Then we just deal with rank 4+1 tensors, which
I denote by
     4    
     |    
 0---T---3
    /|    
   2 1    

I need to deal with the pentagonal ones as well, but we shouldn't have to worry
about boundary conditions as much. I think.
"""

import numpy as np
import scipy.linalg as la
from scipy.stats import special_ortho_group
from misc import *
import warnings
from entropy import S2_disentangler

def unzip_quad(T):
    """ This is really the only unzipper you need, because you don't want to have
    to deal with finite boundary conditions. 

         4            2              3          2       
         |            |             /            \                     
     0---T---3        A        0---B---4      0---C---1                    
        /|           / \          /|                 
       2 1          1   0        2 1                                               
    """

    T_, pipeT = group_legs(T, [[0,1,2],[4,3]])
    B, R = la.qr(T_)
    dim_up, dim_down = get_closest_factors(B.shape[1])
    B = B.reshape(*pipeT[1][0], dim_up, dim_down)

    R = R.reshape(dim_up, dim_down, *pipeT[1][1])
    R_, pipeR = group_legs(R, [[1,3],[0,2]]) # Might be a bug cause
    C, A = la.qr(R_)
    C = C.reshape(*pipeR[1][0], C.shape[1])
    A = A.reshape(A.shape[0], *pipeR[1][1])
    return([A,B,C])

def unzip_pent(T):
    """ Note: The pentagonal tensors are temporary anyway...so I don't think
    there's anything wrong with having a slightly odd index convention 

         5              2             3       3            
         |              |            /         \         
     0---T---4          A       0---B---4   0---C---2
        /|\            / \         /|            \       
      2p 1 3          1   0      2p 1             1     


    """
    T_quad, pipeT = group_legs(T, [[0],[1],[2],[3,4],[5]])
    perm, shape = pipeT
    A,B,C = unzip_quad(T_quad)
    # Only one that needs to be reshaped is C
    T_ = contract_quad(A,B,C)
    T_ = T_.reshape(*T_.shape[:3], *shape[3], T_.shape[4])
    C = C.reshape(C.shape[0], *shape[3], C.shape[2])
    return(A,B,C)

def split(T):
    """ Splits a tensor T, performs disentangling using second Renyi entropy.
    The variable ttype indicates the number of virtual legs of the tensor
    being split.
    """
    ttype = None
    if T.ndim == 3 + 1:
        [A,B,C] = unzip_tri(T)
        ttype = 3
    elif T.ndim == 5 + 1:
        T, pipeT = group_legs(T, [[0,1],[2],[3],[4,5]])
        [A,B,C] = unzip_tri(T)
        ttype = 5
    elif T.ndim == 2 + 1:
        # Adding a trivial leg
        [A,B,C] = unzip_tri(T.reshape((1, *T.shape)))
        ttype = 2
    elif T.ndim == 4 + 1:
        T, pipeT = group_legs(T, [[0,1,2],[3,4]])
        Q, R = la.qr(T)
        Q = Q.reshape((*pipeT[1][0], Q.shape[1]))
        R = R.reshape(R.shape[0], *pipeT[1][1])
        return([Q,R])
    else:
        raise ValueError("T has an invalid numebr of legs.")
    theta = np.tensordot(A, C, [2,1])
    d1, d2 = theta.shape[1], theta.shape[2]

    # Disentangling
    Ui = special_ortho_group.rvs(d1*d2).reshape([d1,d2,d1,d2])
    Un = S2_disentangler(theta, Ui, num_iter=1000)
    theta = np.tensordot(theta, Un, [[1,2],[0,1]]).transpose([0,2,3,1])

    theta, pipetheta = group_legs(theta, [[2,3],[1,0]])
    C, A = la.qr(theta)
    C = C.reshape((*pipetheta[1][0], C.shape[1])).transpose([0,2,1])
    A = A.reshape(A.shape[0], *pipetheta[1][1]).transpose([2,1,0])

    B_outer_legs = [2,3]
    B = np.tensordot(B, Un, [B_outer_legs, [0,1]])

    if ttype == 5:
#        print(B.shape, T.shape)
#        print(pipeT)
        B = B.reshape(*pipeT[1][0], *B.shape[1:])
        C = C.reshape(*C.shape[:2], *pipeT[1][3])
    if ttype == 2:
        B = B.reshape(B.shape[1:])
    return(A,B,C)

# Debugging
def _contract_pent(A,B,C):
    T = np.tensordot(B, C, [4,0])
    T = np.tensordot(T, A, [[3,6],[1,0]])
    return(T)

def _contract_quad(A,B,C):
    """ Debugger, contracts A B C to get T """
    T = np.tensordot(B, C, [4,0])
    T = np.tensordot(T, A, [[3,5],[1,0]])
    return(T)

