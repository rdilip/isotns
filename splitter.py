""" This module provides methods to split and disentangle a tensor, which is
necessary for the Moses move. In particular, the essential idea behind the
disentangler is the same regardless of the tensor shape. The primary function
one should import from this module is split; it calls a different variation of
the same function depending on the shape of the input tensor T.

I use the following index conventions.

Duo (Two virtual legs, one physical leg):

        1             0          1           1        
        |             |          |            \      
        |             |          |             \     
        T---2         A          B----2     0---C---2
       /             / \        /                    
      /             /   \      /                     
    0(p)           1     2    0(p)                   

Triangular (three virtual legs, one physical leg):

        2             0           2           1          
        |             |           |            \         
        |             |           |             \        
   0----T----3        A      0----B----3     0---C---2   
       /             / \         /                       
      /             /   \       /                        
     1(p)          1     2     1(p)                      

Pentagonal (five virtual legs, one physical leg):

         3              0              3     1            
         |              |             /       \           
         |              |            /         \         
     0---T---4          A       0---B---4   0---C---2
        /|\            / \         /|           |             
       / | \          /   \       / |           |       
     2p  1  5        1     2     2p 1           3       
 
"""

import numpy as np
import scipy.linalg as la
from scipy.stats import special_ortho_group
from misc import *
import warnings
from entropy import S2_disentangler

def unzip_tri(T):
    """ This function accepts a rank 4 tensor with three virtual legs and one
    physical leg, and "unzips" it into a tripartite state. One can then find
    a disentangler to minimize the entropy across these states (the moses move).

        0           2           1                 2      
        |           |            \                |      
        |           |             \               |      
        A      0----B----3     0---C---2     0----T----3
       / \         /                             /       
      /   \       /                             /        
     1     2     1(p)                          1(p)      
                                  
                                  
    Parameters
    ----------
    T: A rank 4 tensor. Indices should be as outlined above.

    Returns
    ----------
    ABC: A list of tensors A, B, and C. 
    """
    T, pipeT = group_legs(T, [[0,1],[2,3]])
    # Split T into two tensors with SVD. Store the left tensor as 
    # B, after splitting the leg in two (flooring the square root)
    X, Y, Z = la.svd(T, full_matrices = False)
    keep_dim_0, keep_dim_1 = get_closest_factors(len(Y))
    B = X.reshape((X.shape[0], keep_dim_0 * keep_dim_1))

    pipeB = change_pipe_shape(pipeT, 1, (keep_dim_0, keep_dim_1))
    B = ungroup_legs(B, pipeB)
    # Now split the right most tensor vertically
    right_shape = pipeT[1][1]
    R = np.reshape(np.diag(Y) @ Z, (keep_dim_0, keep_dim_1,
                  right_shape[0], right_shape[1]))

    R, pipeR = group_legs(R, [[1,3],[0,2]]) 
    C, A = la.qr(R)
    R = ungroup_legs(C@A, pipeR)

    C = C.reshape(keep_dim_1, right_shape[1], C.shape[1]).transpose([0,2,1])
    A = A.reshape(A.shape[0], keep_dim_0, right_shape[0]).transpose([2,1,0])
    return([A,B,C])

def unzip_pent(T):
    """ Unzips a rank-6 tensor (counting one physical leg) by grouping  legs
    and calling unzip_tri()


         3              0              3     1            
         |              |             /       \           
         |              |            /         \         
     0---T---4          A       0---B---4   0---C---2
        /|\            / \         /|           |       
       / | \          /   \       / |           |       
     2p  1  5        1     2     2p 1           3       

    """
    warnings.warn("This function is out of date. Are you sure you want to"\
            + " use it?", DeprecationWarning, stacklevel=2)
    T_tri, pipeT = group_legs(T, [[0,1],[2],[3],[4,5]])
    perm, shape = pipeT
    A, B, C = unzip_tri(T_tri)
    # No reshaping for A
    B = B.reshape((*shape[0], *B.shape[1:])).transpose([0, 2, 3, 4, 1])
    C = C.reshape((*C.shape[:2], *shape[3]))
    return([A,B,C])

def contract_pent_split(A,B,C):
    """ Congracts A,B,C where the output tensor T is pentagonal """
    T = np.tensordot(A, C, [2,1])
    T = np.tensordot(B,T,[[3,4],[1,2]])
    return(T)

def contract_tri_split(A,B,C):
    """ Contracts tensors A, B, and C. Useful for debugging """
    tmp = np.tensordot(C, A, [1,2]).transpose([0,2,1,3])
    return(np.tensordot(B, tmp, [[2,3],[3,0]]))

def contract_duo_split(A,B,C):
    """ Contracts tensors A,B,C where the result has two virtual
    indices """
    T = np.tensordot(A, C, [2,1])
    T = np.tensordot(B, T, [[1,2],[1,2]])
    return(T)

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




