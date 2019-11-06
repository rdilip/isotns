""" Module implementing the Moses Move, as described in https://arxiv.org/pdf/1902.05100.pdf

Following Mike's implementation, I have the following conventions for bond
dimensions.
                            
       4                                  |                                           
       |      (group and                  S                                                 
    2--T--3    split_psi())          etaV/ \                                         
      / \                          chiV /   \                                  
     0   1                           ==A --- B==                             
                                      chiH etaH                                                                                                         
     Pent                                                                       

At the end, the leg ordering is

    0          3                                     
    |          |                                     
 3--B--1    0--A--1                                     
    |          |                                 
    2          2                                                                

The internal bond dimensions in the splitting are specified as a subdictionary
under trunc_params as split_trunc_params. Each of eta and chi can be overridden
with eta_max and chi_max.

The primary function moses_move starts from the bottom of a column in B form,
and works its way up, shifting the orthogonality center as it goes. 
"""

from misc import *
import numpy as np
from new_splitter import split_psi
from renyin_splitter import split_psi as mz_split_psi

from debugging_tools import *

def moses_move(Psi, trunc_params,  trunc_flag = False):
    """ Performs a moses move on Psi, starting from the bottom (i.e., the first
    element of Psi and working up. You should view Psi as an MPO before 
    doing this.

    Parameters
    ----------
    Psi: One column wavefunction in B-form.  
    trunc_params: Dictionary of trunc_params. In addition to p_trunc and chi_max,
    also accepts eta_max (for the A tensor) and a dictionary moses_trunc_params,
    which can specifiy etaH_max, chiH_max, etaV_max, chiV_max as shown in the 
    above picture. 

    Returns
    ----------
    A: One column wavefunction in A form

    Lambda: Adjacent zero column wavefunction.
    """
    moses_trunc_params = trunc_params.get("moses_trunc_params", {})
    if 'chi_max' in trunc_params:
        chiH_max = chiV_max = trunc_params["chi_max"]
    else:
        chiH_max = moses_trunc_params.get("chiH_max", 32)
        chiV_max = moses_trunc_params.get("chiV_max", 32)
    if 'eta_max' in trunc_params:
        etaH_max = etaV_max = trunc_params["eta_max"]
    else:
        etaH_max = moses_trunc_params.get("etaH_max", 32)
        etaV_max = moses_trunc_params.get("etaV_max", 32)

    L = len(Psi)
    Lambda = []
    A = []

    pL, pR, _, chi = Psi[0].shape
    eta = 1
    chiV = 1
    errors, d_errors = [], []
    # Reshape bottom most tensor as a pentagonal tensor
    pent = Psi[0].reshape([chiV, eta, pL, pR, chi])



    for j in range(L):
        tri, pipe_tri = group_legs(pent, [[0,2],[4],[1,3]])



        dL = chiV_max
        dR = chiH_max

        if j == L - 1:
            dL = 1
            dR = np.min([etaH_max, tri.shape[0]])
        # TODO

        a, S, B, info = split_psi(tri,
                            (dL, dR),
                            dict(chi_max = etaV_max, 
                            p_trunc = trunc_params["p_trunc"]),
                            flag =False)
        

        errors.append(info.setdefault("error", np.nan))
        d_errors.append(info.setdefault("d_error"))
        dL, dR = a.shape[1:]
        B = B.reshape(dR, B.shape[1], eta, pR).transpose([0,3,2,1])
        a = a.reshape(chiV, pL, dL, dR).transpose([1,3,0,2]) 
        Lambda.append(B)
        A.append(a)

        if j < L - 1:
            pL, pR, _, chi = Psi[j+ 1].shape
            pent = np.tensordot(S, Psi[j + 1], [1,2])
        else:
            Lambda[j] = Lambda[j] * S
        info = dict(errors = errors,
                    error = np.sum(errors),
                    d_errors = d_errors,
                    d_error = np.sum(d_errors))
        chiV = dL
        eta = B.shape[-1]

    return(A, Lambda, info)
 
