import numpy as np
from misc import *
""" Base class for isometric tensor network states. The primary method is 
full_sweep(), which performs a series of sweeps and rotations around the peps. 
Each sweep corresponds to a series of TEBD and moses_moves across the peps in 
it's current orientation. For more details, see 
https://arxiv.org/pdf/1902.05100.pdf. 

The PEPS index convention is
         4                      
         |                         
     1---T---2
        /|                          
       0 3                         

The PEPS is stored as a list of lists of these tensors. PEPS[x] should
give a column.
"""

class isotns:
    """ Base class for isotns.

    Parameters
    ----------
    peps: List of lists of numpy arrays, representing a peps.

    tebd_params: Dictionary of tebd params. Format

                {trunc_params: {chi_max = chi_max, p_trunc = p_trunc,
                                moses_trunc_params: {chiV_max, chiH_max,
                                                     etaV_max, etaH_max}
                                }
                }
    
    Should contain subdictionary 
    trunc_params (which must have p_trunc and chi_max, and can optionally 
    contain moses_move truncation_params)

    Attributes
    ----------
    peps: List of lists of tensors making up the peps.

    Lx: Length in x direction.

    Ly: Length in y direction.

    Ss: List of lists of dimension (Lx - 1, Ly - 1). Each element is the 
    entanglement entropy across a bond (I don't know if this actually is valid).
    """
    def __init__(self, peps, tebd_params):
        self.peps = peps
        self.Lx = len(peps)
        self.Ly = len(peps[0])
        self.tp = tebd_params

    def peps_sweep(self, U, O = None):
        trunc_params = self.tp["trunc_params"]
        Psi = self.peps[0]
        Lx, Ly = self.Lx, self.Ly

        info = dict(expectation_O = [],
                    tebd_err = [],
                    moses_err = [])

        for j in range(Lx):
            Psi, exp_vals, tebd_err = tebd(Psi, U, O, trunc_params, direct = "L")
            info["expectation_O"].append(exp_vals)
            info["tebd_err"].append(tebd_err)
            if j < Lx - 1:
                Psi, pipe = mps_group_legs(Psi, [[0,1],[2]])
                A, Lambda, moses_err = moses_move(Psi, trunc_params)
                A = mps_ungroup_legs(A, pipe)

                self.peps[j] = A
                Psi, pipe = mps_group_legs(peps[j + 1], axes=[[1],[0,2]])
                Psi = contract_mpos(Lambda, Psi)
                Psi = mps_ungroup_legs(Psi, pipe)
                self.peps[j + 1] = Psi
            else:
                Psi = canonical_form(Psi, form='A')
                self.peps[j] = Psi
        return(info)

