import numpy as np
from misc import *
from tebd import tebd, get_time_evol
from moses_move import *
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

This code is written to exactly match Mike and Frank's... keep in mind
that np vs sp do introduce numerical errors.
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

    def __init__(self, peps, trunc_params):
        self.peps = peps
        self.Lx = len(peps)
        self.Ly = len(peps[0])
        self.tp = trunc_params

    def rotate(self):
        """ Rotates a peps counter clockwise by 90 degrees. """
        peps = self.peps
        Lx, Ly = self.Lx, self.Ly
        rpeps = [[None] * Lx for i in range(Ly)] # Rotated dimensions
        for x in range(Lx):
            for y in range(Ly):
                #rpeps[y][Lx - 1 - x] = rotate_CC(peps[x][y]).copy()
                # TODO figure out why this is right...
                rpeps[y][x] = rotate_CC(peps[x][Ly - 1 - y]).copy()
        self.peps = rpeps

    def sweep_with_rotation(self, Us, trunc_params, Os = None):
        """ Performs a series of 4 peps_sweep()s. Between each sweep, rotates the
        peps by 90 degrees CC. In this way, we perform TEBD on all rows and columns.
       
        Parameters
        ----------
        peps: Ly x Lx list of lists of tensors.

        Us: List of 4 lists, one for each sweep direction. 

        trunc_params: truncation_params

        Os: List of 4 lists, one for each sweep direction.

        Returns
        ----------
        peps: Time evolved peps

        info: Dictionary of information about the peps.
        """

        info = dict(expectation_O = [],
                    tebd_err = [0.0] * 4,
                    moses_err = [0.0] * 4
                    )
        if Us is None:
            Us = [None] * 4
        if Os is None:
            Os = [None] * 4
        for i in range(4):
            print("Starting sequence {i} of full sweep".format(i=i))
            info_ = self._full_sweep(Us[i], Os[i])
            info["expectation_O"].append(info_["expectation_O"])
            info["tebd_err"][i] += np.sum(info_["tebd_err"])
            info["moses_err"][i] += np.sum(info_["tebd_err"])

            self.rotate()
        return(info)

    def tebd2(self, Hs, dt, trunc_params, Nsteps = None, min_dE = None):
        """ Performs tebd2 ith second order Trotterization.
        Parameters
        ----------
        Hs: List of vertical and horizontal Hamiltonians (NOT time 
        evolution operator)

        dt: Time interval. 

        trunc_params: truncation param dict

        Nsteps: Number of full sweeps with rotations across peps (i.e.,
        four sweeps back and forth).

        min_dE: Break after the change in energy between sweeps is less than min_dE.

        Returns
        ---------
        info: Dict on final run
        """

        if min_dE is None:
            min_dE = np.float("inf")
        if Nsteps is None:
            Nsteps = np.float("inf")

        Uv = get_time_evol(Hs[0], dt)
        Uh = get_time_evol(Hs[1], dt)
        Uv2 = get_time_evol(Hs[0], dt / 2.) # Second order Trotter

        info = self.sweep_with_rotation([Uv2, Uh, Uv, Uh], trunc_params,
                                        Os=[None, None, None, Hs[1]])
        E_curr = np.sum(info["expectation_O"][3])
        step = 0
        dE = np.float("inf")

        while step < Nsteps and np.abs(dE) < min_dE:
            if step % 10 == 0:
                print("Step {0}".format(step))
            info = self.sweep_with_rotation([Uv, Uh, Uv, Uh], trunc_params,
                                            Os = [None, None, None, Hs[1]])
            E_prev = E_curr
            E_curr = np.sum(info["expectation_O"][3])
            dE = E_curr - E_prev
            step += 1

        info = self.sweep_with_rotation([Uv2, None, None, None],
                                        trunc_params,
                                       Os=[None, None, Hs[0], Hs[1]])
        return(info)

    def _full_sweep(self, U, O = None):
        Psi = self.peps[0]
        Lx = self.Lx
        Ly = self.Ly
        trunc_params = self.tp

        tebd_2_mm_trunc = 0.1 # IDK what this is...
        min_p_trunc = self.tp["p_trunc"]
        # This is adjusted according to the errors. Not sure why exactly
        target_p_trunc = min_p_trunc 

        info = dict(expectation_O = [],
                    tebd_err = [],
                    moses_err = [],
                    moses_d_err = [],
                    nrm = 1.)
        if U is None:
            U = [None]
        if O is None:
            O = [None]
        for j in range(Lx):
            tebd_trunc_params = dict(p_trunc = target_p_trunc,
                                     chi_max = trunc_params["chi_max"])
            Psi, tebd_info = tebd(Psi, U,
                                       O,
                                       tebd_trunc_params, direct = "L")

            info["expectation_O"].append(tebd_info['expectation_O'])
            info["tebd_err"].append(tebd_info['tebd_err'])
            info["nrm"] *= tebd_info["nrm"]

            if j < Lx - 1:
                Psi, pipe = mps_group_legs(Psi, [[0,1],[2]])
                A, Lambda, moses_info = moses_move(Psi, trunc_params)
                info["moses_err"].append(moses_info.setdefault("error", np.nan))
                info["moses_d_err"].append(moses_info.setdefault("d_error", np.nan))

                if not np.isnan(info["moses_err"][-1]):
                    target_p_trunc = np.max([tebd_2_mm_trunc * info['moses_err'][-1] /\
                                            len(Psi), min_p_trunc])

                A = mps_ungroup_legs(A, pipe)
                self.peps[j] = A
                Psi, pipe = mps_group_legs(self.peps[j + 1], axes=[[1],[0,2]])
                self.peps[j + 1] = Psi
              
                Psi = contract_mpos(Lambda, Psi)

                Psi = mps_ungroup_legs(Psi, pipe)
                
            else:
                Psi = canonical_form(Psi, form='A')
                self.peps[j] = Psi
        return(info)

# Static methods
def rotate_CC(T):
    """ Rotates a tensor T counter clockwise """
    if T.ndim != 5:
        raise ValueError("T should have 5 legs")
    return(T.transpose([0,4,3,1,2]))

