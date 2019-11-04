import numpy as np
import time
#import scipy.linalg as la
import numpy.linalg as la
from misc import *
import sys
from misc import svd_theta
from moses_move import moses_move
#from moses_simple import moses_move as moses_move_simple
from pprint import pprint
from debugging_tools import breakpoint, check_bond_dim

""" Implements functions to perform a tebd sweep across a column of a 
PEPS. The PEPS index convention is
         4                      
         |                         
     1---T---2
        /|                          
       0 3                         

The PEPS is stored as a list of lists of these tensors. PEPS[x] should
give a column -- this is not what you see if you print it. Doing this
because it matches Mike's index conventions.
"""

def get_time_evol(H_bonds, dt):
    """ Accepts H_bonds, a list of local operators. Returns operator list for
    imaginary time evolution to find ground state """
    Us = []
    d = H_bonds[0].shape[0] # Local Hilbert space dimension
    for H in H_bonds:
        H = H.reshape([d*d, d*d])
        U = la.expm(-dt * H).reshape([d] * 4)
        Us.append(U)
    return(Us)

def get_TFI_bonds(L, J = 1.0, g = 1.0):
    """ Returns TFI Hamiltonian as a list of local bonds. Default bc is finite. """
    num_bonds = L - 1
    d = 2
    sx = np.array([[0., 1.], [1., 0.]])
    sz = np.array([[1., 0.],[0., -1.]])
    id = np.eye(2)
    
    ops = []
    
    for site in range(num_bonds):
        gL = gR = 0.5 * g
        if site == 0: gL = g
        if site == L - 2: gR = g
        H_local = -J * np.kron(sz, sz) - gL * np.kron(sx, id) - gR * np.kron(id, sx)
        ops.append(np.reshape(H_local, [d] * 4))
    return(ops)

def tebd(Psi, Us = None, Os = None, trunc_params = None, reduced_update = True,
        direct = 'R'):
    """ Performs TEBD across an MPS Psi (which is a column in the PEPS).
    The MPS should start in the B form and will finish in the A form. 

    Parameters
    ---------
    Psi: One column wavefunction (MPS). Can have multiple physical legs.

    U: List of local unitaries to apply. The local unitary U is assumed
    to act only on the first physical leg.

    Os: List of local two-site operators. If Os is not None, then this function
    returns the expectation value of these operators. 

    trunc_params: Dictionary of truncation parameters for SVD.

    reduced_update: If True, takes a QR decomposition to make the unitary
    application less complex. Follows Figure 4 in 
    https://arxiv.org/pdf/1503.05345v2.pdf

    direct: The direction in which to sweep. 

    Returns
    ----------
    None
    """
    if direct not in ['R', 'L']:
        raise ValueError("Direction must be either 'R' or 'L'")
    if trunc_params is None:
        trunc_params = {}
    if direct == 'R':
        Psi, exp_vals, tebd_err = _tebd_sweep(Psi, Us, Os, trunc_params, reduced_update)
    else:
        Psi = mps_invert(Psi)
        if Us is not None:
            Us = operator_invert(Us)
        if Os is not None:
            Os = operator_invert(Os)
        Psi, exp_vals, tebd_err = _tebd_sweep(Psi, Us, Os, trunc_params, reduced_update)
        exp_vals = exp_vals[::-1]
        Psi = mps_invert(Psi)
    return(Psi, exp_vals, tebd_err)

def _tebd_sweep(Psi, U, O, trunc_params, reduced_update = True):
    """ Main work function for sweep(). Performs a sweep from left to right
    on an MPS Psi. """

    L = len(Psi)

    exp_vals = [0 for i in range(L-1)]
    psi = Psi[0] # orthogonality center

    # Number of physical legs -- view the wavefunction as an MPO
    num_p = psi.ndim - 2
    tebd_err = 0.0
    for oc in range(L - 1):
        bp = breakpoint(time.time())
        print("\t\tStarting TEBD on site {0}".format(oc))
        # Going to MPS form (3 leg). Index notation copied.
        psi, pipeL1 = group_legs(psi, [[0], list(range(1, num_p + 1)),\
                                [num_p + 1]])
        B, pipeR1 = group_legs(Psi[oc + 1], [[0], [num_p], list(range(1, num_p))\
                                + list(range(num_p + 1, Psi[oc].ndim))])
        # Left update
        reduced_L, reduced_R = False, False
        if reduced_update and psi.shape[0] * psi.shape[2] < psi.shape[1]:
            reduced_L = True
            psi, pipeL2 = group_legs(psi, [[1], [0,2]])
            QL, RL = np.linalg.qr(psi)
            psi = ungroup_legs(RL, pipeL2)
        if reduced_update and B.shape[2] > B.shape[0] * B.shape[1]:
            reduced_R = True
            B, pipeR2 = group_legs(B, [[0, 1], [2]])
            QR, B = np.linalg.qr(B.T)
            QR = QR.T
            B = B.T
            B = ungroup_legs(B, pipeR2)
        theta = np.tensordot(psi, B, [2,1])
        #theta = np.tensordot(psi, B, axes = [[-1], [-2]])
        if U is not None:
            theta = np.tensordot(U[oc], theta,  [[2,3],[0,2]])
        else:
            theta = theta.transpose([0, 2, 1, 3])
        if O is not None:
            Otheta = np.tensordot(O[oc], theta, [[2,3], [0,1]])
            exp_vals[oc] = np.tensordot(theta.conj(), Otheta, axes=[[0,1,2,3],[0,1,2,3]])
            # This is weirdly slow
            #Otheta = np.tensordot(Otheta, theta.conj(), [[0,1],[0,1]])
            #exp_vals[oc] = np.trace(np.trace(Otheta, axis1=0, axis2=2))
        
        # Grouping one physical and one virtual
        theta, pipe_theta = group_legs(theta, [[0,2],[1,3]])
        # This is taking an oddly long amount of time
       # A, S, B, info = svd_trunc(theta, trunc_params)
       # nrm_t = info["nrm_t"]
       # tebd_err += info["p_trunc"]
       # SB = ((B.T) * S / nrm_t).T
        A, SB, info = svd_theta(theta, trunc_params)
        # Because this is an SVD and might involve truncation, we can't just
        # ungroup legs.
        A = A.reshape(pipe_theta[1][0] + (-1,))
        SB = SB.reshape((-1,) + pipe_theta[1][1]).transpose([1,0,2])
        if reduced_L:
            A = np.tensordot(QL, A, [1,1]).transpose([1, 0, 2])
        if reduced_R:
            SB = np.tensordot(SB, QR, [2,0])
        A = ungroup_legs(A, pipeL1)
        SB = ungroup_legs(SB, pipeR1)


        psi = SB
        Psi[oc] = A

    Psi[L - 1] = SB
    return(Psi, exp_vals, tebd_err)

def peps_sweep(peps, U, trunc_params, O = None, flag = False):
    Psi = peps[0]
    Lx = len(peps)
    Ly = len(Psi)
    """
    if U is None:
        U = [None]
    if O is None:
        O = [None]
    """

    info = dict(expectation_O = [],
                tebd_err = [],
                moses_err = [])

    for j in range(Lx):
        print(j)
        print("\tStarting TEBD on column {0}".format(j))
        start = time.time()
        Psi, exp_vals, tebd_err = tebd(Psi, U, O, trunc_params, direct = "L")
        print("\tTEBD done at time t={0}".format(time.time() - start))
        #info["expectation_O"].append(exp_vals)
        #info["tebd_err"].append(tebd_err)
        if j < Lx - 1:

            #if j == 0 and flag: return(Psi, info)
            Psi, pipe = mps_group_legs(Psi, [[0,1],[2]])
            print("\tStarting moses move on column {0}".format(j))
            start = time.time()
            A, Lambda, moses_err = moses_move(Psi, trunc_params)
            #print("moses simple")
            #A, Lambda, info = moses_move_simple(Psi, trunc_params)

            print("\tMoses done at time t = {0}".format(time.time() - start))
            #info["moses_err"].append(moses_err)

            #if j == 0 and flag: return(A, info)
            A = mps_ungroup_legs(A, pipe)

            peps[j] = A
            Psi, pipe = mps_group_legs(peps[j + 1], axes=[[1],[0,2]])
            Psi = contract_mpos(Lambda, Psi)
            Psi = mps_ungroup_legs(Psi, pipe)
            peps[j + 1] = Psi
        else:
            Psi = canonical_form(Psi, form='A')
            peps[j] = Psi
    return(peps, info)

def rotate_CC(T):
    """ Rotates a tensor T counter clockwise """
    if T.ndim != 5:
        raise ValueError("T should have 5 legs")
    return(T.transpose([0,4,3,1,2]))

def rotate_peps(peps):
    """ Rotates a peps counter clockwise by 90 degrees. """
    Lx, Ly = len(peps), len(peps[0])
    rpeps = [[None] * Lx for i in range(Ly)] # Rotated dimensions
    for x in range(Lx):
        for y in range(Ly):
            #rpeps[y][Lx - 1 - x] = rotate_CC(peps[x][y]).copy()
            # TODO figure out why this is right...
            rpeps[y][x] = rotate_CC(peps[x][Ly - 1 - y]).copy()
    return(rpeps)

def peps_sweep_with_rotation(peps, Us, trunc_params, Os = None):
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
        Us = [[None]] * 4
    if Os is None:
        Os = [[None]] * 4
    flag = False
    for i in range(4):
        if i == 1: flag = True
        print("Starting sequence {i} of full sweep".format(i=i))
        peps, info_ = peps_sweep(peps, Us[i], trunc_params, Os[i], flag = flag)
        #info["expectation_O"].append(info_["expectation_O"])
        #info["tebd_err"][i] += np.sum(info_["tebd_err"])
        #info["moses_err"][i] += np.sum(info_["tebd_err"])


        if i == 0: return(peps, info)
        peps = rotate_peps(peps)




    return(peps, info)

if __name__ == '__main__':
    print("test")
    # All truncation parameters in trunc_params. trunc_params has sub
    # dictionary moses_trunc_params specific to the moses move. 
    # trunc_params also has p_trunc and chi_max to determine how far to truncate.
    # on the normal SVD during the TEBD.
    #trunc_params = dict(chi_max
    # TODO make a standard trunc_params
