import numpy as np
import scipy.linalg as la
from misc import *
from misc import svd_trunc

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
    sx = np.array([[0,1],[1,0]])
    sz = np.array([[1.0, 0.0],[0.0, -1.0]])
    id = np.eye(2)
    
    ops = []
    
    for site in range(num_bonds):
        gL = gR = 0.5 * self.g
        if site == 0: gL = self.g
        if site == L - 2: gR = self.g
        H_local = -self.J * np.kron(sx, sx) - gL * np.kron(sz, idn) - gR * np.kron(idn, sz)
        ops.append(np.reshape(H_local, [d] * 4))
    return(ops)

def sweep(Psi, Us = None, trunc_params = None, reduced_update = True,
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
        Psi, exp_vals = _tebd_sweep(Psi, Us, None, trunc_params, reduced_update)
    else:
        Psi = mps_invert(Psi)
        if Us is not None:
            Us = operator_invert(Us)
        Psi, exp_vals = _tebd_sweep(Psi, Us, None, trunc_params, reduced_update)
        Psi = mps_invert(Psi)
    return(Psi, exp_vals)

def get_expectation_value(Psi_, Os, trunc_params = None):
    """ Gets the expectation value of a list of local operators Os. Psi must be
    in B form. """
    Psi = Psi_.copy()
    Psi, expvals = _tebd_sweep(Psi, U = None, Os = Os, trunc_params = trunc_params)
    return(expvals)

def _tebd_sweep(Psi, U, Os, trunc_params, reduced_update = True):
    """ Main work function for sweep(). Performs a sweep from left to right
    on an MPS Psi. """

    L = len(Psi)

    exp_vals = [0 for i in range(L-1)]
    psi = Psi[0] # orthogonality center

    # Number of physical legs -- view the wavefunction as an MPO
    num_p = psi.ndim - 2

    for oc in range(L - 1):
        # Going to MPS form (3 leg). Index notation copied.
        psi, pipeL1 = group_legs(psi, [[0], list(range(1, num_p + 1)),\
                                [num_p + 1]])

        B, pipeR1 = group_legs(Psi[oc + 1], [[0], [num_p], list(range(1, num_p))\
                                + list(range(num_p + 1, Psi[oc].ndim))])

        # Left update
        reduced_L, reduced_R = False, False
        if reduced_update and psi.shape[0] * psi.shape[2] < psi.shape[1]:
            reduced_L = True
            psi, pipeL2 = group_legs(psi, [[0,2], [1]])
            QL, RL = la.qr(psi)
            psi = ungroup_legs(RL, pipeL2)

        if reduced_update and B.shape[2] > B.shape[0] * B.shape[1]:
            reduced_R = True
            B, pipeR2 = group_legs(B, [[0, 1], [2]])
            QR, RR = la.qr(B.T)
            B = ungroup_legs(RR.T, pipeR2)
        theta = np.tensordot(psi, B, [2,1])
        if U is not None:
            theta = np.tensordot(U[oc], theta,  [[2,3],[0,2]])
        else:
            theta = theta.transpose([0, 2, 1, 3])

        if Os is not None:
            Otheta = np.tensordot(Os[oc], theta, [[2,3], [0,1]])
            Otheta = np.tensordot(Otheta, theta.conj(), [[0,1],[0,1]])
            exp_vals[oc] = np.trace(np.trace(Otheta, axis1=0, axis2=2))
        
        # Grouping one physical and one virtual
        theta, pipe_theta = group_legs(theta, [[0,2],[1,3]])

        A, S, B, nrm_t = svd_trunc(theta, trunc_params)
        SB = ((B.T) * S / nrm_t).T
        # Because this is an SVD and might involve truncation, we can't just
        # ungroup legs.
        A = A.reshape(pipe_theta[1][0] + (-1,))
        SB = SB.reshape((-1,) + pipe_theta[1][1]).transpose([1,0,2])
        if reduced_L:
            A = np.tensordot(A, QL, [1,1]).transpose([1, 0, 2])
        if reduced_R:
            SB = np.tensordot(SB, QR, [2,0])

        A = ungroup_legs(A, pipeL1)
        SB = ungroup_legs(SB, pipeR1)


        psi = SB
        Psi[oc] = A

    Psi[L - 1] = SB
    return(Psi, exp_vals)

        
