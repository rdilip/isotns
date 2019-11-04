""" Splitter function modeled after Mike's code to make it easier to test. """
import numpy as np
from misc import *
import scipy.linalg as la

def split_psi(psi, split_dim, trunc_params, disentangler_params = None, init_from_polar = True):
    """ Splits a tripartite state psi. Finds an approximation psi ~= A.lambda,
    where A is an isometry and lambda is a TEBD style wavefunction. 
    
    Parameters
    ----------
    psi: Rank 3 tensor

    split_dim: A tuple (dL, dR) that describes the dimensions of the first split

    trunc_params: Dictionary of trunc_params. Keys are chi_max
    See Figure 2b in https://arxiv.org/pdf/1902.05100.pdf

    disentangler_params: Optional parameters for disentangler. Valid parameters
    are given in disentangle()

    init_from_polar: Use a polar decomposition as the initial guess
    """
    
    dL, dR = split_dim 

    if disentangler_params is None:
        disentangler_params = {}

    d, mL, mR = psi.shape
    dL = np.min([dL, mL])
    dR = np.min([dR, mR])

    # Mike's version -- better than your factors version
    if dL * dR > d:
        dR = min([int(np.rint(np.sqrt(d))), dR])
        dL = min([d // dR, dL])
    
    # How much are we throwing away by not using get_closest_factors()?
    # dL, dR = get_closest_factors(d)
    tp = dict(p_trunc = 0.0, chi_max = dL * dR)

    X, y, Z, info = svd_trunc(psi.reshape((-1, mL * mR)), tp)
    #X, y, Z, D2, trunc_leg = svd_theta_UsV(psi.reshape(-1, mL * mR), dL * dR, 0.)

    h_err = info["p_trunc"]
    A = X
    theta = (Z.T * y).T
    D2 = len(y)
    init_from_polar = False
    # This next section involves truncating the legs of psi based on eigenvalues
    # of the reduced density matrix (Hermitian and positive so EVs are SVs)
    if init_from_polar:
        psi = theta.reshape([D2, mL, mR])
        if mL > dL:
            rho = np.tensordot(psi, psi.conj(), axes = [[0,2],[0,2]])
            e, u = la.eigh(rho)
            u = u[:, -dL:] # dL largest eigenvalues
            psi = np.tensordot(psi, u.conj(), [1,0]).transpose([0,2,1])

        if mR > dR:
            rho = np.tensordot(psi, psi.conj(), axes = [[0,1],[0,1]])
            e, u = la.eigh(rho)
            u = u[:, -dR:] 
            psi = np.tensordot(psi, u.conj(), [2,0])

        psi /= la.norm(psi)
        u, s, v = svd(psi.reshape(D2, D2))
        # I'm not sure we can do this.
        # psi has shape D2 x mL x mR, which we've truncated to D2 x dL x dR.
        # Then it seems like dL x dR should be D2, but D2 is the number of
        # singular values of psi. If psi does not have a weird rank, then 
        # D2 = rank(psi*psi) = mL * mR. If mL > dL, mR > dR, then this is fine
        # because we haven't done any truncation. If we have truncated then it's
        # unclear why dL * dR = d2...

        Zp = np.dot(u, v)
        A = X @ Zp
        theta = np.dot(Zp.T.conj(), theta)
    # Disentangler
    theta = np.reshape(theta, (dL, dR, mL, mR)) 
    theta = theta.transpose([2, 0, 1, 3]) # left to right


    theta, U, Ss = disentangle(theta, **disentangler_params)
    A = np.tensordot(A, np.reshape(np.conj(U), (dL, dR, dL * dR)), [1, 2])

    # Second splitting

    # theta, pipetheta = group_legs(theta, [[0,1],[2,3]])
    # This appears to be the problem
    theta = theta.transpose([1,0,2,3])
    theta = np.reshape(theta, (dL * mL, dR * mR))
    X, s, Z, info = svd_trunc(theta, trunc_params) # TODO insert trunc params
    v_err = info["p_trunc"]
    S = np.reshape(X, (dL, mL, len(s)))
    S = S * s

    B = np.reshape(Z, (len(s), dR, mR))
    B = B.transpose([1,0,2])

    return(A, S, B, h_err + v_err) # Returns sum of errors

def U2(psi):
    """ Calculates the 2-renyi entropy of a wavefunction psi. Returns the 
    2-renyi entropy and the unitary matrix minimizing the Renyi entropy
    (see the procedure described in https://arxiv.org/pdf/1711.01288.pdf).
    """
    chiL, d1, d2, chiR = psi.shape
    rhoL = np.tensordot(psi, psi.conj(), [[2,3],[2,3]])
    E2 = np.tensordot(rhoL, psi.conj(), [[0,1],[0,1]])
    E2 = np.tensordot(psi, E2, [[0,3],[0,3]])
    E2, pipeE = group_legs(E2, [[0,1],[2,3]])
    S2 = np.trace(E2)
    X, Y, Z = svd(E2)

    return(-np.log(S2), (X @ Z).T.conj())

def disentangle(psi, eps=1e-6, max_iter=120):
    """ Disentangles a wavefunction with 2-renyi polar iteration.
    
    Parameters
    ---------- 
    psi: TEBD style wavefunction. Leg ordering is ancilla - physical -physical -
    ancilla.

    eps: Minimum change between iterations.

    max_iter: Maximum number of iterations

    Returns
    ----------
    psiD: The disentangled wavefunction. psiD = U psi

    U: The unitary disentangler

    Ss: The series of 2-renyi entropies.
    """
    Ss = []
    chiL, d1, d2, chiR = psi.shape
    U = np.eye(d1 * d2)
    for i in range(max_iter):
        S, u = U2(psi)
        Ss.append(S)
        U = u @ U
        u = u.reshape([d1, d2, d1, d2])
        psi = np.tensordot(psi, u, [[1,2],[2,3]]).transpose([0,2,3,1])

        if i > 1 and Ss[-2] - Ss[-1] < eps:
            break
    return(psi, U, Ss)

