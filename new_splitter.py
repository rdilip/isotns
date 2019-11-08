""" Splitter function modeled after Mike's code to make it easier to test. """
import numpy as np
from misc import *
import scipy.linalg as la
from disentanglers import renyi_2_disentangler

def split_psi(psi, split_dim, trunc_params, disentangler_params = None,
                disentangler = renyi_2_disentangler, init_from_polar = True):
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
    tp = dict(p_trunc = 0.0, eta = dL * dR)
#    X, y, Z, trunc_info_H = svd_trunc(psi.reshape((-1, mL * mR)), **tp)
    X,y, Z, D2, trunc_leg = svd_theta_UsV(psi.reshape((-1, mL*mR)), dL*dR, 0.)

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
        Zp = np.dot(u, v)
        A = X @ Zp
        theta = np.dot(Zp.T.conj(), theta)
    # Disentangler
    theta = np.reshape(theta, (dL, dR, mL, mR)) 
    theta = theta.transpose([2, 0, 1, 3]) # left to right
    
    theta, U, Ss = disentangler(theta, **disentangler_params)

    A = np.tensordot(A, np.reshape(np.conj(U), (dL, dR, dL * dR)), [1, 2])
    # Second splitting
    theta = theta.transpose([1,0,2,3])
    theta = np.reshape(theta, (dL * mL, dR * mR))

    X, s, Z, chi_C, trunc_bond = svd_theta_UsV(theta, trunc_params['chi_max'], p_trunc=3e-16)
    errH = trunc_leg
    errV = trunc_bond

    info = dict(error = errH + errV, d_error = errH, sLambda = s)
    S = np.reshape(X, (dL, mL, len(s)))
    S = S * s
    B = np.reshape(Z, (len(s), dR, mR))
    B = B.transpose([1,0,2])
    return(A, S, B, info) # Returns sum of errors


