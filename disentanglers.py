import numpy as np
""" Library of various disentanglers. Each disentangler should take a function
psi and output a disentangled psi and the unitary disentangler U """

def U2(psi):
    """ Calculates the 2-renyi entropy of a wavefunction psi. Returns the 
    2-renyi entropy and the unitary matrix minimizing the Renyi entropy
    (see the procedure described in https://arxiv.org/pdf/1711.01288.pdf).
    Changed to mirror Mike/Frank's code for comparison.
    """
    chiL, d1, d2, chiR = psi.shape
    rhoL = np.tensordot(psi, psi.conj(), [[2,3],[2,3]])
    E2 = np.tensordot(rhoL, psi, [[2,3],[0,1]])
    E2 = np.tensordot(psi.conj(), E2, [[0,3],[0,3]])
    E2 = E2.reshape((d1*d2, -1))
    S2 = np.trace(E2)
    X, Y, Z = np.linalg.svd(E2)
    return -np.log(S2), (np.dot(X, Z).T).conj()

def renyi_2_disentangler(psi, eps=1e-5, max_iter=120):
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
    U = np.eye(d1 * d2, dtype = psi.dtype)
    go = True
    m = 0
    while m < max_iter and go:
        S, u = U2(psi)
        Ss.append(S)
        U = u @ U
        u = u.reshape([d1, d2, d1, d2])
        # ERROR: my commented construction doesn't work. 
        psi = np.tensordot(u, psi, axes=[[2,3],[1,2]]).transpose([2,0,1,3])
#        psi = np.tensordot(psi, u, [[1,2],[2,3]]).transpose([0,2,3,1])

        if m > 1:
            go = Ss[-2] - Ss[-1] > eps
        m += 1

    return(psi, U)

