""" Class for 1D Matrix Product States """
import numpy as np
from scipy.linalg import svd
import pickle
import scipy.linalg as la


class MPS:
    """ 1D MPS class
    Attributes
    ----------
    Ss: Schmidt values at each site (determines the entanglement entropy across
    a bond).

    Bs: B matrices making up the MPS. Once the MPS is converged, it is easy to
    move between the A and B canonical forms via A_n = S_n^{-1} B_n S_{n+1}. 
    The index convention is vL p vR.

    num_bonds: The number of bonds. For the infinite MPS, this is the length
    of the MPS (there is one additional bond to allow the environments to grow
    at each stage). 

    L: Length of the MPS.
    """
    def __init__(self, Ss, Bs, bc="finite"):
        self.Ss = Ss
        self.Bs = Bs
        assert(bc=="finite" or bc=="infinite")
        self.bc = bc
        self.num_bonds = len(Bs) - 1 if bc == "finite" else len(Bs)
        self.L = len(Bs)

    def __copy__(self):
        return(MPS(self.Ss, self.Bs, self.bc))
        
    def get_theta(self, ind, k=2):
        """ Returns the k-site wavefunction for the MPS in mixed canonical 
        form at i, i+1  """
        assert(ind <= self.num_bonds)
        assert k in [1,2]
        if k == 2:
            theta = np.tensordot(np.diag(self.Ss[ind]), self.Bs[ind], axes=[1,0])
            theta = np.tensordot(theta, self.Bs[(ind + 1) % self.L], axes=[2,0])
            return(theta)
        elif k == 1:
            theta = np.tensordot(self.Bs[ind], np.diag(self.Ss[ind]), axes=[0,1])
            return(theta)
    
    def get_bond_exp_val(self, ops):
        """ ops shoudl be a list of local two-site operators. ops[0] should act between sites 0
        and 1. ops[num_bonds - 1] should act between sites L-1 and 0 if infinite, and sites L -2 and
        L -1 if finite. """
        # assert len(ops) == self.num_bonds, "Number of operators must be equal to number of bonds"
        exp_vals = [None] * len(ops)
        for l_site in range(self.num_bonds):
            theta = self.get_theta(l_site)
            theta_ = np.conj(theta)
            exp_vals[l_site] = np.einsum("ijkl,mnjk,imnl", theta, ops[l_site], theta_)
        return(exp_vals)
    
    def get_site_exp_val(self, ops):
        """ Returns the expectation value of a list of local operators. """
        assert(len(ops) == self.L)
        exp_vals = [None] * len(ops)
        for site in range(self.L):
            theta = self.get_theta(site, k=1)
            theta_ = np.conj(theta)
            exp_val = np.tensordot(theta, ops[site], axes=[0,1])
            exp_val = np.tensordot(exp_val, theta_, axes=[[0,2,1],[1,0,2]])
            exp_vals[site] = exp_val
        return(exp_vals)
    
    def correlation_length(self, site_length = 2):
        """ The correlation length is determined by the second largest eigenvalue
        of the transfer matrix, which corresponds to a decay parameter. See
        tenpy docs for more info. """

        assert self.bc == "infinite"
        B = self.Bs[0]
        T = np.tensordot(B, np.conj(B), axes = [1, 1])
        T = np.transpose(T, [0, 2, 1, 3])

        for i in range(1, site_length):
            Bi = self.Bs[i]
            Ti = np.tensordot(Bi, np.conj(Bi), axes = [1, 1])
            Ti = np.transpose(Ti, [0, 2, 1, 3])
            T = np.tensordot(T, Ti, [[2,3],[0,1]])
        chi = B.shape[0]
        T = T.reshape((chi**2, chi**2))
        eta = arp.eigs(T, which = 'LM', k=2, return_eigenvectors=False, ncv=20)
        return (- self.L / np.log(abs(min(eta))))

    def entanglement_entropy(self, k=1, bond = None):
        """ Returns the kth Renyi entropy across a bond. If bond = None, 
        returns the list of Renyi entropies """
        print("Returning {k}th Renyi entropy".format(k=k))
        bonds = range(self.num_bonds)
        result = []
        if bond is not None:
            S = self.Ss[bond].copy()
            Sr = (S * S) ** k
            return(-np.sum(Sr * np.log(Sr)))
        for i in bonds:
            S = self.Ss[i].copy()
            Sr = (S * S) ** k
            result.append(-np.sum(Sr * np.log(Sr)))
        return(np.array(result))

    def update_theta(self, i, theta, shape, chi_max):
        j = (i + 1) % self.L
        [A,Sj,B] = split_and_truncate(theta, shape, chi_max)
        # Return to right canonical form
        Si = self.Ss[i]
        tmp = np.diag(np.power(Si, -1.0))
        Bprev = np.tensordot(np.diag(np.power(Si, -1.0)), A, [1,0])
        Bprev = np.tensordot(Bprev, np.diag(Sj), [2,0])
        oldSV = self.Ss[j]
        self.Ss[j] = Sj
        self.Bs[i] = Bprev
        self.Bs[j] = B
        return(oldSV, Sj)


def inner(psi1, psi2):
    """ Returns the inner product of two matrix product states """
    assert psi1.L == psi2.L
    contracted_tensors = []
    L = psi1.L
    for i in range(L):
        contr = np.tensordot(psi1.Bs[i], psi2.Bs[i], [1,1]).transpose([0,2,1,3])
        contracted_tensors.append(contr)
    outp = contracted_tensors[0]
    for i in range(1, L):
        outp = np.tensordot(outp, contracted_tensors[i], [[2,3],[0,1]])
    return(np.trace(outp, axis1=0, axis2=1).trace(axis1=0, axis2=1))



def get_FM_MPS(L, d, bc="finite"):
    B = np.zeros([1, d, 1], np.float)
    B[0, 0, 0] = 1.
    S = np.ones([1], np.float)
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]
    return(MPS(Ss, Bs, bc))

def split_and_truncate(theta, shape, chi_max, eps=1.e-14):
    """ Splits theta, performs an SVD, and trims the matrices to chi_max. Returns
    A[i], S[i], B[i+1] """
    
    chiL, dL, dR, chiR = shape
    theta_matrix = theta.reshape((chiL * dL, chiR * dR))
    U, Sfull, V = svd(theta_matrix, full_matrices = False)
    
    chi_keep = np.sum(Sfull > eps) # Number of svals > eps
    chi_keep = min(chi_keep, chi_max)
    
    sv_indices = np.argsort(Sfull)[::-1][:chi_keep]
    A = U[:,sv_indices]
    B = V[sv_indices,:]
    S = Sfull[sv_indices]
    S = S / np.linalg.norm(S) # Normalization so schmidt values squared sum to 1
    
    A = A.reshape([chiL, dL, chi_keep])
    B = B.reshape([chi_keep, dL, chiR])
    return([A,S,B])
