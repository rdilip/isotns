# coding: utf-8
""" My personal toy DMRG codes for 1D DMRG. Borrows heavily from tenpy.
"""
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh
import sys
import scipy.sparse.linalg.eigen.arpack as arp
from scipy.linalg import svd
from MPS import *
# For debugging


class DMRG:
    """ DMRG toycode class. """
    def __init__(self, psi, MPO, chi_max, eps=1e-14):
        self.L = psi.L
        self.psi = psi 
        self.MPO = MPO
        self.renvs = [None] * self.L
        self.lenvs = [None] * self.L
        self.chi_max = chi_max
        self.eps = eps
        
        chi = psi.Bs[0].shape[0] # bond dimension 
        D = self.MPO.Ws[0].shape[0] # MPO dimension
        
        lenv = np.zeros((chi, D, chi))
        renv = np.zeros((chi, D, chi))
        
        lenv[:,0,:] = np.eye(chi)
        renv[:, D - 1, :] = np.eye(chi)

        self.lenvs[0] = lenv
        self.renvs[-1] = renv # This is vR
        
        for i in range(self.L - 1, 1, -1):
            self.update_renv(i)

    def sweep(self):
        for i in range(self.psi.num_bonds - 1):
            self.update_bond(i)
        for i in range(self.psi.num_bonds - 1, 0, -1):
            self.update_bond(i)
    
    def update_bond(self, i):
        j = (i + 1) % self.psi.L
        h_eff = H_eff(self.lenvs[i], self.renvs[j], self.MPO.Ws[i], self.MPO.Ws[j])

        theta = self.psi.get_theta(i).reshape(h_eff.shape[0])
        evals, evec = arp.eigsh(h_eff, k=1, v0 = theta, which='SA')
        theta_new = evec.reshape(h_eff.theta_shape)
        [A,Sj,B] = split_and_truncate(theta_new, h_eff.theta_shape, self.chi_max, self.eps)
        # Return to right canonical form
        Si = self.psi.Ss[i]
        Bprev = np.tensordot(np.diag(np.power(Si, -1.0)), A, [1,0])
        Bprev = np.tensordot(Bprev, np.diag(Sj), [2,0])
        self.psi.Ss[j] = Sj
        self.psi.Bs[i] = Bprev
        self.psi.Bs[j] = B
        
        self.update_lenv(i)
        self.update_renv((i + 1) % self.psi.L)
    
    def _readout_filling(self):
        print(np.sum(self.psi.get_site_exp_val(self.num_ops)) / self.L)
             
    def update_lenv(self, i):
        """ Updates the left environment with all tensors right of tensor i """

        j = (i + 1) % self.psi.L
        lenv_i = self.lenvs[i] 
        W = self.MPO.Ws[i]
        B = self.psi.Bs[i]

        S = np.diag(self.psi.Ss[i])
        Sinv = np.diag(np.power(self.psi.Ss[j], -1.0))
        
        G = np.tensordot(S, B, axes=[1,0])
        A = np.tensordot(G, Sinv, [2,0])
        A_ = np.conj(A)
        
        lenv_new = np.tensordot(lenv_i, A, axes=[0,0])
        lenv_new = np.tensordot(lenv_new, W, axes=[[0,2],[0,3]])
        lenv_new = np.tensordot(lenv_new, A_, axes=[[0,3],[0,1]])
        
        self.lenvs[(i+1) % self.L] = lenv_new
        
    def update_renv(self, i):
        """ Updates the right environment with all tensors right of tensor i """
        renv_i = self.renvs[i]
        W = self.MPO.Ws[i]
        B = self.psi.Bs[i]
        B_ = np.conj(B)
        
        renv_new = np.tensordot(B, renv_i, axes=[2, 0]) 
        renv_new = np.tensordot(renv_new, W, axes=[[1, 2], [3, 1]]) 
        renv_new = np.tensordot(renv_new, B_, axes=[[1, 3], [2, 1]])  

        self.renvs[(i-1) % self.L] = renv_new
        
class H_eff(scipy.sparse.linalg.LinearOperator):
    """ Class for a effective Hamiltonian to perform two site DMRG """
    def __init__(self, lenv, renv, W1, W2):
        self.lenv = lenv
        self.renv = renv
        self.W1 = W1
        self.W2 = W2
        
        self.dtype = W1.dtype
        
        chiL, chiR = lenv.shape[0], renv.shape[0] 
        d1, d2 = W1.shape[2], W2.shape[2] # because your indexing is stupid
        self.theta_shape = (chiL, d1, d2, chiR)
        self.shape = (chiL * d1 * d2 * chiR, chiL * d1 * d2 * chiR)
    
    def _matvec(self, theta):
        """ Returns the action of H_eff on a two-site state theta, so that one
        does not need to explicitly construct the matrix H_eff """
        state = theta.reshape(self.theta_shape)
        state = np.tensordot(self.lenv, state, axes=[0,0])
        state = np.tensordot(state, self.W1, axes=[[0,2],[0,3]])
        state = np.tensordot(state, self.W2, axes=[[3,1],[0,3]])
        state = np.tensordot(state, self.renv, axes=[[1,3],[0,1]])
        
        return(state.reshape(self.shape[0]))
    
    
class TFI:
    """ Class for a transverse field ising model """
    def __init__(self, L, g, J, bc="finite"):
        assert(bc == "finite" or bc == "infinite")
        self.bc = bc
        self.g = g
        self.J = J
        self.d = 2
        sx = np.array([[0,1],[1,0]])
        sy = np.array([[0, -1.0j],[1.0j, 0]])
        sz = np.array([[1.0, 0.0],[0.0, -1.0]])
        idn = np.eye(2)
        self.L = L
        
        self.Ws = []
        for i in range(L):
            w = np.zeros((3, 3, self.d, self.d), dtype = np.float)
            w[0,0] = w[2,2,:,:] = idn
            w[0,1] = sx
            w[0,2] = -g * sz
            w[1,2] = -J * sx
            self.Ws.append(w)
            
    def get_H_bonds(self):
        """ Returns a list of local operators corresponding to TFI model. """
        num_bonds = (self.L if self.bc == "infinite" else self.L - 1)
        sx = np.array([[0,1],[1,0]])
        sz = np.array([[1.0, 0.0],[0.0, -1.0]])
        idn = np.eye(2)
        
        ops = []
        
        for site in range(num_bonds):
            gL = gR = 0.5 * self.g
            if self.bc == "finite":
                if site == 0: gL = self.g
                if site == self.L - 2: gR = self.g
            H_local = -self.J * np.kron(sx, sx) - gL * np.kron(sz, idn) - gR * np.kron(idn, sz)
            ops.append(np.reshape(H_local, [self.d] * 4))

        self.H_bonds = ops
        return(ops)
    
if __name__ == '__main__':
    L = 10
    J = 1.0
    g = 1.0
    d = 2
    chi_max = 20

    sx = [[0,1],[1,0]]
    sz = [[1,0],[0,-1]]

    psi = get_FM_MPS(L, d)
    tfi_model = TFIMPO(L, g, J)

    dmrg = DMRG(psi, tfi_model, chi_max)

    ops = tfi_model.get_hamiltonian_local_op()
    for i in range(10):
        dmrg.sweep()
        psi = dmrg.MPS
        print(i, np.sum(psi.get_bond_exp_val(ops)))
    ops_x = [sx for i in range(L)]
    ops_z = [sz for i in range(L)]

    print("Bond dimensions: {0}".format(psi.get_chi()))
    print("Magnetization in x: {0}".format(round(np.sum(psi.get_site_exp_val(ops_x)), 5)))
    print("Magnetization in z: {0}".format(round(np.sum(psi.get_site_exp_val(ops_z)), 5)))
