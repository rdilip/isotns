""" Toy implementation of TEBD, for use in isometric TNS work """
import scipy.linalg as la
import numpy as np

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

def run_tebd(psi, H_bonds, num_iter, dt, chi_max):
    Us = get_time_evol(H_bonds, dt)
    for run in range(num_iter):
        for k in [0,1]:
            for site in range(k, psi.L-1, 2):
                U = Us[site]
                theta = psi.get_theta(site)
                theta_ = np.tensordot(theta, U, [[1,2],[0,1]]).transpose([0,2,3,1])
                psi.update_theta(site, theta_, theta_.shape, chi_max)
    print(np.sum(psi.get_bond_exp_val(H_bonds)))

