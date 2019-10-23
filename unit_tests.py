""" Module to test that basic functionalities of isoTNS are working """
from pprint import pprint
import numpy as np
from rd_model import *
from rd_mps import *
from rd_tebd import *
from tebd import *

# TEBD down one column should output same results as 1D TEBD

def base_tebd(psi, Us, num_iter=100, chi_max=100):
    for run in range(num_iter):
        for k in [0,1]:
            for site in range(k, psi.L-1, 2):
                U = Us[site]
                theta = psi.get_theta(site)
                theta_ = np.tensordot(theta, U, [[1,2],[0,1]]).transpose([0,2,3,1])
                psi.update_theta(site, theta_, theta_.shape, chi_max)
    return(psi)

def test_tebd():
    trunc_params = dict(chi_max=100, p_trunc=1.e-10)
    L = 10
    d = 2
    dt = 0.1
    psi = get_FM_MPS(L, d)
    model = TFI(L, 1.0, 1.0)
    bonds = model.get_H_bonds()
    Us = get_time_evol(bonds, dt)
    pprint("Base Energy: {0}".format(psi.get_bond_exp_val(bonds)))
    pprint("Running TEBD")
    psi = base_tebd(psi, Us)
    pprint("Energy: {0}".format(psi.get_bond_exp_val(bonds)))

    b = np.zeros([2,1,1,1,1])
    b[0,0,0,0,0] = 1.
    Psi = [b.copy() for i in range(L)]
    pprint("Base isoTNS energy: {0}".format(get_expectation_value(Psi, bonds, trunc_params)))

    for i in range(100):
        Psi, expvals = sweep(Psi, Us=Us, trunc_params = trunc_params)
        Psi, expvals = sweep(Psi, Us=Us, trunc_params = trunc_params,\
                            direct="L")
    pprint("isoTNS energy: {0}".format(get_expectation_value(Psi, bonds, trunc_params)))

