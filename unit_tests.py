""" Module to test that basic functionalities of isoTNS are working """
from pprint import pprint
import numpy as np
from tebd import *
import sys

sys.path.insert(1, "../Archive/")
from tebd_copy import * # Mike's code


# TEBD down one column should output same results as 1D TEBD

def get_trunc_params():
    rd_trunc_params = dict(chi_max = 32, eta_max = 32, p_trunc=1.e-10)
    mz_trunc_params = dict(p_trunc=1.e-10, bond_dimensions = dict(chi_max=32, eta_max=32))
    return(rd_trunc_params, mz_trunc_params)


def test_tebd():
    L, J, g, dt = 10, 1., 1., 0.1
    Psi0 = [np.random.random([2,1,1,1,1]) for i in range(10)]
    Psi0 = [t / np.linalg.norm(t) for t in Psi0]
    O = get_TFI_bonds(L, J=J, g=g)
    U = get_time_evol(O, dt)

    rd_tp, mz_tp = get_trunc_params()

    mzPsi = Psi0.copy()
    for i in range(100):
        mzPsi, info = tebd_on_mps(mzPsi, U = U, truncation_par = mz_tp, O = O, order = 'R')
        mzPsi, info = tebd_on_mps(mzPsi, U = U, truncation_par = mz_tp, O = O, order = 'L')
    
    rdPsi = Psi0.copy()
    for i in range(100):
        rdPsi, expvals, err = tebd_on_mps(rdPsi, Us = U, trunc_params=rd_tp, Os = O, order = 'R')
        rdPsi, expvals, err= tebd_on_mps(rdPsi, Us = U, trunc_params=rd_tp, Os = O, order = 'L')

    print(np.all([np.all(np.abs(rdPsi[i] - mzPsi[i]) < 1.e-10) for i in range(L)]))

if __name__ == '__main__':
    test_tebd()
