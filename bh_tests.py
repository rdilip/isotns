import numpy as np
import scipy.linalg as la
import pickle
from tebd import tebd 
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import BoseHubbardChain
from tenpy.algorithms import tebd as tenpy_tebd
from tenpy.algorithms import dmrg as tenpy_dmrg

# TODO
# * Implement second order Trotterization. See if that improves the comparison
# to DMRG.
#       (I don't think this improves the DMRG comparison...because you can't 
#       trotterize the 1D. The Trotterization is a row column thing now, not an
#       even odd thing).
# * Write and run a 2D bose hubbard simulation. Not Hofstadter yet.
# * Compare a 2D bose hubbard simulation to DMRG results on 1D.
# * Implement Hofstadter phase difference.
# * Do the same comparison with Hofstadter phase difference.



def get_N(Nmax = 3):
    return(np.diag(np.arange(Nmax + 1)))

def creation_op(Nmax = 3):
    return(np.diag(np.sqrt(np.arange(1, Nmax + 1)), -1))

def annihilation_op(Nmax = 3):
    return(np.transpose(creation_op(Nmax)).conj())

def get_bose_hubbard_bonds(L = 8, t = 0.1, U = 0.14, Nmax = 3):
    """ Hamiltonian is 

    H = \sum_{i, j} -t \dagger{a_i}a_j + (U / 2) n(n - 1) 

    Note the signs 
    """
    num_bonds = L - 1
    d = Nmax + 1
    N = get_N(Nmax)
    id = np.eye(d)
    ad = creation_op(Nmax)
    a = np.transpose(ad)

    Uop = np.dot(N, N - id)
    ops = []
    for site in range(num_bonds):
        UL = UR = 0.5 * U
        if site == 0:
            UL = U
        if site == L - 2:
            UR = U
        H_local = -t * np.kron(ad, a) - t * np.kron(a, ad) +\
                (UL / 2.) * np.kron(Uop, id) + (UR / 2.) * np.kron(id, Uop)
        ops.append(np.reshape(H_local, [d]*4))
    return(ops)

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

def expectation_value(Psi, Os):
    """ For now O should be a two site operator. We'll need to fix this later.  """
    trunc_params = {"chi_max": 2**32, "p_trunc": 0} # No truncation
    if Os == 'N':
        Os = get_number_op(L = len(Psi), Nmax = Psi[0].shape[0] - 1)

    Psi, _ = tebd(Psi, None, Os, trunc_params,\
                                reduced_update = True, direct = 'R')
    Psi, info = tebd(Psi, None, Os, trunc_params,\
                                reduced_update = True, direct = 'L')
    return(info["expectation_O"])

def get_number_op(L = 8, Nmax = 3):
    """ Returns number operator as a two site operator
    """
    num_bonds = L - 1
    d = Nmax + 1
    N = get_N(Nmax)
    id = np.eye(d)

    ops = []
    for site in range(num_bonds):
        L = R = 0.5
        if site == 0:
            L = 1.0
        if site == L - 2:
            R = 1.0
        num_op = L * np.kron(N, id) + R * np.kron(id, N)
        ops.append(np.reshape(num_op, [d]*4))
    return(ops)

def tenpy_1D_tebd_bose_hubbard(L = 8, t = 0.1, U = 0.14, save = True):
    M = BoseHubbardChain({"L": 8, "t": 0.1, "U": 0.14, "bc_MPS": "finite"})
    psi = MPS.from_product_state(M.lat.mps_sites(), [1,0,0,0,0,0,0,0], "finite")
    tebd_params = {"order": 1, "delta_tau_list": [0.1, 0.001, 1e-5],
        "max_error_E": 1.e-6,
        "trunc_params": {"chi_max": 32, "svd_min": 1.e-10}}
    eng = tenpy_tebd.Engine(psi, M, tebd_params)
    eng.run_GS() # imaginary time evolution with TEBD
    print("E =", sum(psi.expectation_value(M.H_bond[1:])))
    with open("tenpy_bh_tebd_L={0},t={1},U={2}.pkl".format(L, t, U), "wb+") as f:
        pickle.dump([psi, M, tebd_params], f)

def tenpy_1D_dmrg_bose_hubbard(L = 8, t = 0.1, U = 0.14, save = True):
    M = BoseHubbardChain({"L": 8, "t": 0.1, "U": 0.14, "bc_MPS": "finite"})
    psi = MPS.from_product_state(M.lat.mps_sites(), [1,0,0,0,0,0,0,0], "finite")
    dmrg_params = {"trunc_params": {"chi_max": 32, "svd_min": 1.e-10}}
    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi= eng.run() # imaginary time evolution with TEBD
    print("E =", sum(psi.expectation_value(M.H_bond[1:])))
    with open("tenpy_bh_dmrg_L={0},t={1},U={2}.pkl".format(L, t, U), "wb+") as f:
        pickle.dump([psi, E], f)
    return(psi, E)

def iso_1D_tebd_bose_hubbard(L = 8, t = 0.1, U = 0.14, save = True):
    bh_bonds = get_bose_hubbard_bonds()
    b = np.zeros([4,1,1])
    b[0,0,0] = 1.0
    Psi = [b.copy() for i in range(8)]
    Psi[0][1,0,0] = 1.0
    Psi[0][0,0,0] = 0.0
    tp = {"chi_max": 6, "p_trunc": 1.e-10}
    dts = [0.1, 0.001, 1.e-5]
    E_curr = 0.0
    for dt in dts:
        Us = get_time_evol(bh_bonds, dt)
        step = 0
        delta_E = float("inf")
        while np.abs(delta_E) > 1.e-6:
            E_prev = E_curr
            Psi, info = tebd(Psi, Us = Us, Os = bh_bonds, trunc_params = tp, direct = 'R')
            Psi, info = tebd(Psi, Us = Us, Os = bh_bonds, trunc_params = tp, direct = 'L')
            E_curr = np.sum(info["expectation_O"])
            delta_E = E_curr - E_prev
            if step % 10 == 0: 
                print("Step: {0}, Delta E: {1}".format(step, delta_E))
            step += 1
    with open("iso_bh_L={0},t={1},U={2}.pkl".format(L, t, U), "wb+") as f:
        pickle.dump([Psi, info], f)
    return(Psi, info)

def iso_2D_tebd_bose_hubbard(L = 8, t = 0.1, U = 0.14, save = True):
    bh_bonds = get_bose_hubbard_bonds()
    trunc_params = {"chi_max": 32, "p_trunc": 1.e-10}
    peps = isotns(get_peps(Lx=L, Ly=L, fill=(1,8), Nmax=3))
    Tstep = 1.5
    dts = 1.5 * np.exp(-0.5 * np.arange(1,12))
    H = get_bose_hubbard_bonds()
    Hs = [H.copy(), H.copy()]
    Es = []
    for dt in dts:
        info = peps.tebd2(Hs, dt, trunc_params, int(Tstep / dt) + 1)
        with open("iso_bh_tebd_L={0},t={1},U={2}.pkl", "wb+") as f:
            pickle.dump([info, peps, dt])
    print("Done")


def get_Psi(Ly = 8, fill = (1, 8), Nmax = 3):
    a, b = fill
    assert Ly % b == 0
    occ = int(a * Ly / b)
    d = Nmax + 1
    b = np.zeros([d, 1, 1, 1, 1])
    Psi = [b.copy() for i in range(Ly)]
    for i in range(occ):
        Psi[i][1,0,0,0,0] = 1.0
    for i in range(occ, Ly):
        Psi[i][0,0,0,0,0] = 1.0
    np.random.shuffle(Psi)
    return(Psi)

def get_peps(Lx = 8, Ly = 8, fill = (1,8), Nmax = 3):
    peps = []
    for i in range(Lx):
        peps.append(get_Psi(Ly, fill=fill, Nmax=Nmax))
    return(peps)
