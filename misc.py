""" Miscellaneous functions for various tensor operations."""
# TODO Make a general MPS class; a lot of these are functions that specifically
# to an MPS.
import numpy as np
import scipy as sp
import scipy.linalg as la
import warnings
import itertools
import pickle

def group_legs(T, legs):
    """ Function to group legs of a tensor together.
    Parameters
    ----------
    T: np.array tensor.

    legs: List of legs to group together. [[l1,l2,l3],[l4,l5]] corresponds to 
    grouping legs l1,l2,l3 together and l4,l5 together.

    Returns
    ----------
    T_: The grouped tensor

    pipe: A tuple of a permutation and the original leg shape. This can be passed
    to ungroup legs to put the legs back in the original form.
    """
    perm = []
    for leg in legs:
        perm.extend(leg)
    T_ = np.transpose(T, perm)
    m = 0
    new_shape = []
    old_shape = []
    for leg in legs:
        n = len(leg)
        new_shape.append(np.prod(T_.shape[m:m+n]))
        old_shape.append(T_.shape[m:m+n])
        m += n
    pipe = (perm, old_shape)
    T_ = T_.reshape(new_shape)
    return(T_, pipe)

def flatten(myiter):
    return(list(itertools.chain(*myiter)))

def ungroup_legs(T, pipe):
    """ Ungroups the legs.

    Parameters
    ----------
    T: The tensor to ungroup

    pipe: A tuple where the first element is a permutation and the second is 
    the original shape. These are the outputs of group_legs

    Returns
    ----------
    T_: The original tensor
    """
    perm, old_shape = pipe
    if (len(old_shape) != T.ndim):
        raise ValueError("Dimensions of shape and tensor must match")
    shape = []
    
    for i in range(len(old_shape)):
        if len(old_shape[i]) == 1:
            shape.append(T.shape[i])
        else: 
            shape.extend(old_shape[i])
    
    T_ = np.reshape(T, shape)
    T_ = T_.transpose(inverse_transpose(perm))
    return(T_)

def change_pipe_shape(pipe, leg, new_shape):
    """ If you do some operation that changes the size of legs while grouped
    (for example, SVD and truncation) then this function changes the shapes in 
    the pipe so we can use the ungroup function easily. (This is fairly trivial,
    but I keep forgetting the order of the elements in the pipe """
    warnings.warn("Don't use this. Messy solution to a nonexistent problem",\
                    DeprecationWarning)
    shape = []
    for i in range(len(pipe[1])):
        if i != leg: shape.append(pipe[1][i])
        else: shape.append(new_shape)
    new_pipe = [pipe[0], shape]
    return(new_pipe)

def get_closest_factors(N):
    """ Finds the two numbers a and b such that ab = N, |a-b| is minimized """
    for a in range(int(np.sqrt(N)), 0, -1):
        if N % a == 0:
            return(a, int(N / a))
    return(-1)

def inverse_transpose(perm):
    """ Returns the inverse of a permutation """
    inv = [0]* len(perm)
    for i in range(len(perm)):
        inv[perm[i]] = int(i)
    return(inv)

def mps_invert(Psi):
    """ Inverts an MPS. If we assume that the last two legs are the only virtual
    ones, then we just have to flip those """
    num_p = Psi[0].ndim - 2
    return([psi.transpose(list(range(num_p)) + [-1,-2]) for psi in Psi[::-1]])

def operator_invert(ops):
    """ Flips a list of two-site operators. Transposes each individual operator, 
    so methods that sweep along one direction can be easily flipped. """ 
    return([op.transpose([1,0,3,2]) for op in ops[::-1]])

def svd(A, full_matrices = False):
    """ Robust version of svd """
    try:
        return(la.svd(A, full_matrices = full_matrices))
    except np.linalg.linalg.LinAlgError:
        warnings.warn("SVD with LAPACK driver 'gesdd' failed. Using 'gesvd'",\
                      stacklevel=2)
        return(la.svd(A, full_matrices = full_matrices, lapack_driver='gesvd'))

def svd_trunc(A, trunc_params = None):
    """ Performs a truncated SVD. This SVD function is also robust -- if there
    is an error while using the iterative scipy methods it directly calls
    the LAPACK cgsvd method. 
    Parameters
    ---------
    A: Matrix to perform SVD on.

    trunc_params: Dictionary of trunc_params. chi_max is the largest bond
    dimension. p_trunc is the "total probability" that we are discarding -- 
    the sum of all singualr values squared greater than the cut. 
    
    Returns
    ---------
    U, SV: U @ S @ V.T = A
    """
    trunc_info = {}
    if trunc_params is None:
        trunc_params = {}
    U, S, V = svd(A, full_matrices=False)
    nrm = np.linalg.norm(S)

    p_trunc = trunc_params.get("p_trunc", 0.0)
    chi_max = trunc_params.get("chi_max", len(S))
    if p_trunc > 0.0:
        eta = np.count_nonzero(nrm**2 - np.cumsum(S**2) > p_trunc) + 1
        eta_new = np.min((eta, chi_max))
    else:
        eta_new = chi_max

    nrm_t = np.linalg.norm(S[:eta_new])
    trunc_info["p_trunc"] = nrm**2 - nrm_t**2
    trunc_info["nrm_t"] = nrm_t
    trunc_info["eta"] = eta_new
    return(U[:, :eta_new], S[:eta_new], V[:eta_new, :], trunc_info)

# MPS and MPO manipulations

def mps_group_legs(Psi, axes = 'all'):
    """ Given an MPS with a high number of physical legs with B tensors, group
    the physical legs according to axes = [[l1, l2], [l3]]... As usual, for rank
    n tensors, the first n - 2 legs are considered to be physical."""

    if axes == 'all':
        axes = [list(range(Psi[0].ndim - 2))]

    Psi_ = []
    pipes = []
    for j in range(len(Psi)):
        ndim = Psi[j].ndim
        p, pipe = group_legs(Psi[j], axes + [[ndim - 2], [ndim - 1]])
        Psi_.append(p)
        pipes.append(pipe)
    return(Psi_, pipes)

def mps_ungroup_legs(Psi, pipes):
    """ Returns an MPS that was grouped using mps_group_legs """ 
    return([ungroup_legs(Psi[i], pipes[i]) for i in range(len(Psi))])

def canonical_form(Psi, form = 'A', normalize = False):
    """ Puts an MPS into either A or B form (can have arbitrarily many physical
    legs. """
    assert form in ['A', 'B']
    Psi, pipes = mps_group_legs(Psi, axes = 'all')
    if form == 'B':
        Psi = [psi.transpose([0,2,1]) for psi in Psi[::-1]]

    L = len(Psi)
    T = Psi[0]
    for j in range(L - 1):
        T, pipe = group_legs(T, [[0,1],[2]])
        A, S = np.linalg.qr(T)
        #A, S = sp.linalg.qr(T) (For some reason, scipy qr is MUCH slower...)
        Psi[j] = ungroup_legs(A, pipe)
        T = np.tensordot(S, Psi[j + 1], axes = [1,1]).transpose([1,0,2])
    
    if normalize:
        Psi[L - 1] = T / la.norm(T)
    else:
        Psi[L - 1] = T

    if form == 'B':
        Psi = [psi.transpose([0,2,1]) for psi in Psi[::-1]]

    Psi = mps_ungroup_legs(Psi, pipes)
    return(Psi)

def contract_mpos(X, Y, form = None):
    """ Contracts two MPOS, placing final MPO in form if specified """
    if X[0].ndim != 4 or Y[0].ndim != 4:
        raise ValueError("MPOs must have rank 4")
    XY = []
    XY = [group_legs(np.tensordot(x, y, [1,0]), 
                    [[0], [3], [1,4], [2,5]])[0] for x, y in zip(X, Y)]

    if form is not None:
        XY = mps_2form(XY, form)
    return(XY)

# tmp file save and load
def savefile(obj):
    with open("../rd_tmpfile.pkl", "wb+") as f:
        pickle.dump(obj, f)

