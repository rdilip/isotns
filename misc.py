""" Miscellaneous functions for various tensor operations """
import numpy as np
import itertools

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
        if len(old_shape[i]) == 1: shape.append(old_shape[i][0])
        else: shape.extend(old_shape[i])
    T_ = np.reshape(T, shape)
    T_ = T_.transpose(inverse_transpose(perm))
    return(T_)

def change_pipe_shape(pipe, leg, new_shape):
    """ If you do some operation that changes the size of legs while grouped
    (for example, SVD and truncation) then this function changes the shapes in 
    the pipe so we can use the ungroup function easily. (This is fairly trivial,
    but I keep forgetting the order of the elements in the pipe """

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



