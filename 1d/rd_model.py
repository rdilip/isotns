""" My Transverse field ising model """    
import numpy as np
class TFI:
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
