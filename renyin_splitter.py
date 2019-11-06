from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
#import ls

from misc import *


def split_psi(Psi,
              dL,
              dR,
              truncation_par={
                  'chi_max': 32,
                  'p_trunc': 1e-6
              },
              verbose=0,
              n=0.5,
              eps=1e-6,
              max_iter=120,
              init_from_polar=True,
              flag = False):
    """ Given a tripartite state psi.shape = d x mL x mR   , find an approximation
	
			psi = A.Lambda
	
		where A.shape = dL x dR x d  is an isometry that "splits" d --> dL x dR and Lambda.shape = mL dL dR mR 
		is a 2-site TEBD-style wavefunction of unit norm and maximum Schmidt rank 'chi_max.'
	
		The solution for Lambda is given by the MPS-type decomposition

			Lambda_{mL dL dR mR} = \sum_eta  S_{dL mL eta} B_{dR eta mR}

		where 1 < eta <= chi_max, and Lambda has unit-norm

		Arguments:
		
			psi:  shape= d, mL, mR
			
			dL, dR: ints specifying splitting dimensions (dL,dR maybe reduced to smaller values)
			
			truncation_par  = {'chi_max':, 'p_trunc':}  truncation parameters for the SVD of Lambda; p_trunc is acceptable discarded weight
			
			eps: precision of iterative solver routine
			
			max_iter: max iterations of routine (through warning if this is reached!)
		
			verbose:
				
				
		Returns:
		
			A: d x dL x dR
			S: dL x mL x eta
			B: dR x eta x mR
		
			info = {} , a dictionary of (optional) errors
			
				'error': 'trunc_leg','trunc_bond'
	"""
    d, mL, mR = Psi.shape

    dL = np.min([dL, mL])
    dR = np.min([dR, mR])

    if dL * dR > d:
        dR = min([int(np.rint(np.sqrt(d))), dR])
        dL = min([d // dR, dL])


    X, y, Z, D2, trunc_leg = svd_theta_UsV(Psi.reshape((-1, mL * mR)), dL * dR,
                                           0.)  #First SVD down d
    A = X
    theta = (Z.T * y).T
    init_from_polar = False

    if init_from_polar:

        psi = theta.reshape((D2, mL, mR))
        

        #First truncate psi to (D2, dL, dR) based on Schmidt values
        if mL > dL:
            rho = np.tensordot(psi, psi.conj(), axes=[[0, 2], [0, 2]])
            p, u = np.linalg.eigh(rho)
            u = u[:, -dL:]
            psi = np.tensordot(psi, u.conj(), axes=[[1],
                                                    [0]]).transpose([0, 2, 1])

        if mR > dR:
            rho = np.tensordot(psi, psi.conj(), axes=[[0, 1], [0, 1]])
            p, u = np.linalg.eigh(rho)
            u = u[:, -dR:]
            psi = np.tensordot(psi, u.conj(), axes=[[2], [0]])

        psi /= np.linalg.norm(psi)
        u, s, v = svd(psi.reshape(D2, D2), full_matrices=False)
        Zp = np.dot(u, v)
        A = np.dot(A, Zp)

        theta = np.dot(Zp.T.conj(), theta)
    # Disentangle the two-site wavefunction
    theta = np.reshape(theta, (dL, dR, mL, mR))  #view TEBD style
    theta = np.transpose(theta, (2, 0, 1, 3))
    theta, U, info = disentangle2(theta,
                                  eps=10 * eps,
                                  max_iter=max_iter,
                                  verbose=verbose)

    #print "S2s:", info['Ss']
    if n != 2.:
        theta, Un, info = disentangleCG(theta,
                                        n=n,
                                        eps=eps,
                                        max_iter=max_iter,
                                        verbose=verbose,
                                        pt=0.5)
        U = np.dot(Un, U)

    #print "Sns:", info['Ss']

    A = np.tensordot(A,
                     np.reshape(np.conj(U), (dL, dR, dL * dR)),
                     axes=([1], [2]))
    #This code further truncates
    """
    theta, pipe = group_legs(theta, [[1], [0, 2, 3]])
    u, sv, info_L = svd_theta(theta, truncation_par={'p_trunc': 3e-16})
    dL = u.shape[1]
    theta = sv.reshape((-1, mL, dR, mR)).transpose([1, 0, 2, 3])
    A = np.tensordot(A, u, axes=[[1], [0]]).transpose([0, 2, 1])

    theta, pipe = group_legs(theta, [[2], [0, 1, 3]])
    u, sv, info_R = svd_theta(theta, truncation_par={'p_trunc': 3e-16})
    dR = u.shape[1]
    theta = sv.reshape((-1, mL, dL, mR)).transpose([1, 2, 0, 3])
    A = np.tensordot(A, u, axes=[[2], [0]])
    """

    theta = np.transpose(theta, [1, 0, 2, 3])
    theta = np.reshape(theta, (dL * mL, dR * mR))

    X, s, Z, chi_c, trunc_bond = svd_theta_UsV(theta,
                                               truncation_par['chi_max'],
                                               p_trunc=3e-16)
    if flag:
        local_savefile(theta)
        raise ValueError("split psi")


    S = np.reshape(X, (dL, mL, chi_c))
    S = S * s

    B = np.reshape(Z, (chi_c, dR, mR))
    B = np.transpose(B, (1, 0, 2))

    """Psin = np.tensordot(A,
                        np.tensordot(S, B, axes=[[2], [1]]),
                        axes=[[1, 2], [0, 2]])
    print "|dPsi|^2",  np.linalg.norm(Psin - Psi)**2, trunc_leg + trunc_bond
    """
    #TODO: technically theses errors are only good to lowest order I think
    info = {
        'error': trunc_leg + trunc_bond,
        'd_error': trunc_leg,
        's_Lambda': s
       # 'eval': info['eval']
    }
    return A, S, B, info


def renyi(s, n):
    """n-th Renyi entropy from Schmidt spectrum s
	"""
    s = s[s > 1e-16]
    if n == 1:
        return -2 * np.inner(s**2, np.log(s))
    elif n == 'inf':
        return -2 * np.log(s[0])
    else:
        return np.log(np.sum(s**(2 * n))) / (1 - n)


def Sn(psi, n=1):
    """ Given wf. psi, returns spectrum s & nth Renyi entropy via SVD
	"""
    chi = psi.shape
    theta = psi.reshape((chi[0] * chi[1], chi[2] * chi[3]))
    s = np.linalg.svd(theta, compute_uv=False)

    S = renyi(s, n)

    return s, S


def dSn(psi, n=1):
    """ Returns H = dS / dU  and S for nth Renyi
		psi = El, l, r, Er
		
		Returns Sn, dSn
	"""
    chi = psi.shape
    theta = psi.reshape((chi[0] * chi[1], chi[2] * chi[3]))
    X, s, Z = np.linalg.svd(theta, full_matrices=False)

    if n == 'inf':
        r = np.zeros_like(s)
        r[0] = 1 / s[0]
        S = -2 * np.log(s[0])
    elif n == 1:
        p = s**2
        lp = np.log(p)
        r = s * lp * (s > 1e-10)
        S = -np.inner(p, lp)
    else:
        tr_pn = np.sum(s**(2 * n))
        ss = s**(2 * n - 1.)
        r = ss / tr_pn * n / (n - 1)
        S = np.log(tr_pn) / (1 - n)
    #print S, s, r, n
    Epsi = np.dot(X * r, Z).reshape(chi)
    dS = np.tensordot(psi, Epsi.conj(), axes=[[0, 3], [0, 3]])
    dS = dS.reshape((chi[1] * chi[2], -1))
    #dS = -1j*(dS - dS.conj().T)
    dS = (dS - dS.conj().T)
    return S, dS

    #exp[dS]


def U2(psi):
    """Entanglement minimization via 2nd Renyi entropy
	
		psi.shape = mL, dL, dR, mR
		
		Returns  S2 and polar disentangler U
	"""
    chi = psi.shape

    rhoL = np.tensordot(psi, psi.conj(), axes=[[2, 3], [2, 3]])
    dS = np.tensordot(rhoL, psi, axes=[[2, 3], [0, 1]])
    dS = np.tensordot(psi.conj(), dS, axes=[[0, 3], [0, 3]])
    dS = dS.reshape((chi[1] * chi[2], -1))
    s2 = np.trace(dS)

    X, Y, Z = np.linalg.svd(dS)
    return -np.log(s2), (np.dot(X, Z).T).conj()


def disentangle2(psi, eps=1e-6, max_iter=120, verbose=0):
    """Disentangles a wavefunction via 2-renyi polar iteration
		
		Returns psiD, U, Ss
		
		psiD = U psi
		
		Ss are Renyi2 at each step
	"""

    Ss = []
    chi = psi.shape
    U = np.eye(chi[1] * chi[2], dtype=psi.dtype)
    m = 0
    go = True
    #return(psi, U, {"Ss":Ss})
    while m < max_iter and go:
        s, u = U2(psi)
        U = np.dot(u, U)
        u = u.reshape((chi[1], chi[2], chi[1], chi[2]))
        psi = np.tensordot(u, psi, axes=[[2, 3], [1,
                                                  2]]).transpose([2, 0, 1, 3])
        Ss.append(s)
        if m > 1:
            go = Ss[-2] - Ss[-1] > eps
        m += 1

    if verbose:
        print("disentangle2 evaluations:", m, "dS", -np.diff(Ss))

    return psi, U, {'Ss': Ss}


def beta_PR(g1, g2):  #Polak-Ribiere
    return np.max([0, np.real(np.vdot(g2, g2 - g1) / np.vdot(g1, g1))])


def beta_FR(g1, g2):  #Fletcher-Reeves
    return np.real(np.vdot(g2, g2) / np.vdot(g1, g1))


def beta_SD(g1, g2):  #Steepest descent
    return 0.


def disentangleCG(psi,
                  n=2,
                  eps=1e-6,
                  max_iter=30,
                  verbose=0,
                  beta=beta_PR,
                  pt=0.5):
    """Disentangles a wavefunction.
	Returns psi, U, Ss
	"""

    mod = disentangler_model(psi.copy(), n=n, pt=pt)

    res = cg(mod,
             max_iter=max_iter,
             ls_args={'gtol': 0.05},
             eps=eps,
             verbose=verbose,
             beta=beta)
    Ss = res['F']
    dSs = [np.linalg.norm(h) for h in res['Fp']]

    if 0 or verbose:
        print("Iter, Eval:", res['iter'], mod.total)

    if verbose > 1:
        plt.subplot(1, 3, 2)
        plt.plot(Ss, '.-')
        plt.subplot(1, 3, 3)
        plt.plot(dSs, '.-')
        plt.yscale('log')
        #plt.tight_layout()

    return mod.psi_k, mod.U, {'Ss': Ss, 'eval': mod.total}


def cg(model,
       eps=1e-6,
       max_iter=30,
       ls_args={},
       beta=beta_PR,
       verbose=0,
       fmin=None):
    """Conjugate gradient for n-th renyi entropy
	
		eps: target error in |df|
		beta: beta_PR, FR or SD      
	"""

    F = [model.F()]  #Objective
    Fp = [model.Fp()]  #Directional derivative at kth iterate
    H = [0]  #Search dir initiated at kth iterate
    Beta = []
    k = 0
    ts = [0.5]

    while np.linalg.norm(Fp[-1]) > eps and k < max_iter:
        if k > 0 and np.linalg.norm(F[-1] - F[-2]) < eps:
            break

        if k > 0:
            #b1 = beta(model.ls_transport(Fp[-2]), Fp[-1])
            #b2 = beta(Fp[-2], Fp[-1])
            #print b1/b2, np.linalg.norm(model.ls_transport(Fp[-2]) - Fp[-2])
            Beta.append(beta(model.ls_transport(Fp[-2]), Fp[-1]))
        else:
            Beta.append(0.)

        Hnew = -Fp[-1] + Beta[-1] * H[-1]
        if k > 0:
            descent = np.vdot(
                Hnew, Fp[-1]).real  #Restart CG if not a descent direction
            if descent > 0.:
                Beta[-1] = 0.
                Hnew = -Fp[-1]

        H.append(Hnew)
        xk, Gk, pk, = model.ls_setup(H[-1])
        t0 = np.mean(ts[-10:])
        #t0 = ts[-1]
        #print Gk[0], np.vdot(H[-1], Fp[-1]).real
        #Minimize along x = x0 + a d,   input: F0, x0, G0, d, F, G
        MT = ls.pymswolfe.StrongWolfeLineSearch(F[-1],
                                                xk,
                                                Gk,
                                                pk,
                                                model.ls_F_Fp,
                                                stp=t0,
                                                **ls_args)
        MT.search()

        if verbose > 3:
            print("Starting |Fp|:", np.linalg.norm(Fp[-1]))
            print("----- LS --t0-beta-", t0, Beta[-1])
            print("X", MT.xs)
            print("F", MT.fs)
            print("dF/dt", MT.slopes)
            print()
            print("Final t, F, dF/dt", MT.stp, MT.fs[-1], MT.slopes[-1])
            print()

        model.ls_finalize(MT.xs[-1])
        F.append(MT.fs[-1])
        Fp.append(model.Fp())
        ts.append(MT.stp)
        #if verbose:
        #	print F[-1], Fp[-1], "|",
        k += 1

    if verbose > 2:
        plt.subplot(1, 2, 1)
        if fmin == None:
            fmin = F[-1]
        plt.plot([f - fmin + 1e-16 for f in F], '-o')
        plt.yscale('log')
        plt.title('F')

        plt.subplot(1, 2, 2)
        plt.plot([np.linalg.norm(fp) for fp in Fp], '-o')
        plt.yscale('log')
        plt.title('|dF|')
        plt.show()
    #print "CG:", len(Fp), [np.linalg.norm(fp) for fp in Fp]

    return {'F': F, 'Fp': Fp, 'ts': ts, 'iter': k}


"""	model: encode context and objective function for a sequence of conjugate gradient line searches
	
	At iteration k the model is at a "base point" I will denote by 'xk'. Line searches from the basepoint are of the form
	
		F(xk + t*Hk),  dF/dt = F'.Hk
	
	However, for convenience


"""


class disentangler_model(object):
    def __init__(self, psi, n, pt=0.5):

        self.n = n  #Renyi index
        self.psi_k = psi.copy()
        chi = self.chi = psi.shape
        self.U = np.eye(chi[1] * chi[2], dtype=psi.dtype)  #Unitary accumulated
        self.total = 0
        self.pt = pt
        self.F_k, self.Fp_k = dSn(self.psi_k, self.n)

    def ls_setup(self, H):
        """Get ready for a line search U(-i t H) S
			ls_F(t) : Fitness of U(-i t H) S
			ls_Fp(t) : partial_t F(t) (directional derivative)
		"""

        if H.dtype == np.float:
            self.real = True
        else:
            self.real = False

        self.H = H
        self.Id = np.eye(H.shape[0], dtype=H.dtype)
        self.l, self.w = np.linalg.eigh(-1j * H)

        chi = self.chi
        self.psi_k = self.psi_k.transpose([1, 2, 0,
                                           3]).reshape([chi[1] * chi[2], -1])
        self.finalized = False
        self.t = 0
        return np.array([0.]), np.array([np.real(np.vdot(self.Fp_k, self.H))
                                         ]), np.array([1.])

    def ls_F_Fp(self, t):
        """ ls_F(t) : Fitness of U(i t H) S
			
			ls_Fp(t) : partial_t F(t) (directional derivative)
		"""
        t = t[0]
        chi = self.chi

        u = np.dot(self.w * np.exp(1j * self.l * t), self.w.T.conj())
        if self.real:
            u = u.real
        psi_t = np.dot(u, self.psi_k)

        #(1 - t*H/2)^{-1} (self.Id + (t/2)*H)
        #dH = (t/2)*self.H
        #psi_t = np.dot(self.Id + dH, self.psi_k)
        #psi_t = np.linalg.solve(self.Id - dH, self.psi_k)
        #u = None

        psi_t = psi_t.reshape([chi[1], chi[2], chi[0], chi[3]])
        psi_t = psi_t.transpose([2, 0, 1, 3])

        #Calculate derivative
        S, dS = dSn(psi_t, self.n)
        self.total += 1
        self.t = t
        self.F_t, self.Fp_t, self.u_t, self.psi_t = S, dS, u, psi_t
        #print "C", t, np.vdot(dS, self.H)
        return S, np.array([np.real(np.vdot(dS, self.H))])

    def ls_finalize(self, t):
        """ S_{k+1} = U(-i t H) S_k"""
        t = t[0]

        #dH = (t/2)*self.H
        #self.u_t = np.linalg.solve(self.Id - dH, self.Id + dH)

        if t != self.t:
            print(t, self.t)
            raise NotImplemented
        if t != 0:
            self.F_k, self.Fp_k, self.psi_k = self.F_t, self.Fp_t, self.psi_t
            self.psi_k /= np.linalg.norm(self.psi_k)
            self.U = np.dot(self.u_t, self.U)
        else:
            chi = self.chi
            psi_k = self.psi_k.reshape([chi[1], chi[2], chi[0], chi[3]])
            self.psi_k = psi_k.transpose([2, 0, 1, 3])

        u = np.dot(self.w * np.exp(1j * self.l * t * self.pt), self.w.T.conj())
        #print "Ut", np.linalg.norm(self.l*t*self.pt), np.linalg.norm(u - np.eye(u.shape[0]))
        if self.real:
            u = u.real

        #dH = self.pt*dH
        #u = np.linalg.solve(self.Id - dH, self.Id + dH)
        #print "dU", t, np.linalg.norm(np.dot(self.w*np.exp(self.l*t*self.pt), self.w.T.conj()) - u), np.linalg.norm(np.eye(u.shape[0]) - u), self.l*t
        self.sqrtu = u

        self.finalized = True

    def ls_transport(self, v):
        """Transport a tangent vector 'v' in u(N) from S_k ---> S_{k+1}"""
        if not self.finalized:
            raise ValueError
        #return v
        return np.dot(np.dot(self.sqrtu, v), self.sqrtu.T.conj())

    def F(self):
        if self.F_k is None:
            self.F_k, self.Fp_k = dSn(self.psi_k, self.n)
            self.total += 1
        return self.F_k

    def Fp(self):
        if self.Fp_k is None:
            self.F_k, self.Fp_k = dSn(self.psi_k, self.n)
            self.total += 1
        return self.Fp_k


def split_psi_(Psi,
               dL,
               dR,
               truncation_par={
                   'chi_max': 32,
                   'p_trunc': 1e-6
               },
               verbose=0,
               n=0.5,
               eps=1e-6,
               max_iter=20):
    """ Given a tripartite state psi.shape = d x mL x mR   , find an approximation
	
			psi = A.Lambda
	
		where A.shape = dL x dR x d  is an isometry that "splits" d --> dL x dR and Lambda.shape = mL dL dR mR 
		is a 2-site TEBD-style wavefunction of unit norm and maximum Schmidt rank 'chi_max.'
	
		The solution for Lambda is given by the MPS-type decomposition

			Lambda_{mL dL dR mR} = \sum_eta  S_{dL mL eta} B_{dR eta mR}

		where 1 < eta <= chi_max, and Lambda has unit-norm

		Arguments:
		
			psi:  shape= d, mL, mR
			
			dL, dR: ints specifying splitting dimensions (dL,dR maybe reduced to smaller values)
			
			truncation_par  = {'chi_max':, 'p_trunc':}  truncation parameters for the SVD of Lambda; p_trunc is acceptable discarded weight
			
			eps: precision of iterative solver routine
			
			max_iter: max iterations of routine (through warning if this is reached!)
		
			verbose:
				
				
		Returns:
		
			A: d x dL x dR
			S: dL x mL x eta
			B: dR x eta x mR
		
			info = {} , a dictionary of (optional) errors
			
				'error': 'trunc_leg','trunc_bond'
	"""

    d, mL, mR = Psi.shape
    #print "d: mL, dL, dR, mR", d, mL, dL, dR, mR
    X, y, Z, D2, trunc_leg = svd_theta_UsV(Psi.reshape((-1, mL * mR)), d,
                                           3e-16)

    PsiTD = np.dot(Z.T.conj() * y, Z).reshape(
        (mL, mR, mL, mR)).transpose([2, 0, 1, 3])
    U = np.dot(Z.T.conj(), X.T.conj())
    # Disentangle the two-site wavefunction
    #print "PsiTD:", PsiTD.shape

    theta, Un, info = disentangle2(PsiTD,
                                   eps=eps,
                                   max_iter=max_iter,
                                   verbose=verbose)
    U = np.dot(Un, U)
    #print info['Ss']
    if n != 2.:
        theta, Un, info = disentangleCG(theta,
                                        n=n,
                                        eps=eps,
                                        max_iter=max_iter,
                                        verbose=verbose,
                                        pt=0.5)
        U = np.dot(Un, U)
    #print info['Ss']
    U = U.reshape((mL, mR, d))

    #print "dPsi1", np.linalg.norm( theta - np.tensordot(U, Psi, axes = [[2], [0]]).transpose([2, 0, 1, 3]))

    theta, pipe = group_legs(theta, [[1], [0, 2, 3]])
    x, y, z = svd(theta, full_matrices=False)
    #print "sL", y, group_legs(U.conj(), [[0], [1, 2]]).T
    #print np.sqrt(1 - np.linalg.norm(y[:dL])**2),
    x = x[:, :dL]
    y = y[:dL]
    z = z[:dL, :]
    y /= np.linalg.norm(y)
    dL = len(y)
    theta = ((z.T * y).T)
    theta = theta.reshape((dL, mL, mR, mR)).transpose([1, 0, 2, 3])
    #print "PsiTD dL:", theta.shape

    U = np.tensordot(x.conj(), U, axes=[[0], [0]])
    #print "dPsi2", np.linalg.norm( theta - np.tensordot(U, Psi, axes = [[2], [0]]).transpose([2, 0, 1, 3]))

    theta, pipe = group_legs(theta, [[2], [0, 1, 3]])
    x, y, z = svd(theta, full_matrices=False)
    #print "sR", y, 1 - np.linalg.norm(y[:dR])
    #print np.sqrt(1 - np.linalg.norm(y[:dR])**2),
    x = x[:, :dR]
    y = y[:dR]
    z = z[:dR, :]
    y /= np.linalg.norm(y)
    dR = len(y)
    theta = ((z.T * y).T).reshape((dR, mL, dL, mR)).transpose([1, 2, 0, 3])
    U = np.tensordot(x.conj(), U, axes=[[0], [1]]).transpose([1, 0, 2])

    #print "dPsi3", np.linalg.norm( theta - np.tensordot(U, Psi, axes = [[2], [0]]).transpose([2, 0, 1, 3]))
    #print "PsiTD dR:", theta.shape
    def test(t):
        tt = np.dot(t.T.conj(), t)
        return np.linalg.norm(tt - np.eye(tt.shape[0]))

    #A, pipe = group_legs(U.conj(), [[0], [1, 2]]).T
    #U, pipe = group_legs(U, [[0, 1], [2, 3]])
    #U = np.dot(np.dot(U, Z.T.conj(), X.T.conj()).T
    #print "XZ:", test(np.dot(X, Z))
    #print "XZ':", test(np.dot(X, Z).T.conj())
    #print "U:", test(U.T.conj())
    #A = np.dot(np.dot(X, Z), U.T.conj())
    #print A.shape
    #print np.dot(A.T.conj(), A)
    U, pipe = group_legs(U, [[0, 1], [2]])
    x, y, z = svd(U, full_matrices=False)
    #print x.shape, 1-y, z.shape
    U = np.dot(x, z)
    A = U.T.conj()
    U = ungroup_legs(U, pipe)
    #print "AA:", test(A)
    thn = np.tensordot(U, Psi, axes=[[2], [0]]).transpose([2, 0, 1, 3])
    #print "dPsi4", np.linalg.norm( theta - np.tensordot(U, Psi, axes = [[2], [0]]).transpose([2, 0, 1, 3]))
    #print np.tensordot(U, A, axes = [[2], [0]]).reshape((dL*dR, dL*dR))
    #print "th", theta.shape
    #thn = np.tensordot(A.conj(), Psi, axes = [[0], [0]]).reshape((dL, dR, mL, mR))
    #thn = np.transpose(thn, [2, 0, 1, 3])
    #thn = thn/np.linalg.norm(thn)
    #print "dtheta", np.linalg.norm(thn - theta)
    #s, sn = Sn(thn, n)
    thn, Un, info = disentangleCG(thn,
                                  n=n,
                                  eps=eps,
                                  max_iter=max_iter,
                                  verbose=verbose,
                                  pt=0.5)
    #print info['Ss']
    #print "Final:", info['Ss'][0], '-->', info['Ss'][-1]
    A = np.dot(A, Un.T.conj())
    #print np.tensordot(U, A, axes = [[2], [0]]).reshape((dL*dR, dL*dR))
    #print Un.T.conj()
    U, pipe = group_legs(U, [[0, 1], [2]])
    U = np.tensordot(Un, U, axes=[[1], [0]])
    U = ungroup_legs(U, pipe)

    #print np.tensordot(U, A, axes = [[2], [0]]).reshape((dL*dR, dL*dR))
    #print "dPsi5", np.linalg.norm( thn - np.tensordot(U, Psi, axes = [[2], [0]]).transpose([2, 0, 1, 3]))

    theta, pipe = group_legs(thn, [[1, 0], [2, 3]])
    X, s, Z, chi_c, trunc_bond = svd_theta_UsV(theta,
                                               truncation_par['chi_max'],
                                               p_trunc=3e-16)
    #print np.sqrt(trunc_bond)
    S = np.reshape(X, (dL, mL, chi_c))
    S = S * s

    B = np.reshape(Z, (chi_c, dR, mR))
    B = np.transpose(B, (1, 0, 2))

    #A: d x dL x dR
    #S: dL x mL x eta
    #B: dR x eta x mR

    A = A.reshape((d, dL, dR))
    #print np.tensordot(U, A, axes = [[2], [0]]).reshape((dL*dR, dL*dR))

    Psin = np.tensordot(A,
                        np.tensordot(S, B, axes=[[2], [1]]),
                        axes=[[1, 2], [0, 2]])

    print("dPsi", np.linalg.norm(Psin - Psi))

    #TODO: technically theses errors are only good to lowest order I think
    info = {
        'error': trunc_leg + trunc_bond,
        'd_error': trunc_leg,
        's_Lambda': s
    }
    return A, S, B, info
