import sys, os, inspect, numpy as np, scipy.linalg as SL, scipy.sparse as SP, scipy.sparse.linalg as SPL, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cycler import cycler
from math import pi
from .ode_solver import *

class Particle:
    def __init__(self, V, bounds, dr, n, dt, m, hbar):
        self.bounds, self.dt, self.m, self.hbar = bounds, dt, m, hbar
        self.n, self.dr = get_n_h(*bounds, n=n, h=dr, bounds=False)
        self.n_tot, self.dim = np.prod(np.array(self.n)), len(self.n)
        self.r = [np.linspace(*bounds[2*i:2*i+2], self.n[i]+2)[1:-1] for i in range(self.dim)]
        self.R = np.meshgrid(*self.r)
        self.norm_coef = np.sqrt(np.prod(self.dr))
        self.dt = m*min(self.dr)**2/(10*hbar) if dt is None else dt
        self.is_separable = False
        if V is None:
            self.v = np.zeros(self.n_tot, dtype=float)
            if self.dim > 1:
                self.V_iter = self.dim*[None]
                self.is_separable = True
        elif hasattr(V, '__iter__'):
            self.V_iter = V
            self.v = sum([V[i](self.R[i]) for i in range(self.dim)]).flatten()
            self.is_separable = True
        else:
            self.v = V(*self.R).flatten()
        if not self.is_separable:
            self.V = SP.dia_matrix((self.v, 0), shape=(self.n_tot, self.n_tot))
            self.T = -hbar**2/(2*m)*Laplace(*self.r)
            self.H = self.T+self.V
            self.A = (SP.identity(self.n_tot)+1j*self.dt/(2*self.hbar)*self.H).tocsr()
            self.B = (SP.identity(self.n_tot)-1j*self.dt/(2*self.hbar)*self.H).tocsr()
        else:
            self.Q = [Particle_1D(V=self.V_iter[i], bounds=self.bounds[2*i:2*i+2], n=self.n[i]+1) for i in range(self.dim)]
            self.V = lambda: SP.dia_matrix((self.v, 0), shape=(self.n_tot, self.n_tot))
            self.T = lambda: -hbar**2/(2*m)*Laplace(*self.r)
            self.H = lambda: self.T()+self.V()
            self.A = None
            self.B = None
        self.is_diagonalized = False
        self.cond_is_set = False
        self.default_method = None
        self.scale=1
        self.shape = np.flip(self.n)
        self.frames_temp = []
        self.frames = []

    def diagonalize(self, k=None):
        if self.dim == 1:
            '''
            eigh_tridiagonal is faster than eigsh in most cases, so we just keep the first 
            '''
            e, F = SL.eigh_tridiagonal(self.H.diagonal(), self.H.diagonal(1))
            if k is not None:
                self.e, self.F = e[:k], F[:,:k]
            else:
                self.e, self.F = e, F
        elif self.is_separable:
            '''
            H = H_x kronsum H_y.
            eigenvals = E_i + E_j, eigenvectors = f_j kron f_i
            If the eigenvector matrix F_y kron F_x is small enough (eg <= 20000)
            then we might have enough RAM to support it. Otherwise, we just save
            F_x and F_y separately.
            In any case, if we only want the first k eigenvectors, we need to fully diagonalize
            in each dimension, and then sort the column order of the resulting F_y kron Fx matrix,
            in correspondance to the sorted energy array order of elements.
            '''
            diag_data = [q.diagonalize() for q in self.Q]
            E_q, F_q = [data[0] for data in diag_data], [data[1] for data in diag_data]
            e = sum(np.meshgrid(*E_q)).flatten()
            self.e_argsort = e.argsort()
            self.e = e[self.e_argsort]
            if np.prod([f.shape[0] for f in F_q]) <= 20000:
                self.F = np.array([1.])
                for f in F_q[::-1]:
                    self.F = np.kron(self.F, f)
                self.F = self.F[:,self.e_argsort]
                if k is not None:
                    '''
                    We only need the first k columns from the eigenvector matrix
                    '''
                    self.F = self.F[:,:k]
                    self.e = e[:k]
            else:
                '''
                These will be used in case we call the n-th eigenstate
                '''
                self.F = None
                self.E_q, self.F_q = E_q, F_q
        else:
            self.e, self.F = SL.eigh(self.H.toarray()) if k is None else SPL.eigsh(self.H, k=k, which='SM')
        if self.cond_is_set and self.F is not None:
            coefs = self.F.T.dot(self.psi0)
            self.U = self.F*coefs
        self.is_diagonalized = True
        return self.e, self.F

    def set_cond(self, psi_0):
        '''
        Sets the initial condition for a given function psi_0(x)
        '''
        self.psi0 = psi_0(*self.R).flatten()
        self.cn_t, self.cn_psi = 0, self.psi0.copy()
        self.cond_is_set = True
        if self.is_diagonalized:
            coefs = self.F.T.dot(self.psi0) #c_n = <f_n|Ψ_0>
            self.U = self.F*coefs

    def psi(self, t, method = 'default'):
        if method == 'eig':
            return self.psi_eig(t)*self.scale**0.5
        elif method == 'cn':
            return self.psi_cn(t)*self.scale**0.5
        elif method == 'default':
            return self.psi(t, method = self.default_method)
        else:
            raise ValueError('No such available method: '+str(method))

    def psi_eig(self, t):
        if not self.is_diagonalized:
            self.diagonalize()
        return self.U.dot(np.exp(-1j*self.e*t/self.hbar)).reshape(self.shape)

    def psi_cn(self, t):
        if t < self.cn_t:
            t_, psi = 0, self.psi0.copy()
        else:
            t_, psi = self.cn_t, self.cn_psi
        nt = int((t-t_)/self.dt)
        for _ in range(nt):
            psi = SPL.spsolve(self.A, self.B.dot(psi))
        self.cn_t = t_+nt*self.dt
        self.cn_psi = psi
        return psi.reshape(self.shape)

    def pr(self, t, method = 'default'):
        '''
        For a given initian condition, returns the propability density at time 't'
        '''
        return np.abs(self.psi(t, method=method))**2

    def f(self, k):
        if self.is_separable and self.F is None:
            '''
            ONLY works in two dimensions right now. TODO: generalize
            '''
            l = self.e_argsort[k]
            m, n = l % self.nx, l // self.ny
            return np.kron(self.F_q[1][:,n], self.F_q[0][:,m]).reshape(self.shape) / self.norm_coef
        else:
            return self.F[:,k].reshape(self.shape) / self.norm_coef

    def energy_levels(self, k=None):
        '''
        almost same as self.diagonalize, but returns eigenvals only,
        and does not store them anywhere.
        '''
        if self.dim == 1:
            e_levels = SL.eigh_tridiagonal(self.H.diagonal(), self.H.diagonal(1), eigvals_only = True)
            if k is not None:
                e_levels = e_levels[:k]
        elif self.is_separable:
            e_levels = np.sort(sum(np.meshgrid(*[SL.eigh_tridiagonal(q.H.diagonal(), q.H.diagonal(1), eigvals_only = True) for q in self.Q])).flatten())[:k]
        else:
            e_levels = np.sort(SL.eigh(self.H.toarray(), eigvals_only=True) if k is None else SPL.eigsh(self.H, k=k, return_eigenvectors=False, which='SM'))
        return e_levels

    def expected_value(self, operator, t=None, n=1, psi = None, method='default'):
        '''
        <Q> = <ψ(t)|Q|ψ(t)>
        '''
        if psi is None:
            ket = self.psi(t, method=method).flatten()
        else:
            ket = psi.copy()
        bra = ket.conj()
        if self.is_separable:
            operator = operator()
        for _ in range(n):
            ket = operator.dot(ket)
        return bra.dot(ket).real * self.norm_coef**2
    
    def set_wave_packet(self, x_mean, p_mean, sx):
        '''
        e.g:
        In 1D: x_mean = 5, p_mean = 4, sx=1
        In 2D: x_mean = [5, 1], p_mean = [4, 2], sx = [1, 3]
        '''
        self.set_cond(lambda *r: wave_packet(r if self.dim != 1 else r[0], 0, x_mean, p_mean, sx, self.m, self.hbar))

    def dispersion(self, operator, n=1, t=None, psi = None, method='default'):
        '''
        Returns <Q^2> - <Q>^2
        '''
        if psi is None:
            psi = self.psi(t, method=method).flatten()
        return self.expected_value(operator, n=2*n, psi=psi) - self.expected_value(operator, n, psi=psi)**2
        

class Particle_1D(Particle):
    def __init__(self, V=None, bounds=None, dx=None, n=None, dt = None, m=1, hbar=1):
        super().__init__(V=V, bounds=bounds, dr=None if dx is None else (dx,), n=None if n is None else (n,), dt=dt, m=m, hbar=hbar)
        self.methods = dict(eig=2, cn=3)
        self.funcs = [lambda x, t: [], lambda x, t: self.v, lambda x, t: self.pr(t, method='eig'), lambda x, t: self.pr(t, method='cn')]
        self.x, = self.r
        self.dx, = self.dr
        self.x_data = [[], self.x, self.x, self.x]
        self.labels = ['', 'Potential', 'Propagator (dx = %s)' % float('%.3g' % self.dx), 'C-N (dx = %s, dt = %s)'  %(float('%.3g' % self.dx), float('%.3g' % self.dt))]
        self._is_active = [True, V is not None, True, False]
        self.x_view, self.y_view = bounds, None
        self.experiment_is_set = False
        self.default_method = 'eig'
        self.X = spdiag(self.x)
        self.P = -1j*hbar*Diff(self.x)
        

    def compare_to(self, func, label, *params, **x):
        '''
        e.g compare_to(func, label)
        func must have x or t parameters.
        Any other arguments are included in "*params"
        '''
        args = list(inspect.signature(func).parameters)
        if 'self' in args:
            args = list(args)
            args.remove('self')
            args = tuple(args)
        if 'x' not in args and 't' in args:
            assert args[0] == 't'
            self.funcs.append(lambda x, t: self.scale*func(t, *params))
        elif 't' not in args and 'x' in args:
            assert args[0] == 'x'
            self.funcs.append(lambda x, t: self.scale*func(x, *params))
        elif 'x' in args and 't' in args:
            assert 'x' == args[0] and 't' == args[1]
            self.funcs.append(lambda x, t: self.scale*func(x, t, *params))
        else:
            raise ValueError('Function does not depend on x or t')
        if not x:
            self.x_data.append(self.x)
        else:
            if 'x' not in x.keys() or len(x) > 1:
                raise ValueError('x values not specified or more keyword arguments given')
            else:
                self.x_data.append(x['x'].copy())
        self.labels.append(label)
        self._is_active.append(True)

    def uncompare_all(self):
        i = len(self.methods)+2
        self.funcs = self.funcs[:i]
        self.x_data = self.x_data[:i]
        self.labels = self.labels[:i]
        self._is_active = self._is_active[:i]

    def _activate_plots(self, *method):
        if method[0] == 'default' and len(method) == 1:
            self._activate_plots(self.default_method)
        elif method[0] == 'all' and len(method) == 1:
            self._activate_plots(*list(self.methods))
        elif not all([m in self.methods for m in method]):
            raise ValueError('Invalid input in numeric method')
        else:
            for m in self.methods:
                if m in method:
                    self._is_active[self.methods[m]] = True
                else:
                    self._is_active[self.methods[m]] = False

    def view(self, x=None, y=None):
        '''
        Sets the xlim and ylim for future animating ot plotting
        '''
        if x is not None:
            self.x_view = x
        if y is not None:
            self.y_view = y

    def set_V_label(self, label):
        self.labels[1] = label

    def _get_plots(self, ax):
        ax.set(xlim=self.x_view, ylim=self.y_view, title = (6*' ').join(['<T> = %s' % float('%.5g' % self.expected_value(self.T, 0, psi=self.psi0)),'bounds = '+str(self.bounds)]))
        ax.set_prop_cycle(cycler(color='wkbrgcm'))
        plots, labels, ind = [], [], []
        for i in range(len(self.funcs)):
            if self._is_active[i]:
                plots += ax.plot([], [], linewidth=0.5)
                labels.append(self.labels[i])
                ind.append(i)
        return plots, labels, ind

    def plot(self, t, method='default', ax=None):
        self._activate_plots(method)
        self.labels[0] = 't = %s s' % float('%.3g' % t)
        ax_is_None = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax_is_None = True
        plots, labels, ind = self._get_plots(ax)
        for i in range(len(plots)):
            plots[i].set_data(self.x_data[ind[i]], self.funcs[ind[i]](x=self.x_data[ind[i]], t=t))
        ax.legend(plots, labels)
        if ax_is_None:
            return fig, ax
        else:
            return ax

    def animate(self, t, real_time, fps=10, method='default', save=None):
        def update(i):
            labels[0] = 't = %s s' % float('%.3g' % t_[i])
            for j in range(len(plots)):
                plots[j].set_data(*self.frames_temp[i][j])
            return *plots, ax.legend(plots, labels)

        t0, t1, n = *t, int(real_time*fps+1)
        self._activate_plots(method)
        fig, ax = plt.subplots(figsize=(10, 5))
        plots, labels, ind = self._get_plots(ax)
        t_ = np.linspace(t0, t1, n)
        if not self.frames_temp:
            for ti in t_:
                os.system('clear')
                print('Animating:', round(100*(ti-t0)/(t1-t0), 2), '%')
                self.frames_temp.append([(self.x_data[ind[i]], self.funcs[ind[i]](x=self.x_data[ind[i]], t=ti)) for i in range(len(plots))])
        res = FuncAnimation(fig, update, frames = np.arange(len(t_)), interval=1000/fps, blit=False, cache_frame_data = False)
        if save is not None:
            res.save(sys.path[0]+'/'+save+'.mp4', dpi=500)
        else:
            plt.show()
        self.frames = self.frames_temp.copy()
        self.frames_temp = []


class Particle_2D(Particle):
    def __init__(self, V=None, bounds=None, dr=None, n=None, dt = None, m=1, hbar=1):
        super().__init__(V=V, bounds=bounds, dr=None if dr is None else dr, n=None if n is None else n, dt=dt, m=m, hbar=hbar)
        self.x, self.y = self.r
        self.dx, self.dy = self.dr
        self.nx, self.ny = self.n
        self.Lx, self.Ly = bounds[1]-bounds[0], bounds[3]-bounds[2]
        self.default_method = 'cn'
        if self.is_separable:
            self.X = lambda: SP.kron(I(self.y), spdiag(self.x))
            self.Y = lambda: SP.kron(spdiag(self.y), I(self.x))
            self.Px = lambda: -1j*hbar*Diff(self.x, d=0, r=self.r)
            self.Py = lambda: -1j*hbar*Diff(self.y, d=1, r=self.r)
        else:
            self.X = SP.kron(I(self.y), spdiag(self.x))
            self.Y = SP.kron(spdiag(self.y), I(self.x))
            self.Px = -1j*hbar*Diff(self.x, d=0, r=self.r)
            self.Py = -1j*hbar*Diff(self.y, d=1, r=self.r)


    def animate(self, t, real_time, fps=10, axes='2d', method = 'cn', save=None):
        def update(i):
            ax.clear()
            if axes == '2d':
                ax.pcolormesh(self.x, self.y, self.frames_temp[i], shading='auto')
                return fig,
            else:
                ax.plot_surface(*self.R, self.frames_temp[i], cmap='viridis', rstride=5, cstride=1, alpha=None)
                return ax, 

        if axes == '2d':
            scale = max(self.Lx, self.Ly)
            lx = 10*self.Lx/scale
            ly = 10*self.Ly/scale
            fig, ax = plt.subplots(figsize=(lx, ly))
        elif axes == '3d':
            ax=plt.axes(projection='3d')
        else:
            raise ValueError('2d or 3d are the only accepted parameters for "axes"')
        t0, t1, n = *t, int(real_time*fps+1)
        t_ = np.linspace(t0, t1, n)
        if not self.frames_temp:
            for ti in t_:
                os.system('clear')
                print('Animating:', round(100*(ti-t0)/(t1-t0), 2), '%')
                self.frames_temp.append(self.pr(ti, method))
 
        res = FuncAnimation(fig, update, frames = np.arange(len(t_)), interval=1000/fps, cache_frame_data = False)
        if save is not None:
            res.save(sys.path[0]+'/'+save+'.mp4', dpi=500)
        else:
            plt.show()
        self.frames = self.frames_temp.copy()
        self.frames_temp = []

class Simulation:
    def __init__(self, bounds, bins, n, x_mean, p_mean, sp, func=None, free_args=[], V=None, F=None, dt=None):
        if not hasattr(x_mean, '__iter__'):
            x_mean, p_mean, sp = [x_mean], [p_mean], [sp]
        self.x0 = np.concatenate(tuple([np.ones(n)*x_mean_ for x_mean_ in x_mean]))
        self.p0 = np.concatenate(tuple([np.random.normal(p_, sp[i], n) for i, p_ in enumerate(p_mean)]))
        self.phi0 = np.zeros(n*len(x_mean))
        self.delta_x = (bounds[1]-bounds[0])/bins
        self.bins = np.linspace(bounds[0]+self.delta_x/2, bounds[1]-self.delta_x/2, bins)
        self.grid = np.linspace(*bounds, bins+1)
        self.n = n*len(x_mean)
        self.reset()
        self.V = V
        self.F = F
        self.dt = dt
        if func is not None:
            self.simu_func = func
            all_args = [arg for arg in list(inspect.signature(func).parameters) if arg != 't']
            self_args, indep_args = [], []
            for arg in all_args:
                if hasattr(self, arg):
                    self_args.append(arg)
                else:
                    indep_args.append(arg)
            self.simu_self_args = {arg: None for arg in self_args}
            self.simu_free_args = {indep_args[i]: free_args[i] for i in range(len(free_args))}
            
        self.experiment_is_set = True
        
    def reset(self):
        self.x, self.p, self.phi, self.t_sim = self.x0.copy(), self.p0.copy(), self.phi0.copy(), 0  #dynamic variables

    def detect(self, t, return_plain=False):
        if t < self.t_sim:
            self.reset()
        if self.F is None:
            for arg in self.simu_self_args:
                self.simu_self_args[arg] = np.copy(eval('self.'+arg))
            self.x, self.p, self.phi, self.t_sim = self.simu_func(t=t, **self.simu_self_args, **self.simu_free_args)
        else:
            self.x, self.p, self.phi, delta_t = dsolve_motion(t=t-self.t_sim, x0=self.x, v0=self.p, phi0=self.phi, V=self.V, F=self.F, dt=self.dt)
            self.t_sim += delta_t
        if return_plain:
            Neff_plain = np.zeros(len(self.bins))
        Neff = np.zeros(len(self.bins))
        ind = self.x.argsort()
        x = self.x[ind]
        phi = self.phi[ind]
        p = self.p[ind]
        unique_bins, ind = np.unique(np.digitize(x, self.grid), return_index=True)
        for i, bin in enumerate(unique_bins[:-1]):
            ph_bin = phi[ind[i]:ind[i+1]]
            p_bin = p[ind[i]:ind[i+1]]
            n_left, n_right = np.sum(p_bin<0), np.sum(p_bin>=0)
            neff_right = np.sum(np.exp(1j*ph_bin[p_bin>=0]))/np.sqrt(n_right) if n_right>0 else 0
            neff_left = np.sum(np.exp(1j*ph_bin[p_bin<0]))/np.sqrt(n_left) if n_left>0 else 0
            Neff[bin-1] = np.abs(neff_left+neff_right)**2/(self.n*self.delta_x)
            if return_plain:
                Neff_plain[bin-1] = (n_left+n_right)/(self.n*self.delta_x)
        if return_plain:
            return Neff, Neff_plain
        else:
            return Neff

class PDE:
    '''
    Crank-Nicolson algorithm for an nth (currently 1d or 2d supported) dimensional linear pde.
    df/dt = p(x, y)*f + q(x, y)*fx + r(x,y)*fxx + u(x,y)*fy + v(x,y)*fyy + S(x, y)  with stationary boundary contitions
    where f = f(x, y, t), fx = df/dx, fy = df/dy
    '''
    def __init__(self, pde, bounds, conditions, n=None, dr=None, dt=None):
        n, dr = get_n_h(*bounds, n=n if hasattr(n, '__iter__') or n is None else [n], h=dr if hasattr(dr, '__iter__') or dr is None else [dr])
        N, N_in, dim = np.prod(n), np.prod(n-2), len(n)
        self.n=n
        r_aug = [np.linspace(*bounds[2*i:2*i+2], ni) for i, ni in enumerate(n if hasattr(n, '__iter__') else[n])]
        r = [ri[1:-1] for ri in r_aug]
        R = np.meshgrid(*r)

        coefs, source = parse_pde(pde)
        L = Operator(*r, *coefs)
        L_aug = Operator(*r_aug, *coefs)
        
        self.all_indices = [np.meshgrid([np.arange(i) for _ in range(dim-1)])[0] for i in n]
        f = np.zeros(N)
        for i in range(dim):
            f[self.boundary(i, 0)] = conditions[2*i]
            f[self.boundary(i, n[i]-1)] = conditions[2*i+1]
        self.bound_term = L_aug.dot(f).reshape(np.flip(n))[tuple([np.s_[1:-1] for _ in range(dim)])].flatten() * dt

        if not hasattr(source, '__call__'):
            self.source = source*np.ones(N_in) * dt
        else:
            self.source = source(*R).flatten() * dt

        self.x, self.X = r[0], R[0]
        if dim > 1: #most likely dim == 1 or 2, unless you can outlive queen of England and wait until this code runs in a 3D grid
            #UPDATE: she actually died, so a 3D implementation might be possible
            self.y, self.Y = r[1], R[1]

        if dt is None:
            self.dt = np.min(dr)**2/10 # we need dt/dx^2 to be small, i dont know how small
        else:
            self.dt = dt

        self.A, self.B = SP.identity(N_in)-dt*L/2, SP.identity(N_in)+dt*L/2
        self.dim, self.n, self.N, self.N_in, self.r, self.dr, self.R, self.L = dim,n,N,N_in,r,dr,R,L

    def set_cond(self, f0):
        self.psi0 = f0(*self.R).flatten()
        self.psi = self.psi0.copy()
        self.t = 0

    def f(self, t):
        if t < self.t:
            t_, psi = 0, self.psi0.copy()
        else:
            t_, psi = self.t, self.psi
        nt = int((t-t_)/self.dt)
        for _ in range(nt):
            psi = SPL.spsolve(self.A, self.B.dot(psi)+self.bound_term)+self.source*self.dt
        self.t = t_+nt*self.dt
        self.psi = psi
        return psi.reshape(np.flip(self.n)-2)

    def boundary(self, var, i):
        return np.ravel_multi_index(self.all_indices[:var]+[i]+self.all_indices[var+1:], self.n, order='F').flatten()

def parse_pde(s):
    coefs = []
    'ft = p f + q fx + r fxx + u fy + v fyy'
    ind = dict(f=0, fx=1, fxx=2, fy=3, fyy=4)
    right = s.split('=')[1] +' '
    term = ''
    coef = ''
    for k, i in enumerate(right):
        if i in ('f', 'x', 'y'):
            term += i
        else:
            if (term in ind) or k == len(right)-1:
                if 'x' in term:
                    coefs.append([coef, 0, term.count('x')])#0 for first variable 'x'
                elif 'y' in term:
                    coefs.append([coef, 1, term.count('y')])#1 for second variable 'y'
                else:
                    coefs.append([coef, -1, 0])
                coef = i
                term = ''
            else:
                coef += term+i
    source = term+coef
    while ' ' in source:
        source = source.replace(' ', '')
    while '+' in source:
        source = source.replace('+', '')
    if source == '':
        source = 0
    else:
        source = eval(source)

    for i in range(len(coefs)):
        while ' ' in coefs[i][0]:
            coefs[i][0] = coefs[i][0].replace(' ', '')
        while '+' in coefs[i][0]:
            coefs[i][0] = coefs[i][0].replace('+', '')
        while '*' in coefs[i][0]:
            coefs[i][0] = coefs[i][0].replace('*', '')
        if coefs[i][0] == '':
            coefs[i][0] = 1
        elif coefs[i][0] == '-':
            coefs[i][0] = -1
        else:
            coefs[i][0] = eval(coefs[i][0])
    return coefs, source


def get_n_h(*a, n=None, h=None, bounds=True):
    if n is not None:
        if hasattr(n, '__iter__'):
            h = []
            for i, n_ in enumerate(n):
                h.append((a[2*i+1]-a[2*i])/n_)
            return np.array(n, dtype=int)+1 if bounds else np.array(n, dtype=int)-1, np.array(h)
        else:
            h = (a[1] - a[0])/n
            return n+1 if bounds else n-1, h
    else:
        if hasattr(h, '__iter__'):
            n = []
            for i, h_ in enumerate(h):
                n.append(int((a[2*i+1]-a[2*i])/h_))
        else:
            n = int((a[1] - a[0])/h)
            if n == 0:
                return -1, 1
        return get_n_h(*a, n=n, h=None, bounds=bounds)

def spdiag(a, n=None):
    if hasattr(a, '__iter__'):
        return SP.dia_matrix((a, 0), shape=(*a.shape, *a.shape))
    else:
        return SP.diags((a,), (0,), shape=(n, n))

def Operator(*args):
    '''
    operator of the form
    p(x, y) + q(x, y)*d/dx + r(x,y)*d^2/dx^2 + u(x,y)*d/dy + v(x,y)*d^2/dy^2 (+ z derivatives, but would not be of much use)
    '''
    r, C = [], []
    for arg in args:
        if isinstance(arg, np.ndarray):
            r.append(arg)
        else:
            C.append(arg)
    R = np.meshgrid(*r)
    L = 0
    for c, x, diff_order in C:
        if hasattr(c, '__call__'):
            L += spdiag(c(*R).flatten()).dot(Diff(r[x], diff_order, d=x, r=r))
        elif c==0:
            continue
        else:
            L += c*Diff(r[x], diff_order, d=x, r=r)
    return L

def Diff(x, k=1, d=None,r=None):
    '''
    returns a differential operator of k order that acts on the x == r[d] variable of an len(r) dimensional space.
    '''
    n = len(x)
    dx = (x[-1]-x[0])/(n-1)
    if k == 0:
        D = I(x)
    elif k == 1:
        D = SP.diags([-1, 1],[-1, 1], (n, n))/(2*dx)
    elif k == 2:
        D = SP.diags([1, -2, 1],[-1,0,1], (n, n))/dx**2
    if d is None or r is None:
        if not (d is None and r is None):
            raise ValueError('d or r parameter is missing')
        else:
            return D
    else:
        T = np.array([1])
        for i, ri in enumerate(r):
            if i == d:
                T = SP.kron(D, T)
            else:
                T = SP.kron(I(ri), T)
        return T

def Laplace(*r):
    '''
    if r = (x,), returns d^2/dx^2
    if r = (x, y), returns d^2/dx^2 + d^2/dy^2
    '''
    if len(r) == 1:
        return Diff(r[0], 2)
    elif len(r) == 2:
        return SP.kronsum(Diff(r[0], 2), Diff(r[1], 2))

def I(x):
    return SP.identity(len(x))

def wave_packet(x, t, x_mean, p_mean, sx, m=1, hbar=1):
    '''
    if x_mean, p_mean and sx are iterable, then if N components, an N-dimensional wave_packet will be considered:
        e.g, wave_packet_2d = wave_packet_x * wave_packet_y
    '''
    if not hasattr(x_mean, '__iter__'):
        sx_t = (sx**2+t**2*hbar**2/(4*m**2*sx**2))**0.5
        k0, y, a = p_mean/hbar, x-(x_mean+p_mean/m*t), np.angle((sx/sx_t-1j*hbar*t/(2*m*sx*sx_t))**0.5)
        theta = k0*(x-x_mean)-hbar*k0**2/(2*m)*t+a+hbar*t*y**2/(8*m*sx**2*sx_t**2)
        return 1/(2*pi*sx_t**2)**(1/4)*np.exp(-y**2/(4*sx_t**2))*np.exp(1j*theta)
    else:
        w = 1
        for i, _ in enumerate(x_mean):
            w *= wave_packet(x[i], t, x_mean[i], p_mean[i], sx[i], m, hbar)
        return w

def wave_packet_abs_sq(x, t, x_mean, p_mean, sx, m=1, hbar=1): #equivalent to abs(wave_packet)**2
    x_t = x_mean + p_mean/m*t
    sx_t = sx*np.sqrt(1+hbar**2*t**2/(4*m**2*sx**4))
    return 1/np.sqrt(2*pi*sx_t**2)*np.exp(-(x-x_t)**2/(2*sx_t**2))

def barrier(U, a, b, lim=100, k=100):
    from math import atanh

    def sech(x):
        return 1/np.cosh(x)
    '''
    U is the height of the barrier
    a is the start of the barrier
    b is the end
    So the length of the barrier is L = b-a
    "lim" argument is a measure of how square the barrier is. Bigger lim means better approximation of the heaviside function

    This function is fully symmetric (parity +1) around x = (b-a)/2 = L/2

    Transmission length:
    We state that the barrier starts at x0 where f(x0) = U/k, and ends at x1 where f(x1) = U - U/k.

    returns V(x), F(x) = -dV/dx, Transmission length
    '''
    
    dx = (atanh(2*(k-1)/k - 1) - atanh(2/k - 1))/lim
    return lambda x: U*(np.tanh(lim*(x-a))+np.tanh(-lim*(x-b)))/2, lambda x: lim*U/2*(sech(lim*(x-b))**2 - sech(lim*(x-a))**2), dx

def double_slit(l, m, h, Lx, Ly):
    '''
    l: width of each slit
    m: length of the wall between the slits
    h: depth of the walls
    returns the potential wall as a function of x, y

     <------------------------- Lx ------------------------->
    ^                                                        |
    |                                                        |
    |                                                        |
    |                                                        |
    |                                                        |
    |                                                        |
                                                             |
    Ly--------------------  l  --m--  l ---------------------|
                                                             |
    |                                                        |
    |                                                        |
    |                                                        |
    |                                                        |
    |                                                        |
     --------------------------------------------------------|
    '''
    def f(x, y):
        x_mid = np.logical_and(x<=Lx/2+m/2, x>=Lx/2-m/2)
        x_left = x<=Lx/2-m/2-l
        x_right = x>=Lx/2+m/2+l
        is_y = np.logical_and(y>Ly/2-h/2, y<Ly/2+h/2)
        return np.where(np.logical_and(np.logical_or(x_left, np.logical_or(x_mid, x_right)), is_y), 100, 0)
    return f

def slit(l, h, Lx, Ly):
    def f(x ,y):
        is_x = np.logical_or(x<=Lx/2-l/2, x>=Lx/2+l/2)
        is_y = np.logical_and(y>=Ly/2-h/2, y<=Ly/2+h/2)
        return np.where(np.logical_and(is_x, is_y), 100, 0)
    return f

def dsolve_motion(t, x0, v0, phi0, V, F, dt):
    # def x_dot(t, x, v, phi):
    #     return v
    # def v_dot(t, x, v, phi):
    #     return F(x)
    # def phi_dot(t, x, v, phi):
    #     return v**2/2 - V(x)
    # if t > 0:
    #     t, x, v, phi = dsolve(sys=[x_dot, v_dot, phi_dot], ics=[0, x0.copy(), v0.copy(), phi0.copy()], h=dt, range=[0, t])
    #     return x[-1], v[-1], phi[-1], t[-1]
    # else:
    #     return x0.copy(), v0.copy(), phi0.copy(), 0
    n = int(t/dt)
    x, v, phi = x0.copy(), v0.copy(), phi0.copy()
    for _ in range(n):
        f1, f2, f3, f4 = F(x), F(x+v*dt/2), F(x+v*dt/2+dt**2/4*F(x)), F(x+v*dt+dt**2/2*F(x+v*dt/2))
        V1, V2, V3, V4 = V(x), V(x+v*dt/2), V(x+v*dt/2+dt**2/4*F(x)), V(x+v*dt+dt**2/2*F(x+v*dt/2))
        k_x = 6*v+dt*(f1+f2+f3)
        k_v = f1+2*f2+2*f3+f4
        k_phi = 3*v**2 - (V1+2*V2+2*V3+V4) + v*dt*(f1+f2+f3) + dt**2/4*(f1**2 + f2**2 + 2*f3**2)
        x += dt/6*k_x
        v += dt/6*k_v
        phi += dt/6*k_phi
    return x, v, phi, n*dt


'''
1D tutorial
'''

'''
FREE PARTICLE

q = Particle_1D(bounds=(-100, 100), n=2000)
q.diagonalize()
q.set_wave_packet(70, 5, 1/2)
q.view(x=(50, 100))
q.animate(t=[0, 10], real_time=5, fps=20, method='all')

plt.plot(q.x, q.f(0))
plt.plot(q.x, q.f(1))
plt.plot(q.x, q.f(2))
plt.show()

q.set_wave_packet(-90, 5, 1/2)
t = np.linspace(0, 25, 100)
s = [q.dispersion(q.X, t=ti) for ti in t]
plt.plot(t, s)
plt.show()


'''


'''
PARTICLE AGAINST SQUARE BARRIER

V = barrier(U=10, a=0, b=10)[0]

q = Particle_1D(V=V, bounds=(-100, 100), n=5000)
q.set_wave_packet(-20, 5, 1/2)
q.view(x=(-50, 50), y=(0, 30))
q.scale=5
q.animate(t=[0, 10], real_time=5, fps=10)
'''
