"""
What to test
    Sparse counter arrays
    cumsum Qa

"""

from __future__ import division, absolute_import
import unittest
from time import time

from numpy.random import randint, rand, permutation
from numpy import array, ones, zeros, allclose, log, unique
from numpy import asfortranarray
import numpy
from sympy.functions.combinatorial.numbers import stirling

from samplers import utils
from samplers.utils import (sample, 
                   stirlingnumbers, 
                   rstick, 
                   stirlingnumbers2, 
                   stirlingnumbers3)

from samplers.fortransamplers import (hdpmm_latents_gibbs_sampler,
                             hdpmm_latents_gibbs_sampler2,
                             hdpmm_latents_gibbs_sampler_alt,
                             hdpmm_latents_gibbs_sampler_par2,
                             sigma_sampler,
                             sigma_sampler2,
                             sigma_sampler_c,
                             sigma_sampler_c2,
                             gdd_sampler,
                             gamma_sampler,
                             polya_concentration_sampler,
                             polya_sampler_am,
                             polya_sampler_bpsi,
                             polya_sampler_bpsi2,
                             polya_sampler_c,
                             polya_sampler_c2,
                             ddirichlet_sampler,
                             tau_sampler_c,
                             tau_sampler)

from samplers.fortranutils import (get_stirling_numbers, 
                          get_normalized_stirling_numbers,
                          get_normalized_stirling_numbers2,
                          sigma_sampler_rnd_vals,
                          sigma_sampler_c_rnd_vals,
                          get_rbeta,
                          get_rgamma,
                          get_rdirichlet,
                          get_random_integers,
                          bisection_sampler,
                          get_latent_rndvals_permutation)

##########################################################################
##########################################################################



verbose = False
time_started = time()

def tic():
    global time_started
    time_started = time()

def toc():
    return time() - time_started

def rbeta(alpha, beta, seed=None):

    if seed is None:
        seed = randint(101, 10001)

    samples = get_rbeta(alpha, beta, seed=seed)

    if len(samples) == 1:
        return samples[0]
    
    return samples

def _sigma_sampler(Smax, S, rnd_vals, m, a, J, K):

    Stirling = stirlingnumbers(Smax)
    sigma = zeros((J, K), dtype=numpy.int)

    for j in xrange(J):
        for k in xrange(K):

            q = zeros(S[j,k]+1)

            for i in xrange(int(S[j, k]+1)):
                q[i] = Stirling[S[j, k], i] * (a*m[k])**i

            sigma[j, k] = sample(q.cumsum(), rnd_vals[j,k])

    return sigma

def _sigma_sampler_c(Smax, S, rnd_seed, c, V):

    rnd_vals = sigma_sampler_c_rnd_vals(S, rnd_seed)

    Stirling = stirlingnumbers(Smax)
    sigma = zeros(V, dtype=numpy.int)

    for v in xrange(V):

            q = zeros(S[v]+1)

            for i in xrange(int(S[v]+1)):
                q[i] = Stirling[S[v], i] * (c/float(V))**i

            sigma[v] = sample(q.cumsum(), rnd_vals[v])

    return sigma


def _gdd_sampler(S, gamma, seed):

    K, V = S.shape

    sigmasum = S.sum(axis=0)
    alpha = 1 + sigmasum[:-1]
    beta = gamma + array([sigmasum[k+1:].sum() for k in xrange(V-1)])

    m = zeros(V)
    stick = 1.0
    omega = []
    for i,r in enumerate(rbeta(alpha, beta, seed=seed)):

        omega.append(r)

        m[i] = r * stick

        stick = (1-r) * stick


    m[-1] = stick
    return m, array(omega)

def _tau_sampler(R, a, seed):

    J, K = R.shape

    Rj = R.sum(axis=1)
    a = ones(J) * a

    return rbeta(0.01 + a, 0.01 + Rj, seed=seed)


def _polya_concentration_sampler(sigmarowsum, tau, seed):

    shape_parameter = sum(sigmarowsum) + 1
    scale_parameter = 1.0/(1.0 - sum(log(tau)))

    return get_rgamma(shape_parameter, scale_parameter, seed)


class TestCase(unittest.TestCase):

    def setUp(self):

        self._setUp(J=500)

    def _setUp(self, J):

        self.Nj = 250       # Number of words per document
        self.V = 1000       # Number of word types in vocabulary
        self.J = J          # Number of documents
        self.K_rep = 3
        self.K_max = 100    # Number of topics
        self.N = self.J*self.Nj 

        self.x = randint(0, self.K_rep, size=self.N)
        self.w = randint(0, self.V, size=self.N)

        z = []
        for j in xrange(self.J):
            z.extend([j]*self.Nj)

        self.z = array(z)

        self.assertTrue(len(self.x) == len(self.w) == len(self.z) == self.N)

        self.psi = ones(self.V)/self.V

        self.a = self.K_max
        self.b = self.V
        self.c = self.V
        self.gamma = 100.0

        self.m = rstick(self.gamma, self.K_max)

        self.prior_gamma_shape = 1.0
        self.prior_gamma_scale = 1.0

    ##########################################################################
    ######################### Helper functions ###############################
    ##########################################################################

    def get_counts(self):

        """
        Get the S, R, Sk and nonzero totals of S and R.

        """

        Sk = zeros(self.K_max, dtype=numpy.int, order='F')
        S = zeros((self.K_max, self.V), dtype=numpy.int, order='F')
        R = zeros((self.J, self.K_max), dtype=numpy.int, order='F')

        for x, w, z in zip(self.x, self.w, self.z):
            S[x, w] +=1
            R[z, x] +=1
            Sk[x] += 1

        self.assertTrue((S.sum(1) == Sk).all())

        return S, R, Sk, ((S>0).sum(), 
                          (R>0).sum(),
                          (Sk>0).sum())

    def sample_latents_python(self, seed):

        S, R, Sk, _ = self.get_counts()
        x = self.x.copy()

        index_permutation, rndvals = get_latent_rndvals_permutation(self.N,
                                                                    seed)

        tic()
        for i in xrange(self.N):

            w_i = self.w[index_permutation[i]]
            z_i = self.z[index_permutation[i]]
            x_i = x[index_permutation[i]]

            S[x_i, w_i] -= 1
            R[z_i, x_i] -= 1
            Sk[x_i] -=1 

            likelihood = (S[:, w_i] + self.b * self.psi[w_i])/(Sk + self.b)
            prior = R[z_i,:] + self.a * self.m
            f = likelihood * prior

            k_new = sample(f.cumsum(), rndvals[i])

            S[k_new, w_i] +=1 
            R[z_i, k_new] +=1
            Sk[k_new] +=1 

            x[index_permutation[i]] = k_new

        time_elapsed = toc()
        if verbose:
            print('Python time: %2.2f' % time_elapsed)


        return x, S, R, Sk

    def sample_latents_python_alt(self, seed):

        S, R, Sk, _ = self.get_counts()
        x = self.x.copy()

        index_permutation, rndvals = get_latent_rndvals_permutation(self.N,
                                                                    seed)

        tic()
        for i in xrange(self.N):

            w_i = self.w[index_permutation[i]]
            z_i = self.z[index_permutation[i]]
            x_i = x[index_permutation[i]]

            S[x_i, w_i] -= 1
            R[z_i, x_i] -= 1
            Sk[x_i] -=1 


            f = zeros(self.K_max)
            for k in xrange(self.K_rep):
                likelihood = (S[k, w_i] + self.b * self.psi[w_i])/(Sk[k] + self.b)
                prior = R[z_i,k] + self.a * self.m[k]
            
                if k==0:
                    f[k] = likelihood * prior
                else:
                    f[k] = f[k-1] + likelihood * prior

            fk = f[k]
            mu = self.m[self.K_rep:].sum()
            threshold = fk/(fk+self.psi[w_i]*self.a*mu)

            if rndvals[i] < threshold:
                r = rndvals[i]/threshold
                k_new = sample(f[:self.K_rep], r)
            else:
                r = (rndvals[i]-threshold)/(1-threshold)
                k_new = self.K_rep + sample(self.m[self.K_rep:].cumsum(), r)
                self.K_rep = k_new + 1

            S[k_new, w_i] +=1 
            R[z_i, k_new] +=1
            Sk[k_new] +=1 

            x[index_permutation[i]] = k_new

        time_elapsed = toc()
        if verbose:
            print('Alt Python time: %2.2f' % time_elapsed)


        return x, S, R, Sk


    def sample_latents_fortran2(self, seed):

        S, R, Sk, _ = self.get_counts()
        x = self.x.copy()

        S = asfortranarray(S.copy())
        R = asfortranarray(R.copy())
        Sk = Sk.copy()

        tic()

        index_permutation, rndvals = get_latent_rndvals_permutation(self.N,
                                                                    seed)


        hdpmm_latents_gibbs_sampler2(x,
                                    self.w,
                                    self.z,
                                    S,
                                    R,
                                    Sk,
                                    rndvals,
                                    index_permutation,
                                    self.psi,
                                    self.m,
                                    self.a,
                                    self.b,
                                    self.K_max,
                                    self.V,
                                    self.J,
                                    len(index_permutation),
                                    self.N)

        time_elapsed = toc()
        if verbose:
            print('Fortran time: %2.2f' % time_elapsed)

        return x, S, R, Sk


    
    def sample_latents_fortran(self, seed):

        S, R, Sk, _ = self.get_counts()
        x = self.x.copy()

        S = asfortranarray(S.copy())
        R = asfortranarray(R.copy())
        Sk = Sk.copy()

        tic()

        hdpmm_latents_gibbs_sampler(x,
                                    self.w,
                                    self.z,
                                    S,
                                    R,
                                    Sk,
                                    seed,
                                    self.psi,
                                    self.m,
                                    self.a,
                                    self.b,
                                    self.K_max,
                                    self.V,
                                    self.J,
                                    self.N)

        time_elapsed = toc()
        if verbose:
            print('Fortran time: %2.2f' % time_elapsed)

        return x, S, R, Sk

   

    def sample_latents_fortran_alt(self, seed):

        S, R, Sk, _ = self.get_counts()
        x = self.x.copy()

        S = asfortranarray(S.copy())
        R = asfortranarray(R.copy())
        Sk = Sk.copy()

        tic()

        K_rep = self.K_rep
        K_rep_new = hdpmm_latents_gibbs_sampler_alt(x,
                                                    self.w,
                                                    self.z,
                                                    S,
                                                    R,
                                                    Sk,
                                                    seed,
                                                    self.psi,
                                                    K_rep,
                                                    self.m,
                                                    self.a,
                                                    self.b,
                                                    self.K_max,
                                                    self.V,
                                                    self.J,
                                                    self.N)

        self.K_rep = K_rep_new

        time_elapsed = toc()
        if verbose:
            print('alt Fortran time: %2.2f' % time_elapsed)

        return x, S, R, Sk


    def python_get_am_sigma(self, R, rnd_seed):
        
        rnd_vals = sigma_sampler_rnd_vals(R, rnd_seed)

        return _sigma_sampler(R.max(),
                              R, 
                              rnd_vals, 
                              self.m, 
                              self.a, 
                              self.J,
                              self.K_max)


    def python_get_bpsi_sigma(self, S, rnd_seed):

        rnd_vals = sigma_sampler_rnd_vals(S, rnd_seed)

        return _sigma_sampler(S.max(),
                              S, 
                              rnd_vals, 
                              self.psi, 
                              self.b, 
                              self.K_max,
                              self.V)


    def fortran_get_am_sigma(self, R, rnd_seed):

        return sigma_sampler(R.max(),
                             R, 
                             rnd_seed, 
                             self.m, 
                             self.a, 
                             self.J,
                             self.K_max)


    def fortran_get_bpsi_sigma(self, S, rnd_seed):

        return sigma_sampler(S.max(),
                             S, 
                             rnd_seed, 
                             self.psi, 
                             self.b, 
                             self.K_max,
                             self.V)

    def fortran_get_bpsi_sigma2(self, S, rnd_seed):

        I = unique(S)

        return sigma_sampler2(S,
                              I,
                              max(I),
                              rnd_seed,
                              self.psi, 
                              self.b, 
                              len(I),
                              self.K_max,
                              self.V)


    ##########################################################################
    ###################### End Helper functions ##############################
    ##########################################################################

    def test_latent_sampler(self):

        '''
        Test if the Fortran Gibbs sampler subroutine produces the same results
        as a Python one.
        '''

        seed = randint(101, 100001)

        python_results = self.sample_latents_python(seed)

        fortran_results = self.sample_latents_fortran(seed)

        self.assertTrue(
            all([allclose(python_results[k], fortran_results[k]) 
                 for k in (0, 1, 2, 3)])
        )


    def test_latent_sampler_py_par(self):

        '''
        Test the subroutine used for the parallel sampler
        '''

        S, R, Sk, _ = self.get_counts()

        assert 5 < self.J # Lazy way to ensure that we can use next line.
        J = randint(5, self.J//2)
        group_subset = permutation(self.J)[:J]
        I = numpy.where(array([zi in group_subset for zi in self.z]))[0]

        zmap = {xi:i for i,xi in enumerate(group_subset)}
        

        latent = self.x[I]
        observed = self.w[I]
        _group = self.z[I]
        group = array([zmap[zi] for zi in _group])

        N = len(latent)
        rndvals = rand(N)

        hdpmm_latents_gibbs_sampler_par2(self.x,
                                         self.w,
                                         latent,
                                         observed,
                                         group,
                                         rndvals,
                                         self.a,
                                         self.b,
                                         self.psi,
                                         self.m,
                                         J,
                                         self.J,
                                         self.K_max,
                                         self.V,
                                         N,
                                         self.N)


        hdpmm_latents_gibbs_sampler2(self.x,
                                     self.w,
                                     self.z,
                                     S,
                                     R,
                                     Sk,
                                     rndvals,
                                     I,
                                     self.psi,
                                     self.m,
                                     self.a,
                                     self.b,
                                     self.K_max,
                                     self.V,
                                     self.J,
                                     N,
                                     self.N)


        self.assertTrue(all(latent == self.x[I]))


    def test_latent_sampler_py_alt(self):

        '''
        Test if the Fortran Gibbs sampler subroutine produces the same results
        as a Python one.
        '''

        seed = randint(101, 100001)

        python_results_alt = self.sample_latents_python_alt(seed)

        fortran_results = self.sample_latents_fortran(seed)
        fortran_results2 = self.sample_latents_fortran2(seed)

        self.assertTrue(
            all([allclose(fortran_results[k], python_results_alt[k]) 
                 for k in (0, 1, 2, 3)])
        )

        self.assertTrue(
            all([allclose(fortran_results2[k], python_results_alt[k]) 
                 for k in (0, 1, 2, 3)])
        )

    def test_latent_sampler_alt(self):

        '''
        Test if the Fortran Gibbs sampler subroutine produces the same results
        as a Python one.
        '''

        seed = randint(101, 100001)

        fortran_results = self.sample_latents_fortran(seed)

        fortran_results_alt = self.sample_latents_fortran_alt(seed)

        self.assertTrue(
            all([allclose(fortran_results[k], fortran_results_alt[k]) 
                 for k in (0, 1, 2, 3)])
        )

    def latent_sampler_alt_long_test(self):

        '''
        Test if the Fortran Gibbs sampler subroutine produces the same results
        as a Python one.
        '''

        for _ in xrange(100):

            self._setUp(J=randint(100,1000))
            self.gamma = 10.0
            self.m = rstick(self.gamma, self.K_max)

            seed = randint(101, 100001)

            fortran_results = self.sample_latents_fortran(seed)

            fortran_results_alt = self.sample_latents_fortran_alt(seed)

            self.assertTrue(
                all([allclose(fortran_results[k], fortran_results_alt[k]) 
                     for k in (0, 1, 2, 3)])
            )




    def test_stirling_matrices(self):

        """
        Test the calculation of Stirling numbers of first kind. 

        """


        # Test the unnormalized Stirling numbers
        for K in xrange(3, 21):

            pyStirling = []
            
            for k0 in xrange(K+1):
                pyStirling.append(array([int(stirling(k0, k, kind=1)) for k in xrange(K+1)]))

            pyStirling = array(pyStirling)

            S = get_stirling_numbers(K)

            self.assertTrue(allclose(pyStirling, S))


        # Test the normalized Stirling numbers using sympy
        for K in xrange(3, 21):
            pyStirling = []
            
            for k0 in xrange(K+1):
                s = array([int(stirling(k0, k, kind=1)) for k in xrange(K+1)])
                pyStirling.append(s/float(s.max()))

            pyStirling = array(pyStirling)

            S = get_normalized_stirling_numbers(K)

            self.assertTrue(allclose(pyStirling, S))


        # Test the normalized Stirling numbers using hand-made numpy function
        for K in (5, 10, 20, 30, 40, 50, 100, 200, 1000):

            pyStirling = stirlingnumbers(K)

            S = get_normalized_stirling_numbers(K)

            self.assertTrue(allclose(pyStirling, S))


        # Test the lite versions of the Python stirling matrix calculators.
        n = 100 # number of columns in the matrix (less one)
        k = 10  # number of elements in the I index array
        for _ in xrange(100):
            I = sorted(permutation(n)[:k])
            self.assertTrue(allclose(stirlingnumbers(max(I))[I],
                                     stirlingnumbers2(I)))
            self.assertTrue(allclose(stirlingnumbers(max(I))[I],
                                     stirlingnumbers3(I)))

            S = get_normalized_stirling_numbers2(I, max(I), len(I))

            self.assertTrue(allclose(S, stirlingnumbers3(I)))
            self.assertTrue(allclose(S, stirlingnumbers2(I)))
            self.assertTrue(allclose(S, stirlingnumbers(max(I))[I]))

    def test_sigma_sampler2(self):

        S, R, Sk, _ = self.get_counts()
        rnd_seed = 101

        # sigma_col_sum = sigma_sampler(stirling_dim,s,rnd_seed,psi,b,[k,v])
        A = sigma_sampler(S.max(),
                          S, 
                          rnd_seed, 
                          self.psi, 
                          self.b, 
                          self.K_max,
                          self.V)

        I = unique(S)

        B = sigma_sampler2(S,
                           I,
                           max(I),
                           rnd_seed,
                           self.psi, 
                           self.b, 
                           len(I),
                           self.K_max,
                           self.V)


        self.assertTrue(allclose(A,B))


    def test_c_sampler_alt(self):

        S, R, Sk, _ = self.get_counts()
        fortran_sigma_col_sum = self.fortran_get_bpsi_sigma(S, 101)

        c_out = polya_sampler_c(fortran_sigma_col_sum, 
                                self.c, 
                                10001,
                                len(fortran_sigma_col_sum))

        self.assertTrue(allclose(c_out, self._polya_sampler_c(S, 10001)))

        I = unique(fortran_sigma_col_sum)
        c_out2 = polya_sampler_c2(fortran_sigma_col_sum, 
                                  I,
                                  max(I),
                                  self.c, 
                                  10001,
                                  len(I),
                                  len(fortran_sigma_col_sum))


        self.assertTrue(allclose(c_out2, self._polya_sampler_c(S, 10001)))
        self.assertTrue(allclose(c_out2, c_out))

    def test_c_sampler(self):

        S, R, Sk, _ = self.get_counts()
        python_sigma = self.python_get_bpsi_sigma(S, 101)
        fortran_sigma_col_sum = self.fortran_get_bpsi_sigma(S, 101)

        self.assertTrue(allclose(python_sigma.sum(axis=0),
                                  fortran_sigma_col_sum))

        sigmasum_py = _sigma_sampler_c(fortran_sigma_col_sum.max(),
                                       python_sigma.sum(axis=0),
                                       10001,
                                       self.c,
                                       self.V)

        sigmasum = sigma_sampler_c(fortran_sigma_col_sum.max(),
                                   fortran_sigma_col_sum,
                                   10001,
                                   self.c,
                                   self.V)


        I = unique(fortran_sigma_col_sum)
        sigmasum2 = sigma_sampler_c2(fortran_sigma_col_sum,
                                     I,
                                     max(I),
                                     10001,
                                     self.c,
                                     len(I),
                                     self.V)


        self.assertEqual(sigmasum_py.sum(), sigmasum)
        self.assertEqual(sigmasum_py.sum(), sigmasum2)
        self.assertEqual(sigmasum, sigmasum2)

        py_tau = rbeta(self.c, python_sigma.sum(), seed=100)
        fortran_tau = tau_sampler_c(fortran_sigma_col_sum.sum(), 
                                  100,
                                  self.c)

        self.assertTrue(allclose(py_tau, fortran_tau))

        python_c = get_rgamma(1 + sigmasum_py.sum(),
                              1.0/(1.0-log(py_tau)),
                              10110)

        fortran_c = polya_concentration_sampler(sigmasum,
                                                fortran_tau,
                                                10110)     

        self.assertTrue(allclose(python_c, fortran_c))



    def test_gamma_sampler(self):

        S, R, Sk, _ = self.get_counts()

        python_sigma = self.python_get_am_sigma(R, 1001)
        fortran_sigma_rowsums = self.fortran_get_am_sigma(R, 1001)

        python_m, py_omega = _gdd_sampler(python_sigma, self.gamma, 101)
        fortran_m, omega = gdd_sampler(fortran_sigma_rowsums, self.gamma, 101)
        f_gamma = gamma_sampler(omega, 
                                2002, 
                                self.prior_gamma_shape,
                                self.prior_gamma_scale)

        self.assertTrue(allclose(py_omega, omega))

        shape_parameter = self.prior_gamma_shape + len(omega)
        scale_parameter = 1.0/(1/self.prior_gamma_scale - sum(log(1-omega)))

        py_gamma = get_rgamma(shape_parameter, scale_parameter, 2002)

        self.assertTrue(allclose(f_gamma, py_gamma))


    def test_sigma_sampler(self):

        '''
        Test the s and r sigma samplers, i.e. for b*psi and a*m, respectively.

        Also provide a test of bpsi_sampler and bpsi_sampler2

        '''

        S, R, Sk, _ = self.get_counts()
        
        seed = randint(101, 100001)
        sigma_s = self.python_get_bpsi_sigma(S, seed)
        f_sigma_s_sum = self.fortran_get_bpsi_sigma(S, seed)
        f_sigma_s_sum2 = self.fortran_get_bpsi_sigma2(S, seed)
        self.assertTrue(allclose(sigma_s.sum(axis=0), 
                                 f_sigma_s_sum))
        self.assertTrue(allclose(sigma_s.sum(axis=0), 
                                 f_sigma_s_sum2))
        self.assertTrue(allclose(f_sigma_s_sum,
                                 f_sigma_s_sum2))



        seed = randint(101, 100001)
        sigma_r = self.python_get_am_sigma(R, seed)
        f_sigma_r_sum = self.fortran_get_am_sigma(R, seed)
        self.assertTrue(allclose(sigma_r.sum(axis=0),
                                 f_sigma_r_sum))


    def test_gdd_sampler(self):

        '''

        '''

        S, R, Sk, _ = self.get_counts()

        seed = 1010
        python_sigma = self.python_get_am_sigma(R, seed)
        fortran_sigma_rowsums = self.fortran_get_am_sigma(R, seed)

        python_sigma_rowsums = python_sigma.sum(axis=0)

        self.assertEqual(python_sigma_rowsums.shape[0],
                         self.K_max)

        self.assertTrue(allclose(python_sigma_rowsums,
                                 fortran_sigma_rowsums))

        seed = randint(101, 100001)

        python_m, omega = _gdd_sampler(python_sigma, self.gamma, seed)
        fortran_m, omega = gdd_sampler(fortran_sigma_rowsums, self.gamma, seed)

        self.assertTrue(allclose(python_m, fortran_m))


    def test_tau_sampler(self):

        '''
        Unit test of sampling from tau in auxilliary variable based Polya
        sampler.
        '''

        S, R, Sk, _ = self.get_counts()

        seed = randint(101, 100001)

        python_tau = _tau_sampler(R, self.a, seed)
        fortran_tau = tau_sampler(R.sum(axis=1), self.a, seed)

        self.assertTrue(allclose(python_tau, fortran_tau))

    def _polya_sampler_bpsi(self, S, seed):

        '''
        An integration test of the auxilliary variable based sampler of b and
        psi.

        '''


        rnd_seeds = get_random_integers(4, 101, 10001, seed)
        python_sigma = self.python_get_bpsi_sigma(S, rnd_seeds[0])
        python_sigma_rowsums = python_sigma.sum(axis=0)

        python_psi = get_rdirichlet(python_sigma_rowsums + self.c/self.V,
                                    rnd_seeds[1])
        python_tau = _tau_sampler(S, self.b, rnd_seeds[2])
        python_b = _polya_concentration_sampler(python_sigma_rowsums,
                                                python_tau,
                                                rnd_seeds[3])
    
        return python_sigma_rowsums, python_b, python_psi


    def _polya_sampler_am(self, R, seed):

        '''
        An integration test of the auxilliary variable based sampler of a and
        m.

        '''

        rnd_seeds = get_random_integers(4, 101, 10001, seed)

        # sigma sampler test
        python_sigma = self.python_get_am_sigma(R, rnd_seeds[0])
        python_sigma_rowsums = python_sigma.sum(axis=0)

        # m sampler test
        python_m, omega = _gdd_sampler(python_sigma, self.gamma, rnd_seeds[1])

        # tau sampler test
        python_tau = _tau_sampler(R, self.a, rnd_seeds[2])

        # a sampler test
        python_a = _polya_concentration_sampler(python_sigma_rowsums,
                                                python_tau,
                                                rnd_seeds[3])

        return omega, python_a, python_m

    def _polya_sampler_c(self, S, seed):

        S, R, Sk, _ = self.get_counts()
        python_sigma = self.python_get_bpsi_sigma(S, 101)

        rnd_seeds = get_random_integers(3, 101, 10001, seed)

        sigmasum_py = _sigma_sampler_c(python_sigma.sum(axis=0).max(),
                                       python_sigma.sum(axis=0),
                                       rnd_seeds[0],
                                       self.c,
                                       self.V)

        py_tau = rbeta(self.c, python_sigma.sum(), seed=rnd_seeds[1])

        python_c = get_rgamma(1 + sigmasum_py.sum(),
                              1.0/(1.0-log(py_tau)),
                              rnd_seeds[2])

        return python_c

    
    def test_am_polya_sampler(self):

        '''
        An integration test of the auxilliary variable based sampler of a and
        m.

        '''


        S, R, Sk, _ = self.get_counts()

        # sigma sampler test
        python_sigma = self.python_get_am_sigma(R, 101)
        fortran_sigma_rowsums = self.fortran_get_am_sigma(R, 101)

        python_sigma_rowsums = python_sigma.sum(axis=0)

        self.assertEqual(python_sigma_rowsums.shape[0],
                         self.K_max)

        self.assertTrue(allclose(python_sigma_rowsums,
                                 fortran_sigma_rowsums))

        # m sampler test
        seed = randint(101, 100001)

        python_m, omega = _gdd_sampler(python_sigma, self.gamma, seed)
        fortran_m, omega = gdd_sampler(fortran_sigma_rowsums, self.gamma, seed)

        self.assertTrue(allclose(python_m, fortran_m))

        # tau sampler test
        seed = randint(101, 100001)

        python_tau = _tau_sampler(R, self.a, seed)
        fortran_tau = tau_sampler(R.sum(axis=1), self.a, seed)

        self.assertEqual(python_tau.shape[0],
                         self.J)

        self.assertEqual(fortran_tau.shape[0],
                         self.J)

        self.assertTrue(allclose(python_tau, fortran_tau))
        

        # a sampler test
        seed = randint(101, 100001)
        python_a = _polya_concentration_sampler(python_sigma_rowsums,
                                                python_tau,
                                                seed)
    
        fortran_a = polya_concentration_sampler(python_sigma_rowsums.sum(),
                                                fortran_tau,
                                                seed)

        self.assertTrue(allclose(python_a, fortran_a))


    def test_bpsi_polya_sampler(self):

        '''
        An integration test of the auxilliary variable based sampler of b and
        psi.

        Also, test the bspi sigma_sampler2.

        '''

        
        S, R, Sk, _ = self.get_counts()

        # sigma sampler test
        seed = 1010101
        python_sigma = self.python_get_bpsi_sigma(S, seed)
        fortran_sigma_rowsums = self.fortran_get_bpsi_sigma(S, seed)
        fortran_sigma_rowsums2 = self.fortran_get_bpsi_sigma2(S, seed)

        python_sigma_rowsums = python_sigma.sum(axis=0)

        self.assertEqual(python_sigma_rowsums.shape[0],
                         self.V)

        self.assertTrue(allclose(python_sigma_rowsums,
                                 fortran_sigma_rowsums))

        self.assertTrue(allclose(python_sigma_rowsums,
                                 fortran_sigma_rowsums2))

        # psi sampler test
        seed = randint(101, 100001)

        python_psi = get_rdirichlet(python_sigma_rowsums + self.c/self.V, seed)
        fortran_psi = ddirichlet_sampler(fortran_sigma_rowsums, self.c/self.V, seed)

        self.assertTrue(allclose(python_psi, fortran_psi))

        # tau sampler test
        seed = randint(101, 100001)

        python_tau = _tau_sampler(S, self.b, seed)
        fortran_tau = tau_sampler(S.sum(axis=1), self.b, seed)

        self.assertEqual(python_tau.shape[0],
                         self.K_max)

        self.assertEqual(fortran_tau.shape[0],
                         self.K_max)


        self.assertTrue(allclose(python_tau, fortran_tau))
        
        # b sampler test
        seed = randint(101, 100001)
        python_b = _polya_concentration_sampler(python_sigma_rowsums,
                                                python_tau,
                                                seed)
    
        fortran_b = polya_concentration_sampler(fortran_sigma_rowsums.sum(),
                                                fortran_tau,
                                                seed)

        self.assertTrue(allclose(python_b, fortran_b))

    def test_am_polya_sampler_alt(self):

        '''
        An integration test of the auxilliary variable based sampler of a and
        m.
        Here, we test the polya_sampler_am subroutine

        '''

        
        seed = randint(101, 10001)

        S, R, Sk, _ = self.get_counts()

        f_out = polya_sampler_am(R,
                                 self.m,
                                 self.a,
                                 self.gamma,
                                 seed,
                                 self.J,
                                 self.K_max)


        py_out = self._polya_sampler_am(R, seed)
        
        for k in xrange(len(f_out)):
            self.assertTrue(allclose(f_out[k], py_out[k]))


    def test_bpsi_polya_sampler_alt(self):

        '''
        An integration test of the auxilliary variable based sampler of a and
        m.
        Here, we test the polya_sampler_am subroutine

        '''

        seed = randint(101, 10001)

        S, R, Sk, _ = self.get_counts()

        f_out = polya_sampler_bpsi(S,
                                   self.psi,
                                   self.b,
                                   self.c,
                                   seed,
                                   self.K_max,
                                   self.V)

        I = unique(S)
        f_out2 = polya_sampler_bpsi2(S, 
                                     I,
                                     max(I),
                                     self.psi,
                                     self.b,
                                     self.c,
                                     seed,
                                     len(I),
                                     self.K_max,
                                     self.V)

        f_out3 = utils.polya_sampler_bpsi(S, 
                                          self.psi,
                                          self.b,
                                          self.c,
                                          seed)

        py_out = self._polya_sampler_bpsi(S, seed)

        for k in xrange(len(f_out)):
            self.assertTrue(allclose(f_out[k], py_out[k]))
            self.assertTrue(allclose(f_out2[k], py_out[k]))
            self.assertTrue(allclose(f_out3[k], py_out[k]))


    def test_samplers(self):

        '''
        Test fsampler and bisection sampler

        '''

        K = 1000
        N = 10000 

        f = rand(K)
        q = f.cumsum()
        for _ in xrange(N):
           rndval = rand()

           py_sampled_k = sample(q, rndval)

           sampled_k = bisection_sampler(q, rndval)

           self.assertTrue(sampled_k == py_sampled_k)


        # Try with some variable sized array
        for _ in xrange(N):

           f = rand(randint(10, K))
           q = f.cumsum()
           rndval = rand()

           py_sampled_k = sample(q, rndval)

           sampled_k = bisection_sampler(q, rndval)

           self.assertTrue(sampled_k == py_sampled_k)

        # Try with some short arrays
        for _ in xrange(N):

           f = rand(randint(2, 4))

           q = f.cumsum()

           rndval = rand()

           py_sampled_k = sample(q, rndval)

           sampled_k = bisection_sampler(q, rndval)

           self.assertTrue(sampled_k == py_sampled_k)


if __name__ == '__main__':
    unittest.main()
