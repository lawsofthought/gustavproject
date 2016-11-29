from __future__ import division

import os
import time
import datetime

from utils import BagOfWords
from samplers.utils import sample, rstick
import numpy
from numpy.random import randint, dirichlet, rand
from numpy import ones, zeros, unique, empty

from .samplers import fortranutils, fortransamplers

def distributed_latent_sampler(model, index):

    latent = model.x[index]
    observed = model.w[index]
    group = model.z[index]

    zmap = {z:i for i,z in enumerate(unique(group))}
    group = numpy.array([zmap[z] for z in group])

    J = len(zmap)
    N = len(latent)
    rndvals = rand(N)

    fortransamplers.hdpmm_latents_gibbs_sampler_par2(model.x,
                                     model.w,
                                     latent,
                                     observed,
                                     group,
                                     rndvals,
                                     model.a,
                                     model.b,
                                     model.psi,
                                     model.m,
                                     J,
                                     model.J,
                                     model.K_max,
                                     model.V,
                                     N,
                                     model.N)

    return latent


def showtopics(phi, vocabulary, show_probability=True, K=10):
    
    sprintf = {True: lambda k: vocabulary[k] + ' (%2.2f)' % phi[k],
               False: lambda k: vocabulary[k]}

    return ', '.join([sprintf[show_probability](k) 
                      for k in numpy.flipud(phi.argsort())[:K]])

def make_model_id():

    today = datetime.datetime.today()
    model_id = '_'.join(['hdptm', 
                          today.strftime('%d%m%y%H%M%S'),
                          str(randint(1000, 10000))])

    return model_id


class HierarchicalDirichletProcessTopicModel(object):

    '''
    
    Hierarchical Dirichlet Process Mixture Model

    Observed variables: V, J, w

    m ~ dstick(eta)

    psi ~ ddirichlet(c/V)
    for k in 0, 1, 2 ...
        phi[k] ~ dddirichlet(b*psi)

    for j in 0, 1, 2 ... J

        vpi[j] ~ ddirichletprocess(a, m)

        for i in 0, 1, 2, ... n_j
            
            x[j,i] ~ dcat(vpi[j])
            w[j,i] ~ dcat(phi[x[j,i]])

    What about eta, c, b, a?
 

    '''
    number_of_processors = 4

    def load_state(self, filename):

        state = numpy.load(filename)
        for key in ('a', 'b', 'gamma', 'psi', 'x', 'm', 'c'):
            setattr(self, key, state[key])

        #self.get_counts()

    @classmethod
    def restart(cls, data, state, model_id, iteration):

        model = cls(data, 
                    K_min=state['K_min'], 
                    K_max=state['K_max'],
                    model_id=model_id)

        model.iteration = iteration

        for key in ('a', 'b', 'gamma', 'psi', 'x', 'm', 'c', 'K_rep'):
            setattr(model, key, state[key])


        return model


    @classmethod
    def new(cls, data, K_min=5, K_max=1000):

        model_id = make_model_id()

        return cls(data, K_min=K_min, K_max=K_max, model_id=model_id)

    @property
    def date_initiated(self):

        model_type, date_string, uid = self.model_id.split('_')

        return datetime.datetime.strptime(date_string, '%d%m%y%H%M%S')

    def __init__(self, data, K_min=3, K_max=1000, model_id=None):

        if model_id is None:
            model_id = make_model_id()

        self.model_id = model_id

        self.iteration = 0

        self.K_min = K_min
        self.K_max = K_max
        self.K_rep = K_min

        self.load_bag_of_words(data)

        # Hyperparameters and hyperhyperparameters
        # ========================================
        #
        # Following Griffiths & Steyvers (2004), we can set the starting value
        # of b to 0.1*V, psi_v to 1/V, a to 50. Griffiths and
        # Steyvers set each beta_v to 0.1 and set each alpha_k = 50/K, where K
        # is the number of topics.
        # The value of gamma should be such that stick breaking prior should
        # lead to p mass over K_rep topics.
        # 
        # References
        # ----------
        # Griffiths, T. L., & Steyvers, M. (2004). Finding scientific
        # topics.  Proceedings of the National academy of Sciences, 101(suppl
        # 1), 5228-5235.

        self.b = 0.1 * self.V
        self.psi = ones(self.V)/self.V
        self.a = 50.0
        self.gamma = self._initialize_gamma(self.K_rep, p=0.9)
        self.m = rstick(self.gamma, self.K_max)

        self.c = 1.0
        self.prior_gamma_shape = 1.0
        self.prior_gamma_scale = 0.1 # Or e.g. 0.05 to put a prior on low gamma

        self.initialize()

        self.verbose = True
        self.use_slow_latent_sampler = False
        self.use_altalt = True

        self.slow2 = False

        self.save_every_iteration = 10

    def _initialize_gamma(self, J, p=0.9):

        f = lambda gamma : (1/(1+gamma)) * sum([(gamma/(1+gamma))**j for j in xrange(J)])

        gamma = 1.0
        while True:

            if f(gamma) <= p:
                return gamma
            else:
                gamma += 1.0

            if gamma >= 1000.0:
                raise Exception('Run away process of estimating gamma?')

    def _initialize_m(self, p=0.9):

        m = empty(self.K_max)
        m[:self.K_min] = p/self.K_rep
        m[self.K_rep:] = (1-p)/(self.K_max - self.K_rep)

        return m

    def initialize(self, get_counts=False):

        '''
        Re-set latents to random values and re-count.
        '''

        self.x = randint(0, self.K_rep, size=self.N)
        if get_counts:
            self.get_counts()

    def get_counts(self):

        self.S, self.R, self.Sk = fortransamplers.hdpmm_get_counts(self.x, 
                                                   self.w, 
                                                   self.z,
                                                   self.K_max, 
                                                   self.V, 
                                                   self.J, 
                                                   self.N)

        self.K_rep = self.x.max() + 1

    def get_S_counts(self):

        self.S = zeros((self.K_max, self.V), dtype=numpy.int, order='F')
        for w_i, x_i in zip(self.w, self.x):
            self.S[x_i, w_i] += 1

        self.Sk = self.S.sum(axis=1)

        assert self.Sk.size == self.S.shape[0] == self.K_max

    def slow_get_counts(self):

        self.S = zeros((self.K_max, self.V), dtype=numpy.int, order='F')
        self.R = zeros((self.J, self.K_max), dtype=numpy.int, order='F')
        for w_i, x_i, z_i in zip(self.w, self.x, self.z):

            self.R[z_i, x_i] += 1
            self.S[x_i, w_i] += 1

        self.Rj = self.R.sum(axis=1)
        self.Sk = self.S.sum(axis=1)

        assert self.Rj.size == self.R.shape[0] == self.J
        assert self.Sk.size == self.S.shape[0] == self.K_max

    def load_bag_of_words(self, bag_of_words):

        assert type(bag_of_words) is BagOfWords

        self.w = bag_of_words.w
        self.z = bag_of_words.z
        self.N = bag_of_words.N
        self.J = bag_of_words.J
        self.V = bag_of_words.V

        self.vocabulary = bag_of_words.vocabulary

        self.index_to_word = bag_of_words.index_to_word
        self.word_to_index = bag_of_words.word_to_index

        assert len(self.w) == len(self.z) == self.N
        assert (self.z.max() + 1) == self.J
        assert (self.w.max() + 1) == self.V
        assert len(self.vocabulary) == self.V

        assert self.z.min() == 0
        assert self.w.min() == 0

        self.bag_of_words = bag_of_words


    def sample_latent(self, seed=None):


        if self.parallel:
            self.sample_latent_parallel(seed)
        else:
            if self.use_slow_latent_sampler:
                if self.slow2:
                    self.sample_latent_serial_orig2(seed)
                else:
                    self.sample_latent_serial_orig(seed)
            else:
                if self.use_altalt:
                    self.sample_latent_altalt(seed)
                else:
                    self.sample_latent_serial(seed)


    def sample_latent_serial(self, seed=None):

        tic = time.time()

        if self.verbose:
            print('sampling latent serial')


        if seed is None:
            seed = randint(101, 1000001)

        K_rep_start_value = self.K_rep 

        K_rep_new = fortransamplers.hdpmm_latents_gibbs_sampler_alt(self.x,
                                    self.w,
                                    self.z,
                                    self.S,
                                    self.R,
                                    self.Sk,
                                    seed,
                                    self.psi,
                                    K_rep_start_value,
                                    self.m,
                                    self.a,
                                    self.b,
                                    self.K_max,
                                    self.V,
                                    self.J,
                                    self.N)

        self.K_rep = K_rep_new
        self.get_counts()

        if self.verbose:
            print(time.time() - tic)


    def sample_latent_serial_orig2(self, I, seed=None):

        tic = time.time()

        if self.verbose:
            print('sampling latent serial original 2')

        if seed is None:
            seed = randint(101, 1000001)

        N = len(I)
        rndvals = rand(N)
        
        fortransamplers.hdpmm_latents_gibbs_sampler2(self.x,
                                     self.w, 
                                     self.z, 
                                     self.S, 
                                     self.R,
                                     self.Sk,
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
        #self.get_counts()

        if self.verbose:
            print(time.time() - tic)


    def sample_latent_serial_orig(self, seed=None):

        tic = time.time()

        if self.verbose:
            print('sampling latent serial original')

        if seed is None:
            seed = randint(101, 1000001)

        fortransamplers.hdpmm_latents_gibbs_sampler(self.x,
                                    self.w, 
                                    self.z, 
                                    self.S, 
                                    self.R,
                                    self.Sk,
                                    seed,
                                    self.psi,
                                    self.m,
                                    self.a,
                                    self.b,
                                    self.K_max,
                                    self.V,
                                    self.J,
                                    self.N)

        self.get_counts()

        if self.verbose:
            print(time.time() - tic)

    def sample_am(self, seed=None):
        
        tic = time.time()

        if self.verbose:
            print('sampling am')


        if seed is None:
            seed = randint(101, 1000001)


        self.omega, self.a, self.m = fortransamplers.polya_sampler_am(self.R, self.m, self.a, self.gamma, seed,
                                self.J, self.K_max)

        if self.verbose:
            print(time.time() - tic)

    def sample_bpsi(self, seed=None):

        tic = time.time()


        if self.verbose:
            print('sampling bpsi')


        if seed is None:
            seed = randint(101, 1000001)


        I = unique(self.S)
        self.sigma_s_colsums, self.b, self.psi = fortransamplers.polya_sampler_bpsi2(self.S, 
                                                                     I,
                                                                     max(I),
                                                                     self.psi, 
                                                                     self.b, 
                                                                     self.c, 
                                                                     seed,
                                                                     len(I),
                                                                     self.K_max, 
                                                                     self.V)

        if self.verbose:
            print(time.time() - tic)


    def sample_c(self, seed=None):

        tic = time.time()

        if self.verbose:
            print('sampling c')


        if seed is None:
            seed = randint(101, 1000001)

        I = unique(self.sigma_s_colsums)
        self.c = fortransamplers.polya_sampler_c2(self.sigma_s_colsums, 
                                  I,
                                  max(I),
                                  self.c,
                                  seed, 
                                  len(I),
                                  self.V)

        if self.verbose:
            print(time.time() - tic)

    def sample_gamma(self, seed=None):

        if self.verbose:
            print('sampling gamma')

        if seed is None:
            seed = randint(101, 1000001)

        self.gamma = fortransamplers.gamma_sampler(self.omega,
                                   seed,
                                   self.prior_gamma_shape,
                                   self.prior_gamma_scale)

    def _sample(self, iterations=10, am = True, bpsi=True, c=True, gamma=False):

        for iteration in xrange(iterations):
            print('K_rep is %d' % self.K_rep)
            print('Iteration: %d' % iteration)

            self.sample_latent()

            if am:
                self.sample_am()

            if bpsi:
                self.sample_bpsi()

            if c:
                self.sample_c()

            if gamma:
                self.sample_gamma()

            if iteration > 0 and (iteration % self.save_every_iteration == 0):
                self.save_state()

    def update_hyperparameters(self, hyperparameters):

        for hyperparameter in ['a', 'b', 'c', 'gamma', 'm', 'psi']:
            if hyperparameter in hyperparameters:
                setattr(self, hyperparameter, hyperparameters[hyperparameter])


    def update(self, iterations=10, am = True, bpsi=True, c=True, gamma=False):

        self._sample(iterations=iterations, am = am, bpsi=bpsi, c=c, gamma=gamma)


    def sample_phi(self):

        _phi = self.S + self.b*self.psi
        return numpy.array([dirichlet(_phi_i) for _phi_i in _phi])

    def sample(self, samples=1000, thin=10, gamma=True):

        for _ in xrange(samples):
            self._sample(iterations=thin, gamma=gamma)
            

    def save_state(self, root='.'):

        timestamp = datetime.datetime.now()

        state = dict(x = self.x,
                     w = self.w,
                     z = self.z,
                     a = self.a,
                     b = self.b,
                     c = self.c,
                     psi = self.psi,
                     m = self.m,
                     gamma = self.gamma,
                     K_rep = self.K_rep,
                     K_min = self.K_min,
                     K_max = self.K_max)

        filename = self.model_id + '_state_' + str(self.iteration)
        fullpath = os.path.join(root, filename)

        numpy.savez(fullpath, **state)

        return timestamp, filename + '.npz'

    def posterior_predictive(self, words, iterations=100, thin=10, seed=None):

        w = [self.word_to_index[word] for word in words]

        N = len(w)

        x = randint(self.K_max, size=N)

        self.get_S_counts()

        # Initialize counts
        S = self.S.copy()
        Sk = self.Sk.copy()
        R = zeros(self.K_max, dtype=int)

        # Update counts
        for x_i, w_i in zip(x, w):
            S[x_i, w_i] +=1
            Sk[x_i] += 1
            R[x_i] += 1

        w_predicted = zeros(self.V)
        for iteration in xrange(iterations):

            seed = randint(101, 1000001)

            index_permutation, rndvals = fortranutils.get_latent_rndvals_permutation(N, seed)

            for i in xrange(N):

                w_i = w[index_permutation[i]]
                x_i = x[index_permutation[i]]

                S[x_i, w_i] -= 1
                R[x_i] -= 1
                Sk[x_i] -=1 

                likelihood = (S[:, w_i] + self.b * self.psi[w_i])/(Sk + self.b)
                prior = R + self.a * self.m
                f = likelihood * prior

                k_new = sample(f.cumsum(), rndvals[i])

                S[k_new, w_i] +=1 
                R[k_new] +=1
                Sk[k_new] +=1 

                x[index_permutation[i]] = k_new

            if iteration > 0 and (iteration % thin == 0):

                # Draw a sample for phi (based on self.S, not S)
#                phi = zeros((self.K_max, self.V))
#                for k in xrange(self.K_max):
#                    phi[k] = dirichlet(self.S[k] + self.b*self.psi)
#
                # Sample vpi
                vpi = dirichlet(R)# + self.a*self.m)

                # Predictions
                w_predicted += numpy.dot(vpi, self.S)

        return w_predicted/w_predicted.sum()
