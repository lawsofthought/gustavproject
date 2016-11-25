from __future__ import division
from numpy import zeros, float64, ones, array, unique, arange, where
from numpy.random import rand, beta, permutation, shuffle
import numpy
from numpy.linalg import norm
from itertools import cycle
from collections import Counter, defaultdict
import fortransamplers


def sliceit(N, K):
    return zip(list(numpy.arange(K) * (N//K)), list(numpy.arange(1, K) * (N//K)) + [N-1])

class BagOfWords(object):

    '''
    A bag of words data object.

    Input
    -----

    data_filename: Filename of lda formatted data

    E.g.

    0:1 6144:1 3586:2 3:1 4:1 1541:1 8:1 10:1 3927:1 12:7 
    257:1 262:1 1927:2 1032:1 13:1 14:1 
    5829:1 4040:1 2891:1 14:1 1783:1 381:1 

    vocabulary_filename: Newline delimited list of words in vocabulary.


    '''

    def __init__(self, data_filename, vocabulary_filename):

        vocabulary = open(vocabulary_filename).read().strip().split('\n')

        documents = open(data_filename).read().strip().split('\n')

        all_tokens = dict()

        z = []
        w = []

        ii = 0
        start_doc_index = {}
        Nj = {}
        for doc_id, document in enumerate(documents):

            start_doc_index[doc_id] = ii


            nj = 0
            for token_count_tuple in document.strip().split():

                token, count = map(int, 
                                   token_count_tuple.split(':'))

                for _ in xrange(count):

                    z.append(doc_id)
                    w.append(token)

                    ii += 1
                    nj += 1

                all_tokens[token] = None

            Nj[doc_id] = nj

        assert len(vocabulary) == len(all_tokens) == (max(w) + 1)

        assert len(z) == len(w)

        assert len(documents) == (max(z)+1) == (max(Nj.keys()) + 1) == (max(start_doc_index) + 1)

        self.w = numpy.array(w)
        self.z = numpy.array(z)

        self.vocabulary = vocabulary

        self.V = len(vocabulary)
        self.J = len(documents)
        self.N = len(self.z)

        self.index_to_word = dict()
        self.word_to_index = dict()

        for i, word in enumerate(self.vocabulary):
            self.index_to_word[i] = word
            self.word_to_index[word] = i


        self.lims = array([(start_doc_index[j], Nj[j]) for j in
                           xrange(self.J)])


    def _doc_indices(self, j):
        return arange(self.lims[j][0], self.lims[j][0]+self.lims[j][1]) 

    def _lim_test(self):

        for i, lim in enumerate(self.lims):
            truth = self._doc_indices(i) == where(self.z == i)[0]
            assert all(truth)

    def distribute(self, K=8, test=False):

        '''
        Distribute 

        '''

        docs = list(permutation(self.J))

        indices = defaultdict(list)
        assignments = defaultdict(list)

        for k in cycle(xrange(K)):

            if docs:

                doc = docs.pop()
                assignments[k].append(doc)
                indices[k].extend(list(self._doc_indices(doc)))

            else:

                break

        if test:
            for i in xrange(K):
                a_i,b_i = assignments[k], indices[k]
                assert all(array(sorted(a_i)) == numpy.unique(self.z[array(b_i)]))


        start = 0
        I = []
        limits = []
        starts = []
        lengths = []
        stop = []
        for index in indices.values():

            shuffle(index)

            I.extend(index)
            stop = len(I) 
            starts.append(start)
            lengths.append(len(index))
            limits.extend([start, stop])
            start = stop 

        I, limits = array(I), array(limits)

        if test:
            for k in xrange(K):
                i, j = array([0, 1]) + 2*k
                assert all(unique(self.z[I[limits[i]:limits[j]]]) ==
                           array(sorted(assignments[k])))

        return I, tuple(starts), tuple(lengths)


def sample(f, r, input_data_check=False):

    """
    Draw a sample from (unnormalized) cumulative probability mass fnc f.

    Optionally, check if r \in (0, 1) and f is monotonic non-decreasing.

    This is beastly slow and should only be used for testing.

    """

    N = len(f) 

    if input_data_check:

        assert 0.0 <= r <= 1.0, 'r = %2.3f. We need 0.0 <= r <= 1.0' % r

        # f must be monontonic non-decreasing
        for i in xrange(1, N):
            assert f[i] >= f[i-1]

    k = 0
    while f[k]/f[N-1] < r:
        k += 1

    return k


def stirlingnumbers(n):

    '''
    A (n+1 x n+1) numpy array of the first n+1 normalized rows of a unsigned
    Stirling numbers of first kind array.

    These will then be used as a lookup table.

    '''

    assert n > 0

    U = zeros((n+1, n+1), dtype=float64)
    U[0,0] = 1.0
    for i in xrange(1, len(U)):
        U[i] = U[i-1] * (i-1)
        U[i][1:] += U[i-1][:-1]
        U[i] /= U.max()

    return U


def stirlingnumbers2(I):
    
    '''
    
    A (len(I) x n+1) numpy array of normalized rows, rows corresponding 
    to the indices in I, of a unsigned Stirling numbers of first kind array.
    
    These will then be used as a lookup table.
    
    Note: I must be a list of unique integers in ascending order.

    '''
    
    n = max(I)
    
    assert n > 0

    U = zeros((len(I), n+1), dtype=float64)
    
    ind = 0
    i = 0 
    u = zeros(n+1, dtype=float64)
    u[0] = 1.0
    
    if i in I:
        U[ind] = u
        ind += 1
    
    for i in xrange(1, n+1):
        
        _u = u * (i-1)
        _u[1:] += u[:-1]
        u = _u/_u.max()
        
        if i in I:
            U[ind] = u
            ind += 1
            
    return U


def stirlingnumbers3(I):
    
    '''
    
    Identical in function to stirlingnumbers3. Implementated differently.

    '''
        
    n = max(I)

    U = zeros((len(I), n+1), dtype=float64)
    
    ind = 0
    i = 0 
    u = zeros(n+1, dtype=float64)
    u[0] = 1.0
    
    if I[ind] == 0:
        U[ind] = u
        ind += 1
    
    for i in xrange(1, n+1):
        
        _u = u * (i-1)
        _u[1:] += u[:-1]
        u = _u/_u.max()
        
        if I[ind] == i:
            U[ind] = u
            ind += 1
            
    return U


def polya_sampler_bpsi(S, psi, b, c, seed):

    """
    A wrapper to fortransamplers.polya_sampler_bpsi.

    """

    K, V = S.shape

    I = unique(S)

    return fortransamplers.polya_sampler_bpsi2(S, 
                                               I,
                                               max(I),
                                               psi,
                                               b,
                                               c,
                                               seed,
                                               len(I),
                                               K,
                                               V)


def testset(K=10, J=250, N=100, m=None):

    ''' 
    Produces a 'grids' test-set for testing the LDA/HDPMM models.

    K*2 topics. J documents, each with N words.

    '''

    def do_sample(p, size=None):
        f = p.cumsum()
        if size is None:
            r = rand()
            return sample(f, r)
        else:
            samples = []
            for _ in xrange(size):
                r = rand()
                samples.append(sample(f, r))
            return samples

    if m is None:
        m = []
        for _ in xrange(K):
            m.extend([1, 1])
        m = array(m)

    epsilon = 1e-3

    X = zeros((K,K**2))
    Y = zeros((K,K**2))
    for i in xrange(K):
        x = zeros((K, K))
        x[i] = ones(K)
        X[i] = x.flatten()
        Y[i] = x.T.flatten()


    exact_phi = numpy.vstack((X,Y))
    exact_phi = (exact_phi.T/exact_phi.T.sum(axis=0)).T
    phi = numpy.clip(exact_phi,epsilon,1-epsilon)
    phi = (phi.T/phi.T.sum(axis=0)).T

    data = []
    VPI = numpy.random.dirichlet(m, size=J)

    for vpi in VPI:
        x = do_sample(vpi, size=N)
        data.append([do_sample(phi[x_i]) for x_i in x])

    return K, exact_phi, array(data)

def write_testdata(K, 
                   data, 
                   filename='testdata.dat',
                   vocab_file='testdata_vocabulary.txt'):

    with open(vocab_file, 'w') as f:
        S = '\n'.join([str(x) for x in xrange(K**2)])
        f.write(S)

    make_lda_dat_str\
        = lambda doc: ' '.join([':'.join(map(str,x)) for x in Counter(doc).items()])

    with open(filename, 'w') as f:
        f.write('\n'.join([make_lda_dat_str(doc) for doc in data]))



def cmp_phi(exact_phi, phi):

    def cosine(x, y):
        return numpy.inner(x,y) / (norm(x) * norm(y))

    C = zeros((len(phi), len(exact_phi)))
    for k in xrange(len(phi)):
        for l in xrange(len(exact_phi)):

            C[k,l] = cosine(phi[k], exact_phi[l])

    return C

def rstick(gamma, K):

    stick = 1.0
    omega = []
        
    def rbeta():
        return beta(1, gamma)
        
    for k in xrange(K-1):

        beta_prime = rbeta()

        omega.append(beta_prime * stick)

        stick = (1-beta_prime) * stick

    omega.append(stick)

    return numpy.array(omega)
