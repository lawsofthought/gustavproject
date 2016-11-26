from __future__ import division, absolute_import

from numpy import zeros, ones, array, unique, arange, where
from numpy.random import rand, permutation, shuffle
import numpy
from itertools import cycle
from collections import Counter, defaultdict

from .samplers import utils


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



def testset(K=10, J=250, N=100, m=None):

    ''' 
    Produces a 'grids' test-set for testing the LDA/HDPMM models.

    K*2 topics. J documents, each with N words.

    '''

    def do_sample(p, size=None):
        f = p.cumsum()
        if size is None:
            r = rand()
            return utils.sample(f, r)
        else:
            samples = []
            for _ in xrange(size):
                r = rand()
                samples.append(utils.sample(f, r))
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
