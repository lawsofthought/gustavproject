from __future__ import division, absolute_import

from numpy import zeros, ones, array, unique, arange, savez_compressed
from numpy.random import rand, permutation, shuffle
import numpy
from itertools import cycle
from collections import Counter, defaultdict

import cPickle as pickle

from .samplers import utils


def sliceit(N, K):
    return zip(list(numpy.arange(K) * (N//K)), list(numpy.arange(1, K) * (N//K)) + [N-1])

class SparseCountMatrix(object):

    @classmethod
    def new(cls, text_filename, vocabulary_filename):

        sparse_count_matrix = cls(text_filename, vocabulary_filename)
    
        return sparse_count_matrix.ijv

    def __init__(self, text_filename, vocabulary_filename):
        
        self.make_vocabulary(vocabulary_filename)
        self.get_ijv(text_filename)
 
    def get_ijv(self, text_filename, word_sep='|'):

        texts = open(text_filename).read().strip().split('\n')
        
        J = len(texts)
        
        counts = defaultdict(int)
        text_counts = defaultdict(int)
            
        for j, text in enumerate(texts):

            for word in text.strip().split(word_sep):
                try:
                    counts[(j, self.word_to_index[word])] += 1
                    text_counts[j] += 1
                except KeyError:
                    # Ignore words not in vocabulary
                    pass
                
        rows = []
        cols = []
        values = []
        for j,v in counts:
            rows.append(j)
            cols.append(v)
            values.append(counts[(j,v)])

        skdot = []
        for j in text_counts:
            skdot.append(text_counts[j])
            
        data_tuple = tuple(map(numpy.array, 
                               [rows, cols, values, skdot])) + (J, self.V, self.vocabulary)

        self.ijv = dict(zip('rows cols values skdot J V vocabulary'.split(), 
                            data_tuple))
                
    def make_vocabulary(self, vocabulary_filename):

        self.vocabulary = open(vocabulary_filename).read().strip().split('\n')
        self.V = len(self.vocabulary)

        self.index_to_word = dict()
        self.word_to_index = dict()

        for i, word in enumerate(self.vocabulary):
            self.index_to_word[i] = word
            self.word_to_index[word] = i


class BagOfWords(object):

    @classmethod
    def new(cls, Q):

        N, J, w, z, Nj, lims, vocabulary = (int(Q['N']), 
                                            int(Q['J']), 
                                            Q['w'], 
                                            Q['z'], 
                                            Q['Nj'], 
                                            Q['lims'], 
                                            list(Q['vocabulary']))

        return cls(z, w, N, Nj, J, vocabulary, lims)


    def make_vocabulary(self, vocabulary):

        self.vocabulary = vocabulary
        self.V = len(self.vocabulary)

        self.index_to_word = dict()
        self.word_to_index = dict()

        for i, word in enumerate(self.vocabulary):
            self.index_to_word[i] = word
            self.word_to_index[word] = i

    def __init__(self, z, w, N, Nj, J, vocabulary, lims):

        self.make_vocabulary(vocabulary)

        self.z = z
        self.w = w
        assert N == len(self.z)
        self.N = N
        assert sum(Nj) == N
        assert len(Nj) == J
        self.J = J
        self.Nj = Nj

        self.lims = lims

    def _doc_indices(self, j):
        return arange(self.lims[j][0], self.lims[j][0]+self.lims[j][1]) 

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


class BagOfWordsFactory(object):

    @classmethod
    def new(cls, text_filename, vocabulary_filename, save_filename, word_sep='|'):
        
        data = cls(text_filename, vocabulary_filename, word_sep)

        data.save(save_filename)

    def make_lda_formatted_corpus(self, text_filename, word_sep='|'):

        texts = open(text_filename).read().strip().split('\n')

        self.lda_formatted_corpus = []
        for text in texts:

            counts = defaultdict(int)
            for word in text.strip().split(word_sep):
                try:
                    counts[self.word_to_index[word]] += 1
                except KeyError:
                    pass
            
            token_counts = sorted(counts.iteritems(), key=lambda items: items[0])

            self.lda_formatted_corpus.append(token_counts)

        self.J = len(self.lda_formatted_corpus)

    def make_vocabulary(self, vocabulary_filename):

        self.vocabulary = open(vocabulary_filename).read().strip().split('\n')
        self.V = len(self.vocabulary)

        self.index_to_word = dict()
        self.word_to_index = dict()

        for i, word in enumerate(self.vocabulary):
            self.index_to_word[i] = word
            self.word_to_index[word] = i

    def save(self, filename):

        Q = dict(w=self.w,
                 z=self.z,
                 N=self.N,
                 Nj=self.Nj,
                 J=self.J,
                 vocabulary=self.vocabulary,
                 lims=self.lims)

        savez_compressed(filename, **Q)

    def __init__(self, text_filename, vocabulary_filename, word_sep='|'):

        self.make_vocabulary(vocabulary_filename)
        self.make_lda_formatted_corpus(text_filename, word_sep=word_sep)

        all_tokens = dict()

        z = []
        w = []

        ii = 0
        start_doc_index = {}
        self.Nj = {}
        for doc_id, document in enumerate(self.lda_formatted_corpus):

            start_doc_index[doc_id] = ii

            nj = 0
            for token, count in document:

                for _ in xrange(count):

                    z.append(doc_id)
                    w.append(token)

                    ii += 1
                    nj += 1

                all_tokens[token] = None

            self.Nj[doc_id] = nj

        assert self.V == len(self.vocabulary) == len(all_tokens) == (max(w) + 1)

        assert len(z) == len(w)

        assert len(self.lda_formatted_corpus) == (max(z)+1) == (max(self.Nj.keys()) + 1) == (max(start_doc_index) + 1)

        self.w = numpy.array(w)
        self.z = numpy.array(z)

        self.N = len(self.z)

        self.lims = array([(start_doc_index[j], self.Nj[j]) for j in
                           xrange(self.J)])

        self.Nj = [self.Nj[j] for j in xrange(self.J)] # From dict to list


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


def dump(obj, filename):
    
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=2)

def load(filename):

    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj

