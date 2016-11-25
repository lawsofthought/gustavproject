from mpi4py import MPI
import numpy
import utils
import models
from models import distributed_latent_sampler

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

iterations = 1000
use_bnc = True
test = False
verbose = True
sample_hyperparameters = True
load_state = True

def showtopics(rank, model, K=10):

    for i,k in enumerate(numpy.flipud(numpy.argsort(model.Sk))[:K]):

        topic_string = models.showtopics(model.S[k], 
                                         model.vocabulary,
                                         show_probability=False)

        print('Rank %d, Topic %d (%2.2f): %s' % (rank, 
                                                 k, 
                                                 model.Sk[k]/float(model.Sk.sum()),
                                                 topic_string))

if use_bnc:
    data = utils.BagOfWords('notebooks/bnc.dat', 'notebooks/bnc_vocab.txt')
else:
    data = utils.BagOfWords('notebooks/ap2.dat', 'notebooks/vocab.txt')

model = models.HierarchicalDirichletProcessTopicModel(data, 
                                                      K_min = 1000, 
                                                      K_max = 2000,
                                                      prior_gamma_scale=1.0,
                                                      parallel=False)

model.verbose = False

if RANK == 0:
    if load_state:
        model.load_state('model_261016002825_7480state.npz')

COMM.Bcast(model.x, root=0)

if RANK == 0:
    x = numpy.empty(model.N, dtype=int)
else:
    x = None

for iteration in xrange(iterations):

    if RANK == 0:
        if verbose:
            print(iteration)

    ##########################################################################

    ###########
    # Scatter #
    ###########
    
    if RANK == 0:
        index, displacements, counts = data.distribute(K=SIZE)
    else:
        index, displacements, counts = None, None, None

    counts = COMM.bcast(counts, root=0) # bcast a tuple of ints of size SIZE

    _index = numpy.empty(counts[RANK], dtype=int)

    COMM.Scatterv([index, counts, displacements, MPI.LONG], _index)

    ##########################################################################

    ##################
    # Sample latents #
    ##################
 
    _x = distributed_latent_sampler(model, _index)

    ##########################################################################

    ##########
    # Gather #
    ##########
    
    COMM.Gatherv(_x,[x, counts, displacements, MPI.LONG])

    ##########################################################################

    #############
    # Broadcast #
    #############
 
    if RANK == 0:
        model.x[index] = x

    COMM.Bcast(model.x, root=0)

    ##########################
    # Update hyperparameters #
    ##########################
    if sample_hyperparameters:

        if RANK == 0:

            model.get_counts()

            model.sample_am()
            model.sample_bpsi()
            model.sample_c()
            model.sample_gamma()

            hyperparameters = dict(a = model.a,
                                   b = model.b,
                                   c = model.c,
                                   gamma = model.gamma,
                                   m = model.m,
                                   psi = model.psi)

        else:

            hyperparameters = None

        hyperparameters = COMM.bcast(hyperparameters, root=0)

        model.update_hyperparameters(hyperparameters)

    ##############
    # Save state #
    ##############
    if RANK == 0:
        if iteration > 0 and iteration % 10 == 0:
            model.save_state()

if RANK == 0:
    model.get_S_counts()
    showtopics(RANK, model, K=100)
