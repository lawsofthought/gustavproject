import sys
from mpi4py import MPI

import os
import numpy
from gustav import models, utils

if __name__ == '__main__':

    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()

    verbose = True

    (data_filename, 
     state_filename, 
     model_id, 
     _initial_iteration, 
     _iterations,
     sample_hyperparameters) = sys.argv[1:]

    iterations = int(_iterations)
    initial_iteration = int(_initial_iteration)

    Q = numpy.load(data_filename)
    data = utils.BagOfWords.new(Q)
    state = numpy.load(state_filename)
    model = models.HierarchicalDirichletProcessTopicModel.restart(data,
                                                                  state,
                                                                  model_id,
                                                                  initial_iteration)
    if RANK == 0:
        x = numpy.empty(model.N, dtype=int)
    else:
        x = None

    for iteration in xrange(initial_iteration, initial_iteration+iterations):

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
 
    _x = models.distributed_latent_sampler(model, _index)

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
