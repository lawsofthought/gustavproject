import numpy
from mpi4py import MPI

if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    if rank == 0:
        x = numpy.random.rand(1000)
    else:
        x = None

    data = comm.bcast(x, root=0)

    print "hello world from process ", rank, sum(data)
