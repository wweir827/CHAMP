import logging
import pdb
import sys
import tempfile
import traceback
import champ
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import h5py
import pandas as pd
import igraph as ig
from time import time
import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle
import igraph as ig
import numpy as np


DESCRIPTION = ""
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"



def main():
    logging.basicConfig(format=LOG_FORMAT,
                        level=LOG_LEVEL)
    logging.info("Command: %s", " ".join(sys.argv))

    logging.info("Creating Random ER graph and partitioning")
    np.random.seed(0)
    test_graph = ig.Graph.Erdos_Renyi(n=200, p=.1)
    logging.info("Running Parallelized Louvain with 2 processors")
    ensemble = champ.parallel_louvain(test_graph, numprocesses=2, numruns=200, start=0, fin=4, maxpt=4, progress=False)
    print (ensemble)
    print (ensemble.numparts)
    print (ensemble.twin_partitions)
    print (ensemble.unique_coeff_indices.shape)
    print (ensemble.unique_partition_indices.shape)

    logging.info("Number of partitions in CHAMP: %d/%d " %( len(ensemble.ind2doms),ensemble.numparts))


    ens2 = champ.parallel_louvain(test_graph, numprocesses=2, numruns=100, start=0, fin=4, maxpt=4, progress=False)
    logging.info("merging ensembles")
    ens3 = ensemble.merge_ensemble(ens2, new=False)
    return 1

def pydebug(type, value, tb):
    logging.error("Error type:" + str(type) + ": " + str(value))
    traceback.print_tb(tb)
    pdb.pm()


if __name__ == '__main__':
    # sys.excepthook = pydebug
    sys.exit(main())
