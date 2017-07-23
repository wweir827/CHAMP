import logging
# import pdb
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
import cPickle as pickle
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
    times = {}
    run_nums = [100]
    ens = champ.parallel_louvain(test_graph, numprocesses=2, numruns=100, start=0, fin=4, maxpt=4, progress=False)
    logging.info("Number of partitions in CHAMP: %d/%d " %( len(ens.ind2doms),ens.numparts))

    ens.save("name_PartEnsemble_100.hdf5", hdf5=True)

    ens2 = champ.parallel_louvain(test_graph, numprocesses=2, numruns=100, start=0, fin=4, maxpt=4, progress=False)
    logging.info("merging ensembles")
    ens3 = ens.merge_ensemble(ens2, new=False)
    return 1

def pydebug(type, value, tb):
    logging.error("Error type:" + str(type) + ": " + str(value))
    traceback.print_tb(tb)
    pdb.pm()


if __name__ == '__main__':
    # sys.excepthook = pydebug
    sys.exit(main())
