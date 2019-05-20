
import logging
import os,re
import sys

import champ
import pandas as pd
import pickle,gzip
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


def test_multilayer_on_lawyer():
    # We read in each of the seperate layers
    data_dir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/modularity_domains/CHAMP/example_notebook/data"
    lawyer_dir = os.path.join(data_dir, "LazegaLawyers/")
    attributes = pd.read_table(os.path.join(lawyer_dir, 'ELattr.dat'), sep=' ', index_col=0, header=None)
    print('attributes', attributes.shape)
    attributes.index = attributes.index - 1
    attributes.columns = ['status', 'gender', 'office', 'years', 'age', 'practice', 'law_school']
    work = pd.read_table(os.path.join(lawyer_dir, 'ELwork.dat'), sep=' ', header=None)
    print ('work:{}'.format(work.shape))
    advice = pd.read_table(os.path.join(lawyer_dir, 'ELadv.dat'), sep=' ', header=None)
    print ('advice:{}'.format(advice.shape))
    friend = pd.read_table(os.path.join(lawyer_dir, 'ELfriend.dat'), sep=' ', header=None)
    print ('friend:{}'.format(friend.shape))

    layer2name = {0: 'work', 1: 'advice', 2: 'friend'}
    n = work.shape[0]

    # Here we represent intralayer in a single "supra-adjacency"
    super_adj = np.zeros((3 * n, 3 * n))
    super_adj[:n, :n] = work.values
    super_adj[n:2 * n, n:2 * n] = advice.values
    super_adj[2 * n:3 * n, 2 * n:3 * n] = friend.values

    #
    inter_elist = [(i, i + n) for i in range(n)] + \
                  [(i + n, 2 * n + i) for i in range(n)] + \
                  [(2 * n + i, i) for i in range(n)]

    C = np.zeros((3 * n, 3 * n))
    for i, j in inter_elist:
        C[i, j] = 1
        C[j, i] = 1
    # C+=C.T
    layer_vec = np.array([i // n for i in range(3 * n)])
    mult_part_ens = champ.parallel_multilayer_louvain_from_adj(intralayer_adj=super_adj,
                                                               interlayer_adj=C, layer_vec=layer_vec,
                                                               progress=True, numprocesses=2,
                                                               inter_directed=False, intra_directed=False,
                                                               gamma_range=[0, 4], ngamma=5,
                                                               omega_range=[0, 4], nomega=5, maxpt=[4, 10])
    return 0
if __name__ == '__main__':
    # sys.excepthook = pydebug
    sys.exit(test_multilayer_on_lawyer())
