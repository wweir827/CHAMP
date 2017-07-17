import logging
import pdb
import sys
import tempfile
import traceback

import igraph as ig
import numpy as np

import louvain_ext as le

DESCRIPTION = ""
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"



def main():
    logging.basicConfig(format=LOG_FORMAT,
                        level=LOG_LEVEL)
    logging.info("Command: %s", " ".join(sys.argv))

    logging.info("Creating Random ER graph and partitioning")
    test_graph=ig.Graph.Erdos_Renyi(200,p=.05)
    #to handle
    tfile=tempfile.NamedTemporaryFile('wb')
    test_graph.write_graphmlz(tfile.name)
    logging.info("Testing louvain_ext.run_louvain on ER graph")
    PE=le.run_louvain(tfile.name,nruns=10,gamma=1)
    np.random.seed(0)
    logging.info("Testing louvain_ext.parallel_louvain on ER graph")
    PPE=le.parallel_louvain(test_graph,numruns=10,numprocesses=2)
    logging.info("Number of partitions from parallel : %d" %( PPE.numparts))

    return 1

def pydebug(type, value, tb):
    logging.error("Error type:" + str(type) + ": " + str(value))
    traceback.print_tb(tb)
    pdb.pm()


if __name__ == '__main__':
    sys.excepthook = pydebug
    sys.exit(main())
