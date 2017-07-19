import logging
import pdb
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np

import champ

DESCRIPTION = ""
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"



def main():
    logging.basicConfig(format=LOG_FORMAT,
                        level=LOG_LEVEL)
    logging.info("Command: %s", " ".join(sys.argv))
    #create random planes
    test_hs=[]
    np.random.seed(0)

    # print test_hs
    # print test_int_dict

    logging.info("Multilayer Test")
    test_hs_arry=champ.get_random_halfspaces(50)
    logging.info("Coefficent array: ", str(test_hs_arry.shape))

    test_hs=champ.create_halfspaces_from_array(test_hs_arry)

    logging.info("Number of Initial Partitions: %d" %(len(test_hs)) )
    ind_2_doms=champ.get_intersection(test_hs_arry)
    logging.info("Number of Admissible Partitions: %d" %(len(ind_2_doms.keys())))
    #plot domain by domain
    # for i,dom in ind_2_doms.items():
    #     plt.close()
    #     champ.plot_2d_domains(dict([(i,dom)]))
    #     plt.show()
    plt.close()
    ax=champ.plot_2d_domains(ind_2_doms)
    # print ind_2_doms
    plt.show()

    logging.info("Single-layer Test")
    test_hs_arry = champ.get_random_halfspaces(100,dim=2)

    # plt.close()
    # for i in range(test_hs_arry.shape[0]):
    #     plt.plot([0,test_hs_arry[i,0]],[test_hs_arry[i,1],0])
    #
    # plt.show()
    test_hs = champ.create_halfspaces_from_array(test_hs_arry)
    plt.close()
    ax=champ.plot_domains.plot_line_halfspaces(test_hs_arry)
    ax.set_title("Visualization of All Parition Lines")
    plt.show()

    logging.info("Number of Initial Partitions: %d" % (len(test_hs)))
    ind_2_doms = champ.get_intersection(test_hs_arry,max_pt=(10,10))
    logging.info("Number of Admissible Partitions: %d" % (len(ind_2_doms.keys())))
    plt.close()
    ax=champ.plot_single_layer_modularity_domains(ind_2_doms)
    plt.show()

    return 1

def pydebug(type, value, tb):
    logging.error("Error type:" + str(type) + ": " + str(value))
    traceback.print_tb(tb)
    pdb.pm()


if __name__ == '__main__':
    sys.excepthook = pydebug
    sys.exit(main())
