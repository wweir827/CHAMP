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

    logging.info("Multilayer Test")
    # create random planes
    test_hs_array = champ.get_random_halfspaces(50)
    logging.info("Coefficent array: " + str(test_hs_array.shape))

    test_hs = champ.create_halfspaces_from_array(test_hs_array)
    # test_hs_array= np.array([[ 0.01952282, -0.31423295,  0.7157867 ],
    #    [-0.25143036,  0.01881523,  0.50600329],
    #    [ 0.35820641, -0.39882791,  0.50962841],
    #    [ 0.29615668, -0.01859933,  0.39495284],
    #    [ 0.28248495, -0.3065654 ,  0.53979975],
    #    [ 0.19956666, -0.04699647,  0.41592189],
    #    [-0.22016126,  0.00506596,  0.74269582],
    #    [ 0.03799577, -0.04149145,  0.38461615],
    #    [-0.28777492,  0.21594638,  0.65241822],
    #    [ 0.01847376,  0.28849111,  0.17194509]])

    logging.info("Number of Initial Partitions: %d" % (len(test_hs)))
    ind_2_doms = champ.get_intersection(test_hs_array)
    logging.info("Number of Admissible Partitions: %d" % (len(ind_2_doms.keys())))
    # plot domain by domain
    # for i,dom in ind_2_doms.items():
    #     plt.close()
    #     champ.plot_2d_domains(dict([(i,dom)]))
    #     plt.show()
    plt.close()
    ax = champ.plot_2d_domains(ind_2_doms)
    # print ind_2_doms
    plt.show()

    logging.info("Single-layer Test")
    test_hs_arry = champ.get_random_halfspaces(100, dim=2)

    # plt.close()
    # for i in range(test_hs_arry.shape[0]):
    #     plt.plot([0,test_hs_arry[i,0]],[test_hs_arry[i,1],0])
    #
    # plt.show()
    test_hs = champ.create_halfspaces_from_array(test_hs_arry)
    logging.info("Number of Initial Partitions: %d" % (len(test_hs)))
    ind_2_doms = champ.get_intersection(test_hs_arry, max_pt=10)
    logging.info("Number of Admissible Partitions: %d" % (len(ind_2_doms.keys())))

    plt.close()
    f, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4))

    a1 = champ.plot_domains.plot_line_halfspaces(test_hs, ax=a1)
    a1.set_title("Visualization of All Partition Lines")
    a2 = champ.plot_single_layer_modularity_domains(ind_2_doms, ax=a2)
    plt.show()

    return 1

def pydebug(type, value, tb):
    logging.error("Error type:" + str(type) + ": " + str(value))
    traceback.print_tb(tb)
    pdb.pm()


if __name__ == '__main__':
    sys.excepthook = pydebug
    sys.exit(main())
