import champ
import numpy as np
import pyhull
import matplotlib.pyplot as plt
import pdb,sys,traceback,logging

DESCRIPTION = ""
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"

def plot_hs_list(hs2plt,ax=None):
    if ax == None:
        f = plt.figure()
        ax = f.add_subplot(111)
    for hs in hs2plt:
        A=-1.0*hs.offset/hs.normal[0]
        B=-1.0*hs.offset/hs.normal[1]

        ax.plot( [0,A],[B,0] )
    return ax

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
    print test_hs_arry.shape
    print test_hs_arry
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
    print test_hs_arry.shape

    # plt.close()
    # for i in range(test_hs_arry.shape[0]):
    #     plt.plot([0,test_hs_arry[i,0]],[test_hs_arry[i,1],0])
    #
    # plt.show()

    print test_hs_arry
    test_hs = champ.create_halfspaces_from_array(test_hs_arry)
    plt.close()
    plot_hs_list(test_hs)
    plt.show()
    # intpt=champ.get_interior_point(test_hs)
    # qhout=pyhull.qhalf('Fp',test_hs,intpt)
    logging.info("Number of Initial Partitions: %d" % (len(test_hs)))
    ind_2_doms = champ.get_intersection(test_hs_arry,max_pt=(10,10))
    pdb.set_trace()
    logging.info("Number of Admissible Partitions: %d" % (len(ind_2_doms.keys())))
    plt.close()
    ax=champ.plot_single_layer_modularity(ind_2_doms)
    plt.show()

    return 1

def pydebug(type, value, tb):
    logging.error("Error type:" + str(type) + ": " + str(value))
    traceback.print_tb(tb)
    pdb.pm()


if __name__ == '__main__':
    sys.excepthook = pydebug
    sys.exit(main())
