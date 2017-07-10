import champ_functions
import numpy as np
import matplotlib.pyplot as plt


def main():

    #create random planes
    test_hs=[]
    np.random.seed(0)


    # print test_hs
    # print test_int_dict
    test_int_dict=champ_functions.get_random_halfspaces()
    plt.close()
    ax=champ_functions.plot_2d_domains(test_int_dict)
    plt.show()
    return 1

if __name__=='__main__':

    exit(main())