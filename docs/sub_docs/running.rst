.. CHAMP documentation master file, created by
   sphinx-quickstart on Tue Jul 11 15:50:43 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

..  _running:

Running CHAMP
**************

CHAMP uses the quick hull algorithm to find the intersection of the space above all of the planes representing \
the input set of partitions as shown in :ref:`Single Layer <SingleLayer_CHAMP>` and :ref:`Multilayer <Multilayer_CHAMP>`. \
There are many great community detection tools available (both in python and other langauges) that can be used to generate the starting collection of partitions which CHAMP can be applied to.  We have incorporated a python version of the louvain algorithm into CHAMP to identify partitions on the basis of modularity maximization which CHAMP can then be applied to.  See :ref:`Louvain Extension<louvain_ext>`.

Below we detail running of CHAMP for two scenarios:

   1.  Starting from an ensemble of partitions (without the corresponding partition coefficients for calculating the convex hull).
   2.  Starting with the partitions and the coefficients precalculated.


=================================
Starting from Partitions
=================================

If the partitions were generated using a modularity based community detection method, it's better to calculate \
the coefficients while optimizing the communities and feed these into CHAMP directly.  This is especially true, \
if the community detection is being performed in parallel.  However, if the partitions were generated using some \
other form of community detection algorithm, we provide a method to compute these coefficients directly and allow \
for parallelization of this process on supported machines.

.. autofunction:: champ.champ_functions.create_coefarray_from_partitions

------------------------------------
Coeffients from Partitions Example
------------------------------------
::

    import champ
    import numpy as np
    import matplotlib.pyplot as plt
    import igraph as ig


    rand_er_graph=ig.Graph.Erdos_Renyi(n=1000,p=.05)

    for i in range(100):
        ncoms=np.random.choice(range(1,30),size=1)[0]
        if i==0:
            rand_partitions=np.random.choice(range(ncoms),replace=True,size=(1,1000))
        else:
            rand_partitions=np.concatenate([rand_partitions,np.random.choice(range(ncoms),replace=True,size=(1,1000))])

    print(rand_partitions.shape)

    #get the adjacency of ER graph
    A_mat=np.array(rand_er_graph.get_adjacency().data)
    #create null model matrix
    P_mat=np.outer(rand_er_graph.degree(),rand_er_graph.degree())

    ## Create the array of coefficients for the partitions
    coeff_array=champ.champ_functions.create_coefarray_from_partitions(A_mat=A_mat,
                                                                       P_mat=P_mat,
                                                                       partition_array=rand_partitions)

    #Calculate the intersection of all of the halfspaces.  These are the partitions that form the CHAMP set.
    ind2doms=champ.champ_functions.get_intersection(coef_array=coeff_array)
    print(ind2doms)





=======================================
Starting from Partition Coefficients
=======================================

In practice, it is often easier to calculate the coefficients while running performing the community detection \
to generate the input ensemble of partitions, especially if these partitions are being generated in parallel. \
If these have been generated already, one can apply CHAMP directly via the following call.  The same command is \
used in both the Single Layer and Multilayer context, with the output determined automatically by the number \
of coefficients supplied in the input array.

.. autofunction:: champ.champ_functions.get_intersection

------------------------------------------------
Applying CHAMP to Coefficients Array Example
------------------------------------------------
::

    import champ
    import matplotlib.pyplot as plt

    #generate random coefficent matrices
    coeffs=champ.get_random_halfspaces(100,dim=3)
    ind_2_dom=champ.get_intersection(coeffs)


    ax=champ.plot_2d_domains(ind_2_dom)
    plt.show()


Output [1]_ \:

.. _`example1_out`:
.. image::  images/example_2d.jpg
   :width: 50%



* :ref:`genindex`
* :ref:`search`

.. [1] Note that actual output might differ due to random seeding.
