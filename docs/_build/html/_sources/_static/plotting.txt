.. CHAMP documentation master file, created by
   sphinx-quickstart on Tue Jul 11 15:50:43 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Visualizing Results
********************


The CHAMP package offers a number of ways to visualize the results for both single layer and multilayer \
networks.


Single Layer Plots
___________________

.. autofunction:: champ.plot_line_coefficients
.. autofunction:: champ.plot_single_layer_modularity

---------------------
Single Layer Example
---------------------
::

    import champ
    import matplotlib.pyplot as plt


    #generate random coefficent matrices
    coeffs=champ.get_random_halfspaces(100,dim=2)
    ind_2_dom=champ.get_intersection(coeffs)

    plt.close()
    f,axarray=plt.subplots(1,2,figsize=(10,5))
    champ.plot_line_coefficients(coeffs,axarray[0])
    champ.plot_single_layer_modularity(ind_2_dom,axarray[1])
    plt.show()


Output [1]_ \:

.. _`example_sl`:
.. image::  images/example_sl.png
   :width: 90%

-----------------
Heatmap Example
-----------------

In most cases CHAMP reduces the number of considerable parttions drastically.  So much so that it is \
feasible to calculate the similarity between all pairs of paritions and visualize them ordered by their \
domains.  The easier way to do this is to wrap the input partitions into a :class:`louvain_ext.PartitionEnsemble` object \
.  Creation of the :ref:`PartitionEnsemble <louvain_ext.PartitionEnsemble>` object automatically applies CHAMP and allows access to the \
dominant partitions.

.. autofunction:: champ.plot_similarity_heatmap_single_layer


::

    import champ
    from champ import louvain_ext
    import igraph as ig
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)
    test_graph=ig.Graph.Random_Bipartite(n1=100,n2=100,p=.1)

    #parallelized wrapper
    ensemb=louvain_ext.parallel_louvain(test_graph,
                                      numruns=300,start=0,fin=4,
                                      numprocesses=2,
                                      progress=True)



    plt.close()
    a,nmi=champ.plot_similarity_heatmap_single_layer(ensemb.partitions,ensemb.ind2doms,title=True)
    plt.show()

Output [1]_ \:

|
|   Run 0 at gamma = 0.000.  Return time: 0.0275
|   Run 100 at gamma = 1.333.  Return time: 0.0716
|   Run 200 at gamma = 2.667.  Return time: 0.0717
|

.. _`example_nmi`:
.. image::  images/example_nmi.png
   :width: 70%



Multiayer Plots
___________________

In the multilayer case, each domain is a convex ploygon in the :math:`(\gamma,\omega)` plane. \


.. autofunction:: champ.plot_2d_domains

--------------------
Multilayer Example
--------------------
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
   :width: 70%




* :ref:`genindex`
* :ref:`search`

.. [1] Note that actual output might differ due to random seeding.
