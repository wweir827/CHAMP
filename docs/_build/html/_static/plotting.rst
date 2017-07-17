.. CHAMP documentation master file, created by
   sphinx-quickstart on Tue Jul 11 15:50:43 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Visualizing Results
********************

=================================
Plotting Results
=================================
The CHAMP package offers a number of ways to visualize the results for both single layer and multilayer \
networks.


Single Layer Plots
___________________

.. autofunction:: champ.plot_line_coefficients
.. autofunction:: champ.plot_single_layer_modularity
.. autofunction:: champ.create_similarity_heatmap_single_layer

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
    f,axarray=plt.subplots(1,3,figsize=(9,3))
    champ.plot_line_coefficients(coeffs,axarray[0])
    champ.plot_single_layer_modularity(ind_2_dom,axarray[1])
    champ.create_similarity_heatmap_single_layer(ind_2_dom,axarray[2])

    plt.show()


Output\:

.. _`example1_out`:
.. image::  images/example_2d.jpg
   :width: 50%



Multiayer Plots
___________________

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


Output\:

.. _`example1_out`:
.. image::  images/example_2d.jpg
   :width: 50%

Indices


* :ref:`genindex`
* :ref:`search`

