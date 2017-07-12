.. CHAMP documentation master file, created by
   sphinx-quickstart on Tue Jul 11 15:50:43 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Running CHAMP
**************

=======================================
Starting from Partition Coefficients
=======================================

.. autofunction:: champ.champ_functions.get_intersection


=================================
Starting from Partitions
=================================

If the partitions were generated using a modularity based community detection method, it's better to calculate \
the coefficients while optimizing the communities and feed these into CHAMP directly.  This is especially true, \
if the community detection is being performed in parallel.  However, if the partitions were generated using some \
other form of community detection algorithm, we provide a method to compute these coefficients directly and allwo \
for parallization of this process on supported machines. 

.. autofunction:: champ.champ_functions.create_coefarray_from_partitions


Indices

* :ref:`genindex`
* :ref:`search`

