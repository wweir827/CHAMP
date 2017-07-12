.. CHAMP documentation master file, created by
   sphinx-quickstart on Tue Jul 11 15:50:43 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Background
************

=================================
Introduction
=================================

CHAMP (Convex Hull of Admissible Modularity Partitions) is an algorithm to \
find the subset of an ensembles of network partitions that are optimal in terms of modularity.  Thus CHAMP is not \
a community detection algorithm *per say* but a method to assist in interpretation of a collection of partitions \
produced by ones favorite third party detection method ( *e.g.* `Louvain <http://www.traag.net/code/>`_ ).  Instead CHAMP \
identifies the partitions that have a non-empty range of the resolution parameter, :math:`\gamma` over which their modularity \
is larger than any other partition in the input ensemble.  This is done by reformulating the problem in terms of finding \
the convex hull of a set of linear subspaces and solved using the `pyhull <http://pythonhosted.org/pyhull/>`_ implementation \
of the quickhull algorithm.

CHAMP can greatly reduce the number of partitions considerable for future analyses by eliminating all partitions that are \
suboptimal across a given range of the resolution space.



==================
Modularity
==================



Indices


* :ref:`genindex`
* :ref:`search`

