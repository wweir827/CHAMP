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
a community detection algorithm *per* *say* but a method to assist in interpretation of a collection of partitions \
produced by ones favorite third party detection method ( *e.g* Louvain, SBM, Infomap *etc.* ).  Instead CHAMP \
identifies the partitions that have a non-empty range of the resolution parameter, :math:`\gamma` over which their modularity \
is larger than any other partition in the input ensemble.  This is done by reformulating the problem in terms of finding \
the convex hull of a set of linear subspaces and solved using the `pyhull <http://pythonhosted.org/pyhull/>`_ implementation \
of the quickhull :cite:`Barber:1996iv` algorithm.

CHAMP can greatly reduce the number of partitions considerable for future analyses by eliminating all partitions that are \
suboptimal across a given range of the resolution space.  The CHAMP package also allows for visualization of the domains \
using the matplotlib library.  Finally, the CHAMP package also includes a wrapper function for a python implementation \
of Louvain `louvain_igraph <https://github.com/vtraag/louvain-igraph>`_ in parallel over a range of resolutions.

For more details and results see our `preprint <https://arxiv.org/abs/1706.03675>`_


==================
Modularity
==================


    Each partition is represented by a line in :math:`(\gamma,Q)` domain.  CHAMP find the \
    lines that form the outer most surface.

In CHAMP, partitions are compared on the basis of modularity:

    :math:`Q(\gamma)=\frac{1}{2m}\sum_{i,j}{\left( A_{ij}-\gamma \frac{k_ik_j}{2m}\right)\delta(c_i,c_j)}\,,`

Each partition is represented by a line in the :math:`(\gamma,Q)` space that is parameterized by two values:

.. _`single_param`:
.. math::

    \begin{array}
    \hat{A}=\sum{A_{ij}\delta(c_i,c_j)} &\textit{Sum of edges internal to communities}\\
    \hat{P}=\sum{P_{ij}\delta(c_i,c_j)} &\textit{Expected number of edges internal to communities under random null model}
    \end{array}

.. _`SingleLayer_CHAMP`:
.. figure:: images/mod_map_AF.png
    :figwidth: 50%
    :align: center

`SingleLayer_CHAMP`_ depicts graphically the concept behind CHAMP.  Most of the lines lie close to but \
below the outer curve.  CHAMP identifies which partitions are part of the outer envelope of :math:`Q(\gamma)` \
and over which ranges of the resolution parameter, :math:`\gamma` they are dominant.

==================
Multilayer CHAMP
==================

One of the strengths of modularity is that it has been extended in a principled way into a variety of network topologies \
in particular the multilayer context.  The multilayer formulation :cite:`Mucha:2010vk` for modularity incorporates the interlayer \
connectivity of the network in the form of a second adjacency matrix :math:`C_{ij}`

.. math::
    :nowrap:

    \begin{equation}
    Q(\gamma)=\frac{1}{2m}\sum_{i,j}{\left( A_{ij}-\gamma \frac{k_ik_j}{2m} \
    +\omega C_{ij}\right)\delta(c_i,c_j)}
    \end{equation}

Communities in this context group nodes within the layers and across the layers.  The inclusion of the :math:`C_ij` \
boost the modularity for communites that include alot interlayer links.  There is an additional parameter, \
:math:`\omega` that tunes how much weight these interlink ties contribute to the modularity.  With the additional \
parameter, each partitions can be represented in the :math:`(\gamma,\omega,Q)` space by three coefficients. \
The two in equation :ref:`single layer coefficients<single_param>` and \:

.. math::

    \begin{array}
    \hat{C}=\sum{C_{ij}\delta(c_i,c_j)} &\textit{Sum of interlayer edges internal to communities}\\
    \end{array}


.. _`Multilayer_CHAMP`:
.. image::  images/3dplanes_example.jpg
   :width: 30%
.. _`senate_domains`:
.. image::  images/dom_weighted_nmi_senate.png
   :width: 60%

In the multilayer case, we look for the planes that define the intersection of the area above all of the planes \
as depicted in :ref:`3D Planes <Multilayer_CHAMP>`.  These domains are now 2D polygons in the :math:`(\gamma,\omega)` \
space as shown in :ref:`Domains <senate_domains>`.

References
___________

.. bibliography:: biblio.bib
    :style: plain
    :filter: docname in docnames



* :ref:`genindex`
* :ref:`search`

