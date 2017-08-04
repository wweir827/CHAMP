.. CHAMP documentation master file, created by
   sphinx-quickstart on Tue Jul 11 15:50:43 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CHAMP's documentation!
=================================

.. figure::  docs/_static/images/graphs_all_three.png
   :align:   center
   :figwidth: 95%

The CHAMP python package provides two levels of functionality:

* Identifying the subset of partitions from a group of partitions (regardless of how they were discovered) with optimal modularity. See `Running <docs/_static/running.rst>`_.
* Parallelized implementation of modularity based community detection method, `louvain <https://github.com/vtraag/louvain-igraph>`_ with efficient filtering (*ala* CHAMP), management, and storage of the generated partitions. See `Louvain Parallel Extension <docs/_static/louvain_ext.rst>`_ .

For complete documentation, please visit our ReadTheDocs page: \
 `http://champ.readthedocs.io/en/latest/ <http://champ.readthedocs.io/en/latest/>`_



Download and Installation:
____________________________

The CHAMP module is hosted on `PyPi <https://pypi.python.org/pypi/champ>`_.  The easiest way to install is \
via the pip command::

    pip install champ


For installation from source, the latest version of champ can be downloaded from GitHub\:

    `<https://github.com/wweir827/CHAMP>`_

For basic installation:

.. code-block:: bash

    python setup.py install

Dependencies
***************

Most of the dependencies for CHAMP are fairly standard tools for data analysis in Python, with the exception of
`louvain_igraph <https://github.com/vtraag/louvain-igraph>`_.   They include :

+ `NumPy <https://www.scipy.org/scipylib/download.html>`_ \: Python numerical analysis library.
+ `sklearn <http://scikit-learn.org/stable/install.html>`_ \:Machine learning tools for python.
+ `python-igraph <http://igraph.org/python/#downloads>`_ \:igraph python version for manipulation of networks.
+ `matplotlib <https://matplotlib.org/users/installing.html>`_ \:Python data visualization library.
+ `louvain <https://github.com/vtraag/louvain-igraph>`_ \:Vincent Traag's implementation of louvain algorithm.
+ `h5py <https://pypi.python.org/pypi/h5py>`_ \: HDF5 file format library for ptyhon.

These should all be handled automatically if using pip to install.

Citation
___________
Please cite\:

William H. Weir, Scott Emmons, Ryan Gibson, Dane Taylor, and Peter J Mucha. Post-processing partitions to identify domains of modularity optimization. arXiv.org, 2017. arXiv:1706.03675

`bibtex <docs/_static/champ.bib>`_

