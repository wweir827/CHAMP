.. CHAMP documentation master file, created by
   sphinx-quickstart on Tue Jul 11 15:50:43 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CHAMP's documentation!
=================================

.. figure::  _static/images/graphs_all_three.png
   :align:   center
   :figwidth: 95%



Contents:
__________

.. toctree::
    :maxdepth: 2

    _static/intro
    _static/running
    _static/plotting
    _static/louvain_ext


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

+ `NumPy <https://www.scipy.org/scipylib/download.html>`_
+ `sklearn <http://scikit-learn.org/stable/install.html>`_
+ `igraph <http://igraph.org/python/#downloads>`_
+ `matplotlib <https://matplotlib.org/users/installing.html>`_
+ `louvain <https://github.com/vtraag/louvain-igraph>`_



Citation
___________
Please cite\:

.. bibliography:: _static/champ.bib
    :all:
    :style: plain
    :list: none

`bibtex <_static/champ.bib>`_



* :ref:`genindex`
* :ref:`search`

