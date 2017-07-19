CHAMP (Convex Hull of Admissible Modularity Partitions)
==========================================================
by William Weir



.. figure::  docs/_static/images/graphs_all_three.png
   :align:   center
   :figwidth: 95%

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

+ `NumPy <https://www.scipy.org/scipylib/download.html>`_
+ `sklearn <http://scikit-learn.org/stable/install.html>`_
+ `igraph <http://igraph.org/python/#downloads>`_
+ `matplotlib <https://matplotlib.org/users/installing.html>`_
+ `louvain <https://github.com/vtraag/louvain-igraph>`_



Citation
___________
Please cite\:

William H. Weir, Scott Emmons, Ryan Gibson, Dane Taylor, and Peter J Mucha. Post-processing partitions to identify domains of modularity optimization. arXiv.org, 2017. arXiv:1706.03675

`bibtex <docs/_static/champ.bib>`_



