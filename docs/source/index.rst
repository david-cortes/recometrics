.. recometrics documentation master file, created by
   sphinx-quickstart on Fri Jul  9 17:26:17 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RecoMetrics
===========

Library-agnostic evaluation framework for implicit-feedback recommender systems that are based on low-rank matrix factorization models. Calculates per-user metrics based on the ranking of items produced by the model, using efficient multi-threaded routines. Also provides functions for generating train-test splits of data. Writen in C++ with interfaces for Python and R.

For more information, see the project's GitHub page:

`<https://www.github.com/david-cortes/recometrics>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Installation
=================================
Package is available on PyPI, can be installed with
::

    pip install recometrics

Functions
=========

.. autofunction:: recometrics.calc_reco_metrics

.. autofunction:: recometrics.split_reco_train_test


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
