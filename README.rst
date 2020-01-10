sindy
=========

**sindy** is a sparse regression package with several implementations for Sparse Identification of Nonlinear Dynamical systems (SINDy).

Installation
------------

Installing with pip
^^^^^^^^^^^^^^^^^^^

If you are using Linux or macOS you can install sindy with pip:
``pip install sindy``

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
First clone this repository:
``git clone https://github.com/briandesilva/sindy``

Then, to install the package, run:
``python setup.py install``
If you do not have root access, you should add the ``--user`` option to the above line.


Implemented algorithms
----------------------

-  Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
   “Discovering governing equations from data by sparse identification
   of nonlinear dynamical systems.” Proceedings of the National Academy
   of Sciences 113.15 (2016): 3932-3937. `DOI:
   10.1073/pnas.1517384113 <http://dx.doi.org/10.1073/pnas.1517384113>`__

Community guidelines
--------------------

Contributing code
^^^^^^^^^^^^^^^^^
We welcome contributions to sindy. To contribute a new feature please submit a pull request. To be accepted your code should conform to PEP8 (you may choose to use flake8 to test this before submitting your pull request). Your contributed code should pass all unit tests. Upon submission of a pull request, your code will be tested automatically, but you may also choose to test it yourself by running
``pytest``

Reporting issues or bugs
^^^^^^^^^^^^^^^^^^^^^^^^
If you find a bug in the code or want to request a new feature, please open an issue.

Getting help
^^^^^^^^^^^^
For help using sindy please consult the documentation and/or our examples_, or create an issue.

..examples: https://github.com/briandesilva/sindy/tree/master/example 