###############################################
Gustav: A probabilistic topic modelling toolbox
###############################################

:Author: Mark Andrews
:Disclaimer: This is alpha software.

Gustav is a Python and Fortran based probabilistic topic modelling toolbox.

It is released under `GNU General Public License
<http://www.gnu.org/copyleft/gpl.html>`_.

The name ``Gustav`` is named after `Peter Gustav Lejeune Dirichlet`_, the 19th century German
mathematician after whom the `Dirichlet Distribution`_ and `Dirichlet Process`_ are named. Both the `Dirichlet Distribution`_ and `Dirichlet Process`_ are vital components of probabilistic topic models.

Installation
~~~~~~~~~~~~

It is recommended that Gustav is installed using `pip`_ (and in a `virtual
environment`_, though that is a matter of preference).

.. code:: bash

        make all                # compile fortran extension modules
        python setup.py test    # optional
        pip install -e . 

Warning: Alpha software
~~~~~~~~~~~~~~~~~~~~~~~~
Currently, Gustav is a alpha software.

* It implements only a minimal number of probabilistic topic models so far.
* The API to any given sampler or model may change without warning.
* Any future development is likely to be backwards incompatible.
* There is minimial documenation.
* In short, use with caution.

.. _Peter Gustav Lejeune Dirichlet: https://en.wikipedia.org/wiki/Peter_Gustav_Lejeune_Dirichlet
.. _Dirichlet Distribution: https://en.wikipedia.org/wiki/Dirichlet_distribution
.. _Dirichlet Process: https://en.wikipedia.org/wiki/Dirichlet_process
.. _pip: https://pip.pypa.io
.. _virtual environment: https://virtualenv.pypa.io
