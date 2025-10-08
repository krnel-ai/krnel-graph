Running workflows
=================


.. toctree::
   :maxdepth: 2

   cli


Runner interface
----------------
.. currentmodule:: krnel.graph.runners
.. autofunction:: Runner
.. autoclass:: BaseRunner
.. autoclass:: LocalArrowRunner
.. autoclass:: LocalCachedRunner

Operation status
----------------

.. currentmodule:: krnel.graph.runners.op_status
.. autopydantic_model:: OpStatus