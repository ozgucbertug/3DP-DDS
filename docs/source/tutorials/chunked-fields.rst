Chunked Fields
==============

:class:`dds.ChunkedField` stores fixed-size dense chunks only where deposition
touches the domain. It is useful for large workspaces with localized paths.

.. code-block:: python

   from dds.fields import accumulate_chunked_field

   chunked = accumulate_chunked_field(
       domain,
       deposits,
       chunk_shape=(32, 32, 32),
       include_coverage=True,
   )
   dense = chunked.to_dense("implicit")
   result = chunked.to_result(deposits, threshold=0.5)

Chunked storage is a standalone sparse workflow. Use :class:`dds.Simulator`
when you need mutable dense snapshots.
