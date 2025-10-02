krnel-graph documentation
=========================

Krnel-graph is a lightweight Python library for building **strongly typed content-addressable computation graphs**, especially for mechanistic interpretability research.

Think of Krnel-graph as **"git for ML data transformations"** - every operation has a content hash of its parameters and dependencies. Results are cached and you can reproduce any computation exactly.

Krnel-graph is **unopinionated** and **implementation-agnostic.** Each operation's definition contains everything needed to materialize that operation, and each Runner can implement each operation differently. This lets you **swap in different backends**, dataflow executors, orchestrators, etc.


Features
--------

* **Type-safe computation graphs**: Build complex ML pipelines with strong typing
* **Dataset operations**: Load, transform, and manipulate datasets with ease
* **Machine learning**: Train classifiers, extract LLM activations, generate text
* **Visualization**: Create UMAP embeddings and interactive visualizations
* **Flexible execution**: Run locally or on distributed systems



.. toctree::
   :maxdepth: 2

   concepts
   types
   mech-interp/index
   runners
   graph-specification
   extending
   examples

   genindex
   modindex