# CONSENSUS-CSF
**A compressed static function data structure using the CONSENSUS technique [[1]](#1)**.

This data repository only provides a research version of the algorithm and is not production ready.

## Flamegraph

- Install flamegraph `cargo install flamegraph`
- install perf
- run, e.g. `cargo flamegraph --example rand` \
  optionally pass `--no-inline` which shows foll module paths (but also does not resolve inlined functions)
- view `flamegraph.svg`


## References
<a id="1">[1]</a> 
H.-P. Lehmann, P. Sanders, S. Walzer, and J. Ziegler, “Combined Search and Encoding for Seeds, with an Application to Minimal Perfect Hashing,” Feb. 08, 2025, arXiv: arXiv:2502.05613. doi: 10.48550/arXiv.2502.05613.
