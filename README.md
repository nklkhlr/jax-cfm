# `JAX-CFM`: Conditional Flow Matching in JAX

## Overview

This package is a `JAX`-based implementation of Conditional Flow Matching 
(CFM) - an approach for generative modelling based on continuous normalizing 
flows. The API design of this package is closely tied to that of the 
[`TorchCFM`](https://github.com/atong01/conditional-flow-matching/) library to
allow users used to `TorchCFM` who want to migrate to `JAX` an easy transition.

This repository is currently under construction and thus may not be bug-free
or complete at this point.

## Installation


To install `JAX-CFM` clone this repository and run `pip install .` in an
environment with a `python` version >= 3.10.

If you intend to contribute or run examples, please consider installing 
with optional packages as well (e.g. `pip install .[dev]` or 
`pip install .[examples]`).

Eventually, the goal is to make the package available on PyPi.

## Dependencies

`JAX-CFM` relies on [`ott-jax`](https://github.com/ott-jax/ott) for Optimal
Transport-related tasks and uses 
[`equinox`](https://github.com/patrick-kidger/equinox) and 
[`jaxtyping`](https://github.com/google/jaxtyping) for API
design and type annotations and checking.
