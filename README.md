# xjd

[![PyPI - Version](https://img.shields.io/pypi/v/xjd.svg)](https://pypi.org/project/xjd)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xjd.svg)](https://pypi.org/project/xjd)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install xjd
```

## Overview

xjd (jax dag(s)) is a work-in-progress library for machine learning research built on [JAX](https://jax.readthedocs.io/en/latest/index.html).

### Pipelines

xjd is not designed to help one design *individual* machine learning models: that's what JAX is for.

xjd is one level up: it's for designing *pipelines* of such models, in such a way as to promote composability and re-use.

For instance, let's say that we want to compose a PCA-like embedding model with a GMM.

With xjd, we would first write the PCA-like embedding and the GMM as separate JAX model components, before then using xjd to compose them together into a single model.

### Contents

xjd provides:

- base class interfaces for defining re-useable JAX model components.

- an apparatus for specifying how data, parameters, and intermediate results should flow such a model.

- a simple API for model training and application.

Where we obey, as much as possible, the maxim that 'code that is read together should be written together'.

### Tuples

Everything in xjd - models, params, results - is a tuple (or ndarray), which means:

- everything plays nicely with JAX's auto grad functionality, out of the box.

- we can semi-dynamically filter out certain model execution paths (say, for training vs scoring vs testing) without messing with JAX's gradient tracing (see [here](https://jax.readthedocs.io/en/latest/errors.html#jax.errors.UnexpectedTracerError)).

### Work in progress

As mentioned above, this is still a very much work-in-progress project, that I'm currently refactoring out / rewriting from our main code base at [Haven Cove](https://havencove.com/) (where most of our research code was originally written in pytorch).

The test folder is likely the best place to start for an idea of how everything works.

Note, the package for now also includes some other convenience utilities from my day job (until I can find a more appropriate long term home for them).

## License

`xjd` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
