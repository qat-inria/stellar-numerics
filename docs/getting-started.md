# Getting Started

## Requirements

The package requires Python 3.10 or later and earlier than 3.14.

## Install from PyPI

```bash
python -m pip install stellar-numerics
```

## Install from a local checkout

From the repository root

```python
python -m pip install -e .
```

To install the documentation tools as well:

```python
python -m pip install -e ".[docs]"
```

## Build or preview the documentation

From the repository root:

```python
python -m mkdocs serve
```

Open the local address printed by MkDocs to preview the site. To build the
static site, run:

```python
python -m mkdocs build
```
