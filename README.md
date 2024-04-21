# FoMo Project

## Setup

1. Setup required python version using your preferred method (e.g. pyenv, virtualenv, etc.). For pyenv users:

```bash
pyenv install 3.11.6
pyenv local 3.11.6
```

2. Install poetry if needed following the instructions at https://python-poetry.org/docs/#installation
3. Install dependencies:

```bash
poetry install
```

4. Set up the pre-commit hooks:

```bash
poetry run pre-commit install
```

## Datasets

For some of the datasets, you have to download them from the original source. The datasets are not included in this repository.

### Stanford Cars

Download the dataset from https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder to `data/stanford-cars`.
