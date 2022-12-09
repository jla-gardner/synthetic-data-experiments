# synthetic-data-experiments

Code for generating the results found in

<div align="center">

> **[Synthetic data enable experiments in atomistic machine learning](https://arxiv.org/abs/2211.16443)**\
> _[John Gardner](https://twitter.com/jla_gardner), [Zoé Faure Beaulieu](https://twitter.com/ZFaureBeaulieu) and [Volker Deringer](http://deringer.chem.ox.ac.uk)_

</div>

_(see the [sister repo](https://github.com/jla-gardner/synthetic-fine-tuning-experiments) for results pertaining to synthetic fine-tuning and pre-training)_

---

## Repo Overview

- **[results/](results/)**: the results of every model fit are stored as individual `.json` files in this directory.
- **[scripts/](scripts)**: python scripts used to run the experiments.
- **[plotting/](plotting)**: notebooks used to generate all the plots in the paper.
- **[models/](models)**: code used to create [GPR](models/gpr.py), [NN](models/nn.py) and [DKL](models/dkl.py) models respectively.

---

## Reproducing our results

### 1. Clone the repo

```bash
git clone https://github.com/jla-gardner/synthetic-data-experiments
cd synthetic-data-experiments
```

### 2. Install dependencies

We strongly reccomend using a virtual environment. With `conda` installed, this is as simple as:

```bash
conda create -n experiments python=3.8 -y
conda activate experiments
```

All dependencies can then be installed with:

```bash
pip install -r requirements.txt
```

### 3. Download the data

A sample of the full dataset already exists at `all.extxyz`. The complete dataset can be found at [this url](https://github.com/jla-gardner/carbon-data).


### 4. Run the code

The scripts for running the experiments are in `./scripts/`. To run one of these, do:
    
```bash
./run <script-name> [keyword-args]
```

e.g. `./run dkl n_max=6 l_max=6` or `./run scripts/gpr.py m_sparse=200`.
