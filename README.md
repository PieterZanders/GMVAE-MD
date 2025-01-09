# Gaussian Mixture Variational AutoEncoders (GMVAE) for MD Analysis

This repository provides a framework for analyzing molecular dynamics (MD) trajectories using Gaussian Mixture Variational AutoEncoders (GMVAE). The code is designed to process molecular dynamics (MD) trajectories, performing dimensionality reduction and identifying conformational clusters to reveal insights into the molecular system's underlying dynamics.

## Features
- Process molecular dynamics trajectories: Load and Fit Trajectory (Gaussian wRMSD).
- Performs dimensionality reduction.
- Gaussian mixture modeling for clustering.
- Optional: Time-lagged data GMVAE training in order to capture time dependencies and identify slow modes.

## Usage

You can run the main script with the following command-line arguments:

```bash
usage: main.py [-h] [--pdb_path PDB_PATH] [--xtc_path XTC_PATH] [--condition CONDITION] [--norm NORM] [--partition PARTITION] [--stride STRIDE] [--fit_traj] [--ref_pdb REF_PDB] [--sfactor SFACTOR]
               [--hyperparameters HYPERPARAMETERS] [--time_lag TIME_LAG] [--load_model LOAD_MODEL] [--train_model] [--seed SEED]
```

### Arguments:
- `--pdb_path PDB_PATH` : Path to the PDB file.
- `--xtc_path XTC_PATH` : Path to the XTC trajectory file.
- `--condition CONDITION` : Specific condition for the analysis (name of the experiment).
- `--norm NORM` : Normalization option (standard | minmax).
- `--partition PARTITION` : Partitioning scheme for the data.
- `--stride STRIDE` : Stride value of the trajectory frames.
- `--fit_traj` : Flag to fit the trajectory: Gaussian wRMSD.
- `--ref_pdb REF_PDB` : Path to the reference PDB file for fitting.
- `--sfactor SFACTOR` : Scaling factor for the Gaussian wRMSD.
- `--hyperparameters HYPERPARAMETERS` : Path to the GMVAE hyperparameters configuration file.
- `--time_lag TIME_LAG` : Time lag for time-lagged GMVAE.
- `--load_model LOAD_MODEL` : Path to load a pre-trained model.
- `--train_model` : Flag to train the model.
- `--seed SEED` : Random seed for reproducibility.

## Dependencies

The following dependencies are required to run the code:

- `pytorch=2.3.0`
- `numpy=1.26.4`
- `mdtraj=1.9.9`
- `mdanalysis=2.7.0`
- `sklearn=1.5.0`
- `scipy=1.13.0`
- `matplotlib=3.8.4`
- `tqdm=4.66.4`

## Background

This implementation is an updated version of the GMVAE developed by Yasemin Bozkurt Varolgunes. The original repository can be found [here](https://github.com/yabozkurt/gmvae) (Tensorflow 1). The updates in this repository ensure compatibility with newer versions of PyTorch.

### Reference

This project is inspired by the article:
"Interpretable embeddings from molecular simulations using Gaussian mixture variational autoencoders"
(https://doi.org/10.1088/2632-2153/ab80b7).

## Workflow

Jupyter notebooks are included as workflow examples to demonstrate the Gaussian Mixture Variational AutoEncoders (GMVAE) framework. These notebooks provide step-by-step guides on loading trajectories, preprocessing data, training models, and analyzing results.
