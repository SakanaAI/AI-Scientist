# Quantum Chemistry AI Template

This template focuses on the acceleration of quantum chemistry computations using AI. The MACE package, which implements neural network potentials, is used to calculate energies and forces in molecular simulations.

While MACE-type neural networks are great at predicting energies and forces, their use in molecular dynamics (MD) simulations uncovers meaningful connections between the specific architecture of MACE networks and the success of simulations. This example serves as a starting point for open-ended exploration of this relationship.

## Datasets

You can download the required datasets from this [GitHub repository](https://github.com/karsar/molecular_data/archive/refs/heads/main.zip).

After downloading, place the folder named after each respective molecule into the `../templates/MACE/` directory.

## Example Paper

An example of a generated paper can be found [here](https://drive.google.com/file/d/1G_0QDmuBCVzbUGPTvWCSXWEVql9A8MSX/view?usp=sharing).

## Installation

You need to install the following additional packages to run the template:

```bash
pip install mace-torch
pip install MDAnalysis
pip install statsmodels

Depending on your hardware you may need to increase timeout in run_experiment(folder_name, run_num, timeout=7200) function. 
