# QA-PBP-RL-Nexus
This repository contains PPO and Quantum annealing code to generate plastic binding peptides (PBPs) for Developing peptide-based strategies for microplastic pollution via a nexus of biophysical modeling, quantum computing, and artificial intelligence

## Overview
`RL_PPO_Peptides/` contains code to discover and generate peptides in solution space.

`QA_peptide_generator/` contains code to generate optimized amino acid sequences for unique backbones

`sample_data` contains sample data of single body and pairwise energy required to generate peptide for unique backbone

`demo`  contains full body code that helps you generate peptide

## System Requirements
### Operating System
This repository has been developed and tested on macOS, ensuring full compatibility with this operating system. However, the provided scripts are designed to be platform-independent and should work seamlessly on other versions of Windows and macOS.

### Hardware Requirements
The PPO models do not require any non-standard hardware and can be run on a typical computer. The code also has provided the option to use GPU via pytorch library such that one utilizes speedup in the RL process. Quantum Annealing is performed on the Dwave quantum computer, and it requires an API token from Dwave to successfully run the QA part of the code. The code has been tested on a system with the following specifications: 

- Apple M1 Pro
- 16GB of RAM
- macOS Sonora

### Package Requirements
- python 3.7
- pandas 2.0.0
- pytorch 2.0.0
- numpy 1.24.2

## Installation Guide
The source code is not distributed as a package, therefore, no installation is required.

**Clone the repository:** Clone the repository to a local directory using the following command:

```sh
git@github.com:PEESEgroup/QA-PBP-RL-Nexus.git
```
## Demo
Detailed examples of how to use our model to generate PBPs using PPO are provided in `demo/PPO_pbp.py.` The Python file contains the base code and basic hyperparameters required to generate new peptides. Code is structured in a way that one can generate peptides for multiple backbones using a single run. Code also provides various windows to store metadata as a `.pkl` file, which is helpful for restarting the computational job. 

Detailed examples of how to use our model to generate PBPs using Quantum Annealing is provided in `demo/qa_pbp_generation.py.` The Python file contains the base code required to generate new peptides. One needs to get access to Dwave hardware via API token in order to utilize the Quantum Annealer. Code is structured in a way that one can generate peptides for multiple backbones using a single run. Code also provides various windows to store metadata as a `.pkl` file, which is helpful for restarting the computational job. 

## Instructions for Use
To reproduce the results in our paper, please refer to the `.py` files in the `demo/` directory and follow the instructions for the packages and systems specification.

Please note that due to the stochastic nature of the computational models, it is impossible to generate exactly identical peptides in multiple runs, but the same distributions of peptide properties (total energy, distribution of the amino acid) in the paper are expected to be observed with the same design parameters. 
## Citation

```
@article{,
author = {Jeet Dhoriyani, Michael T. Bergman, Carol K. Hall, and Fengqi You},
title = {Developing peptide-based strategies for microplastic pollution via a nexus of biophysical modeling, quantum computing, and artificial intelligence},
journal = {},
volume = {},
number = {},
pages = {},
doi = {},
abstract = {}}
```
