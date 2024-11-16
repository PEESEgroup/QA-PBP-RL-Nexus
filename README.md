# QA-PBP-RL-Nexus
This repository contains code and data to generate and validate plastic binding peptides (PBPs) for the manuscript: "Developing Peptide-Based Strategies for Microplastic Pollution via a Nexus of Biophysical Modeling, Quantum Computing, and Artificial Intelligence."


## Overview
`RL_PPO_Peptides/` contains code to run PPO to discover high affinity peptides for a system conformation and associated Potts model.  

`QA_peptide_generator/` contains code to run QA to discover high affinity peptides for a system conformation and associated Potts model. 

`demo/` contains code, example inputs, and example outputs for peptide discovery with QA and PPO.

`Data Folder/` contains the following data to generate and validate different peptides and results claimed in the manuscript. The organization of the folder is explained below:

- `Data Folder/PottsModel_Energies.zip`: contains zip file with all one-body energies (SingleEnergy.txt) and two-body energies (Pairwise.txt) for all system conformations (or "bb"s) for PE, PP, PS, and PET.

- `Data Folder/PPO_All_Sequences_All_Conformations`: contains all solutions found by PPO for all sampled system conformations. The results for each conformation are in a separate folder named "bb". As described in the manuscript, a sampled amino acid sequence is only considered a solution if its corresponding score is within 5 of the best scores found by QA.

- `Data Folder/Peptide Design with QA+RL PNAS Nexus SI Data File.xlsx`: contains all essential data supporting the SI files. The overview sheet provides a summary of the contents and explains the purpose of each subsequent sheet.

## System Requirements
### Operating System
This repository has been developed and tested on macOS, ensuring full compatibility with this operating system. However, the provided scripts are designed to be platform-independent and should work seamlessly on other versions of macOS and Windows.

### Hardware Requirements
The PPO models do not require any non-standard hardware and can be run on a typical computer. The code also provides the option to utilize a GPU via the PyTorch library for speedup in the reinforcement learning process. Quantum Annealing is performed on the D-Wave quantum computer. An API token from D-Wave is required to successfully run the QA part of the code.

The code has been tested on a system with the following specifications: 

- Apple M1 Pro
- 16GB of RAM
- macOS Sonoma

### Package Requirements
- python 3.9.7
- pandas 2.0.0
- pytorch 2.0.0
- numpy 1.24.2
- dimod 0.10.15
- dwave-hybrid 0.6.10
- tqdm 4.64.1
- openpyxl 3.1.2

## Installation Guide
The source code is not distributed as a package, therefore, no installation is required.

**Clone the repository:** Clone the repository to a local directory using the following command:


`https`
```https
git clone https://github.com/PEESEgroup/QA-PBP-RL-Nexus.git
```

## Demo
Detailed examples of how to use our model to generate PBPs using Proximal Policy Optimization (PPO) are provided in `demo/PPO_pbp.py`; This Python file contains the base code and basic hyperparameters required to train the policy for peptide generation. Additionally, `demo/ppo_pbp_generation.py` contains the code to generate peptides using the learned policy. GPU support using CUDA is provided to enhance performance, and we have attempted compatibility with PyTorch for efficient GPU utilization.

Detailed examples of how to use our model to generate PBPs using Quantum Annealing are provided in `demo/qa_pbp_generation.py`. This Python file contains the base code required to generate new peptides. Access to D-Wave hardware via an API token is necessary to utilize the Quantum Annealer. For demo purposes, we have used a basic sampler to replicate the workflow.

## Instructions for Use
To reproduce the results presented in our paper, please follow these steps:
1.  Navigate to the demo/ directory and examine the readme section and provided `.py` files with sample inputs.
2.  Ensure that your environment meets the package and system specifications outlined in the instructions provided within the repository.

Note: Due to the stochastic nature of the computational models, you may not generate identical peptides in multiple runs. However, you can expect to observe the same distributions of peptide properties (such as total energy and amino acid distribution) as reported in the paper, provided the same design parameters are used. 
## Citation

```
@article{,
author = {Jeet Dhoriyani, Michael T. Bergman, Carol K. Hall, and Fengqi You},
title = {Integrating biophysical modeling, quantum computing, and AI to discover plastic-binding peptides that combat microplastic pollution},
journal = {},
volume = {},
number = {},
pages = {},
doi = {},
abstract = {}}
```
