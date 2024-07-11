# QA-PBP-RL-Nexus
This repository contains PPO and Quantum annealing code to generate plastic binding peptides (PBPs) for Developing peptide-based strategies for microplastic pollution via a nexus of biophysical modeling, quantum computing, and artificial intelligence

## Overview
`RL_PPO_Peptides/` contains code to discover and generate peptides in solution space.

`QA_peptide_generator/` contains code to generate optimized amino acid sequences for unique backbones

`sample_data` contains sample data of single body and pairwise energy required to generate peptide for unique backbone

`example`  contains full body code that helps you generate peptide

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
git clone https://github.com/PEESEgroup/DL-PBP-Design.git
```
