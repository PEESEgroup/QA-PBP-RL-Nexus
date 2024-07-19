# QA-PBP-RL-Nexus
This repository contains code and data to generate and validate plastic binding peptides (PBPs) for the manuscript: Developing peptide-based strategies for microplastic pollution via a nexus of biophysical modeling, quantum computing, and artificial intelligence.

## Overview
`RL_PPO_Peptides/` contains code to discover and generate peptides in solution space. 

`QA_peptide_generator/` contains code to generate optimized amino acid sequences for unique backbones

`sample_data/` contains sample data of a single body and pairwise energy required to generate peptide for the unique backbone.

`demo/` contains a full body code with sample inputs that help you generate peptides via both QA and PPO.

`Peptide design QA + RL/` folder contains the following data to generate and validate different peptides and results claimed in the manuscript. Organization of the folder is explained below:

- `Peptide Design QA+RL/AllDesigns.xlsx`: contains the best designs for each system conformation for all plastics using either quantum annealing (QA), PepBD, or Proximal Policy Optimization (PPO). There is a separate tab for the plastics polyethylene (PE), polypropylene (PP), polystyrene (PS), and polyethylene terephthalate (PET). For each system conformation, the best score and the corresponding amino acid sequence are provided. Note that for polyethylene, QA and PPO do not have solutions for all confirmations
	
- `Peptide Design QA+RL/AminoAcidFrequencies.xlsx`: contains the amino acid frequencies for QA designs for all four plastics and PepBD designs for polyethylene.
	
- `Peptide Design QA+RL/Example_PPO_Score_Trajectories`: provides two examples of the evolution of the peptide score over the course of PPO
	
- `Peptide Design QA+RL/MD_Data.xlsx`: provides the adsorption enthalpy (dH) and adsorption free energy (dG) for peptides, with values calculated using the MM/GBSA method. Separate tabs are provided for the PE, PP, PS, and PET. Each tab lists results for peptides found by QA, PepBD, PPO, or generation of a random amino acid sequence. Each entry lists the peptide amino acid sequence, the adsorption enthalpy, and adsorption-free energy.
	
- `Peptide Design QA+RL/Peptide_Properties.csv`: contains the Camsol solubility score and the net peptide charge for peptides designed by QA, PepBD, and PPO. For QA and PPO designs, the system conformation corresponding to the design is listed.
	
- `Peptide Design QA+RL/PottsModel_Energies`: Contains all one-body energies (SingleEnergy.txt) and two-body energies (Pairwise.txt) for all system conformations (or "bb"s) for PE, PP, PS, and PET
	
- `Peptide Design QA+RL/PPO_All_Sequences_All_Conformations`: Contains all solutions found by PPO for all sampled system conformations. The results for each conformation are in a separate folder named "bb". As described in the manuscript, a sampled amino acid sequence is only considered a solution if its corresponding score is within 5 of the best score found by QA
	
- `Peptide Design QA+RL/PPO_NumUnique_vs_Score.csv`: Contains the total number of solutions found per system confirmation, as well as the best score found by QA for that confirmation.
	
- `PPO_Seq_2_SideChainEnvironment_Analysis`: Contains analysis of the relationship between side chain geometric environment (SideChainEnvironment.csv) and the most frequent amino acid found in the environment (SequenceAnalysis.csv). 

- `Peptide Design QA+RL/PPO_Seq_2_SideChainEnvironment_Analysis/SequenceAnalysis.csv`: For each system conformation, list the best score found by PPO, the number of solutions found, and the most common amino acid at each of the 12 residues in the peptide

- `Peptide Design QA+RL/PPO_Seq_2_SideChainEnvironment_Analysis/SideChainEnvironment.csv`: for each system conformation, provides the distance between beta carbon and top of surface (COM), angle between alpha carbon - beta carbon vector and surface normal vector (Angle), and solvent accessible surface area (SASA) for each of the 12 residues.


## System Requirements
### Operating System
This repository has been developed and tested on macOS, ensuring full compatibility with this operating system. However, the provided scripts are designed to be platform-independent and should work seamlessly on other versions of Windows and macOS.

### Hardware Requirements
The PPO models do not require any non-standard hardware and can be run on a typical computer. The code also has provided the option to use GPU via pytorch library such that one utilizes speedup in the RL process. Quantum Annealing is performed on the Dwave quantum computer, and it requires an API token from Dwave to successfully run the QA part of the code. The code has been tested on a system with the following specifications: 

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

## Installation Guide
The source code is not distributed as a package, therefore, no installation is required.

**Clone the repository:** Clone the repository to a local directory using the following command:


`https`
```https
git clone https://github.com/PEESEgroup/QA-PBP-RL-Nexus.git
```

## Demo
Detailed examples of how to use our model to generate PBPs using PPO are provided in `demo/PPO_pbp.py.` The Python file contains the base code and basic hyperparameters required to train policy for peptide generation, and `demo/ppo_pbp_generation.py` contains code to generate peptides using the learned policy. Code is structured in a way that one can generate peptides for multiple backbones using a single run. Code also provides various windows to store metadata as a `.pkl` file, which is helpful for restarting the computational job. We have provided GPU support using Cuda as well for the code.

Detailed examples of how to use our model to generate PBPs using Quantum Annealing is provided in `demo/qa_pbp_generation.py.` The Python file contains the base code required to generate new peptides. One needs to get access to Dwave hardware via API token in order to utilize the Quantum Annealer. Code is structured in a way that one can generate peptides for multiple backbones using a single run. Code also provides various windows to store metadata as a `.pkl` file, which is helpful for restarting the computational job. 

## Instructions for Use
To reproduce the results presented in our paper, please follow these steps:
1.  Navigate to the demo/ directory and examine the readme section and provided `.py` files with sample inputs.
2.  Ensure that your environment meets the package and system specifications outlined in the instructions provided within the repository.

Note: Due to the stochastic nature of the computational models, you may not generate identical peptides in multiple runs. However, you can expect to observe the same distributions of peptide properties (such as total energy and amino acid distribution) as reported in the paper, provided the same design parameters are used. 
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
