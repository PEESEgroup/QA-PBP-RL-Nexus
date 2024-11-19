`demo/sample_data` contains files to show peptide design in action. Users are recommended to use their own files to generate peptides.
  
  - `demo/sample_data/PPO_Policy.pth` is an example policy learned using demo/PPO_pbp.py
  - `demo/sample_data/PairwiseEnergy_sample.xlsx` is an example pairwise energies file in .xlsx format
  - `demo/sample_data/PairwiseEnergy.txt` is an example pairwise energies file in .txt format
  - `demo/sample_data/SingleEnergy_sample.xlsx` is an example single body energy file in .xlsx format
  - `demo/sample_data/SingleEnergy.txt` is an example single body energy file in .txt format

`demo/PPO_pbp.py` is a demo script to train a PPO policy using the files in `sample_data` 

`demo/ppo_pbp_generation.py` generates peptides using a trained PPO policy.

`demo/qa_pbp_generation.py` generates peptides using Quantum Annealing and the the files in `sample_data`. For demonstration purposes, a random sampler is used to illustrate the peptide design workflow.


