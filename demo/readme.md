`demo/sample_data`: This folder consists of sample data of PairwiseEnergy and SingleEnergy for demo folder. It serves as input to demo files to generate peptides using QA and PPO.

`demo/sample_data/PPO_Policy.pth` : This file consists of sample policy already trained for the demo purpose. User recommended to use their own file to genereate peptide.

`demo/PPO_pbp.py`: PPO_pbp.py consists of a demo script with sample input to train policy for peptide design. 

`demo/ppo_pbp_generation.py`: This file consists demo script to generate peptides using PPO-policy.

`demo/qa_pbp_generation.py`: This file consists demo to generate peptides using Quantum annealing. For the demo purpose we are using a random sampler to demonstrate the flow of the peptide design.

