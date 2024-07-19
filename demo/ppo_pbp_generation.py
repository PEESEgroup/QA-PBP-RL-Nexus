import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os

# Constants
NUM_AMINO_ACIDS = 19
SEQUENCE_LENGTH = 12

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Amino acids dictionary
AMINO_ACIDS = {0: 'G', 1: 'I', 2: 'L', 3: 'M', 4: 'F', 5: 'W', 6: 'Y', 7: 'V', 8: 'R', 9: 'K', 10: 'S', 11: 'T', 12: 'N', 13: 'Q', 14: 'H', 15: 'A', 16: 'C', 17: 'D', 18: 'E'}

# Policy network class
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, embedding_size):
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(num_inputs, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_outputs)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return F.log_softmax(x, dim=-1)
    
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.int64).unsqueeze(0).to(device)
        log_probs = self.forward(state)
        action_probs = F.softmax(log_probs, dim=-1).detach()
        action = torch.multinomial(action_probs, num_samples=1)
        return action.item(), log_probs[0, action.item()]

def generate_peptide(policy, num_steps):
    policy.eval()
    state = np.random.choice(NUM_AMINO_ACIDS, SEQUENCE_LENGTH)
    for _ in range(num_steps):
        action, _ = policy.get_action(state)
        position = action // NUM_AMINO_ACIDS
        amino_acid = action % NUM_AMINO_ACIDS
        state[position] = amino_acid
    return state

def generate_multiple_peptides(policy_path, num_steps, num_peptides, output_file):
    policy = PolicyNetwork(NUM_AMINO_ACIDS, NUM_AMINO_ACIDS * SEQUENCE_LENGTH, 256, 128).to(device)
    policy.load_state_dict(torch.load(policy_path))
    generated_peptides = [generate_peptide(policy, num_steps) for _ in range(num_peptides)]
    generated_peptide_strs = ["".join([AMINO_ACIDS[aa] for aa in peptide]) for peptide in generated_peptides]
    peptides_df = pd.DataFrame(generated_peptide_strs, columns=["Peptide"])
    peptides_df.to_csv(output_file, index=False)
    for peptide_str in generated_peptide_strs:
        print(peptide_str)

if __name__ == "__main__":
    policy_path = 'demo/sample_data/PPO_Policy.pth'
    num_steps = 50
    num_peptides = 10
    output_file = 'generated_peptides.csv'
    generate_multiple_peptides(policy_path, num_steps, num_peptides, output_file)
