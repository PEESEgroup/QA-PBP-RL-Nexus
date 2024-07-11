import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
from collections import deque

# Constants
NUM_AMINO_ACIDS = 19
SEQUENCE_LENGTH = 12
STATE_SIZE = NUM_AMINO_ACIDS * SEQUENCE_LENGTH
ACTION_SIZE = NUM_AMINO_ACIDS * SEQUENCE_LENGTH
DATA_FOLDER = 'path/to/data/folder'  # Ensure this folder is correctly specified in your runtime environment
# Define the amino acids mapping given the example:
AMINO_ACIDS ={0: 'G', 1: 'I', 2: 'L', 3: 'M', 4: 'F', 5: 'W', 6: 'Y', 7: 'V', 8: 'R', 9: 'K', 10: 'S', 11: 'T', 12: 'N', 13: 'Q', 14: 'H', 15: 'A', 16: 'C', 17: 'D', 18: 'E'}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fetch_data(backbone_index):
    """
    Fetch single and pairwise energy values from files.
    
    Args:
    backbone_index (int): Index of the backbone for which to fetch data.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Single energy values and pairwise energy matrix.
    """
    backbone_folder = os.path.join(DATA_FOLDER, f"bb{backbone_index}")
    single_energy_file = os.path.join(backbone_folder, "path/to/single_energy_file")
    pairwise_energy_file = os.path.join(backbone_folder, "path/to/pairwise_energy_file")

    single_energy_values = pd.read_excel(single_energy_file)["Energy"].values
    pairwise_energy_values = pd.read_excel(pairwise_energy_file)["Energy"].values

    pairwise_energy_matrix = np.zeros((SEQUENCE_LENGTH, NUM_AMINO_ACIDS, SEQUENCE_LENGTH, NUM_AMINO_ACIDS))
    index = 0
    for i, j in combinations(range(SEQUENCE_LENGTH), 2):
        for k in range(NUM_AMINO_ACIDS):
            for l in range(NUM_AMINO_ACIDS):
                pairwise_energy_matrix[i, k, j, l] = pairwise_energy_values[index]
                index += 1

    return single_energy_values, pairwise_energy_matrix

def load_starting_points(index):
    """
    Load starting points from a CSV file.

    Args:
    index (int): Index to filter the starting points.

    Returns:
    List[List[int]]: List of starting points.
    """
    filename = 'path/to/starting_points_file'
    df = pd.read_csv(filename)
    df_filtered = df[df['Backbone'] == index]

    if df_filtered.empty:
        print(f"No starting points found for backbone: {index}")
        return []

    starting_points = df_filtered['Solution'].apply(lambda seq: [list(AMINO_ACIDS.values()).index(aa) for aa in seq]).tolist()
    return starting_points

def calculate_total_energy(solution, single_energy, pairwise_matrix):
    """
    Calculate the total energy of a solution.

    Args:
    solution (List[int]): List of amino acid indices.
    single_energy (np.ndarray): Array of single energies.
    pairwise_matrix (np.ndarray): 4D matrix of pairwise energies.

    Returns:
    float: Total energy of the solution.
    """
    solution = np.array(solution, dtype=int)
    total_energy = np.sum(single_energy[solution])

    for i, j in combinations(range(SEQUENCE_LENGTH), 2):
        total_energy += pairwise_matrix[i, solution[i], j, solution[j]]

    return total_energy

class PeptideEnv:
    """
    Environment for peptide folding simulation.
    """
    def __init__(self, single_energy_data, pairwise_energy_matrix, starting_points, reward_scale=1):
        self.single_energy_data = single_energy_data
        self.pairwise_energy_matrix = pairwise_energy_matrix
        self.starting_points = starting_points
        self.reward_scale = reward_scale
        self.state = self.reset()

    def reset(self):
        self.state = random.choice(self.starting_points) if self.starting_points else np.random.randint(NUM_AMINO_ACIDS, size=SEQUENCE_LENGTH)
        return self.state

    def step(self, action):
        position = action // NUM_AMINO_ACIDS
        amino_acid = action % NUM_AMINO_ACIDS
        old_energy = self.calculate_energy(self.state)
        
        self.state[position] = amino_acid
        new_energy = self.calculate_energy(self.state)
        reward = (old_energy - new_energy) * self.reward_scale
        return self.state, reward, False

    def calculate_energy(self, solution):
        return calculate_total_energy(solution, self.single_energy_data, self.pairwise_energy_matrix)

class PolicyNetwork(nn.Module):
    """
    Policy network based on GRU for generating peptide sequences.
    """
    def __init__(self, num_inputs, num_outputs, hidden_size, embedding_size):
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(num_inputs, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        return F.log_softmax(self.fc(x[:, -1, :]), dim=-1)

class ValueNetwork(nn.Module):
    """
    Value network based on GRU for evaluating peptide sequences.
    """
    def __init__(self, num_inputs, hidden_size, embedding_size):
        super(ValueNetwork, self).__init__()
        self.embedding = nn.Embedding(num_inputs, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        return self.fc(x[:, -1, :])

class Memory:
    """
    Memory for storing experiences during PPO training.
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPO:
    """
    Proximal Policy Optimization algorithm.
    """
    def __init__(self, policy_network, value_network, lr, betas, gamma, k_epochs, eps_clip):
        self.policy = policy_network
        self.value = value_network
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr, betas=betas)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        old_states = torch.tensor(memory.states, dtype=torch.long).to(device)
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(device)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(device)

        for _ in range(self.k_epochs):
            logprobs, entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - self.value(old_states).detach().squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(self.value(old_states).squeeze(), rewards) - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def main():
    indices = []   # Backbone indices to process

    # PPO Parameters
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99
    k_epochs = 4
    eps_clip = 0.2
    max_episodes = 12000
    max_timesteps = 800
    update_timestep = 2000
    log_interval = 10
    saved_interval = 500

    for index in indices:
        starting_points = load_starting_points(index)
        single_energy_data, pairwise_energy_matrix = fetch_data(index)
        env = PeptideEnv(single_energy_data, pairwise_energy_matrix, starting_points)

        memory = Memory()
        ppo = PPO(PolicyNetwork(NUM_AMINO_ACIDS, ACTION_SIZE, 256, 128), 
                ValueNetwork(NUM_AMINO_ACIDS, 256, 128), lr, betas, gamma, k_epochs, eps_clip)
        ppo.policy.to(device)
        ppo.value.to(device)

        for episode in range(max_episodes):
            state = env.reset()
            for t in range(max_timesteps):
                action, log_prob = ppo.policy.get_action(state)
                new_state, reward, done = env.step(action)
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(log_prob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                state = new_state

                if t % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()

            if episode % log_interval == 0:
                avg_reward = np.mean(memory.rewards[-log_interval:])
                print(f'Episode {episode}\tLast reward: {reward}\tAverage reward: {avg_reward}')

            if episode % saved_interval == 0:
                torch.save(ppo.policy.state_dict(), f'./PE_PPO_Policy_{index}.pth')

if __name__ == "__main__":
    main()
