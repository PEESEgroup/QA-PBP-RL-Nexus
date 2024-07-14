import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
from collections import deque
from itertools import combinations
import pandas as pd
import os

# Constants
NUM_AMINO_ACIDS = 19
SEQUENCE_LENGTH = 12
n = 12
m = 19
STATE_SIZE = NUM_AMINO_ACIDS * SEQUENCE_LENGTH
ACTION_SIZE = NUM_AMINO_ACIDS * SEQUENCE_LENGTH

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Amino acids dictionary
AMINO_ACIDS = {0: 'G', 1: 'I', 2: 'L', 3: 'M', 4: 'F', 5: 'W', 6: 'Y', 7: 'V', 8: 'R', 9: 'K', 10: 'S', 11: 'T', 12: 'N', 13: 'Q', 14: 'H', 15: 'A', 16: 'C', 17: 'D', 18: 'E'}

# Fetch data function
def fetch_data(data_folder):
    backbone_folder = os.path.join(data_folder)
    single_energy_file = os.path.join(backbone_folder, "SingleEnergy_sample.xlsx")
    pairwise_energy_file = os.path.join(backbone_folder, "PairwiseEnergy_sample.xlsx")
    
    single_energy_values = pd.read_excel(single_energy_file)["Energy"].values
    pairwise_energy_values = pd.read_excel(pairwise_energy_file)["Energy"].values
    
    return single_energy_values, pairwise_energy_values

# Load starting points function
def load_starting_points(filename, index):
    df = pd.read_csv(filename)
    df_filtered = df[df['Backbone'] == index]
    if df_filtered.empty:
        print(f"No starting points found for backbone: {index}")
        return []
    starting_points = df_filtered['Solution'].apply(lambda seq: [list(AMINO_ACIDS.values()).index(aa) for aa in seq]).values.tolist()
    return starting_points

# Calculate total energy function
def calculate_total_energy(solution, single_energy_values, pairwise_energy_values):
    solution = np.array(solution, dtype=int)
    single_energy_matrix = single_energy_values.reshape(n, m)
    total_energy = np.sum(single_energy_matrix[np.arange(n), solution])
    pairwise_energy_matrix = np.zeros((n, m, n, m))
    pairwise_index = 0
    for i, j in combinations(range(n), 2):
        for k in range(m):
            for l in range(m):
                pairwise_energy_matrix[i, k, j, l] = pairwise_energy_values[pairwise_index]
                pairwise_index += 1
    for i, j in combinations(range(n), 2):
        total_energy += pairwise_energy_matrix[i, solution[i], j, solution[j]]
    return total_energy

# Peptide environment class
class PeptideEnv:
    def __init__(self, single_energy_data, pairwise_energy_data, reward_scale=1):
        self.single_energy_data = single_energy_data
        self.pairwise_energy_data = pairwise_energy_data
        self.reward_scale = reward_scale
        self.state = self.reset()

    def reset(self, starting_point=None):
        if starting_point is None:
            self.state = np.random.choice(NUM_AMINO_ACIDS, SEQUENCE_LENGTH)
        else:
            self.state = starting_point
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
        return calculate_total_energy(solution, self.single_energy_data, self.pairwise_energy_data)

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
    
    def get_logprobs(self, states, actions):
        states = torch.tensor(states, dtype=torch.int64).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        log_probs = self.forward(states)
        return log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze()

    def get_entropy(self, states):
        states = torch.tensor(states, dtype=torch.int64).to(device)
        log_probs = self.forward(states)
        return -(log_probs.exp() * log_probs).sum(dim=-1)
    
    def evaluate(self, states, actions):
        states = torch.tensor(states, dtype=torch.int64).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        log_probs = self.forward(states)
        action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze()
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return action_log_probs, entropy

# Value network class
class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, embedding_size):
        super(ValueNetwork, self).__init__()
        self.embedding = nn.Embedding(num_inputs, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.fc(x)
        return x
    
    def get_value(self, state):
        state = torch.tensor(state, dtype=torch.int64).unsqueeze(0).to(device)
        return self.forward(state).item()

# Memory class for PPO
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

# PPO algorithm class
class PPO:
    def __init__(self, policy_network, value_network, lr, betas, gamma, k_epochs, eps_clip):
        self.policy = policy_network
        self.value = value_network
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr, betas=betas)
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
            advantages = rewards - self.value(old_states).detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            entropy_weight = 0.001
            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(self.value(old_states), rewards) - entropy_weight * entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

# Main training loop
def train_network(data_folder, starting_points_file, lr=0.0005, betas=(0.9, 0.999), gamma=0.995, k_epochs=4, eps_clip=0.2, max_episodes=5000, max_timesteps=300, update_timestep=2000, log_interval=10, saved_interval=500):
    # Ensure lr is a float
    lr = float(lr)
    
    single_energy_data, pairwise_energy_data = fetch_data(data_folder)
    env = PeptideEnv(single_energy_data, pairwise_energy_data)
    memory = Memory()
    ppo = PPO(PolicyNetwork(NUM_AMINO_ACIDS, NUM_AMINO_ACIDS*SEQUENCE_LENGTH, 256, 128), 
              ValueNetwork(NUM_AMINO_ACIDS, 256, 128), lr, betas, gamma, k_epochs, eps_clip)
    ppo.policy = ppo.policy.to(device)
    ppo.value = ppo.value.to(device)

    state_energy_data = []
    time_step = 0
    for episode in range(max_episodes):
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            action, log_prob = ppo.policy.get_action(state)
            new_state, reward, done = env.step(action)
            energy = env.calculate_energy(new_state)
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            state = new_state
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
        if episode % log_interval == 0:
            print(f'Episode {episode}\tLast reward: {reward}\tAverage reward: {np.mean(memory.rewards[-log_interval:])}')
            print(f'State: {new_state}\tEnergy: {energy}')
        if episode != 0 and episode % saved_interval == 0:
            torch.save(ppo.policy.state_dict(), f'./PPO_Policy.pth')
        state_energy_data.append({'episode': episode, 'state': new_state, 'energy': energy})
    df = pd.DataFrame(state_energy_data)
    df.to_csv('work_random_PE.csv', index=False)

if __name__ == "__main__":
    data_folder = '/Users/jeetdhoriyani/Library/CloudStorage/Box-Box/peptide_workspace/QA-PBP-RL-Nexus/sample_data'
    starting_points_file = ''
    train_network(data_folder, starting_points_file)


