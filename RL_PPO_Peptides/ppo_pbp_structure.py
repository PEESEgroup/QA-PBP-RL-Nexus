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

def fetch_data(data_folder: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Fetch single and pairwise energy values from the specified folder.

    Args:
        data_folder (str): The folder containing the energy data files.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of single and pairwise energy values.
    """
    backbone_folder = os.path.join(data_folder)
    single_energy_file = os.path.join(backbone_folder, "SingleEnergy_sample.xlsx")
    pairwise_energy_file = os.path.join(backbone_folder, "PairwiseEnergy_sample.xlsx")
    
    single_energy_values = pd.read_excel(single_energy_file)["Energy"].values
    pairwise_energy_values = pd.read_excel(pairwise_energy_file)["Energy"].values
    
    return single_energy_values, pairwise_energy_values

def calculate_total_energy(solution: list[int], single_energy_values: np.ndarray, pairwise_energy_values: np.ndarray) -> float:
    """
    Calculate the total energy of a given solution based on single and pairwise energy values.

    Args:
        solution (list[int]): The solution sequence represented as a list of amino acid indices.
        single_energy_values (np.ndarray): Array of single energy values.
        pairwise_energy_values (np.ndarray): Array of pairwise energy values.

    Returns:
        float: The total energy of the solution.
    """
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

class PeptideEnv:
    def __init__(self, single_energy_data: np.ndarray, pairwise_energy_data: np.ndarray, reward_scale: float = 1):
        """
        Initialize the Peptide environment with single and pairwise energy data.

        Args:
            single_energy_data (np.ndarray): Array of single energy values.
            pairwise_energy_data (np.ndarray): Array of pairwise energy values.
            reward_scale (float): Scaling factor for rewards.
        """
        self.single_energy_data = single_energy_data
        self.pairwise_energy_data = pairwise_energy_data
        self.reward_scale = reward_scale
        self.state = self.reset()

    def reset(self, starting_point: list[int] = None) -> np.ndarray:
        """
        Reset the environment to a starting point or a random state.

        Args:
            starting_point (list[int], optional): The starting state. Defaults to None.

        Returns:
            np.ndarray: The initial state.
        """
        if starting_point is None:
            self.state = np.random.choice(NUM_AMINO_ACIDS, SEQUENCE_LENGTH)
        else:
            self.state = starting_point
        return self.state

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Take an action in the environment, return the new state, reward, and done status.

        Args:
            action (int): The action to take.

        Returns:
            tuple[np.ndarray, float, bool]: The new state, the reward, and the done status.
        """
        position = action // NUM_AMINO_ACIDS
        amino_acid = action % NUM_AMINO_ACIDS
        old_energy = self.calculate_energy(self.state)
        self.state[position] = amino_acid
        new_energy = self.calculate_energy(self.state)
        reward = (old_energy - new_energy) * self.reward_scale
        return self.state, reward, False

    def calculate_energy(self, solution: list[int]) -> float:
        """
        Calculate the total energy of the current solution.

        Args:
            solution (list[int]): The solution sequence.

        Returns:
            float: The total energy of the solution.
        """
        return calculate_total_energy(solution, self.single_energy_data, self.pairwise_energy_data)

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, hidden_size: int, embedding_size: int):
        """
        Initialize the policy network with input/output sizes, hidden layer size, and embedding size.

        Args:
            num_inputs (int): Number of input features.
            num_outputs (int): Number of output features.
            hidden_size (int): Size of the hidden layer.
            embedding_size (int): Size of the embedding layer.
        """
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(num_inputs, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with log probabilities.
        """
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return F.log_softmax(x, dim=-1)
    
    def get_action(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        """
        Get an action based on the current state.

        Args:
            state (np.ndarray): The current state.

        Returns:
            tuple[int, torch.Tensor]: The chosen action and its log probability.
        """
        state = torch.tensor(state, dtype=torch.int64).unsqueeze(0).to(device)
        log_probs = self.forward(state)
        action_probs = F.softmax(log_probs, dim=-1).detach()
        action = torch.multinomial(action_probs, num_samples=1)
        return action.item(), log_probs[0, action.item()]
    
    def get_logprobs(self, states: list[np.ndarray], actions: list[int]) -> torch.Tensor:
        """
        Get the log probabilities of the given states and actions.

        Args:
            states (list[np.ndarray]): List of states.
            actions (list[int]): List of actions.

        Returns:
            torch.Tensor: Log probabilities of the actions.
        """
        states = torch.tensor(states, dtype=torch.int64).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        log_probs = self.forward(states)
        return log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze()

    def get_entropy(self, states: list[np.ndarray]) -> torch.Tensor:
        """
        Calculate the entropy of the given states.

        Args:
            states (list[np.ndarray]): List of states.

        Returns:
            torch.Tensor: Entropy of the states.
        """
        states = torch.tensor(states, dtype=torch.int64).to(device)
        log_probs = self.forward(states)
        return -(log_probs.exp() * log_probs).sum(dim=-1)
    
    def evaluate(self, states: list[np.ndarray], actions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the given states and actions, return log probabilities and entropy.

        Args:
            states (list[np.ndarray]): List of states.
            actions (list[int]): List of actions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Log probabilities of the actions and entropy of the states.
        """
        states = torch.tensor(states, dtype=torch.int64).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        log_probs = self.forward(states)
        action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze()
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return action_log_probs, entropy

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs: int, hidden_size: int, embedding_size: int):
        """
        Initialize the value network with input size, hidden layer size, and embedding size.

        Args:
            num_inputs (int): Number of input features.
            hidden_size (int): Size of the hidden layer.
            embedding_size (int): Size of the embedding layer.
        """
        super(ValueNetwork, self).__init__()
        self.embedding = nn.Embedding(num_inputs, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with state value.
        """
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.fc(x)
        return x
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get the value of the current state.

        Args:
            state (np.ndarray): The current state.

        Returns:
            float: The value of the state.
        """
        state = torch.tensor(state, dtype=torch.int64).unsqueeze(0).to(device)
        return self.forward(state).item()

class Memory:
    def __init__(self):
        """
        Initialize memory for storing states, actions, log probabilities, rewards, and terminal status.
        """
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self) -> None:
        """
        Clear the stored memory.
        """
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

class PPO:
    def __init__(self, policy_network: PolicyNetwork, value_network: ValueNetwork, lr: float, betas: tuple, gamma: float, k_epochs: int, eps_clip: float):
        """
        Initialize the PPO algorithm with policy/value networks, learning rate, betas, gamma, epochs, and epsilon clip.

        Args:
            policy_network (PolicyNetwork): The policy network.
            value_network (ValueNetwork): The value network.
            lr (float): Learning rate.
            betas (tuple): Betas for Adam optimizer.
            gamma (float): Discount factor.
            k_epochs (int): Number of epochs for updating.
            eps_clip (float): Epsilon clip for PPO.
        """
        self.policy = policy_network
        self.value = value_network
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr, betas=betas)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
    
    def update(self, memory: Memory) -> None:
        """
        Update the policy and value networks using the memory.

        Args:
            memory (Memory): The memory object containing experiences.
        """
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
def train_network(data_folder: str, lr: float = 0.0005, betas: tuple = (0.9, 0.999), gamma: float = 0.995, k_epochs: int = 4, eps_clip: float = 0.2, max_episodes: int = 5000, max_timesteps: int = 300, update_timestep: int = 2000, log_interval: int = 10, saved_interval: int = 500) -> None:
    """
    Train the PPO network using the provided data and hyperparameters.

    Args:
        data_folder (str): The folder containing the energy data files.
        lr (float, optional): Learning rate. Defaults to 0.0005.
        betas (tuple, optional): Betas for Adam optimizer. Defaults to (0.9, 0.999).
        gamma (float, optional): Discount factor. Defaults to 0.995.
        k_epochs (int, optional): Number of epochs for updating. Defaults to 4.
        eps_clip (float, optional): Epsilon clip for PPO. Defaults to 0.2.
        max_episodes (int, optional): Maximum number of training episodes. Defaults to 5000.
        max_timesteps (int, optional): Maximum number of timesteps per episode. Defaults to 300.
        update_timestep (int, optional): Number of timesteps before updating the policy. Defaults to 2000.
        log_interval (int, optional): Interval for logging progress. Defaults to 10.
        saved_interval (int, optional): Interval for saving the model. Defaults to 500.
    """
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
