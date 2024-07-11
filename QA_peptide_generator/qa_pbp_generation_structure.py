import os
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
import dimod
from hybrid.reference.kerberos import KerberosSampler
import argparse
import sys

def fetch_data(backbone_index, pairwise_factor, data_directory):
    """
    Load energy values from Excel files, apply transformation and scaling.

    Args:
        backbone_index (int): Index of the backbone to process.
        pairwise_factor (float): Scaling factor for pairwise energies.
        data_directory (str): Base directory containing energy files.

    Returns:
        tuple: Tuple containing arrays of single and pairwise energy values.
    """
    single_energy_file = os.path.join(data_directory, f"bb{backbone_index}", "Single_energy_files.xlsx")
    pairwise_energy_file = os.path.join(data_directory, f"bb{backbone_index}", "PairwiseEnergy_energy_files.xlsx")

    single_energy_values = pd.read_excel(single_energy_file)["Energy"].values
    pairwise_energy_values = pd.read_excel(pairwise_energy_file)["Energy"].values * pairwise_factor


    return single_energy_values, pairwise_energy_values

def generate_qubo(single_energy, pairwise_energy, penalty, num_positions=12, num_amino_acids=19):
    """
    Generate a QUBO matrix for the given energy profiles and penalty.

    Args:
        single_energy (np.array): Array of single point energies.
        pairwise_energy (np.array): Array of pairwise energies.
        penalty (float): Penalty value to enforce constraints.
        num_positions (int): Number of positions (default: 12).
        num_amino_acids (int): Number of amino acids (default: 19).

    Returns:
        dict: Dictionary representing the QUBO problem.
    """
    qubo_size = num_positions * num_amino_acids
    qubo_matrix = np.zeros((qubo_size, qubo_size))

    # Fill diagonal with single point energies
    for i in range(num_positions):
        for j in range(num_amino_acids):
            qubo_index_i = i * num_amino_acids + j
            qubo_matrix[qubo_index_i, qubo_index_i] = single_energy[qubo_index_i]

    # Fill off-diagonal with pairwise energies
    pairwise_index = 0
    for i, j in combinations(range(num_positions), 2):
        for k in range(num_amino_acids):
            for l in range(num_amino_acids):
                index_ik = i * num_amino_acids + k
                index_jl = j * num_amino_acids + l
                qubo_matrix[index_ik, index_jl] = pairwise_energy[pairwise_index]
                qubo_matrix[index_jl, index_ik] = pairwise_energy[pairwise_index]
                pairwise_index += 1

    # Apply penalties to enforce constraints
    for i in range(num_positions):
        for j in range(num_amino_acids):
            for k in range(j + 1, num_amino_acids):
                qubo_index_j = i * num_amino_acids + j
                qubo_index_k = i * num_amino_acids + k
                qubo_matrix[qubo_index_j, qubo_index_k] += penalty

    # Convert to dictionary form
    qubo_problem = {(i, j): qubo_matrix[i, j] for i in range(qubo_size) for j in range(qubo_size) if qubo_matrix[i, j] != 0}
    return qubo_problem

def solve_qubo_with_hybrid(qubo_problem):
    """
    Solve the QUBO problem using a hybrid quantum-classical approach.

    Args:
        qubo_problem (dict): QUBO problem dictionary.

    Returns:
        dict: Optimal sample from the solution set.
    """
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_problem)
    sampler = KerberosSampler()
    sampleset = sampler.sample(bqm)
    sample = sampleset.first.sample
    return sample

def decode_solutions(samples, num_positions=12, num_amino_acids=19):
    """
    Decode samples into human-readable solutions.

    Args:
        samples (list): List of sample solutions from the QUBO solver.
        num_positions (int): Number of positions (default: 12).
        num_amino_acids (int): Number of amino acids (default: 19).

    Returns:
        list: List of decoded solutions.
    """
    solutions = []
    for sample in samples:
        solution = np.zeros(num_positions)
        for i in range(num_positions):
            for j in range(num_amino_acids):
                qubo_index = i * num_amino_acids + j
                if sample[qubo_index] == 1:
                    solution[i] = j
        solutions.append(solution)
    return solutions

def main(args):
    """
    Main function to process data, solve QUBO problems, and save results.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    data_directory = args.data_directory
    backbones = os.listdir(data_directory)
    backbone_indices = [int(bb.split('bb')[-1]) for bb in backbones if 'bb' in bb]
    backbone_indices.sort()

    df_solutions = pd.DataFrame(columns=['Backbone', 'Solution'])
    checkpoint_filename = f'checkpoint_file{args.pairwise_factor}_Pen{args.penalty}.pkl'

    # Check for existing data to resume
    if os.path.exists(checkpoint_filename) and args.start_from_backbone == 0:
        df_solutions = pd.read_pickle(checkpoint_filename)
        last_processed_backbone = df_solutions['Backbone'].iloc[-1]
        args.start_from_backbone = backbone_indices.index(last_processed_backbone) + 1

    for i in tqdm(backbone_indices[args.start_from_backbone:], desc="Processing Backbones"):
        single_energy_values_i, pairwise_energy_values_i = fetch_data(i, args.pairwise_factor, data_directory)
        qubo_problem_i = generate_qubo(single_energy_values_i, pairwise_energy_values_i, args.penalty)

        for repeat in range(args.num_repeats):
            sample = solve_qubo_with_hybrid(qubo_problem_i)
            d1 = decode_solutions([sample])
            df_solutions = pd.concat([df_solutions, pd.DataFrame({'Backbone': [i], 'Solution': [d1]})], ignore_index=True)

        df_solutions.to_pickle(checkpoint_filename)

    file_name = f'aPS_final_solvent_PE_Pen{args.penalty}_Pairwise.csv'
    df_solutions.to_csv(file_name, index=False)
