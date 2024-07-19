import os
import pandas as pd
import numpy as np
from itertools import combinations
import dimod



def fetch_data(data_directory, pairwise_factor=1.0):
    """
    Load energy values from Excel files, apply transformation and scaling.

    Args:
        data_directory (str): Base directory containing energy files.
        pairwise_factor (float): Scaling factor for pairwise energies.

    Returns:
        tuple: Tuple containing arrays of single and pairwise energy values.
    """
    single_energy_file = os.path.join(data_directory, "SingleEnergy_sample.xlsx")
    pairwise_energy_file = os.path.join(data_directory, "PairwiseEnergy_sample.xlsx")

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
    print("QUBO problem generated.")
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
    sampler = dimod.RandomSampler()
    sampleset = sampler.sample(bqm)
    sample = sampleset.first.sample
    return sample

def decode_solutions(sample, num_positions=12, num_amino_acids=19):
    """
    Decode samples into human-readable solutions.

    Args:
        sample (list): Sample solution from the QUBO solver.
        num_positions (int): Number of positions (default: 12).
        num_amino_acids (int): Number of amino acids (default: 19).

    Returns:
        list: Decoded solution.
    """
    solution = np.zeros(num_positions)
    for i in range(num_positions):
        for j in range(num_amino_acids):
            qubo_index = i * num_amino_acids + j
            if sample[qubo_index] == 1:
                solution[i] = j
    return solution

def main():
    """
    Main function to process data, solve QUBO problems, and save results.
    """
    data_directory = "demo/sample_data"
    pairwise_factor = 1.0
    penalty = 10.0

    single_energy_values, pairwise_energy_values = fetch_data(data_directory, pairwise_factor)
    qubo_problem = generate_qubo(single_energy_values, pairwise_energy_values, penalty)

    df_solutions = pd.DataFrame(columns=['Solution'])
    sample = solve_qubo_with_hybrid(qubo_problem)
    decoded_solution = decode_solutions(sample)
    df_solutions = pd.concat([df_solutions, pd.DataFrame({'Solution': [decoded_solution]})], ignore_index=True)

    file_name = 'sample_peptide_10.csv'
    df_solutions.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")

if __name__ == "__main__":
    main()
