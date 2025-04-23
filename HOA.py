import numpy as np
import opfunu
from matplotlib import pyplot as plt
from matplotlib import cm
import math
import time
import pandas as pd

# Initialization function
def initialization(N, Dim, UB, LB):
    """
    Initializes the positions of particles in the search space.
    
    :param N: Number of particles in the population
    :param Dim: Number of dimensions of the problem
    :param UB: Upper bounds for each dimension
    :param LB: Lower bounds for each dimension
    :return: Initialized particle positions
    """
    B_no = len(UB)
    X = np.zeros((N, Dim))

    if B_no == 1:
        # If there is only one boundary, initialize all particles within the same range
        X = np.random.rand(N, Dim) * (UB - LB) + LB
    elif B_no > 1:
        # For each dimension, initialize the particles within the respective boundaries
        for i in range(Dim):
            Ub_i = UB[i]
            Lb_i = LB[i]
            X[:, i] = np.random.rand(N) * (Ub_i - Lb_i) + Lb_i

    return X

def HOA(N, T, LB, UB, Dim, F_obj, identifier):
    """
    HOA algorithm for optimizing a given objective function.
    
    :param N: Number of particles (population size)
    :param T: Number of iterations
    :param LB: Lower bounds for each dimension
    :param UB: Upper bounds for each dimension
    :param Dim: Number of dimensions
    :param F_obj: Objective function to be minimized
    :param identifier: Unique identifier for the algorithm (for logging or differentiation)
    :return: Best fitness value, best solution, and convergence history
    """
    Best_P = np.zeros(Dim)  # Best position (solution)
    Best_FF = float('inf')  # Best fitness value (initially set to infinity)
    
    # Initialize positions and velocities of particles
    X = initialization(N, Dim, UB, LB)
    X_new = X.copy()
    
    # Store the fitness values
    Ffun = np.zeros(N)
    Ffun_new = np.zeros(N)
    
    t = 1  # Current iteration
    conv = []  # Convergence history
    
    total_time = 0  # Initialize total time
    iteration_times = []  # Store iteration times
    GbestScores = []  # Store best fitness values
    GbestPositions = []  # Store best positions

    # Find the initial leader (best solution)
    for i in range(N):
        X_new[i, :] = X[i, :]
        Ffun_new[i] = F_obj(X_new[i, :])
        Ffun[i] = Ffun_new[i]
        if Ffun[i] < Best_FF:
            Best_FF = Ffun[i]
            Best_P = X[i, :]

    # Main optimization loop (iterates T times)
    while t < T + 1:
        start_time = time.time()  # Record the start time of the iteration
        
        for i in range(N):      
            # Randomly generate slope angle (theta)
            theta = np.random.randint(low=0, high=50, size=1)
            
            # Calculate slope (s)
            s = math.tan(theta)
            
            # Scanning factor (SF) for randomness
            SF = np.random.uniform(low=1.0, high=2.0, size=1)
            
            # Initial velocity calculation
            Vel = 6 * math.exp(-3.5 * abs(s + 0.05))
            
            # Update velocity using the position of the leader and a random component
            newVel = X_new[i, :].copy()
            newVel = Vel + np.random.randn(1, Dim) * (Best_P - SF * X_new[i, :])

            # Update position of the particle
            X_new[i, :] = X_new[i, :] + newVel

            # Boundary constraint: ensure the particle stays within bounds
            F_UB = X_new[i, :] > UB
            F_LB = X_new[i, :] < LB
            X_new[i, :] = (X_new[i, :] * ~(F_UB + F_LB)) + UB * F_UB + LB * F_LB

            # Evaluate the new fitness of the particle
            Ffun_new[i] = F_obj(X_new[i, :])
            
            # If the new fitness is better, update the particle's position
            if Ffun_new[i] < Ffun[i]:
                X[i, :] = X_new[i, :]
                Ffun[i] = Ffun_new[i]
            
            # If this particle's fitness is the best so far, update the global best
            if Ffun[i] < Best_FF:
                Best_FF = Ffun[i]
                Best_P = X[i, :]

        end_time = time.time()  # Record the end time of the iteration
        iteration_time = end_time - start_time  # Calculate the iteration time
        total_time += iteration_time  # Add the current iteration time to the total time
        average_time = total_time / t  # Calculate the average iteration time

        # Store the iteration time, best fitness value, and best position
        iteration_times.append(iteration_time)
        GbestScores.append(Best_FF)
        GbestPositions.append(Best_P.tolist())

        # Print the progress
        if t % 1 == 0:
            print(f'At iteration {t}, the best solution fitness is {Best_FF:.8f}')
        conv.append(Best_FF)
        
        # Print detailed iteration time statistics
        print(f"Iteration {t} time: {iteration_time:.4f} seconds")
        print(f"Average iteration time: {average_time:.4f} seconds")

        t += 1

    return Best_FF, Best_P, conv  # Return the best fitness value, best solution, and convergence history
