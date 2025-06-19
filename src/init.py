import random
import math

# Function to maximize


def f(x, y):
    """
    The function to maximize: f(x,y) = |e^(-x) - y^2 + 1| + 10^-4
    """
    return abs(math.exp(-x) - (y**2) + 1) + (10**-4)

# Genetic Algorithm Parameters


INTERVAL_START = -10
INTERVAL_END = 10
PRECISION = 0.005
POPULATION_SIZE = 100  # Number of individuals in the population
MUTATION_RATE = 0.05   # 5% mutation rate as an example
NUM_GENERATIONS = 500  # Number of generations to run the algorithm

# Calculate the number of bits needed for each variable
# Formula for precision: (sup - inf) / (2^k - 1)
# We need to find k such that (INTERVAL_END - INTERVAL_START) / (2^k - 1) <= PRECISION
# (2^k - 1) >= (INTERVAL_END - INTERVAL_START) / PRECISION
# 2^k >= (INTERVAL_END - INTERVAL_START) / PRECISION + 1
# k >= log2((INTERVAL_END - INTERVAL_START) / PRECISION + 1)
range_val = INTERVAL_END - INTERVAL_START
min_bits = math.log2(range_val / PRECISION + 1)
BITS_PER_VARIABLE = math.ceil(min_bits)
CHROMOSOME_LENGTH = BITS_PER_VARIABLE * 2  # Two variables (x and y)

# Helper function to convert binary to real value


def bin_to_real(binary_str, interval_start, interval_end, bits):
    """
    Converts a binary string to a real number within a given interval.
    """
     
    r_i = int(binary_str, 2)
    real_val = interval_start + ((interval_end - interval_start) / (2**bits - 1)) * r_i
    return real_val

# Helper function to evaluate an individual (chromosome)


def evaluate_individual(chromosome):
    """
    Evaluates the fitness of an individual (chromosome) by decoding its genes
    into x and y values and then calculating f(x,y).
    Since the genetic algorithm is for maximization, a higher f(x,y) value
    means better fitness.
    """

    mid_point = CHROMOSOME_LENGTH // 2
    x_binary = chromosome[:mid_point]
    y_binary = chromosome[mid_point:]

    x = bin_to_real(x_binary, INTERVAL_START, INTERVAL_END, BITS_PER_VARIABLE)
    y = bin_to_real(y_binary, INTERVAL_START, INTERVAL_END, BITS_PER_VARIABLE)

    # We want to maximize f(x,y). If f(x,y) can return 0 or negative,
    # it's usually good practice to adjust the fitness function for roulette wheel
    # selection, e.g., fitness = 1 + f(x,y) or a similar scaling.
    # However, given f(x,y) = |...| + 10^-4, the minimum value is 10^-4, which is positive.
    # So, f(x,y) itself can serve as the fitness function directly for maximization.
    return f(x, y)

# 1. Initialize the population


def initialize_population(size, chromosome_length):
    """
    Initializes a population of individuals with random binary chromosomes.
    """
    population = []
    for _ in range(size):
        chromosome = ''.join(random.choice('01') for _ in range(chromosome_length))
        population.append(chromosome)
    return population

# 2. Evaluate each individual in the population


def calculate_fitness(population):
    """
    Calculates the fitness for each individual in the population.

    Returns a list of (individual, fitness) tuples.
    """
    fitness_scores = []
    for individual in population:
        fitness = evaluate_individual(individual)
        fitness_scores.append((individual, fitness))
    return fitness_scores

# 3. Select parents for generating new individuals (Roulette Wheel Selection)


def select_parents(fitness_scores, num_parents):
    """
    Selects parents using the roulette wheel selection method.
    Individuals with higher fitness have a higher probability of being selected.
    """
    total_fitness = sum(fitness for _, fitness in fitness_scores)

    # Handle cases where total_fitness might be zero or very small to avoid division by zero errors
    if total_fitness == 0:
        # If all fitnesses are zero, select parents randomly
        return random.sample([ind for ind, _ in fitness_scores], num_parents)

    probabilities = [(individual, fitness / total_fitness) for individual, fitness in fitness_scores]

    parents = []
    for _ in range(num_parents):
        r = random.uniform(0, 1)
        cumulative_probability = 0
        for individual, prob in probabilities:
            cumulative_probability += prob
            if r <= cumulative_probability:
                parents.append(individual)
                break
    return parents

# 4. Apply crossover and mutation to generate new individuals


def crossover(parent1, parent2):
    """
    Performs a single-point crossover between two parents.
    """
    crossover_point = random.randint(1, len(parent1) - 1)  # A position between two genes 
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(chromosome, mutation_rate):
    """
    Applies mutation to a chromosome by flipping bits with a given probability.
    """
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.uniform(0, 1) < mutation_rate:
            mutated_chromosome[i] = '1' if mutated_chromosome[i] == '0' else '0'
    return "".join(mutated_chromosome)

# Main Genetic Algorithm loop


def run_genetic_algorithm():
    population = initialize_population(POPULATION_SIZE, CHROMOSOME_LENGTH)

    best_overall_individual = None
    best_overall_fitness = -float('inf')

    print(f"Number of bits per variable: {BITS_PER_VARIABLE}")
    print(f"Chromosome length: {CHROMOSOME_LENGTH}\n")

    for generation in range(NUM_GENERATIONS):
        # b) Evaluate each individual in the population. 
        fitness_scores = calculate_fitness(population)

        # Find the best individual in the current generation
        current_best_individual, current_best_fitness = max(fitness_scores, key=lambda item: item[1])

        # Update overall best
        if current_best_fitness > best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_individual = current_best_individual

        # c) Select parents for generating new individuals. 
        parents = select_parents(fitness_scores, POPULATION_SIZE) # Select enough parents to create a new population

        new_population = []
        # d) Apply the operators of recombination (crossover) and mutation to generate new individuals. 
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i+1]
                child1, child2 = crossover(parent1, parent2)

                child1 = mutate(child1, MUTATION_RATE)
                child2 = mutate(child2, MUTATION_RATE)

                new_population.extend([child1, child2])
            else:
                # If there's an odd number of parents, just add the last one (possibly mutated)
                new_population.append(mutate(parents[i], MUTATION_RATE))

        # Ensure the new population size matches the original population size
        # Truncate or pad if necessary (though with an even number of parents, it should be fine)
        new_population = new_population[:POPULATION_SIZE]

        # e) Erase old members of the population.
        # f) Evaluate all new individuals and insert them into the population.
        population = new_population

        if (generation + 1) % 50 == 0 or generation == NUM_GENERATIONS - 1:
            x_val = bin_to_real(best_overall_individual[:CHROMOSOME_LENGTH//2], INTERVAL_START, INTERVAL_END, BITS_PER_VARIABLE)
            y_val = bin_to_real(best_overall_individual[CHROMOSOME_LENGTH//2:], INTERVAL_START, INTERVAL_END, BITS_PER_VARIABLE)
            print(f"Generation {generation + 1}:")
            print(f"  Best fitness so far: {best_overall_fitness:.6f}")
            print(f"  Corresponding (x, y): ({x_val:.4f}, {y_val:.4f})\n")

    # g) If time is over or the best individual satisfies performance requirements, return it. 
    final_x = bin_to_real(best_overall_individual[:CHROMOSOME_LENGTH//2], INTERVAL_START, INTERVAL_END, BITS_PER_VARIABLE)
    final_y = bin_to_real(best_overall_individual[CHROMOSOME_LENGTH//2:], INTERVAL_START, INTERVAL_END, BITS_PER_VARIABLE)

    print("\n--- Genetic Algorithm Results ---")
    print(f"Global Maximum found:")
    print(f"  x = {final_x:.4f}")
    print(f"  y = {final_y:.4f}")
    print(f"  f(x,y) = {best_overall_fitness:.6f}")
    print(f"  Chromosome: {best_overall_individual}")


if __name__ == "__main__":
    run_genetic_algorithm()

