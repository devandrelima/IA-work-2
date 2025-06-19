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
POPULATION_SIZE = 100
MUTATION_RATE = 0.05
NUM_GENERATIONS = 500

# Calculate the number of bits needed for each variable
# Formula for precision: (sup - inf) / (2^k - 1)
# We need to find k such that:

# (INTERVAL_END - INTERVAL_START) / (2^k - 1) <= PRECISION
# (2^k - 1) >= (INTERVAL_END - INTERVAL_START) / PRECISION
# 2^k >= (INTERVAL_END - INTERVAL_START) / PRECISION + 1
# k >= log2((INTERVAL_END - INTERVAL_START) / PRECISION + 1)

range_val = INTERVAL_END - INTERVAL_START
min_bits = math.log2(range_val / PRECISION + 1)
BITS_PER_VARIABLE = math.ceil(min_bits)
CHROMOSOME_LENGTH = BITS_PER_VARIABLE * 2  # Two variables (x and y)

# Helper function to convert binary to real value


def bin_to_real(binary_str, interval_start, interval_end, bits):
    r_i = int(binary_str, 2)
    real_val = interval_start + ((interval_end - interval_start) / (2**bits - 1)) * r_i
    return real_val

# Helper function to evaluate an individual (chromosome)


def evaluate_individual(chromosome):
    mid_point = CHROMOSOME_LENGTH // 2
    x_binary = chromosome[:mid_point]
    y_binary = chromosome[mid_point:]

    x = bin_to_real(x_binary, INTERVAL_START, INTERVAL_END, BITS_PER_VARIABLE)
    y = bin_to_real(y_binary, INTERVAL_START, INTERVAL_END, BITS_PER_VARIABLE)

    return f(x, y)

# 1. Initialize the population


def generate_chromosome(length):
    result = ''

    for _ in range(length):
        result += random.choice(['0', '1'])

    return result


def initialize_population(size, chromosome_length):
    population = []

    for _ in range(size):
        population.append(generate_chromosome(chromosome_length))

    return population

# 2. Evaluate each individual in the population


def calculate_fitness(population):
    fitness_scores = []

    for individual in population:
        fitness = evaluate_individual(individual)
        fitness_scores.append((individual, fitness))

    return fitness_scores

# 3. Select parents for generating new individuals (Roulette Wheel Selection)


def select_parents(fitness_scores, num_parents):
    total_fitness = sum(fitness for _, fitness in fitness_scores)

    if total_fitness == 0:
        return random.sample([ind for ind, _ in fitness_scores], num_parents)

    cumulative_wheel_pieces = []
    current_cumulative_value = 0

    for individual, fitness in fitness_scores:
        piece_size = (fitness / total_fitness) * 360
        current_cumulative_value += piece_size
        cumulative_wheel_pieces.append((individual, current_cumulative_value))

    parents = []
    for _ in range(num_parents):
        r = random.uniform(0, 360)

        for individual, cumulative_upper_bound in cumulative_wheel_pieces:
            if r <= cumulative_upper_bound:
                parents.append(individual)
                break

    return parents

# 4. Apply crossover and mutation to generate new individuals


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(chromosome, mutation_rate):
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
