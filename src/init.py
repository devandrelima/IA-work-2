import random
import math

def funcaoMaximizacao(x, y):
    return abs(math.exp(-x) - (y**2) + 1) + (10**-4)

INTERVALO_INICIO = -10
INTERVAL_FINAL = 10
PRECISAO = 0.005
TAMANHO_POPULACAO = 100
TAXA_MUTACAO = 0.05
NUMERO_GERACOES = 500
NUMERO_REPETICOES = 150

MINIMO_BITS = math.log2(INTERVAL_FINAL - INTERVALO_INICIO / PRECISAO + 1)
BITS_POR_VARIAVEL = math.ceil(MINIMO_BITS)
TAMANHO_CROMOSSOMO = BITS_POR_VARIAVEL * 2

def binarioParaReal(stringBinario, inicioDeIntervalo, finalDeIntervalo, bits):
    return inicioDeIntervalo + ((finalDeIntervalo - inicioDeIntervalo) / (2**bits - 1)) * int(stringBinario, 2)

def avalidarIndividuo(cromossomo):
    meio = TAMANHO_CROMOSSOMO // 2
    binarioX = cromossomo[:meio]
    binarioY = cromossomo[meio:]

    x = binarioParaReal(binarioX, INTERVALO_INICIO, INTERVAL_FINAL, BITS_POR_VARIAVEL)
    y = binarioParaReal(binarioY, INTERVALO_INICIO, INTERVAL_FINAL, BITS_POR_VARIAVEL)

    return funcaoMaximizacao(x, y)

def gerarCromossomo(tamanho):
    resultado = ''

    for _ in range(tamanho):
        resultado += random.choice(['0', '1'])

    return resultado


def inicializarPopulacao(tamanhoDaPopulacao, tamanhoDoCromossomo):
    populacao = []

    for _ in range(tamanhoDaPopulacao):
        populacao.append(gerarCromossomo(tamanhoDoCromossomo))

    return populacao

def avaliarPopulacao(populacao):
    pontuacoes = []

    for individuo in populacao:
        avaliacao = avalidarIndividuo(individuo)
        pontuacoes.append((individuo, avaliacao))

    return pontuacoes

def selecionarPais(pontuacoes, tamanhoDaPopulacao):
    somatorioDeAvaliacoes = sum(avaliacao for _, avaliacao in pontuacoes)

    if somatorioDeAvaliacoes == 0:
        return random.sample([individuo for individuo, _ in pontuacoes], tamanhoDaPopulacao)

    porcoesDaRoleta = []
    ultimaPorcaoSelecionada = 0

    for individuo, avaliacao in pontuacoes:
        tamanhoDaPorcao = (avaliacao / somatorioDeAvaliacoes) * 360
        ultimaPorcaoSelecionada += tamanhoDaPorcao
        porcoesDaRoleta.append((individuo, ultimaPorcaoSelecionada))

    pais = []
    for _ in range(tamanhoDaPopulacao):
        porcaoAleatoria = random.uniform(0, 360)

        for individuo, topoDaPorcaoSelecionada in porcoesDaRoleta:
            if porcaoAleatoria <= topoDaPorcaoSelecionada:
                pais.append(individuo)
                break

    return pais

def crossover(pai1, pai2):
    meio = random.randint(1, len(pai1) - 1)
     
    filho1 = pai1[:meio] + pai2[meio:]
    filho2 = pai2[:meio] + pai1[meio:]
     
    return filho1, filho2


def mutar(cromossomo, taxaDeMutacao):
    cromossomoMutado = list(cromossomo)

    for i in range(len(cromossomoMutado)):
        if random.uniform(0, 1) < taxaDeMutacao:
            cromossomoMutado[i] = '1' if cromossomoMutado[i] == '0' else '0'

    return "".join(cromossomoMutado)

# Main Genetic Algorithm loop
def run_genetic_algorithm():
    population = inicializarPopulacao(TAMANHO_POPULACAO, TAMANHO_CROMOSSOMO)

    best_overall_individual = None
    best_overall_fitness = -float('inf')
    generations_since_last_improvement = 0

    print(f"Number of bits per variable: {BITS_POR_VARIAVEL}")
    print(f"Chromosome length: {TAMANHO_CROMOSSOMO}\n")

    for generation in range(NUMERO_GERACOES):
        # b) Evaluate each individual in the population.
        fitness_scores = avaliarPopulacao(population)

        # Find the best individual in the current generation
        current_best_individual, current_best_fitness = max(fitness_scores, key=lambda item: item[1])

        # Update overall best
        if current_best_fitness > best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_individual = current_best_individual
            generations_since_last_improvement = 0
        else:
            generations_since_last_improvement += 1

        # c) Select parents for generating new individuals.
        parents = selecionarPais(fitness_scores, TAMANHO_POPULACAO) # Select enough parents to create a new population

        new_population = []
        # d) Apply the operators of recombination (crossover) and mutation to generate new individuals. 
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i+1]
                child1, child2 = crossover(parent1, parent2)

                child1 = mutar(child1, TAXA_MUTACAO)
                child2 = mutar(child2, TAXA_MUTACAO)

                new_population.extend([child1, child2])
            else:
                new_population.append(mutar(parents[i], TAXA_MUTACAO))

        new_population = new_population[:TAMANHO_POPULACAO]

        # e) Erase old members of the population.
        # f) Evaluate all new individuals and insert them into the population.
        population = new_population

        if (generation + 1) % 50 == 0 or generation == NUMERO_GERACOES - 1:
            x_val = binarioParaReal(best_overall_individual[:TAMANHO_CROMOSSOMO//2], INTERVALO_INICIO, INTERVAL_FINAL, BITS_POR_VARIAVEL)
            y_val = binarioParaReal(best_overall_individual[TAMANHO_CROMOSSOMO//2:], INTERVALO_INICIO, INTERVAL_FINAL, BITS_POR_VARIAVEL)

            print(f"Generation {generation + 1}:")
            print(f"  Best fitness so far: {best_overall_fitness:.6f}")
            print(f"  Corresponding (x, y): ({x_val:.4f}, {y_val:.4f})\n")

        if generations_since_last_improvement >= NUMERO_REPETICOES:
            print(f"Algorithm stopped due to stagnation. No significant improvement for {NUMERO_REPETICOES} generations.")
            break

    # g) If time is over or the best individual satisfies performance requirements, return it. 
    final_x = binarioParaReal(best_overall_individual[:TAMANHO_CROMOSSOMO//2], INTERVALO_INICIO, INTERVAL_FINAL, BITS_POR_VARIAVEL)
    final_y = binarioParaReal(best_overall_individual[TAMANHO_CROMOSSOMO//2:], INTERVALO_INICIO, INTERVAL_FINAL, BITS_POR_VARIAVEL)

    print("\n--- Genetic Algorithm Results ---")
    print("Global Maximum found:")
    print(f"  x = {final_x:.4f}")
    print(f"  y = {final_y:.4f}")
    print(f"  f(x,y) = {best_overall_fitness:.6f}")
    print(f"  Chromosome: {best_overall_individual}")


if __name__ == "__main__":
    run_genetic_algorithm()
