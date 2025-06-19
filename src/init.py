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

def rodarAlgoritmoGenetico():
    populacao = inicializarPopulacao(TAMANHO_POPULACAO, TAMANHO_CROMOSSOMO)

    melhorIndividuo = None
    melhorAvaliacao = -float('inf')
    ultimaGeracaoAprimorada = 0

    print(f"Número de bits por variável: {BITS_POR_VARIAVEL}")
    print(f"Tamanho do cromossomo: {TAMANHO_CROMOSSOMO}\n")

    for geracao in range(NUMERO_GERACOES):
        # b) Avaliar popução
        avaliacoes = avaliarPopulacao(populacao)

        melhorIndividuoAtual, melhorAvaliacaoAtual = max(avaliacoes, key=lambda item: item[1])

        if melhorAvaliacaoAtual > melhorAvaliacao:
            melhorAvaliacao = melhorAvaliacaoAtual
            melhorIndividuo = melhorIndividuoAtual
             
            ultimaGeracaoAprimorada = 0
        else:
            ultimaGeracaoAprimorada += 1

        # c) Selecionar pais
        pais = selecionarPais(avaliacoes, TAMANHO_POPULACAO)

        novaPopulacao = []
         
        # d) Aplicar crossover e mutações
        for i in range(0, len(pais), 2):
            if i + 1 < len(pais):
                pai1 = pais[i]
                pai2 = pais[i+1]
                filho1, filho2 = crossover(pai1, pai2)

                filho1 = mutar(filho1, TAXA_MUTACAO)
                filho2 = mutar(filho2, TAXA_MUTACAO)

                novaPopulacao.extend([filho1, filho2])
            else:
                novaPopulacao.append(mutar(pais[i], TAXA_MUTACAO))

        novaPopulacao = novaPopulacao[:TAMANHO_POPULACAO]

        # e) Apagar população anterior
        # f) Avaliar novos indivíduos
        populacao = novaPopulacao

        if (geracao + 1) % 50 == 0 or geracao == NUMERO_GERACOES - 1:
            realX = binarioParaReal(melhorIndividuo[:TAMANHO_CROMOSSOMO//2], INTERVALO_INICIO, INTERVAL_FINAL, BITS_POR_VARIAVEL)
            realY = binarioParaReal(melhorIndividuo[TAMANHO_CROMOSSOMO//2:], INTERVALO_INICIO, INTERVAL_FINAL, BITS_POR_VARIAVEL)

            print(f"Geração {geracao + 1}:")
            print(f"  Melhor pontuação: {melhorAvaliacao:.6f}")
            print(f"  Correspondente (x, y): ({realX:.4f}, {realY:.4f})\n")

        # g) Se o tempo acabou ou o melhor Indivíduo satisfaz os requerimentos e desempenho, retorne-o, caso contrário, volte para o passo c).
        if ultimaGeracaoAprimorada >= NUMERO_REPETICOES:
            print(f"Algoritmo parou devido à estagnação. Não houve melhorias há {NUMERO_REPETICOES} gerações.")
            break

    finalX = binarioParaReal(melhorIndividuo[:TAMANHO_CROMOSSOMO//2], INTERVALO_INICIO, INTERVAL_FINAL, BITS_POR_VARIAVEL)
    finalY = binarioParaReal(melhorIndividuo[TAMANHO_CROMOSSOMO//2:], INTERVALO_INICIO, INTERVAL_FINAL, BITS_POR_VARIAVEL)

    print("\n--- Resultados ---")
    print("Valores máximos encontrados:")
    print(f"  x = {finalX:.4f}")
    print(f"  y = {finalY:.4f}")
    print(f"  f(x,y) = {melhorAvaliacao:.6f}")
    print(f"  Cromossomo: {melhorIndividuo}")


if __name__ == "__main__":
    rodarAlgoritmoGenetico()
