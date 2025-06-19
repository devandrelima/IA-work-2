# Trabalho de Implementação de MLP e Backpropagation

## Pergunta Respondida com Sucesso

Este documento detalha o trabalho de implementação de um algoritmo de Backpropagation para uma Rede Neural Perceptron de Múltiplas Camadas (MLP), com o objetivo de resolver um problema de classificação de números de 0 a 7.

Vamos detalhar o que você precisa saber, estudar e fazer, passo a passo, seguido por uma explicação do código em Python.

### Entendendo o Problema e os Requisitos

* **Disciplina**: Inteligência Artificial
* **Professor**: Danniel Cavalcante Lopes
* **Período**: 2025.1

**O que fazer:**
Implementar o algoritmo Backpropagation para uma MLP para classificar números de 0 a 7.

**Restrições/Dicas Importantes:**
* **Não é permitido** usar bibliotecas prontas como Weka, Toolbox do Matlab, Tensorflow, Scikit-learn, etc. Isso significa que você terá que implementar as operações matemáticas e a lógica do Backpropagation "do zero".
* Usar **função não linear na camada oculta**. (Muito importante! Geralmente Sigmoide ou ReLU).
* Testar diversas configurações. (Número de neurônios na camada oculta, taxa de aprendizado, número de épocas, etc.).
* A entrega fora do prazo acarreta perda de 0,5 pontos para cada dia passado.
* A apresentação será em dupla.
* **Data limite para entrega:** 11/06/2025 e 16/06/2025 (de acordo com o sorteio).

**Dados de Entrada e Saída (Sugestão):**
A tabela de dados a seguir fornece os valores:
| Entrada (Decimal) | Entrada (Normalizada) | Saída Desejada |
| :---------------- | :-------------------- | :------------- |
| 0                 | 0.0                   | 0,0,0          |
| 1                 | 0.14                  | 0,0,1          |
| 2                 | 0.28                  | 0,1,0          |
| 3                 | 0.42                  | 0,1,1          |
| 4                 | 0.57                  | 1,0,0          |
| 5                 | 0.71                  | 1,0,1          |
| 6                 | 0.85                  | 1,1,0          |
| 7                 | 1.0                   | 1,1,1          |

### Passo a Passo: O Que Saber, Estudar e Fazer

#### Parte 1: Fundamentos Teóricos (O que estudar)

1.  **Redes Neurais Artificiais (RNAs):**
    * **Conceitos Básicos:** O que são neurônios, camadas (entrada, oculta, saída), pesos, vieses (bias).
    * **MLP (Multi-Layer Perceptron):** Entender a arquitetura de uma rede com múltiplas camadas (pelo menos uma camada oculta, além da entrada e saída).

2.  **Funções de Ativação:**
    * **Função Não Linear:** Entender por que são essenciais em camadas ocultas para permitir que a rede aprenda padrões complexos (sem elas, uma MLP seria equivalente a uma regressão linear).
    * **Funções Comuns:**
        * **Sigmoide (Logística):** $f(x) = \frac{1}{1 + e^{-x}}$. Sua derivada é $f'(x) = f(x) * (1 - f(x))$. É uma boa escolha para a camada oculta e para a camada de saída se as saídas forem probabilísticas ou binárias.
        * **ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$. Sua derivada é $1$ para $x > 0$ e $0$ para $x \le 0$. É popular em camadas ocultas.
        * **Linear/Identidade:** $f(x) = x$. Geralmente usada na camada de saída para problemas de regressão. Para classificação binária (0 ou 1), a Sigmoide é mais adequada. Para múltiplas classes (como aqui), a Sigmoide ou a Softmax podem ser consideradas na saída, mas a Sigmoide para cada bit independente funciona bem para a representação binária.

3.  **Algoritmo Backpropagation:**
    * **O Coração do Aprendizado:** Compreender como ele funciona para ajustar os pesos e vieses da rede.
    * **Passo a Passo:**
        * **Feedforward (Propagação para Frente):** Calcular as saídas dos neurônios camada por camada, da entrada até a saída.
        * **Cálculo do Erro:** Comparar a saída da rede com a saída desejada (real) usando uma **função de custo/perda** (ex: Erro Quadrático Médio - MSE). A fórmula do MSE é $MSE = \frac{1}{N} \sum_{i=1}^{N} (y_{real,i} - y_{previsto,i})^2$.
        * **Backpropagation (Propagação para Trás):** Calcular o gradiente do erro em relação a cada peso e viés da rede, começando da camada de saída e retrocedendo até a camada de entrada. Isso envolve a **regra da cadeia** do cálculo.
        * **Atualização de Pesos e Vieses:** Ajustar os pesos e vieses usando o gradiente e a **taxa de aprendizado** (learning rate). A fórmula geral é $Novo\_Peso = Peso\_Antigo - Taxa\_Aprendizado \times Gradiente\_do\_Erro\_em\_relação\_ao\_Peso$.
        * **Épocas:** Entender que o processo de feedforward, backpropagation e atualização de pesos é repetido por um número de épocas para que a rede "aprenda" os padrões nos dados.

#### Parte 2: Pré-processamento de Dados (O que fazer)

* **Entradas e Saídas:** Você já tem uma sugestão de dados.
    * **Entradas:** Os números normalizados de $0.0$ a $1.0$. Para os números de $0$ a $7$, a entrada normalizada é um valor único ($0.0$, $0.14$, ..., $1.0$). Portanto, sua camada de entrada terá $1$ neurônio.
    * **Saídas Desejadas:** Representação binária de $3$ bits (ex: $[0,0,0]$ para $0$, $[0,0,1]$ para $1$, etc.). Sua camada de saída terá $3$ neurônios.

#### Parte 3: Estrutura da Rede (O que fazer e estudar)

* **Camada de Entrada:** $1$ neurônio (para o valor normalizado).
* **Camada Oculta:**
    * Escolha um número de neurônios (ex: $2, 3, 4, 5$). O trabalho pede para testar diversas configurações, então comece com um e experimente outros.
    * Aplique uma função de ativação não linear aqui (Sigmoide é uma boa pedida).
* **Camada de Saída:** $3$ neurônios (para os $3$ bits binários).
    * Aqui, a função de ativação Sigmoide é uma boa escolha, pois ela espreme a saída entre $0$ e $1$, o que é útil para representar probabilidades ou valores binários.

#### Parte 4: Implementação (O que fazer - em Python)

* **Inicialização de Pesos e Vieses:**
    * Gerar pesos e vieses aleatoriamente (valores pequenos, como entre $-1$ e $1$ ou $-0.5$ e $0.5$) para cada conexão entre neurônios e para cada neurônio.
    * Isso é crucial para que a rede não comece em um ponto "morto" e possa aprender.
* **Funções de Ativação e Suas Derivadas:**
    * Implemente a função Sigmoide e sua derivada.
    * Se for experimentar, implemente ReLU e sua derivada também.
* **Função de Erro (Loss Function):**
    * Implemente o Erro Quadrático Médio (MSE).
* **Loop de Treinamento (Épocas):**
    * Para cada época:
        * Para cada par (entrada, saída\_desejada) no seu dataset:
            * **Feedforward:**
                * Calcular as saídas da camada oculta.
                * Calcular as saídas da camada de saída.
            * **Backpropagation:**
                * Calcular o erro na camada de saída.
                * Propagar o erro de volta para a camada oculta.
            * **Atualização de Pesos e Vieses:**
                * Usar os gradientes calculados para ajustar os pesos e vieses de ambas as camadas (oculta e saída) usando a taxa de aprendizado.
* **Testes e Avaliação:**
    * Após o treinamento, teste a rede com as entradas de $0$ a $7$ e veja se as saídas se aproximam dos valores binários desejados.
    * Você pode arredondar as saídas para $0$ ou $1$ para fazer a classificação final.

#### Parte 5: Teste de Configurações (O que fazer)

* **Número de Neurônios na Camada Oculta:** Experimente $2, 3, 4, 5$ neurônios, por exemplo.
* **Taxa de Aprendizado:** Comece com $0.1$ e experimente $0.01, 0.5$, etc.
* **Número de Épocas:** Comece com $1000, 5000$ e aumente se a rede não estiver aprendendo bem.
* **Funções de Ativação:** Embora o requisito seja "não linear na camada oculta", você pode testar Sigmoide vs. ReLU na oculta (se quiser ir além do básico) e a Sigmoide na saída é geralmente a melhor escolha para saída binária.