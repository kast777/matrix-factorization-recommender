# Fatorização de Matrizes em Sistemas de Recomendação: Otimização de Modelos Latentes e Análise de Escala

Este projeto foi desenvolvido no âmbito da Unidade Curricular de **Métodos Matemáticos em Inteligência Artificial** do Mestrado em **Matemática Aplicada para a Indústria** ministrado pelo **ISEL** (Instituto Superior de Engenharia de Lisboa).

> **Classificação Final do Trabalho: 20 / 20**

---

## 🎯 Objetivo

O foco central deste trabalho foi a exploração de modelos de filtragem colaborativa baseados em **Fatorização de Matrizes**. O desafio consistiu em implementar, otimizar e comparar modelos de **NMF** (Non-negative Matrix Factorization) e **WMF** (Weighted Matrix Factorization) num cenário de **feedback implícito**, utilizando o dataset MovieLens 25M (+25 milhões de interações) em ambiente de hardware restrito.

---

## 🔬 Análise de Performance e Hiperparâmetros

### 1. O Custo da Exploração: Manual vs. Otimizado

A narrativa central deste projeto destaca o impacto da eficiência algorítmica na fase de experimentação:

* **Implementação Manual (NumPy + SGD):** A natureza estocástica do SGD faz com que o modelo não convirja para um ponto fixo, mas sim para um intervalo de estabilidade "ruidoso". O custo computacional de realizar uma **Grid Search de 16 combinações** de hiperparâmetros atingiu as **5 horas**, evidenciando a lentidão dos ciclos `for` em Python para grandes volumes de dados.
* **Otimizada (Implicit/Scikit-Learn):** Através de bibliotecas que utilizam **ALS** (Alternating Least Squares) e rotinas em C++/Cython, essa exploração é reduzida para **segundos**, permitindo uma iteração científica muito mais ágil.

### 2. Warm Start vs. Cold Start e a Divisão de Dados

Uma das lições fundamentais foi a correção da metodologia de teste:

* Inicialmente, uma divisão por utilizadores gerou um cenário de **Cold Start**, onde o NMF falhava por não possuir perfis latentes para novos utilizadores.
* A transição para uma **Divisão por Interações (Warm Start)** permitiu que o modelo utilizasse o conhecimento prévio dos utilizadores para prever os itens "escondidos", elevando a Precision@10 de patamares marginais para resultados competitivos.

### 3. RMSE vs. Precision@K: O Paradoxo do Erro

Observou-se que a minimização do erro quadrático (RMSE) nem sempre corrobora a qualidade da recomendação. Em sistemas de feedback implícito, a **ordenação relativa (Ranking)** é mais valiosa que a precisão do valor previsto. O ranking estabiliza a capacidade de sugestão muito antes do erro atingir o seu patamar mínimo de oscilação.

---

## ⚙️ Metodologia e Implementação

### Engenharia de Dados e Esparsidade

* **Feedback Implícito:** Tratamos a ausência de dados não como "desagrado", mas como incerteza. O **WMF** revelou-se superior ao introduzir pesos diferenciados para interações observadas e não observadas.
* **Amostragem Negativa:** No WMF manual, a implementação de *Negative Sampling* foi crucial para ensinar o modelo a distinguir entre o que o utilizador consome e o vasto universo de itens não interagidos.

### Interpretabilidade Latente (t-SNE)

Para validar a aprendizagem, aplicámos **t-SNE** sobre a matriz de itens . O resultado revelou agrupamentos geométricos coerentes: filmes do mesmo género aglomeraram-se no espaço latente sem que o algoritmo tivesse acesso a qualquer metadado (títulos ou categorias) durante o treino.

---

## ⚠️ Reflexão Crítica: Lições de um Projeto de Escala

Este trabalho reforçou que, ao lidar com 25 milhões de interações, a **infraestrutura domina a teoria**:

1. **Otimização é Viabilidade:** O abismo temporal entre a exploração manual e a otimizada define se um projeto é academicamente interessante ou industrialmente aplicável.
2. **Ruído Estocástico:** Aceitar que o SGD "estaciona" num intervalo de erro e não num valor exato é fundamental para definir critérios de paragem (*early stopping*).
3. **Hardware como Restrição:** A necessidade de gerir memória forçou a utilização de matrizes esparsas e operações vetorizadas, competências essenciais para qualquer Engenheiro de ML.

---

## 🛠️ Tecnologias Utilizadas

* **Core:** Python, NumPy, Pandas, SciPy.
* **ML:** Scikit-Learn, Implicit (ALS).
* **Visualização:** Matplotlib, Seaborn, t-SNE.

---

## 📖 Relatório Completo
O estudo detalhado, incluindo a fundamentação matemática (decomposição matricial e gradiente descendente), está disponível em [PDF](./docs/Fatorizacao_de_Matrizes_em_Sistemas_de_Recomendacao.pdf).
