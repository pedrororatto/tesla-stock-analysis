# Tesla Stock Analysis

Este projeto consiste em uma aplicação interativa desenvolvida em Python para análise de dados históricos das ações da Tesla. O objetivo principal é oferecer insights financeiros por meio de estatísticas descritivas, testes estatísticos e visualizações gráficas, além de simular a evolução de um investimento baseado em uma data de compra selecionada pelo usuário.

## 🎯 Funcionalidades

- **Estatísticas Descritivas**:
  - Média, Mediana, Moda, Variância, Desvio Padrão e Desvio Médio Absoluto dos preços de fechamento.
  - Histograma interativo dos preços.

- **Teste T de Student**:
  - Comparação da média dos preços de fechamento com um valor hipotético para determinar significância estatística.

- **Previsão com Regressão Linear**:
  - Modelo de regressão linear para prever os preços futuros das ações.

- **Evolução do Investimento**:
  - Simulação da evolução de um investimento a partir de uma data de compra, considerando rentabilidade acumulada ao longo do tempo.

- **Volume total de vendar por ano e mensal**:
  - Exemplo gráfico de como foram as vendas ao longo do ano e por mês.

- **Indicadores de compra e venda**:
  - Com base nos dados históricos, indica quando foi bom ter comprado e quando foi melhor ter vendido.

- **Correlação entre preço e volume**
  - Gráfico de correlação entre essas duas variáveis. 

## 🛠️ Tecnologias Utilizadas

- **Linguagem de Programação**: Python
- **Bibliotecas Principais**:
  - [Streamlit](https://streamlit.io/): Para criação de dashboards interativos.
  - [Pandas](https://pandas.pydata.org/): Manipulação e análise de dados.
  - [Plotly](https://plotly.com/): Criação de gráficos interativos.
  - [NumPy](https://numpy.org/): Cálculos matemáticos e estatísticos.
  - [Scikit-learn](https://scikit-learn.org/): Modelagem preditiva com regressão linear.

## ⚙️ Estrutura do Projeto

- **app.py**: Código principal da aplicação, incluindo processamento de dados, cálculos estatísticos e geração de gráficos.
- **Dataset/**: Diretório contendo os dados históricos das ações da Tesla (CSV).
- **README.md**: Este arquivo explicando o projeto.
- **requirements.txt**: Este arquivo consiste nas dependências do projeto.
