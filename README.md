# MCOM-2025 – Métodos Computacionais

Disciplina de mestrado focada na implementação **manual** de rotinas numéricas fundamentais (operações vetoriais, matriciais, entre outras) para compreender como funcionam internamente bibliotecas científicas como o NumPy.  

## Objetivo do repositório

- Implementar, a partir do zero, funções numéricas básicas, construindo uma mini-biblioteca própria em Python.  
- Utilizar testes automatizados para validar o comportamento de cada função e desenvolver boas práticas de programação científica.  

## Arquivos principais

- `functions_mat.py`: arquivo onde devem ser implementadas as funções numéricas solicitadas (operações com vetores, matrizes e funções auxiliares).  
- `test_functions.py`: arquivo de testes que define o comportamento esperado de cada função e verifica se a implementação em `functions_mat.py` está correta.  

## Como trabalhar com as funções

1. Abrir `functions_mat.py` e analisar as assinaturas, comentários e docstrings para entender o que cada função deve fazer.  
2. Implementar cada rotina passo a passo, evitando o uso direto de funções prontas equivalentes do NumPy quando indicado.  
3. Executar `test_functions.py` (por exemplo, com `pytest` ou `python test_functions.py`) para verificar quais funções passam nos testes.  
4. Refinar a implementação até que todos os testes sejam aprovados, consolidando sua própria biblioteca numérica.  

## Pré-requisitos

- Python 3 instalado e editor de texto ou IDE de preferência.  
- Conhecimentos básicos de programação em Python.  
- Noções de álgebra linear e cálculo numérico.  

## Contato

Dúvidas sobre o uso deste repositório ou sobre as implementações devem ser encaminhadas pelos canais oficiais da disciplina ou diretamente ao docente responsável.
