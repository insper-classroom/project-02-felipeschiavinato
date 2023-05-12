# Projeto de Reinforcement Learning para o Problema do Caixeiro Viajante na Fórmula 1

Este projeto aplica técnicas de aprendizado por reforço, especificamente Pointer Networks, para resolver o problema do Caixeiro Viajante para 23 nós, onde cada nó representa uma cidade em que ocorre uma corrida da Fórmula 1 ao longo do ano. O objetivo é encontrar o caminho mais curto entre todas as cidades. Além disso, o projeto experimenta com diferentes algoritmos para avaliar e comparar seu desempenho.

## Sumário



1. [Introdução](#introdução)
2. [Metodologia](#metodologia)
3. [Instalação](#instalação)
4. [Referencias](#referencias)

## Introdução

O problema do Caixeiro Viajante é um problema clássico em ciência da computação. Ele envolve encontrar a rota mais curta que permite visitar uma série de locais uma única vez e retornar ao ponto de origem. Neste projeto, aplicamos esse problema ao contexto da Fórmula 1, onde cada "local" é uma cidade que hospeda uma corrida.

## Metodologia

Este projeto utiliza Pointer Networks, uma variante da arquitetura de redes neurais sequenciais que pode aprender a ordem dos elementos, esta rede é ultil pois permite um output variavel uma vez que a cada cidade visitada no percurso, ela nao pode mais ser um output. Além disso, experimentaremos com vários algoritmos de aprendizado por reforço para comparar e avaliar seu desempenho no problema do Caixeiro Viajante. Finalmente a validação será atravez do algorítimo de Held-Karp, que é um algorítimo exato para o problema do caixeiro viajante que nos garante uma política ótima.


## Referencias

[Neural Combinatorial Optimization With Reinforcement Learning](https://arxiv.org/pdf/1611.09940.pdf)
