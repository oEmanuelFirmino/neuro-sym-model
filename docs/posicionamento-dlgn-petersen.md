# Posicionamento: DLG vs. Differentiable Logic Gate Networks (Petersen et al.)

> Material de apoio para a Seção 2 do artigo — item M1 do parecer (colisão de
> nome e proposta com linha não citada). **Conferir as afirmações técnicas
> contra os artigos originais antes de citar** — escrito de memória da
> literatura.
>
> Referências primárias a adicionar à bibliografia:
> - PETERSEN, F.; BORGELT, C.; KUEHNE, H.; DEUSSEN, O. *Deep Differentiable
>   Logic Gate Networks*. NeurIPS 2022.
> - PETERSEN, F. et al. *Convolutional Differentiable Logic Gate Networks*.
>   NeurIPS 2024.

## 1. O que a linha de Petersen et al. é

Redes em que cada nó é uma porta lógica binária de 2 entradas com
conectividade fixa e aleatória. Aprendizado: cada nó mantém uma distribuição
(softmax) aprendível sobre os 16 operadores booleanos possíveis; no treino,
cada operador é substituído por sua relaxação contínua real-valuada — a
mesma álgebra de produto usada no nosso artigo (AND → a·b, OR → a+b−ab) — e a
saída do nó é a esperança sob a distribuição. Após o treino, cada nó é
**discretizado** para a porta de maior probabilidade, produzindo um circuito
booleano puro de inferência extremamente rápida em CPU/FPGA (sem ponto
flutuante). A versão de 2024 adiciona kernels convolucionais de árvores
lógicas e pooling por OR. O objetivo declarado é **eficiência de inferência
em hardware** (linha de pesquisa vizinha a redes binarizadas/quantizadas).

## 2. Tabela de diferenciação

| Eixo | DLGN (Petersen et al.) | DLG (este trabalho) |
|---|---|---|
| Objeto aprendido | qual porta cada nó é (estrutura do circuito) | groundings (embeddings + MLPs de predicados) sob regras dadas |
| Nível lógico | proposicional (bits, sem semântica de domínio) | primeira ordem (predicados, constantes, axiomas) |
| Papel da relaxação | truque de treino; inferência final é discreta | semântica de inferência permanente (graus contínuos) |
| Conhecimento prévio | nenhum; conectividade aleatória | axiomas de domínio na perda e na inferência composta |
| Objetivo | velocidade/eficiência em hardware | integração de conhecimento + explicabilidade |
| Explicabilidade | circuito discretizado formalmente inspecionável, mas não interpretável na prática (milhões de portas aleatórias); não é alegação deles | atribuição por gradiente sobre o DAG de prova, com fidelidade causal medida (deletion/insertion) |

## 3. Formulação de diferenciação (proposta de texto para a Seção 2)

> "Apesar da semelhança de nomenclatura, os Differentiable Logic Gate
> Networks de Petersen et al. resolvem um problema disjunto do aqui tratado:
> empregam relaxações contínuas da lógica booleana (a mesma álgebra de
> produto adotada neste trabalho) como mecanismo de treino para aprender a
> estrutura de circuitos proposicionais que são, ao final, discretizados
> visando inferência eficiente em hardware, sem qualquer conhecimento
> simbólico prévio. Em contraste, os DLGs mantêm a semântica contínua na
> própria inferência, com o propósito de integrar axiomas de primeira ordem
> fornecidos a priori e extrair explicações diferenciáveis do processo
> dedutivo. A ferramenta matemática é compartilhada; o objeto aprendido, o
> nível lógico e a finalidade são distintos."

## 4. O espectro completo do M1 (organização sugerida da Seção 2 revisada)

Os três trabalhos exigidos pelo parecer formam um espectro coerente de "o
que é aprendido":

1. **Petersen et al. (DLGN)** — aprende a *estrutura do circuito*
   (proposicional, sem conhecimento prévio);
2. **∂ILP (Evans & Grefenstette, 2018)** — aprende as *regras* (indução
   diferenciável de programas lógicos de primeira ordem);
3. **DLG (este trabalho)** — assume as regras e aprende os *groundings*, com
   inferência composta explicável.

Apresentar os três nessa ordem torna a diferenciação natural e responde ao
M1 de uma vez.

## 5. Risco do nome (decisão pendente com o orientador)

"Differentiable Logic Graphs" (DLG) vs. "Differentiable Logic Gate Networks"
(DLGN) é uma colisão que sobreviverá a qualquer parágrafo de diferenciação —
a linha de Petersen é consolidada em NeurIPS e domina a busca pelo termo.
Recomendação: considerar renomear a arquitetura na revisão (candidato
natural pós-refoco: *Differentiable Proof Graphs*, já que a inferência
composta via DAG de prova é agora o centro da contribuição). Mudar o nome
agora custa pouco; carregar a ambiguidade pela revisão e pela vida do artigo
custa caro.
