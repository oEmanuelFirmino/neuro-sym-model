# Posicionamento: DLG vs. NTP e DeepProbLog (eixo explicabilidade)

> Material de apoio para a revisão da Seção 2 do artigo, sob a tese
> reposicionada (explicabilidade + consistência lógica). Compara o DLG, como
> de fato implementado neste repositório, com os dois sistemas de proving
> diferenciável mais próximos. **Antes de citar, conferir as afirmações
> técnicas contra os artigos originais** — este documento foi escrito de
> memória da literatura e deve ser validado ponto a ponto.
>
> Referências primárias:
> - NTP: Rocktäschel & Riedel, *End-to-End Differentiable Proving*, NeurIPS 2017
>   (já citado como [25] no manuscrito); extensões: Minervini et al., *Greedy
>   NTPs* (AAAI 2020) e *Learning Reasoning Strategies* (CTP, ICML 2020).
> - DeepProbLog: Manhaeve et al., NeurIPS 2018 (já citado como [5]).

## 1. O que cada sistema é

| | NTP | DeepProbLog | DLG (este trabalho) |
|---|---|---|---|
| Semântica | Similaridade de embeddings (unificação suave por kernel RBF) | Probabilística exata (ProbLog + nADs) | Fuzzy (Product T-norm) |
| Inferência | Backward-chaining estilo Prolog; escore de prova = min das unificações no caminho; agregação max/top-k entre provas alternativas | Grounding → compilação (SDD/d-DNNF) → Weighted Model Counting | Avaliação direta da fórmula de prova composta dinamicamente, sem compilação |
| Regras | **Induz** regras via templates | Dadas no programa lógico | Dadas (axiomas) |
| Custo de inferência | Busca de prova cara (exige aproximações top-k/greedy nas extensões) | Compilação #P-difícil no caso geral | Uma avaliação forward do DAG grounded, O(|V|+|E|) |

## 2. O que é "a explicação" em cada um

Este é o eixo em que o DLG precisa se diferenciar — e onde há espaço real.

**NTP**: a explicação é o **caminho de prova discreto** de maior escore
(argmax sobre provas). É um artefato simbólico, legível — e nisso o NTP é
*mais* interpretável que um mapa de atribuição numérica. Mas:
- a agregação por max roteia o gradiente (e o crédito) por **um único
  caminho vencedor**; atribuição graduada entre provas alternativas não é
  natural no formalismo;
- obter a explicação exige a busca de prova (o mesmo custo que torna o NTP
  caro), não vem de graça da inferência;
- o artigo do NTP **demonstra** provas extraídas, mas não **valida
  quantitativamente** a fidelidade delas (nenhum protocolo causal).

**DeepProbLog**: a semântica probabilística permite explicações bem
fundamentadas (ex.: Most Probable Explanation, enumeração de provas com
probabilidades). Mas:
- a explicação requer **inferência adicional** sobre o circuito compilado
  (MPE é outro problema de inferência, não um subproduto do forward);
- o circuito é produto de compilação estática por consulta — a estrutura
  explicativa não é o grafo de execução do aprendizado;
- também não há, no artigo original, validação causal quantitativa da
  qualidade explicativa.

**DLG**: a explicação é a **atribuição por gradiente sobre os embeddings,
computada pelo mesmo backward pass usado no treinamento**, sobre o mesmo DAG
que produziu a resposta. Propriedades que a diferenciam:
1. **Custo zero adicional**: um backward no DAG já avaliado — O(|V|+|E|),
   coerente com o argumento de amortização da Seção 4.7 do artigo.
2. **Atribuição graduada por todas as provas simultaneamente**: a Product
   T-norm/T-conorm é suave, então o gradiente flui por **todos** os ramos da
   disjunção de provas, com peso proporcional à contribuição de cada um — em
   contraste com o winner-take-all do max no NTP. (Medido: ~45% da massa de
   gradiente nos intermediários do caminho, `evidence/kinship_proof_dag/`.)
3. **Fidelidade causal verificada, não presumida**: o protocolo
   deletion/insertion com baseline aleatório (Δ=0.93 ao deletar os
   intermediários da prova vs. 0.0 no preditor plano) mostra que a atribuição
   aponta para entidades das quais a predição **de fato depende**. Nenhum dos
   dois sistemas comparados publica protocolo equivalente.
4. **Delimitação de escopo explícita**: mostramos que sem encadeamento de
   regras na inferência a "explicabilidade intrínseca" é tautológica
   (concentração=1.0 garantida pela arquitetura). Assumir essa delimitação no
   texto é um diferencial de maturidade metodológica.

## 3. Formulação honesta da contribuição (proposta de texto)

> "Diferentemente dos NTPs, cuja explicação é o caminho de prova discreto
> selecionado por agregação max — legível, porém obtido ao custo da busca de
> prova e sem atribuição graduada entre provas alternativas — e do
> DeepProbLog, em que explicações probabilísticas exigem inferência adicional
> sobre o circuito compilado, o DLG obtém a atribuição como subproduto direto
> do backward pass sobre o próprio DAG de avaliação, com custo O(|V|+|E|) e
> crédito distribuído continuamente por todas as provas ativas. Crucialmente,
> não presumimos a fidelidade dessa explicação: ela é validada por protocolos
> causais de deleção/inserção contra baseline aleatório, e caracterizamos
> formalmente as condições sob as quais a explicabilidade intrínseca é
> informativa (inferência com encadeamento de regras) versus estruturalmente
> trivial (consultas atômicas)."

## 4. Contrapontos que o texto deve reconhecer (antes que o revisor o faça)

1. **Legibilidade**: um caminho de prova simbólico (NTP) é mais legível para
   humanos que um vetor de atribuições. Resposta possível: os dois são
   complementares — no DLG o suporte da atribuição *identifica* o subgrafo de
   prova relevante (as entidades com massa > 0), de onde um caminho simbólico
   pode ser lido; e a atribuição graduada quantifica o que o caminho discreto
   não expressa (importância relativa).
2. **Semântica**: DeepProbLog tem probabilidades calibradas; graus fuzzy da
   Product T-norm **não são probabilidades** e o texto não deve tratá-los
   como tal.
3. **Indução de regras**: NTP aprende regras; o DLG as pressupõe. Já está nas
   limitações do artigo (Seção 7.2) — manter.
4. **Saturação da T-conorm**: a disjunção de provas satura sem cobertura de
   negativos no predicado base (achado nosso, custo O(|E|²) — registrado na
   Fase 4 do plano). O DeepProbLog, com inferência exata, não tem o problema
   análogo. Reconhecer como trade-off do relaxamento fuzzy.
5. **Escala**: os domínios avaliados são pequenos (p=13; árvores de ~10
   entidades). NTP/CTP reportam KBs maiores (ex.: benchmarks de kinship
   reais, WordNet). Mitigação: enquadrar como estudo controlado de
   propriedades, não como sistema de escala.

## 5. Itens de M1 que este documento NÃO cobre

O parecer também exige citar e diferenciar **Petersen et al. (Differentiable
Logic Gate Networks, NeurIPS 2022/2024)** — colisão de nome e proposta — e
**∂ILP (Evans & Grefenstette, 2018)**. São eixos diferentes (aprendizado de
estrutura de circuito booleano; indução de regras diferenciável) e merecem
parágrafos próprios na Seção 2; não confundir com o eixo de explicabilidade
tratado aqui.
