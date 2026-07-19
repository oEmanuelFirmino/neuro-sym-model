# Posicionamento: DLG vs. ∂ILP (Evans & Grefenstette, 2018)

> Material de apoio para a Seção 2 do artigo — fecha o item M1 do parecer
> junto com `posicionamento-dlgn-petersen.md` e
> `posicionamento-ntp-deepproblog.md`. **Conferir as afirmações técnicas
> contra o artigo original antes de citar** — escrito de memória da
> literatura.
>
> Referência primária a adicionar à bibliografia:
> - EVANS, R.; GREFENSTETTE, E. *Learning Explanatory Rules from Noisy Data*.
>   Journal of Artificial Intelligence Research (JAIR), v. 61, 2018.

## 1. O que o ∂ILP é

Indução de programas lógicos diferenciável: o sistema **aprende regras de
primeira ordem** a partir de exemplos positivos e negativos. O espaço de
regras candidatas é gerado por templates de cláusulas; a cada candidata
associa-se um peso contínuo (softmax), e a inferência é um encadeamento
progressivo (forward chaining) diferenciável — iterações do operador de
consequência sobre valorações contínuas de átomos ground. Ao final do
treino, os pesos concentram-se nas cláusulas corretas e o resultado é um
**programa lógico legível por humanos**, robusto a ruído nos dados. Custo
característico: a combinatória dos templates torna o método intensivo em
memória, restringindo aridade, número de predicados e escala.

## 2. Tabela de diferenciação

| Eixo | ∂ILP | DLG (este trabalho) |
|---|---|---|
| Objeto aprendido | **as regras** (estrutura do programa lógico) | **os groundings** (embeddings + predicados neurais) sob regras dadas |
| Símbolos | discretos; valorações sobre átomos ground, sem embeddings | constantes em R^d, predicados como MLPs |
| Inferência | forward chaining diferenciável sobre o espaço de templates | avaliação da fórmula de prova composta dinamicamente (Product T-norm) |
| Percepção subsimbólica | fora do escopo (entradas simbólicas) | nativa (embeddings treináveis, predicados neurais) |
| Explicabilidade | o *produto* é interpretável: um programa legível | o *processo* é explicável: atribuição por gradiente sobre o DAG de prova, com fidelidade causal medida |
| Gargalo | memória (combinatória de templates) | cobertura de negativos do predicado base (saturação da t-conorm) |

## 3. Formulação de diferenciação (proposta de texto para a Seção 2)

> "O ∂ILP de Evans e Grefenstette ataca o problema inverso ao aqui tratado:
> dadas as valorações dos fatos, induz as regras do programa lógico por
> relaxação diferenciável de um espaço de templates de cláusulas. Os DLGs
> assumem as regras como conhecimento de domínio e aprendem as
> representações contínuas (groundings e predicados neurais) que as
> satisfazem, permitindo integrar percepção subsimbólica — fora do escopo do
> ∂ILP, cujos símbolos permanecem discretos. As abordagens são
> complementares: a indução de regras à la ∂ILP é uma direção natural para
> eliminar a principal limitação dos DLGs, a especificação manual dos
> axiomas (Seção 7.2), e sua integração é apontada como trabalho futuro
> (Seção 7.3)."

Nota estratégica: o artigo **já** lista a especificação manual de axiomas
como limitação (7.2) e a integração com ILP como trabalho futuro (7.3) — a
diferenciação com ∂ILP não é defensiva, é a formalização de algo que o texto
já reconhece. É o mais fácil dos três itens do M1.

## 4. Fechamento do M1 — parágrafo-síntese do espectro

Com os três documentos de posicionamento, a Seção 2 revisada pode organizar
a literatura diretamente relacionada num espectro de "o que é aprendido",
tornando a lacuna do DLG explícita:

> "As abordagens diferenciáveis para lógica distinguem-se pelo objeto de
> aprendizado. Os Differentiable Logic Gate Networks aprendem a estrutura de
> circuitos proposicionais para inferência eficiente, sem conhecimento
> prévio; o ∂ILP induz as próprias regras de primeira ordem a partir de
> exemplos; NTPs aprendem representações realizando busca de prova
> diferenciável com indução de regras via templates; e o DeepProbLog aprende
> os parâmetros de predicados neurais sob semântica probabilística exata,
> ao custo de compilação simbólica. Os DLGs ocupam uma posição distinta:
> assumem os axiomas como dados, aprendem groundings contínuos, e mantêm a
> semântica fuzzy na própria inferência — o que torna a atribuição por
> gradiente sobre o DAG de prova um subproduto direto da execução, com
> fidelidade causal verificável."

## 5. Checklist de citações novas exigidas pelo M1/m4 (estado)

- [x] Análise Petersen et al. 2022/2024 (`posicionamento-dlgn-petersen.md`)
- [x] Análise ∂ILP (este documento)
- [x] Análise NTP/DeepProbLog aprofundada (`posicionamento-ntp-deepproblog.md`)
- [ ] Incorporar os parágrafos propostos ao manuscrito (Seção 2)
- [ ] Adicionar as entradas bibliográficas (Petersen ×2, Evans & Grefenstette,
      Minervini et al. se citar as extensões do NTP)
- [ ] m4 (relacionado): Power et al. 2022 e Nanda et al. 2023 para ancorar a
      discussão de grokking — necessários mesmo com as alegações amenizadas
