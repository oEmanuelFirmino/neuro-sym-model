# Estrutura do Artigo Revisado — blueprint seção a seção

> Documento normativo para a reescrita do manuscrito sob o redirecionamento
> estratégico (consistência lógica + explicabilidade intrínseca como
> contribuições centrais; generalização por restrições semânticas como achado
> secundário). Cada seção indica: o que muda, qual evidência do repositório a
> sustenta, e quais itens do parecer ela resolve.
>
> Fontes: `plano-correcoes-artigo.md` (resultados), `posicionamento-*.md`
> (Seção 2), `experiments/evidence/` (tabelas e figuras).

---

## Título

**Sai**: "Generalização Acelerada" (não sustentado — a transição de grokking
não foi observada).
**Proposta**: *"Grafos de Prova Diferenciáveis: Arquitetura Neuro-Simbólica
com Consistência Lógica e Explicabilidade Intrínseca Verificável"*.
**Decisão pendente com o orientador**: renomear DLG → DPG (Differentiable
Proof Graphs) para eliminar a colisão com Petersen et al. (ver
`posicionamento-dlgn-petersen.md`, Seção 5). Se o nome DLG for mantido,
parágrafo de desambiguação obrigatório na Seção 2.

## Resumo (reescrever por completo — resolve item editorial "resumo truncado")

Estrutura em 6 frases: (1) problema da integração neuro-simbólica; (2) a
arquitetura: predicados neurais + composição dinâmica de fórmulas de prova
com Product T-norms + regularização L1; (3) alegação 1 com número:
consistência dedutiva — relações derivadas nunca treinadas discriminam
0.94/0.02 (sintético) e ≥0.75/≤0.003 nas 4 relações do dataset de Hinton
(1986); (4) alegação 2 com número: explicabilidade causalmente fiel — 38-45%
da massa de gradiente nos intermediários do caminho de prova e Δ=0.77-0.93 na
deleção causal, contra zero estrutural em preditores planos; (5) achado
secundário: restrições semânticas melhoram generalização (31-36% vs 0% na
formulação por classificação), sem alegar aceleração de grokking; (6)
esparsidade calibrada (80% com γ=1e-2) e código/dados públicos.

## 1. Introdução

- **1.1 Contexto**: mantém, enxugando a motivação por grokking.
- **1.2 Objetivo**: reescrever — o objetivo é uma arquitetura cuja inferência
  atravessa a estrutura lógica, tornando consistência e explicação
  propriedades mensuráveis da execução, não pós-processamentos.
- **1.3 Justificativa**: mantém o argumento de explicabilidade intrínseca,
  agora com a promessa de *verificação* (protocolos causais) — diferencial
  sobre a versão anterior, que apenas postulava.
- Parágrafo final de contribuições (novo, lista numerada):
  C1 consistência dedutiva medida; C2 explicabilidade causalmente fiel com
  protocolo quantitativo; C3 delimitação formal de quando a explicabilidade
  intrínseca é informativa; C4 esparsidade estrutural calibrada; C5 achado
  secundário sobre generalização semântica (com resultado negativo honesto da
  formulação relacional).

## 2. Revisão Bibliográfica (reestruturar — resolve M1 e m4)

Organizar pelo espectro "o que é aprendido" (parágrafo-síntese pronto em
`posicionamento-dilp.md`, Seção 4):

- **2.1 Restrições semânticas na perda** (Semantic Loss — mantém o texto atual).
- **2.2 Programação lógica probabilística** (DeepProbLog — mantém + parágrafo
  de explicabilidade de `posicionamento-ntp-deepproblog.md`).
- **2.3 Semânticas contínuas** (LTN — mantém).
- **2.4 Proving diferenciável e indução de estrutura** (NOVA):
  NTP (+ extensões Greedy NTP/CTP), ∂ILP, Petersen et al. 2022/2024 — os três
  parágrafos de diferenciação já redigidos nos documentos de posicionamento.
  ⚠ Validar afirmações técnicas contra os artigos originais antes de citar.
- **2.5 Contribuição proposta**: reescrever com C1-C5; a frase-chave é a
  "combinação integrada + execução dinâmica via DAG" exigida por M1(iii),
  agora com o DAG de prova como mecanismo concreto.
- **m4**: citar Power et al. (2022) e Nanda et al. (2023) na discussão de
  generalização (Seção 6.4), mesmo com alegações amenizadas.

## 3. Fundamentação Teórica

- 3.1-3.3: mantêm (elogiadas no parecer).
- **3.4 (resolve M4)**: qualificar a alegação da Product T-norm — reconhecer
  vanishing gradient em conjunções longas (van Krieken et al.) e explicitar
  que a média aritmética em L_semantic é o mitigador; a vantagem defendida
  passa a ser (i) suavidade vs. regiões planas de Łukasiewicz (evidência:
  teste de gradiente nulo em `test_fuzzy_operators.py`) e (ii) fluxo de
  crédito por todos os ramos da prova (base da explicabilidade graduada).
- 3.5: mantém; nota de implementação: ordenação topológica iterativa.
- **3.6 (resolve m2)**: redefinir T_g operacionalmente com janela de
  `patience` (como em `training/metrics.py`); discutir robustez a flutuações.
  Rebaixar de "métrica central" para "métrica da análise de generalização".
- **3.7**: corrigir notação do limite (|E| → |E_minimal|); condicionar a
  emergência de esparsidade ao γ calibrado (a curva da Fase 5 mostra que
  γ=1e-4 não produz esparsidade).
- **3.8 (NOVA — núcleo do refoco)**: formalizar a composição de fórmulas de
  prova: ancestor_d(x,z) e as regras com gênero do domínio de Hinton;
  definir fecho da consulta, massa de gradiente em intermediários,
  concentração, e os protocolos deletion/insertion. Enunciar a proposição da
  trivialidade estrutural: sem encadeamento, concentração ≡ 1 (C3).

## 4. Formulação Matemática

- 4.1-4.4: mantêm (grounding, predicados, T-norms, L_total). Corrigir a
  vírgula espúria em L_total (item editorial).
- **4.5**: expandir de "norma do gradiente" para o conjunto completo de
  métricas de explicabilidade formalizado em 3.8.
- 4.6: absorve a redefinição de T_g (3.6).
- 4.7: mantém amortização O(|V|+|E|) — agora com papel extra: é o argumento
  de que a explicação tem custo de um backward. Remover a cerca ```latex
  solta (item editorial).

## 5. Metodologia

- **5.1 (resolve Q4)**: descrever o motor real — autodiff próprio em
  NumPy/Python, DFS iterativa, sem PyTorch/CUDA. Atualizar Tabela 3
  (infraestrutura: CPU). O texto das linhas sobre DFS/hash-set já está
  correto no manuscrito atual; remover só as menções a PyTorch/CUDA/GPU.
- **5.2 Domínios (3 subseções)**:
  a) Adição modular p=13/97 — papel: axiomas algébricos, fidelidade,
     esparsidade, análise de generalização; split 3-vias 30/35/35 (resolve m1);
  b) Parentesco sintético (fecho transitivo) — papel: dedução composta e
     explicabilidade em profundidade >1;
  c) **Hinton (1986)** — dataset real (resolve M2), 8 predicados base + 4
     derivados por regras com gênero; simplificação consanguínea documentada.
- **5.3 Baselines**: MLP puro e LTN com protocolo idêntico (mesmos
  embeddings/otimizador/orçamento — resolve M3-i/ii); preditores planos como
  baseline de explicabilidade; declarar Semantic Loss/DeepProbLog como
  trabalho futuro de comparação (ou implementar antes da submissão).
- **5.4 Protocolo**: AdamW (lr, wd por experimento — tabela); γ=1e-2
  justificado pela curva de trade-off; **fixar N de seeds único em todo o
  artigo** (resolve Q6); semânticas de medição: Reichenbach para fidelidade
  (comparabilidade — resolve m3), argmax para acurácia.
- **5.5 Reprodutibilidade (resolve M3-iii)**: repositório público, um comando
  por experimento, evidências versionadas em `experiments/evidence/`.

## 6. Resultados e Discussão (reordenar pela tese)

- **6.1 Consistência dedutiva** (evidência central): tabelas de
  `kinship_proof_dag` e `hinton_family`; discussão da assimetria
  uncle/aunt vs nephew/niece (propagação do erro do predicado base);
  limitação da saturação da t-conorm (cobertura O(|E|²) de negativos).
- **6.2 Explicabilidade verificável**: massa em intermediários + deleção
  causal (composta vs. plana, dois domínios); resultado de trivialidade na
  adição modular como validação de C3; fidelidade deletion/insertion vs.
  aleatório (margens pequenas no domínio sem encadeamento — reportar com
  honestidade). Substitui o heatmap único (resolve M5).
- **6.3 Fidelidade axiomática e esparsidade**: tabela DLG/MLP/LTN com as
  duas ressalvas (identidade coincide com axiomas de treino; piso ~0.5 da
  implicação de Reichenbach) + curva γ×esparsidade×fidelidade.
- **6.4 Análise de generalização (honesta)**: formulação relacional não
  generaliza (nenhuma config); formulação por classificação: 31-36% com
  axiomas vs 0% sem — evidência a favor de L_semantic, com a ressalva
  explícita de que o DAG/T-norm não participam dessa inferência; sem
  alegação de aceleração de grokking (τ=0.95 nunca atingido). Citar
  Power/Nanda aqui.
- **6.5 Limitações**: escala dos domínios; axiomas manuais; saturação da
  t-conorm; custo do motor NumPy.

## 7. Conclusão

Reescrever espelhando C1-C5. "Generalização acelerada" sai das conclusões;
entra "consistência e explicação como propriedades mensuráveis da execução".

## Figuras (plano — resolve Seções 6-8 do parecer)

| # | Conteúdo | Fonte | Ação |
|---|---|---|---|
| 1 | Grounding (conceitual) | atual | manter, enxugar texto embutido |
| 2 | MLP do predicado | atual | manter, prosa → legenda |
| 3 | Product T-norm | atual | dividir em 2; corrigir rótulo: S-implicação (Reichenbach), não resíduo |
| 4 | Superfície de perda | atual | rotular como esquema conceitual OU substituir por paisagem real |
| 5 | DAG de prova + atribuição (Hinton) | nova | diagrama da consulta uncle(arthur,colin) com massas de gradiente |
| 6 | Curvas de treino/validação | `evidence/*/curves_*.pdf` | vetoriais, inglês, sem títulos "Tabela N" |
| 7 | Barras: dedução composta vs. plana | `hinton_family/derived_report.json` | nova |
| 8 | Curva trade-off γ×esparsidade×fidelidade | `fidelity_sparsity_p13` | substitui o radar |

Todas vetoriais (PDF), rótulos em inglês, prosa nas legendas.

## Checklist editorial final (itens pontuais do parecer)

- [ ] Resumo completo com números
- [ ] Vírgula espúria em L_total
- [ ] Notação do limite em 3.7
- [ ] Cerca ```latex e artefatos "L ∗ data"
- [ ] Rótulo da Fig. 3 (S-implicação)
- [ ] N de execuções único em texto e figuras
- [ ] Títulos internos "Tabela 1/2/3" removidos das figuras
- [ ] Bibliografia nova: Petersen 2022, Petersen 2024, Evans & Grefenstette
      2018, Power 2022, Nanda 2023 (+ Minervini se citar extensões do NTP;
      Li et al. 2018 se refizer a Fig. 4)
