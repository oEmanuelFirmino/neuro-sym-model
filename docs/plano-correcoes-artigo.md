# Plano de Implementação — Correções do Artigo DLG

> Baseado no parecer do orientador (`correcoes_do_artigo.pdf`) e no estado real do
> repositório `neuro-sym-model` em 2026-07-18. Decisões já tomadas com o autor:
>
> 1. **O motor de execução permanece o autodiff próprio em NumPy/Python** (não há
>    migração para PyTorch/CUDA). O texto do artigo é que será corrigido para
>    refletir isso — não o código.
> 2. **Nenhum dos experimentos do artigo (Tabelas 4/5, Figuras 6-8, adição modular)
>    existe ainda.** Precisam ser construídos e rodados de verdade neste framework
>    antes de qualquer número entrar no texto final.

Isso muda a natureza do trabalho: não é só "revisar o texto", é **construir a
infraestrutura experimental que falta e só então escrever os resultados**. O plano
abaixo separa o que é edição pura de texto (rápido, sem dependências) do que exige
engenharia nova no framework (a maior parte do esforço).

---

## Visão geral do gap atual

O repositório hoje só tem dois exemplos de brinquedo (`examples/socrates`,
`examples/fraud_detection`) e a infraestrutura mínima de um framework neuro-simbólico:
`Tensor` com autodiff reverso, backends Python/NumPy, `Interpreter` com Product
T-norm, `Trainer` com SGD puro, e um `explainer.py` que só faz top-k por gradiente.

Faltam, comparado ao que o artigo descreve:

| Área | Existe hoje | Precisa existir |
|---|---|---|
| Regularização L1 estrutural | ❌ não implementada no `Trainer`/loss | ✅ termo `γ‖W‖₁` no funcional |
| Peso `λ` da perda semântica | ❌ satisfação de regra entra com peso 1 fixo | ✅ hiperparâmetro `λ` configurável |
| Otimizador AdamW | ❌ só existe `SGD` | ✅ AdamW com weight decay |
| Domínio de Adição Modular | ❌ não existe | ✅ dataset, grounding, axiomas |
| Split treino/val/teste | ⚠️ loader suporta `test_facts_file`, mas exemplos atuais não usam holdout real | ✅ split 3-vias genuíno |
| Métrica T_g (tempo p/ generalização) | ❌ não implementada | ✅ com robustez a flutuações |
| Baselines (Semantic Loss, LTN, DeepProbLog, MLP puro) | ⚠️ só Product T-norm existe; Łukasiewicz/Gödel não implementados | ✅ 4 baselines comparáveis |
| Runner multi-seed (10 execuções) | ❌ não existe | ✅ harness com seeds fixas + agregação |
| Sparsity comensurável entre arquiteturas | ❌ | ✅ métrica normalizada única |
| Explicabilidade quantitativa (fidelidade, deletion/insertion) | ⚠️ só heatmap anedótico | ✅ métricas sobre N consultas |
| Segundo domínio experimental | ❌ | ✅ tarefa relacional ou variante ruidosa |
| Geração de figuras (vetorial, a partir de dados reais) | ❌ nenhum script de plot no repo | ✅ scripts reprodutíveis |

---

## Fase 0 — Decisões já resolvidas (não repetir)

- Motor: NumPy próprio, não PyTorch. ✅
- Resultados: precisam ser gerados do zero. ✅

Decisão pendente que **você** ainda precisa tomar antes da Fase 4:

- **Segundo domínio (M2):** recomendo uma tarefa relacional de **fecho transitivo /
  parentesco** (kinship) sobre um grafo pequeno. Motivo: reaproveita 100% do motor
  FOL/DAG já existente (nenhuma mudança estrutural), e reforça diretamente a
  alegação de explicabilidade via "fecho transitivo" que hoje é só anedótica (M5).
  Alternativa mais barata: adição modular com **ruído no rótulo** (mesmo pipeline,
  só perturbar uma fração dos exemplos) — mais rápido, mas mais fraco como evidência
  de generalidade porque ainda é o mesmo domínio algébrico. Posso detalhar as duas
  quando você decidir.

---

## Fase 1 — Correções de texto sem dependência de código

Não bloqueiam nem são bloqueadas pela Fase 2+. Podem ser feitas em paralelo e
entregues primeiro para mostrar progresso rápido ao orientador.

1. **Resumo truncado** — reescrever completo, mas só pode receber os números finais
   (fidelidade, esparsidade, T_g) depois da Fase 3/4. Por ora, redigir a estrutura
   com placeholders claros.
2. **M1(iii) — reformular a contribuição** como *"combinação integrada de execução
   dinâmica via DAG + Product T-norm + regularização L1"*, deixando explícito que
   nenhuma peça isolada é nova.
3. **Citações faltantes** — adicionar ao texto e à bibliografia:
   - Petersen et al., *Deep Differentiable Logic Gate Networks* (NeurIPS 2022) + versão
     convolucional (NeurIPS 2024) — citar na Seção 2 e diferenciar explicitamente do DLG.
   - Evans & Grefenstette, *∂ILP* (2018) — Seção 2.
   - Power et al. (2022) e Nanda et al. (2023, *Progress measures for grokking*) —
     Seção 3.6, ancorando o uso do arcabouço de grokking.
   - Li et al. (2018, filter normalization) — só necessária se a Fig. 4 for
     recalculada com paisagem de perda real (Fase 7).
4. **M4 — qualificar a alegação sobre Product T-norm** (Seções 3.4 e 4.3): o texto
   hoje afirma preservação de gradiente em cadeias profundas sem ressalva, o que
   contradiz a própria referência [11] (van Krieken et al.) já citada no artigo.
   Reescrever para: Product T-norm ainda sofre vanishing gradient em conjunções
   longas (∂(ab)/∂a = b), e é a média aritmética em `L_semantic` — não o produto —
   que mitiga isso. Isso é 100% consistente com o código (`fuzzy_operators.py` usa
   produto puro; `interpreter.py` usa `mean()`/`min()`/`p_mean()` para agregação).
5. **Q4 — corrigir Seção 5.1/Tabela 3**: remover toda menção a PyTorch/CUDA, GPU
   RTX 5070 etc. Descrever fielmente o motor implementado: DFS com hash-set de nós
   visitados para topological sort, chain rule aplicada em ordem reversa,
   `backend/numpy_backend.py` para as operações vetorizadas. Esse texto já existe
   quase pronto na Seção 5.1.1 do artigo — só precisa trocar a menção a PyTorch/CUDA
   pela descrição real do motor (que aliás já está descrita corretamente logo depois,
   nas linhas 638–675 do texto extraído — o artigo se contradiz internamente).
6. **Correções tipográficas pontuais**: vírgula espúria em `L_total`, seta faltante
   em `lim |E| → |E_minimal|`, remover a cerca ` ```latex ` solta antes da Seção 4.7.
7. **m1 — terminologia do split**: depende da Fase 2 (ver abaixo). Se implementarmos
   holdout de teste real, renomear para treino/validação/teste. Se não, ao menos
   renomear "validação" para "holdout" e deixar explícito que não há seleção de
   hiperparâmetros nesse conjunto.

---

## Fase 2 — Fundação experimental (bloqueia quase tudo depois) ✅ concluída (2026-07-18)

Trabalho de engenharia no `src/neurosym`. Sem isso, nenhum número em Tabelas 4/5 é
confiável.

1. ✅ **Regularização L1 estrutural** — `training/trainer.py` agora computa
   `loss = L_data + λ·L_semantic + γ‖W‖₁`, com `lambda_semantic`/`gamma_l1`
   configuráveis no construtor do `Trainer` (`gamma_l1=0.0` por padrão). O termo
   `‖W‖₁` soma apenas as matrizes de peso das camadas `Linear` dos predicados
   (`Module.l1_weight_parameters()` / `Linear.l1_weight_parameters()`, novo),
   excluindo bias — consistente com a Seção 4.4 do artigo. `l1_penalty` é sempre
   registrado nos logs por época, mesmo com `gamma_l1=0`, para permitir monitorar
   a esparsidade nos cenários de ablação que não usam a penalidade.
   Dependência nova: `Tensor.abs()` com gradiente (`sign(x)`, aproximado de forma
   branch-free para funcionar nos dois backends).
2. ✅ **AdamW** — `training/optimizer.py:AdamW`, com momentos de 1ª/2ª ordem,
   correção de viés e weight decay desacoplado, testado com um passo calculado à
   mão (`test/test_optimizer.py`).
3. ⚠️ **Split treino/validação/teste** — mecanismo pronto: `Trainer.fit()` aceita
   `val_facts` opcional e calcula `val_accuracy` por época sem tocar em gradientes
   (`Trainer.evaluate_accuracy`), e `test_facts` é avaliado só uma vez ao final via
   o runner multi-seed. O que falta é a Fase 3 usar isso de fato com um split real
   de 3 vias no domínio de Adição Modular (aqui só a infraestrutura existe).
4. ⏸ **Inicialização Kaiming/Xavier** — não mexido nesta fase; `module/factory.py`
   segue com a inicialização existente (`sqrt(2/in_features)` em `Linear`). Avaliar
   na Fase 3 se precisa de inicializadores dedicados por camada.
5. ✅ **Métrica T_g** — `training/metrics.py:time_to_generalization`, com parâmetro
   `patience` (janela sustentada) em vez de "cruzamento único ∀t'≥t", endereçando
   diretamente o item m2. Also `post_threshold_dip_count` como diagnóstico de
   robustez da transição. Testado com um caso de flutuação pós-transição
   deliberada (`test/test_metrics.py`).
6. ✅ **Runner multi-seed** — `experiments/run_multiseed.py`, agnóstico de domínio
   (recebe um `build_fn(seed) -> ExperimentSpec`), agrega T_g / acurácia final /
   `l1_penalty` final em média±desvio padrão e serializa tudo (agregados + curvas
   por época de cada seed) em JSON. Testado com um experimento sintético (não o
   exemplo Socrates, que ainda não tem `config.yaml` — gap pré-existente, fora do
   escopo desta fase) e com verificação de determinismo (mesma seed → mesmo
   resultado).

   **Gap conhecido para a Fase 3:** `KnowledgeBaseLoader.load_domain`
   (`data_manager/loader.py`) gera os embeddings iniciais via `hash(nome + i)`,
   que não é fixado por `random.seed()`/`np.random.seed()` (hash de string em
   Python não é determinístico entre processos por padrão). Isso quebra a
   reprodutibilidade por seed do runner multi-seed *se* o domínio de Adição
   Modular usar esse loader para gerar embeddings. Precisa trocar por
   inicialização baseada em `random`/`np.random` antes da Fase 3 usar seeds de
   verdade.

---

## Fase 3 — Domínio de Adição Modular + Baselines (M2 parcial, M3)

1. **Dataset**: gerar as 9409 triplas `(a,b,c=(a+b) mod 97)`, split 30/… conforme
   protocolo — mas revisar já a decisão dos 30% treino / 70% "validação" à luz do
   item m1 (Fase 1.7).
2. **Axiomas** como `Formula` (`Forall` + `Implies`/`And` já existentes em
   `logic/logic.py`): comutatividade `Add(a,b,c) → Add(b,a,c)` e identidade
   `Add(a,0,a)`.
3. **Baselines**, cada um com protocolo de tuning documentado e curvas de
   convergência salvas (exigência explícita de M3):
   - **MLP puro** (sem lógica) — o valor atual do artigo (14,2%) é seu próprio
     placeholder, não um resultado real; ao rodar de verdade, orçar épocas e weight
     decay suficientes para não ficar com um baseline artificialmente fraco — isso
     é literalmente o que o parecer aponta como suspeito.
   - **Semantic Loss** (penalização via WMC) — versão simplificada: usar a
     satisfação booleana agregada como penalidade estática (sem DAG dinâmico, sem
     L1), para isolar o que é "DAG + L1" de "só ter uma perda simbólica".
   - **LTN** (Łukasiewicz + p-mean) — `fuzzy_operators.py` só tem Product T-norm
     hoje; precisa adicionar `lukasiewicz_tnorm`/`_tconorm`/`_implication`
     (`max(0, a+b-1)` etc.) ao `OPERATOR_MAP`. O agregador `p_mean` já existe em
     `interpreter.py`.
   - **DeepProbLog** (nAD + WMC) — o mais custoso de reproduzir fielmente; avaliar
     escopo reduzido (aproximação com compilação probabilística simplificada) e
     documentar claramente as simplificações assumidas, já que o parecer só exige
     "protocolo descrito + reprodutibilidade", não paridade total com a
     implementação original.
4. **Liberar código e dados** (M3-iii) — garantir que o repo público reproduz os
   números do artigo com um comando (`README` com instruções, configs versionadas).

---

## Fase 4 — Segundo domínio (M2)

Depende da decisão pendente da Fase 0. Reaproveita toda a Fase 2 (T_g, L1, split,
runner multi-seed) — só troca o domínio/dataset e os axiomas.

## Fase 5 — Sparsity comensurável entre arquiteturas (m3)

Definir uma métrica única aplicável aos 4 mecanismos inferenciais (WMC, agregadores
fuzzy, nAD, DAG+L1) — por exemplo "fração de parâmetros/arestas com contribuição
efetiva acima de um limiar ε, normalizada pelo total de parâmetros/arestas
possíveis do mesmo predicado". Isso precisa de instrumentação por baseline (cada um
reporta sua contagem "ativa" na mesma unidade) **e** de um parágrafo metodológico
explícito no artigo justificando a comparação — sem isso o parecer vai repetir a
objeção.

## Fase 6 — Explicabilidade quantitativa (M5)

1. **Métrica de fidelidade/concentração**: para um conjunto de N consultas
   (idealmente amostradas do(s) domínio(s) da Fase 3/4), medir a fração da massa do
   gradiente que cai sobre entidades do fecho transitivo da consulta vs. fora dele.
   Generalizar `explainability/explainer.py` (hoje só devolve top-k por consulta
   única) para rodar em lote e agregar estatísticas.
2. **Protocolo deletion/insertion**: zerar progressivamente os embeddings mais
   influentes (por gradiente) e medir a degradação do grau de verdade previsto
   (curva + AUC); fazer o inverso (inserção) para completude.
3. **Comparação com XAI post-hoc**: como o motor é próprio (sem Captum/SHAP
   prontos), implementar uma baseline leve de perturbação (leave-one-out) ou
   gradientes integrados usando o autodiff já existente, só para servir de
   contraponto ao método por gradiente puro.

## Fase 7 — Figuras (regeneradas a partir de dados reais)

Só depois que Fases 2–6 produzirem números de verdade. Novo diretório
`experiments/plots/` com scripts reprodutíveis, saída vetorial (PDF/SVG).

- **Fig. 3**: separar painel algébrico do comparativo de fluxo de gradiente; corrigir
  rótulo da implicação `1-a+ab` para **S-implicação (Reichenbach)** — não é o
  resíduo/Goguen. Isso já bate com o operador implementado em
  `fuzzy_operators.py:15-16` (`product_implication`), então é só correção de rótulo,
  não de código.
- **Fig. 4**: rotular como esquema conceitual explícito, ou (preferível, agora que
  haverá modelo treinado de verdade) substituir por paisagem de perda real via
  *filter normalization* (Li et al., 2018) calculada a partir dos pesos treinados.
- **Fig. 6/7/8**: remover títulos internos "Tabela 1/2/3"; gerar com o **mesmo N**
  de execuções em todo lugar (Fase 2.6); substituir o radar da Fig. 8 por barras
  agrupadas ou tabela.
- Verificar se o periódico-alvo é nacional ou internacional antes de decidir se as
  figuras precisam ser refeitas em inglês.

## Fase 8 — Consolidação final

- Reescrever o Resumo com os números reais finais.
- Conferir consistência de N execuções em todo o texto/figuras.
- Passagem final de revisão contra cada item do parecer (checklist abaixo).
- Só então considerar o manuscrito pronto para submissão.

---

## Ordem recomendada de execução

```
Fase 1 (texto, paralelo)  ──┐
                             ├──> Fase 8 (consolidação final)
Fase 2 (fundação) ──> Fase 3 (adição modular + baselines) ──┐
                  └─> Fase 4 (2º domínio) ────────────────────┼──> Fase 7 (figuras)
                  └─> Fase 5 (sparsity comensurável) ─────────┤
                  └─> Fase 6 (explicabilidade quantitativa) ──┘
```

A Fase 2 é o gargalo: nada em Tabelas 4/5, T_g ou figuras é confiável antes dela
estar pronta.

## Checklist de rastreamento (mapeado ao parecer)

- [ ] M1 — novidade reposicionada + citações Petersen/∂ILP
- [ ] M2 — segundo domínio
- [ ] M3 — baselines justos + protocolo + código/dados públicos
- [ ] M4 — afirmação sobre Product T-norm qualificada
- [ ] M5 — explicabilidade quantitativa (N consultas, deletion/insertion, XAI post-hoc)
- [ ] m1 — terminologia do split
- [ ] m2 — robustez de T_g a flutuações
- [ ] m3 — sparsity comensurável entre arquiteturas
- [ ] m4 — citações Power et al. / Nanda et al.
- [ ] Resumo reescrito completo
- [ ] Fig. 3 rótulo da implicação corrigido
- [ ] Erro tipográfico em L_total
- [ ] Notação do limite em 3.7
- [ ] Artefatos de compilação (```latex solta etc.)
- [ ] Consistência do número de execuções (Q6)
- [ ] Q4 — Seção 5.1/Tabela 3 corrigida (motor real, sem PyTorch/CUDA)
- [ ] Fig. 4 rotulada/recalculada
- [ ] Fig. 8 substituída por barras/tabela
- [ ] Títulos internos "Tabela 1/2/3" removidos das Figs. 6/7/8
- [ ] Idioma das figuras (se aplicável)
