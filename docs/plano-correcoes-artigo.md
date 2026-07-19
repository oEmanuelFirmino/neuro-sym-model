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

## ⚑ Redirecionamento estratégico (decisão do autor, 2026-07-18)

Após os experimentos da Fase 3 (nenhuma configuração da formulação relacional
generaliza; a formulação softmax generaliza graças ao termo semântico, mas sem
usar o DAG/T-norm na inferência), o autor decidiu **amenizar as alegações de
grokking/generalização acelerada e reposicionar o artigo em explicabilidade +
consistência lógica** — a saída que o próprio parecer oferece em M2 ("Ou
podemos amenizar as alegações"). Consequências:

1. **Tese revisada do artigo**: o DLG é um framework neuro-simbólico
   diferenciável cuja arquitetura (DAG dinâmico + Product T-norm + L1) entrega
   (i) *consistência lógica mensurável* (fidelidade axiomática alta em dados
   não vistos) e (ii) *explicabilidade intrínseca mensurável* (atribuição por
   gradiente concentrada no fecho da consulta, validada por protocolos de
   deletion/insertion). A melhora de generalização induzida por restrições
   semânticas vira um achado secundário, reportado com honestidade (evidência:
   experimento softmax, 31-36% vs 0%), sem alegar aceleração de grokking — a
   transição completa não foi observada.
2. **Título/resumo precisarão mudar**: "Generalização Acelerada" sai do título
   ou é substituído por algo defensável (ex.: "Consistência Lógica e
   Explicabilidade Intrínseca").
3. **Prioridade experimental invertida**: as Fases 5 (esparsidade) e 6
   (explicabilidade) passam à frente da caça ao grokking em p=97. Em
   particular, a explicabilidade só é evidência *da arquitetura* num domínio
   com **encadeamento de regras na inferência** (o fecho transitivo maior que
   as constantes literais da consulta) — o que torna o segundo domínio
   (Fase 4, parentesco/ancestral com consultas compostas via DAG de prova) o
   experimento central do artigo reposicionado, não um complemento.
4. **T_g e as curvas de grokking** permanecem no artigo como análise honesta
   (incluindo o resultado negativo da formulação relacional e o positivo do
   termo semântico na formulação softmax), mas saem do posto de contribuição
   principal.

---

## Visão geral do gap atual

O repositório hoje só tem dois exemplos de brinquedo (`examples/socrates`,
`examples/fraud_detection`) e a infraestrutura mínima de um framework neuro-simbólico:
`Tensor` com autodiff reverso, backends Python/NumPy, `Interpreter` com Product
T-norm, `Trainer` com SGD puro, e um `explainer.py` que só faz top-k por gradiente.

Estado no início do plano (2026-07-18), comparado ao que o artigo descreve. Esta
tabela é o snapshot original — o estado atual (atualizado a cada fase) fica nas
seções de cada fase abaixo; a coluna "Precisa existir" permanece como referência
do alvo.

| Área | Existia no início | Precisa existir |
|---|---|---|
| Regularização L1 estrutural | ❌ não implementada no `Trainer`/loss | ✅ termo `γ‖W‖₁` no funcional — **feito (Fase 2)** |
| Peso `λ` da perda semântica | ❌ satisfação de regra entra com peso 1 fixo | ✅ hiperparâmetro `λ` configurável — **feito (Fase 2)** |
| Otimizador AdamW | ❌ só existe `SGD` | ✅ AdamW com weight decay — **feito (Fase 2)** |
| Domínio de Adição Modular | ❌ não existe | ✅ dataset, grounding, axiomas — **feito (Fase 3)** |
| Split treino/val/teste | ⚠️ loader suporta `test_facts_file`, mas exemplos atuais não usam holdout real | ✅ split 3-vias genuíno — **feito (Fase 3)** |
| Métrica T_g (tempo p/ generalização) | ❌ não implementada | ✅ com robustez a flutuações — **feito (Fase 2)** |
| Baselines (Semantic Loss, LTN, DeepProbLog, MLP puro) | ⚠️ só Product T-norm existe; Łukasiewicz/Gödel não implementados | ✅ 4 baselines comparáveis — **3/4 feitos (Fase 3): DLG, MLP, LTN; Semantic Loss e DeepProbLog adiados p/ Fase 3b** |
| Runner multi-seed (10 execuções) | ❌ não existe | ✅ harness com seeds fixas + agregação — **feito (Fase 2)** |
| Sparsity comensurável entre arquiteturas | ❌ | ✅ métrica normalizada única — pendente (Fase 5) |
| Explicabilidade quantitativa (fidelidade, deletion/insertion) | ⚠️ só heatmap anedótico | ✅ métricas sobre N consultas — pendente (Fase 6) |
| Segundo domínio experimental | ❌ | ✅ tarefa relacional ou variante ruidosa — pendente (Fase 4) |
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

## Fase 3 — Domínio de Adição Modular + Baselines (M2 parcial, M3) ⚠️ parcialmente concluída (2026-07-18)

1. ✅ **Dataset + split** — `experiments/modular_addition/dataset.py`. Gera as p²
   triplas `(a,b,(a+b) mod p)` e particiona em **treino/validação/teste
   genuinamente separados** (resolve o item m1: o artigo original usa 30% treino /
   70% "validação" com seleção de modelo no mesmo conjunto, sem holdout real). A
   fração de treino pequena (30% por padrão) é preservada — é o que induz o efeito
   de grokking — mas o restante (70%) agora é dividido em validação (acompanha a
   curva/T_g durante o treino) e teste (nunca visto até a avaliação final), metade
   cada por padrão. Cada par `(a,b)` gera 1 fato positivo (`c` correto) + N fatos
   negativos (`c` incorreto amostrado) para que a acurácia não seja "gamed" por um
   preditor trivial que sempre responde verdadeiro. Embeddings gerados via
   `random.uniform` seedado (`build_grounding_env`), não o `hash()` não-determinístico
   do `KnowledgeBaseLoader` (fecha o gap sinalizado ao final da Fase 2).
2. ✅ **Axiomas** — `experiments/modular_addition/axioms.py`: comutatividade
   `Add(a,b,c) → Add(b,a,c)` e identidade `Add(a,0,a)`. Implementados como
   instâncias *grounded* por par de treino, não via `Forall` sobre todo o
   domínio — com p=97 um `Forall` de duas variáveis livres exigiria iterar 97²
   combinações por avaliação, e o `Interpreter.eval_formula` atual não vetoriza
   quantificadores (custo O(|domínio|) por variável, sequencial em Python).
   Instanciar por exemplo de treino é a leitura mais direta do funcional
   L_semantic do artigo (Seção 4.4) e o único jeito computacionalmente tratável
   dado o interpretador atual.
3. ✅ **Métrica de avaliação do domínio** — `experiments/modular_addition/evaluation.py`:
   acurácia binária genérica (limiar 0.5) não bastava aqui, porque um preditor
   trivial "sempre >= 0.5" acertaria todo fato positivo sem aprender a função. A
   métrica correta para uma relação `Add(a,b,c)` é: dado `(a,b)`, o candidato `c`
   com maior grau de verdade dentre todos os `p` candidatos é o `c` correto?
   `Trainer` ganhou um parâmetro `accuracy_fn` (Fase 3) para plugar essa métrica
   sem forkar a classe, e `val_eval_every` para não recalculá-la (cara: O(p) por
   consulta) a cada época em domínios maiores.
4. ✅ **3 de 4 baselines** — `experiments/modular_addition/run.py`, todos
   compartilhando embeddings/MLP/split/época, só o mecanismo de inferência muda
   (Tabela 1 do artigo):
   - **DLG** (`make_dlg_build_fn`): Product T-norm (padrão do `Interpreter`) +
     axiomas + regularização L1.
   - **MLP puro** (`make_mlp_baseline_build_fn`): sem axiomas, sem L_semantic, sem
     L1 — só `L_data`. Resolve a suspeita de M3 sobre o baseline de 14,2% do
     artigo (que era um placeholder, não um resultado real): aqui ele roda com o
     mesmo otimizador/orçamento de época que o DLG, então uma diferença de
     desempenho reflete o mecanismo, não um baseline subtreinado.
   - **LTN** (`make_ltn_baseline_build_fn`): troca Product por operadores de
     Łukasiewicz (`lukasiewicz_and`/`_or`/`_implies`, novos em
     `fuzzy_operators.py`), sem L1. **Simplificação registrada**: a agregação
     p-mean da LTN original não é exercida, porque os axiomas são instâncias
     grounded (não `Forall`) e `Trainer.fit` agrega `L_semantic` como média
     aritmética simples — o diferencial testado aqui é o operador T-norm
     (Product vs. Łukasiewicz), que é o ponto central da discussão de M4, não a
     forma de agregação.
5. ⏸ **Semantic Loss e DeepProbLog — adiados nesta passada.** Semantic Loss
   (penalização via WMC) e DeepProbLog (nAD + compilação probabilística) não têm
   um mapeamento direto para o `Interpreter` atual (que só sabe avaliar T-normas
   fuzzy compostas via DAG dinâmico) sem trabalho adicional de design — diferente
   de Łukasiewicz, que é só mais um `FuzzyOperator`. Ficam para uma Fase 3b
   dedicada, com as simplificações explicitamente documentadas quando chegar lá
   (o parecer exige "protocolo descrito + reprodutibilidade", não paridade total
   com as implementações originais).
6. ✅ **Piloto em pequena escala, executado** — `experiments/modular_addition/run_pilot.py`,
   `p=13`, embeddings=8, hidden=24, 200 épocas, 2 seeds, os 3 baselines disponíveis
   (DLG/MLP/LTN). Resultados brutos em `experiments/modular_addition/pilot_results/`.

   | Arquitetura | val_accuracy final | test_accuracy | T_g (τ=0.95) | tempo (2 seeds) |
   |---|---|---|---|---|
   | DLG | 5.1% ± 0.0 | 5.0% ± 1.7 | nunca atingido | 255s |
   | MLP puro | 7.6% ± 2.5 | 5.8% ± 4.2 | nunca atingido | 111s |
   | LTN | 4.2% ± 0.8 | 5.0% ± 1.7 | nunca atingido | 236s |

   **Leitura honesta**: nenhuma arquitetura generalizou dentro de 200 épocas —
   acurácia ficou perto do acaso (1/13 ≈ 7.7%), então **isto não é evidência de
   que o DLG generaliza mais rápido**, nem o contrário. O que o piloto confirma é
   que o *pipeline* está correto: a curva de treino do DLG mostra exatamente o
   comportamento esperado de um platô de memorização pré-grokking —
   `loss` caindo de forma monótona (0.58→0.14) e a satisfação dos axiomas subindo
   para 96%, enquanto `val_accuracy` fica estagnada e ruidosa (~5-15%) o tempo
   todo — e `T_g` corretamente retorna `None` em vez de inventar um valor, porque
   o limiar de 95% nunca foi atingido de forma sustentada. Também ficou visível
   que `l1_penalty` do DLG *cresceu* ao longo do treino (263→304) apesar de
   `gamma_l1=1e-4` — nesta escala/hiperparâmetros o termo L1 é fraco demais
   perto de `L_data`; o `gamma_l1` precisa ser recalibrado quando formos atrás
   de esparsidade de verdade.

   200 épocas é pouco para observar a transição de grokking em si (a literatura
   -- Power et al. 2022 -- tipicamente usa 10³-10⁵ passos); então o piloto
   deliberadamente não tentou responder "o DLG groka mais rápido", só "o
   encanamento funciona e mede o que diz medir". Essa pergunta científica fica
   para uma execução bem mais longa (ver nota de desempenho e decisão pendente
   abaixo).

   **Rodada longa (DLG vs. MLP puro, 3000 épocas, 2 seeds, mesmo p=13,
   `run_pilot_long.py`), executada em 2026-07-18:**

   | Arquitetura | val_accuracy final | test_accuracy | T_g (τ=0.95) | tempo (2 seeds) |
   |---|---|---|---|---|
   | DLG | 5.1% ± 1.7 | 6.7% ± 1.7 | nunca atingido | 4994s (~83 min) |
   | MLP puro | 8.5% ± 1.7 | 5.0% ± 5.0 | nunca atingido | 2335s (~39 min) |

   **Resultado honesto e negativo**: mesmo com 15x mais épocas, nenhuma das
   duas arquiteturas generalizou -- acurácia continua no nível do acaso
   (1/13 ≈ 7,7%). A curva do DLG (`pilot_results/dlg_long.json`) mostra que o
   treino **memoriza quase perfeitamente por volta da época 300**
   (`l_data` cai para ~0,02, satisfação dos axiomas para 99,4%) e depois fica
   mais **2700 épocas** nesse platô sem qualquer tendência de subida em
   `val_accuracy` (oscila entre 5-15% o tempo todo, sem tendência). Ou seja:
   isto não é "ainda não convergiu", é um platô de memorização estável que não
   se rompeu mesmo em ~10x o tempo que levou para memorizar -- diferente do
   padrão relatado na literatura de grokking, onde a transição eventualmente
   ocorre.

   Hipóteses mais prováveis para investigar antes de escalar para p=97 (nenhuma
   testada ainda):
   - **weight decay fraco demais** relativo à velocidade de memorização (o
     padrão do AdamW aqui é `1e-2`; a literatura de grokking trata isso como o
     hiperparâmetro mais sensível para induzir a transição, e por vezes exige
     ordens de magnitude mais passos pós-memorização do que os 10x testados
     aqui).
   - **p=13 pode ser pequeno demais** para o efeito aparecer de forma limpa
     (grande parte da literatura usa p≥97 especificamente).
   - **capacidade do modelo** (embedding_dim=8, hidden=24) pode ser
     insuficiente para representar a solução generalizante, mesmo que baste
     para memorizar.
   - **a formulação relacional em si** (avaliar a trinca `Add(a,b,c)` via
     embeddings concatenados, em vez de prever `c` por classificação softmax
     entre p classes, como é comum na literatura de grokking) pode ser
     intrinsecamente mais difícil de grokar -- essa é uma pergunta metodológica
     em aberto sobre a própria formulação do artigo, não só um problema de
     hiperparâmetro, e vale investigar antes de comprometer o tempo de uma
     rodada em p=97.

   **Recomendação**: não escalar direto para p=97 sem antes fazer uma busca de
   hiperparâmetros (weight decay, lr, capacidade) em pequena escala e
   confirmar que a transição de grokking é alcançável neste framework para
   *algum* ponto do espaço de configuração. Caso contrário corre-se o risco de
   gastar um job de várias horas em p=97 só para reproduzir o mesmo platô.

   **Formulação softmax executada (2026-07-18, `run_softmax.py`, evidências em
   `experiments/evidence/softmax_*`)** — p=13, 3000 épocas, 2 seeds, wd=1.0
   (canônico da literatura de grokking), classificação entre p classes com
   cross-entropy; axiomas traduzidos para o cenário distribucional
   (`softmax_formulation.py`):

   | Variante | val final | teste | Comportamento |
   |---|---|---|---|
   | softmax_dlg (com axiomas) | **31.4% ± 4.2** | **33.3% ± 5.0** | subida contínua desde ~época 100; seed 1 ainda subindo em 3000 |
   | softmax_mlp (sem axiomas) | 0.0% ± 0.0 | 0.8% ± 0.8 | memoriza (l_data→0.005) e colapsa a validação a 0% |

   **Primeiro resultado positivo genuíno do projeto**, com três leituras:
   1. **A hipótese da formulação estava certa**: a formulação relacional
      (triplas + MSE) não mostrou nenhum sinal de aprendizado generalizante em
      nenhuma configuração; a formulação por classificação mostra subida clara
      e monotônica — 4x acima do acaso e ainda subindo no fim do orçamento.
   2. **Primeira evidência real a favor da alegação central do artigo**: a
      diferença entre 31-36% e 0% é exatamente o termo L_semantic (axiomas de
      comutatividade/identidade). O baseline puro memoriza e generaliza *pior
      que o acaso*; com os axiomas, o modelo aprende estrutura transferível.
   3. **Ressalvas para o texto**: (i) ainda não é a transição de grokking — o
      limiar τ=0.95 não foi atingido e T_g segue indefinido; (ii) nesta
      formulação o "DLG" se reduz ao termo semântico distribucional — o DAG
      dinâmico/Product T-norm do artigo não participam (a inferência é um
      classificador puro), então o resultado sustenta L_semantic, não a
      arquitetura completa; o artigo precisa decidir como conciliar isso.

   **Próximos passos recomendados**: (a) rodada mais longa (10k+ épocas) e/ou
   leve ajuste de lr — a curva da seed 1 ainda subia em 3000; (b) para p=97,
   o custo por época da formulação softmax com axiomas (~0.5s em p=13, ~56x
   mais pares em p=97) exige antes as otimizações de desempenho já listadas
   (AdamW vetorizado, mini-batching).

   **Sweep de hiperparâmetros executado (2026-07-18, `run_sweep.py`,
   resultados em `sweep_results/`)** — DLG apenas, p=13, 1500 épocas, seed 0,
   5 configurações cobrindo as hipóteses 1-3:

   | Config | val final | val pico | teste | T_g |
   |---|---|---|---|---|
   | wd=0.05 (8/24) | 5.1% | 15.3% | 8.3% | nunca |
   | wd=0.1 (8/24) | 5.1% | 15.3% | 6.7% | nunca |
   | wd=0.3 (8/24) | 6.8% | 15.3% | 5.0% | nunca |
   | wd=0.1, capacidade 16/64 | 8.5% | 15.3% | 11.7% | nunca |
   | wd=0.1, 3 negativos/positivo | 13.6% | 15.3% | 3.3% | nunca |

   **Nenhuma configuração rompeu o platô.** Diagnóstico pelas curvas:
   - O weight decay *está* agindo (a `l1_penalty` da config 16/64 despenca de
     1084→234 ao longo do treino — os pesos encolhem bastante), e ainda assim
     a memorização persiste com validação no nível do acaso. Ou seja, a
     hipótese "wd fraco demais" foi testada de verdade e **refutada** neste
     domínio/formulação (até 30x o valor original).
   - Capacidade maior e mais negativos também não mudam o quadro (com 3
     negativos o modelo nem memoriza completamente: `l_data` estaciona em
     ~0.195, sugerindo subajuste dentro do orçamento).
   - O "val pico" idêntico (15.3% = 9/59 acertos) em todas as configs é
     granularidade da métrica com 59 consultas de validação, não um sinal.

   **Conclusão**: hiperparâmetros, capacidade e densidade de negativos não
   destravam grokking nesta formulação. A hipótese remanescente — e agora a
   principal suspeita — é a **hipótese 4: a própria formulação relacional**.
   No setup clássico de grokking (Power et al.), o modelo prevê `c` por
   classificação softmax entre p classes com cross-entropy, o que força
   estrutura compartilhada entre todos os candidatos a cada exemplo. Aqui, o
   predicado avalia cada tripla `Add(a,b,c)` isoladamente com MSE, e o treino
   só mostra 2-4 dos 13 candidatos por par — sinal muito mais esparso, e a
   literatura já documenta que trocar cross-entropy por MSE altera (e pode
   suprimir) a dinâmica de grokking. Isso tem implicação metodológica direta
   para o artigo: se a formulação relacional do DLG não groka nem em p=13,
   as alegações do artigo sobre aceleração de grokking precisam ser
   re-avaliadas ou a formulação experimental precisa mudar (por exemplo,
   negativos completos p-1 por par, aproximando o setup de classificação, ou
   uma cabeça softmax explícita). **Próxima decisão a tomar com o autor antes
   de gastar mais horas de computação.**
7. ⏸ **Pendente para fechar M3 por completo**: rodar a escala cheia (p=97, 10
   seeds, todas as 4 arquiteturas) — job bem mais longo, ver nota de desempenho
   abaixo — e documentar o protocolo de tuning de cada baseline + curvas de
   convergência salvas (exigência explícita de M3). Liberar código e dados
   (M3-iii) fica natural uma vez que a Fase 3b estiver completa.

**Nota de desempenho para a Fase 3b:** `Trainer.fit` avalia a lista `rules`/`facts`
inteira a cada época (sem mini-batching) e `AdamW.step` itera parâmetro-por-parâmetro
em Python puro (sem vetorização). Em `p=13` (embeddings=8, hidden=24), ~1s/época;
a escala do artigo (p=97, split 30/35/35 → milhares de fatos + eixo de avaliação
O(p) por consulta) provavelmente exige horas por seed nesta implementação. Antes
de rodar a escala cheia, vale considerar: (a) mini-batching real no `Trainer`, (b)
vetorizar `AdamW.step` via o backend NumPy em vez de Python puro, e (c) reduzir a
frequência/tamanho da avaliação O(p) em domínios grandes.

---

## Fase 4 — Segundo domínio (M2) ✅ implementado e com resultados (2026-07-18)

Promovido a **experimento central** pelo redirecionamento estratégico.
`experiments/kinship/`: árvores genealógicas pequenas; predicado base
`parent(x,y)` aprendido de fatos; predicado derivado `ancestor(x,z)` **nunca
treinado** — avaliado em tempo de consulta compondo dinamicamente a fórmula de
prova `parent(x,z) ∨ ⋁_m [parent(x,m) ∧ ancestor(m,z)]` com Product T-norms
(a "execução dinâmica via DAG" do artigo, concreta). Baseline de contraste:
`ancestor_flat`, MLP treinado diretamente nos pares, sem composição.

**Resultados (500 épocas, evidência em `experiments/evidence/kinship_proof_dag/`):**

| Métrica | Composta (DAG de prova) | Plana (MLP direto) |
|---|---|---|
| Verdade em pares ancestral verdadeiros | 0.942 ± 0.017 | 0.992 ± 0.005 |
| Verdade em pares falsos | **0.017 ± 0.014** | 0.003 ± 0.005 |
| Massa de gradiente nos intermediários do caminho | **0.447 ± 0.086** | **0.000** |
| Concentração no fecho da consulta | 0.9996 | 1.0 |
| Δ verdade ao deletar os intermediários | **0.931 ± 0.020** | **0.000** |

Duas evidências centrais para o artigo reposicionado:
1. **Consistência lógica / transferência dedutiva**: a inferência composta
   discrimina quase perfeitamente (0.94 vs 0.02) **sem nunca ter sido treinada
   no conceito derivado** — o raciocínio emerge da composição do predicado base
   pelo DAG. O predicado plano precisa de supervisão direta para o mesmo.
2. **Explicabilidade arquitetural**: ~45% da massa de gradiente recai sobre os
   intermediários do caminho de prova (estruturalmente **zero** no plano), e
   deletá-los destrói a predição composta (Δ=0.93) sem afetar em nada a plana —
   a explicação é causalmente fiel à cadeia de raciocínio, propriedade que só
   existe porque a inferência atravessa o DAG.

**Limitação metodológica registrada (para o texto)**: a composição via
t-conorm satura se o predicado base não estiver calibrado em todos os
não-arcos (com negativos amostrados 2:1, a verdade dos pares falsos era 0.92!);
o OR de ~|E| ramos com valores ~0.5 tende a 1. A composição dedutiva exige
cobertura de negativos no treino do predicado base — custo O(|E|²) que limita
a escala e deve ser discutido honestamente.

### Fase 4b — Dataset real: árvore genealógica de Hinton (1986) ✅ (2026-07-19)

Replicação do resultado sintético num **benchmark canônico e citável**
(Hinton, *Learning Distributed Representations of Concepts*, 1986 — a origem
dos kinship benchmarks usados por NTP/∂ILP, dando comparabilidade direta com
os sistemas dos documentos de posicionamento). `experiments/hinton_family/`:
12 pessoas, 8 predicados base treinados (cobertura completa de negativos,
split 85/15), 4 relações derivadas **nunca treinadas** compostas por regras
com gênero via DAG de prova (ex.: `uncle(x,y) = ⋁_z brother(x,z) ∧
(father(z,y) ∨ mother(z,y))`). Evidência completa (curvas de treino PNG/PDF,
tabelas, JSONs) em `experiments/evidence/hinton_family/`.

**Resultados (500 épocas, ~20 min; acurácia held-out nos fatos base: 93,0%):**

| Relação | Composta pos/neg | Plana (treinada) pos/neg |
|---|---|---|
| uncle | 0.945 / 0.003 | 0.965 / 0.001 |
| aunt | 0.954 / 0.000 | 0.980 / 0.000 |
| nephew | 0.752 / 0.002 | 0.974 / 0.000 |
| niece | 0.778 / 0.000 | 0.971 / 0.000 |

| Explicabilidade | Composta | Plana |
|---|---|---|
| Massa de gradiente no intermediário do caminho | **0.385** | 0.000 |
| Δ verdade ao deletar o intermediário | **0.771** | 0.000 |

Leituras:
1. **A dedução composta discrimina nas 4 relações** (pos ≥ 0.75, neg ≤ 0.003)
   sem nunca ter visto o conceito derivado — replica o achado sintético num
   dataset real, com regras assimétricas de gênero.
2. **Assimetria interessante para o texto**: uncle/aunt (~0.95) > nephew/niece
   (~0.76). As regras de nephew/niece compõem `son/daughter ∧ sibling`, cujos
   predicados-base têm menos exemplos positivos por entidade — a qualidade da
   dedução herda a calibração do elo mais fraco da regra. Discussão honesta de
   como o erro do predicado base se propaga pela composição.
3. **Explicabilidade arquitetural confirmada no dataset real**: 38,5% da massa
   de gradiente no intermediário z da regra (estruturalmente 0 no plano) e
   deleção causal Δ=0.77 vs 0.00.
4. Bugs de infraestrutura encontrados e corrigidos ao escalar: RecursionError
   no backward (topological sort agora iterativa — também desbloqueia p=97) e
   encoding cp1252 do console Windows.

## Fase 5 — Sparsity comensurável entre arquiteturas (m3) ✅ métrica + calibração (2026-07-18)

Métrica única implementada (`training/metrics.py:weight_sparsity`): fração de
pesos |w| < ε (ε=1e-3) nas matrizes das camadas Linear dos predicados — a mesma
régua para qualquer mecanismo inferencial, respondendo ao item m3. Medição e
calibração executadas em p=13 (400 épocas, evidência em
`experiments/evidence/fidelity_sparsity_p13/`):

**Fidelidade axiomática em instanciações held-out + esparsidade por arquitetura:**

| Arquitetura | Fidelidade geral | Comutatividade (held-out) | Identidade | Esparsidade |
|---|---|---|---|---|
| DLG | 0.882 | 0.856 | **0.998** | 2.0% |
| MLP puro | 0.777 | 0.836 | **0.511** | 0.3% |
| LTN | 0.879 | 0.854 | 0.995 | 0.2% |

**Varredura de gamma_l1 (DLG) — trade-off esparsidade × fidelidade:**

| gamma_l1 | Esparsidade | Fidelidade |
|---|---|---|
| 1e-4 (antigo) | 2.0% | 0.882 |
| 1e-3 | 34.4% | 0.857 |
| **1e-2** | **80.3%** | **0.820** |
| 1e-1 | 82.3% | 0.730 |

Leituras para o texto:
1. **gamma=1e-2 é o ponto de operação**: 80% dos pesos zerados com perda modesta
   de fidelidade (0.88→0.82) — a alegação de esparsidade/auditabilidade do
   artigo agora tem uma curva de trade-off real por trás, em vez do gamma=1e-4
   original que não produzia esparsidade alguma.
2. **Treinar com axiomas produz modelos que os satisfazem**: gap de identidade
   0.998 vs 0.511 contra o MLP puro. **Ressalva de honestidade**: as
   instanciações de identidade medidas são as mesmas usadas como axioma de
   treino (o axioma cobre todo o domínio), então esse número mistura
   generalização com memorização do axioma; já a comutatividade é medida em
   pares held-out genuínos — e aí a vantagem sobre o MLP é modesta
   (0.856 vs 0.836). O texto deve reportar as duas separadamente.
3. **Piso da métrica**: a implicação de Reichenbach dá I(t1,t2) ≥ 1−t1, então
   predicados incertos (t≈0.5) já pontuam ~0.5+ sem saber nada — o parágrafo
   metodológico do artigo deve explicitar isso para não inflar a leitura.

## Fase 6 — Explicabilidade quantitativa (M5) ⚠️ parcialmente concluída (2026-07-18)

1. ✅ **Métrica de fidelidade/concentração** — `src/neurosym/explainability/metrics.py`:
   `compute_influences` (norma L1 do gradiente por embedding), `concentration`
   (fração da massa sobre as constantes relevantes), agregação sobre N consultas
   via `evaluate_queries`. Testado com predicado de estrutura conhecida.
2. ✅ **Protocolo deletion/insertion** — mesmo módulo: curvas de deleção/inserção
   cumulativa com restauração garantida dos embeddings, AUC normalizada, e
   comparação contra ordem aleatória como baseline de fidelidade.
3. ⏸ **Comparação com XAI post-hoc** (leave-one-out / gradientes integrados) —
   ainda pendente.

**Primeira avaliação real (DLG memorizado, p=13, 25 consultas de validação,
`experiments/evidence/explainability_dlg_p13/`):**

| Métrica | Resultado |
|---|---|
| Concentração do gradiente | **1.000 ± 0.000** |
| Deletion AUC (ordem do gradiente) | 0.712 ± 0.064 |
| Deletion AUC (ordem aleatória) | 0.753 ± 0.213 |
| Insertion AUC (ordem do gradiente) | 0.698 ± 0.335 |
| Insertion AUC (ordem aleatória) | 0.683 ± 0.240 |

**Leitura honesta, importante para o texto do artigo:**
- A concentração perfeita (1.0) **não é evidência de explicação aprendida**
  neste domínio: como as consultas são atômicas e grounded (sem encadeamento
  de regras na inferência), o grafo computacional só toca os 3 embeddings da
  tripla consultada — o gradiente das demais constantes é estruturalmente
  zero. É uma propriedade da arquitetura, não do treinamento. A alegação do
  artigo de "concentração no fecho transitivo" só se torna informativa num
  domínio com encadeamento real de regras (ex.: o segundo domínio relacional
  da Fase 4), onde o fecho é maior que as constantes literais da consulta.
  O artigo não deve usar este número como evidência forte sem essa ressalva.
- A fidelidade por deletion/insertion vai na direção certa (deletar na ordem
  do gradiente degrada mais rápido: 0.712 < 0.753; inserir recupera mais
  rápido: 0.698 > 0.683), mas as margens são pequenas e a variância alta —
  em parte porque, com concentração 1.0, a ordem além das 3 constantes
  relevantes é arbitrária (todas têm gradiente zero). Números medidos sobre
  um modelo **memorizado, não generalizante** — devem ser re-medidos quando
  houver um modelo que generalize.

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
