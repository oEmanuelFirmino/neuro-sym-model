from typing import List, Optional


def time_to_generalization(
    val_curve: List[float],
    threshold: float = 0.95,
    patience: int = 1,
) -> Optional[int]:
    """Tempo para Generalização (T_g).

    A definição original (artigo, Seção 3.6) é T_g = min{t | Aval(t') >= threshold,
    para todo t' >= t}: uma vez cruzado o limiar, a acurácia de validação nunca
    mais poderia cair abaixo dele. Isso pressupõe uma curva monotonicamente
    estável após a transição, o que curvas de grokking nem sempre exibem, e não é
    computável online (exigiria conhecer o futuro da curva) -- ponto levantado no
    parecer do orientador (item m2).

    Aqui T_g é redefinido operacionalmente como o primeiro índice `t` a partir do
    qual a acurácia permanece >= `threshold` por uma janela sustentada de
    `patience` avaliações consecutivas. `patience=1` recupera o comportamento de
    "primeiro cruzamento" (frágil a flutuações isoladas); valores maiores tornam
    a métrica robusta a oscilações passageiras sem exigir estabilidade perpétua.

    Args:
        val_curve: acurácia de validação por época, em ordem cronológica.
        threshold: limiar de generalização (tau no artigo; tau=0.95 no protocolo).
        patience: número mínimo de avaliações consecutivas acima do limiar para
            aceitar a transição como estável.

    Returns:
        O índice (época) da primeira janela sustentada acima do limiar, ou None
        se a curva nunca atinge esse critério.
    """
    if patience < 1:
        raise ValueError("patience deve ser >= 1")

    n = len(val_curve)
    for t in range(n):
        window = val_curve[t : t + patience]
        if len(window) < patience:
            break
        if all(acc >= threshold for acc in window):
            return t
    return None


def post_threshold_dip_count(
    val_curve: List[float], threshold: float = 0.95, patience: int = 1
) -> Optional[int]:
    """Quantas vezes a curva volta a cair abaixo do limiar depois do T_g sustentado.

    Diagnóstico de robustez (item m2 do parecer): mede o quão "definitiva" foi a
    transição detectada por `time_to_generalization`. Zero dips indica uma
    transição estável; valores altos indicam que o limiar de patience escolhido
    pode estar mascarando uma curva ainda instável.
    """
    t_g = time_to_generalization(val_curve, threshold=threshold, patience=patience)
    if t_g is None:
        return None
    return sum(1 for acc in val_curve[t_g:] if acc < threshold)
