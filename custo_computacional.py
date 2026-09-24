"""
Etapa 3 — custo computacional: formulas fechadas de flops e duas reducoes.

Convencao (a mesma dos solvers): a - b*c conta 2 flops; cada divisao conta 1.

Conteudo:
  - formulas_gauss / formulas_lu / formulas_cholesky: flops por fase em
    funcao de n (conferem com os contadores dos solvers).
  - imprimir_comparacao_assintotica: totais para n = 5, 50, 500.
  - eliminacao_gaussiana_esparsa: Gauss que pula os zeros de G.
  - raiz_quadrada_chute: Newton com chute inicial melhor e parada antecipada.

Bibliotecas: so typing (aliases). Nenhuma conta usa biblioteca.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Formulas fechadas (inteiros; conferem com os contadores dos solvers)
# ---------------------------------------------------------------------------

def formulas_gauss(n: int) -> Dict[str, int]:
    """
    Eliminacao em A: para cada coluna k ha m = n-1-k linhas abaixo, cada uma
    com 1 divisao + 2m flops -> soma m(1 + 2m) = n(n-1)/2 + n(n-1)(2n-1)/3.
    Atualizacao de b: 2 flops por linha eliminada -> n(n-1).
    Retroativa: 2 por termo + 2 por linha -> n^2 + n.
    """
    return {
        "eliminacao em A": n * (n - 1) // 2 + n * (n - 1) * (2 * n - 1) // 3,
        "atualizacao de b": n * (n - 1),
        "substituicao retroativa": n * n + n,
    }


def formulas_lu(n: int) -> Dict[str, int]:
    """
    Fatoracao: mesmas contas da eliminacao em A.
    Direta com Lii = 1: 2 por termo + 1 subtracao por linha (i > 0) -> n^2 - 1.
    Retroativa: n^2 + n.
    """
    return {
        "fatoracao A = LU": n * (n - 1) // 2 + n * (n - 1) * (2 * n - 1) // 3,
        "substituicao direta Ly = b": n * n - 1,
        "substituicao retroativa Ux = y": n * n + n,
    }


def formulas_cholesky(n: int) -> Dict[str, int]:
    """
    Fatoracao sem raizes: coluna j tem 2j flops na diagonal e (n-1-j)(2j+1)
    abaixo dela -> soma = (2n^3 + 3n^2 - 5n)/6 ~ n^3/3.
    Direta e retroativa (Lii != 1): n^2 + n cada.
    As raizes (Newton) dependem do numero de iteracoes: sem formula fechada.
    """
    return {
        "fatoracao A = LL^T (sem raizes)": (2 * n ** 3 + 3 * n ** 2 - 5 * n) // 6,
        "substituicao direta Ly = b": n * n + n,
        "substituicao retroativa L^T x = y": n * n + n,
    }


def imprimir_tabela_fases(
    titulo: str, contados: Sequence[int], formulas: Dict[str, int]
) -> None:
    """Tabela Fase | contado | formula(n)."""
    print(f"\n{titulo}")
    print(f"  {'Fase':<36}{'contado':>9}{'formula':>9}")
    for (fase, esperado), contado in zip(formulas.items(), contados):
        marca = "" if contado == esperado else "  <-- diferente"
        print(f"  {fase:<36}{contado:>9}{esperado:>9}{marca}")
    print(f"  {'total':<36}{sum(contados):>9}{sum(formulas.values()):>9}")


def imprimir_comparacao_assintotica(ns: Sequence[int] = (5, 50, 500)) -> None:
    """Totais por metodo; a razao Cholesky/LU tende a 1/2 quando n cresce."""
    print(f"  {'n':>5}{'Gauss':>14}{'LU':>14}{'Cholesky*':>14}{'Chol/LU':>10}")
    for n in ns:
        g = sum(formulas_gauss(n).values())
        lu = sum(formulas_lu(n).values())
        ch = sum(formulas_cholesky(n).values())
        print(f"  {n:>5}{g:>14}{lu:>14}{ch:>14}{ch / lu:>10.3f}")
    print("  * Cholesky sem as n raizes (custo linear em n, desprezivel perto de n^3/3).")
    print("  Ordens: Gauss ~ 2n^3/3, LU ~ 2n^3/3, Cholesky ~ n^3/3.")


# ---------------------------------------------------------------------------
# Reducao A: Gauss que explora os zeros de G (esparsidade)
# ---------------------------------------------------------------------------

def eliminacao_gaussiana_esparsa(
    A: List[List[float]], b: List[float]
) -> Tuple[List[float], int]:
    """
    Mesmo algoritmo da eliminacao gaussiana, mas:
      - se M[i][k] == 0 a linha i nao precisa ser eliminada (fator 0);
      - termos com M[k][j] == 0 (ou x[j] multiplicado por 0) sao pulados.
    Em G cada no so se liga aos vizinhos, entao ha zeros estruturais.
    Conferir zero e uma comparacao, nao entra na conta de flops.
    """
    n = len(A)
    M = [row[:] for row in A]
    y = b[:]
    flops = 0

    for k in range(n - 1):
        pivo = M[k][k]
        if abs(pivo) < 1e-15:
            raise ZeroDivisionError(f"Pivo nulo na coluna {k} (sem pivoteamento).")
        for i in range(k + 1, n):
            if M[i][k] == 0.0:
                continue
            fator = M[i][k] / pivo
            flops += 1
            M[i][k] = 0.0
            for j in range(k + 1, n):
                if M[k][j] != 0.0:
                    M[i][j] = M[i][j] - fator * M[k][j]
                    flops += 2
            if y[k] != 0.0:
                y[i] = y[i] - fator * y[k]
                flops += 2

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            if M[i][j] != 0.0:
                soma = soma + M[i][j] * x[j]
                flops += 2
        x[i] = (y[i] - soma) / M[i][i]
        flops += 2

    return x, flops


# ---------------------------------------------------------------------------
# Reducao B: raiz por Newton com chute melhor e parada antecipada
# ---------------------------------------------------------------------------

def raiz_quadrada_chute(s: float) -> Tuple[float, int]:
    """
    sqrt(s) por Newton, mais barato que raiz_quadrada:
      1. reducao de faixa: s = s' * 4^k com s' em [1/4, 4] (2 flops por passo,
         e 1 flop para desfazer no fim), pois sqrt(4^k) = 2^k;
      2. chute x0 = (1 + s')/2 (2 flops), a reta tangente de sqrt em s' = 1;
      3. parada quando |x_novo - x| < 1e-8: como Newton converge
         quadraticamente, o erro do novo x ja e ~ d^2 / (2x) < 1e-16,
         entao a iteracao extra de confirmacao e dispensavel.
    """
    if s <= 0.0:
        raise ValueError(f"Argumento nao positivo para raiz: {s} (matriz nao SPD).")

    flops = 0
    escala = 1.0
    while s > 4.0:
        s = s / 4.0
        escala = escala * 2.0
        flops += 2
    while s < 0.25:
        s = s * 4.0
        escala = escala / 2.0
        flops += 2

    x = (1.0 + s) / 2.0
    flops += 2
    for _ in range(60):
        x_novo = 0.5 * (x + s / x)
        flops += 3
        d = x_novo - x
        if d < 0.0:
            d = -d
        x = x_novo
        if d < 1e-8:
            break

    if escala != 1.0:
        x = x * escala
        flops += 1
    return x, flops
