"""
Etapa 3 — fatoracao de Cholesky para o sistema nodal A x = b.

Entrada do circuito Falstad (.txt): igual a solvers_nodal.py
  - Run sem argumentos: menu interativo (default = circuito_falstad.txt)
  - python fatoracao_cholesky.py caminho/arquivo.txt
  - python fatoracao_cholesky.py --paste

A = G (matriz de condutancias) e SPD neste circuito, portanto
A = L L^T com L triangular inferior e diagonal positiva.
A raiz da diagonal sai por Newton (Heron), sem math.sqrt.

Bibliotecas em uso (nenhuma resolve Ax=b nem fatora):
  - __future__.annotations — adia avaliacao de type hints
  - sys — stdin/stderr e saida do processo
  - typing — aliases; sem matematica em runtime
  - solvers_nodal — parser Falstad, I/O de matrizes e Gauss (so para check)
  - fatoracao_LU — resolver_lu (comparacao de custo pedida no PDF)
  - custo_computacional — formulas de flops por fase (so para exibir o custo)

Proibido para fatorar/resolver: numpy / scipy / math (ou equivalentes).
A fatoracao de Cholesky e 100% loops manuais com + - * /.
"""

from __future__ import annotations

import sys
from typing import Callable, List, Optional, Sequence, Tuple

from custo_computacional import formulas_cholesky, imprimir_tabela_fases
from fatoracao_LU import resolver_lu
from solvers_nodal import (
    Matrix,
    Vector,
    copiar_vetor,
    eliminacao_gaussiana,
    imprimir_cabecalho,
    imprimir_matriz,
    imprimir_secao,
    imprimir_vetor,
    max_abs_diff_matriz,
    max_abs_diff_vetor,
    obter_sistema,
    residuo,
    verificar_referencia,
)


# ---------------------------------------------------------------------------
# Raiz quadrada por Newton (Heron)
# ---------------------------------------------------------------------------

def raiz_quadrada(s: float) -> Tuple[float, int]:
    """
    sqrt(s) por iteracao x <- 0.5 * (x + s/x).

    Etapa: cada atualizacao conta 3 flops (divisao, soma, produto).
    A comparacao de parada nao entra na conta.
    """
    if s <= 0.0:
        raise ValueError(f"Argumento nao positivo para raiz: {s} (matriz nao SPD).")

    x = s if s >= 1.0 else 1.0
    flops = 0
    for _ in range(60):
        x_novo = 0.5 * (x + s / x)
        flops += 3
        d = x_novo - x
        if d < 0.0:
            d = -d
        x = x_novo
        if d < 1e-15:
            break
    return x, flops


# ---------------------------------------------------------------------------
# Fatoracao de Cholesky (Banachiewicz, L inferior)
# ---------------------------------------------------------------------------

def decomposicao_cholesky(
    A: Matrix,
    raiz: Callable[[float], Tuple[float, int]] = raiz_quadrada,
) -> Tuple[Matrix, int, List[int]]:
    """
    Fatora A = L L^T.

    Etapa: coluna j. A diagonal e a raiz do residual;
    abaixo da diagonal, L[i][j] = (A[i][j] - soma) / L[j][j].
    Convencao de flops: a - b*c conta 2; cada divisao conta 1.
    Os flops das raizes (Newton) ficam separados, um valor por raiz, para
    que a parte aritmetica seja comparavel com a LU. `raiz` permite trocar
    a funcao de raiz (ver raiz_quadrada_chute em custo_computacional).

    Retorna L, flops aritmeticos (sem raizes) e a lista de flops de cada raiz.
    """
    n = len(A)
    L: Matrix = [[0.0] * n for _ in range(n)]
    flops = 0
    flops_raizes: List[int] = []

    for j in range(n):
        s = A[j][j]
        for k in range(j):
            s = s - L[j][k] * L[j][k]
            flops += 2

        L[j][j], flops_sqrt = raiz(s)
        flops_raizes.append(flops_sqrt)
        if abs(L[j][j]) < 1e-15:
            raise ZeroDivisionError(f"Pivo nulo na coluna {j} (Cholesky).")

        for i in range(j + 1, n):
            s = A[i][j]
            for k in range(j):
                s = s - L[i][k] * L[j][k]
                flops += 2
            L[i][j] = s / L[j][j]
            flops += 1

    return L, flops, flops_raizes


def substituicao_direta(L: Matrix, b: Vector) -> Tuple[Vector, int]:
    """
    Resolve L y = b (forward). Lii nao e 1: ha divisao.

    Etapa: y[i] = (b[i] - soma_j<i L[i][j] * y[j]) / L[i][i].
    """
    n = len(L)
    y = copiar_vetor(b)
    flops = 0

    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma = soma + L[i][j] * y[j]
            flops += 2
        y[i] = (b[i] - soma) / L[i][i]
        flops += 2

    return y, flops


def substituicao_retroativa_Lt(L: Matrix, y: Vector) -> Tuple[Vector, int]:
    """
    Resolve L^T x = y (backward) lendo L[j][i], sem montar L^T.

    Etapa: x[i] = (y[i] - soma_j>i L[j][i] * x[j]) / L[i][i].
    """
    n = len(L)
    x = [0.0] * n
    flops = 0

    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma = soma + L[j][i] * x[j]
            flops += 2
        x[i] = (y[i] - soma) / L[i][i]
        flops += 2

    return x, flops


def resolver_cholesky(
    A: Matrix, b: Vector
) -> Tuple[Matrix, Vector, Vector, int, List[int], int, int, int]:
    """
    Encadeia: A = L L^T, L y = b, L^T x = y.

    Retorna L, y, x e os flops (fatoracao sem raizes, lista por raiz,
    forward, backward, total).
    """
    L, flops_fat, flops_raizes = decomposicao_cholesky(A)
    y, flops_fwd = substituicao_direta(L, b)
    x, flops_bwd = substituicao_retroativa_Lt(L, y)
    flops_total = flops_fat + sum(flops_raizes) + flops_fwd + flops_bwd
    return L, y, x, flops_fat, flops_raizes, flops_fwd, flops_bwd, flops_total


# ---------------------------------------------------------------------------
# Verificacoes (loops manuais, sem biblioteca)
# ---------------------------------------------------------------------------

def multiplicar_LLt(L: Matrix) -> Matrix:
    """Reconstroi A_hat = L * L^T  com  (L L^T)_ij = soma_k L[i][k] * L[j][k]."""
    n = len(L)
    A_hat: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            soma = 0.0
            for k in range(n):
                soma = soma + L[i][k] * L[j][k]
            A_hat[i][j] = soma
    return A_hat


def executar_cholesky(
    A: Matrix, b: Vector, flops_lu_fat: int, flops_lu: int
) -> Tuple[Vector, int, int]:
    """
    Fatora, resolve e imprime L, y, x, custos, verificacao e a
    comparacao de custo com a LU. Retorna x, flops sem raizes e total.
    """
    imprimir_secao(
        "Fatoracao de Cholesky (Banachiewicz)\n"
        "A = L L^T;  L y = b;  L^T x = y"
    )
    (
        L, y, x, flops_fat, flops_raizes, flops_fwd, flops_bwd, flops_total
    ) = resolver_cholesky(A, b)
    flops_raiz = sum(flops_raizes)
    flops_sem_raiz = flops_total - flops_raiz
    imprimir_matriz("L", L)
    imprimir_vetor("\ny (Ly=b)", y)
    imprimir_vetor("x (Cholesky)", x)
    print(f"Flops (fatoracao Cholesky, sem raizes): {flops_fat}")
    print(f"Flops (raizes por Newton, {len(A)} raizes): {flops_raiz}")
    print(f"Flops (forward Ly=b): {flops_fwd}")
    print(f"Flops (backward L^T x=y): {flops_bwd}")
    print(f"Flops (Cholesky, sem raizes): {flops_sem_raiz}")
    print(f"Flops (Cholesky, total com raizes): {flops_total}")
    imprimir_tabela_fases(
        f"Custo computacional - Cholesky, sem raizes (n = {len(A)})",
        [flops_fat, flops_fwd, flops_bwd],
        formulas_cholesky(len(A)),
    )
    print(f"  Raizes por Newton (flops por raiz L[j][j]): {flops_raizes}"
          f" = {flops_raiz}")
    print("  (3 flops por iteracao; depende de quantas iteracoes cada raiz leva.)")

    imprimir_secao("Verificacao Cholesky")
    print(f"max |A - L L^T| = {max_abs_diff_matriz(A, multiplicar_LLt(L)):.3e}")
    print(f"residuo max |A x - b| = {residuo(A, x, b):.3e}")

    imprimir_secao("Comparacao de custo: Cholesky x LU")
    print(f"Fatoracao:  LU = {flops_lu_fat}  |  Cholesky = {flops_fat} (sem raizes)")
    print(f"Total:      LU = {flops_lu}  |  Cholesky = {flops_sem_raiz} (sem raizes)"
          f"  /  {flops_total} (com raizes)")
    print("Cholesky explora a simetria de A: so calcula L, com menos trabalho")
    print("que a LU (tende a metade para n grande). As raizes por Newton sao um custo")
    print("extra, porque aqui a raiz e iterativa (sem math.sqrt).")
    return x, flops_sem_raiz, flops_total


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    A, b, nos, origem, check_ref = obter_sistema(argv, metodo="fatoracao de Cholesky")
    imprimir_cabecalho("Etapa 3: Fatoracao de Cholesky", origem, nos, A, b)

    _, _, x_lu, flops_lu_fat, _, _, flops_lu = resolver_lu(A, b)
    x_ch, _, _ = executar_cholesky(A, b, flops_lu_fat, flops_lu)

    imprimir_secao("Cholesky x Gauss x LU")
    x_g, _ = eliminacao_gaussiana(A, b)
    imprimir_vetor("x (Gauss)", x_g)
    imprimir_vetor("x (LU)", x_lu)
    print(f"max |x_Cholesky - x_Gauss| = {max_abs_diff_vetor(x_ch, x_g):.3e}")
    print(f"max |x_Cholesky - x_LU| = {max_abs_diff_vetor(x_ch, x_lu):.3e}")

    if check_ref:
        verificar_referencia(A, b, x_ch)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, ZeroDivisionError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
    except EOFError:
        print("\nEntrada interrompida.", file=sys.stderr)
        sys.exit(1)
