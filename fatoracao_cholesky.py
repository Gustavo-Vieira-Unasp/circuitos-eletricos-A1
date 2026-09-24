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
  - fatoracao_LU — resolver_lu e max_abs_diff_vetor (comparacao pedida no PDF)

Proibido para fatorar/resolver: numpy / scipy / math (ou equivalentes).
A fatoracao de Cholesky e 100% loops manuais com + - * /.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence, Tuple

from fatoracao_LU import max_abs_diff_vetor, resolver_lu
from solvers_nodal import (
    Matrix,
    Vector,
    copiar_vetor,
    eliminacao_gaussiana,
    imprimir_matriz,
    imprimir_vetor,
    obter_sistema,
    sistema_referencia,
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

def decomposicao_cholesky(A: Matrix) -> Tuple[Matrix, int]:
    """
    Fatora A = L L^T.

    Etapa: coluna j. A diagonal e a raiz do residual;
    abaixo da diagonal, L[i][j] = (A[i][j] - soma) / L[j][j].
    Convenção de flops: a - b*c conta 2; cada divisao conta 1;
    a raiz soma os flops de Newton.
    """
    n = len(A)
    L: Matrix = [[0.0] * n for _ in range(n)]
    flops = 0

    for j in range(n):
        s = A[j][j]
        for k in range(j):
            s = s - L[j][k] * L[j][k]
            flops += 2

        L[j][j], flops_sqrt = raiz_quadrada(s)
        flops += flops_sqrt
        if abs(L[j][j]) < 1e-15:
            raise ZeroDivisionError(f"Pivo nulo na coluna {j} (Cholesky).")

        for i in range(j + 1, n):
            s = A[i][j]
            for k in range(j):
                s = s - L[i][k] * L[j][k]
                flops += 2
            L[i][j] = s / L[j][j]
            flops += 1

    return L, flops


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
) -> Tuple[Matrix, Vector, Vector, int, int, int, int]:
    """
    Encadeia: A = L L^T, L y = b, L^T x = y.

    Retorna L, y, x e os flops (fatoracao, forward, backward, total).
    """
    L, flops_fat = decomposicao_cholesky(A)
    y, flops_fwd = substituicao_direta(L, b)
    x, flops_bwd = substituicao_retroativa_Lt(L, y)
    flops_total = flops_fat + flops_fwd + flops_bwd
    return L, y, x, flops_fat, flops_fwd, flops_bwd, flops_total


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


def max_abs_diff_matriz(A: Matrix, B: Matrix) -> float:
    m = 0.0
    for i in range(len(A)):
        for j in range(len(A[i])):
            d = abs(A[i][j] - B[i][j])
            if d > m:
                m = d
    return m


def residuo(A: Matrix, x: Vector, b: Vector) -> float:
    """max_i | (A x)_i - b_i |."""
    n = len(A)
    rmax = 0.0
    for i in range(n):
        soma = 0.0
        for j in range(n):
            soma = soma + A[i][j] * x[j]
        d = abs(soma - b[i])
        if d > rmax:
            rmax = d
    return rmax


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    A, b, nos, origem, check_ref = obter_sistema(argv)

    print()
    print("=" * 60)
    print("Etapa 3: Fatoracao de Cholesky")
    print(f"Circuito: {origem}")
    print(f"Nos livres (ordem de x), reps (x,y): {nos}")
    print("=" * 60)

    imprimir_matriz("A", A)
    imprimir_vetor("\nb", b)

    print("\n" + "-" * 60)
    print("Fatoracao de Cholesky (Banachiewicz)")
    print("A = L L^T;  L y = b;  L^T x = y")
    print("-" * 60)
    L, y, x_ch, flops_fat, flops_fwd, flops_bwd, flops_total = resolver_cholesky(
        A, b
    )
    imprimir_matriz("L", L)
    imprimir_vetor("\ny (Ly=b)", y)
    imprimir_vetor("x (Cholesky)", x_ch)
    print(f"Flops (fatoracao Cholesky): {flops_fat}")
    print(f"Flops (forward Ly=b): {flops_fwd}")
    print(f"Flops (backward L^T x=y): {flops_bwd}")
    print(f"Flops (Cholesky, total): {flops_total}")

    print("\n" + "-" * 60)
    print("Verificacao")
    print("-" * 60)
    A_hat = multiplicar_LLt(L)
    print(f"max |A - L L^T| = {max_abs_diff_matriz(A, A_hat):.3e}")
    print(f"residuo max |A x - b| = {residuo(A, x_ch, b):.3e}")

    x_g, _ = eliminacao_gaussiana(A, b)
    imprimir_vetor("x (Gauss)", x_g)
    print(f"max |x_Cholesky - x_Gauss| = {max_abs_diff_vetor(x_ch, x_g):.3e}")

    _, _, x_lu, _, _, _, flops_lu = resolver_lu(A, b)
    imprimir_vetor("x (LU)", x_lu)
    print(f"max |x_Cholesky - x_LU| = {max_abs_diff_vetor(x_ch, x_lu):.3e}")
    print(f"Flops (LU, total): {flops_lu}")
    print(f"Flops (Cholesky, total): {flops_total}")

    if check_ref:
        print("\n" + "-" * 60)
        print("Sanity check vs formulacao_nodal.md (circuito default)")
        print("-" * 60)
        A_ref, b_ref = sistema_referencia()
        print(f"max |A - A_ref| = {max_abs_diff_matriz(A, A_ref):.3e}")
        print(f"max |b - b_ref| = {max_abs_diff_vetor(b, b_ref):.3e}")
        _, _, x_ref, _, _, _, _ = resolver_cholesky(A_ref, b_ref)
        imprimir_vetor("x (ref Etapa 2, Cholesky)", x_ref)
        print(f"max |x - x_ref| = {max_abs_diff_vetor(x_ch, x_ref):.3e}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, ZeroDivisionError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
    except EOFError:
        print("\nEntrada interrompida.", file=sys.stderr)
        sys.exit(1)
