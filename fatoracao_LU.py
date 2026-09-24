"""
Etapa 3 — decomposicao LU (Doolittle) para o sistema nodal A x = b.

Entrada do circuito Falstad (.txt): igual a solvers_nodal.py
  - Run sem argumentos: menu interativo (default = circuito_falstad.txt)
  - python fatoracao_LU.py caminho/arquivo.txt
  - python fatoracao_LU.py --paste

A = G (matriz de condutancias) e SPD neste circuito, portanto a
fatoracao segue sem pivoteamento parcial: os pivos de U permanecem
positivos. L tem diagonal 1; U e triangular superior.

Bibliotecas em uso (nenhuma resolve Ax=b nem fatora):
  - __future__.annotations — adia avaliacao de type hints
  - sys — stdin/stderr e saida do processo
  - typing — aliases; sem matematica em runtime
  - solvers_nodal — parser Falstad, I/O de matrizes e Gauss (so para check)

Proibido para fatorar/resolver: numpy / scipy / math (ou equivalentes).
A decomposicao LU e 100% loops manuais com + - * /.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence, Tuple

from solvers_nodal import (
    Matrix,
    Vector,
    copiar_matriz,
    copiar_vetor,
    eliminacao_gaussiana,
    imprimir_matriz,
    imprimir_vetor,
    obter_sistema,
    sistema_referencia,
)


# ---------------------------------------------------------------------------
# Decomposicao LU (Doolittle, sem pivoteamento)
# ---------------------------------------------------------------------------

def decomposicao_lu(A: Matrix) -> Tuple[Matrix, Matrix, int]:
    """
    Fatora A = L U.

    Etapa: copia A em U; L comeca como identidade (Lii = 1).
    Para cada coluna k, o multiplicador L[i][k] = U[i][k] / U[k][k]
    elimina a subdiagonal de U (j = k+1 ... n-1).
    """
    n = len(A)
    U = copiar_matriz(A)
    L: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    flops = 0

    for k in range(n - 1):
        pivo = U[k][k]
        if abs(pivo) < 1e-15:
            raise ZeroDivisionError(f"Pivo nulo na coluna {k} (sem pivoteamento).")

        for i in range(k + 1, n):
            # multiplicador guarda-se em L; 1 divisao
            L[i][k] = U[i][k] / pivo
            flops += 1
            U[i][k] = 0.0
            for j in range(k + 1, n):
                U[i][j] = U[i][j] - L[i][k] * U[k][j]
                flops += 2

    return L, U, flops


def substituicao_direta(L: Matrix, b: Vector) -> Tuple[Vector, int]:
    """
    Resolve L y = b (forward). Lii = 1, logo nao ha divisao.

    Etapa: y[i] = b[i] - soma_j<i L[i][j] * y[j].
    """
    n = len(L)
    y = copiar_vetor(b)
    flops = 0

    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma = soma + L[i][j] * y[j]
            flops += 2
        y[i] = b[i] - soma
        if i > 0:
            flops += 1

    return y, flops


def substituicao_retroativa(U: Matrix, y: Vector) -> Tuple[Vector, int]:
    """
    Resolve U x = y (backward). Mesma convencao de flops da Gauss:
    +2 por termo da soma; +2 na linha (subtracao + divisao pelo pivo).
    """
    n = len(U)
    x = [0.0] * n
    flops = 0

    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma = soma + U[i][j] * x[j]
            flops += 2
        x[i] = (y[i] - soma) / U[i][i]
        flops += 2

    return x, flops


def resolver_lu(
    A: Matrix, b: Vector
) -> Tuple[Matrix, Matrix, Vector, int, int, int, int]:
    """
    Encadeia: A = LU, L y = b, U x = y.

    Retorna L, U, x e os flops (fatoracao, forward, backward, total).
    """
    L, U, flops_fat = decomposicao_lu(A)
    y, flops_fwd = substituicao_direta(L, b)
    x, flops_bwd = substituicao_retroativa(U, y)
    flops_total = flops_fat + flops_fwd + flops_bwd
    return L, U, x, flops_fat, flops_fwd, flops_bwd, flops_total


# ---------------------------------------------------------------------------
# Verificacoes (loops manuais, sem biblioteca)
# ---------------------------------------------------------------------------

def multiplicar_LU(L: Matrix, U: Matrix) -> Matrix:
    """Reconstroi A_hat = L * U para checar a fatoracao."""
    n = len(L)
    A_hat: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            soma = 0.0
            for k in range(n):
                soma = soma + L[i][k] * U[k][j]
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


def max_abs_diff_vetor(u: Vector, v: Vector) -> float:
    m = 0.0
    for i in range(len(u)):
        d = abs(u[i] - v[i])
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
    print("Etapa 3: Decomposicao LU")
    print(f"Circuito: {origem}")
    print(f"Nos livres (ordem de x), reps (x,y): {nos}")
    print("=" * 60)

    imprimir_matriz("A", A)
    imprimir_vetor("\nb", b)

    print("\n" + "-" * 60)
    print("Decomposicao LU (Doolittle, sem pivoteamento)")
    print("A = L U;  L y = b;  U x = y")
    print("-" * 60)
    L, U, x_lu, flops_fat, flops_fwd, flops_bwd, flops_total = resolver_lu(A, b)
    imprimir_matriz("L", L)
    imprimir_matriz("U", U)
    imprimir_vetor("\nx (LU)", x_lu)
    print(f"Flops (fatoracao LU): {flops_fat}")
    print(f"Flops (forward Ly=b): {flops_fwd}")
    print(f"Flops (backward Ux=y): {flops_bwd}")
    print(f"Flops (LU, total): {flops_total}")

    print("\n" + "-" * 60)
    print("Verificacao")
    print("-" * 60)
    A_hat = multiplicar_LU(L, U)
    print(f"max |A - L U| = {max_abs_diff_matriz(A, A_hat):.3e}")
    print(f"residuo max |A x - b| = {residuo(A, x_lu, b):.3e}")

    x_g, _ = eliminacao_gaussiana(A, b)
    imprimir_vetor("x (Gauss)", x_g)
    print(f"max |x_LU - x_Gauss| = {max_abs_diff_vetor(x_lu, x_g):.3e}")

    if check_ref:
        print("\n" + "-" * 60)
        print("Sanity check vs formulacao_nodal.md (circuito default)")
        print("-" * 60)
        A_ref, b_ref = sistema_referencia()
        print(f"max |A - A_ref| = {max_abs_diff_matriz(A, A_ref):.3e}")
        print(f"max |b - b_ref| = {max_abs_diff_vetor(b, b_ref):.3e}")
        _, _, x_ref, _, _, _, _ = resolver_lu(A_ref, b_ref)
        imprimir_vetor("x (ref Etapa 2, LU)", x_ref)
        print(f"max |x - x_ref| = {max_abs_diff_vetor(x_lu, x_ref):.3e}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, ZeroDivisionError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
    except EOFError:
        print("\nEntrada interrompida.", file=sys.stderr)
        sys.exit(1)
