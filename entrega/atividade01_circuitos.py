"""
ATIVIDADE AVALIATIVA 01 - Circuitos Eletricos e Metodos Numericos
Prof. Me. Jonas Mateus Bettiol
Alunos: Gustavo Vieira e Philipe Custódio

Arquivo unico de entrega. Acompanha o export do Falstad circuito_falstad.txt
(deixe os dois na mesma pasta para usar o circuito padrao).

Como rodar:
  python atividade01_circuitos.py            -> menu (Enter = circuito_falstad.txt)
  python atividade01_circuitos.py outro.txt  -> outro export do Falstad
  python atividade01_circuitos.py --paste    -> cola o netlist, termina com END

=============================================================================
Etapa 1 - Circuito no Falstad (circuito_falstad.txt)
=============================================================================
Ponte resistiva com diagonal, resistores Rk = k ohms (k = 1..8) e duas fontes
de corrente independentes (fontes de corrente deixam G simetrica e definida
positiva, o que permite Cholesky). Terra no trilho inferior -> 5 nos livres.

  No 1 (TL) (112,112)   No 2 (TM) (208,112)   No 3 (TR) (304,112)
  No 4 (ML) (112,208)   No 5 (C)  (208,208)   Terra: y = 304 e (304,208)

  R1 = 1 ohm  TL-TM        R5 = 5 ohms TR-C (diagonal)
  R2 = 2 ohms TM-TR        R6 = 6 ohms ML-C
  R3 = 3 ohms TL-ML        R7 = 7 ohms C-terra
  R4 = 4 ohms TM-C         R8 = 8 ohms C-terra
  Is1 = 3 A  terra -> ML (entra no no 4)
  Is2 = 2 A  terra -> TR (entra no no 3)

=============================================================================
Etapa 2 - Formulacao nodal  A x = b  (A = G, x = tensoes nodais v1..v5)
=============================================================================
Condutancias Yk = 1/Rk. KCL (corrente saindo = corrente injetada):

  No 1: Y1(v1-v2) + Y3(v1-v4)                           = 0
  No 2: Y1(v2-v1) + Y2(v2-v3) + Y4(v2-v5)               = 0
  No 3: Y2(v3-v2) + Y5(v3-v5)                           = Is2 = 2
  No 4: Y3(v4-v1) + Y6(v4-v5)                           = Is1 = 3
  No 5: Y4(v5-v2) + Y5(v5-v3) + Y6(v5-v4) + (Y7+Y8) v5  = 0

        |  4/3   -1      0     -1/3    0       |        | 0 |
        | -1      7/4   -1/2    0     -1/4     |        | 0 |
  A  =  |  0     -1/2    7/10   0     -1/5     |   b =  | 2 |
        | -1/3    0      0      1/2   -1/6     |        | 3 |
        |  0     -1/4   -1/5   -1/6   743/840  |        | 0 |

A e simetrica e os pivos da eliminacao sao todos positivos
(4/3, 1, 9/20, 23/72, 15/56), logo A e SPD.
Solucao exata: x = [1819/69, 580/23, 1808/69, 2056/69, 56/3] V.

=============================================================================
Etapa 3 - Metodos implementados do zero (loops com + - * /)
=============================================================================
  3.1 Eliminacao gaussiana + substituicao retroativa
  3.2 Decomposicao LU (Doolittle, sem pivoteamento): imprime L e U
  3.3 Fatoracao de Cholesky A = L L^T (raiz por Newton): imprime L e
      compara o custo com a LU
Convencao de flops: a - b*c conta 2; cada divisao conta 1.

Bibliotecas usadas (nenhuma fatora nem resolve A x = b):
  __future__ (type hints), argparse (linha de comando), sys (stdin/saida),
  pathlib (caminho do .txt), typing (aliases de tipo), fractions (so para
  os literais exatos da matriz de referencia da Etapa 2).
Sem numpy / scipy / math.
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ===========================================================================
# Leitura do circuito, utilitarios e Etapa 3.1 - Eliminacao gaussiana
# (origem: solvers_nodal.py)
# ===========================================================================

Matrix = List[List[float]]
Vector = List[float]
Point = Tuple[int, int]

DIR_SCRIPT = Path(__file__).resolve().parent
ARQUIVO_DEFAULT = DIR_SCRIPT / "circuito_falstad.txt"


# ---------------------------------------------------------------------------
# Referencia Etapa 2 (sanity check quando o default e usado)
# ---------------------------------------------------------------------------

def sistema_referencia() -> Tuple[Matrix, Vector]:
    """A = G e b de formulacao_nodal.md (Fraction so para literais exatos)."""
    A_frac = [
        [Fraction(4, 3), Fraction(-1), Fraction(0), Fraction(-1, 3), Fraction(0)],
        [Fraction(-1), Fraction(7, 4), Fraction(-1, 2), Fraction(0), Fraction(-1, 4)],
        [Fraction(0), Fraction(-1, 2), Fraction(7, 10), Fraction(0), Fraction(-1, 5)],
        [Fraction(-1, 3), Fraction(0), Fraction(0), Fraction(1, 2), Fraction(-1, 6)],
        [Fraction(0), Fraction(-1, 4), Fraction(-1, 5), Fraction(-1, 6), Fraction(743, 840)],
    ]
    b_frac = [Fraction(0), Fraction(0), Fraction(2), Fraction(3), Fraction(0)]
    A = [[float(x) for x in row] for row in A_frac]
    b = [float(x) for x in b_frac]
    return A, b


# ---------------------------------------------------------------------------
# Union-Find para fios / nos eletricos
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[Point, Point] = {}

    def add(self, p: Point) -> None:
        if p not in self.parent:
            self.parent[p] = p

    def find(self, p: Point) -> Point:
        self.add(p)
        while self.parent[p] != p:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def union(self, a: Point, b: Point) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # parent preferivel: menor (x,y) para representante estavel
            if ra <= rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Parser Falstad (subconjunto: r, i, w, g)
# ---------------------------------------------------------------------------

def ler_texto_circuito(texto: str) -> Tuple[Matrix, Vector, List[Point]]:
    """
    Converte export Falstad (texto) em G x = b (nos livres).

    Corrente `i`: de post1 -> post2 (injeta em post2, sai de post1),
    como stampCurrentSource(n1, n2, I) no CircuitJS.
    """
    uf = UnionFind()
    resistors: List[Tuple[Point, Point, float]] = []
    currents: List[Tuple[Point, Point, float]] = []
    wires: List[Tuple[Point, Point]] = []
    grounds: List[Point] = []

    for raw in texto.splitlines():
        line = raw.strip()
        if not line or line.startswith("$"):
            continue
        parts = line.split()
        kind = parts[0]

        if kind == "r":
            # r x1 y1 x2 y2 flags R
            x1, y1, x2, y2 = map(int, parts[1:5])
            R = float(parts[6])
            p1, p2 = (x1, y1), (x2, y2)
            uf.add(p1)
            uf.add(p2)
            resistors.append((p1, p2, R))

        elif kind == "i":
            # i x1 y1 x2 y2 flags I [maxV]
            x1, y1, x2, y2 = map(int, parts[1:5])
            I = float(parts[6])
            p1, p2 = (x1, y1), (x2, y2)
            uf.add(p1)
            uf.add(p2)
            currents.append((p1, p2, I))

        elif kind == "w":
            x1, y1, x2, y2 = map(int, parts[1:5])
            p1, p2 = (x1, y1), (x2, y2)
            uf.union(p1, p2)
            wires.append((p1, p2))

        elif kind == "g":
            x, y = int(parts[1]), int(parts[2])
            p = (x, y)
            uf.add(p)
            grounds.append(p)

    if not grounds:
        raise ValueError("Circuito sem terra (linha 'g').")

    # Se o terra cai no meio de um fio (comum no Falstad), une os extremos ao terra.
    def no_segmento(p: Point, a: Point, b: Point) -> bool:
        (px, py), (ax, ay), (bx, by) = p, a, b
        if ax == bx == px:  # vertical
            return min(ay, by) <= py <= max(ay, by)
        if ay == by == py:  # horizontal
            return min(ax, bx) <= px <= max(ax, bx)
        return False

    for g in grounds:
        for a, b in wires:
            if no_segmento(g, a, b):
                uf.union(g, a)
                uf.union(g, b)

    ground_roots = {uf.find(g) for g in grounds}

    for p1, p2, _ in resistors:
        uf.add(p1)
        uf.add(p2)
    for p1, p2, _ in currents:
        uf.add(p1)
        uf.add(p2)

    all_points = list(uf.parent.keys())
    free_roots = sorted(
        {uf.find(p) for p in all_points if uf.find(p) not in ground_roots},
        key=lambda p: (p[1], p[0]),  # y depois x → TL,TM,TR,ML,C no circuito default
    )
    if not free_roots:
        raise ValueError("Nenhum no livre apos aterrar o circuito.")

    index = {root: k for k, root in enumerate(free_roots)}
    n = len(free_roots)
    G: Matrix = [[0.0] * n for _ in range(n)]
    b: Vector = [0.0] * n

    def idx(p: Point) -> Optional[int]:
        r = uf.find(p)
        if r in ground_roots:
            return None
        return index[r]

    for p1, p2, R in resistors:
        if abs(R) < 1e-18:
            raise ValueError(f"Resistencia nula entre {p1} e {p2}.")
        Y = 1.0 / R
        i, j = idx(p1), idx(p2)
        if i is None and j is None:
            continue
        if i is not None and j is None:
            G[i][i] += Y
        elif i is None and j is not None:
            G[j][j] += Y
        else:
            assert i is not None and j is not None
            G[i][i] += Y
            G[j][j] += Y
            G[i][j] -= Y
            G[j][i] -= Y

    for p1, p2, I in currents:
        i, j = idx(p1), idx(p2)
        if i is not None:
            b[i] -= I
        if j is not None:
            b[j] += I

    return G, b, free_roots


def carregar_de_arquivo(caminho: Path) -> Tuple[Matrix, Vector, List[Point], str]:
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")
    texto = caminho.read_text(encoding="utf-8")
    A, b, nos = ler_texto_circuito(texto)
    return A, b, nos, str(caminho)


def carregar_de_paste() -> Tuple[Matrix, Vector, List[Point], str]:
    print("Cole o export Falstad. Termine com uma linha so com END (ou Ctrl+Z/Ctrl+D).")
    linhas: List[str] = []
    for line in sys.stdin:
        if line.strip() == "END":
            break
        linhas.append(line)
    texto = "".join(linhas)
    if not texto.strip():
        raise ValueError("Nenhum texto colado.")
    A, b, nos = ler_texto_circuito(texto)
    return A, b, nos, "(paste)"


def menu_interativo(
    metodo: str = "eliminacao gaussiana",
) -> Tuple[Matrix, Vector, List[Point], str, bool]:
    """Menu quando o usuario aperta Run sem argumentos CLI."""
    print("=" * 60)
    print(f"Circuito nodal - {metodo}")
    print("=" * 60)
    print(f"1) [Enter] Default: {ARQUIVO_DEFAULT.name}")
    print(f"          ({ARQUIVO_DEFAULT})")
    print("2) Caminho de um .txt Falstad")
    print("3) Colar netlist (termine com END)")
    print()
    escolha = input("Opcao [1/2/3]: ").strip() or "1"

    if escolha == "1":
        if not ARQUIVO_DEFAULT.is_file():
            raise FileNotFoundError(
                f"Default nao encontrado: {ARQUIVO_DEFAULT}\n"
                "Use a opcao 2 (caminho) ou 3 (colar)."
            )
        print(f"\nUsando circuito padrao do projeto: {ARQUIVO_DEFAULT.name}")
        A, b, nos, origem = carregar_de_arquivo(ARQUIVO_DEFAULT)
        return A, b, nos, origem, True

    if escolha == "2":
        caminho = Path(input("Caminho do .txt: ").strip().strip('"'))
        A, b, nos, origem = carregar_de_arquivo(caminho)
        return A, b, nos, origem, False

    if escolha == "3":
        A, b, nos, origem = carregar_de_paste()
        return A, b, nos, origem, False

    raise ValueError(f"Opcao invalida: {escolha!r} (use 1, 2 ou 3).")


def obter_sistema(
    argv: Optional[Sequence[str]] = None,
    metodo: str = "eliminacao gaussiana",
) -> Tuple[Matrix, Vector, List[Point], str, bool]:
    """
    Retorna A, b, nos_livres, origem, usar_ref_check.
    Sem args CLI -> menu interativo. Com args -> modo nao interativo.
    `metodo` so aparece no titulo do menu e na ajuda do argparse.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Run / python script.py sem args -> menu
    if len(argv) == 0:
        return menu_interativo(metodo)

    parser = argparse.ArgumentParser(
        description=f"Resolve o sistema nodal A x = b ({metodo}) a partir de um export Falstad (.txt)."
    )
    parser.add_argument(
        "arquivo",
        nargs="?",
        default=None,
        help="Caminho do .txt Falstad",
    )
    parser.add_argument(
        "--paste",
        action="store_true",
        help="Ler netlist da entrada padrao ate END",
    )
    args = parser.parse_args(argv)

    if args.paste and args.arquivo:
        parser.error("Use --paste OU um caminho de arquivo, nao ambos.")

    if args.paste:
        A, b, nos, origem = carregar_de_paste()
        return A, b, nos, origem, False

    if args.arquivo:
        caminho = Path(args.arquivo)
        A, b, nos, origem = carregar_de_arquivo(caminho)
        return A, b, nos, origem, False

    return menu_interativo(metodo)


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------

def copiar_matriz(M: Matrix) -> Matrix:
    return [row[:] for row in M]


def copiar_vetor(v: Vector) -> Vector:
    return v[:]


def imprimir_matriz(nome: str, M: Matrix, casas: int = 6) -> None:
    print(f"\n{nome} =")
    for row in M:
        cells = "  ".join(f"{val:{casas + 4}.{casas}f}" for val in row)
        print(f"  [ {cells} ]")


def imprimir_vetor(nome: str, v: Vector, casas: int = 6) -> None:
    cells = ", ".join(f"{val:.{casas}f}" for val in v)
    print(f"{nome} = [ {cells} ]")


def imprimir_secao(titulo: str) -> None:
    print("\n" + "-" * 60)
    print(titulo)
    print("-" * 60)


def imprimir_cabecalho(
    titulo: str, origem: str, nos: List[Point], A: Matrix, b: Vector
) -> None:
    """Titulo da execucao, origem do circuito, ordem dos nos, A e b."""
    print()
    print("=" * 60)
    print(titulo)
    print(f"Circuito: {origem}")
    print(f"Nos livres (ordem de x), reps (x,y): {nos}")
    print("=" * 60)
    imprimir_matriz("A", A)
    imprimir_vetor("\nb", b)


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
# Eliminacao gaussiana + substituicao retroativa
# ---------------------------------------------------------------------------

def eliminacao_gaussiana(A: Matrix, b: Vector) -> Tuple[Vector, int]:
    n = len(A)
    M = copiar_matriz(A)
    y = copiar_vetor(b)
    flops = 0

    for k in range(n - 1):
        pivo = M[k][k]
        if abs(pivo) < 1e-15:
            raise ZeroDivisionError(f"Pivo nulo na coluna {k} (sem pivoteamento).")

        for i in range(k + 1, n):
            fator = M[i][k] / pivo
            flops += 1
            M[i][k] = 0.0
            for j in range(k + 1, n):
                M[i][j] = M[i][j] - fator * M[k][j]
                flops += 2
            y[i] = y[i] - fator * y[k]
            flops += 2

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma = soma + M[i][j] * x[j]
            flops += 2
        x[i] = (y[i] - soma) / M[i][i]
        flops += 2

    return x, flops


def executar_gauss(A: Matrix, b: Vector) -> Tuple[Vector, int]:
    """Resolve por Gauss e imprime x e o custo. Retorna x e flops."""
    imprimir_secao("Eliminacao gaussiana + substituicao retroativa")
    x, flops = eliminacao_gaussiana(A, b)
    imprimir_vetor("x (Gauss)", x)
    print(f"Flops (Gauss, total): {flops}")
    return x, flops


def verificar_referencia(A: Matrix, b: Vector, x: Vector) -> None:
    """Compara A, b e x com a formulacao da Etapa 2 (circuito default)."""
    imprimir_secao("Sanity check vs formulacao_nodal.md (circuito default)")
    A_ref, b_ref = sistema_referencia()
    x_ref, _ = eliminacao_gaussiana(A_ref, b_ref)
    print(f"max |A - A_ref| = {max_abs_diff_matriz(A, A_ref):.3e}")
    print(f"max |b - b_ref| = {max_abs_diff_vetor(b, b_ref):.3e}")
    imprimir_vetor("x (ref Etapa 2)", x_ref)
    print(f"max |x - x_ref| = {max_abs_diff_vetor(x, x_ref):.3e}")


# ===========================================================================
# Etapa 3.2 - Decomposicao LU (Doolittle)
# (origem: fatoracao_LU.py)
# ===========================================================================

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


def substituicao_direta_unitaria(L: Matrix, b: Vector) -> Tuple[Vector, int]:
    """
    Resolve L y = b (forward) com Lii = 1, logo nao ha divisao.

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
    y, flops_fwd = substituicao_direta_unitaria(L, b)
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


def executar_lu(A: Matrix, b: Vector) -> Tuple[Vector, int, int]:
    """
    Fatora, resolve e imprime L, U, x, custos e verificacao.
    Retorna x, flops da fatoracao e flops totais.
    """
    imprimir_secao(
        "Decomposicao LU (Doolittle, sem pivoteamento)\n"
        "A = L U;  L y = b;  U x = y"
    )
    L, U, x, flops_fat, flops_fwd, flops_bwd, flops_total = resolver_lu(A, b)
    imprimir_matriz("L", L)
    imprimir_matriz("U", U)
    imprimir_vetor("\nx (LU)", x)
    print(f"Flops (fatoracao LU): {flops_fat}")
    print(f"Flops (forward Ly=b): {flops_fwd}")
    print(f"Flops (backward Ux=y): {flops_bwd}")
    print(f"Flops (LU, total): {flops_total}")

    imprimir_secao("Verificacao LU")
    print(f"max |A - L U| = {max_abs_diff_matriz(A, multiplicar_LU(L, U)):.3e}")
    print(f"residuo max |A x - b| = {residuo(A, x, b):.3e}")
    return x, flops_fat, flops_total


# ===========================================================================
# Etapa 3.3 - Fatoracao de Cholesky
# (origem: fatoracao_cholesky.py)
# ===========================================================================

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

def decomposicao_cholesky(A: Matrix) -> Tuple[Matrix, int, int]:
    """
    Fatora A = L L^T.

    Etapa: coluna j. A diagonal e a raiz do residual;
    abaixo da diagonal, L[i][j] = (A[i][j] - soma) / L[j][j].
    Convenção de flops: a - b*c conta 2; cada divisao conta 1.
    Os flops das raizes (Newton) ficam num contador separado, para que
    a parte aritmetica seja comparavel com a LU.

    Retorna L, flops aritmeticos (sem raizes) e flops das raizes.
    """
    n = len(A)
    L: Matrix = [[0.0] * n for _ in range(n)]
    flops = 0
    flops_raiz = 0

    for j in range(n):
        s = A[j][j]
        for k in range(j):
            s = s - L[j][k] * L[j][k]
            flops += 2

        L[j][j], flops_sqrt = raiz_quadrada(s)
        flops_raiz += flops_sqrt
        if abs(L[j][j]) < 1e-15:
            raise ZeroDivisionError(f"Pivo nulo na coluna {j} (Cholesky).")

        for i in range(j + 1, n):
            s = A[i][j]
            for k in range(j):
                s = s - L[i][k] * L[j][k]
                flops += 2
            L[i][j] = s / L[j][j]
            flops += 1

    return L, flops, flops_raiz


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
) -> Tuple[Matrix, Vector, Vector, int, int, int, int, int]:
    """
    Encadeia: A = L L^T, L y = b, L^T x = y.

    Retorna L, y, x e os flops (fatoracao sem raizes, raizes, forward,
    backward, total).
    """
    L, flops_fat, flops_raiz = decomposicao_cholesky(A)
    y, flops_fwd = substituicao_direta(L, b)
    x, flops_bwd = substituicao_retroativa_Lt(L, y)
    flops_total = flops_fat + flops_raiz + flops_fwd + flops_bwd
    return L, y, x, flops_fat, flops_raiz, flops_fwd, flops_bwd, flops_total


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
        L, y, x, flops_fat, flops_raiz, flops_fwd, flops_bwd, flops_total
    ) = resolver_cholesky(A, b)
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


# ===========================================================================
# main - executa os tres metodos no mesmo sistema
# (origem: main.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    A, b, nos, origem, check_ref = obter_sistema(argv, metodo="Gauss, LU e Cholesky")
    imprimir_cabecalho(
        "Etapa 3: Gauss, LU e Cholesky no mesmo sistema", origem, nos, A, b
    )

    x_g, flops_g = executar_gauss(A, b)
    x_lu, flops_lu_fat, flops_lu = executar_lu(A, b)
    x_ch, flops_ch, flops_ch_raiz = executar_cholesky(A, b, flops_lu_fat, flops_lu)

    imprimir_secao("Resumo: solucoes e custo computacional")
    imprimir_vetor("x (Gauss)   ", x_g)
    imprimir_vetor("x (LU)      ", x_lu)
    imprimir_vetor("x (Cholesky)", x_ch)
    print(f"max |x_LU - x_Gauss|       = {max_abs_diff_vetor(x_lu, x_g):.3e}")
    print(f"max |x_Cholesky - x_Gauss| = {max_abs_diff_vetor(x_ch, x_g):.3e}")
    print()
    print(f"{'Metodo':<30}{'Flops':>8}")
    print(f"{'Gauss + retroativa':<30}{flops_g:>8}")
    print(f"{'LU (fatoracao + 2 triang.)':<30}{flops_lu:>8}")
    print(f"{'Cholesky (sem raizes)':<30}{flops_ch:>8}")
    print(f"{'Cholesky (com raizes Newton)':<30}{flops_ch_raiz:>8}")

    if check_ref:
        verificar_referencia(A, b, x_g)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, ZeroDivisionError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
    except EOFError:
        print("\nEntrada interrompida.", file=sys.stderr)
        sys.exit(1)


# ===========================================================================
# Saida no console (python atividade01_circuitos.py, Enter = circuito padrao)
# ===========================================================================
# ============================================================
# Circuito nodal - Gauss, LU e Cholesky
# ============================================================
# 1) [Enter] Default: circuito_falstad.txt
#           (circuito_falstad.txt)
# 2) Caminho de um .txt Falstad
# 3) Colar netlist (termine com END)
#
# Opcao [1/2/3]:
# Usando circuito padrao do projeto: circuito_falstad.txt
#
# ============================================================
# Etapa 3: Gauss, LU e Cholesky no mesmo sistema
# Circuito: circuito_falstad.txt
# Nos livres (ordem de x), reps (x,y): [(112, 112), (208, 112), (304, 112), (112, 208), (208, 208)]
# ============================================================
#
# A =
#   [   1.333333   -1.000000    0.000000   -0.333333    0.000000 ]
#   [  -1.000000    1.750000   -0.500000    0.000000   -0.250000 ]
#   [   0.000000   -0.500000    0.700000    0.000000   -0.200000 ]
#   [  -0.333333    0.000000    0.000000    0.500000   -0.166667 ]
#   [   0.000000   -0.250000   -0.200000   -0.166667    0.884524 ]
#
# b = [ 0.000000, 0.000000, 2.000000, 3.000000, 0.000000 ]
#
# ------------------------------------------------------------
# Eliminacao gaussiana + substituicao retroativa
# ------------------------------------------------------------
# x (Gauss) = [ 26.362319, 25.217391, 26.202899, 29.797101, 18.666667 ]
# Flops (Gauss, total): 120
#
# ------------------------------------------------------------
# Decomposicao LU (Doolittle, sem pivoteamento)
# A = L U;  L y = b;  U x = y
# ------------------------------------------------------------
#
# L =
#   [   1.000000    0.000000    0.000000    0.000000    0.000000 ]
#   [  -0.750000    1.000000    0.000000    0.000000    0.000000 ]
#   [   0.000000   -0.500000    1.000000    0.000000    0.000000 ]
#   [  -0.250000   -0.250000   -0.277778    1.000000    0.000000 ]
#   [   0.000000   -0.250000   -0.722222   -1.000000    1.000000 ]
#
# U =
#   [   1.333333   -1.000000    0.000000   -0.333333    0.000000 ]
#   [   0.000000    1.000000   -0.500000   -0.250000   -0.250000 ]
#   [   0.000000    0.000000    0.450000   -0.125000   -0.325000 ]
#   [   0.000000    0.000000    0.000000    0.319444   -0.319444 ]
#   [   0.000000    0.000000    0.000000    0.000000    0.267857 ]
#
# x (LU) = [ 26.362319, 25.217391, 26.202899, 29.797101, 18.666667 ]
# Flops (fatoracao LU): 70
# Flops (forward Ly=b): 24
# Flops (backward Ux=y): 30
# Flops (LU, total): 124
#
# ------------------------------------------------------------
# Verificacao LU
# ------------------------------------------------------------
# max |A - L U| = 2.776e-17
# residuo max |A x - b| = 8.882e-15
#
# ------------------------------------------------------------
# Fatoracao de Cholesky (Banachiewicz)
# A = L L^T;  L y = b;  L^T x = y
# ------------------------------------------------------------
#
# L =
#   [   1.154701    0.000000    0.000000    0.000000    0.000000 ]
#   [  -0.866025    1.000000    0.000000    0.000000    0.000000 ]
#   [   0.000000   -0.500000    0.670820    0.000000    0.000000 ]
#   [  -0.288675   -0.250000   -0.186339    0.565194    0.000000 ]
#   [   0.000000   -0.250000   -0.484481   -0.565194    0.517549 ]
#
# y (Ly=b) = [ 0.000000, 0.000000, 2.981424, 6.290857, 9.660918 ]
# x (Cholesky) = [ 26.362319, 25.217391, 26.202899, 29.797101, 18.666667 ]
# Flops (fatoracao Cholesky, sem raizes): 50
# Flops (raizes por Newton, 5 raizes): 72
# Flops (forward Ly=b): 30
# Flops (backward L^T x=y): 30
# Flops (Cholesky, sem raizes): 110
# Flops (Cholesky, total com raizes): 182
#
# ------------------------------------------------------------
# Verificacao Cholesky
# ------------------------------------------------------------
# max |A - L L^T| = 1.110e-16
# residuo max |A x - b| = 6.217e-15
#
# ------------------------------------------------------------
# Comparacao de custo: Cholesky x LU
# ------------------------------------------------------------
# Fatoracao:  LU = 70  |  Cholesky = 50 (sem raizes)
# Total:      LU = 124  |  Cholesky = 110 (sem raizes)  /  182 (com raizes)
# Cholesky explora a simetria de A: so calcula L, com menos trabalho
# que a LU (tende a metade para n grande). As raizes por Newton sao um custo
# extra, porque aqui a raiz e iterativa (sem math.sqrt).
#
# ------------------------------------------------------------
# Resumo: solucoes e custo computacional
# ------------------------------------------------------------
# x (Gauss)    = [ 26.362319, 25.217391, 26.202899, 29.797101, 18.666667 ]
# x (LU)       = [ 26.362319, 25.217391, 26.202899, 29.797101, 18.666667 ]
# x (Cholesky) = [ 26.362319, 25.217391, 26.202899, 29.797101, 18.666667 ]
# max |x_LU - x_Gauss|       = 0.000e+00
# max |x_Cholesky - x_Gauss| = 1.776e-14
#
# Metodo                           Flops
# Gauss + retroativa                 120
# LU (fatoracao + 2 triang.)         124
# Cholesky (sem raizes)              110
# Cholesky (com raizes Newton)       182
#
# ------------------------------------------------------------
# Sanity check vs formulacao_nodal.md (circuito default)
# ------------------------------------------------------------
# max |A - A_ref| = 0.000e+00
# max |b - b_ref| = 0.000e+00
# x (ref Etapa 2) = [ 26.362319, 25.217391, 26.202899, 29.797101, 18.666667 ]
# max |x - x_ref| = 0.000e+00
