"""
Etapa 3 — eliminacao gaussiana para o sistema nodal A x = b.

Entrada do circuito Falstad (.txt):
  - Run sem argumentos: menu interativo (default = circuito_falstad.txt)
  - python solvers_nodal.py caminho/arquivo.txt
  - python solvers_nodal.py --paste

Bibliotecas em uso (nenhuma resolve Ax=b):
  - __future__.annotations — adia avaliacao de type hints (typing mais limpo)
  - argparse — le flags/caminhos da linha de comando (--paste, arquivo.txt)
  - sys — stdin/stderr e saida do processo
  - pathlib.Path — caminhos de arquivo do .txt Falstad
  - typing — aliases de tipo (Matrix, Vector, ...); sem matematica em runtime
  - fractions.Fraction — literais exatos so na matriz de referencia da Etapa 2
    (4/3 etc.); convertido para float antes da eliminacao gaussiana

Proibido para fatorar/resolver: numpy / scipy / math (ou equivalentes).
A eliminacao gaussiana e 100% loops manuais com + - * /.
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def menu_interativo() -> Tuple[Matrix, Vector, List[Point], str, bool]:
    """Menu quando o usuario aperta Run sem argumentos CLI."""
    print("=" * 60)
    print("Circuito nodal — eliminacao gaussiana")
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


def obter_sistema(argv: Optional[Sequence[str]] = None) -> Tuple[Matrix, Vector, List[Point], str, bool]:
    """
    Retorna A, b, nos_livres, origem, usar_ref_check.
    Sem args CLI -> menu interativo. Com args -> modo nao interativo.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Run / python script.py sem args -> menu
    if len(argv) == 0:
        return menu_interativo()

    parser = argparse.ArgumentParser(
        description="Resolve o sistema nodal (Gauss) a partir de um export Falstad (.txt)."
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

    return menu_interativo()


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
            for j in range(k, n):
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    A, b, nos, origem, check_ref = obter_sistema(argv)

    print()
    print("=" * 60)
    print("Etapa 3: Eliminacao gaussiana")
    print(f"Circuito: {origem}")
    print(f"Nos livres (ordem de x), reps (x,y): {nos}")
    print("=" * 60)

    imprimir_matriz("A", A)
    imprimir_vetor("\nb", b)

    print("\n" + "-" * 60)
    print("Eliminacao gaussiana + substituicao retroativa")
    print("-" * 60)
    x_g, flops_g = eliminacao_gaussiana(A, b)
    imprimir_vetor("x (Gauss)", x_g)
    print(f"Flops (Gauss, total): {flops_g}")

    if check_ref:
        print("\n" + "-" * 60)
        print("Sanity check vs formulacao_nodal.md (circuito default)")
        print("-" * 60)
        A_ref, b_ref = sistema_referencia()
        x_ref, _ = eliminacao_gaussiana(A_ref, b_ref)
        dif_ref = max(abs(x_g[i] - x_ref[i]) for i in range(len(x_g)))
        imprimir_vetor("x (ref Etapa 2)", x_ref)
        print(f"max |x - x_ref| = {dif_ref:.3e}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, ZeroDivisionError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
    except EOFError:
        print("\nEntrada interrompida.", file=sys.stderr)
        sys.exit(1)
