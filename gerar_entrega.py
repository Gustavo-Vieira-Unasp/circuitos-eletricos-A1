"""
Gera o arquivo unico de entrega: entrega/atividade01_circuitos.py.

O PDF pede "o arquivo de codigo fonte em Python (.py)" junto do .txt do
Falstad. O desenvolvimento fica nos modulos separados; este script junta
solvers_nodal.py, fatoracao_LU.py, fatoracao_cholesky.py e main.py num so
arquivo, sem imports entre modulos, e copia circuito_falstad.txt para a
mesma pasta. Por fim roda a entrega e anexa a saida do console como
comentario no fim do arquivo (o PDF pede os outputs). Rodar de novo apos
qualquer mudanca nos modulos:

    python gerar_entrega.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

RAIZ = Path(__file__).resolve().parent
PASTA_ENTREGA = RAIZ / "entrega"
ARQUIVO_ENTREGA = PASTA_ENTREGA / "atividade01_circuitos.py"
CIRCUITO = "circuito_falstad.txt"

# (arquivo, titulo da secao, manter o main? None = modulo sem main)
PARTES = [
    ("solvers_nodal.py", "Leitura do circuito, utilitarios e Etapa 3.1 - Eliminacao gaussiana", False),
    ("fatoracao_LU.py", "Etapa 3.2 - Decomposicao LU (Doolittle)", False),
    ("fatoracao_cholesky.py", "Etapa 3.3 - Fatoracao de Cholesky", False),
    ("custo_computacional.py", "Custo computacional - formulas e reducoes de custo", None),
    ("main.py", "main - executa os tres metodos no mesmo sistema", True),
]

MODULOS_INTERNOS = {"solvers_nodal", "fatoracao_LU", "fatoracao_cholesky", "custo_computacional"}
IMPORTS_PERMITIDOS = {"__future__", "argparse", "sys", "fractions", "pathlib", "typing"}

IMPORTS = """\
from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
"""

CABECALHO = '''\
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

=============================================================================
Custo computacional (flops contados = formulas fechadas; n = 5)
=============================================================================
  Gauss:    eliminacao em A   n(n-1)/2 + n(n-1)(2n-1)/3   = 70
            atualizacao de b  n(n-1)                      = 20
            retroativa        n^2 + n                     = 30   total 120
  LU:       fatoracao         (mesma da eliminacao)       = 70
            direta (Lii = 1)  n^2 - 1                     = 24
            retroativa        n^2 + n                     = 30   total 124
  Cholesky: fatoracao         (2n^3 + 3n^2 - 5n)/6        = 50
            direta            n^2 + n                     = 30
            retroativa        n^2 + n                     = 30   total 110
            + 5 raizes por Newton (3 flops por iteracao)  = 72   total 182

  Ordens: Gauss ~ 2n^3/3, LU ~ 2n^3/3, Cholesky ~ n^3/3 (usa a simetria).
  Com n = 5 as triangulares (~n^2) pesam muito: Cholesky/LU = 0.89;
  para n = 500 a razao ja e 0.505.

  Reducoes implementadas e medidas (ver saida no fim do arquivo):
    - Gauss que pula os zeros de G (esparsidade): 76 flops contra 120.
    - Newton com chute (1 + s)/2 e parada antecipada: 61 contra 72 nas raizes.
  Outras ideias: reusar a fatoracao para um novo b (so as triangulares:
  54 na LU, 60 no Cholesky); LDL^T evita as raizes; sem pivoteamento
  porque G e SPD. __pycache__ guarda o bytecode dos modulos importados e
  economiza so o tempo de compilacao ao iniciar (0 flops); por isso fica
  no .gitignore.

Bibliotecas usadas (nenhuma fatora nem resolve A x = b):
  __future__ (type hints), argparse (linha de comando), sys (stdin/saida),
  pathlib (caminho do .txt), typing (aliases de tipo), fractions (so para
  os literais exatos da matriz de referencia da Etapa 2).
Sem numpy / scipy / math.
"""
'''


def remover_docstring(linhas: List[str]) -> List[str]:
    if not linhas or not linhas[0].startswith('"""'):
        return linhas
    for i in range(1, len(linhas)):
        if linhas[i].rstrip().endswith('"""'):
            return linhas[i + 1:]
    raise ValueError("Docstring do modulo sem fechamento.")


def remover_imports(linhas: List[str], nome: str) -> List[str]:
    saida: List[str] = []
    i = 0
    while i < len(linhas):
        linha = linhas[i]
        m = re.match(r"(?:from|import)\s+([\w.]+)", linha)
        if m:
            modulo = m.group(1).split(".")[0]
            if modulo not in MODULOS_INTERNOS and modulo not in IMPORTS_PERMITIDOS:
                raise ValueError(f"{nome}: import nao permitido: {linha.strip()}")
            if linha.rstrip().endswith("("):
                while not linhas[i].strip().startswith(")"):
                    i += 1
            i += 1
            continue
        saida.append(linha)
        i += 1
    return saida


def cortar_main(linhas: List[str], nome: str) -> List[str]:
    """Remove o bloco do main (banner '# main' + def main + __main__)."""
    for i, linha in enumerate(linhas):
        if linha.startswith("def main("):
            j = i
            while j > 0 and (not linhas[j - 1].strip() or linhas[j - 1].startswith("#")):
                j -= 1
            return linhas[:j]
    raise ValueError(f"{nome}: def main( nao encontrado.")


def aparar(linhas: List[str]) -> List[str]:
    while linhas and not linhas[0].strip():
        linhas = linhas[1:]
    while linhas and not linhas[-1].strip():
        linhas = linhas[:-1]
    return linhas


def gerar() -> str:
    blocos: List[str] = [CABECALHO, IMPORTS]
    nomes_definidos: List[str] = []

    for arquivo, titulo, manter_main in PARTES:
        linhas = (RAIZ / arquivo).read_text(encoding="utf-8").splitlines()
        linhas = remover_docstring(linhas)
        linhas = remover_imports(linhas, arquivo)
        if manter_main is False:
            linhas = cortar_main(linhas, arquivo)
        linhas = aparar(linhas)

        for linha in linhas:
            m = re.match(r"(?:def|class)\s+(\w+)", linha)
            if m:
                nomes_definidos.append(m.group(1))

        banner = "# " + "=" * 75
        blocos.append(f"\n{banner}\n# {titulo}\n# (origem: {arquivo})\n{banner}\n")
        blocos.append("\n".join(linhas) + "\n")

    repetidos = sorted({n for n in nomes_definidos if nomes_definidos.count(n) > 1})
    if repetidos:
        raise ValueError(f"Nomes definidos mais de uma vez: {repetidos}")

    return "\n".join(blocos)


def capturar_saida() -> str:
    """Roda a entrega com Enter (circuito padrao) e devolve o console."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, ARQUIVO_ENTREGA.name],
        input="\n",
        cwd=PASTA_ENTREGA,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"A entrega falhou (exit {proc.returncode}):\n{proc.stderr}")
    return proc.stdout.replace(str(PASTA_ENTREGA) + os.sep, "")


def bloco_saida(saida: str) -> str:
    banner = "# " + "=" * 75
    linhas = [
        "",
        "",
        banner,
        "# Saida no console (python atividade01_circuitos.py, Enter = circuito padrao)",
        banner,
    ]
    linhas += [("# " + linha).rstrip() for linha in saida.rstrip().splitlines()]
    return "\n".join(linhas) + "\n"


def main() -> None:
    PASTA_ENTREGA.mkdir(exist_ok=True)
    codigo = gerar()
    ARQUIVO_ENTREGA.write_text(codigo, encoding="utf-8", newline="\n")
    shutil.copyfile(RAIZ / CIRCUITO, PASTA_ENTREGA / CIRCUITO)
    saida = capturar_saida()
    ARQUIVO_ENTREGA.write_text(codigo + bloco_saida(saida), encoding="utf-8", newline="\n")
    print(f"Gerado: {ARQUIVO_ENTREGA.relative_to(RAIZ)} (com saida do console)")
    print(f"Copiado: {(PASTA_ENTREGA / CIRCUITO).relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
