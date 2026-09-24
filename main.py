"""
Etapa 3 — executa os tres metodos sobre o mesmo sistema nodal A x = b.

Le o circuito uma vez (mesmo menu dos outros scripts) e roda, em ordem:
  1. Eliminacao gaussiana (solvers_nodal.py)
  2. Decomposicao LU (fatoracao_LU.py)
  3. Fatoracao de Cholesky (fatoracao_cholesky.py)
No fim imprime um resumo com os x de cada metodo e a tabela de flops.

Entrada: python main.py | python main.py arquivo.txt | python main.py --paste
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from custo_computacional import (
    eliminacao_gaussiana_esparsa,
    imprimir_comparacao_assintotica,
    raiz_quadrada_chute,
)
from fatoracao_cholesky import decomposicao_cholesky, executar_cholesky
from fatoracao_LU import executar_lu
from solvers_nodal import (
    Matrix,
    Vector,
    executar_gauss,
    imprimir_cabecalho,
    imprimir_secao,
    imprimir_vetor,
    max_abs_diff_matriz,
    max_abs_diff_vetor,
    obter_sistema,
    verificar_referencia,
)


def secao_custo(A: Matrix, b: Vector, x_g: Vector, flops_g: int, flops_lu: int,
                flops_ch: int) -> None:
    """Comparacao assintotica e formas de reduzir o custo (medidas e explicadas)."""
    n = len(A)

    imprimir_secao("Custo computacional: comparacao assintotica (formulas)")
    imprimir_comparacao_assintotica()
    print(f"  Para n = {n} as substituicoes triangulares (~n^2) ainda pesam muito,")
    print(f"  por isso Cholesky fica {flops_ch} contra {flops_lu} da LU, e nao metade.")

    imprimir_secao("Como reduzir o custo - implementado e medido")
    x_esp, flops_esp = eliminacao_gaussiana_esparsa(A, b)
    zeros = sum(1 for linha in A for v in linha if v == 0.0)
    print(f"1) Gauss explorando os zeros de G ({zeros} de {n * n} entradas sao zero):")
    print(f"   flops {flops_esp} contra {flops_g} "
          f"({flops_g - flops_esp} a menos, {100 * (flops_g - flops_esp) / flops_g:.0f}%)")
    print(f"   max |x_esparso - x_Gauss| = {max_abs_diff_vetor(x_esp, x_g):.3e}")

    L_orig, _, raizes_orig = decomposicao_cholesky(A)
    L_chute, _, raizes_chute = decomposicao_cholesky(A, raiz=raiz_quadrada_chute)
    r_orig, r_chute = sum(raizes_orig), sum(raizes_chute)
    print("2) Raiz por Newton com chute (1 + s)/2 e parada antecipada:")
    print(f"   flops por raiz {raizes_chute} = {r_chute} contra "
          f"{raizes_orig} = {r_orig} ({r_orig - r_chute} a menos)")
    print(f"   Cholesky total com raizes: {flops_ch + r_chute} contra {flops_ch + r_orig}")
    print(f"   max |L_chute - L| = {max_abs_diff_matriz(L_chute, L_orig):.3e}")

    imprimir_secao("Como reduzir o custo - outras ideias")
    print("- Reusar a fatoracao para um novo b (outra configuracao de fontes):")
    print(f"  LU refaz so as triangulares ({2 * n * n + n - 1} flops) e Cholesky "
          f"{2 * (n * n + n)}, contra {flops_g} de um Gauss completo.")
    print("- Simetria: Cholesky calcula so L (metade da matriz), ~n^3/3 contra ~2n^3/3.")
    print("- LDL^T: mesma ordem do Cholesky, sem nenhuma raiz quadrada.")
    print("- As n raizes custam O(n); para n grande somem perto de n^3/3.")
    print("- Pivoteamento nao e necessario: G e SPD, os pivos ja sao positivos.")
    print("- __pycache__: guarda o bytecode compilado dos modulos importados, entao")
    print("  as execucoes seguintes nao recompilam o codigo. Economiza milissegundos")
    print("  de inicializacao e 0 flops; o arquivo unico rodado como script nao e")
    print("  cacheado. A pasta fica no .gitignore por ser gerada automaticamente.")


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

    secao_custo(A, b, x_g, flops_g, flops_lu, flops_ch)

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
