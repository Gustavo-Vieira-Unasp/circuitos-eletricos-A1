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

from fatoracao_cholesky import executar_cholesky
from fatoracao_LU import executar_lu
from solvers_nodal import (
    executar_gauss,
    imprimir_cabecalho,
    imprimir_secao,
    imprimir_vetor,
    max_abs_diff_vetor,
    obter_sistema,
    verificar_referencia,
)


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
