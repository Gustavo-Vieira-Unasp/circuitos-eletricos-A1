# Diretrizes para agentes

Fonte da atividade: `ATIVIDADE_AVALIATIVA_01_CIRCUITOS_ELETRICOS.pdf`.

## Registro de IA (obrigatório)

- Toda parte feita com IA deve ser registrada em [`REGISTRO_IA.md`](REGISTRO_IA.md).
- Colunas: **Parte | Etapa | O que foi feito | IA / agente**.
- **Não** incluir coluna de data.
- Acrescentar uma nova linha ao final da tabela a cada entrega parcial.

A diretriz compartilhada no remoto é este arquivo + `REGISTRO_IA.md`. Regras em `.cursor/rules/` são só contexto local (gitignored).

## Decisões do projeto

- Análise **nodal** apenas (sistema governante `A x = b` via nós).
- Resistores em **ohms pequenos** (ex.: 1…8 Ω), não kΩ.
- Preferir fontes de corrente independentes para `G` SPD (Cholesky).
- Documentar `G` / `A` com **frações** (não decimais) em `formulacao_nodal.md`.
- Circuito: `circuito_falstad.txt`. Formulação: `formulacao_nodal.md`.

## Solvers

Etapa 3: eliminação gaussiana em `solvers_nodal.py`; decomposição LU (Doolittle, sem pivoteamento) em `fatoracao_LU.py`; Cholesky (`A = L L^T`, raiz por Newton, sem `math`) em `fatoracao_cholesky.py`. Os três usam loops manuais `+ - * /` com contagem de flops.

`main.py` lê o circuito uma vez e roda os três métodos em sequência, com resumo de flops. `gerar_entrega.py` junta os três módulos + `main.py` num arquivo único, `entrega/atividade01_circuitos.py` (sem imports entre módulos), e copia `circuito_falstad.txt` para `entrega/`. Rodar `python gerar_entrega.py` após mudar qualquer módulo. O e-mail ao professor leva o conteúdo de `entrega/` (o `.py` único + o `.txt` do Falstad).

Bibliotecas em uso (nenhuma resolve `A x = b`):

- `__future__.annotations` — adia avaliação de type hints
- `argparse` — CLI (`--paste`, caminho do `.txt`)
- `sys` — stdin/stderr e exit
- `pathlib.Path` — caminhos do export Falstad
- `typing` — aliases (`Matrix`, `Vector`, …); sem matemática em runtime
- `fractions.Fraction` — literais exatos só na matriz de referência (Etapa 2); vira `float` antes do Gauss

**Proibido:** `numpy` / `scipy` / `math` (ou equivalentes) para fatorar ou resolver o sistema.
