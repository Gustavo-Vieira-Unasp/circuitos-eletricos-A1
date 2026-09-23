# Diretrizes para agentes

Fonte da atividade: `ATIVIDADE_AVALIATIVA_01_CIRCUITOS_ELETRICOS.pdf`.

## Registro de IA (obrigatório)

- Toda parte feita com IA deve ser registrada em [`REGISTRO_IA.md`](REGISTRO_IA.md).
- Colunas: **Parte | Etapa | O que foi feito | IA / agente**.
- **Não** incluir coluna de data.
- Acrescentar uma nova linha ao final da tabela a cada entrega parcial.

A diretriz compartilhada no remoto é este arquivo + `REGISTRO_IA.md`. Regras em `.cursor/rules/` são só contexto local (gitignored).

## Decisões do projeto

- Análise **nodal** apenas (sistema governante \(\mathbf{A}\mathbf{x}=\mathbf{b}\) via nós).
- Resistores em **ohms pequenos** (ex.: 1…8 Ω), não kΩ.
- Preferir fontes de corrente independentes para \(\mathbf{G}\) SPD (Cholesky).
- Circuito: `circuito_falstad.txt`. Formulação: `formulacao_nodal.md`.

## Solvers

Implementar do zero (sem `scipy.linalg` ou equivalentes): eliminação gaussiana, LU e Cholesky, com contagem de flops.
