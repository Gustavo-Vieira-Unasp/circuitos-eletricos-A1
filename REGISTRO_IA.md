# Registro de IA

Registro de qual IA / agente produziu cada parte do trabalho.

| Parte | Etapa | O que foi feito | IA / agente |
|-------|-------|-----------------|-------------|
| Circuito Falstad (pathway + 5 malhas) | 1 (versão anterior) | Export inicial com kΩ e análise por malhas | Cursor Auto (Composer) |
| Retune Falstad nodal | 1 | Resistores 1…8 Ω; fontes de corrente 3 A / 2 A; 5 nós livres; `circuito_falstad.txt` | Cursor Auto (Composer) |
| Formulação G v = i | 2 | KCLs, matriz 5×5 G, vetor b; `formulacao_nodal.md` (malhas removidas) | Cursor Auto (Composer) |
| Regra local Cursor | — | Atualização nodal-only / Ω / registro de IA em `.cursor/rules/` | Cursor Auto (Composer) |
| Diretriz de IA no remoto | — | `AGENTS.md` + `REGISTRO_IA.md` sem coluna de data | Cursor Auto (Composer) |
| Formulação legível no preview | 2 | Matrizes/equações em tabelas Markdown (sem LaTeX) em `formulacao_nodal.md` | Cursor Auto (Composer) |
| Verificação + só frações | 1–2 | Remoção de decimais; checklist de auditoria Etapas 1–2 em `formulacao_nodal.md` | Cursor Auto (Composer) |
| Solvers Gauss | 3 | `solvers_nodal.py`: eliminação gaussiana com flops (sem LU) | Cursor Auto (Composer) |
| Entrada Falstad .txt | 3 | Parser r/i/w/g; default / caminho / `--paste` em `solvers_nodal.py` | Cursor Auto (Composer) |
| Menu Run + so Gauss | 3 | Remoção de LU; menu interativo com default `circuito_falstad.txt` | Cursor Auto (Composer) |
| Desc. soft das libs | 3 | Blurbs do que cada import faz (I/O/typing/ref); nenhuma resolve o sistema | Cursor Auto (Composer) |

Convenção: novas linhas são **acrescentadas** ao final da tabela a cada entrega parcial (sem coluna de data).
