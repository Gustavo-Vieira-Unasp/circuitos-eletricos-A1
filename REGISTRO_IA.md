# Registro de IA

Registro de qual IA / agente produziu cada parte do trabalho.

| Parte | Etapa | O que foi feito | IA / agente |
|-------|-------|-----------------|-------------|
| Circuito Falstad (pathway + 5 malhas) | 1 (versão anterior) | Export inicial com kΩ e análise por malhas | Cursor Auto (Composer) |
| Retune Falstad nodal | 1 | Resistores 1…8 Ω; fontes de corrente 3 A / 2 A; 5 nós livres; `circuito_falstad.txt` | Cursor Auto (Composer) |
| Formulação \(\mathbf{G}\mathbf{v}=\mathbf{i}\) | 2 | KCLs, matriz 5×5 \(\mathbf{G}\), vetor \(\mathbf{b}\); `formulacao_nodal.md` (malhas removidas) | Cursor Auto (Composer) |
| Regra local Cursor | — | Atualização nodal-only / Ω / registro de IA em `.cursor/rules/` | Cursor Auto (Composer) |
| Diretriz de IA no remoto | — | `AGENTS.md` + `REGISTRO_IA.md` sem coluna de data | Cursor Auto (Composer) |

Convenção: novas linhas são **acrescentadas** ao final da tabela a cada entrega parcial (sem coluna de data).
