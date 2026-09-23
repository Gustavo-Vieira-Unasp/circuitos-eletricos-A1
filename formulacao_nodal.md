# Formulação nodal — circuito Falstad (5×5)

**Etapa 2** da Atividade Avaliativa 01.  
Circuito: [`circuito_falstad.txt`](circuito_falstad.txt)  
Método: **análise nodal** (apenas nós). Sistema `A x = b` com `A = G` (matriz de condutâncias).

## Etapa 1 (resumo)

Mantida a ponte com diagonal do pathway; alterações:

- Resistores em **ohms pequenos**: `Rk = k Ω` para `k = 1 … 8` (sem kΩ)
- Fontes de tensão trocadas por **fontes de corrente independentes** (para `G` ser SPD / Cholesky)
- Terra no trilho inferior; fio direito mantém MR no terra → **5 nós livres**

## Nós independentes

| Índice | Rótulo | Coordenada Falstad |
|--------|--------|--------------------|
| 1 | TL | (112,112) |
| 2 | TM | (208,112) |
| 3 | TR | (304,112) |
| 4 | ML | (112,208) |
| 5 | C | (208,208) |
| 0 | Terra | trilho y=304 e (304,208) |

Desconhecidos `x`:

|   |   |
|---|---|
| v1 | v_TL |
| v2 | v_TM |
| v3 | v_TR |
| v4 | v_ML |
| v5 | v_C |

## Componentes

| Símbolo | Valor | Ligação |
|---------|-------|---------|
| R1 | 1 Ω | TL–TM |
| R2 | 2 Ω | TM–TR |
| R3 | 3 Ω | TL–ML |
| R4 | 4 Ω | TM–C |
| R5 | 5 Ω | TR–C (diagonal) |
| R6 | 6 Ω | ML–C |
| R7 | 7 Ω | C–terra (via MR) |
| R8 | 8 Ω | C–terra |
| Is1 | 3 A | terra → ML (entra no nó 4) |
| Is2 | 2 A | terra → TR (entra no nó 3) |

Condutâncias `Yk = 1/Rk`:

```
Y1 = 1
Y2 = 1/2
Y3 = 1/3
Y4 = 1/4
Y5 = 1/5
Y6 = 1/6
Y7 = 1/7
Y8 = 1/8
```

## KCL em cada nó

Corrente saindo pelo resistor entre nós `i` e `j`: `Y * (vi - vj)`.  
Fontes no lado direito = correntes **injetadas** no nó.

**Nó 1 (TL):**

```
Y1*(v1 - v2) + Y3*(v1 - v4) = 0
```

**Nó 2 (TM):**

```
Y1*(v2 - v1) + Y2*(v2 - v3) + Y4*(v2 - v5) = 0
```

**Nó 3 (TR):**

```
Y2*(v3 - v2) + Y5*(v3 - v5) = Is2 = 2
```

**Nó 4 (ML):**

```
Y3*(v4 - v1) + Y6*(v4 - v5) = Is1 = 3
```

**Nó 5 (C):**

```
Y4*(v5 - v2) + Y5*(v5 - v3) + Y6*(v5 - v4) + Y7*v5 + Y8*v5 = 0
```

## Matriz de condutâncias G

Montagem: `Gii = soma das Y no nó i`; `Gij = -Yij` se i e j estão ligados.

### Forma simbólica

|   | v1 | v2 | v3 | v4 | v5 |
|---|----|----|----|----|----|
| 1 | Y1+Y3 | -Y1 | 0 | -Y3 | 0 |
| 2 | -Y1 | Y1+Y2+Y4 | -Y2 | 0 | -Y4 |
| 3 | 0 | -Y2 | Y2+Y5 | 0 | -Y5 |
| 4 | -Y3 | 0 | 0 | Y3+Y6 | -Y6 |
| 5 | 0 | -Y4 | -Y5 | -Y6 | Y4+Y5+Y6+Y7+Y8 |

### Valores numéricos (frações) — A = G

Fonte de verdade do sistema (sem decimais). Na Etapa 3 os solvers podem converter para float em runtime.

|   | v1 | v2 | v3 | v4 | v5 |
|---|----|----|----|----|----|
| 1 | 4/3 | -1 | 0 | -1/3 | 0 |
| 2 | -1 | 7/4 | -1/2 | 0 | -1/4 |
| 3 | 0 | -1/2 | 7/10 | 0 | -1/5 |
| 4 | -1/3 | 0 | 0 | 1/2 | -1/6 |
| 5 | 0 | -1/4 | -1/5 | -1/6 | 743/840 |

### Vetor b

| nó | bi |
|----|-----|
| 1 | 0 |
| 2 | 0 |
| 3 | 2 |
| 4 | 3 |
| 5 | 0 |

```
b = [0, 0, 2, 3, 0]^T
```

Sistema governante:

```
G x = b
A x = b    (com A = G)
```

## SPD (Cholesky)

- `G` é simétrica por construção.
- Rede resistiva conexa com referência a terra ⇒ `G` **definida positiva**.
- Autovalores (verificação): todos > 0 (λ_min ≈ 0,041).

Cholesky `G = L L^T` é aplicável sem retocar o circuito.

## Verificação Etapas 1 e 2

Auditoria independente a partir de `circuito_falstad.txt` (aritmética exata com frações):

| Check | Resultado |
|-------|-----------|
| Netlist: R1…R8 = 1…8 Ω; Is1 = 3 A → ML; Is2 = 2 A → TR; MR aterrado | OK |
| Nós livres = 5 (TL, TM, TR, ML, C) | OK |
| `G` documentada (frações) = matriz remontada ramo a ramo | match exato |
| `G` simétrica | OK |
| SPD (todos os autovalores > 0) | OK (λ_min ≈ 0,041) |
| Resíduo de `G v = b` | ~1e-15 |

**Conclusão:** Etapas 1 e 2 estão corretas; `A = G` e `b` acima são a fonte de verdade para a Etapa 3.

## Importar no Falstad

1. [falstad.com/circuit](https://www.falstad.com/circuit/circuitjs.html)
2. **File → Import from Text…**
3. Colar [`circuito_falstad.txt`](circuito_falstad.txt)

## Próximo passo

**Etapa 3:** implementar no Python (do zero) eliminação gaussiana, LU e Cholesky sobre este `A`, `b`, com contagem de flops.
