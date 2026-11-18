# Fase 4 — avaliação no teste de referência
400 comentários (80 por nota), rótulos de referência do **gpt-6.1-sol** com o guia de rotulagem (**sem validação humana**). IC 95%: bootstrap pareado, 2000 reamostragens, seed 42.
## Métricas
|  | Baseline TF-IDF + LR | BERTimbau | Tucano2 1.5B QLoRA |
|---|---|---|---|
| F1 macro | 0.760 | 0.807 | 0.822 |
| IC 95% F1 macro | 0.703–0.808 | 0.752–0.848 | 0.769–0.865 |
| Acurácia | 0.812 | 0.855 | 0.875 |
| F1 atraso (n=81) | 0.830 | 0.873 | 0.916 |
| F1 errado/incompleto (n=69) | 0.837 | 0.844 | 0.869 |
| F1 defeito (n=32) | 0.769 | 0.814 | 0.833 |
| F1 qualidade (n=49) | 0.739 | 0.792 | 0.804 |
| F1 atendimento (n=17) | 0.743 | 0.812 | 0.812 |
| F1 elogio (n=130) | 0.880 | 0.940 | 0.946 |
| F1 outro (n=22) | 0.524 | 0.578 | 0.571 |
| Erros (de 400) | 75 | 58 | 50 |
| Confiança média nos erros | 0.54 | 0.90 | 0.89 |
| Erros com confiança ≥ 0,95 | 1 | 33 | 29 |
| ECE (10 faixas) | 0.060 | 0.113 | 0.100 |
| Latência GPU, mediana (ms/exemplo) | — | 7.3 | 99.0 |
| Latência GPU, p95 (ms) | — | 8.2 | 105.2 |
| Latência CPU, mediana (ms/exemplo) | 2.2 | 61.2 | 963.0 |
| Latência CPU, p95 (ms) | 2.6 | 89.7 | 1272.6 |
| Tamanho em disco (MB) | 2 | 416 | 2947 |

Latência e tamanho medidos no mesmo hardware — GPU: Tesla T4; CPU: Intel(R) Xeon(R) CPU @ 2.00GHz (4 CPUs lógicas) (2 threads); lote de 1, tokenização incluída, cada chamada cronometrada; mediana e p95 em ms; 400 textos após 10 de aquecimento. Tucano2: GPU em 4 bits + adaptador; CPU com adaptador mesclado em base fp32. Tamanho do Tucano2 = base (2880 MB) + adaptador (67 MB). BERTimbau em fp32 nos dois dispositivos.
Fidelidade do Tucano2 mesclado em CPU: concordância de 95.2% com o modelo em 4 bits; F1 macro 0.811 (CPU) vs 0.822 (GPU 4 bits), ambos medidos nesta mesma sessão.

## Diferenças de F1 macro (bootstrap pareado)
| Comparação | Diferença | IC 95% | P(diferença > 0) |
|---|---|---|---|
| BERTimbau − Baseline | +0.047 | -0.009 a +0.105 | 94.8% |
| Tucano2 − Baseline | +0.061 | +0.014 a +0.108 | 99.4% |
| Tucano2 − BERTimbau | +0.014 | -0.034 a +0.062 | 73.0% |

## Matrizes de confusão (linhas = referência, colunas = predição)

**Baseline TF-IDF + LR**

| ref \ pred | atraso | errado/incompleto | defeito | qualidade | atendimento | elogio | outro |
|---|---|---|---|---|---|---|---|
| atraso | 66 | 1 | 0 | 0 | 3 | 7 | 4 |
| errado/incompleto | 4 | 59 | 2 | 2 | 0 | 2 | 0 |
| defeito | 2 | 1 | 25 | 2 | 1 | 0 | 1 |
| qualidade | 0 | 6 | 3 | 34 | 0 | 6 | 0 |
| atendimento | 2 | 0 | 0 | 0 | 13 | 0 | 2 |
| elogio | 2 | 2 | 2 | 4 | 1 | 117 | 2 |
| outro | 2 | 3 | 1 | 1 | 0 | 4 | 11 |

**BERTimbau**

| ref \ pred | atraso | errado/incompleto | defeito | qualidade | atendimento | elogio | outro |
|---|---|---|---|---|---|---|---|
| atraso | 72 | 4 | 0 | 0 | 0 | 1 | 4 |
| errado/incompleto | 3 | 62 | 2 | 2 | 0 | 0 | 0 |
| defeito | 0 | 2 | 24 | 5 | 0 | 1 | 0 |
| qualidade | 0 | 6 | 1 | 40 | 1 | 0 | 1 |
| atendimento | 3 | 1 | 0 | 0 | 13 | 0 | 0 |
| elogio | 2 | 1 | 0 | 4 | 0 | 118 | 5 |
| outro | 4 | 2 | 0 | 1 | 1 | 1 | 13 |

**Tucano2 1.5B QLoRA**

| ref \ pred | atraso | errado/incompleto | defeito | qualidade | atendimento | elogio | outro |
|---|---|---|---|---|---|---|---|
| atraso | 76 | 2 | 0 | 0 | 0 | 1 | 2 |
| errado/incompleto | 2 | 63 | 0 | 2 | 1 | 0 | 1 |
| defeito | 0 | 2 | 25 | 3 | 0 | 1 | 1 |
| qualidade | 0 | 6 | 3 | 39 | 0 | 1 | 0 |
| atendimento | 1 | 0 | 0 | 0 | 13 | 0 | 3 |
| elogio | 3 | 1 | 0 | 3 | 0 | 122 | 1 |
| outro | 3 | 2 | 0 | 1 | 1 | 3 | 12 |

## Discordâncias

Os três modelos discordam entre si em 85 dos 400 itens; os três erram juntos (em relação à referência) em 28. Lista completa em `results/fase4/discordancias.csv`; exemplos comentados em `reports/fase4_exemplos_discordancia.md`.
