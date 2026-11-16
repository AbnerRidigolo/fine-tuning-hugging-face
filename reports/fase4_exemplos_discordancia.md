# Fase 4 — 5 exemplos reais em que os modelos discordam

Comentários do teste de referência (texto original, sem correção). "Referência" é o rótulo do gpt-6.1-sol
com o guia de rotulagem (não há rótulo humano). Entre parênteses, a probabilidade que o modelo atribuiu à
classe prevista. Os números vêm de `results/fase4/discordancias.csv`; as explicações são interpretações
qualitativas, não medidas.

Os três modelos discordam entre si em 85 dos 400 itens.

---

### 1. G002 — elogio em volta de um problema
> "A loja postou bem rápido, embalagem simples, mas a encomenda estava bem protegido dentro da caixa. O correio atrasou a entrega."

| Referência | TF-IDF + LR | BERTimbau | Tucano2 QLoRA |
|---|---|---|---|
| atraso | **elogio** | atraso (1,00) | atraso (1,00) |

O TF-IDF soma palavras: "bem rápido", "bem protegido" pesam mais que a única frase do problema, no fim.
Os dois modelos contextuais aplicam a regra 1 do guia (problema concreto vence elogio). É o erro típico
do baseline: 7 dos 81 atrasos da referência viraram `elogio` no TF-IDF, contra 1 em cada modelo contextual.

### 2. G342 — expressão idiomática
> "produto chegou pela metade"

| Referência | TF-IDF + LR | BERTimbau | Tucano2 QLoRA |
|---|---|---|---|
| produto_errado_ou_incompleto | **elogio** | **atraso** (0,97) | produto_errado_ou_incompleto (1,00) |

"Pela metade" significa incompleto (regra 4). O BERTimbau provavelmente se apoiou em "produto chegou",
muito frequente em comentários sobre entrega, e errou com alta confiança. O Tucano2, pré-treinado em muito
mais texto em português, acertou o sentido da expressão.

### 3. G027 — "demorou, mas dentro do prazo" (regra 7)
> "Demora absurda para entrega, dentro do prazo especificado por eles, mas ainda sim muito demorado!"

| Referência | TF-IDF + LR | BERTimbau | Tucano2 QLoRA |
|---|---|---|---|
| outro | **atraso** | outro (0,84) | **atraso** (0,86) |

Pela regra 7, demora que ficou dentro do prazo não é atraso. A pista decisiva é uma oração no meio do
texto ("dentro do prazo especificado"), cercada de palavras fortes de reclamação. Aqui o BERTimbau acertou
e o Tucano2 errou: o modelo maior não ganha em todos os casos.

### 4. G079 — defeito, qualidade ou produto errado?
> "A parte da capa onde fica os botões de volume e liga/desliga acionam mais de um botão ao mesmo tempo."

| Referência | TF-IDF + LR | BERTimbau | Tucano2 QLoRA |
|---|---|---|---|
| defeito_ou_avaria | defeito_ou_avaria | **qualidade_abaixo_do_esperado** (0,68) | **produto_errado_ou_incompleto** (0,97) |

Cada modelo deu uma classe diferente. O texto descreve algo que não funciona direito (defeito, regra 5),
mas sem palavras como "quebrado" ou "não funciona". É a fronteira mais confusa do problema: nas matrizes de
confusão, `qualidade` → `errado/incompleto` aparece 6 vezes em cada um dos três modelos, e foi também a
principal fonte de discordância entre os dois rotuladores LLM na Fase 1.

### 5. G137 — "ainda não usei" (regra 8), com excesso de confiança
> "ainda não usei mais recomendo"

| Referência | TF-IDF + LR | BERTimbau | Tucano2 QLoRA |
|---|---|---|---|
| elogio | elogio | elogio (0,59) | **outro** (1,00) |

Pela regra 8, "ainda não usei" acompanhado de algo positivo ("recomendo") é `elogio`. O Tucano2 errou com
probabilidade 1,00. Não é um caso isolado, e não é só do Tucano2 — os dois modelos contextuais são
confiantes demais quando erram:

| | TF-IDF + LR | BERTimbau | Tucano2 QLoRA |
|---|---|---|---|
| Erros (de 400) | 75 | 58 | 50 |
| Confiança média nos erros | 0,54 | 0,90 | 0,89 |
| Erros com confiança ≥ 0,95 | 1 | 33 | 29 |
| Erro de calibração (ECE, 10 faixas) | 0,060 | 0,113 | 0,100 |

Para o Space, a "confiança" exibida precisa vir com essa ressalva; calibrar com *temperature scaling* na
validação é um próximo passo natural.
