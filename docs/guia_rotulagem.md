# Guia de rotulagem — motivo principal da avaliação

**Tarefa:** ler o comentário de uma avaliação de compra (Olist) e atribuir **um único rótulo**: o
**motivo principal** que levou o cliente a escrever aquilo. Leia só o texto do comentário; não use a nota.

Este guia é usado por pessoas (rotulagem do conjunto gold) e, sem alterações de conteúdo, como prompt
do rotulador LLM (rótulos silver).

> Observação sobre os dados: a Olist anonimizou os nomes das lojas com nomes de casas de *Game of
> Thrones* ("lannister", "baratheon", "stark"...). Trate-os como nome de loja.

Todos os exemplos abaixo são comentários reais do dataset, copiados sem correção, e **nenhum faz parte
do conjunto gold**.

---

## Classes

### 1. `atraso_ou_nao_entrega`
Problema de **logística/entrega**: o pedido atrasou, ainda não chegou, nunca chegou, o rastreio diz
"entregue" mas não foi, a carga foi extraviada/roubada, ou o cliente teve que buscar nos Correios
apesar de ter pago a entrega. Vale também quando o pedido chegou, mas a reclamação é sobre a demora.

- "Até o presente momento não recebi meu produto. Gostaria de um posicionamento sobre a entrega."
- "Ainda não recebi o meu pedido, estava programado para o dia 13/12 e até agora nada, hoje já é dia 17/12"
- "Não recebi o produto é consta no rastreamento que já foi entregue, um absurdo."

### 2. `produto_errado_ou_incompleto`
Chegou **algo diferente do que foi comprado** (outro modelo, cor, tamanho, versão) ou **faltou algo**:
um item do pedido, uma unidade, uma peça ou acessório que deveria vir junto.

- "Bom comprei uma capinha Sony xa1 e veio xa1 ultra a película também veio errado"
- "Recebi o produto faltando 02 (duas) luminárias"
- "BOA NOITE, MINHA ENCOMENDA VEIO FALTANDO A PELICULA DE VIDRO,SÓ A CAPA CARTEIRA QUE CHEGOU."

### 3. `defeito_ou_avaria`
O produto certo chegou, mas **danificado ou sem funcionar**: quebrado, trincado, amassado, rasgado,
não liga, parou de funcionar, peça montada errada de fábrica.

- "o produto veio com defeito, não esta ligando"
- "Bom dia recebi o produto mas ao usar percebi que está com defeito nas lâminas, estão desalinhados e o pente montado torto como faço para trocar?"
- "Boa noite recebi a mercadoria em parte pois uma cuba esta quebrada, como devo fazer? fico no prejuízo ou quem e que ira me repor esta quebrada"

### 4. `qualidade_abaixo_do_esperado`
O produto certo chegou e **funciona**, mas o cliente está insatisfeito com ele: material fraco,
acabamento ruim, menor/diferente do que parecia no anúncio, não faz bem o que promete, "não gostei".

- "Achei o produto muito pequeno, e o material fraco.Como os produtos do baratheon são de ótima qualidade deixou a desejar."
- "o material do produto é de qualidade inferior ao esperado."
- "Produto com medida menor do que a informada no site"

### 5. `atendimento_vendedor`
Problema com a **loja/vendedor ou com o processo da compra**, e não com o produto ou o transporte em
si: não responde, não resolve, cancelamento, estorno/reembolso, cobrança indevida, nota fiscal,
falta de estoque depois da compra, dificuldade para trocar ou devolver.

- "Eu cancelei a compra e quero meu estorno"
- "Recebi um e-mail de que não enviariam pois não tem o produto em estoque, mas estornariam o dinheiro.. porém veio a cobrança no meu cartão da primeira parcela, como vamos resolver?"
- "PERCEBI UM ERRO APÓS FECHAR A COMPRA. IMPOSSIVEL CANCELAR, MESMO ANTES DE EMITIR NF. NÃO RESPONDEM EMAIL, CONTATO TELEFONICO NAO FUNCIONA. PÉSSIMA EXPERIENCIA"

### 6. `elogio`
Satisfação com o produto, a entrega ou a loja, **sem nenhum problema concreto** relatado.

- "Recebi antes do prazo de entrega informado e o produto veio devidamente embalado e com NF. Recomendo 100%"
- "Entrega muito antes do prazo. O relógio é bem delicado. Muito satisfeita com a compra!"
- "Muito Bom! Gostei Bastante! Tecido Ótimo! Atendeu Minhas Espectativas! 👏👏👏 Super Recomendo! 👍"

### 7. `outro`
Não há motivo identificável entre as classes acima: comentário neutro ou informativo, ainda não usou
e não opina, reclamação só de preço ou frete caro, sugestão, texto ininteligível.

- "Achei muito caro o custo do frete."
- "Para melhor avaliação só usando e como ainda não usei???"
- "Por se tratar de um produto de simples transporte e ter o valor do frete maior que o valor do produto. E desestimulante a compra"

---

## Regras de desempate

Aplique na ordem; a primeira regra que resolver o caso decide.

1. **Problema concreto vence elogio.** Se o texto relata um problema de uma das classes 1–5, o rótulo
   é o problema, mesmo com elogios em volta ou com o problema já resolvido.
   - "o produto veio errado mas o fornecedor resolveu o problema de maneira rápida e gentil." → `produto_errado_ou_incompleto`
   - "Bem menor Do que eu esperava mas muito boa" → `qualidade_abaixo_do_esperado`
   - Exceção: se o próprio cliente diz que o problema não afetou nada, é `elogio`.
     "A caixa veio com leves amassados e arranhões, mas nada que tenha chegado ao produto." → `elogio`

2. **Causa vence consequência.** Quando o cliente reclama do atendimento *por causa* de outro problema
   (não chegou, veio errado, quebrou) e ninguém responde, rotule o problema de origem.
   - "O produto veio errado e até agora não tive nenhuma resposta da loja" → `produto_errado_ou_incompleto`
   - "Não recomendo. Não recebi. Quero reembolso." → `atraso_ou_nao_entrega`
   - Use `atendimento_vendedor` quando a origem é a própria loja/processo (cancelou, sem estoque,
     cobrou errado, não emitiu NF, não aceita troca) ou quando não há outro problema descrito.

3. **Vários problemas de produto/entrega: vale o que o cliente tem em mãos.** Prioridade:
   `defeito_ou_avaria` > `produto_errado_ou_incompleto` > `atraso_ou_nao_entrega` > `qualidade_abaixo_do_esperado`.
   - "O produto chegou atrasado..e veio quebrado..e até agora não consegui contato com vcs para trocar." → `defeito_ou_avaria`
   - "Produto chegou atrasado, mal embalado e faltando um Ilhós." → `produto_errado_ou_incompleto`

4. **Faltou parte do pedido = incompleto, não atraso.** Se chegou parte do pedido e falta um item,
   unidade ou peça, é `produto_errado_ou_incompleto`. Se não chegou nada, é `atraso_ou_nao_entrega`.
   - "Comprei dois produtos , chegou apenas um, quero meu boné" → `produto_errado_ou_incompleto`

5. **Quebrado/não funciona = defeito; funciona mas decepciona = qualidade.** Diferença em relação ao
   anúncio em *medida, material ou aparência* é `qualidade_abaixo_do_esperado`; *outro modelo, cor ou
   versão* é `produto_errado_ou_incompleto`.
   - "O farol estava como preto e veio cromado." → `produto_errado_ou_incompleto`
   - "Produto com funcionamento correto, porem qualidade do material muito baixa." → `qualidade_abaixo_do_esperado`

6. **Preço e frete não são classe de problema.** Se a única crítica é preço/frete caro e o resto é
   elogio, é `elogio`; se o comentário é só essa crítica, é `outro`.
   - "Produto chegou bem embalado e no prazo. Gostei, só não ganhou 5 estrelas porque o frete é caro." → `elogio`
   - Cobrança de frete indevida (cobrou o que era grátis) é processo da loja → `atendimento_vendedor`.

7. **Ter que buscar nos Correios é problema de entrega.**
   - "Paguei o frete mas tive que buscar nós Correios. Insatisfeita." → `atraso_ou_nao_entrega`

8. **"Ainda não usei"**: com algo positivo (chegou rápido, bem embalado, bonito) é `elogio`; sozinho
   ou só com dúvida é `outro`.
   - "Tudo ok, a entrega foi rápida comparando ao prazo informado. Ainda não testei o produto mas está tudo ok!" → `elogio`

9. **Perguntas**: se a pergunta revela um problema, rotule o problema ("Cadê o último???" após faltar
   uma almofada → `produto_errado_ou_incompleto`); se não revela, é `outro`.

10. **Elogios curtos e genéricos** ("Muito bom", "Ótimo", "Recomendo", "Ok") são `elogio`. Texto que
    não dá para entender é `outro`.

11. **Na dúvida real entre duas classes**, escolha a que um atendente precisaria acionar primeiro para
    resolver o caso do cliente, e marque confiança 1 na planilha.
