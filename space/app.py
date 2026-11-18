"""Space Gradio: classifica o motivo principal de uma avaliação de e-commerce em português.

Usa o BERTimbau ajustado (repositório em MODELO_ID, variável do Space). Roda em CPU.
"""
import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELO_ID = os.environ.get("MODELO_ID", "")
NOMES = {
    "atraso_ou_nao_entrega": "Atraso ou não entrega",
    "produto_errado_ou_incompleto": "Produto errado ou incompleto",
    "defeito_ou_avaria": "Defeito ou avaria",
    "qualidade_abaixo_do_esperado": "Qualidade abaixo do esperado",
    "atendimento_vendedor": "Atendimento do vendedor",
    "elogio": "Elogio",
    "outro": "Outro",
}

tok = AutoTokenizer.from_pretrained(MODELO_ID)
modelo = AutoModelForSequenceClassification.from_pretrained(MODELO_ID).eval()
torch.set_num_threads(max(1, os.cpu_count() or 1))


def classificar(texto: str):
    texto = (texto or "").strip()
    if not texto:
        return {}, ""
    with torch.inference_mode():
        logits = modelo(**tok(texto, return_tensors="pt", truncation=True, max_length=128)).logits[0]
    prob = logits.softmax(-1).tolist()
    saida = {NOMES[modelo.config.id2label[i]]: p for i, p in enumerate(prob)}
    melhor = max(saida, key=saida.get)
    return saida, f"**{melhor}** — confiança {saida[melhor]:.0%}"


# os valores entre __ são preenchidos por src/fase5_preparar_hub.py a partir de results/fase4/metricas.json
AVISO = """O modelo é um BERTimbau ajustado em __N_TREINO__ comentários do dataset público da Olist, com rótulos gerados
por LLM. No teste de 400 comentários (rótulos de outro LLM, **sem validação humana**), o F1 macro foi __F1__.

**Sobre a confiança:** o modelo é confiante demais quando erra. No teste, __ERROS_095__ dos __ERROS__ erros tiveram
confiança ≥ 95%. Trate a porcentagem como uma ordem de preferência entre as classes, não como uma probabilidade
calibrada.

Dados: Brazilian E-Commerce Public Dataset by Olist (CC BY-NC-SA 4.0). Uso não comercial."""

EXEMPLOS = [
    "Produto chegou quebrado e a loja não responde meus e-mails.",
    "Comprei azul e veio vermelho.",
    "Já passou 10 dias do prazo e nada do meu pedido.",
    "Ótimo produto, chegou antes do prazo. Recomendo!",
    "O tecido é bem mais fino do que parecia na foto.",
]

with gr.Blocks(title="Motivo da avaliação") as demo:
    gr.Markdown("# Qual é o motivo principal desta avaliação?\nCole um comentário de compra online em português.")
    entrada = gr.Textbox(label="Comentário", lines=3, placeholder="Ex.: o produto veio com defeito")
    botao = gr.Button("Classificar", variant="primary")
    resumo = gr.Markdown()
    rotulos = gr.Label(label="Probabilidade por classe", num_top_classes=7)
    gr.Examples(EXEMPLOS, inputs=entrada)
    gr.Markdown(AVISO)
    botao.click(classificar, entrada, [rotulos, resumo])
    entrada.submit(classificar, entrada, [rotulos, resumo])

if __name__ == "__main__":
    demo.launch()
