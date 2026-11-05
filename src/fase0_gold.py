"""Fase 0 - separa 400 comentários para o teste gold (estratificado pela nota) e gera a planilha de rotulagem.

Regras do pool (aplicadas antes de qualquer amostragem):
  1. Só comentários não vazios.
  2. Um registro por review_id (o CSV repete review_id quando a avaliação cobre vários pedidos).
  3. Remove textos sem nenhuma letra ("." , "10", "100%"...).
  4. Deduplica pelo texto em minúsculas, para que o mesmo texto nunca apareça no gold e no treino.

Amostragem do gold: 80 por nota (1 a 5), seed 42. A alocação igual por nota é proposital: com a
alocação proporcional, ~74% do gold seriam notas 4-5 (quase tudo "elogio") e as classes de
reclamação ficariam com pouquíssimos exemplos para medir F1 por classe.

A planilha mostra só o comentário (sem nota e sem qualquer rótulo de modelo). A nota fica em
data/gold/gold_meta.csv, que você não precisa abrir para rotular.
"""
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from dados import RAIZ, carregar_comentarios

SEED = 42
POR_NOTA = 80
CLASSES = [
    "atraso_ou_nao_entrega",
    "produto_errado_ou_incompleto",
    "defeito_ou_avaria",
    "qualidade_abaixo_do_esperado",
    "atendimento_vendedor",
    "elogio",
    "outro",
]


def montar_pool() -> pd.DataFrame:
    df = carregar_comentarios()
    df = df.drop_duplicates("review_id", keep="first")
    df = df[df["texto"].str.contains(r"[A-Za-zÀ-ÿ]", regex=True)]
    df["texto_chave"] = df["texto"].str.lower()
    df = df.drop_duplicates("texto_chave", keep="first")
    return df.reset_index(drop=True)


def escrever_planilha(gold: pd.DataFrame, caminho) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "rotular"
    cab = ["id_gold", "comentario", "rotulo", "confianca_1a3", "observacao"]
    ws.append(cab)
    for _, r in gold.iterrows():
        ws.append([r["id_gold"], r["texto"], None, None, None])

    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="444444")
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 90
    ws.column_dimensions["C"].width = 32
    ws.column_dimensions["D"].width = 14
    ws.column_dimensions["E"].width = 40
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"

    n = len(gold) + 1
    dv = DataValidation(type="list", formula1='"' + ",".join(CLASSES) + '"', allow_blank=True)
    dv.error, dv.errorTitle = "Escolha uma das 7 classes do guia.", "Rótulo inválido"
    ws.add_data_validation(dv)
    dv.add(f"C2:C{n}")
    dv_conf = DataValidation(type="list", formula1='"1,2,3"', allow_blank=True)
    ws.add_data_validation(dv_conf)
    dv_conf.add(f"D2:D{n}")

    leg = wb.create_sheet("classes")
    leg.append(["classe", "resumo (detalhes em docs/guia_rotulagem.md)"])
    resumo = {
        "atraso_ou_nao_entrega": "atrasou, não chegou, rastreio parado, entrega em local errado",
        "produto_errado_ou_incompleto": "veio item/cor/tamanho/modelo diferente, faltou peça ou unidade",
        "defeito_ou_avaria": "chegou quebrado, amassado, não funciona, parou de funcionar",
        "qualidade_abaixo_do_esperado": "funciona, mas é fraco/pequeno/frágil/diferente da foto",
        "atendimento_vendedor": "vendedor/loja não responde, não resolve, troca ou estorno negado",
        "elogio": "satisfação sem reclamação relevante",
        "outro": "neutro, ininteligível, pergunta, sobre a Olist/plataforma, sem motivo claro",
    }
    for k in CLASSES:
        leg.append([k, resumo[k]])
    leg.column_dimensions["A"].width = 32
    leg.column_dimensions["B"].width = 80
    leg2 = wb.create_sheet("confianca")
    leg2.append(["1 = chute / ambíguo", "2 = provável", "3 = certo"])
    wb.save(caminho)


def main() -> None:
    pool = montar_pool()
    gold = (
        pool.groupby("review_score", group_keys=False)
        .sample(n=POR_NOTA, random_state=SEED)
        .sample(frac=1, random_state=SEED)  # embaralha para não agrupar por nota
        .reset_index(drop=True)
    )
    gold.insert(0, "id_gold", [f"G{i:03d}" for i in range(1, len(gold) + 1)])

    pasta = RAIZ / "data" / "gold"
    pasta.mkdir(parents=True, exist_ok=True)
    gold[["id_gold", "review_id", "order_id", "review_score", "texto"]].to_csv(
        pasta / "gold_meta.csv", index=False, encoding="utf-8"
    )
    escrever_planilha(gold, pasta / "gold_para_rotular.xlsx")

    # Pool restante (sem nada do gold) para guia, silver e treino.
    resto = pool[~pool["texto_chave"].isin(set(gold["texto_chave"]))]
    resto = resto[["review_id", "order_id", "review_score", "texto"]]
    proc = RAIZ / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    resto.to_csv(proc / "pool_sem_gold.csv", index=False, encoding="utf-8")

    print(f"pool total: {len(pool)} | gold: {len(gold)} | pool sem gold: {len(resto)}")
    print("pool por nota:", pool["review_score"].value_counts().sort_index().to_dict())
    print("gold por nota:", gold["review_score"].value_counts().sort_index().to_dict())
    print("comprimento médio gold (chars):", round(gold["texto"].str.len().mean(), 1))


if __name__ == "__main__":
    main()
