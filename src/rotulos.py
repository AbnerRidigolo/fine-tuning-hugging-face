"""Classes, leitura do silver e do gold rotulado (compartilhado pelas fases 1 a 4)."""
import json

import pandas as pd

from dados import RAIZ

CLASSES = [
    "atraso_ou_nao_entrega",
    "produto_errado_ou_incompleto",
    "defeito_ou_avaria",
    "qualidade_abaixo_do_esperado",
    "atendimento_vendedor",
    "elogio",
    "outro",
]

SILVER_JSONL = RAIZ / "data" / "silver" / "rotulos_silver.jsonl"
SILVER_CSV = RAIZ / "data" / "silver" / "silver.csv"
GOLD_XLSX = RAIZ / "data" / "gold" / "gold_para_rotular.xlsx"
GOLD_META = RAIZ / "data" / "gold" / "gold_meta.csv"


def ler_jsonl(caminho) -> pd.DataFrame:
    regs = [json.loads(l) for l in open(caminho, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(regs)
    # se um id foi re-tentado, vale o último registro sem erro
    df = df[df["erro"].isna()].drop_duplicates("id", keep="last")
    return df


def carregar_silver() -> pd.DataFrame:
    """silver.csv: review_id, review_score, texto, rotulo, confianca, split (treino/validacao)."""
    return pd.read_csv(SILVER_CSV, dtype={"review_id": str})


GOLD_REFERENCIA = RAIZ / "data" / "gold" / "rotulos_referencia.jsonl"


def carregar_gold() -> pd.DataFrame:
    """Conjunto de teste (400) com rótulos de REFERÊNCIA gerados por LLM (gpt-6.1-sol, guia como prompt).

    Decisão do projeto: não há rotulagem humana; as métricas medem concordância com o LLM de referência.
    Falha se faltar rótulo para algum item.
    """
    meta = pd.read_csv(GOLD_META, dtype={"review_id": str})
    ref = ler_jsonl(GOLD_REFERENCIA).rename(columns={"id": "id_gold"})
    gold = meta.merge(ref[["id_gold", "rotulo", "confianca", "modelo"]], on="id_gold", how="left", validate="1:1")
    if gold["rotulo"].isna().any():
        raise ValueError(f"gold incompleto: {gold['rotulo'].isna().sum()} itens sem rótulo de referência")
    return gold
