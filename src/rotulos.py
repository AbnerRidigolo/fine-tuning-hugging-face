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


def carregar_gold() -> pd.DataFrame:
    """Gold com os rótulos humanos da planilha. Falha se faltar rótulo ou houver classe inválida."""
    if not GOLD_XLSX.exists():
        raise FileNotFoundError(GOLD_XLSX)
    plan = pd.read_excel(GOLD_XLSX, sheet_name="rotular", dtype=str)
    plan["rotulo"] = plan["rotulo"].str.strip()
    faltando = plan["rotulo"].isna().sum()
    invalidos = sorted(set(plan["rotulo"].dropna()) - set(CLASSES))
    if faltando or invalidos:
        raise ValueError(f"gold incompleto: {faltando} sem rótulo, classes inválidas: {invalidos}")
    meta = pd.read_csv(GOLD_META, dtype={"review_id": str})
    gold = meta.merge(plan[["id_gold", "rotulo", "confianca_1a3"]], on="id_gold", validate="1:1")
    return gold
