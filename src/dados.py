"""Carregamento e limpeza mínima dos reviews da Olist (compartilhado entre fases)."""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CSV_BRUTO = RAIZ / "data" / "raw" / "olist_order_reviews_dataset.csv"


def normalizar_texto(s: pd.Series) -> pd.Series:
    """Colapsa espaços/quebras de linha; mantém caixa e acentos."""
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def carregar_reviews() -> pd.DataFrame:
    df = pd.read_csv(CSV_BRUTO, dtype=str)
    df["review_score"] = df["review_score"].astype(int)
    return df


def carregar_comentarios() -> pd.DataFrame:
    """Apenas linhas com comentário não vazio, com a coluna `texto` normalizada."""
    df = carregar_reviews()
    df["texto"] = normalizar_texto(df["review_comment_message"])
    return df[df["texto"] != ""].reset_index(drop=True)
