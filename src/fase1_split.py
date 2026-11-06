"""Fase 1 - junta os rótulos silver aos textos e separa treino/validação (85/15, estratificado, seed 42).

A mesma divisão é usada pelo baseline, pelo BERTimbau e pelo QLoRA. O gold nunca entra aqui.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from dados import RAIZ
from rotulos import SILVER_CSV, SILVER_JSONL, ler_jsonl

SEED = 42


def main() -> None:
    amostra = pd.read_csv(RAIZ / "data" / "processed" / "silver_amostra.csv", dtype={"review_id": str})
    rot = ler_jsonl(SILVER_JSONL).rename(columns={"id": "review_id"})
    df = amostra.merge(rot[["review_id", "rotulo", "confianca", "modelo"]], on="review_id", how="inner")
    faltam = len(amostra) - len(df)
    tr, va = train_test_split(df.index, test_size=0.15, stratify=df["rotulo"], random_state=SEED)
    df["split"] = "treino"
    df.loc[va, "split"] = "validacao"
    df = df[["review_id", "review_score", "texto", "rotulo", "confianca", "modelo", "split"]]
    df.to_csv(SILVER_CSV, index=False, encoding="utf-8")
    print(f"silver: {len(df)} (faltando/erro: {faltam}) -> {SILVER_CSV}")
    print(pd.crosstab(df["rotulo"], df["split"], margins=True))


if __name__ == "__main__":
    main()
