"""Fase 1 - sorteia os comentários que serão rotulados pelo LLM (silver), sem sobreposição com o gold.

Parte de data/processed/pool_sem_gold.csv (já sem nenhum texto do gold, comparado em minúsculas).
1.000 por nota (5.000 no total), seed 42: mesma lógica do gold, para dar exemplos suficientes às
classes de reclamação no treino.
"""
import pandas as pd

from dados import RAIZ

SEED = 42
POR_NOTA = 1000


def main() -> None:
    pool = pd.read_csv(RAIZ / "data" / "processed" / "pool_sem_gold.csv", dtype={"review_id": str})
    gold = pd.read_csv(RAIZ / "data" / "gold" / "gold_meta.csv")
    amostra = (
        pool.groupby("review_score", group_keys=False)
        .sample(n=POR_NOTA, random_state=SEED)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    assert not set(amostra["review_id"]) & set(gold["review_id"])
    assert not set(amostra["texto"].str.lower()) & set(gold["texto"].str.lower())
    saida = RAIZ / "data" / "processed" / "silver_amostra.csv"
    amostra.to_csv(saida, index=False, encoding="utf-8")
    print(f"{len(amostra)} comentários -> {saida}")
    print(amostra["review_score"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
