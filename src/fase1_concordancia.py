"""Fase 1 - concordância do rotulador silver (gpt-6.1-terra) com o rotulador de referência do teste
(gpt-6.1-sol) nos 400 itens: kappa de Cohen + matriz de confusão.

Não há rótulo humano: isto mede concordância entre dois LLMs com o mesmo guia, não acerto.

Pré-requisitos:
    python src\\rotulador.py --entrada data\\gold\\gold_meta.csv --col-id id_gold --saida data\\gold\\rotulos_referencia.jsonl --modelo gpt-6.1-sol --esforco medium
    python src\\rotulador.py --entrada data\\gold\\gold_meta.csv --col-id id_gold --saida data\\gold\\rotulos_silver_labeler.jsonl --modelo gpt-6.1-terra --esforco none
"""
import json

import mlflow
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from dados import RAIZ
from rotulos import CLASSES, carregar_gold, ler_jsonl


def main() -> None:
    gold = carregar_gold()
    sil = ler_jsonl(RAIZ / "data" / "gold" / "rotulos_silver_labeler.jsonl").rename(
        columns={"id": "id_gold", "rotulo": "rotulo_silver", "modelo": "modelo_silver"})
    df = gold.merge(sil[["id_gold", "rotulo_silver", "modelo_silver"]], on="id_gold", how="left", validate="1:1")
    if df["rotulo_silver"].isna().any():
        raise ValueError(f"{df['rotulo_silver'].isna().sum()} itens sem rótulo do rotulador silver")

    kappa = cohen_kappa_score(df.rotulo, df.rotulo_silver, labels=CLASSES)
    acc = (df.rotulo == df.rotulo_silver).mean()
    f1 = f1_score(df.rotulo, df.rotulo_silver, labels=CLASSES, average="macro", zero_division=0)
    cm = pd.DataFrame(confusion_matrix(df.rotulo, df.rotulo_silver, labels=CLASSES),
                      index=[f"ref:{c}" for c in CLASSES], columns=[f"silver:{c}" for c in CLASSES])

    saida = RAIZ / "reports" / "fase1_concordancia"
    saida.mkdir(parents=True, exist_ok=True)
    cm.to_csv(saida / "matriz_confusao_referencia_vs_silver.csv", encoding="utf-8")
    df[df.rotulo != df.rotulo_silver][["id_gold", "review_score", "texto", "rotulo", "rotulo_silver"]].rename(
        columns={"rotulo": "rotulo_referencia"}).to_csv(saida / "discordancias.csv", index=False, encoding="utf-8")
    res = {"kappa_cohen": kappa, "concordancia": acc, "f1_macro_silver_vs_referencia": f1, "n": len(df)}
    (saida / "metricas.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("motivo-avaliacoes-olist")
    with mlflow.start_run(run_name="rotulador_silver_vs_referencia"):
        mlflow.log_params({"rotulador_silver": df["modelo_silver"].iloc[0],
                           "rotulador_referencia": df["modelo"].iloc[0], "referencia_humana": False})
        mlflow.log_metrics(res)
        mlflow.log_artifacts(str(saida), artifact_path="concordancia")

    print(json.dumps(res, indent=2))
    print("distribuição referência:", df.rotulo.value_counts().to_dict())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(cm)


if __name__ == "__main__":
    main()
