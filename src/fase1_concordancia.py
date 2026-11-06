"""Fase 1 - concordância do rotulador LLM com os rótulos humanos no gold (kappa de Cohen + matriz de confusão).

Pré-requisitos: planilha gold rotulada e
    python src\\rotulador.py --entrada data\\gold\\gold_meta.csv --col-id id_gold --saida data\\gold\\rotulos_llm_gold.jsonl --modelo gpt-6.1-terra --esforco none
"""
import json

import mlflow
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from dados import RAIZ
from rotulos import CLASSES, carregar_gold, ler_jsonl


def main() -> None:
    gold = carregar_gold()
    llm = ler_jsonl(RAIZ / "data" / "gold" / "rotulos_llm_gold.jsonl").rename(columns={"id": "id_gold", "rotulo": "rotulo_llm"})
    df = gold.merge(llm[["id_gold", "rotulo_llm"]], on="id_gold", how="left", validate="1:1")
    if df["rotulo_llm"].isna().any():
        raise ValueError(f"{df['rotulo_llm'].isna().sum()} itens do gold sem rótulo do LLM")

    kappa = cohen_kappa_score(df.rotulo, df.rotulo_llm, labels=CLASSES)
    acc = (df.rotulo == df.rotulo_llm).mean()
    f1 = f1_score(df.rotulo, df.rotulo_llm, labels=CLASSES, average="macro", zero_division=0)
    cm = pd.DataFrame(confusion_matrix(df.rotulo, df.rotulo_llm, labels=CLASSES),
                      index=[f"humano:{c}" for c in CLASSES], columns=[f"llm:{c}" for c in CLASSES])

    saida = RAIZ / "reports" / "fase1_concordancia"
    saida.mkdir(parents=True, exist_ok=True)
    cm.to_csv(saida / "matriz_confusao_humano_vs_llm.csv", encoding="utf-8")
    df[df.rotulo != df.rotulo_llm][["id_gold", "review_score", "texto", "rotulo", "rotulo_llm"]].to_csv(
        saida / "discordancias.csv", index=False, encoding="utf-8")
    res = {"kappa_cohen": kappa, "concordancia": acc, "f1_macro_llm_vs_humano": f1, "n": len(df)}
    (saida / "metricas.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("motivo-avaliacoes-olist")
    with mlflow.start_run(run_name="rotulador_llm_vs_gold"):
        mlflow.log_param("rotulador", llm["modelo"].iloc[0])
        mlflow.log_metrics(res)
        mlflow.log_artifacts(str(saida), artifact_path="concordancia")

    print(json.dumps(res, indent=2))
    pd.set_option("display.width", 250)
    print(cm)


if __name__ == "__main__":
    main()
