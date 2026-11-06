"""Fase 1 - baseline TF-IDF + regressão logística, registrado no MLflow.

Treina no split `treino` do silver, escolhe C na `validacao` (F1 macro) e, se o gold rotulado existir,
avalia nele. Ver a UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from dados import RAIZ
from rotulos import CLASSES, carregar_gold, carregar_silver

SEED = 42
GRADE_C = [0.5, 1.0, 2.0, 5.0, 10.0]


def montar(c: float) -> Pipeline:
    feats = FeatureUnion([
        ("palavras", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(lowercase=True, strip_accents="unicode", analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=SEED)
    return Pipeline([("tfidf", feats), ("clf", clf)])


def avaliar(modelo, textos, y, prefixo: str, pasta: Path) -> dict:
    t0 = time.perf_counter()
    pred = modelo.predict(textos)
    lat_ms = 1000 * (time.perf_counter() - t0) / len(textos)
    rel = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=CLASSES)
    met = {f"{prefixo}_f1_macro": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
           f"{prefixo}_acuracia": rel["accuracy"],
           f"{prefixo}_latencia_cpu_ms_por_exemplo_lote": lat_ms}
    for c in CLASSES:
        met[f"{prefixo}_f1_{c}"] = rel[c]["f1-score"]
    (pasta / f"{prefixo}_relatorio.json").write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    (pasta / f"{prefixo}_matriz_confusao.json").write_text(
        json.dumps({"classes": CLASSES, "matriz": cm.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return met


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("motivo-avaliacoes-olist")
    df = carregar_silver()
    tr, va = df[df.split == "treino"], df[df.split == "validacao"]

    melhor = None
    for c in GRADE_C:
        m = montar(c).fit(tr.texto, tr.rotulo)
        f1 = f1_score(va.rotulo, m.predict(va.texto), labels=CLASSES, average="macro", zero_division=0)
        print(f"C={c:<5} F1 macro validação={f1:.4f}")
        if melhor is None or f1 > melhor[1]:
            melhor = (c, f1, m)
    c, _, modelo = melhor

    pasta = RAIZ / "models" / "baseline_tfidf_lr"
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, pasta / "modelo.joblib")

    with mlflow.start_run(run_name="baseline_tfidf_lr"):
        mlflow.log_params({"modelo": "tfidf(word1-2+char_wb2-5)+logreg", "C": c, "class_weight": "balanced",
                           "seed": SEED, "n_treino": len(tr), "n_validacao": len(va),
                           "rotulador_silver": df["modelo"].iloc[0]})
        met = avaliar(modelo, va.texto, va.rotulo, "validacao", pasta)
        try:
            gold = carregar_gold()
            met |= avaliar(modelo, gold.texto, gold.rotulo, "gold", pasta)
        except FileNotFoundError:
            print("gold ainda não rotulado: avaliação no gold fica para depois")
        except ValueError as e:
            print(f"gold ignorado: {e}")
        met["tamanho_disco_mb"] = (pasta / "modelo.joblib").stat().st_size / 2**20
        mlflow.log_metrics(met)
        mlflow.log_artifacts(str(pasta), artifact_path="baseline")
    print(json.dumps({k: round(v, 4) for k, v in met.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
