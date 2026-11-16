"""Fase 4 - avaliação dos três modelos no teste de referência (400 itens, rótulos gpt-6.1-sol).

Usa as predições oficiais de cada execução (baseline recalculado do .joblib; BERTimbau e Tucano2 dos
arquivos results/<run>/gold_predicoes.csv gerados no próprio treino) e calcula:
  - F1 macro, acurácia, F1 por classe e matriz de confusão;
  - IC 95% por bootstrap pareado (2.000 reamostragens, seed 42) do F1 macro e das diferenças entre modelos;
  - itens em que os modelos discordam (results/fase4/discordancias.csv);
  - junta latência e tamanho de results/fase4/latencia.json (se existir).
Escreve results/fase4/metricas.json, reports/fase4_avaliacao.md e registra no MLflow.

    python src\\fase4_avaliacao.py
"""
import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from dados import RAIZ
from rotulos import CLASSES, carregar_gold

SEED, N_BOOT = 42, 2000
SAIDA = RAIZ / "results" / "fase4"
MODELOS = {"baseline_tfidf_lr": "Baseline TF-IDF + LR", "bertimbau": "BERTimbau", "qlora_tucano2_1.5b": "Tucano2 1.5B QLoRA"}
CURTO = {"atraso_ou_nao_entrega": "atraso", "produto_errado_ou_incompleto": "errado/incompleto",
         "defeito_ou_avaria": "defeito", "qualidade_abaixo_do_esperado": "qualidade",
         "atendimento_vendedor": "atendimento", "elogio": "elogio", "outro": "outro"}


def predicoes(gold: pd.DataFrame) -> pd.DataFrame:
    df = gold[["id_gold", "review_score", "texto", "rotulo"]].copy()
    m = joblib.load(RAIZ / "models" / "baseline_tfidf_lr" / "modelo.joblib")
    prob = m.predict_proba(df.texto)
    df["pred_baseline_tfidf_lr"] = m.classes_[prob.argmax(1)]
    df["conf_baseline_tfidf_lr"] = prob.max(1)
    for run in ("bertimbau", "qlora_tucano2_1.5b"):
        p = pd.read_csv(RAIZ / "results" / run / "gold_predicoes.csv")
        p = p.rename(columns={"pred": f"pred_{run}", "confianca": f"conf_{run}"})
        df = df.merge(p[["id_gold", f"pred_{run}", f"conf_{run}"]], on="id_gold", how="left", validate="1:1")
        assert df[f"pred_{run}"].notna().all(), run
    return df


def calibracao(df: pd.DataFrame, k: str) -> dict:
    """Confiança nos erros e ECE (10 faixas iguais de confiança)."""
    ok = (df[f"pred_{k}"] == df.rotulo).to_numpy()
    c = df[f"conf_{k}"].to_numpy()
    faixa = np.clip(np.digitize(c, np.linspace(0, 1, 11)) - 1, 0, 9)
    ece = sum((faixa == b).mean() * abs(ok[faixa == b].mean() - c[faixa == b].mean()) for b in range(10) if (faixa == b).any())
    return {"erros": int((~ok).sum()), "conf_media_erros": float(c[~ok].mean()), "conf_media_acertos": float(c[ok].mean()),
            "erros_conf_ge_095": int((c[~ok] >= 0.95).sum()), "ece": float(ece)}


def f1m(y, p) -> float:
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)


def bootstrap(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    y = df.rotulo.to_numpy()
    preds = {k: df[f"pred_{k}"].to_numpy() for k in MODELOS}
    amostras = {k: [] for k in MODELOS}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(df), len(df))
        for k, p in preds.items():
            amostras[k].append(f1m(y[idx], p[idx]))
    amostras = {k: np.array(v) for k, v in amostras.items()}
    out = {k: {"ic95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in amostras.items()}
    pares = [("bertimbau", "baseline_tfidf_lr"), ("qlora_tucano2_1.5b", "baseline_tfidf_lr"),
             ("qlora_tucano2_1.5b", "bertimbau")]
    out["diferencas"] = {}
    for a, b in pares:
        d = amostras[a] - amostras[b]
        out["diferencas"][f"{a}_menos_{b}"] = {
            "observada": f1m(y, preds[a]) - f1m(y, preds[b]),
            "ic95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "prob_maior_que_zero": float((d > 0).mean())}
    return out


def tabela_md(cab, linhas) -> str:
    s = "| " + " | ".join(cab) + " |\n|" + "---|" * len(cab) + "\n"
    return s + "".join("| " + " | ".join(str(c) for c in l) + " |\n" for l in linhas)


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    gold = carregar_gold()
    df = predicoes(gold)
    df.to_csv(SAIDA / "predicoes_gold_tres_modelos.csv", index=False, encoding="utf-8")

    met, matrizes = {}, {}
    for k in MODELOS:
        rel = classification_report(df.rotulo, df[f"pred_{k}"], labels=CLASSES, output_dict=True, zero_division=0)
        met[k] = {"f1_macro": rel["macro avg"]["f1-score"], "acuracia": rel["accuracy"],
                  **{f"f1_{c}": rel[c]["f1-score"] for c in CLASSES}, **calibracao(df, k)}
        matrizes[k] = confusion_matrix(df.rotulo, df[f"pred_{k}"], labels=CLASSES)
        pd.DataFrame(matrizes[k], index=[f"ref:{c}" for c in CLASSES], columns=[f"pred:{c}" for c in CLASSES]) \
          .to_csv(SAIDA / f"matriz_confusao_{k}.csv", encoding="utf-8")
    boot = bootstrap(df)
    lat_path = SAIDA / "latencia.json"
    lat = json.loads(lat_path.read_text(encoding="utf-8")) if lat_path.exists() else None

    preds = df[[f"pred_{k}" for k in MODELOS]]
    disc = df[preds.nunique(axis=1) > 1].copy()
    disc["n_acertos"] = sum((disc[f"pred_{k}"] == disc.rotulo).astype(int) for k in MODELOS)
    disc.to_csv(SAIDA / "discordancias.csv", index=False, encoding="utf-8")

    resumo = {"n": len(df), "metricas": met, "bootstrap": boot, "latencia_tamanho": lat,
              "n_discordancias": int(len(disc)),
              "n_todos_erram": int((~(preds.eq(df.rotulo, axis=0)).any(axis=1)).sum())}
    (SAIDA / "metricas.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- relatório ----------
    nomes = list(MODELOS.values())
    r = ["# Fase 4 — avaliação no teste de referência\n",
         f"400 comentários (80 por nota), rótulos de referência do **gpt-6.1-sol** com o guia de rotulagem "
         f"(**sem validação humana**). IC 95%: bootstrap pareado, {N_BOOT} reamostragens, seed {SEED}.\n",
         "## Métricas\n"]
    linhas = [["F1 macro"] + [f"{met[k]['f1_macro']:.3f}" for k in MODELOS],
              ["IC 95% F1 macro"] + [f"{boot[k]['ic95'][0]:.3f}–{boot[k]['ic95'][1]:.3f}" for k in MODELOS],
              ["Acurácia"] + [f"{met[k]['acuracia']:.3f}" for k in MODELOS]]
    linhas += [[f"F1 {CURTO[c]} (n={int((df.rotulo == c).sum())})"] + [f"{met[k]['f1_' + c]:.3f}" for k in MODELOS]
               for c in CLASSES]
    linhas += [["Erros (de 400)"] + [str(met[k]["erros"]) for k in MODELOS],
               ["Confiança média nos erros"] + [f"{met[k]['conf_media_erros']:.2f}" for k in MODELOS],
               ["Erros com confiança ≥ 0,95"] + [str(met[k]["erros_conf_ge_095"]) for k in MODELOS],
               ["ECE (10 faixas)"] + [f"{met[k]['ece']:.3f}" for k in MODELOS]]
    if lat:
        hw = lat["hardware"]
        fmt = lambda v: "—" if v is None else f"{v:.1f}"
        linhas += [["Latência GPU, mediana (ms/exemplo)"] + [fmt(lat[k].get("gpu_ms")) for k in MODELOS],
                   ["Latência GPU, p95 (ms)"] + [fmt(lat[k].get("gpu_ms_p95")) for k in MODELOS],
                   ["Latência CPU, mediana (ms/exemplo)"] + [fmt(lat[k].get("cpu_ms")) for k in MODELOS],
                   ["Latência CPU, p95 (ms)"] + [fmt(lat[k].get("cpu_ms_p95")) for k in MODELOS],
                   ["Tamanho em disco (MB)"] + [f"{lat[k]['disco_mb']:.0f}" for k in MODELOS]]
    r.append(tabela_md(["", *nomes], linhas))
    if lat:
        t = lat["qlora_tucano2_1.5b"]
        r.append(f"\nLatência e tamanho medidos no mesmo hardware — GPU: {hw['gpu']}; CPU: {hw['cpu']} "
                 f"({hw['torch_threads_cpu']} threads); {hw['protocolo']}; {hw['n_textos']} textos após "
                 f"{hw['aquecimento']} de aquecimento. Tucano2: GPU em 4 bits + adaptador; CPU com adaptador mesclado "
                 f"em base fp32. Tamanho do Tucano2 = base ({t['disco_base_mb']:.0f} MB) + adaptador "
                 f"({t['disco_adaptador_mb']:.0f} MB). BERTimbau em fp32 nos dois dispositivos.\n")
        if "concordancia_cpu_vs_gpu" in t:
            r.append(f"Fidelidade do Tucano2 mesclado em CPU: concordância de {t['concordancia_cpu_vs_gpu']:.1%} com o "
                     f"modelo em 4 bits; F1 macro {t['f1_macro_cpu_mesclado_fp32']:.3f} (CPU) vs "
                     f"{t['f1_macro_gpu_4bit']:.3f} (GPU 4 bits, local).\n")
    r.append("\n## Diferenças de F1 macro (bootstrap pareado)\n")
    rot = {"bertimbau_menos_baseline_tfidf_lr": "BERTimbau − Baseline",
           "qlora_tucano2_1.5b_menos_baseline_tfidf_lr": "Tucano2 − Baseline",
           "qlora_tucano2_1.5b_menos_bertimbau": "Tucano2 − BERTimbau"}
    r.append(tabela_md(["Comparação", "Diferença", "IC 95%", "P(diferença > 0)"],
                       [[rot[k], f"{v['observada']:+.3f}", f"{v['ic95'][0]:+.3f} a {v['ic95'][1]:+.3f}",
                         f"{v['prob_maior_que_zero']:.1%}"] for k, v in boot["diferencas"].items()]))
    r.append("\n## Matrizes de confusão (linhas = referência, colunas = predição)\n")
    for k, nome in MODELOS.items():
        r.append(f"\n**{nome}**\n\n")
        r.append(tabela_md(["ref \\ pred", *[CURTO[c] for c in CLASSES]],
                           [[CURTO[c], *matrizes[k][i]] for i, c in enumerate(CLASSES)]))
    r.append(f"\n## Discordâncias\n\nOs três modelos discordam entre si em {len(disc)} dos 400 itens; "
             f"os três erram juntos (em relação à referência) em {resumo['n_todos_erram']}. "
             "Lista completa em `results/fase4/discordancias.csv`; exemplos comentados em "
             "`reports/fase4_exemplos_discordancia.md`.\n")
    (RAIZ / "reports" / "fase4_avaliacao.md").write_text("".join(r), encoding="utf-8")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("motivo-avaliacoes-olist")
    with mlflow.start_run(run_name="fase4_avaliacao_comparativa"):
        mlflow.log_params({"n_bootstrap": N_BOOT, "seed": SEED, "referencia": "gpt-6.1-sol (sem humano)",
                           **({"hardware_gpu": lat["hardware"]["gpu"], "hardware_cpu": lat["hardware"]["cpu"]} if lat else {})})
        for k in MODELOS:
            mlflow.log_metrics({f"{k}_{m}": v for m, v in met[k].items()})
            mlflow.log_metrics({f"{k}_f1_macro_ic95_inf": boot[k]["ic95"][0], f"{k}_f1_macro_ic95_sup": boot[k]["ic95"][1]})
            if lat:
                mlflow.log_metrics({f"{k}_{m}": v for m, v in lat[k].items() if isinstance(v, (int, float))})
        for k, v in boot["diferencas"].items():
            mlflow.log_metrics({f"dif_{k}": v["observada"], f"dif_{k}_ic95_inf": v["ic95"][0],
                                f"dif_{k}_ic95_sup": v["ic95"][1], f"dif_{k}_prob_pos": v["prob_maior_que_zero"]})
        mlflow.log_artifacts(str(SAIDA), artifact_path="fase4")
        mlflow.log_artifact(str(RAIZ / "reports" / "fase4_avaliacao.md"))
    print("".join(r))


if __name__ == "__main__":
    main()
