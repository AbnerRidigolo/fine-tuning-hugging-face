"""Importa no MLflow local o resultado de um notebook de GPU (saida_faseN.zip).

    python src\\importar_resultados.py caminho\\saida_fase2.zip

Extrai em results/<run_name>/ (sem o diretório do modelo, que fica em models/<run_name>/), registra
params, métricas, histórico de validação por época e os artefatos no experimento do projeto.
"""
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import mlflow

from dados import RAIZ


def main(caminho_zip: str) -> None:
    tmp = RAIZ / "results" / "_tmp_import"
    shutil.rmtree(tmp, ignore_errors=True)
    zipfile.ZipFile(caminho_zip).extractall(tmp)
    reg = json.loads((tmp / "registro_mlflow.json").read_text(encoding="utf-8"))
    nome = reg["run_name"]

    destino = RAIZ / "results" / nome
    shutil.rmtree(destino, ignore_errors=True)
    for sub in ("modelo", "adaptador"):
        if (tmp / sub).exists():
            dm = RAIZ / "models" / nome
            shutil.rmtree(dm, ignore_errors=True)
            shutil.move(str(tmp / sub), dm)
    tmp.rename(destino)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("motivo-avaliacoes-olist")
    with mlflow.start_run(run_name=nome):
        mlflow.log_params({k: str(v)[:500] for k, v in reg["params"].items()})
        mlflow.log_metrics({k: float(v) for k, v in reg["metrics"].items()})
        for h in reg.get("historico_validacao", []):
            passo = int(round(h["epoch"]))
            mlflow.log_metric("val_f1_macro_por_epoca", h["eval_f1_macro"], step=passo)
            mlflow.log_metric("val_loss_por_epoca", h["eval_loss"], step=passo)
        mlflow.log_artifacts(str(destino), artifact_path=nome)
    print(f"run '{nome}' registrado; artefatos em {destino}")
    print(json.dumps({k: round(float(v), 4) for k, v in reg["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
