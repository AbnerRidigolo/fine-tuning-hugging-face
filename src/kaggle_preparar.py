"""Monta as pastas de envio ao Kaggle em kaggle/ (dataset privado + notebook da Fase 3).

    python src\\kaggle_preparar.py
    python src\\kaggle_cli.py datasets create -p kaggle\\dataset          (1a vez; privado por padrão)
    python src\\kaggle_cli.py datasets version -p kaggle\\dataset -m "..." (atualizações)
    python src\\kaggle_cli.py kernels push -p kaggle\\fase3 --accelerator NvidiaTeslaT4
    python src\\kaggle_cli.py kernels status <usuario>/olist-fase3-qlora-tucano2
    python src\\kaggle_cli.py kernels output <usuario>/olist-fase3-qlora-tucano2 -p results\\_kaggle_fase3
"""
import json
import os
import shutil
import sys
import zipfile

from dados import RAIZ

USUARIO = os.environ.get("KAGGLE_USERNAME", "abnerridigolo")
DATASET = f"{USUARIO}/olist-motivo-avaliacoes"
KERNEL = f"{USUARIO}/olist-fase3-qlora-tucano2"
DATASET_MODELOS = f"{USUARIO}/olist-motivo-modelos"
KERNEL_FASE4 = f"{USUARIO}/olist-fase4-latencia"


def kernel_metadata(pasta, kid, arquivo, fontes) -> None:
    (pasta / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": kid.split("/")[1], "code_file": arquivo, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": fontes, "competition_sources": [], "kernel_sources": [],
    }, indent=2), encoding="utf-8")


def preparar_fase4() -> None:
    """proj.zip com src/ (código da medição), data/gold/ (teste de referência) e models/ (os três modelos)."""
    base = RAIZ / "kaggle"
    ds = base / "modelos"
    shutil.rmtree(ds, ignore_errors=True)
    ds.mkdir(parents=True)
    arquivos = [RAIZ / "src" / n for n in ("dados.py", "rotulos.py", "fase4_latencia.py")]
    arquivos += [RAIZ / "data" / "gold" / n for n in ("gold_meta.csv", "rotulos_referencia.jsonl")]
    for d in ("baseline_tfidf_lr", "bertimbau", "qlora_tucano2_1.5b"):
        arquivos += [f for f in (RAIZ / "models" / d).iterdir() if f.is_file() and f.name != "training_args.bin"]
    with zipfile.ZipFile(ds / "proj.zip", "w", zipfile.ZIP_STORED) as z:
        for f in arquivos:
            z.write(f, f.relative_to(RAIZ).as_posix())
    (ds / "dataset-metadata.json").write_text(json.dumps({
        "title": "Olist motivo de avaliacoes - modelos treinados",
        "id": DATASET_MODELOS,
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
        "description": "Baseline TF-IDF+LR, BERTimbau e adaptador QLoRA do Tucano2 1.5B treinados sobre dados "
                       "derivados da Olist (CC BY-NC-SA 4.0), mais o código de medição de latência.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    nb = base / "fase4"
    shutil.rmtree(nb, ignore_errors=True)
    nb.mkdir(parents=True)
    shutil.copy(RAIZ / "notebooks" / "fase4_latencia_kaggle.ipynb", nb)
    kernel_metadata(nb, KERNEL_FASE4, "fase4_latencia_kaggle.ipynb", [DATASET_MODELOS])
    print(f"pronto: {ds / 'proj.zip'} ({(ds / 'proj.zip').stat().st_size / 2**20:.0f} MB) e {nb}")


def main() -> None:
    base = RAIZ / "kaggle"
    ds = base / "dataset"
    shutil.rmtree(ds, ignore_errors=True)
    ds.mkdir(parents=True)
    zipfile.ZipFile(RAIZ / "dist" / "dados_treino.zip").extractall(ds)
    (ds / "dataset-metadata.json").write_text(json.dumps({
        "title": "Olist motivo de avaliacoes (silver + teste)",
        "id": DATASET,
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
        "description": "Derivado do Brazilian E-Commerce Public Dataset by Olist (CC BY-NC-SA 4.0). "
                       "Rótulos silver gerados por gpt-6.1-terra; teste rotulado por gpt-6.1-sol.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    nb = base / "fase3"
    shutil.rmtree(nb, ignore_errors=True)
    nb.mkdir(parents=True)
    shutil.copy(RAIZ / "notebooks" / "fase3_qlora_tucano2.ipynb", nb)
    (nb / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL,
        "title": "olist-fase3-qlora-tucano2",
        "code_file": "fase3_qlora_tucano2.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2), encoding="utf-8")
    print(f"pronto: {ds} e {nb}")


if __name__ == "__main__":
    preparar_fase4() if sys.argv[1:] == ["fase4"] else main()
