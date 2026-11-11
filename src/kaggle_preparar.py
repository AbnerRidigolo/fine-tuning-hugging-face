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
import zipfile

from dados import RAIZ

USUARIO = os.environ.get("KAGGLE_USERNAME", "abnerridigolo")
DATASET = f"{USUARIO}/olist-motivo-avaliacoes"
KERNEL = f"{USUARIO}/olist-fase3-qlora-tucano2"


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
    main()
