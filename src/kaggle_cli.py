"""Roda o Kaggle CLI com o token do .env (sem expor o token em linha de comando ou log).

    python src\\kaggle_cli.py kernels status abnerridigolo/olist-fase3-qlora-tucano2
"""
import subprocess
import sys

from dotenv import load_dotenv

from dados import RAIZ

if __name__ == "__main__":
    load_dotenv(RAIZ / ".env")
    sys.exit(subprocess.call([sys.executable, "-m", "kaggle", *sys.argv[1:]]))
