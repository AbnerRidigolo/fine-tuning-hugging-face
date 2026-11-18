"""Gera notebooks/fase4_latencia_kaggle.ipynb: mede latência e tamanho dos três modelos na mesma sessão
do Kaggle (GPU T4 + CPU da sessão), rodando src/fase4_latencia.py sobre o pacote proj.zip
(src/, data/gold/, models/) anexado como dataset privado.

    python notebooks\\gerar_fase4.py
"""
from pathlib import Path

import nbformat as nbf

MD, CODE = "md", "code"
CELULAS = [
(MD, """# Fase 4 — latência e tamanho dos três modelos (mesma sessão Kaggle)

Mede, no mesmo hardware, o baseline TF-IDF + LR, o BERTimbau e o Tucano2 1.5B QLoRA com o protocolo de
`src/fase4_latencia.py`: um comentário por chamada (lote de 1), tokenização incluída, 400 textos do teste
de referência após 10 de aquecimento, cada chamada cronometrada; mediana, p95 e média.

Requer o dataset privado com `proj.zip` (código, teste de referência e modelos treinados), GPU T4 e Internet
(para baixar o modelo base do Tucano2). Saída: `saida_fase4.zip` com `latencia.json`."""),
(CODE, """%pip install -q transformers==5.22.2 accelerate==1.15.0 peft==0.21.2 bitsandbytes==0.50.2 scikit-learn==1.9.1
# a imagem do Kaggle traz torchao 0.10, que o peft 0.21 rejeita ao injetar o LoRA; não usamos torchao
%pip uninstall -q -y torchao"""),
(CODE, """import glob, os, shutil, sys, zipfile
PROJ = "/tmp/proj"  # fora de /kaggle/working: os modelos não viram saída do notebook
achado = glob.glob("/kaggle/input/**/src/fase4_latencia.py", recursive=True)
if achado:                       # o Kaggle extraiu o zip
    shutil.copytree(os.path.dirname(os.path.dirname(achado[0])), PROJ, dirs_exist_ok=True)
else:                            # o zip veio inteiro
    zipfile.ZipFile(glob.glob("/kaggle/input/**/proj.zip", recursive=True)[0]).extractall(PROJ)
os.environ["FASE4_SAIDA"] = "/kaggle/working/saida_fase4"
sys.path.insert(0, f"{PROJ}/src")
print(sorted(os.listdir(PROJ)), sorted(os.listdir(f"{PROJ}/models")))"""),
(CODE, """import torch, platform
print(torch.__version__, torch.cuda.get_device_name(0), "| CPUs:", os.cpu_count(), "| threads torch:", torch.get_num_threads())
import fase4_latencia
fase4_latencia.main()"""),
(CODE, """shutil.make_archive("/kaggle/working/saida_fase4", "zip", "/kaggle/working/saida_fase4")
print(open("/kaggle/working/saida_fase4/latencia.json").read())"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if t == MD else nbf.v4.new_code_cell(c) for t, c in CELULAS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    destino = Path(__file__).parent / "fase4_latencia_kaggle.ipynb"
    nbf.write(nb, destino)
    print(destino)


if __name__ == "__main__":
    main()
