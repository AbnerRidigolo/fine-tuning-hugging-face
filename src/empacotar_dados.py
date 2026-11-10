"""Empacota os dados que os notebooks de GPU (Kaggle/Colab) precisam em dist/dados_treino.zip.

Conteúdo:
  silver.csv            review_id, texto, rotulo, split (treino/validacao)
  teste_referencia.csv  id_gold, texto, rotulo (rótulo de referência gpt-6.1-sol)

Kaggle: crie um Dataset privado com o zip (o Kaggle extrai sozinho) e anexe ao notebook.
Colab:  o notebook pede o upload do zip.
"""
import zipfile

from dados import RAIZ
from rotulos import carregar_gold, carregar_silver


def main() -> None:
    dist = RAIZ / "dist"
    dist.mkdir(exist_ok=True)
    silver = carregar_silver()[["review_id", "texto", "rotulo", "split"]]
    teste = carregar_gold()[["id_gold", "texto", "rotulo"]]
    destino = dist / "dados_treino.zip"
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("silver.csv", silver.to_csv(index=False))
        z.writestr("teste_referencia.csv", teste.to_csv(index=False))
    print(f"{destino}  silver={len(silver)}  teste={len(teste)}")


if __name__ == "__main__":
    main()
