# Motivo de avaliações de e-commerce em português: BERTimbau vs LLM pequeno com QLoRA

> Em construção. Resultados, conclusão e links entram aqui ao final do estudo, a partir das execuções registradas.

**Pergunta:** um LLM pequeno ajustado com QLoRA supera o BERTimbau o suficiente para justificar o custo de inferência?

Dados: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(CC BY-NC-SA 4.0). Dataset e modelos derivados são distribuídos sob a mesma licença, com crédito à Olist.

## Ambiente (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env   # preencha os tokens; o .env não vai para o git
```

## Fase 0

```powershell
# baixa o CSV de reviews (dataset público, sem token) para data\raw\
python -c "import kagglehub,shutil,zipfile; p=kagglehub.dataset_download('olistbr/brazilian-ecommerce', path='olist_order_reviews_dataset.csv'); (zipfile.ZipFile(p).extractall('data/raw') if zipfile.is_zipfile(p) else shutil.copy(p,'data/raw/'))"
python src\fase0_perfil.py   # -> reports\fase0_perfil.json
python src\fase0_gold.py     # -> data\gold\gold_para_rotular.xlsx, data\gold\gold_meta.csv
```

Guia de rotulagem: [docs/guia_rotulagem.md](docs/guia_rotulagem.md).
