# Motivo de avaliações de e-commerce em português: BERTimbau vs LLM pequeno com QLoRA

**Pergunta:** um LLM pequeno ajustado com QLoRA supera o BERTimbau o suficiente para justificar o custo de inferência?

**Resposta curta:** não, nesta tarefa. O Tucano2 1.5B com QLoRA teve F1 macro 0,822 contra 0,807 do BERTimbau, mas a
diferença (+0,014) fica dentro do ruído do teste (IC 95% de −0,034 a +0,062). O custo é bem maior: latência 13,6×
maior na GPU, 15,7× maior na CPU e 7× mais espaço em disco. Para produção, o BERTimbau é a escolha.

Dados: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(CC BY-NC-SA 4.0). Dataset e modelos derivados são distribuídos sob a mesma licença, com crédito à Olist.

## Links

<!-- preenchidos na publicação -->
- Modelo BERTimbau: _a publicar_
- Adaptador LoRA do Tucano2: _a publicar_
- Dataset (silver e teste): _a publicar_
- Demo (Space Gradio): _a publicar_

## Tarefa

Dado o comentário de uma avaliação, prever o motivo **principal** em 7 classes: `atraso_ou_nao_entrega`,
`produto_errado_ou_incompleto`, `defeito_ou_avaria`, `qualidade_abaixo_do_esperado`, `atendimento_vendedor`, `elogio`
e `outro`. As definições, os exemplos e as regras de desempate estão no [guia de rotulagem](docs/guia_rotulagem.md).

## Desenho do estudo

| Etapa | O que foi feito |
|---|---|
| Dados | 40.950 comentários não vazios em 99.224 avaliações; depois de deduplicar, sobra um pool de 34.947 ([perfil](reports/fase0_perfil.json)). |
| Teste | 400 comentários (80 por nota de 1 a 5, seed 42), separados antes de tudo. Rótulos de referência do **gpt-6.1-sol** (raciocínio médio) seguindo o guia. **Não há rótulo humano** (veja [Limitações](#limitações)). |
| Treino (silver) | 5.000 comentários do restante do pool (1.000 por nota), rotulados pelo **gpt-6.1-terra** via API com saída estruturada (Pydantic). Treino de 4.250 e validação de 750, estratificados. |
| Qualidade do silver | Nos 400 do teste, os dois rotuladores concordam em 96,0% (kappa de Cohen 0,950) ([métricas](reports/fase1_concordancia/metricas.json)). |
| Custo de rotulagem | US$ 1,86 em API no total (silver, referência e concordância). |
| Modelos | (1) TF-IDF de palavras e caracteres + regressão logística; (2) BERTimbau base com ajuste fino completo; (3) [Tucano2-qwen-1.5B-Instruct](https://huggingface.co/Polygl0t/Tucano2-qwen-1.5B-Instruct) (Apache 2.0) com QLoRA em 4 bits e cabeça de classificação. Seed 42; early stopping pelo F1 macro da validação. |
| Avaliação | F1 macro e por classe no teste, IC 95% por bootstrap pareado (2.000 reamostragens), calibração e matrizes de confusão. A latência foi medida com lote de 1 e tokenização incluída, numa única sessão do Kaggle (GPU T4 e CPU Xeon de 2 threads), para os três modelos rodarem no mesmo hardware. |

Todos os runs ficam no MLflow (`sqlite:///mlflow.db`, experimento `motivo-avaliacoes-olist`).

## Resultados no teste (400 comentários)

Fonte: [reports/fase4_avaliacao.md](reports/fase4_avaliacao.md), gerado por `src/fase4_avaliacao.py`.

| | TF-IDF + LR | BERTimbau | Tucano2 1.5B QLoRA |
|---|---|---|---|
| **F1 macro** | 0,760 | 0,807 | **0,822** |
| IC 95% F1 macro | 0,703–0,808 | 0,752–0,848 | 0,769–0,865 |
| Acurácia | 0,812 | 0,855 | 0,875 |
| F1 atraso (n=81) | 0,830 | 0,873 | 0,916 |
| F1 errado/incompleto (n=69) | 0,837 | 0,844 | 0,869 |
| F1 defeito (n=32) | 0,769 | 0,814 | 0,833 |
| F1 qualidade (n=49) | 0,739 | 0,792 | 0,804 |
| F1 atendimento (n=17) | 0,743 | 0,812 | 0,812 |
| F1 elogio (n=130) | 0,880 | 0,940 | 0,946 |
| F1 outro (n=22) | 0,524 | 0,578 | 0,571 |
| Latência GPU T4, mediana / p95 (ms) | — | 7,3 / 8,2 | 99,0 / 105,2 |
| Latência CPU 2 threads, mediana / p95 (ms) | 2,2 / 2,6 | 61,2 / 89,7 | 963,0 / 1.272,6 |
| Tamanho em disco (MB) | 2 | 416 | 2.947 (base 2.880 + adaptador 67) |
| Erros com confiança ≥ 0,95 | 1 de 75 | 33 de 58 | 29 de 50 |
| ECE (10 faixas) | 0,060 | 0,113 | 0,100 |

O Tucano2 roda em 4 bits na GPU, como no treino. Na CPU ele roda com o adaptador mesclado num base fp32, e aí o F1
cai para 0,811 (95,2% de concordância com a versão em 4 bits).

**Diferenças de F1 macro (bootstrap pareado):**

| Comparação | Diferença | IC 95% | P(diferença > 0) |
|---|---|---|---|
| BERTimbau − TF-IDF | +0,047 | −0,009 a +0,105 | 94,8% |
| Tucano2 − TF-IDF | +0,061 | +0,014 a +0,108 | 99,4% |
| Tucano2 − BERTimbau | +0,014 | −0,034 a +0,062 | 73,0% |

Cinco exemplos reais de discordância, com explicação: [reports/fase4_exemplos_discordancia.md](reports/fase4_exemplos_discordancia.md).

## Conclusão

1. **Os dois modelos contextuais superam o TF-IDF**, mas só o ganho do Tucano2 tem IC inteiramente acima de zero.
   O TF-IDF erra sobretudo quando um elogio cerca o problema (7 dos 81 atrasos viraram `elogio`).
2. **O Tucano2 não se separa do BERTimbau.** Ele é melhor em `atraso` (0,916 contra 0,873) e tem 8 erros a menos em 400,
   mas a diferença no F1 macro é compatível com zero. Com 400 itens de teste, não dá para afirmar que ele é melhor.
3. **O custo, por outro lado, é claro e grande:** 99 ms contra 7,3 ms por comentário na T4, quase 1 s contra 61 ms na
   CPU, 2,9 GB contra 416 MB, e uma GPU necessária para servir em 4 bits. Para classificar avaliações em volume, o
   BERTimbau entrega quase a mesma qualidade por uma fração do custo. Por isso o Space usa o BERTimbau.
4. **Nenhum dos dois é bem calibrado.** Mais da metade dos erros saem com confiança ≥ 0,95. A confiança exibida não deve
   ser lida como probabilidade sem recalibrar antes.

O que mudaria a conclusão: um teste maior, com rótulo humano, para estreitar o IC; e mais épocas para o Tucano2, cuja
melhor época foi a última (o early stopping não chegou a agir).

## Limitações

- **O teste foi rotulado por LLM, sem validação humana.** As métricas medem concordância com o gpt-6.1-sol, não
  acerto em relação a pessoas. O kappa de 0,950 entre os dois rotuladores mede consistência entre LLMs, não correção.
- O teste tem 400 itens e é balanceado por nota; a distribuição real de classes é outra (50% dos comentários têm nota 5).
- O BERTimbau foi treinado numa RTX 3050 local e o Tucano2 numa T4 do Kaggle. A latência, por outro lado, foi medida
  para os três modelos na mesma sessão.
- A CPU de latência é a do Kaggle (Xeon de 2,0 GHz, 2 threads no torch). Os tempos absolutos
  dependem do hardware, e a medição não foi repetida em outras máquinas.

## Como reproduzir (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# torch: escolha UM dos dois
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu     # só CPU
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132   # GPU NVIDIA (CUDA 13.x)
python -m pip install -r requirements.txt
Copy-Item .env.example .env   # preencha os tokens; o .env não vai para o git
```

| Fase | Comandos | Saída |
|---|---|---|
| 0. Dados e teste | `python src\fase0_perfil.py`; `python src\fase0_gold.py` | `reports\fase0_perfil.json`, `data\gold\` |
| 1. Rotulagem e baseline | `python src\fase1_amostra_silver.py`; `python src\rotulador.py ...` (custa API); `python src\fase1_split.py`; `python src\fase1_concordancia.py`; `python src\fase1_baseline.py` | `data\silver\`, `reports\fase1_concordancia\`, MLflow |
| 2. BERTimbau | `python src\empacotar_dados.py`; rodar `notebooks\fase2_bertimbau.ipynb` (GPU local, Kaggle ou Colab); `python src\importar_resultados.py <zip>` | `results\bertimbau\`, MLflow |
| 3. Tucano2 QLoRA | `python src\kaggle_preparar.py`; `python src\kaggle_cli.py kernels push -p <pasta absoluta>\kaggle\fase3 --accelerator NvidiaTeslaT4` | `results\qlora_tucano2_1.5b\` |
| 4. Avaliação | `python src\kaggle_preparar.py fase4` e push de `kaggle\fase4` (latência na T4); `python src\fase4_avaliacao.py` | `results\fase4\`, `reports\fase4_avaliacao.md`, MLflow |
| 5. Publicação | `python src\fase5_hub.py preparar --usuario <usuario-hf>` (revisão local); `python src\fase5_hub.py publicar` | `dist\hub\`, Hugging Face Hub |

Para baixar o CSV de avaliações (dataset público, sem token):

```powershell
python -c "import kagglehub,shutil,zipfile; p=kagglehub.dataset_download('olistbr/brazilian-ecommerce', path='olist_order_reviews_dataset.csv'); (zipfile.ZipFile(p).extractall('data/raw') if zipfile.is_zipfile(p) else shutil.copy(p,'data/raw/'))"
```

Para ver os experimentos: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Licença

Dados, dataset derivado e modelos: **CC BY-NC-SA 4.0**, derivados
do *Brazilian E-Commerce Public Dataset by Olist*. O uso é não comercial e exige atribuição à Olist.
