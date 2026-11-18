"""Fase 5 - prepara e publica no Hugging Face Hub: modelo BERTimbau, adaptador LoRA do Tucano2, dataset e Space.

Todos os números dos cards vêm de arquivos de execução:
  results/fase4/metricas.json (teste), results/fase4/latencia.json (Kaggle), results/<run>/registro_mlflow.json
  (treino) e reports/fase1_concordancia/metricas.json (kappa entre rotuladores).

    python src\\fase5_hub.py preparar --usuario <usuario-hf>   # monta dist/hub/ para revisão (sem rede)
    python src\\fase5_hub.py publicar                          # usa HF_TOKEN do .env; SÓ com autorização
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd

from dados import RAIZ
from rotulos import CLASSES, carregar_gold, carregar_silver

DIST = RAIZ / "dist" / "hub"
NOMES_REPO = {"modelo": "bertimbau-motivo-avaliacoes-olist", "adaptador": "tucano2-1.5b-qlora-motivo-avaliacoes-olist",
              "dataset": "olist-motivo-avaliacoes", "space": "motivo-avaliacoes-olist"}
OLIST = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
GITHUB = "https://github.com/AbnerRidigolo/fine-tuning-hugging-face"
DESC = {
    "atraso_ou_nao_entrega": "o pedido atrasou (passou do prazo prometido) ou não chegou",
    "produto_errado_ou_incompleto": "veio outro produto, outra cor/tamanho, ou faltando itens/partes",
    "defeito_ou_avaria": "chegou quebrado, danificado ou não funciona",
    "qualidade_abaixo_do_esperado": "funciona, mas a qualidade ou o material decepcionou",
    "atendimento_vendedor": "problema de comunicação, troca, reembolso ou nota com o vendedor/loja",
    "elogio": "comentário positivo, sem problema concreto",
    "outro": "nenhuma das anteriores (inclui demora dentro do prazo, comentários neutros ou sem motivo claro)",
}


LIMITES = """## Limitações e vieses

- **Rótulos de LLM, sem validação humana.** As métricas medem concordância com outro LLM que seguiu o mesmo guia,
  não acerto em relação a pessoas. Erros sistemáticos compartilhados pelos dois rotuladores não aparecem nas métricas.
- **Confiança mal calibrada.** O modelo erra com probabilidade alta com frequência (veja a calibração abaixo da tabela);
  não use a probabilidade como medida de certeza sem recalibrar (por exemplo, *temperature scaling* na validação).
- **Domínio.** Comentários curtos (mediana de {mediana_chars} caracteres, máximo de {max_chars}) de um único marketplace
  brasileiro, escritos entre {inicio} e {fim}. Outros domínios (restaurantes, entregas de comida), gírias recentes e
  textos longos não foram avaliados.
- **Amostragem.** Treino e teste foram balanceados por nota (1 a 5); no dataset original, {pct_nota5} dos comentários
  têm nota 5. A distribuição de classes em produção será diferente.
- **Classe mais difícil:** `{pior_classe}` (menor F1 no teste). A fronteira entre `qualidade_abaixo_do_esperado`
  e `produto_errado_ou_incompleto` também concentra confusões.
- **Uso pretendido:** triagem e análise agregada de avaliações. Não use para decisões automáticas sobre clientes ou
  vendedores.
"""


def perfil(met, k):
    p = ler("reports/fase0_perfil.json")
    r = pd.read_csv(RAIZ / "data" / "raw" / "olist_order_reviews_dataset.csv", usecols=["review_comment_message", "review_creation_date"])
    datas = pd.to_datetime(r[r.review_comment_message.notna()].review_creation_date)
    nc = p["notas_com_comentario"]
    return {"mediana_chars": int(p["chars"]["p50"]), "max_chars": p["chars"]["max"],
            "inicio": datas.min().strftime("%m/%Y"), "fim": datas.max().strftime("%m/%Y"),
            "pct_nota5": f"{nc['5'] / sum(nc.values()):.0%}",
            "pior_classe": min(CLASSES, key=lambda cl: met[k][f"f1_{cl}"])}


def ler(caminho):
    return json.loads((RAIZ / caminho).read_text(encoding="utf-8"))


def f3(v):
    return f"{v:.3f}".replace(".", ",")


def dec(v, casas):
    return f"{v:.{casas}f}".replace(".", ",")


def pct(v, casas=1):
    return f"{v:.{casas}%}".replace(".", ",")


def milhar(n):
    return f"{n:,}".replace(",", ".")


def f1c(v):
    return "—" if v is None else f"{v:.1f}".replace(".", ",")


def tabela(cab, linhas):
    s = "| " + " | ".join(cab) + " |\n|" + "---|" * len(cab) + "\n"
    return s + "".join("| " + " | ".join(str(c) for c in l) + " |\n" for l in linhas)


def contexto(usuario):
    m, lat = ler("results/fase4/metricas.json"), ler("results/fase4/latencia.json")
    return {"u": usuario, "met": m["metricas"], "boot": m["bootstrap"], "lat": lat, "kappa": ler("reports/fase1_concordancia/metricas.json"),
            "reg_b": ler("results/bertimbau/registro_mlflow.json"), "reg_t": ler("results/qlora_tucano2_1.5b/registro_mlflow.json"),
            "repo": {k: f"{usuario}/{v}" for k, v in NOMES_REPO.items()}}


def bloco_comum(c, k):
    """Seções compartilhadas pelos dois model cards: dados, classes, avaliação, limitações e licença."""
    kap, met, boot, hw = c["kappa"], c["met"], c["boot"], c["lat"]["hardware"]
    classes = tabela(["Classe", "Quando usar"], [[f"`{k}`", v] for k, v in DESC.items()])
    nomes = {"baseline_tfidf_lr": "TF-IDF + LR", "bertimbau": "BERTimbau", "qlora_tucano2_1.5b": "Tucano2 1.5B QLoRA"}
    linhas = [["F1 macro"] + [f3(met[k]["f1_macro"]) for k in nomes],
              ["IC 95% (bootstrap)"] + [f"{f3(boot[k]['ic95'][0])}–{f3(boot[k]['ic95'][1])}" for k in nomes],
              ["Acurácia"] + [f3(met[k]["acuracia"]) for k in nomes]]
    linhas += [[f"F1 `{cl}`"] + [f3(met[k][f"f1_{cl}"]) for k in nomes] for cl in CLASSES]
    linhas += [["Latência GPU T4, mediana (ms)"] + [f1c(c["lat"][k].get("gpu_ms")) for k in nomes],
               [f"Latência CPU ({hw['torch_threads_cpu']} threads), mediana (ms)"] + [f1c(c["lat"][k]["cpu_ms"]) for k in nomes],
               ["Tamanho em disco (MB)"] + [f"{c['lat'][k]['disco_mb']:.0f}" for k in nomes]]
    dif = boot["diferencas"]["qlora_tucano2_1.5b_menos_bertimbau"]
    return {
        "dados": f"""## Dados

Comentários de avaliações do [Brazilian E-Commerce Public Dataset by Olist]({OLIST}), só as linhas com
comentário não vazio, deduplicadas. Dataset derivado com os rótulos: [{c['repo']['dataset']}](https://huggingface.co/datasets/{c['repo']['dataset']}).

- **Treino e validação (silver):** {milhar(c['reg_b']['params']['n_treino'])} + {c['reg_b']['params']['n_validacao']} comentários
  (amostra estratificada por nota), rotulados pelo **gpt-6.1-terra** seguindo um guia de rotulagem com 7 classes e regras de desempate.
- **Teste (referência):** {c['reg_b']['params']['n_teste']} comentários (80 por nota de 1 a 5), separados antes de tudo,
  rotulados por um modelo maior (**gpt-6.1-sol**, raciocínio médio) com o mesmo guia. **Não há rótulo humano.**
  A concordância entre os dois rotuladores LLM nesses 400 itens foi kappa de Cohen {f3(kap['kappa_cohen'])}
  ({pct(kap['concordancia'])} de concordância).
""",
        "classes": "## Classes\n\nUm único rótulo por comentário: o motivo **principal**.\n\n" + classes,
        "avaliacao": f"""## Avaliação no teste de referência (400 comentários)

Os três modelos do estudo, avaliados no mesmo conjunto. Latência por exemplo com lote de 1 e tokenização incluída,
mediana de 400 chamadas após 10 de aquecimento, medida numa única sessão do Kaggle (GPU {hw['gpu']}; CPU {hw['cpu']}).
O Tucano2 roda em 4 bits na GPU e com o adaptador mesclado em fp32 na CPU.

{tabela(['', *nomes.values()], linhas)}
Diferença de F1 macro Tucano2 − BERTimbau: {dif['observada']:+.3f} (IC 95% {dif['ic95'][0]:+.3f} a {dif['ic95'][1]:+.3f};
P(diferença > 0) = {pct(dif['prob_maior_que_zero'], 0)}, bootstrap pareado com 2.000 reamostragens).
""".replace("+0.", "+0,").replace("-0.", "−0,"),
        "limites": LIMITES.format(**perfil(met, k)),
        "licenca": f"""## Licença e crédito

Os dados de origem são da **Olist** ([Brazilian E-Commerce Public Dataset by Olist]({OLIST})), licenciados sob
**CC BY-NC-SA 4.0**. Este modelo é derivado desses dados e é distribuído sob a mesma licença: uso **não comercial**,
com atribuição à Olist e compartilhamento pela mesma licença.
""" + (f"\nCódigo do estudo: {GITHUB}\n" if GITHUB else ""),
    }


def calibracao_md(met, k):
    m = met[k]
    d2 = lambda v: f"{v:.2f}".replace(".", ",")
    return (f"\n**Calibração no teste:** {m['erros']} erros em 400; confiança média nos erros {d2(m['conf_media_erros'])}"
            f" (nos acertos, {d2(m['conf_media_acertos'])}); {m['erros_conf_ge_095']} erros com confiança ≥ 0,95; "
            f"ECE (10 faixas) {f3(m['ece'])}.\n")


def card_modelo(c):
    b, p = bloco_comum(c, "bertimbau"), c["reg_b"]["params"]
    yaml = f"""---
language: pt
license: cc-by-nc-sa-4.0
library_name: transformers
pipeline_tag: text-classification
base_model: neuralmind/bert-base-portuguese-cased
datasets:
- {c['repo']['dataset']}
tags:
- portuguese
- e-commerce
- reviews
- olist
metrics:
- f1
---
"""
    return yaml + f"""# BERTimbau — motivo principal de avaliações de e-commerce

Classifica um comentário de compra online em português em uma de 7 classes de motivo (atraso, produto errado,
defeito, qualidade, atendimento, elogio, outro). É o `neuralmind/bert-base-portuguese-cased` com ajuste fino completo.

Faz parte de um estudo que comparou este modelo a um LLM de 1,5 B parâmetros ajustado com QLoRA
([{c['repo']['adaptador']}](https://huggingface.co/{c['repo']['adaptador']})) e a um baseline TF-IDF.
Demo: [Space](https://huggingface.co/spaces/{c['repo']['space']}).

## Uso

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{c['repo']['modelo']}", top_k=None)
clf("O produto chegou quebrado e a loja não responde.")
```

{b['classes']}
{b['dados']}
## Treino

Ajuste fino completo, lr {p['lr']}, lote {p['batch']}, weight decay {p['weight_decay']}, warmup de {p['warmup_frac']:.0%},
`max_length` {p['max_len']}, até {p['epocas_max']} épocas com early stopping (paciência {p['paciencia']}) pelo F1 macro
da validação; melhor época {p['melhor_epoca']:.0f} de {p['epocas_treinadas']:.0f}. Seed {p['seed']}. GPU: {p['gpu']}.
F1 macro na validação (silver): {f3(c['reg_b']['metrics']['validacao_f1_macro'])}.

{b['avaliacao']}{calibracao_md(c['met'], 'bertimbau')}
{b['limites']}
{b['licenca']}"""


def card_adaptador(c):
    b, p, r = bloco_comum(c, "qlora_tucano2_1.5b"), c["reg_t"]["params"], c["reg_t"]["metrics"]
    t = c["lat"]["qlora_tucano2_1.5b"]
    yaml = f"""---
language: pt
license: cc-by-nc-sa-4.0
library_name: peft
pipeline_tag: text-classification
base_model: Polygl0t/Tucano2-qwen-1.5B-Instruct
datasets:
- {c['repo']['dataset']}
tags:
- portuguese
- e-commerce
- reviews
- olist
- lora
- qlora
metrics:
- f1
---
"""
    lab = "{i: c for i, c in enumerate(CLASSES)}"
    return yaml + f"""# Tucano2 1.5B + LoRA (QLoRA) — motivo principal de avaliações de e-commerce

Adaptador LoRA para classificação de sequência sobre o
[Polygl0t/Tucano2-qwen-1.5B-Instruct](https://huggingface.co/Polygl0t/Tucano2-qwen-1.5B-Instruct) (Apache 2.0),
treinado com QLoRA (base em 4 bits). Classifica um comentário de compra online em português em uma de 7 classes de motivo.
Este repositório tem só o adaptador ({t['disco_adaptador_mb']:.0f} MB) e a cabeça de classificação; o modelo base é baixado à parte.

Comparação do estudo: [BERTimbau]({'https://huggingface.co/' + c['repo']['modelo']}) e baseline TF-IDF.

## Uso (GPU, 4 bits, como no treino)

```python
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

ADAPT = "{c['repo']['adaptador']}"
CLASSES = {CLASSES}
tok = AutoTokenizer.from_pretrained(ADAPT)
base = AutoModelForSequenceClassification.from_pretrained(
    "Polygl0t/Tucano2-qwen-1.5B-Instruct", num_labels=len(CLASSES),
    id2label={lab}, label2id={{c: i for i, c in enumerate(CLASSES)}},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
    device_map={{"": 0}}, dtype=torch.float16)
base.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(base, ADAPT).eval()

with torch.inference_mode():
    logits = model(**tok("Veio a cor errada.", return_tensors="pt").to(0)).logits
print(CLASSES[logits.argmax(-1).item()])
```

Em CPU, carregue o base em `torch.float32` sem `quantization_config` e use `PeftModel.from_pretrained(...).merge_and_unload()`.
Na medição do estudo, a versão mesclada em CPU concordou com a de 4 bits em {pct(t['concordancia_cpu_vs_gpu'])} dos 400 itens de teste.

{b['classes']}
{b['dados']}
## Treino

QLoRA: base quantizado em NF4 com double quant, compute em fp16; LoRA r={p['lora_r']}, alpha={p['lora_alpha']},
dropout {p['lora_dropout']}, em `{p['lora_alvos']}`, mais a cabeça de classificação treinada por inteiro
({f1c(p['params_treinaveis'] / 1e6)} M parâmetros treináveis de {dec(p['params_total'] / 1e9, 2)} B). lr {p['lr']}, lote {p['batch']},
cosine com warmup de {p['warmup_frac']:.0%}, `max_length` {p['max_len']}, até {p['epocas_max']} épocas com early stopping
(paciência {p['paciencia']}); a melhor época foi a última ({p['melhor_epoca']:.0f}), então o early stopping não chegou a agir.
Seed {p['seed']}. GPU: {p['gpu']}; pico de memória no treino {f1c(r['pico_memoria_gpu_treino_gb'])} GB; tempo de treino
{r['tempo_treino_s'] / 60:.0f} min. F1 macro na validação (silver): {f3(r['validacao_f1_macro'])}.

{b['avaliacao']}{calibracao_md(c['met'], 'qlora_tucano2_1.5b')}
{b['limites']}
{b['licenca']}""".replace("Este modelo é derivado", "Este adaptador é derivado")


def card_dataset(c, n):
    kap = c["kappa"]
    yaml = f"""---
language: pt
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
tags:
- portuguese
- e-commerce
- reviews
- olist
- llm-labeled
pretty_name: Motivo principal de avaliações Olist (rótulos por LLM)
size_categories:
- 1K<n<10K
configs:
- config_name: silver
  data_files:
  - split: train
    path: silver/treino.csv
  - split: validation
    path: silver/validacao.csv
- config_name: teste_referencia
  data_files:
  - split: test
    path: teste/teste_referencia.csv
---
"""
    dist = tabela(["Classe", "silver treino", "silver validação", "teste"],
                  [[f"`{k}`", n["treino"].get(k, 0), n["validacao"].get(k, 0), n["teste"].get(k, 0)] for k in CLASSES])
    return yaml + f"""# Motivo principal de avaliações de e-commerce (Olist) — rótulos por LLM

Comentários em português do [Brazilian E-Commerce Public Dataset by Olist]({OLIST}), cada um com o motivo
**principal** da avaliação em 7 classes. Os rótulos foram gerados por LLM seguindo um guia de rotulagem
(`guia_rotulagem.md`, incluído aqui), **sem revisão humana**.

## Configurações

| Configuração | Split | Linhas | Rotulador |
|---|---|---|---|
| `silver` | `train` | {milhar(sum(n['treino'].values()))} | gpt-6.1-terra (sem raciocínio) |
| `silver` | `validation` | {sum(n['validacao'].values())} | gpt-6.1-terra (sem raciocínio) |
| `teste_referencia` | `test` | {sum(n['teste'].values())} | gpt-6.1-sol (raciocínio médio) |

O teste foi sorteado primeiro (80 comentários por nota, seed 42) e excluído do pool do silver. O silver tem 1.000
comentários por nota, com split treino/validação estratificado pelo rótulo (seed 42).
Nos 400 itens do teste, o rotulador do silver concordou com o de referência em {pct(kap['concordancia'])}
(kappa de Cohen {f3(kap['kappa_cohen'])}).

```python
from datasets import load_dataset
silver = load_dataset("{c['repo']['dataset']}", "silver")
teste = load_dataset("{c['repo']['dataset']}", "teste_referencia")
```

## Colunas

- `review_id`, `review_score` (1 a 5) e `texto`: do dataset original da Olist (texto sem correção).
- `rotulo`: uma das 7 classes abaixo.
- `confianca`: confiança declarada pelo LLM (`alta`, `media` ou `baixa`), não calibrada.
- `modelo_rotulador`: o LLM que gerou o rótulo.
- `id_teste` (só no teste): identificador estável usado nos relatórios do estudo.

## Classes

{tabela(['Classe', 'Quando usar'], [[f'`{k}`', v] for k, v in DESC.items()])}
## Distribuição

{dist}
## Limitações

- Rótulos de LLM sem validação humana; no teste, servem como referência, não como verdade.
- Amostragem balanceada por nota, diferente da distribuição original (metade dos comentários com nota 5).
- Comentários de 10/2016 a 08/2018 de um único marketplace.

## Licença e crédito

Dados de origem: **Olist**, [Brazilian E-Commerce Public Dataset by Olist]({OLIST}), **CC BY-NC-SA 4.0**. Este dataset
derivado é distribuído sob a mesma licença (uso não comercial, com atribuição e compartilhamento pela mesma licença).
"""


def card_space(c):
    return f"""---
title: Motivo da avaliação
emoji: 🛒
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
license: cc-by-nc-sa-4.0
models:
- {c['repo']['modelo']}
datasets:
- {c['repo']['dataset']}
short_description: Motivo principal de avaliações de e-commerce em português
---

Demo do modelo [{c['repo']['modelo']}](https://huggingface.co/{c['repo']['modelo']}). Dados: Olist (CC BY-NC-SA 4.0).
"""


def preparar(usuario: str) -> None:
    c = contexto(usuario)
    if os.path.exists(DIST):
        shutil.rmtree(DIST)
    # modelo BERTimbau
    d = DIST / "modelo"
    d.mkdir(parents=True)
    for f in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(RAIZ / "models" / "bertimbau" / f, d)
    (d / "README.md").write_text(card_modelo(c), encoding="utf-8")
    # adaptador
    d = DIST / "adaptador"
    d.mkdir()
    for f in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(RAIZ / "models" / "qlora_tucano2_1.5b" / f, d)
    (d / "README.md").write_text(card_adaptador(c), encoding="utf-8")
    # dataset
    d = DIST / "dataset"
    (d / "silver").mkdir(parents=True)
    (d / "teste").mkdir()
    s = carregar_silver().rename(columns={"modelo": "modelo_rotulador"})
    cols = ["review_id", "review_score", "texto", "rotulo", "confianca", "modelo_rotulador"]
    for split, nome in (("treino", "treino"), ("validacao", "validacao")):
        s[s.split == split][cols].to_csv(d / "silver" / f"{nome}.csv", index=False, encoding="utf-8")
    g = carregar_gold().rename(columns={"modelo": "modelo_rotulador", "id_gold": "id_teste"})
    g[["id_teste", *cols]].to_csv(d / "teste" / "teste_referencia.csv", index=False, encoding="utf-8")
    shutil.copy(RAIZ / "docs" / "guia_rotulagem.md", d)
    n = {"treino": s[s.split == "treino"].rotulo.value_counts().to_dict(),
         "validacao": s[s.split == "validacao"].rotulo.value_counts().to_dict(), "teste": g.rotulo.value_counts().to_dict()}
    (d / "README.md").write_text(card_dataset(c, n), encoding="utf-8")
    # space
    d = DIST / "space"
    d.mkdir()
    m = c["met"]["bertimbau"]
    app = (RAIZ / "space" / "app.py").read_text(encoding="utf-8")
    for k, v in {"__N_TREINO__": f"{c['reg_b']['params']['n_treino']:,}".replace(",", "."), "__F1__": f3(m["f1_macro"]),
                 "__ERROS_095__": str(m["erros_conf_ge_095"]), "__ERROS__": str(m["erros"])}.items():
        app = app.replace(k, v)
    assert "__" not in app.split("AVISO", 1)[1].split('"""', 2)[1], "placeholder não preenchido"
    (d / "app.py").write_text(app, encoding="utf-8")
    shutil.copy(RAIZ / "space" / "requirements.txt", d)
    (d / "README.md").write_text(card_space(c), encoding="utf-8")
    (DIST / "repos.json").write_text(json.dumps(c["repo"], indent=2), encoding="utf-8")
    print(f"pronto para revisão em {DIST}")
    for k, v in c["repo"].items():
        print(f"  {k:10s} -> {v}")


def publicar() -> None:
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(RAIZ / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN vazio no .env")
    api = HfApi(token=token)
    usuario = api.whoami()["name"]
    preparar(usuario)
    repo = json.loads((DIST / "repos.json").read_text(encoding="utf-8"))
    for chave, tipo in (("dataset", "dataset"), ("modelo", "model"), ("adaptador", "model")):
        api.create_repo(repo[chave], repo_type=tipo, private=False, exist_ok=True)
        api.upload_folder(repo_id=repo[chave], repo_type=tipo, folder_path=DIST / chave,
                          commit_message="Publicação do estudo (fase 5)")
        print("publicado:", f"https://huggingface.co/{'datasets/' if tipo == 'dataset' else ''}{repo[chave]}")
    api.create_repo(repo["space"], repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    api.add_space_variable(repo["space"], "MODELO_ID", repo["modelo"])
    api.upload_folder(repo_id=repo["space"], repo_type="space", folder_path=DIST / "space",
                      commit_message="Publicação do estudo (fase 5)")
    print("publicado:", f"https://huggingface.co/spaces/{repo['space']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preparar").add_argument("--usuario", required=True)
    sub.add_parser("publicar")
    a = ap.parse_args()
    preparar(a.usuario) if a.cmd == "preparar" else publicar()


if __name__ == "__main__":
    main()
