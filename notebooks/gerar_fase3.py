"""Gera notebooks/fase3_qlora_tucano2.ipynb (fonte única das células).

    python notebooks\\gerar_fase3.py            -> escreve o .ipynb
    python notebooks\\gerar_fase3.py --smoke X  -> escreve X.py com as células (sem pip) para teste local
"""
import sys
from pathlib import Path

import nbformat as nbf

MD, CODE = "md", "code"
CELULAS = [
(MD, """# Fase 3 — Tucano2 1.5B com QLoRA para o motivo de avaliações (Olist)

`Polygl0t/Tucano2-qwen-1.5B-Instruct` (Apache 2.0) quantizado em 4 bits (NF4) + LoRA, com cabeça de
classificação de 7 classes (`AutoModelForSequenceClassification`). Treinado nos rótulos silver
(gpt-6.1-terra) e avaliado no teste de referência de 400 itens (rótulos gpt-6.1-sol, **sem validação humana**).

Feito para caber numa GPU T4 de 16 GB (Kaggle/Colab). O pico de memória de GPU é registrado no fim.

**Como rodar**
- **Kaggle:** anexe o dataset com `dados_treino.zip`, ative *GPU T4* e *Internet*, rode tudo.
- **Colab:** *GPU T4*, rode tudo e envie o `dados_treino.zip` quando pedido.

Saída: `saida_fase3.zip` (adaptador LoRA + cabeça, métricas, predições, registro MLflow).
Localmente: `python src\\importar_resultados.py saida_fase3.zip`.

Dados: Olist (CC BY-NC-SA 4.0)."""),
(CODE, """%pip install -q transformers==5.22.2 accelerate==1.15.0 peft==0.21.2 bitsandbytes==0.50.2 scikit-learn==1.9.1"""),
(CODE, """import glob, json, os, platform, shutil, time, zipfile
import numpy as np, pandas as pd, torch, transformers, peft, bitsandbytes
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

SMOKE = os.environ.get("SMOKE") == "1"   # teste local rápido; no Kaggle/Colab fica False
SEED = 42
MODELO_BASE = "Polygl0t/Tucano2-qwen-1.5B-Instruct"
CLASSES = ["atraso_ou_nao_entrega", "produto_errado_ou_incompleto", "defeito_ou_avaria",
           "qualidade_abaixo_do_esperado", "atendimento_vendedor", "elogio", "outro"]
L2I = {c: i for i, c in enumerate(CLASSES)}
SAIDA = "/kaggle/working/saida_fase3" if os.path.isdir("/kaggle/working") else os.path.abspath("saida_fase3")
os.makedirs(SAIDA, exist_ok=True)

transformers.set_seed(SEED)
assert torch.cuda.is_available(), "QLoRA precisa de GPU: ative a T4"
GPU_NOME = torch.cuda.get_device_name(0)
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__,
      "| bitsandbytes", bitsandbytes.__version__, "| GPU:", GPU_NOME)"""),
(MD, "## Dados"),
(CODE, """def achar_dados():
    for base in ["/kaggle/input", os.environ.get("DADOS_DIR", ""), "."]:
        if base and (hits := glob.glob(f"{base}/**/silver.csv", recursive=True)):
            return os.path.dirname(hits[0])
    try:
        from google.colab import files  # Colab: pede o zip
        nome = next(iter(files.upload()))
        zipfile.ZipFile(nome).extractall("dados")
        return "dados"
    except ImportError:
        raise FileNotFoundError("silver.csv não encontrado: anexe o dataset ou defina DADOS_DIR")

DIR = achar_dados()
silver = pd.read_csv(f"{DIR}/silver.csv")
teste = pd.read_csv(f"{DIR}/teste_referencia.csv")
treino, valid = silver[silver.split == "treino"], silver[silver.split == "validacao"]
if SMOKE:
    treino, valid, teste = treino.sample(64, random_state=SEED), valid.sample(32, random_state=SEED), teste.head(16)
print(len(treino), len(valid), len(teste))"""),
(CODE, """tok = AutoTokenizer.from_pretrained(MODELO_BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
n_tok = [len(tok(t)["input_ids"]) for t in silver.texto]
print("tokens por comentário: p50=%d p95=%d p99=%d max=%d" % tuple(np.percentile(n_tok, [50, 95, 99, 100])))
MAX_LEN = 128

class DS(torch.utils.data.Dataset):
    def __init__(self, df):
        self.enc = tok(list(df.texto), truncation=True, max_length=MAX_LEN)
        self.y = [L2I[r] for r in df.rotulo]
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.enc.items()}, "labels": self.y[i]}

ds_tr, ds_va, ds_te = DS(treino), DS(valid), DS(teste)"""),
(MD, "## Modelo em 4 bits + LoRA"),
(CODE, """bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
modelo = AutoModelForSequenceClassification.from_pretrained(
    MODELO_BASE, num_labels=len(CLASSES), id2label=dict(enumerate(CLASSES)), label2id=L2I,
    quantization_config=bnb_cfg, device_map={"": 0}, dtype=torch.float16)
modelo.config.pad_token_id = tok.pad_token_id
modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=False)  # textos curtos

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
modelo = get_peft_model(modelo, LoraConfig(task_type=TaskType.SEQ_CLS, **LORA))
modelo.print_trainable_parameters()
treinaveis, total = modelo.get_nb_trainable_parameters()"""),
(MD, "## Treino (seed fixa, validação separada, early stopping no F1 macro)"),
(CODE, """def metricas(p):
    pred = p.predictions.argmax(-1)
    return {"f1_macro": f1_score(p.label_ids, pred, average="macro", labels=range(len(CLASSES)), zero_division=0),
            "acuracia": (pred == p.label_ids).mean()}

HP = dict(lr=2e-4, batch=16, epocas_max=5, paciencia=2, weight_decay=0.0, warmup_frac=0.1)
if SMOKE: HP.update(epocas_max=1, batch=8)
passos_ep = int(np.ceil(len(ds_tr) / HP["batch"]))

args = TrainingArguments(
    output_dir=f"{SAIDA}/checkpoints", seed=SEED, data_seed=SEED,
    learning_rate=HP["lr"], per_device_train_batch_size=HP["batch"], per_device_eval_batch_size=32,
    num_train_epochs=HP["epocas_max"], weight_decay=HP["weight_decay"],
    warmup_steps=int(HP["warmup_frac"] * passos_ep * HP["epocas_max"]), lr_scheduler_type="cosine",
    eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
    save_total_limit=2, fp16=True, report_to="none")

trainer = Trainer(model=modelo, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                  processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=metricas, callbacks=[EarlyStoppingCallback(HP["paciencia"])])
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer.train()
tempo_treino = time.time() - t0
pico_treino_gb = torch.cuda.max_memory_allocated() / 2**30
historico = [h for h in trainer.state.log_history if "eval_f1_macro" in h]
for h in historico: print(f"época {h['epoch']:.0f}: F1 macro val={h['eval_f1_macro']:.4f}  loss val={h['eval_loss']:.4f}")
print("melhor checkpoint:", trainer.state.best_model_checkpoint, "| treino (s):", round(tempo_treino),
      "| pico de memória GPU no treino (GB):", round(pico_treino_gb, 2))"""),
(MD, "## Avaliação na validação e no teste de referência"),
(CODE, """def avaliar(ds, df, nome):
    pred = trainer.predict(ds)
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    prob = torch.softmax(torch.tensor(logits, dtype=torch.float32), -1).numpy()
    y, yhat = pred.label_ids, prob.argmax(-1)
    rel = classification_report(y, yhat, labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0)
    met = {f"{nome}_f1_macro": rel["macro avg"]["f1-score"], f"{nome}_acuracia": rel["accuracy"]}
    met |= {f"{nome}_f1_{c}": rel[c]["f1-score"] for c in CLASSES}
    cm = confusion_matrix(y, yhat, labels=range(len(CLASSES)))
    json.dump({"classes": CLASSES, "matriz": cm.tolist()}, open(f"{SAIDA}/{nome}_matriz_confusao.json", "w"), indent=2)
    json.dump(rel, open(f"{SAIDA}/{nome}_relatorio.json", "w"), indent=2, ensure_ascii=False)
    out = df.copy(); out["pred"] = [CLASSES[i] for i in yhat]; out["confianca"] = prob.max(-1)
    out.to_csv(f"{SAIDA}/{nome}_predicoes.csv", index=False)
    print(nome, {k: round(v, 4) for k, v in met.items()})
    return met

met = avaliar(ds_va, valid, "validacao") | avaliar(ds_te, teste, "gold")"""),
(MD, """## Latência em GPU e tamanho
Latência por exemplo com lote de 1, modelo em 4 bits, média sobre os textos do teste após aquecimento.
A latência em CPU (modelo mesclado, sem quantização) e a comparação no mesmo hardware ficam para a Fase 4."""),
(CODE, """def latencia_ms(mdl, textos, n_aquec=10):
    mdl.eval()
    with torch.inference_mode():
        for t in textos[:n_aquec]:
            mdl(**tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cuda"))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for t in textos:
            mdl(**tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cuda"))
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / len(textos)

melhor = trainer.model
met["gold_latencia_gpu_ms"] = latencia_ms(melhor, list(teste.texto))
met["pico_memoria_gpu_treino_gb"] = pico_treino_gb
met["memoria_modelo_4bit_gb"] = melhor.get_memory_footprint() / 2**30

dir_adapt = f"{SAIDA}/adaptador"
melhor.save_pretrained(dir_adapt); tok.save_pretrained(dir_adapt)
shutil.rmtree(f"{SAIDA}/checkpoints", ignore_errors=True)
met["tamanho_adaptador_mb"] = sum(os.path.getsize(f) for f in glob.glob(f"{dir_adapt}/*")) / 2**20
print({k: round(v, 2) for k, v in met.items() if any(s in k for s in ["latencia", "memoria", "tamanho"])})"""),
(MD, "## Registro para o MLflow e pacote de saída"),
(CODE, """registro = {
    "run_name": "qlora_tucano2_1.5b" + ("_smoke" if SMOKE else ""),
    "params": {"modelo_base": MODELO_BASE, "quantizacao": "4bit nf4 double-quant, compute fp16",
               "lora_r": LORA["r"], "lora_alpha": LORA["lora_alpha"], "lora_dropout": LORA["lora_dropout"],
               "lora_alvos": ",".join(LORA["target_modules"]), "params_treinaveis": treinaveis, "params_total": total,
               "max_len": MAX_LEN, "seed": SEED, **HP,
               "epocas_treinadas": historico[-1]["epoch"] if historico else 0,
               "melhor_epoca": max(historico, key=lambda h: h["eval_f1_macro"])["epoch"] if historico else 0,
               "n_treino": len(ds_tr), "n_validacao": len(ds_va), "n_teste": len(ds_te),
               "rotulos_treino": "silver gpt-6.1-terra", "rotulos_teste": "referencia gpt-6.1-sol (sem humano)",
               "gpu": GPU_NOME, "plataforma": platform.platform(), "torch": torch.__version__,
               "transformers": transformers.__version__, "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__},
    "metrics": {**met, "tempo_treino_s": tempo_treino},
    "historico_validacao": historico,
}
json.dump(registro, open(f"{SAIDA}/registro_mlflow.json", "w"), indent=2, ensure_ascii=False)
shutil.make_archive(SAIDA, "zip", SAIDA)
print("pronto:", SAIDA + ".zip")"""),
]


def main() -> None:
    aqui = Path(__file__).parent
    if "--smoke" in sys.argv:
        destino = Path(sys.argv[sys.argv.index("--smoke") + 1])
        codigo = [c for t, c in CELULAS if t == CODE and not c.startswith("%pip")]
        destino.write_text("\n\n".join(codigo), encoding="utf-8")
        print(destino)
        return
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if t == MD else nbf.v4.new_code_cell(c) for t, c in CELULAS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"
    nbf.write(nb, aqui / "fase3_qlora_tucano2.ipynb")
    print(aqui / "fase3_qlora_tucano2.ipynb")


if __name__ == "__main__":
    main()
