"""Fase 4 - latência por exemplo (CPU e GPU) e tamanho em disco dos três modelos, no MESMO hardware.

Protocolo (igual para todos):
  - um comentário por chamada (lote de 1), como numa API; a tokenização entra no tempo;
  - os 400 textos do teste de referência, após 10 chamadas de aquecimento; cada chamada é cronometrada
    individualmente (com sincronização da GPU) e reportamos mediana, p95 e média em ms por exemplo;
  - GPU: BERTimbau em fp32; Tucano2 em 4 bits NF4 + adaptador (como foi treinado). Baseline não usa GPU.
  - CPU: BERTimbau em fp32; Tucano2 com o adaptador mesclado num base fp32 (não há 4 bits em CPU).
    As predições do Tucano2 mesclado em CPU são comparadas às de 4 bits, para registrar a fidelidade.

Saída: results/fase4/latencia.json e results/fase4/predicoes_tucano2_local.csv
(a pasta de saída pode ser trocada com a variável FASE4_SAIDA, usada no Kaggle).

    python src\\fase4_latencia.py
"""
import glob
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from dados import RAIZ
from rotulos import CLASSES, carregar_gold

N_AQUEC = 10
SAIDA = Path(os.environ.get("FASE4_SAIDA") or RAIZ / "results" / "fase4")
BASE_TUCANO = "Polygl0t/Tucano2-qwen-1.5B-Instruct"


def nome_cpu() -> str:
    if os.path.exists("/proc/cpuinfo"):
        for linha in open("/proc/cpuinfo"):
            if linha.startswith("model name"):
                return linha.split(":", 1)[1].strip() + f" ({os.cpu_count()} CPUs lógicas)"
    return f"{platform.processor()} ({os.cpu_count()} CPUs lógicas)"


def tamanho_mb(caminhos) -> float:
    return sum(os.path.getsize(f) for c in caminhos for f in glob.glob(str(c)) if os.path.isfile(f)) / 2**20


def cronometrar(fn, textos, cuda=False) -> dict:
    """Cronometra cada chamada separadamente; devolve mediana, p95 e média em ms."""
    for t in textos[:N_AQUEC]:
        fn(t)
    tempos = []
    for t in textos:
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(t)
        if cuda:
            torch.cuda.synchronize()
        tempos.append(1000 * (time.perf_counter() - t0))
    a = np.array(tempos)
    return {"mediana": float(np.median(a)), "p95": float(np.percentile(a, 95)), "media": float(a.mean())}


def registrar(res: dict, disp: str, est: dict) -> None:
    res[f"{disp}_ms"] = est["mediana"]
    res[f"{disp}_ms_p95"] = est["p95"]
    res[f"{disp}_ms_media"] = est["media"]


def medir_baseline(textos) -> dict:
    caminho = RAIZ / "models" / "baseline_tfidf_lr" / "modelo.joblib"
    m = joblib.load(caminho)
    res = {"gpu_ms": None, "disco_mb": tamanho_mb([caminho])}
    registrar(res, "cpu", cronometrar(lambda t: m.predict_proba([t]), textos))
    return res


def medir_bertimbau(textos) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    d = RAIZ / "models" / "bertimbau"
    tok = AutoTokenizer.from_pretrained(d)
    res = {"disco_mb": tamanho_mb([d / "*.safetensors", d / "*.json"])}
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        m = AutoModelForSequenceClassification.from_pretrained(d).to(dev).eval()

        def f(t, m=m, dev=dev):
            with torch.inference_mode():
                return m(**tok(t, return_tensors="pt", truncation=True, max_length=128).to(dev)).logits
        registrar(res, "gpu" if dev == "cuda" else "cpu", cronometrar(f, textos, cuda=dev == "cuda"))
        del m
        torch.cuda.empty_cache()
    return res


def medir_tucano2(textos, gold) -> dict:
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    adapt = RAIZ / "models" / "qlora_tucano2_1.5b"
    base_dir = Path(snapshot_download(BASE_TUCANO, allow_patterns=["*.json", "*.safetensors", "*.jinja"]))
    tok = AutoTokenizer.from_pretrained(adapt)
    kw = dict(num_labels=len(CLASSES), id2label=dict(enumerate(CLASSES)), label2id={c: i for i, c in enumerate(CLASSES)})
    res = {"disco_adaptador_mb": tamanho_mb([adapt / "adapter_model.safetensors", adapt / "adapter_config.json"]),
           "disco_base_mb": tamanho_mb([base_dir / "*.safetensors"])}
    res["disco_mb"] = res["disco_adaptador_mb"] + res["disco_base_mb"]

    def preditor(m, dev):
        def f(t):
            with torch.inference_mode():
                return m(**tok(t, return_tensors="pt", truncation=True, max_length=128).to(dev)).logits
        return f

    preds = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForSequenceClassification.from_pretrained(base_dir, quantization_config=bnb,
                                                                  device_map={"": 0}, dtype=torch.float16, **kw)
        base.config.pad_token_id = tok.pad_token_id
        m = PeftModel.from_pretrained(base, adapt).eval()
        f = preditor(m, "cuda")
        registrar(res, "gpu", cronometrar(f, textos, cuda=True))
        res["gpu_memoria_inferencia_gb"] = torch.cuda.max_memory_allocated() / 2**30
        preds["gpu_4bit"] = [int(f(t).argmax(-1)) for t in textos]
        del m, base
        torch.cuda.empty_cache()

    base = AutoModelForSequenceClassification.from_pretrained(base_dir, dtype=torch.float32, **kw)
    base.config.pad_token_id = tok.pad_token_id
    m = PeftModel.from_pretrained(base, adapt).merge_and_unload().eval()
    f = preditor(m, "cpu")
    registrar(res, "cpu", cronometrar(f, textos))
    preds["cpu_mesclado_fp32"] = [int(f(t).argmax(-1)) for t in textos]

    y = [CLASSES.index(r) for r in gold.rotulo]
    for k, p in preds.items():
        res[f"f1_macro_{k}"] = f1_score(y, p, labels=range(len(CLASSES)), average="macro", zero_division=0)
    if len(preds) == 2:
        res["concordancia_cpu_vs_gpu"] = float(np.mean(np.array(preds["gpu_4bit"]) == np.array(preds["cpu_mesclado_fp32"])))
    out = gold[["id_gold", "rotulo"]].copy()
    for k, p in preds.items():
        out[f"pred_{k}"] = [CLASSES[i] for i in p]
    out.to_csv(SAIDA / "predicoes_tucano2_local.csv", index=False, encoding="utf-8")
    return res


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    gold = carregar_gold()
    textos = list(gold.texto)
    torch.manual_seed(42)
    hw = {"cpu": nome_cpu(), "torch_threads_cpu": torch.get_num_threads(),
          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          "torch": torch.__version__, "n_textos": len(textos), "aquecimento": N_AQUEC,
          "protocolo": "lote de 1, tokenização incluída, cada chamada cronometrada; mediana e p95 em ms",
          "plataforma": platform.platform()}
    res = {"hardware": hw}
    for nome, fn in [("baseline_tfidf_lr", lambda: medir_baseline(textos)),
                     ("bertimbau", lambda: medir_bertimbau(textos)),
                     ("qlora_tucano2_1.5b", lambda: medir_tucano2(textos, gold))]:
        print(f"medindo {nome}...", flush=True)
        res[nome] = fn()
        print(json.dumps(res[nome], indent=2), flush=True)
        (SAIDA / "latencia.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
