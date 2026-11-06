"""Fase 1 - rotulador LLM (OpenAI) com o guia de rotulagem como prompt e saída validada por Pydantic.

Uso (PowerShell, com OPENAI_API_KEY no .env):
    python src\\rotulador.py --entrada data\\processed\\silver_amostra.csv --saida data\\silver\\rotulos.jsonl --modelo gpt-5.4-mini

- Retoma de onde parou: ids já presentes no .jsonl de saída são pulados.
- Cada linha de saída guarda rótulo, confiança, tokens usados e o modelo, para auditar custo depois.
- Saídas que não validam no Pydantic são registradas com erro e podem ser re-tentadas.
"""
import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

RAIZ = Path(__file__).resolve().parents[1]
GUIA = RAIZ / "docs" / "guia_rotulagem.md"

Classe = Literal[
    "atraso_ou_nao_entrega",
    "produto_errado_ou_incompleto",
    "defeito_ou_avaria",
    "qualidade_abaixo_do_esperado",
    "atendimento_vendedor",
    "elogio",
    "outro",
]


class Rotulo(BaseModel):
    rotulo: Classe
    confianca: Literal["alta", "media", "baixa"]


def montar_instrucoes() -> str:
    guia = GUIA.read_text(encoding="utf-8")
    return (
        "Você é um anotador de dados. Siga estritamente o guia abaixo para rotular o comentário "
        "de avaliação enviado pelo usuário. Responda apenas no formato JSON pedido, com o rótulo "
        "do motivo principal e sua confiança (alta, media ou baixa; use baixa nos casos em que o "
        "guia manda marcar confiança 1).\n\n" + guia
    )


async def rotular_um(cliente, sem, modelo, instrucoes, esforco, rid, texto):
    async with sem:
        for tentativa in range(5):
            try:
                kwargs = {"reasoning": {"effort": esforco}} if esforco else {}
                resp = await cliente.responses.parse(
                    model=modelo,
                    instructions=instrucoes,
                    input=f"Comentário:\n{texto}",
                    text_format=Rotulo,
                    **kwargs,
                )
                u = resp.usage
                base = {
                    "id": rid,
                    "modelo": resp.model,
                    "tokens_entrada": u.input_tokens,
                    "tokens_cache": getattr(u.input_tokens_details, "cached_tokens", 0),
                    "tokens_saida": u.output_tokens,
                    "tokens_raciocinio": getattr(u.output_tokens_details, "reasoning_tokens", 0),
                }
                try:
                    r = Rotulo.model_validate_json(resp.output_text)
                    return {**base, "rotulo": r.rotulo, "confianca": r.confianca, "erro": None}
                except ValidationError as e:
                    return {**base, "rotulo": None, "confianca": None, "erro": str(e)[:300]}
            except Exception as e:  # rede / rate limit: backoff exponencial
                if tentativa == 4:
                    return {"id": rid, "rotulo": None, "confianca": None, "erro": repr(e)[:300]}
                await asyncio.sleep(2 ** tentativa)


async def main_async(args) -> None:
    from openai import AsyncOpenAI

    load_dotenv(RAIZ / ".env")
    cliente = AsyncOpenAI()
    instrucoes = montar_instrucoes()

    df = pd.read_csv(args.entrada, dtype={args.col_id: str})
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    feitos = set()
    if saida.exists():
        for linha in saida.read_text(encoding="utf-8").splitlines():
            reg = json.loads(linha)
            if reg.get("erro") is None:
                feitos.add(reg["id"])
    pend = df[~df[args.col_id].isin(feitos)]
    if args.limite:
        pend = pend.head(args.limite)
    print(f"{len(feitos)} já rotulados, {len(pend)} pendentes")

    sem = asyncio.Semaphore(args.concorrencia)
    tarefas = [
        rotular_um(cliente, sem, args.modelo, instrucoes, args.esforco, r[args.col_id], r[args.col_texto])
        for _, r in pend.iterrows()
    ]
    with saida.open("a", encoding="utf-8") as f:
        for i, fut in enumerate(asyncio.as_completed(tarefas), 1):
            reg = await fut
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
            f.flush()
            if i % 200 == 0:
                print(f"  {i}/{len(tarefas)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--modelo", required=True)
    p.add_argument("--esforco", default=None, help="reasoning effort (ex.: none, minimal, low)")
    p.add_argument("--col-id", default="review_id")
    p.add_argument("--col-texto", default="texto")
    p.add_argument("--concorrencia", type=int, default=8)
    p.add_argument("--limite", type=int, default=0, help="rotula só os N primeiros pendentes (piloto)")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
