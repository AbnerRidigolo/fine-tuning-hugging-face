"""Fase 0 - perfil dos dados. Gera reports/fase0_perfil.json e imprime um resumo."""
import json

import pandas as pd

from dados import RAIZ, carregar_comentarios, carregar_reviews


def main() -> None:
    bruto = carregar_reviews()
    df = carregar_comentarios()

    n_chars = df["texto"].str.len()
    n_palavras = df["texto"].str.split().str.len()
    qs = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    dup_texto = df["texto"].duplicated(keep="first")
    dup_texto_ci = df["texto"].str.lower().duplicated(keep="first")
    top_dups = df.loc[df["texto"].duplicated(keep=False), "texto"].value_counts().head(15)

    perfil = {
        "linhas_csv": int(len(bruto)),
        "review_id_unicos": int(bruto["review_id"].nunique()),
        "review_id_repetidos_linhas": int(bruto["review_id"].duplicated().sum()),
        "comentarios_nao_vazios": int(len(df)),
        "pct_com_comentario": round(100 * len(df) / len(bruto), 2),
        "com_titulo": int(df["review_comment_title"].notna().sum()),
        "chars": {
            "media": round(float(n_chars.mean()), 1),
            **{f"p{int(q*100)}": float(n_chars.quantile(q)) for q in qs},
            "max": int(n_chars.max()),
        },
        "palavras": {
            "media": round(float(n_palavras.mean()), 1),
            **{f"p{int(q*100)}": float(n_palavras.quantile(q)) for q in qs},
            "max": int(n_palavras.max()),
        },
        "comentarios_ate_3_chars": int((n_chars <= 3).sum()),
        "notas_todos_reviews": bruto["review_score"].value_counts().sort_index().to_dict(),
        "notas_com_comentario": df["review_score"].value_counts().sort_index().to_dict(),
        "pct_notas_com_comentario": (
            df["review_score"].value_counts(normalize=True).sort_index().mul(100).round(2).to_dict()
        ),
        "textos_duplicados_exatos": int(dup_texto.sum()),
        "textos_duplicados_ignorando_caixa": int(dup_texto_ci.sum()),
        "textos_unicos": int(df["texto"].nunique()),
        "top_textos_repetidos": {k: int(v) for k, v in top_dups.items()},
    }

    saida = RAIZ / "reports" / "fase0_perfil.json"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text(json.dumps(perfil, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(perfil, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
