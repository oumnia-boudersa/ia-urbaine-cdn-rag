#!/usr/bin/env python3
"""
Run the RAG strategies from the terminal (no Streamlit).

Interactive, one config:
    python run_cli.py --config configs/Llama-2-7b-chat/IvadoCDN/SR-RAG-SPARSE.json

Batch -> CSV for make_examples_tex.py (label=config_path, one or more):
    python run_cli.py \
      --run no_rag=configs/Llama-2-7b-chat/IvadoCDN/wo-RAG.json \
      --run single_sparse=configs/Llama-2-7b-chat/IvadoCDN/SR-RAG-SPARSE.json \
      --run single_dense=configs/Llama-2-7b-chat/IvadoCDN/SR-RAG-DENSE.json \
      --run dragin=configs/Llama-2-7b-chat/IvadoCDN/DRAGIN.json \
      --gold gold_eval_set_phase1.json --out outputs.csv

Use --question "..." (repeatable) instead of --gold for a few ad-hoc questions.
Adjust the paths above to your real config file names.
"""
import argparse, csv, gc, json, logging, sys
from pathlib import Path
from types import SimpleNamespace

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))
logging.basicConfig(level=logging.WARNING)

from data import IvadoCDN                                    # noqa: E402
from generate_ivado import (BasicRAG, SingleRAG, FixLengthRAG, TokenRAG,   # noqa: E402
                            EntityRAG, AttnWeightRAG, Counter)


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("shuffle", False)
    cfg.setdefault("use_counter", True)
    if "complexity_judge_model_path" not in cfg:
        cand = BASE_DIR / "query complexity evaluator" / "results_complexity_evaluator_t5-base" / "best_model"
        if cand.is_dir():
            cfg["complexity_judge_model_path"] = str(cand)
    cfg["config_path"] = str(path)
    return SimpleNamespace(**cfg)


def build_model(a):
    m = a.method
    if m == "non-retrieval": return BasicRAG(a)
    if m == "single-retrieval": return SingleRAG(a)
    if m in {"fix-length-retrieval", "fix-sentence-retrieval"}: return FixLengthRAG(a)
    if m == "token": return TokenRAG(a)
    if m == "entity": return EntityRAG(a)
    if m in {"attn_prob", "dragin"}: return AttnWeightRAG(a)
    raise ValueError(f"Unsupported method: {m}")


def build_demo(fewshot):
    demo = []
    for ex in IvadoCDN.examplars[:fewshot]:
        q = IvadoCDN.demo_input_template(IvadoCDN, ex["question"])
        out = IvadoCDN.output_template(IvadoCDN, ex.get("cot"), ex.get("answer"))
        if out:
            q += "" if q[-1:] in {"\n", " "} else " "
            q += out
        demo.append({"question": ex["question"], "case": q, "ctxs": ex.get("ctxs", [])})
    return demo


def ask(model, args, demo, question, show_ctx):
    model.counter = Counter()
    case = IvadoCDN.test_input_template(IvadoCDN, question)
    answer = model.inference(question, demo, case).strip()
    docs = []
    for rec in getattr(model, "retrieved_docs", []) or []:
        docs.extend(rec.get("docs", []))
    if show_ctx and docs:
        print("\n--- Retrieved passages ---")
        for i, d in enumerate(docs, 1):
            print(f"[{i}] {str(d)[:300].replace(chr(10), ' ')}")
    return answer, docs


def gold_questions(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [it["question"] for lv in d["personas"].values() for items in lv.values() for it in items]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="single config -> interactive mode")
    p.add_argument("--run", action="append", default=[], help="label=config_path (batch mode)")
    p.add_argument("--gold"); p.add_argument("--question", action="append", default=[])
    p.add_argument("--out", default="outputs.csv")
    p.add_argument("--fewshot", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None, help="override generate_max_length")
    p.add_argument("--topk", type=int, default=None, help="override retrieve_topk")
    p.add_argument("--show-passages", action="store_true")
    a = p.parse_args()

    def overrides(args, model):
        for k, v in (("generate_max_length", a.max_length), ("retrieve_topk", a.topk)):
            if v is not None:
                setattr(args, k, v); setattr(model, k, v)

    if a.config:                                   # ---- interactive
        args = load_config(a.config); model = build_model(args); overrides(args, model)
        demo = build_demo(a.fewshot or min(args.fewshot, len(IvadoCDN.examplars)))
        print(f"Loaded {a.config} (method={args.method}). Empty line or Ctrl-D to quit.")
        while True:
            try: q = input("\nQuestion> ").strip()
            except EOFError: break
            if not q: break
            ans, _ = ask(model, args, demo, q, a.show_passages or True)
            print("\n=== Answer ===\n" + ans)
        return

    if not a.run:
        p.error("give --config (interactive) or one/more --run label=path (batch)")
    questions = a.question or (gold_questions(a.gold) if a.gold else [])
    if not questions:
        p.error("give --gold or --question for batch mode")

    rows = []
    from tqdm.auto import tqdm
    for spec in a.run:
        label, path = spec.split("=", 1)
        args = load_config(path); model = build_model(args); overrides(args, model)
        demo = build_demo(a.fewshot or min(args.fewshot, len(IvadoCDN.examplars)))
        for q in tqdm(questions, desc=label, unit="q"):
            ans, docs = ask(model, args, demo, q, False)
            rows.append({"strategy": label, "question": q, "answer": ans,
                         "n_passages": len(docs),
                         "passages": " || ".join(str(d)[:200].replace("\n", " ") for d in docs)})
        del model; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "question", "answer", "n_passages", "passages"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()