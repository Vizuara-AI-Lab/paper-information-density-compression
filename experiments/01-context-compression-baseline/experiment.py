"""
Context Compression Without Critical Information Loss — baseline experiment.

Question: Across compression families (truncation, random, extractive, importance-
weighted, abstractive, iterative-abstractive, structured memory), how much of the
information needed to answer a question survives at compression ratios r in
{0.1, 0.25, 0.5}? Which family achieves the best information-density (answer
recall per surviving token)?

Dataset: LongBench `multifieldqa_en` (English multi-field QA subset).
Sample: 25 examples per seed (deterministic shuffle).
Seeds: [0, 1, 2].

Per (method x budget x example) we compute:
  - answer_recall      : token-level F1 between compressed context and gold answer string
  - answer_em          : 1.0 if gold answer occurs verbatim (case-insensitive) in compressed text
  - embed_sim          : cosine similarity between MiniLM embedding of compressed text and gold answer
  - compression_ratio  : len(compressed_tokens) / len(original_tokens)
  - info_density       : answer_recall / max(compression_ratio, eps)

Modal-shaped: one `run_seed` function per seed, fanned out via `.map()`.
Local fallback at `__main__` lets the script run on CPU without Modal.

Outputs (per seed) go to /outputs (Modal Volume) or ./experiment_logs (local):
  - metrics.jsonl        : one JSON line per (seed, method, budget, example)
  - per_seed_summary.json: aggregated mean/std per (method, budget)

The local entrypoint then aggregates across seeds and writes summary.json.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import re
import time
from collections import Counter

import modal

APP_NAME = "paper-experiment-b8e2"
VOL_NAME = "paper-outputs-20260423-120645-b8e2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.44.2",
        "sentence-transformers==3.0.1",
        "datasets==2.20.0",
        "sumy==0.11.0",
        "nltk==3.9.1",
        "numpy",
        "tqdm",
    )
    .run_commands(
        "python -c \"import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)\""
    )
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


# ----------------------------- core experiment ----------------------------- #

DATASET_HF = "zai-org/LongBench"
DATASET_SUBSET = "multifieldqa_en"
N_EXAMPLES_PER_SEED = 25
BUDGETS = [0.10, 0.25, 0.50]
METHODS = [
    "truncate_head",
    "truncate_tail",
    "random_sentences",
    "lexrank",
    "textrank",
    "abstractive_bart",
    "iterative_abstractive",
    "structured_memory",
]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUMM_MODEL = "sshleifer/distilbart-cnn-12-6"
EPS = 1e-9


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(pred_text: str, gold_text: str) -> float:
    pred = _tokenize(pred_text)
    gold = _tokenize(gold_text)
    if not pred or not gold:
        return 0.0
    pred_c, gold_c = Counter(pred), Counter(gold)
    overlap = sum((pred_c & gold_c).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(pred)
    r = overlap / len(gold)
    return 2 * p * r / (p + r)


def exact_match(pred_text: str, gold_text: str) -> float:
    return 1.0 if _normalize(gold_text) in _normalize(pred_text) else 0.0


def split_sentences(text: str) -> list[str]:
    # cheap sentence splitter; nltk.sent_tokenize would be better but adds load
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def truncate_by_tokens(text: str, target_tokens: int, side: str) -> str:
    toks = text.split()
    if target_tokens >= len(toks):
        return text
    if side == "head":
        return " ".join(toks[:target_tokens])
    return " ".join(toks[-target_tokens:])


def random_sentences(text: str, target_tokens: int, rng: random.Random) -> str:
    sents = split_sentences(text)
    if not sents:
        return text
    rng.shuffle(sents)
    out, used = [], 0
    for s in sents:
        n = len(s.split())
        if used + n > target_tokens and out:
            break
        out.append(s)
        used += n
    return " ".join(out)


def lexrank_summary(text: str, target_tokens: int) -> str:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summ = LexRankSummarizer()
    sents = list(parser.document.sentences)
    n = max(1, min(len(sents), max(1, target_tokens // 18)))
    chosen = summ(parser.document, n)
    return " ".join(str(s) for s in chosen)


def textrank_summary(text: str, target_tokens: int) -> str:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summ = TextRankSummarizer()
    sents = list(parser.document.sentences)
    n = max(1, min(len(sents), max(1, target_tokens // 18)))
    chosen = summ(parser.document, n)
    return " ".join(str(s) for s in chosen)


def abstractive_bart(text: str, target_tokens: int, summarizer) -> str:
    text = text[:8000]  # bart input cap
    max_len = max(20, min(target_tokens, 400))
    min_len = max(10, max_len // 3)
    out = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        truncation=True,
        do_sample=False,
        num_beams=2,
    )
    return out[0]["summary_text"]


def iterative_abstractive(text: str, target_tokens: int, summarizer) -> str:
    intermediate = abstractive_bart(text, max(target_tokens * 2, 200), summarizer)
    return abstractive_bart(intermediate, target_tokens, summarizer)


def structured_memory(text: str, target_tokens: int) -> str:
    # Cheap extractive structured schema: pull noun-phrase-like spans, dates, numerics,
    # and quoted strings; format as a compact key-value list.
    facts = []
    facts.extend(re.findall(r"\b\d{4}\b", text)[:30])  # years
    facts.extend(re.findall(r"\$?\d[\d,]*(?:\.\d+)?%?", text)[:60])  # numerics
    facts.extend(re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text)[:120])  # proper nouns
    facts.extend(re.findall(r"\"([^\"]{3,80})\"", text)[:30])
    seen, uniq = set(), []
    for f in facts:
        f = f.strip()
        if f and f.lower() not in seen:
            seen.add(f.lower())
            uniq.append(f)
    blob = "; ".join(uniq)
    toks = blob.split()
    if len(toks) > target_tokens:
        toks = toks[:target_tokens]
    return " ".join(toks)


def apply_method(method: str, text: str, target_tokens: int, rng, summarizer) -> str:
    if method == "truncate_head":
        return truncate_by_tokens(text, target_tokens, "head")
    if method == "truncate_tail":
        return truncate_by_tokens(text, target_tokens, "tail")
    if method == "random_sentences":
        return random_sentences(text, target_tokens, rng)
    if method == "lexrank":
        return lexrank_summary(text, target_tokens)
    if method == "textrank":
        return textrank_summary(text, target_tokens)
    if method == "abstractive_bart":
        return abstractive_bart(text, target_tokens, summarizer)
    if method == "iterative_abstractive":
        return iterative_abstractive(text, target_tokens, summarizer)
    if method == "structured_memory":
        return structured_memory(text, target_tokens)
    raise ValueError(method)


def _load_models(stub: bool):
    if stub:
        return None, None
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    import torch as _torch
    dev = 0 if _torch.cuda.is_available() else -1
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if dev == 0 else "cpu")
    summarizer = pipeline("summarization", model=SUMM_MODEL, device=dev)
    return embedder, summarizer


def _load_examples(seed: int, n: int):
    from datasets import load_dataset

    ds = load_dataset(DATASET_HF, DATASET_SUBSET, split="test", trust_remote_code=True)
    idx = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    chosen = idx[:n]
    examples = []
    for i in chosen:
        row = ds[i]
        gold_answers = row.get("answers") or [row.get("answer", "")]
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
        gold = max(gold_answers, key=len) if gold_answers else ""
        examples.append(
            {
                "id": row.get("_id") or row.get("id") or str(i),
                "context": row["context"],
                "question": row["input"],
                "gold": gold,
            }
        )
    return examples


def _run_seed_body(seed: int) -> dict:
    stub = bool(os.environ.get("EXPERIMENT_STUB"))
    out_dir = pathlib.Path("/outputs") if pathlib.Path("/outputs").exists() else pathlib.Path("experiment_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"metrics_seed{seed}.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    embedder, summarizer = _load_models(stub)
    examples = _load_examples(seed, N_EXAMPLES_PER_SEED) if not stub else [
        {"id": f"stub{i}", "context": "x " * 4000, "question": "q", "gold": "answer"}
        for i in range(5)
    ]

    rng = random.Random(seed)
    t0 = time.time()
    rows = []

    for ex in examples:
        ctx_tokens = ex["context"].split()
        n_orig = len(ctx_tokens)
        gold_emb = (
            embedder.encode([ex["gold"] or "_"], normalize_embeddings=True)[0]
            if embedder
            else None
        )
        for budget in BUDGETS:
            target = max(20, int(round(n_orig * budget)))
            for method in METHODS:
                try:
                    compressed = apply_method(method, ex["context"], target, rng, summarizer)
                except Exception as e:
                    compressed = ""
                    err = repr(e)[:200]
                else:
                    err = None
                comp_tokens = compressed.split()
                ratio = len(comp_tokens) / max(n_orig, 1)
                ar = token_f1(compressed, ex["gold"])
                em = exact_match(compressed, ex["gold"])
                if embedder:
                    comp_emb = embedder.encode([compressed or "_"], normalize_embeddings=True)[0]
                    sim = float((gold_emb * comp_emb).sum())
                else:
                    sim = 0.0
                density = ar / max(ratio, EPS)
                row = {
                    "seed": seed,
                    "ex_id": ex["id"],
                    "method": method,
                    "budget": budget,
                    "compression_ratio": ratio,
                    "answer_recall": ar,
                    "answer_em": em,
                    "embed_sim": sim,
                    "info_density": density,
                    "n_orig_tokens": n_orig,
                    "n_comp_tokens": len(comp_tokens),
                    "error": err,
                }
                rows.append(row)
                log_f.write(json.dumps(row) + "\n")
    log_f.close()

    # aggregate per (method, budget) for this seed
    agg = {}
    for r in rows:
        key = (r["method"], r["budget"])
        bucket = agg.setdefault(
            key,
            {
                "method": r["method"],
                "budget": r["budget"],
                "answer_recall": [],
                "answer_em": [],
                "embed_sim": [],
                "compression_ratio": [],
                "info_density": [],
            },
        )
        for k in ("answer_recall", "answer_em", "embed_sim", "compression_ratio", "info_density"):
            bucket[k].append(r[k])

    summary = []
    for (m, b), bucket in agg.items():
        s = {"method": m, "budget": b, "n": len(bucket["answer_recall"])}
        for k in ("answer_recall", "answer_em", "embed_sim", "compression_ratio", "info_density"):
            xs = bucket[k]
            mean = sum(xs) / len(xs)
            var = sum((x - mean) ** 2 for x in xs) / len(xs)
            s[f"{k}_mean"] = mean
            s[f"{k}_std"] = var ** 0.5
        summary.append(s)

    overall_density = sum(s["info_density_mean"] for s in summary) / len(summary)
    out = {
        "seed": seed,
        "primary_metric_name": "info_density_mean_across_methods",
        "primary_metric": overall_density,
        "training_time_s": time.time() - t0,
        "peak_gpu_memory_gb": 0.0,
        "per_method_budget": summary,
        "n_examples": len(examples),
        "n_methods": len(METHODS),
        "n_budgets": len(BUDGETS),
    }
    summary_path = out_dir / f"per_seed_summary_{seed}.json"
    summary_path.write_text(json.dumps(out, indent=2))
    print(f"RESULT: metric={out['primary_metric']:.4f} seed={seed}")
    return out


# ----------------------------- Modal entrypoints ----------------------------- #

@app.function(
    gpu="A10G",
    cpu=4.0,
    memory=8192,
    timeout=int(os.environ.get("COMPUTE_TIMEOUT_S", "3600")),
    volumes={"/outputs": vol},
)
def run_seed(seed: int) -> dict:
    out = _run_seed_body(seed)
    vol.commit()
    return out


@app.local_entrypoint()
def main(seeds: str = "0,1,2"):
    seed_list = [int(s) for s in seeds.split(",")]
    results = list(run_seed.map(seed_list))

    primary = [r["primary_metric"] for r in results]
    mean = sum(primary) / len(primary)
    std = (sum((p - mean) ** 2 for p in primary) / len(primary)) ** 0.5

    # cross-seed aggregation per (method, budget)
    bucket = {}
    for r in results:
        for s in r["per_method_budget"]:
            key = (s["method"], s["budget"])
            bucket.setdefault(key, []).append(s)
    cross = []
    for (m, b), entries in bucket.items():
        row = {"method": m, "budget": b, "n_seeds": len(entries)}
        for k in ("answer_recall", "answer_em", "embed_sim", "compression_ratio", "info_density"):
            means = [e[f"{k}_mean"] for e in entries]
            mu = sum(means) / len(means)
            sd = (sum((x - mu) ** 2 for x in means) / len(means)) ** 0.5
            row[f"{k}_mean"] = mu
            row[f"{k}_std"] = sd
        cross.append(row)

    summary = {
        "primary_metric": {
            "name": results[0]["primary_metric_name"],
            "mean": mean,
            "std": std,
            "n_seeds": len(primary),
        },
        "per_seed": results,
        "cross_seed": cross,
    }
    out_local = pathlib.Path("experiment_logs")
    out_local.mkdir(exist_ok=True)
    (out_local / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SUMMARY: {json.dumps(summary)[:1500]}")


# ----------------------------- local fallback ----------------------------- #

if __name__ == "__main__":
    out_dir = pathlib.Path("experiment_logs")
    out_dir.mkdir(exist_ok=True)
    seed_list = [0, 1, 2]
    results = [_run_seed_body(s) for s in seed_list]
    primary = [r["primary_metric"] for r in results]
    mean = sum(primary) / len(primary)
    std = (sum((p - mean) ** 2 for p in primary) / len(primary)) ** 0.5
    bucket = {}
    for r in results:
        for s in r["per_method_budget"]:
            key = (s["method"], s["budget"])
            bucket.setdefault(key, []).append(s)
    cross = []
    for (m, b), entries in bucket.items():
        row = {"method": m, "budget": b, "n_seeds": len(entries)}
        for k in ("answer_recall", "answer_em", "embed_sim", "compression_ratio", "info_density"):
            means = [e[f"{k}_mean"] for e in entries]
            mu = sum(means) / len(means)
            sd = (sum((x - mu) ** 2 for x in means) / len(means)) ** 0.5
            row[f"{k}_mean"] = mu
            row[f"{k}_std"] = sd
        cross.append(row)
    summary = {
        "primary_metric": {
            "name": results[0]["primary_metric_name"],
            "mean": mean,
            "std": std,
            "n_seeds": len(primary),
        },
        "per_seed": results,
        "cross_seed": cross,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"SUMMARY: {json.dumps(summary)[:1500]}")
