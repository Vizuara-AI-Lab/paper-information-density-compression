# Experiment Plan v1

## Question
Across families of context-compression methods, how much answer-relevant
information survives at fixed compression ratios on long-context QA, and which
family achieves the best information-density (answer-relevant content per
surviving token)? We treat the *original passage* as the upper bound and
measure each compressor's ability to retain enough of it that a downstream
QA system could still recover the gold answer.

## Hypothesis
1. **Truncation** (head or tail) sets a low floor: positional bias produces
   high variance in answer recall.
2. **Extractive** methods (LexRank, TextRank) plateau in information-density
   because they pick whole sentences and waste budget on connective tissue.
3. **Abstractive** summarization (BART) achieves higher density at small
   budgets but loses long-tail facts on multi-fact questions.
4. **Iterative abstractive** compaction degrades faster than single-pass
   abstractive — each pass discards information, and errors compound.
5. **Structured memory schema** (entity/numeric extraction) maximizes density
   on entity-centric questions and underperforms on synthesis questions.

## Dataset + preprocessing
- LongBench `multifieldqa_en` (English multi-field QA, ~150 examples,
  contexts averaging ~4k tokens). License: MIT. Source:
  `huggingface.co/datasets/zai-org/LongBench`.
- Per seed, deterministically shuffle by seed and take the first 25 examples.
- Per example we use the raw `context`, `input` (question), and `answers`
  fields. The longest gold answer is treated as the reference string.

## Compression methods (8)
| Family                  | Method                  | Mechanism                                                                       |
| ----------------------- | ----------------------- | ------------------------------------------------------------------------------- |
| Truncation              | `truncate_head`         | First N tokens                                                                  |
| Truncation              | `truncate_tail`         | Last N tokens                                                                   |
| Random                  | `random_sentences`      | Uniformly sampled sentences until token budget reached                          |
| Extractive              | `lexrank`               | LexRank graph-centrality summary (`sumy`)                                       |
| Extractive              | `textrank`              | TextRank summary (`sumy`)                                                       |
| Abstractive             | `abstractive_bart`      | `sshleifer/distilbart-cnn-12-6` single-pass summary                             |
| Abstractive (iterative) | `iterative_abstractive` | distilbart at 2× budget, then re-summarize to budget                            |
| Structured memory       | `structured_memory`     | Years + numerics + proper nouns + quoted strings, deduped, joined to budget     |

## Compression budgets
Three target ratios `r ∈ {0.10, 0.25, 0.50}` of the original token count.

## Metrics
| Name                | Formula                                                                                |
| ------------------- | -------------------------------------------------------------------------------------- |
| `compression_ratio` | `len(compressed_tokens) / len(original_tokens)`                                        |
| `answer_recall`     | Token-F1 between compressed text and the gold answer string                            |
| `answer_em`         | Exact substring match of gold answer (lower-cased) inside compressed text              |
| `embed_sim`         | Cosine similarity between MiniLM embedding of compressed text and gold answer          |
| `info_density`      | `answer_recall / max(compression_ratio, eps)` — answer-relevant content per used token |

## Evaluation protocol
- Seeds: `[0, 1, 2]`. Each seed picks its own 25 examples and seeds any
  stochastic compressor (only `random_sentences` is stochastic).
- Per seed we compute mean ± std of every metric across the 25 examples.
  Cross-seed we report mean and std of the per-seed means.

## Models (CPU-friendly)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` (~80MB).
- Abstractive summarizer: `sshleifer/distilbart-cnn-12-6` (~300MB), beam=2, no
  sampling.

## Compute budget
- Modal default: `cpu=4.0`, `memory=8192MB`, `timeout=1800s` (no GPU needed —
  the chosen models run comfortably on CPU). Three seeds run in parallel via
  `run_seed.map`.
- Local fallback: `python experiment.py` runs seeds sequentially. Expected
  wall-clock on a recent laptop CPU: 8–15 minutes for 3 seeds × 25 examples ×
  3 budgets × 8 methods.
- Estimated Modal cost for the full run: well under \\, no GPU billed.

## Expected artifacts
| Path                                       | Content                                  |
| ------------------------------------------ | ---------------------------------------- |
| `metrics_seed{N}.jsonl`                    | One JSON line per (method, budget, ex)   |
| `per_seed_summary_{N}.json`                | Per-seed mean/std per (method, budget)   |
| `summary.json`                             | Cross-seed primary metric + table        |

## Stop criteria
- Cross-seed `info_density_mean_across_methods` is computed and finite.
- Per-method, per-budget rows have `n_seeds = 3`.
- No NaNs in `answer_recall`, `embed_sim`, or `compression_ratio`.

## Risks / limitations to note in the paper
- Token-F1 against a short gold answer rewards verbose compressors; we report
  `embed_sim` and `info_density` as counterweights.
- `multifieldqa_en` contexts are ~4k tokens, shorter than the long-context
  regime where compression matters most. We treat this as a controlled stress
  test, not a long-context end-to-end evaluation.
- We measure information *survival in text*, not downstream QA accuracy under
  a frozen LLM. That follow-on is left to a Stage-2 study.
