# Information Density: Comparing Context-Compression Families Under a Reader-Independent Protocol

We investigate how much answer-relevant information survives when a long QA context is compressed by methods spanning truncation, random sampling, extractive (LexRank/TextRank), abstractive (distilBART), iterative abstractive, and structured-memory schemas at fixed token-budget ratios {0.

## Paper

- **PDF (ACL two-column)**: [tex/main.pdf](tex/main.pdf)
- **Source**: [tex/main.tex](tex/main.tex), compile with `tectonic -X compile main.tex`
- **Peer review**: [review.md](review.md) (sealed-PDF reviewer output)

## Primary result

**info_density_mean_across_methods: 1.2828 ± 0.3139** across 3 seeds. Single-pass distilBART-CNN attains information density `ID = 4.24` at compression budget `r = 0.10` on LongBench `multifieldqa_en`, roughly an order of magnitude above every non-abstractive family at the two looser budgets.

## How to reproduce

```bash
modal run experiments/01-context-compression-baseline/experiment.py
```

The script evaluates eight compressors (`truncate_head`, `truncate_tail`, `random_sentences`, `lexrank`, `textrank`, `structured_memory`, `abstractive_bart`, `iterative_abstractive`) on three target budgets (`r ∈ {0.10, 0.25, 0.50}`) over three seeds of 25 LongBench examples each. Modal launches one A10G worker per seed; total wall-clock is under ten minutes. The script can also be run locally (`python experiments/01-.../experiment.py`) with roughly equivalent wall-clock on a mid-range CPU.

## Figures

| File | Caption (first sentence) |
| --- | --- |
| `fig-taxonomy.png` | The eight compressors studied, grouped by family: truncation (head, tail), random (sentence-level), extractive graph (LexRank, TextRank), training-free structured (structured memory), and neural abstractive (single-pass BART, iterative BART). |
| `fig-pipeline.png` | Each example in LongBench multifieldqa_en is passed through a compressor at a fixed target budget r, yielding a compressed context. |
| `fig-main-info-density.png` | Information density (ID = answer recall / realized compression ratio) at target budget r = 0. |
| `fig-budget-sweep.png` | Information density as a function of target budget r. |
| `fig-cr-vs-target.png` | Realized compression ratio CR for each method at each target budget r. |
| `fig-iterative-vs-single.png` | Information density of single-pass abstractive_bart versus iterative_abstractive at each of the three target budgets. |

## Recommended venues

- **ACL** (Annual Meeting of the Association for Computational Linguistics) — Strong.
- **EMNLP** (Conference on Empirical Methods in Natural Language Processing) — Strong.
- **COLM** (Conference on Language Modeling) — Good.
- **TMLR** (Transactions on Machine Learning Research) — Strong.
- **TACL** (Transactions of the Association for Computational Linguistics) — Good.

## Authors

Vikash Chandra Mishra

## License

MIT, see [LICENSE](LICENSE).

## Provenance

Session id: `20260423-120645-b8e2`. See [log.md](log.md) and [state.json](state.json) for the full per-stage audit trail.
