# Peer Review: Information Density: Comparing Context-Compression Families Under a Reader-Independent Protocol

**Reviewer:** balanced / intermediate
**Recommendation:** accept
**Confidence:** 3
**Score:** 7/10

## Summary of contributions

The paper benchmarks eight context-compression methods across five families (truncation, random sampling, extractive graph, structured memory, neural abstractive) on the LongBench `multifieldqa_en` sub-task, scored on four reader-independent signals (token-F1 answer recall, exact match, MiniLM embedding similarity, realised compression ratio) and a derived information-density metric `ID = AR / CR`. Two results carry the paper: single-pass distilBART-CNN attains `ID = 4.24` at budget `r = 0.10` and stays in the `3.75–4.24` band across budgets, outperforming every non-abstractive family by more than an order of magnitude at the looser budgets; and iterative two-pass compaction reliably loses 7%–11% of single-pass information density, consistent with compounded per-pass loss. A supporting observation is that a regex-based structured-memory schema self-limits at the document's entity capacity (`CR ≈ 0.045` regardless of budget), which the authors argue is a desirable property for hard-capped memory systems.

## Strengths

1. **Clean taxonomic design.** Eight compressors across five families, with explicit sentence-, fact-, and sub-sentence-level baselines (§3.2, Figure 1). The within-experiment baseline suite is honest and the paper disclaims numerical comparison with prior LLM-reader-based studies (§4.4) rather than forcing apples-to-oranges comparisons.
2. **Reader-independent evaluation is well motivated and now self-critical.** The argument in §1 that downstream QA accuracy under a frozen large reader conflates compressor quality with reader robustness is plausible, and the AR/EM/ES/CR/ID instrument cleanly decouples the two. §3.3 now explicitly flags that `ID` is gameable by budget overshoot and points the reader at the `random_sentences` entry as the illustration, which is exactly the right level of candour.
3. **Honest negative result on iterative abstractive compaction.** §5.2 documents 7%–11% loss versus single-pass, with a consistent widening-then-narrowing `ΔID` across budgets. This result is not over-interpreted.
4. **Novel self-limiting structured-memory framing.** §5.3 presents `CR ≈ 0.045` and flat `ID ≈ 1.45` as a feature for hard-capped systems rather than a bug, which is a fresh way to look at a training-free baseline.
5. **Limitations paragraph in §6 names the right follow-on experiments.** The four caveats (no downstream-accuracy pairing; `multifieldqa_en` only; single abstractive backbone; token-F1 favours verbose compressors) match the weaknesses a reviewer would flag, and the two named follow-ups (downstream reader pairing; learned structured-memory extractor) are specific enough to be actionable.

## Weaknesses

1. **Single dataset, single sub-task** (severity: high, acknowledged). The headline claims ("single-pass abstractive is the only family that improves information density by an order of magnitude over the naive baselines at the tightest budget") are stated for context compression generally but evidence comes from `multifieldqa_en` only (25 examples per seed × 3 seeds = 75 unique example positions). The related-work section names NarrativeQA, MuSiQue, QASPER, GovReport as the natural sister tasks. The paper acknowledges this in §6, but a reviewer at a top venue will still mark it down. A supplementary table over 2–3 LongBench sub-tasks in the next revision would promote this from "weak accept" to "clear accept".
2. **No empirical link from `ID` to downstream task accuracy** (severity: high, acknowledged). This is the headline limitation the authors themselves call out. The metric is presented as if it predicts task utility but no experiment in the paper establishes that link. The promised §6 follow-on ("feeding each compressed context into a frozen 7B-parameter reader") is the experiment the paper needs to make its case stick; it is worth prioritising for the camera-ready.
3. **Single abstractive backbone** (severity: medium, acknowledged). All abstractive numbers come from `distilBART-CNN-12-6`. A 2024-vintage instruction-tuned summariser (Flan-T5-base, Phi-3-mini, Qwen2.5-1.5B-Instruct) could shift the magnitudes and possibly the iterative-vs-single ordering. A 2-row Table 1 addition would close this concern.
4. **`ID` gameability is now acknowledged but not resolved in a metric** (severity: medium). §3.3's new caveat that `ID` can be inflated by budget overshoot is the right disclosure, but the paper still reports `ID` as the cell-level summary throughout Table 1 and §5. An AR-at-matched-realised-CR column (say, AR interpolated at `CR = 0.025`) would give a comparison that is not gameable; this would be a one-afternoon addition for the camera-ready.
5. **Figure 6 y-axis legend overlap** (severity: low). The legend sits near the `0.01` tick on the log y-axis. Moving it outside the plot area (or shrinking its font) would help.
6. **One bibliography entry still shows an ASCII fallback** (severity: low). Non-ASCII author names (e.g. Ko\v{c}isk\'y, O\u{g}uz) now render via LaTeX accent macros rather than placeholder boxes, which is a clear improvement over iteration 1. A couple of accent marks may still look slightly off depending on font rendering; switching to `lualatex` + `fontspec` would give the fully native rendering.

## Specific comments

- §1 contribution bullet 3: the qualified phrasing "more than `1.5×` the next non-abstractive method even at `r = 0.10` where a random-sampling budget-overshoot artefact narrows the gap" is honest but heavy. Consider leading with the artefact sentence in §5.4 first and then stating the contribution more concisely.
- §2 paragraph "Long-context architectures and benchmarks": six long-context benchmarks and three long-input QA benchmarks are listed but only `multifieldqa_en` is evaluated. A single line stating that these are the planned next-step evaluation targets would align expectations.
- §3.2 "Extractive graph summarisation": the `n ≈ max(1, ⌊r·|x|/18⌋)` heuristic assumes an 18-token average sentence length. Worth citing or empirically grounding that 18.
- §3.3 new caveat on `ID` gameability: the reference to §5.1 (`\ref{sec:results-main}`) is the right forward pointer; consider also adding a one-line note in Table 1's caption that says the effect is specifically visible in the `random_sentences / r=0.10` cell.
- Table 1 page 9: consider a `Δr = 0.50 − 0.10` column for each method to make the budget-sensitivity story of §5 pop off the page.
- §6 reproducibility commitment: "We release the full evaluation code…" — the URL/repo name is missing and will be checked at camera-ready; fill it in before submission.

## Recommendation justification

The paper has a clean problem statement, a sensible eight-method comparison grid, an honest negative result on iterative compaction, and a useful self-limiting-structured-memory observation. The core empirical claim (single-pass abstractive dominates information density at the tightest budget, with >10× gap at looser budgets) is well-supported by the within-experiment evidence. Iteration-2 rewrites have fixed the previous iteration's blocking issues: three bibliography entries with wrong author lists are now hand-corrected, the stranded Claude citation has been resolved, and a methodological caveat on `ID` gameability has been added where it belongs in §3.3. The remaining weaknesses (single dataset, single backbone, no downstream-accuracy pairing) are all acknowledged openly and are all addressable in a camera-ready or a follow-up paper. I lean toward accept on the strength of the core finding and the paper's honesty; the outstanding items are extensions, not blockers.

## Minor issues

- The author block lists "Independent" as affiliation; the email domain references a different name. If the affiliation is intentional, leave it; otherwise align the two for a cleaner first page.
- §4.3 "Compute": the 3–6 minute A10G wall-clock is a useful number. Worth a one-line note that the pipeline is CPU-runnable in roughly the same wall-clock so groups without GPUs can reproduce.
- The title wraps to two lines naturally and is balanced; no change needed.
