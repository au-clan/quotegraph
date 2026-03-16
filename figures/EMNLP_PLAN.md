# EMNLP Plan: LIWC Example Application Section

> This file will be written to `figures/EMNLP_PLAN.md` upon approval.

## Context

Quotegraph is a dataset paper for EMNLP with 3 example applications: (1) first-name vs. last-name gender bias (existing Section 6), (2) LIWC linguistic analysis of directed quotations (this plan), (3) TBD. The LIWC section must be concise (~1.5 pages), statistically rigorous, and focused on findings that **require** Quotegraph's directed speaker→mention structure. The full internal analysis (`liwc_literature_comparison.md`, 391 lines) is far too detailed and has critical issues documented in `PAPEREMNLP.md`.

---

## REJECTION RISK: Items Requiring Special Attention

> These are the items most likely to cause a desk reject or a confident-reject score. Ordered by danger level. If you address nothing else, address these.

### REJECT-1: No statistical tests (F1)
**Risk**: Instant reject at any top venue. Descriptive-only analysis with eyeballed values from plots is not publishable.
**What to do**: Fit mixed-effects regressions (Sec 3a). Report coefficients, 95% CIs, FDR-corrected p-values. This is non-negotiable.
**Trap**: With 8.63M edges, *everything* will be significant. You MUST also report effect sizes (partial η², Cohen's f²). A reviewer will immediately note that p < 0.001 is meaningless at this scale if the effect size is η² = 0.001. Frame results around effect magnitude, not significance.

### REJECT-2: Occupation confound uncontrolled (M1)
**Risk**: A reviewer can dismiss every finding with: "This is just 'politicians use more power/conflict language than athletes' — controlling for occupation would likely eliminate these effects."
**What to do**: Include occupation as a covariate in all regressions. Run within-occupation stratification (politicians-only, sportspeople-only) for headline findings. If the conflict inverted-U or Clout-Power mirror vanish within occupation, demote them honestly. If they survive, that is your strongest evidence.
**Trap**: If you don't run within-occupation analyses, a reviewer will ask, and you'll have no answer during rebuttal.

### REJECT-3: Causal language (F3)
**Risk**: Easy ammunition for any reviewer. "The authors claim X 'elicits' Y without any causal identification strategy."
**What to do**: Find-and-replace audit of the entire section. Zero tolerance for: elicits, drives, produces, suppresses, causes, leads to, makes. Replace with: is associated with, co-occurs with, correlates with, predicts.
**Trap**: This is a low-effort fix but is easy to miss in the heat of writing. Do a literal text search before submission.

### REJECT-4: Cherry-picked LIWC categories (F2)
**Risk**: "The authors report 6 of 80+ LIWC categories without justification. How many did they test?"
**What to do**: State the selection criterion before presenting results: "We selected categories with established speaker-level predictions from Pennebaker & Stone (2003) and Kacewicz et al. (2014)." Then apply FDR correction across ALL tested categories (not just the ones you show). State explicitly: "We tested K categories; N survived FDR correction at q < 0.05."
**Trap**: If you add new categories after seeing results, that's HARKing. Decide the set now and commit.

### REJECT-5: LIWC domain validity (M2)
**Risk**: "LIWC was validated on essays and social media. What evidence do the authors have that it works on news quotations?"
**What to do**: (a) Report quote length distribution and apply minimum length filter (≥ 25 words). (b) State the speaker-level replications (I×age, Clout×age) as indirect validation — known effects replicate, suggesting LIWC captures meaningful signal. (c) Acknowledge the domain gap in limitations.
**Trap**: Don't oversell the replication as "validation." It's necessary but not sufficient.

### REJECT-6: Pseudoreplication (A4) — NEWLY IDENTIFIED
**Risk**: A methods-savvy reviewer will notice that the same speaker appears in thousands of quotes. Treating each quote as independent inflates significance massively.
**What to do**: Use random intercepts for source_id and target_id in mixed-effects models (Sec 3a). At minimum, cluster standard errors by dyad (source, target pair). Report the number of unique speakers, targets, and dyads alongside total N quotes.
**Trap**: If you report p-values based on N = 8M without accounting for clustering, a reviewer can compute that the effective N is orders of magnitude smaller.

### REJECT-7: Static PageRank on dynamic network (A11) — NEWLY IDENTIFIED
**Risk**: "PageRank is computed on the full 2008–2020 aggregated network. But a person's prominence in 2009 differs from 2018. Using static PageRank introduces look-ahead bias — a quote from 2009 is scored against prominence computed partly from 2015–2020 data."
**What to do**: Acknowledge this in limitations. If feasible, compute year-specific PageRank and show robustness. At minimum, state that PageRank captures long-run average prominence, not time-varying prominence.

### REJECT-8: Effect-size-at-scale illusion (A12) — NEWLY IDENTIFIED
**Risk**: Closely related to REJECT-1. All the reported differences (e.g., "Clout 51→60 across age-diff axis") sound large in raw LIWC units but may explain <1% of variance. Without R² or η², a reviewer can't assess practical significance.
**What to do**: For every reported pattern, state partial η² or R². Frame the result appropriately: "The age-gap accounts for X% of variance in Clout scores, which is [small/medium] by Cohen's conventions but noteworthy given the heterogeneity of public discourse." Even small effects can be meaningful in observational data — but you must be transparent.

---

## ToDo Checklist

> Each item maps to an issue from `PAPEREMNLP.md` or a newly identified gap.
> Items are marked [x] when the plan addresses them.

### Fatal issues (from PAPEREMNLP.md)
- [x] **F1: No statistical tests** → Sec 3 + REJECT-1
- [x] **F2: Cherry-picked categories** → Sec 2 + REJECT-4
- [x] **F3: Causal language** → Sec 5 + REJECT-3

### Major issues (from PAPEREMNLP.md)
- [x] **M1: Occupation confound uncontrolled** → Sec 3 + REJECT-2
- [x] **M2: LIWC domain validity** → Sec 4, 6 + REJECT-5
- [x] **M3: Selection bias (who gets quoted)** → Sec 5 (scope all claims)
- [x] **M4: Mention ≠ interaction** → Sec 5 (drop CAT, reframe through CDA)
- [x] **M5: Eyeballed numerical values** → Sec 3 (exact computed values)
- [x] **M6: Heatmap patterns unquantified** → Sec 3b (variance decomposition)

### Moderate issues (from PAPEREMNLP.md)
- [x] **m1: Inconsistent verdict taxonomy** → Sec 5 (4-level scheme)
- [x] **m2: Novel count overclaimed** → Sec 2 (reduced to 3–4 findings)
- [x] **m3: B&L over-invoked** → Sec 5 (cite once)
- [x] **m4: CAT about-vs-with ignored** → Sec 5 (drop CAT)
- [x] **m5: SST contradict/extend inconsistency** → Sec 5 (conditional framing)
- [x] **m6: No null model** → Sec 3d (permutation test)
- [x] **m7: Temporal dimension absent** → Sec 3d (temporal split)

### Minor issues (from PAPEREMNLP.md)
- [x] **p1: Loaded terminology** → Sec 5 (neutral descriptions)
- [x] **p2: Incomplete citations** → Sec 5 (full references required)
- [x] **p3: Inconsistent precision** → Sec 3 (2dp from computed data)
- [x] **p4: Three archetypes untested** → Sec 2 (dropped)
- [x] **p5: Priority claim unverified** → Sec 5 ("to our knowledge")

### Newly identified issues (this pass)
- [x] **A1: LIWC computation code missing from repo** → Sec 7
- [x] **A2: Simpson's paradox risk** → Sec 3 (occupation stratification)
- [x] **A3: PageRank–conflict tautology** → Sec 6 (limitations)
- [x] **A4: Pseudoreplication** → Sec 3a + REJECT-6
- [x] **A5: Quote attribution uncertainty** → Sec 3c, 3d, 6
- [x] **A6: Mention disambiguation accuracy** → Sec 6
- [x] **A7: Non-native English speakers** → Sec 3, 6
- [x] **A8: Ecological fallacy** → Sec 5 (scope claims)
- [x] **A9: Missing covariates (nationality, party)** → Sec 3 (nationality covariate)
- [x] **A10: Reproducibility** → Sec 7
- [x] **A11: Static PageRank on dynamic network** → REJECT-7
- [x] **A12: Effect-size-at-scale illusion** → REJECT-8
- [x] **A13: Binning sensitivity** → Sec 3d (see below)

---

## 1. Scope and Positioning

**Goal**: Demonstrate that Quotegraph's directed structure enables linguistic analyses impossible with speaker-only data (Quotebank) or undirected networks.

**Length**: ~1.5 pages (one of three example applications in a dataset paper). NOT a full psycholinguistics study.

**Core argument** (one sentence): *In news quotations, the linguistic properties of a quote are associated as much with whom the speaker mentions as with who the speaker is — and the gap between speaker and target prominence or age predicts systematic asymmetries in power, conflict, and self-reference language.*

**What this section is NOT**:
- Not a replication of Pennebaker / Kacewicz / Newman (speaker-level effects are validation, not contribution)
- Not a psycholinguistic theory paper (no causal claims, no theoretical framework testing)
- Not an exhaustive survey of 80+ LIWC categories

---

## 2. Finding Selection (3–4 headline findings)

### Selection rationale (state explicitly in paper)

> "We select LIWC-22 categories with established prior predictions from speaker-level studies (Pennebaker & Stone 2003; Kacewicz et al. 2014), enabling direct comparison of speaker-only vs. speaker+target decomposition. We focus on categories where the target or interaction term is significant after controlling for speaker properties — i.e., findings that require Quotegraph's directed structure."

### Selected findings

| # | Finding | Why it survives scrutiny | Figure |
|---|---------|------------------------|--------|
| 1 | **Clout–Power mirror across age gap** | Genuine interaction: Clout ↗ and Power ↘ across age-diff axis. Hard to explain by occupation alone (both track hierarchy, not topic). Validates directed edges decompose speaker self-positioning (Clout) from target characterization (Power). | `fig_age_diffs` (c)+(d) |
| 2 | **Conflict peaks at zero age difference** | Genuine interaction (inverted-U). Neither speaker age nor target age alone produces this shape. Critical test: does it hold within-politicians? | `fig_age_diffs` (a) |
| 3 | **"Punching up" conflict asymmetry (PageRank)** | ~2× more conflict when source << target. Consistent with media criticism patterns (van Dijk 1993). The asymmetry is the finding. | `fig_pagerank_diffs` (b) |
| 4 | **Peer solidarity in I/we/Tone (PageRank)** | Three categories peak at PR-diff ≈ 0 (inverted-U). Coherent profile. Extends Kacewicz: status effects are relational, not absolute. | `fig_pagerank_diffs` (c)+(e)+(f) |

### What to demote to supplementary or drop

- **Speaker-level replications** (I×age, Clout×age): one-sentence validation, no figures in main text
- **Gender findings**: supplementary (replication of known effects; heavy confounds)
- **Occupation-confounded target effects** (Power×age, Anger×age): supplementary, framed honestly as compositional
- **"Three archetypes" framework**: drop entirely (post-hoc, untested)
- **"12 novel findings"**: reduce to 3–4 well-defended; honesty > inflation

### Figures for paper body

| Figure | Content | Findings |
|--------|---------|----------|
| `fig_age_diffs` | 2×3 age-difference curves | Findings 1 + 2 |
| `fig_pagerank_diffs` | 2×3 PageRank-difference curves | Findings 3 + 4 |

### Figures for supplementary

`fig_gender_core`, `fig_gender_emotion`, `fig_age_core`, `fig_age_interactions`, `fig_pagerank_core` — with brief captions.

---

## 3. Statistical Framework

### 3a. Primary analysis: Mixed-effects regression

For each selected LIWC category, fit:

```
LIWC_score ~ source_attr + target_attr + source_attr × target_attr
             + occupation_source + occupation_target + nationality_source
             + log(quote_length)
             + (1 | source_id) + (1 | target_id)
```

- **Random intercepts** for source and target (REJECT-6: pseudoreplication)
- **Occupation** as fixed-effect covariate (REJECT-2)
- **Nationality** as covariate (A7, A9)
- **Quote length** as covariate (M2)
- Report: coefficient estimates, 95% CIs, FDR-corrected p-values, **partial η²** (REJECT-1, REJECT-8)

For difference-plot analyses (age gap, PageRank gap):
```
LIWC_score ~ poly(gap, 2) + source_attr + occupation_source + occupation_target
             + nationality_source + log(quote_length)
             + (1 | source_id) + (1 | target_id)
```
- Quadratic term tests inverted-U / U-shapes
- Report R², AIC comparison (linear vs. quadratic), partial η² for gap terms

**Computational note**: With ~8M quotes and ~500k random-effect levels, use `lme4` (R) or `MixedModels.jl` (Julia). If infeasible, fall back to OLS with **dyad-clustered standard errors** (Stata/R `sandwich` package) and state this.

### 3b. Variance decomposition (for supplementary heatmaps)

For each heatmap, report partial η² for source main effect, target main effect, interaction, residual. Replaces "pure vertical gradient" / "diagonal brightening" with numbers.

### 3c. Sample sizes and data quality

- **Report N** (quotes) per heatmap cell and per diff-plot bin (m6)
- **Minimum cell size**: exclude bins with N < 100
- **Quote length filter**: ≥ 25 words (REJECT-5)
- **Report**: total N quotes after filtering, N unique speakers, N unique targets, N unique dyads
- **Report** Quotebank attribution accuracy from Vaucher et al. (2021) (A5)
- **Report** entity linking accuracy from Culjak et al. (2022/2023) (A6)

### 3d. Robustness checks (supplementary)

1. **Occupation stratification** (REJECT-2): Repeat age-diff findings within politicians-only and sportspeople-only. If conflict inverted-U survives within-politicians, that is your strongest single piece of evidence.
2. **Temporal split** (m7): 2008–2014 vs. 2015–2020. Key patterns replicate?
3. **Permutation baseline** (m6): Shuffle source-target assignments (preserve marginals). Show patterns are absent in shuffled data.
4. **Attribution confidence** (A5): Restrict to quotes with Quotebank P(speaker) > 0.8.
5. **Binning sensitivity** (A13): Recompute difference plots with half and double the bin width. Inverted-U / V-shape should be robust to binning choice.

---

## 4. Data Preprocessing

1. **Quote length filter**: ≥ 25 words (LIWC-22 summary dims unreliable below this)
2. **Self-loop removal**: already done per paper
3. **Deduplication**: Quotebank grouping already applied
4. **LIWC version**: state LIWC-22 explicitly; note Tone/Clout/Authentic/Analytic are composite algorithms
5. **Age computation**: use age at time of quotation (not current age) — if not already done, this must be fixed
6. **Binning**: age 5yr bins; age-diff 5yr bins; PageRank log₁₀ quantile bins; PR-diff bins
7. **Report**: total N after filtering

---

## 5. Writing Strategy

### Framing (M3, M4, m3, m4)

- **Scope every claim**: "In English-language news quotations" or "among quoted public figures." Never generalize to "people" unqualified.
- **Drop CAT entirely**. It's about face-to-face conversation; Quotegraph is about discourse about third parties.
- **Cite Brown & Levinson once** as motivation. Do not invoke W = D + P + R repeatedly.
- **Primary frame**: media discourse analysis (van Dijk 1993; Entman 1993 framing theory).
- **Secondary frame**: Kacewicz et al. (2014) — the specific prior LIWC study we extend to relational status.

### Language discipline (REJECT-3, p1)

- **Zero causal verbs**: replace "elicits", "drives", "suppresses", "produces" → "is associated with", "co-occurs with", "correlates with", "predicts"
- **Drop invented terms**: no "paternalistic solidarity", no "age-homophilous adversarialism"
- **Define "punching up/down"** on first use: "quotations where log₁₀PR_source − log₁₀PR_target < 0 [> 0]"
- **Priority claim**: "To our knowledge, this is the first large-scale LIWC analysis of directed speaker→mention quotation data"

### Verdict taxonomy (m1)

4 labels only: **Replicates** | **Extends** | **Diverges** | **Novel**

### SST framing (m5)

"The speaker-marginal age–Tone relationship diverges from SST's positivity effect, likely reflecting occupational composition. The direction-conditional pattern (older→younger = more positive Tone) is consistent with a contextualized version of SST at the relational level."

### Ecological fallacy (A8)

State: "All reported effects are associations at the demographic-bin level and should not be interpreted as individual-level psychological traits."

---

## 6. Limitations Paragraph

All 9 items must appear:

1. **Domain validity**: LIWC-22 validated on personal writing; news quotes are a different register, selected/edited by journalists. (M2, REJECT-5)
2. **Selection bias**: Only people quoted in English-language news. Not representative of general language. (M3)
3. **Occupation confound**: Age and PageRank partly reflect occupation. Stratified analyses in supplementary. (M1, REJECT-2)
4. **Mention ≠ interaction**: Edges are discourse *about*, not conversation *with*. (M4)
5. **PageRank circularity**: Controversy drives mentions, inflating PageRank; conflict–PageRank association may be partly tautological. (A3)
6. **Attribution and linking noise**: Quotebank speaker attribution and heuristic entity linking introduce errors. Sensitivity analyses in supplementary. (A5, A6)
7. **Language proficiency**: ~48% of top nodes are non-US nationals; LIWC norms may not apply equally. (A7)
8. **Static PageRank on dynamic network**: PageRank computed on 2008–2020 aggregate; individual prominence varies over time. (A11, REJECT-7)
9. **Temporal scope**: Patterns aggregated over 12 years; political discourse shifted (e.g., post-2016). Split in supplementary. (m7)

---

## 7. Reproducibility

- [ ] Commit LIWC analysis pipeline (quote extraction, scoring, aggregation, regressions)
- [ ] State LIWC-22 license requirement; provide code even though LIWC is proprietary
- [ ] Release **aggregated data**: per-cell means, SEs, Ns for all bins — allows verification without LIWC license
- [ ] Include regression output tables in supplementary

---

## 8. Suggested Paper Section Structure (~1.5 pages)

**N.1 Setup** (~0.25 pages)
- Quotegraph's directed structure decomposes quotation language into source (speaker) and target (mentioned person).
- We apply LIWC-22 and analyze how linguistic properties vary with attributes of both endpoints.
- Category selection rationale [state explicitly].
- Data: N quotes (≥25 words), N speakers, N targets.

**N.2 Speaker-level validation** (~0.2 pages)
- One sentence each: "I" decreases with speaker age, Clout increases — replicating Pennebaker & Stone (2003) and Kacewicz et al. (2014). No figures.
- Purpose: confirms LIWC captures meaningful signal in this domain.

**N.3 Age-gap decomposition** (~0.4 pages)
- Figure: `fig_age_diffs`
- Clout–Power mirror (Finding 1) + conflict inverted-U (Finding 2)
- Regression results inline (coefficients, CIs, η²)
- Occupation confound note → supplementary

**N.4 Prominence-gap decomposition** (~0.4 pages)
- Figure: `fig_pagerank_diffs`
- Punching-up conflict asymmetry (Finding 3) + peer solidarity (Finding 4)
- Regression results inline

**N.5 Discussion** (~0.25 pages)
- Directed structure reveals LIWC categories (Clout, I, we) are modulated by speaker–target gap
- Extends Kacewicz et al. (2014) from absolute to relational status
- Limitations cross-reference

---

## 9. Potential Expansions: Other Signals Beyond LIWC

### 9a. Transformer-based sentiment / stance
- Fine-tuned RoBERTa stance classifier on quotation text.
- Stance (pro/against/neutral toward the mentioned person) directly labels the signed valence of each directed edge.
- **Recommended as the strongest third example application**: most directly demonstrates directed-edge value; complements LIWC (how language varies) with what valence it carries; established shared-task models provide credibility.

### 9b. Toxicity / incivility scoring
- Perspective API or detoxify model. Directed toxicity network: who is toxic toward whom, by gender/age/prominence.

### 9c. Moral Foundations analysis
- Extended Moral Foundations Dictionary (eMFD; Hopp et al. 2021). Five dimensions: care/harm, fairness/cheating, loyalty/betrayal, authority/subversion, purity/degradation.
- Do speakers invoke authority/subversion when "punching up"? Loyalty/betrayal toward peers?

### 9d. Quotation verb analysis (editorial signal)
- The verb journalists use to introduce quotes ("said" vs. "claimed" vs. "argued"). This is NOT a speaker signal — it's editorial framing. Available in Quotebank article-centric format.

### 9e. Topic modeling conditioned on edge attributes
- BERTopic/LDA on quotes, conditioned on source-target attribute pairs. What topics dominate in cross-party vs. same-party quotations?

### 9f. Embedding-based linguistic divergence
- Sentence embeddings per quote; measure how a speaker's language shifts when mentioning different target types. Tests register-shift hypothesis directly.

---

## 10. Implementation Steps

1. Find/recommit LIWC analysis code (or rewrite)
2. Apply quote-length filter (≥ 25 words)
3. Compute exact cell means, SEs, Ns for all bins
4. Fit mixed-effects regressions with covariates
5. Run FDR correction; report effect sizes (η²)
6. Run robustness checks (occupation stratification, temporal split, permutation, attribution confidence, binning sensitivity)
7. Rewrite paper section per Sec 8 structure
8. Regenerate `fig_age_diffs` and `fig_pagerank_diffs` with larger fonts
9. Move other figures to supplementary
10. Write limitations paragraph (all 9 items)
11. Commit pipeline + release aggregated data

---

## Verification Checklist

- [ ] Mixed-effects regressions run for all selected categories
- [ ] FDR-corrected p-values AND partial η² reported for every claim
- [ ] Occupation stratification: conflict inverted-U tested within-politicians
- [ ] Temporal split shows stability
- [ ] Permutation baseline shows patterns are non-random
- [ ] All values extracted from data (no "~" estimates)
- [ ] Paper section ≤ 1.5 pages
- [ ] Causal language audit: zero causal verbs
- [ ] Limitations paragraph addresses all 9 items
- [ ] Analysis code committed; aggregated data released
- [ ] Binning sensitivity checked (half/double width)
- [ ] N per bin reported; low-N bins excluded
