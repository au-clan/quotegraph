# Critical Review: LIWC Example Application for EMNLP Submission

> Self-audit of `liwc_literature_comparison.md` and associated figures.
> Goal: identify every issue an EMNLP reviewer would raise, ranked by severity.

---

## FATAL — Will sink the paper if not addressed

### F1. No statistical tests anywhere

The entire analysis is descriptive. Every claim ("~4–5 power", "peak at diff ≈ 0") is read off heatmaps or line plots by eye. EMNLP reviewers expect:

- Confidence intervals or standard errors on every reported value
- Significance tests (or Bayesian equivalents) for every comparison
- Multiple-comparison corrections — with 80+ LIWC categories × 3 dimensions × source/target, the family-wise error rate is enormous
- Effect sizes (Cohen's d or η²), not just "dramatically higher"

**Fix**: Run regression models. For heatmap effects: two-way ANOVA or mixed-effects model with source-age-bin × target-age-bin, report F-statistics and partial η². For difference plots: report Spearman ρ or polynomial regression fits with R². For categorical (gender): report the 2×2 cell means with bootstrap CIs. Apply Bonferroni or FDR correction across all tested categories and state the correction method. Present the exact values from data, not visual estimates.

### F2. Cherry-picked categories with no selection rationale

The analysis examines ~15 of 80+ LIWC categories. There is no stated criterion for why these 15 and not others. A reviewer will ask: "Did you look at all 80 and pick the ones with nice patterns?"

**Fix**: Either (a) pre-register the category selection with a stated rationale (e.g., "categories with prior literature predictions" — but state this explicitly and cite the predictions before showing results), or (b) run the full battery of 80+ categories and report how many show significant effects, with FDR correction. Option (a) is more practical for a paper section; option (b) belongs in supplementary material.

### F3. Causal language without causal design

Pervasive throughout. Examples:
- "elicits anger" (Sec 4c) — implies the target *causes* anger
- "drives power language" — implies the gap *produces* power words
- "speakers accommodate" — implies a deliberate behavioral adaptation
- "produces maximally confrontational quotations" (Sec 5a)
- "suppresses self-reference" (Sec 3c)

This is observational data. All effects are correlational. EMNLP reviewers are trained to catch this.

**Fix**: Systematic language audit. Replace all causal verbs with associative ones: "is associated with", "co-occurs with", "correlates with", "predicts" (in the statistical sense). Reserve causal language for the literature references where the original authors used experimental designs.

---

## MAJOR — Serious weaknesses that reviewers will flag

### M1. Occupation confound acknowledged but not controlled

Theme 5 honestly admits the confound (young = sports, old = politics, high-PR = politics). But then the rest of the analysis proceeds as if the effects are about age/prominence per se. Nearly every "novel" finding in Sections 4–6 could be rewritten as: "Politicians are discussed with more power/conflict/anger language than athletes."

**Fix**: At minimum, report the occupation distribution across age bins and PageRank bins (a simple table). Ideally, add occupation as a covariate in regressions and show which effects survive. If Quotegraph has Wikidata occupation labels, stratify the analysis: show the age-difference plots *within* politicians only and *within* athletes only. If the patterns hold within occupation, the finding is much stronger. If they vanish, be honest — the "age effect" is really an occupation effect.

### M2. LIWC validity in news quotations is unexamined

LIWC-22 was developed and validated on personal writing (essays, blogs, social media, emails). News quotations are a fundamentally different register:
- Quotes are *selected* by journalists (selection bias on what gets quoted)
- Quotes may be *edited*, truncated, or paraphrased
- Quotes are *contextualized* — the surrounding article frames interpretation
- Quote length varies enormously; short quotes have unreliable LIWC scores

The analysis applies LIWC as if its psychometric properties transfer directly to this domain. No validation is offered.

**Fix**: (a) Report the distribution of quote lengths and filter or flag quotes below LIWC's recommended minimum (typically 50 words; some categories need 100+). (b) Acknowledge the domain shift explicitly in the paper's limitations. (c) If possible, validate on a small sample — e.g., do human annotators agree with LIWC's Tone/Conflict scores on a random sample of Quotegraph quotes?

### M3. Selection bias: who gets quoted is not random

Not everyone speaks in the news. The sample is overwhelmingly politicians, athletes, executives, celebrities. Claims about "age effects on language" are really claims about "age effects on language *among people prominent enough to be quoted in English-language news media*." Generalizing to "people" or "speakers" is inappropriate.

**Fix**: Frame all findings explicitly as properties of *public discourse in news media*, not of language in general. E.g., "In news quotations, older targets are associated with more power language" — not "Mentioning older people elicits power language." Add a limitations paragraph noting the sample is restricted to quoted public figures.

### M4. "Mention" ≠ "interaction"

The analysis frequently invokes Communication Accommodation Theory, politeness theory, and face theory — all developed for *dyadic interaction*. But a Quotegraph edge means "A was quoted saying something that mentions B." This is:
- Not a conversation between A and B
- Not necessarily directed *at* B
- Possibly about B in their absence
- Filtered through journalistic selection

The CAT/B&L framing is acknowledged as imperfect (e.g., "topic-driven register shift" caveat in Theme 2), but then the analysis still heavily relies on it.

**Fix**: Be more disciplined about when interaction theories apply. Use them as *analogies* with explicit hedging, not as theoretical frameworks that "predict" your results. Consider framing through media studies / framing theory / critical discourse analysis instead — these theories are actually designed for public discourse about third parties. Reduce reliance on CAT; keep Brown & Levinson only where the original formulation explicitly covers discourse *about* a referent (their variable R = rating of the imposition does not require face-to-face interaction).

### M5. Numerical values are eyeballed, not computed

Throughout: "~4–5 power", "peak at diff ≈ 0 (~0.45)", "faint diagonal brightening." These are visual readings from plots, not computed statistics. Some are hedged with "~" but EMNLP expects exact values.

**Fix**: Extract the actual aggregated values from the underlying data and report them to 2 decimal places. Replace all "~" estimates with exact computed means. For difference plots, fit curves (LOESS or polynomial) and report inflection points / extrema analytically.

### M6. Heatmap patterns described without quantification

Claims like "pure vertical gradient" (Sec 4a), "bright diagonal block" (Sec 4b), and "faint diagonal brightening" (Sec 4d) are subjective visual interpretations. Different readers may see different patterns.

**Fix**: Quantify. For "vertical gradient" (target-dominated): compute variance explained by target-age vs. source-age in a two-way ANOVA. For "diagonal block": compute the mean LIWC score on-diagonal vs. off-diagonal and test the difference. For "faint brightening": if it's too faint to quantify, don't claim it.

---

## MODERATE — Will weaken the paper; should fix if possible

### m1. Inconsistent verdict taxonomy in summary table

The "Verdict" column uses at least 10 different labels (Strongly consistent, Consistent, Partially consistent, Novel pattern, Counterintuitive, Contradicts, Extends, Resolves, etc.). This makes it hard to compare across rows and is not a principled taxonomy.

**Fix**: Use a 4-level scheme consistently: **Replicates** (same direction, similar magnitude), **Extends** (same direction with new moderator/dimension), **Diverges** (different direction, explainable), **Contradicts** (different direction, unexplained). Map each current verdict to one of these four.

### m2. "Novel" count of 12 is overclaimed

Several "novel findings" are direct consequences of occupation composition:
- Power × age target dominance (finding #1): politicians hold power; older targets are politicians
- Anger × age target (finding #9): people are angry about politics; older targets are politicians
- Power × age diff mirror (#2): restatement of #1 in difference-plot form

These are not psycholinguistic novelties — they are descriptive properties of who appears in news. Claiming 12 novel findings when several reduce to "politicians are discussed with political language" will draw reviewer skepticism.

**Fix**: Separate findings into (a) genuinely novel interaction effects that hold after considering composition (e.g., conflict inverted-U at age diff ≈ 0, Clout–Power mirror) and (b) descriptive patterns that reflect sample composition. Be honest about which is which. A smaller number of well-defended novel findings is far stronger than an inflated count.

### m3. Brown & Levinson (1987) is over-invoked

B&L's politeness framework is cited 8+ times, often as loose justification. B&L's original weight function W = D + P + R was about face-to-face politeness strategies between an S and an H. Applying it to LIWC scores on news quotes about a third-party referent is a substantial theoretical leap. Reviewers in computational social science are familiar with B&L and will notice the stretch.

**Fix**: Cite B&L once as motivation for the hypothesis that power/distance variables shape language. Do not repeatedly invoke the "W = D + P + R" formula as if it makes quantitative predictions for this setting. Acknowledge that the mapping from B&L's politeness strategies to LIWC category scores is not established.

### m4. CAT "about" vs. "with" distinction is acknowledged but then ignored

The analysis correctly notes (Theme 2) that CAT is about conversation *with* someone, while Quotegraph captures discourse *about* someone. But then Sections 4d, 5c, 5f continue to invoke CAT as if it predicts these patterns.

**Fix**: State the caveat once, clearly, in the introduction to the example application. Then consistently use language like "parallels CAT-like patterns" rather than "consistent with CAT predictions." Better yet, cite media framing / agenda-setting / critical discourse analysis literature (which IS about public discourse about third parties) as the primary theoretical frame.

### m5. "Contradicts SST" and "Extends SST" used for the same phenomenon

Tone × age (Sec 2, fig_age_core) is labeled "Contradicts" SST. Tone × age diff (Sec 5b) is labeled "Extends SST directionally." But the age-diff plot is derived from the same underlying data as the heatmap. The analysis can't both contradict and extend the same theory with the same data — unless the framing is made much more precise about what exactly contradicts (speaker-level prediction) and what extends (directional prediction).

**Fix**: Rewrite to be precise: "The speaker-marginal contradicts SST's positivity effect, likely due to occupation confounds. However, the *direction-conditional* pattern (old→young = positive) is consistent with a weaker, contextualized version of SST, suggesting the positivity effect operates at the relational level rather than the speaker level."

### m6. No null model or baseline

When you see that conflict peaks at age-diff ≈ 0, how do you know this isn't an artifact of the data structure? If same-age pairs are simply more common (more data = more extreme means), or if same-age pairs come from a specific domain, the pattern could be spurious.

**Fix**: Report N (number of quotation pairs) per age-diff bin or heatmap cell. Show that the patterns are not driven by low-N bins at the extremes. Ideally, compare to a permutation baseline: shuffle source-target assignments while preserving marginals and show that the real data has patterns absent in the shuffled version.

### m7. Temporal dimension is completely absent

Quotegraph spans 2008–2020 (from Quotebank). Are the patterns stable across this period? Political language shifted dramatically (e.g., post-2016 polarization). If the "punching up" conflict asymmetry is entirely driven by 2017–2020 Trump coverage, that changes the interpretation.

**Fix**: At minimum, mention the time span. Ideally, replicate one or two key findings in separate time windows (e.g., 2008–2014 vs. 2015–2020) to show temporal robustness.

---

## MINOR — Polish issues

### p1. Loaded terminology

"Paternalistic solidarity" (Sec 4e, 5e) is a sociologically charged term. "Age-homophilous adversarialism" (Sec 4b) sounds invented. "Punching up/down" is informal. These will land differently with different reviewers.

**Fix**: Use "paternalistic solidarity" only if you can cite it as an established construct. Otherwise, describe the pattern neutrally ("older speakers use more affiliative language toward younger targets"). "Punching up/down" is vivid but should be introduced with scare-quotes and defined precisely on first use.

### p2. Literature citations are incomplete

Several citations lack publication venue or are ambiguous:
- "Luoto 2021" — which Luoto 2021? Full reference needed
- "Eckert 2003" — likely "Language and Adolescent Peer Groups" but not specified
- "Kram 1985" — management/mentoring literature, unusual for EMNLP audience
- "Gonzales et al. 2010" — LSM paper, needs full citation

**Fix**: Ensure every citation has a complete reference. For EMNLP, check whether the cited work will be familiar to the CL/NLP audience; if not, provide brief context.

### p3. Inconsistent precision

Some values reported as "~4–5" (wide range), others as "~0.453" (three decimal places). The mix suggests some were read from plots and others from data tables. This inconsistency undermines confidence.

**Fix**: Report all values at consistent precision from the actual computed data.

### p4. The "three archetypes" (Theme 2) are post-hoc

The peer/punching-up/punching-down trichotomy is a nice narrative framework but is not tested. It's an interpretive overlay, not a discovered structure (e.g., via clustering).

**Fix**: Either (a) present it explicitly as interpretive framing ("we organize the findings into three descriptive profiles...") rather than as a discovered result, or (b) actually test it — e.g., cluster the LIWC profiles by age-diff or PR-diff bin and show that three clusters emerge.

### p5. "First LIWC study with directed network" — verify the claim

The analysis claims "Quotegraph is the first large-scale LIWC study with a directed speaker→mention network." This is a strong priority claim. If any prior work has done LIWC on directed discourse networks (even smaller-scale), a reviewer finding it will be damaging.

**Fix**: Do a thorough literature search for LIWC + directed networks / quotation networks / speaker-mention. Check Quotebank papers, media analysis papers. If the claim holds, keep it. If not, soften to "one of the first" or "the first at this scale."

---

## Action Priority Matrix

| Priority | Issue | Effort | Impact on paper |
|----------|-------|--------|-----------------|
| 1 | F1: Add statistical tests | High | Paper rejected without this |
| 2 | F3: Remove causal language | Low | Easy fix, high reviewer sensitivity |
| 3 | F2: State category selection rationale | Low | One paragraph fixes it |
| 4 | M1: Address occupation confound | Medium | Controls/stratification needed |
| 5 | M3: Frame as "public discourse" | Low | Language changes throughout |
| 6 | M5: Replace eyeballed values | Medium | Needs data access |
| 7 | M2: Discuss LIWC domain validity | Low | Limitations paragraph |
| 8 | M4: Fix interaction-theory framing | Low | Rewrite theoretical sections |
| 9 | m6: Add sample sizes per bin | Medium | Needs data access |
| 10 | m2: Reduce novel-findings count | Low | Honesty strengthens the paper |
| 11 | m5: Fix SST contradiction/extension | Low | One paragraph rewrite |
| 12 | m7: Mention temporal span | Low | One sentence minimum |

---

## What IS strong about the analysis

To be fair — the core contribution is genuinely valuable:

1. **The directed decomposition idea is sound.** Separating speaker from target effects is a real methodological advance over speaker-only LIWC studies.
2. **The Clout–Power mirror** (age diff, PR diff) is the cleanest finding. It's hard to explain by confounds and validates the directed structure.
3. **The conflict × age-diff inverted-U** is a genuine interaction effect (not reducible to marginals).
4. **The PR-diff plots** as a whole tell a coherent story about hierarchy in public discourse.
5. **The occupation confound is honestly acknowledged** — many papers would hide it.

The analysis doesn't need to be scrapped — it needs to be *tightened*. Fewer claims, each better defended, with statistical backing and honest scope limitations. An EMNLP paper with 4 well-defended novel findings and proper statistics is far stronger than one claiming 12 novel findings without tests.
