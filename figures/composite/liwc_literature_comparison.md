# LIWC Analysis of Quotegraph: Literature Comparison

> Mapping of composite figures to psycholinguistics literature findings.
> Quotegraph LIWC scores are computed on quotation text.

## Figure–Panel Reference

| Figure | Panel | Category | Dimension |
|--------|-------|----------|-----------|
| `fig_gender_core` | (a) | First-person singular (*I*) | Gender |
| `fig_gender_core` | (b) | Analytic thinking | Gender |
| `fig_gender_core` | (c) | Authenticity | Gender |
| `fig_gender_core` | (d) | Clout | Gender |
| `fig_gender_core` | (e) | Tone | Gender |
| `fig_gender_core` | (f) | Power | Gender |
| `fig_gender_emotion` | (a) | Anger | Gender |
| `fig_gender_emotion` | (b) | Anxiety | Gender |
| `fig_gender_emotion` | (c) | Sadness | Gender |
| `fig_gender_emotion` | (d) | Conflict | Gender |
| `fig_gender_emotion` | (e) | First-person plural (*we*) | Gender |
| `fig_gender_emotion` | (f) | Third-person plural (*they*) | Gender |
| `fig_age_core` | (a) | First-person singular (*I*) | Age |
| `fig_age_core` | (b) | Clout | Age |
| `fig_age_core` | (c) | Past focus | Age |
| `fig_age_core` | (d) | Analytic thinking | Age |
| `fig_age_core` | (e) | Tone | Age |
| `fig_age_core` | (f) | First-person plural (*we*) | Age |
| `fig_pagerank_core` | (a) | Tone | PageRank |
| `fig_pagerank_core` | (b) | Power | PageRank |
| `fig_pagerank_core` | (c) | First-person singular (*I*) | PageRank |
| `fig_pagerank_core` | (d) | Clout | PageRank |
| `fig_pagerank_core` | (e) | First-person plural (*we*) | PageRank |
| `fig_pagerank_core` | (f) | Analytic thinking | PageRank |
| `fig_age_interactions` | (a) | Power | Age (heatmap) |
| `fig_age_interactions` | (b) | Conflict | Age (heatmap) |
| `fig_age_interactions` | (c) | Anger | Age (heatmap) |
| `fig_age_interactions` | (d) | Authenticity | Age (heatmap) |
| `fig_age_interactions` | (e) | Affiliation | Age (heatmap) |
| `fig_age_interactions` | (f) | Tone | Age (heatmap) |
| `fig_age_diffs` | (a) | Conflict | Age difference |
| `fig_age_diffs` | (b) | Tone | Age difference |
| `fig_age_diffs` | (c) | Clout | Age difference |
| `fig_age_diffs` | (d) | Power | Age difference |
| `fig_age_diffs` | (e) | Affiliation | Age difference |
| `fig_age_diffs` | (f) | First-person plural (*we*) | Age difference |
| `fig_pagerank_diffs` | (a) | Power | PageRank difference |
| `fig_pagerank_diffs` | (b) | Conflict | PageRank difference |
| `fig_pagerank_diffs` | (c) | Tone | PageRank difference |
| `fig_pagerank_diffs` | (d) | Clout | PageRank difference |
| `fig_pagerank_diffs` | (e) | First-person singular (*I*) | PageRank difference |
| `fig_pagerank_diffs` | (f) | First-person plural (*we*) | PageRank difference |

---

## 1. Gender Findings

### fig_gender_core (a): First-person singular "I"
- **Literature**: Women >> Men (d=0.66; Pennebaker 2011)
- **Quotegraph**: FF (~4.07) >> FM (~3.22) > MM (~2.89) > MF (~2.82)
- **Assessment**: **Strongly consistent.** F speakers (~3.6 avg) use dramatically more "I" than M speakers (~2.85 avg). FF is 43% higher than MM. MF < MM — men use fewer "I" words when mentioning women.

### fig_gender_core (b): Analytic thinking
- **Literature**: Men > Women (d=0.68; Luoto 2021)
- **Quotegraph**: MF (~53.5) > FM (~52) > MM (~50.5) > FF (~45.9)
- **Assessment**: **Consistent for speaker effect.** M speakers (~52) > F speakers (~49). FF is dramatically lowest. Cross-gender mention increases Analytic for both sexes (MF > MM, FM > FF).

### fig_gender_core (c): Authenticity
- **Literature**: Women > Men (inferred from component features)
- **Quotegraph**: MM (~39.3) ≈ FF (~39.0) > FM (~37.4) > MF (~35.9)
- **Assessment**: **Novel pattern — same-gender > cross-gender, not F > M.** Same-gender pairs (FF, MM) have higher Authenticity than cross-gender pairs (FM, MF). Speakers adopt a more guarded, self-monitored communication style when discussing the opposite gender.

### fig_gender_core (d): Clout
- **Literature**: Men >= Women (Kacewicz et al. 2014)
- **Quotegraph**: MF (~55.1) > FM (~54.2) > MM (~54.0) > FF (~52.5)
- **Assessment**: **Consistent for speaker effect.** M speakers (~54.5) > F speakers (~53.3). Small effect. Cross-gender mentions boost Clout.

### fig_gender_core (e): Tone
- **Literature**: Women use more positive emotion words (Newman et al. 2008; Park et al. 2016, d=0.63)
- **Quotegraph**: FF (~51) > MM (~48) > MF (~47) > FM (~45.7)
- **Assessment**: **Partially consistent, with a novel twist.** FF has the highest tone (confirming positivity finding), but FM has the *lowest* — lower than any male-speaker pair. When women speak about men in the news, it is disproportionately in conflictual/negative contexts.

### fig_gender_core (f): Power
- **Literature**: Expected M > F
- **Quotegraph**: FM (~2.80) >> MF (~2.51) > MM (~2.43) >> FF (~1.89)
- **Assessment**: **Mention-driven, not speaker-driven.** Male mentions receive more power language (avg ~2.62) than female mentions (avg ~2.20). FM is dramatically highest — women talking about men use the most power vocabulary. Likely reflects that men who are mentioned are disproportionately politicians/leaders.

### fig_gender_emotion (a): Anger
- **Literature**: Men > Women (Newman et al. 2008; d=0.32)
- **Quotegraph**: FM (~0.138) > FF (~0.131) > MF (~0.123) > MM (~0.118)
- **Assessment**: **Contradicts the speaker-level prediction.** Female speakers use MORE anger words (F avg ~0.135, M avg ~0.120). Cross-gender mentions increase anger. The reversal may reflect that female public figures who get quoted in news are disproportionately in adversarial contexts.

### fig_gender_emotion (b): Anxiety
- **Literature**: Women > Men (Newman et al. 2008)
- **Quotegraph**: FM (~0.110) > FF (~0.096) ≈ MF (~0.094) > MM (~0.081)
- **Assessment**: **Consistent.** Female speakers use more anxiety words. FM is notably highest.

### fig_gender_emotion (c): Sadness
- **Literature**: Women > Men (Newman et al. 2008)
- **Quotegraph**: FF (~0.101) > FM (~0.097) > MF (~0.090) > MM (~0.068)
- **Assessment**: **Consistent.** Clear speaker-gender gradient with F >> M. FF uses ~50% more sadness words than MM.

### fig_gender_emotion (d): Conflict
- **Literature**: Men > Women (Park et al. 2016)
- **Quotegraph**: FM (~0.453) > MM (~0.420) > MF (~0.404) > FF (~0.363)
- **Assessment**: **Mention-driven.** Mentioning men = more conflict language. FM (women mentioning men) is highest. FF is lowest. The mention effect dominates.

### fig_gender_emotion (e): First-person plural "we"
- **Literature**: No consistent gender difference (Newman et al. 2008); used more by high-status speakers (Kacewicz et al. 2014)
- **Quotegraph**: MM (~1.63) >> MF (~1.34) ≈ FM (~1.33) > FF (~1.27)
- **Assessment**: **Men use significantly more "we."** May be confounded with status — men in Quotegraph are more prominent/central.

### fig_gender_emotion (f): Third-person plural "they"
- **Literature**: Women > Men for third-person pronouns (Newman et al. 2008)
- **Quotegraph**: MM (~0.76) > MF (~0.73) > FM (~0.71) ≈ FF (~0.70)
- **Assessment**: **Contradicts.** M speakers use MORE "they." Effect size is small.

---

## 2. Age Findings

### fig_age_core (a): First-person singular "I"
- **Literature**: Decreases with age (Pennebaker & Stone 2003, one of the most robust findings)
- **Quotegraph**: Young speakers (20–29) use ~5+ "I" words, old speakers (65–74) use ~2.5–3. Clean gradient.
- **Assessment**: **Strongly consistent.** One of the clearest patterns across all heatmaps. Steep, monotonic decrease with speaker age.

### fig_age_core (b): Clout
- **Literature**: Increases with age/status (Kacewicz et al. 2014)
- **Quotegraph**: Older speakers (45–74) have Clout ~60–64, young speakers (15–29) have ~48. The strongest age pattern.
- **Assessment**: **Strongly consistent.** Clout increases ~16 points from youngest to oldest. Speakers also show more Clout when mentioning younger targets — linguistically more deferential toward older people.

### fig_age_core (c): Past focus
- **Literature**: Past tense DECREASES with age (counterintuitive; Pennebaker & Stone 2003)
- **Quotegraph**: Young speakers (15–29) use the MOST past-tense language (~6), older speakers (55–74) use less (~4.5–5).
- **Assessment**: **Consistent with the counterintuitive finding.** Younger speakers use more past tense, confirming that older people do NOT dwell on the past more. Likely domain-confounded: young sportspeople recount events; politicians use present/future tense.

### fig_age_core (d): Analytic thinking
- **Literature**: Increases with age (inferred from component features; Pennebaker 2011)
- **Quotegraph**: Dominated by TARGET age: mentioning older targets (55–74) = high Analytic (~54), mentioning younger targets (15–29) = low (~42). Speaker age secondary positive effect.
- **Assessment**: **Consistent for both speaker and target.** Older speakers are more analytic, and mentioning older people elicits more analytic language. Likely occupation-confounded.

### fig_age_core (e): Tone
- **Literature**: Positive emotion INCREASES with age (Pennebaker & Stone 2003; socioemotional selectivity theory)
- **Quotegraph**: Young speakers mentioning young targets have HIGHEST tone (~56). Old × old = LOWEST (~42–44).
- **Assessment**: **Contradicts the speaker-level prediction.** Older speakers have lower, not higher, tone. Massive **occupation confound**: young = sportspeople (victory language); old = politicians (conflict language). Literature studied personal writing/social media; Quotegraph captures public discourse.

### fig_age_core (f): First-person plural "we"
- **Literature**: Increases with age (Pennebaker & Stone 2003; Seider et al. 2009)
- **Quotegraph**: Middle-aged (40–55) use the most "we" (~2.4). Young (15–24) use less (~1.2–1.5). Very old (70–74) also use less.
- **Assessment**: **Partially consistent — inverted U.** Increases from youth to middle age (consistent) then plateaus/declines. Peak at 45–55 likely reflects the politician cohort.

---

## 3. PageRank (Prominence) Findings

### fig_pagerank_core (a): Tone
- **Literature**: High status possibly more emotionally restrained
- **Quotegraph**: Low-prominence × low-prominence = positive (~50–52). High-prominence × high-prominence = very negative (~35–40).
- **Assessment**: **Consistent.** One of the cleanest patterns: prominence correlates with negativity. The most central nodes are politicians in adversarial discourse.

### fig_pagerank_core (b): Power
- **Literature**: High-status speakers may use more power words
- **Quotegraph**: TARGET-driven: mentioning very high-prominence targets (-2.8 to -3.4) = ~5 power words. Low-prominence targets = ~2. Speaker prominence has minimal effect.
- **Assessment**: **Target dominance.** Power language reflects WHOM you're talking about, not who is speaking. Most prominent targets (presidents, heads of state) naturally elicit power vocabulary.

### fig_pagerank_core (c): First-person singular "I"
- **Literature**: High status = fewer "I" words (Kacewicz et al. 2014; Pennebaker 2011)
- **Quotegraph**: Inverted U — medium-prominence speakers (-4.2 to -4.8) use the MOST "I" (~3.5). Most prominent (-3.7) and least prominent (-6.4) use less (~2.5–3). Very high-prominence targets (-2.8) dramatically reduce "I" (~1.5).
- **Assessment**: **Partially consistent.** Most prominent speakers DO use less "I" than medium speakers, per Kacewicz. Non-linear: least-prominent also use less (rarely get personal when quoted). Novel TARGET effect: mentioning very prominent people suppresses self-reference.

### fig_pagerank_core (d): Clout
- **Literature**: High status = high Clout (by construction in LIWC; Boyd et al. 2022)
- **Quotegraph**: Low-prominence speakers mentioning low-prominence targets show HIGHEST Clout (~60+). Medium-high prominence = LOWEST (~50–52).
- **Assessment**: **Counterintuitive.** Highest Clout from low-prominence speakers. Possible selection bias: low-prominence speakers only get quoted when making authoritative/confident statements. Alternatively, Clout algorithm (optimized on lab/email) may not transfer cleanly to news quotations.

### fig_pagerank_core (e): First-person plural "we"
- **Literature**: High status = more "we" (Kacewicz et al. 2014)
- **Quotegraph**: Medium-prominence speakers (-4.8 to -5.1) mentioning low-prominence targets = highest "we" (~2.0–2.4). Most prominent (-3.7) use LESS (~1.25).
- **Assessment**: **Contradicts for the most prominent.** The very top of the hierarchy uses less collective language. May reflect that the most prominent speakers (heads of state) speak from personal authority ("I think…"), while mid-prominence politicians use more collective framing.

### fig_pagerank_core (f): Analytic thinking
- **Literature**: High status = higher Analytic
- **Quotegraph**: U-shaped by TARGET: mentioning very low- and very high-prominence targets = high Analytic (~55–60). Medium targets = low (~45). Speaker effect weaker.
- **Assessment**: **Complex, target-driven.** Not a simple linear increase. Prominence extremes elicit more analytical language; the middle band is more narrative/personal.

---

## 4. Age Target Effects (heatmaps) — `fig_age_interactions`

> These panels foreground target-driven and interaction effects that are invisible in speaker-only analyses. Quotegraph's directed structure enables decomposing language into source (speaker) and target (mentioned person) contributions.

### fig_age_interactions (a): Power × Age
- **Quotegraph**: Pure vertical gradient (target-dominated). Mentioning 65–74yr targets → ~4–5 power; 15–24yr targets → ~1–2. Speaker age has near-zero effect.
- **Literature**: Brown & Levinson (1987) politeness theory predicts that perceived power of the addressee/topic increases the weight of face-threatening acts and formal/deferential language. Kacewicz et al. (2014) showed pronoun patterns reflect status hierarchies. However, these studies measured power as a *speaker* trait. No prior LIWC study has decomposed power language by *target* properties.
- **Assessment**: **Novel target-dominance finding.** Power vocabulary in quotations tracks whom you mention, not who you are. Older targets — who in Quotegraph are disproportionately political leaders — naturally elicit references to authority, governance, and control. This cannot be explained by speaker accommodation alone; it reflects the *topic's inherent power valence*. Partial occupation confound: older targets are more likely to hold political office, so the age gradient partly reflects role, not age per se.

### fig_age_interactions (b): Conflict × Age
- **Quotegraph**: Strong target effect (older targets → more conflict) plus a bright diagonal block at source 25–39 × target 25–39 suggesting same-age conflict.
- **Literature**: In Brown & Levinson's (1987) framework, face-threatening acts carry more weight at greater social distance — but *low* distance (as between age peers) enables more direct confrontation. Generational cohort research finds that people of similar age compete for overlapping social/professional positions. No prior LIWC study has examined age-peer conflict in public discourse at scale.
- **Assessment**: **Novel interaction effect.** The diagonal block hints at **age-homophilous adversarialism** — same-age pairs engage in more conflictual discourse, likely reflecting within-cohort political/professional rivalry. The target gradient (older targets = more conflict) parallels the power finding and is partly occupation-confounded.

### fig_age_interactions (c): Anger × Age
- **Quotegraph**: Clearly target-dominated. Mentioning 65–74yr targets → ~0.2 anger; 15–19yr targets → ~0.07. ~3× range across the target axis.
- **Literature**: Pennebaker & Stone (2003) found older adults use *fewer* negative emotion words (speaker effect). Socioemotional selectivity theory (Carstensen 2006) predicts that older adults regulate emotions more effectively, showing a "positivity effect." But these are speaker-level predictions. No prior study examined whether *being mentioned* (as a target) elicits anger from others.
- **Assessment**: **Novel and potentially the most striking target effect.** The anger directed at older targets — regardless of speaker age — is dramatically higher. This likely reflects the political context: older public figures are disproportionately targets of criticism and outrage in news quotations. The finding inverts the speaker-level prediction: older speakers may be calmer (per Carstensen), but being old makes you a target of others' anger.

### fig_age_interactions (d): Authentic × Age
- **Quotegraph**: Strong speaker effect (young ≈ 45 Authentic, old ≈ 35) with target modulation — mentioning similarly-aged targets produces subtly higher Authenticity (faint diagonal brightening).
- **Literature**: Pennebaker & Stone (2003) found increasing cognitive complexity with age, which could reduce Authenticity scores (Authenticity inversely correlates with self-monitoring). The same-age pattern echoes the gender finding (same-gender = more Authentic) and aligns with Communication Accommodation Theory (Giles et al. 1991): speakers accommodate less — and therefore appear more "authentic" — with in-group members.
- **Assessment**: **Speaker effect consistent; target modulation is novel.** The age-similarity → Authenticity pattern mirrors the same-gender finding and extends it to a continuous dimension. Speakers are more guarded (lower Authenticity) when the target is dissimilar in age.

### fig_age_interactions (e): Affiliation × Age
- **Quotegraph**: Mixed pattern. Brightest region: older speakers (45–74) × younger targets (15–29). Older speakers × older targets = lowest affiliation.
- **Literature**: Intergenerational communication research documents "elderspeak" and overaccommodation — including increased use of collective "we" and affiliative language — by younger speakers toward older adults (Williams et al. 1997; Giles et al. 1992). Quotegraph shows the *reverse* pattern: it is **older speakers mentioning younger targets** who use the most affiliation language.
- **Assessment**: **Different context produces opposite directionality to elderspeak.** In Quotegraph's public discourse context, older speakers (politicians, leaders) use affiliative language when mentioning younger people — a "paternalistic solidarity" or mentoring register. Elderspeak research documents the reverse (younger → older) in clinical/care settings. The two findings are not contradictory but context-dependent: in care settings, the young accommodate downward toward perceived frailty; in public discourse, the old accommodate downward from positions of authority toward perceived juniors. Both reflect top-down accommodation — but "top" is defined by physical capability in one context and institutional authority in the other.

### fig_age_interactions (f): Tone × Age
- **Quotegraph**: Diagonal gradient: young speaker × young target = most positive (~56); old × old = most negative (~42). Both speaker and target age matter.
- **Literature**: Socioemotional selectivity theory (Carstensen & Mikels 2005; Pennebaker & Stone 2003) predicts a "positivity effect" in aging — older adults focus on positive information. The heatmap contradicts this at the speaker level (young = more positive) and reveals a strong target contribution: older targets are mentioned in more negative contexts regardless of speaker age.
- **Assessment**: **Contradicts SST for speaker-level; reveals target contribution.** The contradiction is likely occupation-confounded: young targets are disproportionately sportspeople (positive, victory-language contexts), while older targets are disproportionately politicians (negative, adversarial contexts). The diagonal structure shows that *both* speaker and target age contribute, ruling out a purely target-driven explanation.

---

## 5. Age Difference Interactions — `fig_age_diffs`

> The age difference plots (X = Source age − Target age) are the **central novel contribution** for the age dimension. They directly isolate interaction effects that cannot be recovered from speaker-only analysis. Negative values = speaker younger than target; positive = speaker older.

### fig_age_diffs (a): Conflict × Age Difference — **Inverted U (peer conflict)**
- **Pattern**: Peak conflict at diff ≈ 0 (~0.45). Falls to ~0.28 at large negative gaps and ~0.20 at large positive gaps.
- **Literature**: In Brown & Levinson's (1987) framework, the weight of face-threatening acts is modulated by social distance (D) — lower distance enables more direct confrontation. Age peers have minimal social distance, potentially licensing more explicit conflict language. Eckert (2003) documented how language marks same-age peer group boundaries and within-group competition in adolescent communities. No prior LIWC study has demonstrated a conflict peak at zero age difference in adult public discourse.
- **Assessment**: **Novel at scale; consistent with face theory's distance variable.** Same-age public figures are more likely to be direct political rivals (e.g., competing politicians of the same generation), producing maximally confrontational quotations. Large age gaps introduce deference/formality that suppresses overt conflict language. This is a **genuine interaction effect** — neither speaker age nor target age alone produces an inverted-U.

### fig_age_diffs (b): Tone × Age Difference — **U-shaped (minimum near peers)**
- **Pattern**: Minimum Tone at diff ≈ −10 to −15 (~47). Both extremes are more positive, especially old→young (diff +30 to +40, reaching ~53–55).
- **Literature**: The positivity effect in aging (Carstensen & Mikels 2005) predicts older adults focus on positive information. Tone × age diff reveals a *directional* version: older speakers mentioning younger targets are most positive, consistent with a benevolent/mentoring framing. The minimum near −10 (young speakers mentioning somewhat older targets) aligns with the high conflict zone — younger political figures criticizing established older ones.
- **Assessment**: **Extends socioemotional selectivity theory to directed discourse.** The age-asymmetric pattern (old→young = positive, young→old = negative) adds directionality to the known "positivity of aging" effect. The dip near zero/slightly negative confirms the peer-conflict zone.

### fig_age_diffs (c): Clout × Age Difference — **Monotonic S-curve (deference gradient)**
- **Pattern**: Old→young (positive diff) = Clout ~59–60; young→old (negative diff) = ~50–51. ~9-point swing.
- **Literature**: Kacewicz et al. (2014) found Clout increases with status. CAT (Giles et al. 1991) predicts convergence/divergence based on interlocutor status — younger speakers accommodate upward (deference), older speakers accommodate downward (authority). Brown & Levinson (1987): higher perceived power of the hearer → more politeness/deference → lower Clout. Intergenerational communication research (Giles et al. 1992) documents that younger speakers adopt more deferential registers with older interlocutors.
- **Assessment**: **Consistent with CAT and politeness theory, extended to population-scale public discourse.** The monotonic gradient confirms that Clout is not just a speaker trait but is modulated by the speaker–target age gap. Note that CAT describes accommodation in conversation *with* someone, whereas Quotegraph captures discourse *about* someone — this is a topic-driven register shift rather than interlocutor accommodation. Nevertheless, the pattern (confident when discussing juniors, deferential when discussing seniors) parallels CAT predictions closely.

### fig_age_diffs (d): Power × Age Difference — **Monotonic decreasing (looking-up = power)**
- **Pattern**: Young→old (negative diff) = highest power (~2.7); old→young (positive diff) = lowest (~1.5).
- **Literature**: This is the **mirror image** of Clout. Where Clout reflects the *speaker's* self-positioning (confident when addressing juniors), power reflects the *target's* position (mentioning seniors = power vocabulary). Brown & Levinson (1987) predicted that face-threat weightiness increases with addressee power. Kacewicz et al. (2014) focused on speaker status → pronoun use but did not decompose by target.
- **Assessment**: **Novel decomposition.** Clout and power move in opposite directions across the age-gap axis, confirming they track different loci: Clout = speaker self-positioning, power = target characterization. This Clout–power split is a key finding that validates Quotegraph's directional structure.

### fig_age_diffs (e): Affiliation × Age Difference — **Monotonic increasing (looking-down = affiliative)**
- **Pattern**: Old→young (positive diff) = affiliation ~3.2; young→old (negative diff) = ~2.5.
- **Literature**: Elderspeak research (Williams et al. 1997) found that speakers use patronizing but *affiliative* language toward older adults. Quotegraph reverses the direction: **older speakers are more affiliative toward younger targets.** This aligns with the "mentoring register" documented in workplace communication (Kram 1985) where senior figures adopt inclusive language when addressing juniors.
- **Assessment**: **Reverses the elderspeak direction; consistent with mentoring register.** In public discourse, it is the *older* speakers who adopt togetherness language toward the young, not vice versa. Combined with high Clout and high "we," this forms a coherent "paternalistic solidarity" profile: confident, collective, and affiliative when addressing younger targets.

### fig_age_diffs (f): "We" × Age Difference — **Monotonic S-curve (collective when looking down)**
- **Pattern**: Old→young (positive diff) = ~2.0–2.2; young→old (negative diff) = ~1.3. Sharp inflection at diff ≈ 0.
- **Literature**: Seider et al. (2009) found "we" increases with age in couples, reflecting shared identity. Kacewicz et al. (2014) found "we" increases with status. Pennebaker & Stone (2003) found "we" increases with age (speaker effect). But none examined *directional* "we" use across an age gap. Intergenerational research (Giles et al. 1992) noted that collective "we" is a feature of elderspeak used *toward* older adults.
- **Assessment**: **Extends and clarifies the age–"we" relationship.** The increase is not purely a speaker-age effect — it is modulated by the gap. Older speakers mentioning younger targets maximize "we," forming a "we includes you" framing. Younger speakers mentioning older targets minimize "we," maintaining individual framing ("I"). This directional pattern is consistent with the mentoring/paternalistic profile above.

---

## 6. PageRank Difference Interactions — `fig_pagerank_diffs`

> The PageRank difference plots (X = log₁₀Source − log₁₀Target) are the **central novel contribution** for the prominence dimension. They reveal "punching up" vs. "punching down" asymmetries that are unique to directed network analysis. Negative X = source less prominent ("punching up"); positive X = source more prominent ("punching down").

### fig_pagerank_diffs (a): Power × PageRank Difference — **V-shaped (prominence gap drives power)**
- **Pattern**: Minimum power at diff ≈ 0 (~2.0). Left side (source less prominent, "punching up") rises steeply to ~5+. Right side (source more prominent) rises to ~4+.
- **Literature**: Kacewicz et al. (2014) measured status as a speaker trait. Brown & Levinson (1987) predicted that perceived power of the hearer increases the formality and indirectness of communication. No prior LIWC study measured how the *prominence gap* between speaker and target shapes power vocabulary.
- **Assessment**: **Novel V-shape, with asymmetric "punching up" effect.** Any prominence gap elicits power language, but mentioning someone more prominent produces the most. This is consistent with politeness theory's weight function (W = D + P + R): the power variable P increases the weight of what is said about high-status targets. The asymmetry shows that "upward-directed" discourse is linguistically distinct from "downward-directed" discourse — a finding only possible with directed network data.

### fig_pagerank_diffs (b): Conflict × PageRank Difference — **Steep "punching up" gradient**
- **Pattern**: Source << target (diff ≈ −5) = ~0.75. Peer (diff ≈ 0) = ~0.37. Source >> target (diff ≈ +3) = mild rise to ~0.43.
- **Literature**: Critical discourse analysis (van Dijk 1993) examines how power asymmetries structure discourse — elite actors are both producers and targets of discursive power. Media studies show that public criticism is disproportionately directed at high-status figures, as they are the consequential decision-makers whose actions merit scrutiny. CAT (Giles et al. 1991) predicts that speakers may diverge from higher-status interlocutors to assert distinctiveness, though here the effect is discourse *about* rather than *with* the target.
- **Assessment**: **Strongly consistent with upward-directed criticism.** The asymmetry (~2× more conflict when punching up vs. down) quantifies what media scholars have observed qualitatively. Less-prominent speakers mentioning more-prominent targets produce dramatically more adversarial language. This is the PageRank analogue of the target-driven conflict effect seen in gender (mentioning men = more conflict) and age (mentioning older targets = more conflict).

### fig_pagerank_diffs (c): Tone × PageRank Difference — **Inverted U (peers are most positive)**
- **Pattern**: Peak at diff ≈ 0 (~50). Falls to ~35 (punching up) and ~43 (punching down).
- **Literature**: Socioemotional selectivity theory has no direct analogue for prominence. However, social identity theory (Tajfel & Turner 1979) predicts that in-group (same-status peer) interactions are more positive than out-group (cross-status) interactions. Language style matching research (Gonzales et al. 2010) found that linguistic similarity (which peaks among peers) predicts group cohesion.
- **Assessment**: **Novel; consistent with in-group favoritism and peer solidarity.** Same-prominence peers produce the most positive discourse. The asymmetry (punching up = far more negative than punching down) reinforces the conflict finding. This pattern cannot be explained by a simple "high prominence = negative" story — it is the *gap* that matters.

### fig_pagerank_diffs (d): Clout × PageRank Difference — **Asymmetric U (gap = confidence)**
- **Pattern**: Minimum at diff ≈ −2 (~53). Source >> target (diff ≈ +3) = highest Clout (~60). Source << target (diff ≈ −5) ≈ 54.
- **Literature**: Kacewicz et al. (2014) found Clout increases with status. The PageRank difference plot *extends* this: Clout reflects not just absolute status but *relative* status. The minimum at diff ≈ −2 (not at 0) suggests speakers are least confident when they are moderately below their target in prominence — a "slightly outranked" effect.
- **Assessment**: **Resolves the earlier "counterintuitive" finding.** The Clout × PageRank heatmap (Section 3) seemed to show low-prominence speakers having high Clout, which was puzzling. The difference plot clarifies: what matters is the *gap*. Low-prominence speakers mentioning similarly low-prominence targets have a near-zero gap → they feel relatively confident. The finding confirms Kacewicz et al. directionally but adds the crucial moderating role of target prominence.

### fig_pagerank_diffs (e): "I" × PageRank Difference — **Inverted U (self-reference is for peers)**
- **Pattern**: Peak at diff ≈ −1 to 0 (~3.4). Falls to ~1.5 (punching up, diff ≈ −5) and ~1.5 (punching down, diff ≈ +3).
- **Literature**: Kacewicz et al. (2014): high-status speakers use fewer "I" words. Pennebaker (2011): "I" reflects self-focus and lower status. The difference plot extends this to *relational* status: "I" peaks when speaker ≈ target in prominence, and is suppressed whenever hierarchy crossing occurs. The stronger suppression when punching up aligns with Kacewicz's finding that low status relative to your interlocutor increases self-focus — but in Quotegraph, mentioning a much more prominent person *suppresses* "I" (the quotation becomes about the target, not the self).
- **Assessment**: **Partially contradicts, partially extends Kacewicz.** Kacewicz predicted low-status individuals use more "I," but Quotegraph shows that when the gap is very large (punching up), "I" actually drops — the speaker self-effaces in deference to the prominent target. The inverted-U shape shows that moderate status equality maximizes self-expression.

### fig_pagerank_diffs (f): "We" × PageRank Difference — **Inverted U (collective language for peers)**
- **Pattern**: Peak at diff ≈ 0 (~1.7–1.8). Falls to ~1.1–1.2 at both extremes.
- **Literature**: Kacewicz et al. (2014): "we" increases with status. Seider et al. (2009): "we" reflects shared identity. The PageRank difference plot shows that "we" actually peaks at *peer-level* prominence and is *suppressed* at both ends of the hierarchy gap. This contradicts the simple "high status → more we" prediction and aligns with a solidarity/in-group interpretation.
- **Assessment**: **Contradicts Kacewicz for large gaps; supports solidarity interpretation.** "We" is a marker of perceived peer solidarity, not unilateral high status. Large prominence gaps (whether up or down) suppress collective framing. This finding resolves the earlier paradox where the most prominent speakers in PageRank heatmaps used less "we" — they were disproportionately discussing people far below them in prominence.

---

## 7. Cross-Cutting Themes

### Theme 1: "You are whom you mention" — target effects dominate

Target properties (gender, age, prominence) shape quotation language as much or more than speaker properties. Power, conflict, anger, and Tone are all target-dominated across age and PageRank.

**Literature context**: All major LIWC studies (Pennebaker & Stone 2003; Newman et al. 2008; Kacewicz et al. 2014; Luoto 2021) analyzed language as a function of *speaker* properties alone. This was an inherent limitation of their data (essays, social media posts, emails) which had no directed structure. Quotegraph is the first large-scale LIWC study with a directed speaker→mention network, enabling the target decomposition. The finding that language is shaped by whom you discuss — not just who you are — fundamentally extends the LIWC literature.

### Theme 2: Three interaction archetypes

- **Peer discourse** (diff ≈ 0): personal (high I), collective (high we), positive (high Tone), conflictual (high conflict in age, NOT in PageRank).
- **"Punching up"** (mentioning older/more prominent): deferential (low Clout), power-laden, negative Tone, high conflict, low self-reference.
- **"Punching down"** (mentioning younger/less prominent): confident (high Clout), affiliative, low power, positive Tone, collective (high we in age).

**Literature context**: Communication Accommodation Theory (Giles et al. 1991) predicts convergence (toward similar others) and divergence (from dissimilar others) in dyadic interaction. **Important caveat**: CAT was developed for face-to-face conversation *with* someone, whereas Quotegraph captures public discourse *about* someone. The patterns we observe are therefore not conversational accommodation but *topic-driven register shifts* — a speaker's language changes based on whom they are discussing, not whom they are addressing. Nevertheless, the parallels are striking: peer discourse is "convergent" (personal, solidary); hierarchy-crossing is "divergent" (formal, impersonal), with the gap direction determining deference (upward) vs. authority (downward). Brown & Levinson's (1987) face theory, which is framed more generally around the perceived power/distance of the referent (not just the addressee), provides a better theoretical fit: face-threat weight W = D + P + R, where P (power of referent) and D (social distance) vary systematically across the gap axis.

### Theme 3: Age-peer vs. prominence-peer conflict divergence

Same-age peers are maximally conflictual (inverted U in conflict × age diff). Same-prominence peers are minimally conflictual (U-shape in conflict × PageRank diff).

**Literature context**: Social identity theory (Tajfel & Turner 1979) predicts in-group favoritism, but also acknowledges within-group competition when individuals contest the same resources. Age peers in public life (e.g., politicians of the same generation) compete for overlapping positions and media attention — hence within-cohort rivalry. Prominence peers, by contrast, may be coalition partners operating at a similar level in different domains rather than direct rivals. This age vs. prominence divergence in peer-conflict direction is a novel contribution with no direct literature precedent. It illustrates how the *type* of similarity (demographic vs. structural) shapes whether peer interactions are cooperative or adversarial.

### Theme 4: The Clout–Power complementarity

"Looking up" and "looking down" produce distinct linguistic signatures that are mirror images: Clout ↔ Power, Affiliation ↔ Anger, We ↔ I. The complementarity is visible in both age and PageRank differences.

**Literature context**: Kacewicz et al. (2014) found that Clout (high status speaker trait) and first-person singular (low status speaker trait) are inversely related. Quotegraph extends this to a *two-dimensional* decomposition: Clout tracks the speaker's self-positioning relative to the target, while Power tracks the target's attributed position. These two dimensions are formally orthogonal but empirically anti-correlated across the gap axis — confirming that they capture complementary aspects of hierarchical discourse.

### Theme 5: Occupation confound in age/prominence

Young targets are disproportionately sportspeople (positive, personal, past-tense). Older/prominent targets are disproportionately politicians (negative, formal, power-laden). Many age and PageRank patterns likely reflect underlying occupation composition. This confound does not invalidate the findings — it *is* the mechanism through which age and prominence affect public discourse — but it means the effects should be interpreted as "age-as-socially-constituted" rather than as pure age or prominence effects.

---

## Summary Table

| Category | Dim. | Literature prediction | Quotegraph result | Verdict |
|----------|------|----------------------|-------------------|---------|
| *I* | Gender | F >> M | FF >> FM > MM > MF | **Strongly consistent** |
| *I* | Age | Decreases with age | Clear monotonic decrease | **Strongly consistent** |
| *I* | PageRank | High status → fewer *I* | Inverted U; top uses less | **Partially consistent** |
| *I* | PR diff | Low status → more *I* (Kacewicz) | Inverted U: peak at peer, suppressed at large gaps | **Partially contradicts** |
| Analytic | Gender | M > F | MF > FM > MM > FF | **Consistent** |
| Analytic | Age | Increases with age | Target-dominated increase | **Consistent** |
| Authentic | Gender | F > M | Same-gender > cross-gender | **Novel pattern** |
| Authentic | Age | Decreases with age (cognitive complexity) | Speaker-driven decrease; age-similarity diagonal | **Consistent + novel target modulation** |
| Clout | Gender | M >= F | M > F (small) | **Consistent** |
| Clout | Age | Increases with age | +16 pts youngest→oldest | **Strongly consistent** |
| Clout | Age diff | Status → Clout (Kacewicz) | Monotonic S-curve: 51→60 across gap | **Extends to relational status** |
| Clout | PageRank | High status → high Clout | Low-prominence highest | **Counterintuitive** |
| Clout | PR diff | Status → Clout (Kacewicz) | Asymmetric U: min at diff ≈ −2 | **Resolves heatmap paradox** |
| Tone | Gender | F more positive | FF highest, FM lowest | **Partially consistent** |
| Tone | Age | Increases with age (SST) | Decreases with age | **Contradicts** (confound) |
| Tone | Age diff | Positivity of aging (SST) | U-shaped: old→young most positive | **Extends SST directionally** |
| Tone | PageRank | High status restrained | Prominence → negativity | **Consistent** |
| Tone | PR diff | — | Inverted U: peers most positive | **Novel** |
| Anger | Gender | M > F | F > M | **Contradicts** |
| Anger | Age | Older speakers calmer (SST) | Target-driven: older targets ~3× more anger | **Novel: inverts SST for targets** |
| Anxiety | Gender | F > M | F > M | **Consistent** |
| Sadness | Gender | F > M | F >> M | **Consistent** |
| Power | Gender | M > F | Mention-driven (M-target) | **Novel: target effect** |
| Power | Age | — | Pure vertical gradient (target age) | **Novel: target dominance** |
| Power | Age diff | — | Monotonic ↘: 2.7→1.5 | **Novel: mirror of Clout** |
| Power | PR diff | — | V-shaped: gap drives power | **Novel: first LIWC prominence-gap finding** |
| Conflict | Gender | M > F | Mention-driven (M-target) | **Novel: target effect** |
| Conflict | Age diff | Low distance → direct FTA (B&L) | Inverted U: peak at diff ≈ 0 | **Novel: consistent with face theory** |
| Conflict | PR diff | Upward criticism (van Dijk) | Steep ↘: punching up = 2× conflict | **Strongly consistent** |
| Affiliation | Age | Elderspeak: young→old affiliative | Reversed: old→young affiliative | **Reverses direction** (context-dependent) |
| Affiliation | Age diff | Elderspeak direction | Monotonic ↗: 2.5→3.2 | **Mentoring register** |
| Past focus | Age | Decreases with age | Decreases with age | **Consistent** |
| *We* | Age | Increases with age | Inverted U, peak at 45–55 | **Partially consistent** |
| *We* | Age diff | Increases with age/status | Monotonic ↗: 1.3→2.2 | **Extends to directed gap** |
| *We* | PageRank | High status → more *we* | Top uses less | **Contradicts** |
| *We* | PR diff | High status → more *we* (Kacewicz) | Inverted U: peers highest | **Contradicts; supports solidarity** |

### Robust replications (6)
*I* × age, *I* × gender, Clout × age, past focus × age, anxiety × gender, sadness × gender

### Notable contradictions (5)
Anger × gender, Tone × age, Clout × PageRank, *we* × PageRank, *we* × PR diff (Kacewicz)

### Novel findings requiring directed network (12)
1. Power × age: pure target dominance
2. Power × age diff: mirror of Clout (speaker vs. target loci)
3. Power × PR diff: V-shaped prominence-gap effect
4. Conflict × age diff: inverted-U peer conflict
5. Conflict × PR diff: "punching up" asymmetry
6. Tone × PR diff: inverted-U peer positivity
7. Clout × age diff: relational deference gradient
8. Clout × PR diff: resolves heatmap paradox
9. Anger × age: target-driven (inverts SST)
10. Affiliation × age diff: reverses elderspeak direction
11. *I* × PR diff: self-effacement at large gaps
12. *We* × PR diff: solidarity, not status

### Prior replications extended to interactions (4)
Authentic same-gender → extended to age-similarity diagonal; Tone × age diff extends SST directionally; *We* × age diff extends speaker-level finding; Clout × age diff extends Kacewicz to relational status
