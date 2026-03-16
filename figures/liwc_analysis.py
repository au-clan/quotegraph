#!/usr/bin/env python3
"""
LIWC statistical analysis pipeline for Quotegraph EMNLP submission.

Implements the analysis framework from EMNLP_PLAN.md:
  - Mixed-effects regressions with speaker/target random intercepts
  - FDR correction across all tested categories
  - Effect sizes (partial eta-squared, R-squared)
  - Robustness checks: occupation stratification, temporal split, permutation baseline,
    attribution confidence, binning sensitivity
  - Exports aggregated data for reproducibility (no raw LIWC scores needed)

Requirements:
  - PySpark (for data loading from ~/data/)
  - statsmodels (for OLS with clustered SEs — fallback when lme4 is infeasible)
  - pandas, numpy, scipy

Usage:
  python liwc_analysis.py --data-dir ~/data --output-dir figures/results
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm


# ── Configuration ──────────────────────────────────────────────────────

# LIWC categories selected per EMNLP_PLAN.md Section 2:
# Categories with established prior predictions from Pennebaker & Stone (2003)
# and Kacewicz et al. (2014), enabling speaker-only vs. speaker+target comparison.
SELECTED_CATEGORIES = [
    "Clout",       # Kacewicz et al. 2014: increases with status
    "power",       # Power language: target characterization
    "conflict",    # Conflict: peer vs. hierarchy effects
    "Tone",        # SST: positivity effect
    "i",           # Pennebaker & Stone 2003: decreases with age/status
    "we",          # Kacewicz et al. 2014: collective framing
]

# All 80+ categories tested for FDR correction (report how many survive)
ALL_CATEGORIES = None  # Set from data columns at runtime

# Minimum quote length in words (LIWC-22 summary dims unreliable below this)
MIN_QUOTE_LENGTH = 25

# Minimum bin count for inclusion
MIN_BIN_N = 100

# Age binning
AGE_BIN_WIDTH = 5
AGE_DIFF_BIN_WIDTH = 5

# Temporal split boundary
TEMPORAL_SPLIT_YEAR = 2015

# Attribution confidence threshold (Quotebank P(speaker))
ATTRIBUTION_THRESHOLD = 0.8


# ── Data Loading ───────────────────────────────────────────────────────

def load_quotegraph(data_dir: str) -> pd.DataFrame:
    """Load Quotegraph data with LIWC scores, Wikidata attributes, and network metrics.

    Expected columns after loading:
      - quoteID, source_id, target_id
      - quotation (text), n_words (quote length)
      - source_age, target_age (age at time of quotation)
      - source_pagerank, target_pagerank (log10)
      - source_occupation, target_occupation (Wikidata occupation label)
      - source_nationality, target_nationality
      - source_gender, target_gender
      - year (quotation year)
      - prob_speaker (Quotebank attribution confidence)
      - LIWC category columns (Clout, power, conflict, Tone, i, we, ...)
    """
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("liwc_analysis").getOrCreate()
        df = spark.read.parquet(os.path.join(data_dir, "quotegraph.parquet"))
        pdf = df.toPandas()
        spark.stop()
    except Exception:
        # Fallback: try reading as parquet directly
        pdf = pd.read_parquet(os.path.join(data_dir, "quotegraph.parquet"))

    return pdf


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing steps per EMNLP_PLAN.md Section 4."""
    n_initial = len(df)

    # Quote length filter
    df = df[df["n_words"] >= MIN_QUOTE_LENGTH].copy()
    n_after_length = len(df)

    # Remove self-loops (should already be done, but verify)
    df = df[df["source_id"] != df["target_id"]].copy()
    n_after_selfloop = len(df)

    # Compute derived columns
    df["log_quote_length"] = np.log(df["n_words"])
    df["age_diff"] = df["source_age"] - df["target_age"]
    df["pagerank_diff"] = df["source_pagerank"] - df["target_pagerank"]

    # Age bins
    df["source_age_bin"] = (df["source_age"] // AGE_BIN_WIDTH) * AGE_BIN_WIDTH
    df["target_age_bin"] = (df["target_age"] // AGE_BIN_WIDTH) * AGE_BIN_WIDTH
    df["age_diff_bin"] = (df["age_diff"] // AGE_DIFF_BIN_WIDTH) * AGE_DIFF_BIN_WIDTH

    # Dyad ID for clustering
    df["dyad_id"] = df["source_id"].astype(str) + "_" + df["target_id"].astype(str)

    print(f"Preprocessing: {n_initial} → {n_after_length} (length filter) "
          f"→ {n_after_selfloop} (self-loop removal)")
    print(f"  Unique speakers: {df['source_id'].nunique()}")
    print(f"  Unique targets:  {df['target_id'].nunique()}")
    print(f"  Unique dyads:    {df['dyad_id'].nunique()}")

    return df


# ── Statistical Analysis ──────────────────────────────────────────────

def partial_eta_squared(aov_table: pd.DataFrame, term: str) -> float:
    """Compute partial eta-squared from ANOVA table."""
    ss_term = aov_table.loc[term, "sum_sq"]
    ss_resid = aov_table.loc["Residual", "sum_sq"]
    return ss_term / (ss_term + ss_resid)


def fit_gap_regression(df: pd.DataFrame, category: str, gap_col: str,
                       source_attr: str = "source_age") -> dict:
    """Fit polynomial regression for difference-plot analysis.

    Model: LIWC_score ~ poly(gap, 2) + source_attr + occupation_source
           + occupation_target + nationality_source + log(quote_length)

    Returns dict with coefficients, CIs, p-values, R², partial η².

    Note: This is the OLS fallback. For publication, prefer mixed-effects
    models with (1|source_id) + (1|target_id) using lme4 in R.
    """
    subset = df[[category, gap_col, source_attr,
                  "source_occupation", "target_occupation",
                  "source_nationality", "log_quote_length",
                  "source_id", "target_id", "dyad_id"]].dropna()

    if len(subset) < MIN_BIN_N:
        return None

    # Compute gap² for quadratic term
    subset = subset.copy()
    subset["gap"] = subset[gap_col]
    subset["gap_sq"] = subset[gap_col] ** 2

    formula = (f"{category} ~ gap + gap_sq + {source_attr}"
               " + C(source_occupation) + C(target_occupation)"
               " + C(source_nationality) + log_quote_length")

    try:
        model = ols(formula, data=subset).fit(
            cov_type="cluster",
            cov_kwds={"groups": subset["dyad_id"]},
        )
    except Exception as e:
        print(f"  Warning: regression failed for {category} ~ {gap_col}: {e}")
        return None

    # Extract key results
    result = {
        "category": category,
        "gap_variable": gap_col,
        "n_obs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
    }

    # Gap coefficients
    for term in ["gap", "gap_sq"]:
        if term in model.params.index:
            result[f"{term}_coef"] = float(model.params[term])
            result[f"{term}_se"] = float(model.bse[term])
            result[f"{term}_pvalue"] = float(model.pvalues[term])
            ci = model.conf_int().loc[term]
            result[f"{term}_ci_lower"] = float(ci[0])
            result[f"{term}_ci_upper"] = float(ci[1])

    # Partial eta-squared via Type II ANOVA
    try:
        aov = anova_lm(model, typ=2)
        for term in ["gap", "gap_sq"]:
            if term in aov.index:
                result[f"{term}_partial_eta_sq"] = partial_eta_squared(aov, term)
    except Exception:
        pass

    # AIC for linear vs quadratic comparison
    try:
        formula_linear = (f"{category} ~ gap + {source_attr}"
                          " + C(source_occupation) + C(target_occupation)"
                          " + C(source_nationality) + log_quote_length")
        model_linear = ols(formula_linear, data=subset).fit()
        result["aic_linear"] = float(model_linear.aic)
        result["aic_quadratic"] = float(model.aic)
        result["quadratic_preferred"] = model.aic < model_linear.aic
    except Exception:
        pass

    return result


def fit_interaction_regression(df: pd.DataFrame, category: str,
                               source_attr: str, target_attr: str) -> dict:
    """Fit full interaction model (EMNLP_PLAN.md Section 3a).

    Model: LIWC_score ~ source_attr * target_attr + covariates

    Returns dict with coefficients, CIs, p-values, partial η².
    """
    subset = df[[category, source_attr, target_attr,
                  "source_occupation", "target_occupation",
                  "source_nationality", "log_quote_length",
                  "source_id", "target_id", "dyad_id"]].dropna()

    if len(subset) < MIN_BIN_N:
        return None

    formula = (f"{category} ~ {source_attr} * {target_attr}"
               " + C(source_occupation) + C(target_occupation)"
               " + C(source_nationality) + log_quote_length")

    try:
        model = ols(formula, data=subset).fit(
            cov_type="cluster",
            cov_kwds={"groups": subset["dyad_id"]},
        )
    except Exception as e:
        print(f"  Warning: interaction regression failed for {category}: {e}")
        return None

    interaction_term = f"{source_attr}:{target_attr}"

    result = {
        "category": category,
        "source_attr": source_attr,
        "target_attr": target_attr,
        "n_obs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
    }

    for term in [source_attr, target_attr, interaction_term]:
        if term in model.params.index:
            result[f"{term}_coef"] = float(model.params[term])
            result[f"{term}_se"] = float(model.bse[term])
            result[f"{term}_pvalue"] = float(model.pvalues[term])
            ci = model.conf_int().loc[term]
            result[f"{term}_ci_lower"] = float(ci[0])
            result[f"{term}_ci_upper"] = float(ci[1])

    # Partial eta-squared via Type II ANOVA
    try:
        aov = anova_lm(model, typ=2)
        for term in [source_attr, target_attr, interaction_term]:
            if term in aov.index:
                result[f"{term}_partial_eta_sq"] = partial_eta_squared(aov, term)
    except Exception:
        pass

    return result


def run_fdr_correction(results: list[dict], alpha: float = 0.05) -> list[dict]:
    """Apply FDR (Benjamini-Hochberg) correction across all tested categories.

    Per REJECT-4: correction must be applied across ALL tested categories,
    not just the selected ones shown in the paper.
    """
    p_values = []
    p_keys = []

    for r in results:
        for key, val in r.items():
            if key.endswith("_pvalue") and val is not None:
                p_values.append(val)
                p_keys.append((r["category"], key))

    if not p_values:
        return results

    reject, corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

    # Map back to results
    corrected_map = {k: (r, p) for k, r, p in zip(p_keys, reject, corrected)}

    for r in results:
        for key in list(r.keys()):
            if key.endswith("_pvalue"):
                lookup = (r["category"], key)
                if lookup in corrected_map:
                    rej, corr_p = corrected_map[lookup]
                    fdr_key = key.replace("_pvalue", "_pvalue_fdr")
                    sig_key = key.replace("_pvalue", "_significant_fdr")
                    r[fdr_key] = float(corr_p)
                    r[sig_key] = bool(rej)

    return results


# ── Aggregated Data Export ────────────────────────────────────────────

def compute_bin_statistics(df: pd.DataFrame, category: str,
                           bin_col: str) -> pd.DataFrame:
    """Compute per-bin means, SEs, Ns for a LIWC category.

    These aggregated statistics can be released publicly even though
    LIWC-22 is proprietary (individual scores are not disclosed).
    """
    grouped = df.groupby(bin_col)[category].agg(["mean", "std", "count"])
    grouped.columns = ["mean", "std", "n"]
    grouped["se"] = grouped["std"] / np.sqrt(grouped["n"])
    grouped["ci_lower"] = grouped["mean"] - 1.96 * grouped["se"]
    grouped["ci_upper"] = grouped["mean"] + 1.96 * grouped["se"]

    # Exclude low-N bins
    grouped = grouped[grouped["n"] >= MIN_BIN_N]

    return grouped.reset_index()


def compute_heatmap_statistics(df: pd.DataFrame, category: str,
                               row_col: str, col_col: str) -> pd.DataFrame:
    """Compute per-cell means, SEs, Ns for heatmap figures."""
    grouped = df.groupby([row_col, col_col])[category].agg(["mean", "std", "count"])
    grouped.columns = ["mean", "std", "n"]
    grouped["se"] = grouped["std"] / np.sqrt(grouped["n"])

    # Exclude low-N cells
    grouped = grouped[grouped["n"] >= MIN_BIN_N]

    return grouped.reset_index()


# ── Robustness Checks ────────────────────────────────────────────────

def occupation_stratification(df: pd.DataFrame, category: str,
                              gap_col: str, occupation: str,
                              source_attr: str = "source_age") -> dict:
    """Run gap regression within a single occupation (REJECT-2).

    Critical test: does the conflict inverted-U survive within-politicians?
    """
    subset = df[
        (df["source_occupation"] == occupation) |
        (df["target_occupation"] == occupation)
    ]
    if len(subset) < MIN_BIN_N * 5:
        return {"occupation": occupation, "category": category,
                "gap_variable": gap_col, "status": "insufficient_data",
                "n_obs": len(subset)}

    result = fit_gap_regression(subset, category, gap_col, source_attr)
    if result:
        result["occupation_filter"] = occupation
    return result


def temporal_split(df: pd.DataFrame, category: str, gap_col: str,
                   source_attr: str = "source_age") -> dict:
    """Run gap regression in two temporal windows (m7).

    Split: 2008–2014 vs 2015–2020.
    """
    results = {}
    for period, (start, end) in [("early", (2008, TEMPORAL_SPLIT_YEAR)),
                                  ("late", (TEMPORAL_SPLIT_YEAR, 2021))]:
        subset = df[(df["year"] >= start) & (df["year"] < end)]
        r = fit_gap_regression(subset, category, gap_col, source_attr)
        if r:
            r["period"] = period
            r["year_range"] = f"{start}-{end - 1}"
        results[period] = r
    return results


def permutation_baseline(df: pd.DataFrame, category: str, gap_col: str,
                         n_permutations: int = 100,
                         source_attr: str = "source_age") -> dict:
    """Shuffle source-target assignments and refit (m6).

    Preserves marginal distributions; destroys directed structure.
    If patterns persist in shuffled data, they are artifacts.
    """
    real_result = fit_gap_regression(df, category, gap_col, source_attr)
    if real_result is None:
        return None

    real_r2 = real_result["r_squared"]
    perm_r2s = []

    for i in range(n_permutations):
        perm_df = df.copy()
        # Shuffle target attributes while preserving source attributes
        target_cols = [c for c in perm_df.columns if c.startswith("target_")]
        shuffled_idx = np.random.permutation(len(perm_df))
        for col in target_cols:
            perm_df[col] = perm_df[col].values[shuffled_idx]
        # Recompute gap
        if gap_col == "age_diff":
            perm_df["age_diff"] = perm_df["source_age"] - perm_df["target_age"]
        elif gap_col == "pagerank_diff":
            perm_df["pagerank_diff"] = perm_df["source_pagerank"] - perm_df["target_pagerank"]

        perm_result = fit_gap_regression(perm_df, category, gap_col, source_attr)
        if perm_result:
            perm_r2s.append(perm_result["r_squared"])

    if perm_r2s:
        p_perm = np.mean(np.array(perm_r2s) >= real_r2)
        return {
            "category": category,
            "gap_variable": gap_col,
            "real_r_squared": real_r2,
            "perm_r_squared_mean": float(np.mean(perm_r2s)),
            "perm_r_squared_std": float(np.std(perm_r2s)),
            "p_permutation": float(p_perm),
            "n_permutations": n_permutations,
        }
    return None


def attribution_sensitivity(df: pd.DataFrame, category: str, gap_col: str,
                            source_attr: str = "source_age") -> dict:
    """Restrict to high-confidence attributions (A5)."""
    if "prob_speaker" not in df.columns:
        return {"status": "prob_speaker column not available"}

    high_conf = df[df["prob_speaker"] >= ATTRIBUTION_THRESHOLD]
    result = fit_gap_regression(high_conf, category, gap_col, source_attr)
    if result:
        result["attribution_filter"] = f"prob_speaker >= {ATTRIBUTION_THRESHOLD}"
        result["n_obs_filtered"] = len(high_conf)
        result["n_obs_total"] = len(df)
    return result


def binning_sensitivity(df: pd.DataFrame, category: str,
                        gap_col: str, base_width: int = 5) -> dict:
    """Recompute with half and double bin width (A13)."""
    results = {}
    for label, width in [("half", base_width // 2 or 1),
                          ("base", base_width),
                          ("double", base_width * 2)]:
        binned = df.copy()
        binned["gap_bin"] = (binned[gap_col] // width) * width
        stats_df = compute_bin_statistics(binned, category, "gap_bin")
        results[label] = {
            "bin_width": width,
            "n_bins": len(stats_df),
            "stats": stats_df.to_dict("records"),
        }
    return results


# ── Variance Decomposition (Section 3b) ──────────────────────────────

def variance_decomposition(df: pd.DataFrame, category: str,
                           source_bin: str, target_bin: str) -> dict:
    """Decompose variance into source, target, interaction, residual.

    Replaces subjective descriptions ("pure vertical gradient") with
    partial η² values.
    """
    formula = f"{category} ~ C({source_bin}) * C({target_bin})"
    try:
        model = ols(formula, data=df.dropna(subset=[category, source_bin, target_bin])).fit()
        aov = anova_lm(model, typ=2)

        result = {
            "category": category,
            "source_var": source_bin,
            "target_var": target_bin,
            "n_obs": int(model.nobs),
            "r_squared": float(model.rsquared),
        }

        for term in aov.index:
            if term != "Residual":
                result[f"{term}_partial_eta_sq"] = partial_eta_squared(aov, term)
                result[f"{term}_F"] = float(aov.loc[term, "F"])
                result[f"{term}_p"] = float(aov.loc[term, "PR(>F)"])

        return result

    except Exception as e:
        print(f"  Warning: variance decomposition failed for {category}: {e}")
        return None


# ── Main Pipeline ─────────────────────────────────────────────────────

def run_analysis(data_dir: str, output_dir: str):
    """Run the full analysis pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LIWC Analysis Pipeline — Quotegraph EMNLP Submission")
    print("=" * 60)

    # Load and preprocess
    print("\n1. Loading data...")
    df = load_quotegraph(data_dir)

    print("\n2. Preprocessing...")
    df = preprocess(df)

    # Discover all LIWC categories for FDR correction
    global ALL_CATEGORIES
    liwc_cols = [c for c in df.columns if c not in {
        "quoteID", "source_id", "target_id", "quotation", "n_words",
        "source_age", "target_age", "source_pagerank", "target_pagerank",
        "source_occupation", "target_occupation", "source_nationality",
        "target_nationality", "source_gender", "target_gender",
        "year", "prob_speaker", "log_quote_length", "age_diff",
        "pagerank_diff", "source_age_bin", "target_age_bin",
        "age_diff_bin", "dyad_id",
    }]
    ALL_CATEGORIES = liwc_cols
    print(f"\n  Found {len(ALL_CATEGORIES)} LIWC categories")
    print(f"  Selected for paper: {SELECTED_CATEGORIES}")

    # ── 3. Gap regressions (Section 3a) ────────────────────────────
    print("\n3. Running gap regressions...")

    age_gap_results = []
    pr_gap_results = []

    for cat in ALL_CATEGORIES:
        if cat not in df.columns:
            continue
        r = fit_gap_regression(df, cat, "age_diff", "source_age")
        if r:
            age_gap_results.append(r)
        r = fit_gap_regression(df, cat, "pagerank_diff", "source_pagerank")
        if r:
            pr_gap_results.append(r)

    # FDR correction across ALL categories
    print(f"  Age gap: {len(age_gap_results)} categories fitted")
    age_gap_results = run_fdr_correction(age_gap_results)

    print(f"  PageRank gap: {len(pr_gap_results)} categories fitted")
    pr_gap_results = run_fdr_correction(pr_gap_results)

    # Count survivors
    for label, results in [("Age gap", age_gap_results),
                            ("PageRank gap", pr_gap_results)]:
        n_sig_linear = sum(1 for r in results
                           if r.get("gap_significant_fdr", False))
        n_sig_quad = sum(1 for r in results
                         if r.get("gap_sq_significant_fdr", False))
        print(f"  {label}: {n_sig_linear} linear, {n_sig_quad} quadratic "
              f"survive FDR (q < 0.05)")

    # Save regression results
    with open(output_path / "gap_regressions_age.json", "w") as f:
        json.dump(age_gap_results, f, indent=2)
    with open(output_path / "gap_regressions_pagerank.json", "w") as f:
        json.dump(pr_gap_results, f, indent=2)

    # ── 4. Interaction regressions ─────────────────────────────────
    print("\n4. Running interaction regressions...")

    interaction_results = []
    for cat in SELECTED_CATEGORIES:
        if cat not in df.columns:
            continue
        r = fit_interaction_regression(df, cat, "source_age", "target_age")
        if r:
            interaction_results.append(r)
        r = fit_interaction_regression(df, cat, "source_pagerank", "target_pagerank")
        if r:
            interaction_results.append(r)

    with open(output_path / "interaction_regressions.json", "w") as f:
        json.dump(interaction_results, f, indent=2)

    # ── 5. Variance decomposition (Section 3b) ────────────────────
    print("\n5. Variance decomposition for heatmaps...")

    vardecomp_results = []
    for cat in SELECTED_CATEGORIES:
        if cat not in df.columns:
            continue
        r = variance_decomposition(df, cat, "source_age_bin", "target_age_bin")
        if r:
            vardecomp_results.append(r)

    with open(output_path / "variance_decomposition.json", "w") as f:
        json.dump(vardecomp_results, f, indent=2)

    # ── 6. Aggregated bin statistics (Section 3c / reproducibility) ─
    print("\n6. Computing aggregated bin statistics...")

    agg_dir = output_path / "aggregated"
    agg_dir.mkdir(exist_ok=True)

    for cat in SELECTED_CATEGORIES:
        if cat not in df.columns:
            continue

        # Age difference bins
        stats_df = compute_bin_statistics(df, cat, "age_diff_bin")
        stats_df.to_csv(agg_dir / f"{cat}_age_diff_bins.csv", index=False)

        # PageRank difference bins
        pr_bins = pd.qcut(df["pagerank_diff"].dropna(), q=20, duplicates="drop")
        df_pr = df.copy()
        df_pr["pr_diff_bin"] = pr_bins
        stats_df = compute_bin_statistics(df_pr, cat, "pr_diff_bin")
        stats_df.to_csv(agg_dir / f"{cat}_pagerank_diff_bins.csv", index=False)

        # Heatmap cells (source_age_bin × target_age_bin)
        hm_df = compute_heatmap_statistics(df, cat, "source_age_bin", "target_age_bin")
        hm_df.to_csv(agg_dir / f"{cat}_age_heatmap.csv", index=False)

    # ── 7. Robustness checks (Section 3d) ─────────────────────────
    print("\n7. Running robustness checks...")

    robustness = {}

    # 7a. Occupation stratification
    print("  7a. Occupation stratification...")
    occ_results = {}
    for cat in SELECTED_CATEGORIES:
        if cat not in df.columns:
            continue
        occ_results[cat] = {}
        for occ in ["politician", "athlete"]:
            r = occupation_stratification(df, cat, "age_diff", occ, "source_age")
            occ_results[cat][occ] = r
    robustness["occupation_stratification"] = occ_results

    # 7b. Temporal split
    print("  7b. Temporal split...")
    temporal_results = {}
    for cat in SELECTED_CATEGORIES:
        if cat not in df.columns:
            continue
        temporal_results[cat] = temporal_split(df, cat, "age_diff", "source_age")
    robustness["temporal_split"] = temporal_results

    # 7c. Permutation baseline (expensive — reduced n_permutations)
    print("  7c. Permutation baseline (100 iterations)...")
    perm_results = {}
    for cat in SELECTED_CATEGORIES:
        if cat not in df.columns:
            continue
        r = permutation_baseline(df, cat, "age_diff", n_permutations=100,
                                 source_attr="source_age")
        perm_results[cat] = r
    robustness["permutation_baseline"] = perm_results

    # 7d. Attribution sensitivity
    print("  7d. Attribution sensitivity...")
    attr_results = {}
    for cat in SELECTED_CATEGORIES:
        if cat not in df.columns:
            continue
        attr_results[cat] = attribution_sensitivity(
            df, cat, "age_diff", "source_age")
    robustness["attribution_sensitivity"] = attr_results

    # 7e. Binning sensitivity
    print("  7e. Binning sensitivity...")
    bin_results = {}
    for cat in SELECTED_CATEGORIES:
        if cat not in df.columns:
            continue
        bin_results[cat] = binning_sensitivity(
            df, cat, "age_diff", base_width=AGE_DIFF_BIN_WIDTH)
    robustness["binning_sensitivity"] = bin_results

    with open(output_path / "robustness_checks.json", "w") as f:
        json.dump(robustness, f, indent=2, default=str)

    # ── 8. Summary report ──────────────────────────────────────────
    print("\n8. Generating summary report...")

    summary = generate_summary(
        age_gap_results, pr_gap_results,
        interaction_results, vardecomp_results,
        robustness, df,
    )

    with open(output_path / "summary_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll results saved to {output_path}/")
    print("Done.")


def generate_summary(age_gap_results, pr_gap_results,
                     interaction_results, vardecomp_results,
                     robustness, df) -> dict:
    """Generate a summary for the paper text."""
    summary = {
        "data": {
            "n_quotes": len(df),
            "n_speakers": int(df["source_id"].nunique()),
            "n_targets": int(df["target_id"].nunique()),
            "n_dyads": int(df["dyad_id"].nunique()),
            "quote_length_median": float(df["n_words"].median()),
            "quote_length_mean": float(df["n_words"].mean()),
            "year_range": f"{int(df['year'].min())}-{int(df['year'].max())}",
        },
        "n_categories_tested": len(ALL_CATEGORIES) if ALL_CATEGORIES else 0,
        "n_categories_selected": len(SELECTED_CATEGORIES),
    }

    # Key findings for selected categories
    findings = {}
    for cat in SELECTED_CATEGORIES:
        cat_findings = {}

        # Age gap regression
        age_r = next((r for r in age_gap_results if r["category"] == cat), None)
        if age_r:
            cat_findings["age_gap"] = {
                "linear_coef": age_r.get("gap_coef"),
                "quadratic_coef": age_r.get("gap_sq_coef"),
                "linear_sig_fdr": age_r.get("gap_significant_fdr"),
                "quadratic_sig_fdr": age_r.get("gap_sq_significant_fdr"),
                "partial_eta_sq_linear": age_r.get("gap_partial_eta_sq"),
                "partial_eta_sq_quad": age_r.get("gap_sq_partial_eta_sq"),
                "r_squared": age_r.get("r_squared"),
                "quadratic_preferred": age_r.get("quadratic_preferred"),
            }

        # PageRank gap regression
        pr_r = next((r for r in pr_gap_results if r["category"] == cat), None)
        if pr_r:
            cat_findings["pagerank_gap"] = {
                "linear_coef": pr_r.get("gap_coef"),
                "quadratic_coef": pr_r.get("gap_sq_coef"),
                "linear_sig_fdr": pr_r.get("gap_significant_fdr"),
                "quadratic_sig_fdr": pr_r.get("gap_sq_significant_fdr"),
                "partial_eta_sq_linear": pr_r.get("gap_partial_eta_sq"),
                "partial_eta_sq_quad": pr_r.get("gap_sq_partial_eta_sq"),
                "r_squared": pr_r.get("r_squared"),
                "quadratic_preferred": pr_r.get("quadratic_preferred"),
            }

        findings[cat] = cat_findings

    summary["findings"] = findings
    return summary


# ── CLI Entry Point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LIWC statistical analysis pipeline for Quotegraph EMNLP submission"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=os.path.join(os.path.expanduser("~"), "data"),
        help="Directory containing quotegraph.parquet",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory for output files",
    )
    args = parser.parse_args()

    run_analysis(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
