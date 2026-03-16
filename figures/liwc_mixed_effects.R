#!/usr/bin/env Rscript
#
# Mixed-effects regression analysis for Quotegraph LIWC section.
#
# This is the PREFERRED analysis (EMNLP_PLAN.md Section 3a).
# The Python pipeline (liwc_analysis.py) provides a fallback with
# OLS + dyad-clustered standard errors if lme4 is computationally
# infeasible at N ~ 8M.
#
# Models:
#   Gap model: LIWC ~ poly(gap, 2) + source_attr + occupation_source
#              + occupation_target + nationality_source + log(n_words)
#              + (1 | source_id) + (1 | target_id)
#
#   Interaction model: LIWC ~ source_attr * target_attr + covariates
#                      + (1 | source_id) + (1 | target_id)
#
# Usage:
#   Rscript liwc_mixed_effects.R --input data.csv --output results/

library(lme4)
library(lmerTest)    # p-values via Satterthwaite
library(effectsize)  # eta_squared()
library(performance) # r2()
library(arrow)       # read_parquet()
library(dplyr)
library(tidyr)
library(jsonlite)

# ── Configuration ──────────────────────────────────────────────────

SELECTED_CATEGORIES <- c("Clout", "power", "conflict", "Tone", "i", "we")
MIN_QUOTE_LENGTH <- 25
MIN_BIN_N <- 100
FDR_ALPHA <- 0.05

# ── Argument parsing ──────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
input_path <- ifelse(length(args) >= 2 && args[1] == "--input",
                     args[2], "~/data/quotegraph_liwc.csv")
output_dir <- ifelse(length(args) >= 4 && args[3] == "--output",
                     args[4], "results")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ── Load and preprocess ──────────────────────────────────────────

cat("Loading data from", input_path, "\n")
if (grepl("\\.parquet$", input_path)) {
  df <- read_parquet(input_path)
} else {
  df <- read.csv(input_path)
}

cat("Initial N:", nrow(df), "\n")

# Quote length filter
df <- df[df$n_words >= MIN_QUOTE_LENGTH, ]
cat("After length filter (>=", MIN_QUOTE_LENGTH, "words):", nrow(df), "\n")

# Remove self-loops
df <- df[df$source_id != df$target_id, ]
cat("After self-loop removal:", nrow(df), "\n")

# Derived columns
df$log_n_words <- log(df$n_words)
df$age_diff <- df$source_age - df$target_age
df$age_diff_sq <- df$age_diff^2
df$pagerank_diff <- df$source_pagerank - df$target_pagerank
df$pagerank_diff_sq <- df$pagerank_diff^2

# Factor encoding for random effects
df$source_id <- as.factor(df$source_id)
df$target_id <- as.factor(df$target_id)
df$source_occupation <- as.factor(df$source_occupation)
df$target_occupation <- as.factor(df$target_occupation)
df$source_nationality <- as.factor(df$source_nationality)

cat("Unique speakers:", nlevels(df$source_id), "\n")
cat("Unique targets:", nlevels(df$target_id), "\n")

# ── Gap models ────────────────────────────────────────────────────

fit_gap_model <- function(df, category, gap_col, source_attr) {
  gap_sq_col <- paste0(gap_col, "_sq")

  formula_str <- paste0(
    category, " ~ ", gap_col, " + ", gap_sq_col, " + ", source_attr,
    " + source_occupation + target_occupation + source_nationality",
    " + log_n_words + (1 | source_id) + (1 | target_id)"
  )

  cat("  Fitting:", formula_str, "\n")

  tryCatch({
    model <- lmer(as.formula(formula_str), data = df,
                  REML = FALSE,  # ML for AIC comparison
                  control = lmerControl(optimizer = "bobyqa",
                                        optCtrl = list(maxfun = 50000)))

    # Also fit linear-only for AIC comparison
    formula_linear <- paste0(
      category, " ~ ", gap_col, " + ", source_attr,
      " + source_occupation + target_occupation + source_nationality",
      " + log_n_words + (1 | source_id) + (1 | target_id)"
    )
    model_linear <- lmer(as.formula(formula_linear), data = df,
                         REML = FALSE,
                         control = lmerControl(optimizer = "bobyqa",
                                               optCtrl = list(maxfun = 50000)))

    # Extract results
    coefs <- summary(model)$coefficients
    ci <- confint(model, method = "Wald",
                  parm = c(gap_col, gap_sq_col))

    # Effect sizes (partial eta-squared)
    eta <- eta_squared(model, partial = TRUE, ci = 0.95)

    # R-squared (marginal and conditional)
    r2_vals <- r2(model)

    result <- list(
      category = category,
      gap_variable = gap_col,
      n_obs = nobs(model),
      # Linear term
      gap_coef = coefs[gap_col, "Estimate"],
      gap_se = coefs[gap_col, "Std. Error"],
      gap_pvalue = coefs[gap_col, "Pr(>|t|)"],
      gap_ci_lower = ci[gap_col, 1],
      gap_ci_upper = ci[gap_col, 2],
      # Quadratic term
      gap_sq_coef = coefs[gap_sq_col, "Estimate"],
      gap_sq_se = coefs[gap_sq_col, "Std. Error"],
      gap_sq_pvalue = coefs[gap_sq_col, "Pr(>|t|)"],
      gap_sq_ci_lower = ci[gap_sq_col, 1],
      gap_sq_ci_upper = ci[gap_sq_col, 2],
      # Model fit
      r_squared_marginal = r2_vals$R2_marginal,
      r_squared_conditional = r2_vals$R2_conditional,
      aic_quadratic = AIC(model),
      aic_linear = AIC(model_linear),
      quadratic_preferred = AIC(model) < AIC(model_linear),
      # Random effects variance
      var_source = as.numeric(VarCorr(model)$source_id),
      var_target = as.numeric(VarCorr(model)$target_id),
      var_residual = sigma(model)^2
    )

    # Add partial eta-squared if available
    for (i in seq_len(nrow(eta))) {
      param <- eta$Parameter[i]
      if (param == gap_col) {
        result$gap_partial_eta_sq <- eta$Eta2_partial[i]
      } else if (param == gap_sq_col) {
        result$gap_sq_partial_eta_sq <- eta$Eta2_partial[i]
      }
    }

    return(result)
  }, error = function(e) {
    cat("    ERROR:", conditionMessage(e), "\n")
    return(list(category = category, gap_variable = gap_col,
                status = "failed", error = conditionMessage(e)))
  })
}

# ── Run gap models for all categories ─────────────────────────────

cat("\n=== Age Gap Models ===\n")
age_gap_results <- list()
for (cat_name in SELECTED_CATEGORIES) {
  if (!(cat_name %in% colnames(df))) next
  result <- fit_gap_model(df, cat_name, "age_diff", "source_age")
  age_gap_results[[cat_name]] <- result
}

cat("\n=== PageRank Gap Models ===\n")
pr_gap_results <- list()
for (cat_name in SELECTED_CATEGORIES) {
  if (!(cat_name %in% colnames(df))) next
  result <- fit_gap_model(df, cat_name, "pagerank_diff", "source_pagerank")
  pr_gap_results[[cat_name]] <- result
}

# ── FDR correction ────────────────────────────────────────────────

apply_fdr <- function(results_list) {
  # Collect all p-values
  p_vals <- c()
  p_labels <- c()

  for (r in results_list) {
    if (!is.null(r$gap_pvalue)) {
      p_vals <- c(p_vals, r$gap_pvalue)
      p_labels <- c(p_labels, paste0(r$category, "_gap"))
    }
    if (!is.null(r$gap_sq_pvalue)) {
      p_vals <- c(p_vals, r$gap_sq_pvalue)
      p_labels <- c(p_labels, paste0(r$category, "_gap_sq"))
    }
  }

  if (length(p_vals) == 0) return(results_list)

  corrected <- p.adjust(p_vals, method = "BH")
  names(corrected) <- p_labels

  # Map back
  for (i in seq_along(results_list)) {
    r <- results_list[[i]]
    gap_label <- paste0(r$category, "_gap")
    gap_sq_label <- paste0(r$category, "_gap_sq")

    if (gap_label %in% names(corrected)) {
      results_list[[i]]$gap_pvalue_fdr <- corrected[gap_label]
      results_list[[i]]$gap_significant_fdr <- corrected[gap_label] < FDR_ALPHA
    }
    if (gap_sq_label %in% names(corrected)) {
      results_list[[i]]$gap_sq_pvalue_fdr <- corrected[gap_sq_label]
      results_list[[i]]$gap_sq_significant_fdr <- corrected[gap_sq_label] < FDR_ALPHA
    }
  }

  return(results_list)
}

age_gap_results <- apply_fdr(age_gap_results)
pr_gap_results <- apply_fdr(pr_gap_results)

# ── Occupation stratification (REJECT-2) ──────────────────────────

cat("\n=== Occupation Stratification ===\n")
occ_results <- list()
for (occ in c("politician", "athlete")) {
  cat("  Occupation:", occ, "\n")
  occ_df <- df[df$source_occupation == occ | df$target_occupation == occ, ]
  cat("    N:", nrow(occ_df), "\n")

  if (nrow(occ_df) < MIN_BIN_N * 10) {
    cat("    Skipping: insufficient data\n")
    next
  }

  occ_results[[occ]] <- list()
  for (cat_name in SELECTED_CATEGORIES) {
    if (!(cat_name %in% colnames(occ_df))) next
    result <- fit_gap_model(occ_df, cat_name, "age_diff", "source_age")
    result$occupation_filter <- occ
    occ_results[[occ]][[cat_name]] <- result
  }
}

# ── Temporal split (m7) ──────────────────────────────────────────

cat("\n=== Temporal Split ===\n")
temporal_results <- list()
for (period in list(list(name = "early", start = 2008, end = 2014),
                    list(name = "late", start = 2015, end = 2020))) {
  cat("  Period:", period$name, "(", period$start, "-", period$end, ")\n")
  period_df <- df[df$year >= period$start & df$year <= period$end, ]
  cat("    N:", nrow(period_df), "\n")

  temporal_results[[period$name]] <- list()
  for (cat_name in SELECTED_CATEGORIES) {
    if (!(cat_name %in% colnames(period_df))) next
    result <- fit_gap_model(period_df, cat_name, "age_diff", "source_age")
    result$period <- period$name
    result$year_range <- paste0(period$start, "-", period$end)
    temporal_results[[period$name]][[cat_name]] <- result
  }
}

# ── Save results ──────────────────────────────────────────────────

cat("\nSaving results to", output_dir, "\n")

write_json(toJSON(age_gap_results, auto_unbox = TRUE, pretty = TRUE),
           file.path(output_dir, "mixed_effects_age_gap.json"))
write_json(toJSON(pr_gap_results, auto_unbox = TRUE, pretty = TRUE),
           file.path(output_dir, "mixed_effects_pagerank_gap.json"))
write_json(toJSON(occ_results, auto_unbox = TRUE, pretty = TRUE),
           file.path(output_dir, "mixed_effects_occupation_stratification.json"))
write_json(toJSON(temporal_results, auto_unbox = TRUE, pretty = TRUE),
           file.path(output_dir, "mixed_effects_temporal_split.json"))

# ── Summary table for paper ───────────────────────────────────────

cat("\n=== Summary for Paper Text ===\n")
cat("Copy these values into the paper (Section N.3 and N.4):\n\n")

for (cat_name in SELECTED_CATEGORIES) {
  r_age <- age_gap_results[[cat_name]]
  r_pr <- pr_gap_results[[cat_name]]

  cat(sprintf("--- %s ---\n", cat_name))

  if (!is.null(r_age) && is.null(r_age$status)) {
    cat(sprintf("  Age gap (linear):    β = %.3f [%.3f, %.3f], p_FDR = %.2e, η²p = %.4f\n",
                r_age$gap_coef, r_age$gap_ci_lower, r_age$gap_ci_upper,
                r_age$gap_pvalue_fdr, r_age$gap_partial_eta_sq))
    cat(sprintf("  Age gap (quadratic): β = %.3f [%.3f, %.3f], p_FDR = %.2e, η²p = %.4f\n",
                r_age$gap_sq_coef, r_age$gap_sq_ci_lower, r_age$gap_sq_ci_upper,
                r_age$gap_sq_pvalue_fdr, r_age$gap_sq_partial_eta_sq))
    cat(sprintf("  R²_marginal = %.4f, R²_conditional = %.4f, AIC_quad = %.1f vs AIC_lin = %.1f\n",
                r_age$r_squared_marginal, r_age$r_squared_conditional,
                r_age$aic_quadratic, r_age$aic_linear))
  }

  if (!is.null(r_pr) && is.null(r_pr$status)) {
    cat(sprintf("  PR gap (linear):     β = %.3f [%.3f, %.3f], p_FDR = %.2e, η²p = %.4f\n",
                r_pr$gap_coef, r_pr$gap_ci_lower, r_pr$gap_ci_upper,
                r_pr$gap_pvalue_fdr, r_pr$gap_partial_eta_sq))
    cat(sprintf("  PR gap (quadratic):  β = %.3f [%.3f, %.3f], p_FDR = %.2e, η²p = %.4f\n",
                r_pr$gap_sq_coef, r_pr$gap_sq_ci_lower, r_pr$gap_sq_ci_upper,
                r_pr$gap_sq_pvalue_fdr, r_pr$gap_sq_partial_eta_sq))
    cat(sprintf("  R²_marginal = %.4f, R²_conditional = %.4f, AIC_quad = %.1f vs AIC_lin = %.1f\n",
                r_pr$r_squared_marginal, r_pr$r_squared_conditional,
                r_pr$aic_quadratic, r_pr$aic_linear))
  }

  cat("\n")
}

cat("Done.\n")
