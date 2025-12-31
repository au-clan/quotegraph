#!/usr/bin/env python3
"""
Article sampling script for QuoteGraph data quality analysis.

Samples a specified number of quotes from each year in the dataset for manual review.
"""

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import quotegraph.utils as u


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample articles from QuoteGraph parquet data by year.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/dlabdata1/culjak/data/quotegraph.parquet",
        help="Path to input QuoteGraph parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="yearly_samples.pkl",
        help="Path to output file (pickle or CSV based on extension)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2008,
        help="Start year for sampling (inclusive)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2020,
        help="End year for sampling (inclusive)",
    )
    parser.add_argument(
        "--samples-per-year",
        type=int,
        default=20,
        help="Number of samples to take per year",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--print-samples",
        action="store_true",
        help="Print samples to stdout in a readable format",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=24,
        help="Number of Spark threads",
    )
    parser.add_argument(
        "--driver-memory",
        type=str,
        default="120g",
        help="Spark driver memory",
    )
    return parser.parse_args()


def sample_by_year(spark, input_path, years, samples_per_year, seed=None):
    """Sample quotes from each year in the dataset."""
    yearly_samples = []

    for year in years:
        year_df = spark.read.parquet(input_path).where(u.year("quoteID") == str(year))
        count = year_df.count()

        if count == 0:
            print(f"Year {year}: No data found, skipping.")
            continue

        # Sample with a fraction that gives approximately 2x the desired samples
        # then limit to exact count
        fraction = min(1.0, (samples_per_year * 2) / count)
        sample_df = year_df.sample(withReplacement=False, fraction=fraction, seed=seed)
        sample_df = sample_df.limit(samples_per_year)

        sample_pandas = sample_df.toPandas()
        yearly_samples.append(sample_pandas)
        print(f"Year {year}: Sampled {len(sample_pandas)} quotes from {count} total.")

    return yearly_samples


def print_samples(yearly_samples, years):
    """Print samples in a readable format with Wikidata links."""
    for year, samples in zip(years, yearly_samples):
        print(f"\n{'=' * 100}")
        print(f"YEAR {year}")
        print(f"{'=' * 100}\n")

        for idx, row in samples.iterrows():
            print(f"Speaker: https://www.wikidata.org/wiki/{row.speaker}")
            print(f"Target: https://www.wikidata.org/wiki/{row.target}")
            print("Quote:")
            print("\n".join(textwrap.wrap(str(row.quotation), width=80)))
            search_query = str(row.quotation).replace(" ", "+")
            print(
                f"Search: https://quotebank.dlab.tools/search?target=quotation&text={search_query}"
                "&from_date=2008-09-01&to_date=2020-04-17"
            )
            print(f"quoteID: {row.quoteID}")
            print("-" * 100)


def save_samples(yearly_samples, output_path, years):
    """Save samples to pickle or CSV file."""
    output_path = Path(output_path)

    if output_path.suffix == ".csv":
        import pandas as pd

        # Combine all samples with year column
        combined = []
        for year, samples in zip(years, yearly_samples):
            samples = samples.copy()
            samples["sample_year"] = year
            combined.append(samples)
        combined_df = pd.concat(combined, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        print(f"Saved {len(combined_df)} samples to {output_path}")
    else:
        u.save_pickle(yearly_samples, str(output_path))
        print(f"Saved samples to {output_path}")


def main():
    args = parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    print(f"Sampling {args.samples_per_year} quotes per year for years {years[0]}-{years[-1]}")

    spark = u.start_spark(
        appName="article_sampling",
        n_threads=args.n_threads,
        driver_memory=args.driver_memory,
    )

    try:
        yearly_samples = sample_by_year(
            spark,
            args.input,
            years,
            args.samples_per_year,
            seed=args.seed,
        )

        if args.print_samples:
            print_samples(yearly_samples, years[: len(yearly_samples)])

        save_samples(yearly_samples, args.output, years[: len(yearly_samples)])

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
