import argparse
import quotegraph.utils as u
import pyspark.sql.functions as F


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qb", type=str, required=True, help="Path to quotebank parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to output parquet file")
    parser.add_argument("--n_threads", type=int, required=True, help="Number of threads to use")
    
    args = parser.parse_args()
    spark = u.start_spark(n_threads=args.n_threads)
    quotebank = spark.read.parquet(args.qb)

    quotebank.select(
        F.explode("quotations").alias("quotation"),
        u.extract_outlet_domain("url").alias("outlet_domain"),
    ).select("quotation.quoteID", "outlet_domain").distinct().groupby("quoteID").agg(
        F.countDistinct("outlet_domain").alias("num_news_outlets")
    ).write.parquet(args.output, mode="overwrite")
