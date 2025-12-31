import argparse
import quotegraph.utils as u
import pyspark.sql.functions as F


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wikidata_path", type=str, required=True)
    parser.add_argument("output_path", type=str, required=True)
    parser.add_argument("--n_threads", type=int, default=24)
    args = parser.parse_args()

    spark = u.start_spark(n_threads=args.n_threads)
    wikidata = spark.read.parquet(args.wikidata_path)

    wikidata = wikidata.select(
        u.get_json_object(u.col("instance"), "$.id").alias("id"),
        u.get_alias_list(
            F.get_json_object(F.col("instance"), "$.aliases.en")
        ).alias("aliases"),
        u.get_label(F.get_json_object(F.col("instance"), "$.labels.en")).alias(
            "label"
        ),
        u.get_label(
            F.get_json_object(F.col("instance"), "$.descriptions.en")
        ).alias("description"),
        u.get_item_statements(
            F.get_json_object(F.col("instance"), "$.claims")
        ).alias("statements"),
        u.get_titles(
            F.get_json_object(F.col("instance"), "$.sitelinks")
        ).alias("sitelinks"),
    ).write.parquet(args.output_path)
