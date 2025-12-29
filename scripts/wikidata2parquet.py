import argparse
import quotegraph.utils as u
import pyspark.sql.functions as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wikidata_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--n_threads', type=int, default=24)
    args = parser.parse_args()

    spark = u.start_spark()

    df = spark.read.json(args.wikidata_file)
    df.write.parquet(args.output_file)