import quotegraph.utils as u

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_json_path', help='Path to the file or directory in JSON format.')
    parser.add_argument('output_parquet_path', help='Path where the data will be saved in Parquet format.')
    parser.add_argument('--n_threads', type=int, default=24, help='Number of threads to use for the conversion.')
    args = parser.parse_args()

    spark = u.start_spark(n_threads=args.n_threads)
    df = spark.read.json(args.input_json_path)
    df.write.mode('overwrite').parquet(args.output_parquet_path)