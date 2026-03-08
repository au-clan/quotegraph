from quotegraph.utils import *
from tqdm.notebook import tqdm


import pyspark.sql.functions as F
import os
import random

spark = start_spark()

    
DATA_DIR = os.path.join(os.path.expanduser("~"), "data")

quotegraph = spark.read.parquet(f"{DATA_DIR}/quotegraph.parquet")
quotebank = spark.read.parquet(f"{DATA_DIR}/quotebank")

quotegraph.show()


unique_ids_df = quotegraph.groupBy("quoteID").count().filter(F.col("count") == 1).select("quoteID")
unique_quoteIDs = set(row.quoteID for row in unique_ids_df.collect())

sampled_quoteIDs = random.sample(list(unique_quoteIDs), 1000) if len(unique_quoteIDs) >= 1000 else list(unique_quoteIDs)

save_pickle(sampled_quoteIDs, f"{DATA_DIR}/sampled_quoteIDs.pkl")







