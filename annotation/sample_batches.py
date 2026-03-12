import argparse
import os
import random
import pyspark.sql.functions as F

from quotegraph.utils import start_spark, load_pickle
from collections import defaultdict
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_articles_per_phase", type=int, default=750)
    parser.add_argument("--n_threads", type=int, default=56)
    args = parser.parse_args()

    spark = start_spark(n_threads=args.n_threads)
    DATA_DIR = os.path.join(os.path.expanduser("~"), "data")

    quotegraph_articleIDs = spark.read.parquet(f'{DATA_DIR}/quotegraph_articles_.parquet').toPandas()

    article2quoteIDs = dict(zip(quotegraph_articleIDs['articleID'], quotegraph_articleIDs['quoteIDs']))

    quoteIDs2articleIDs = defaultdict(set)

    for articleID, quoteIDs in tqdm(article2quoteIDs.items()):
        for quoteID in quoteIDs:
            quoteIDs2articleIDs[quoteID].add(articleID)
    
    article2article_overlaps = defaultdict(set)

    for articleID in tqdm(article2quoteIDs):
        for quoteID in article2quoteIDs[articleID]:
            for other_articleID in quoteIDs2articleIDs[quoteID]:
                if other_articleID != articleID:
                    article2article_overlaps[articleID].add(other_articleID)
   

    phase_AD_articles = quotegraph_articleIDs.where(F.col('phase') != 'E').select('articleID', 'quoteIDs').toPandas()
    phase_E_articles = quotegraph_articleIDs.where(F.col('phase') == 'E').select('articleID', 'quoteIDs').toPandas()

    quotebank = spark.read.parquet(f"{DATA_DIR}/quotebank") 
    quotegraph = spark.read.parquet(f"{DATA_DIR}/quotegraph.parquet")
    

    unique_quoteIDs = load_pickle(f"{DATA_DIR}/unique_quoteIDs.pkl")

    sampled_quoteIDs = random.sample(list(unique_quoteIDs), args.n_articles_per_phase)



    quotebank.show()