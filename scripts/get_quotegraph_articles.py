import os
import pyspark.sql.functions as F

from quotegraph.utils import start_spark

if __name__ == "__main__":
    spark = start_spark(n_threads=56)
    DATA_DIR = os.path.join(os.path.expanduser("~"), "data")

    quotegraph = spark.read.parquet(f"{DATA_DIR}/quotegraph.parquet")    
    articles = spark.read.parquet(f"{DATA_DIR}/quotebank")

    quotegraph_quoteIDs = quotegraph.select('quoteID').distinct()
    
    articles_exploded_quoteIDs = articles.select('articleID', F.explode("quotations").alias("quotation")).select('articleID', F.col('quotation.quoteID').alias('quoteID'))

    quotegraph_articleIDs = articles_exploded_quoteIDs.join(F.broadcast(quotegraph_quoteIDs), on="quoteID", how="inner").select('articleID', 'quoteID').groupby('articleID').agg(F.collect_list('quoteID').alias('quoteIDs'))


    spark.read.parquet(f'{DATA_DIR}/quotegraph_articles.parquet').join(quotegraph_articleIDs, on="articleID", how="inner").select('articleID', 'quoteIDs', 'phase').write.parquet(f"{DATA_DIR}/quotegraph_articles_.parquet", mode="overwrite")
    quotegraph_articles = articles.join(F.broadcast(quotegraph_articleIDs), on="articleID", how="inner")
    quotegraph_articles.write.parquet(f"{DATA_DIR}/quotegraph_articles.parquet", mode="overwrite")
