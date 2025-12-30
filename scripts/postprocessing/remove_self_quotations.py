import sys
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

spark = (SparkSession.builder.master("local[24]")
         .appName("Quotegraph")
         .config("spark.driver.memory", "32g")
         .config("spark.executor.memory", "32g")
         .config('spark.local.dir', '/scratch/culjak/tmp')
         .getOrCreate())
sc = spark.sparkContext

PATH_TO_QUOTEGRAPH = sys.argv[1]
PATH_TO_OUT = sys.argv[2]

quotegraph = spark.read.parquet(PATH_TO_QUOTEGRAPH)

@F.udf(T.IntegerType())
def self_quotation(source, target):
    return 1 * (source == target)

quotegraph.select('articleID', 'quoteID', 'source', self_quotation('source', 'target').alias('self_quotation'))\
    .groupby('articleID', 'source', 'quoteID')\
    .agg(F.sum('self_quotation').alias('self_quotation'))\
    .where('self_quotation == 0')\
    .join(quotegraph, on=['articleID', 'source', 'quoteID'])\
    .drop('self_quotation')\
    .write.parquet(PATH_TO_OUT)