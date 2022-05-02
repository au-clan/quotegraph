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
PATH_TO_WHITE_QUOTEGRAPH = sys.argv[2]

with open('data/mention_blacklist.txt', 'r') as f:
    mention_blacklist = set([line.strip() for line in f])

with open('data/name_blacklist.txt', 'r') as f:
    name_blacklist = set([line.strip() for line in f])

@F.udf(T.BooleanType())
def all_blacklisted(mentions):
    return all(mention in mention_blacklist for mention in mentions.split('|'))

@F.udf(T.BooleanType())
def node_blacklisted(node):
    return node.lower() in name_blacklist


quotegraph = spark.read.parquet(PATH_TO_QUOTEGRAPH)

quotegraph.where(~all_blacklisted('target_mentions'))\
    .where(~node_blacklisted('target'))\
    .where(~node_blacklisted('source'))\
    .write.parquet(PATH_TO_WHITE_QUOTEGRAPH)
