import sys
import pyspark.sql.functions as F
import pyspark.sql.types as T
import datetime
from pyspark.sql import SparkSession

PATH_TO_REDUCED_QUOTEBANK = sys.argv[1]
PATH_TO_OUT = sys.argv[2]

spark = (SparkSession.builder.master("local[24]")
         .appName("Quotegraph")
         .config("spark.driver.memory", "32g")
         .config("spark.executor.memory", "32g")
         .config('spark.local.dir', '/dlabdata1/culjak/tmp')
         .getOrCreate())

qb = spark.read.parquet(PATH_TO_REDUCED_QUOTEBANK)

@F.udf(T.ArrayType(T.ArrayType(T.StringType())))
def extract_edges(content, quotations, names, ends):
    edges = []
    for name in names:
        # The offset list is currently a string so the first step is to convert it to a list of pairs denoting the
        # starting and the ending offset
        name['offsets'] = [list(map(int, i.split(', '))) for i in name['offsets'][2:-2].replace('], [', '|').split('|')]

    split_content = None
    for quotation, end in zip(quotations, ends):
        if quotation.localTopSpeaker == 'None':
            continue
        source = quotation.localTopSpeaker
        for name in names:
            mentions = [] # Entity mentions inside quotations will be saved for post-processing
            for offset in name['offsets']:
                if quotation.quotationOffset <= offset[0] < end:
                    if split_content is None:
                        split_content = content.split()
                    mention = ' '.join(split_content[offset[0]: offset[1]])
                    if offset[1] - offset[0] == 1:
                        if len(mention) == 1: # If a mention has only one letter, it is faulty
                            continue
                        if not any(c.isalpha() for c in mention): # If there are no letters in a mention, it is faulty
                            continue
                    mentions.append(mention)
            if len(mentions) != 0:
                edges.append([source, name['name'], quotation.quotation, quotation.quoteID, '|'.join(mentions)])
    return edges


qb.select('articleID', 'url', F.explode(extract_edges('content', 'quotations', 'names', 'ends')).alias('edge')) \
    .select('articleID', 'url', F.col('edge').getItem(0).alias('source'),
            F.col('edge').getItem(1).alias('target'),
            F.col('edge').getItem(2).alias('quotation'),
            F.col('edge').getItem(3).alias('quoteID'),
            F.col('edge').getItem(4).alias('target_mentions')) \
    .write.parquet(PATH_TO_OUT)