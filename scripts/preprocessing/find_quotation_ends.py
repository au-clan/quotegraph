import sys
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

PATH_TO_ARTICLE_QUOTEBANK = sys.argv[1]
PATH_TO_OUT = sys.argv[2]

spark = (SparkSession.builder.master("local[24]")
         .appName("Quotegraph")
         .config("spark.driver.memory", "32g")
         .config("spark.executor.memory", "32g")
         .config('spark.local.dir', '/dlabdata1/culjak/tmp')
         .getOrCreate())

articles = spark.read.parquet(PATH_TO_ARTICLE_QUOTEBANK)


@F.udf(T.ArrayType(T.IntegerType()))
def get_ends(content, quotations):
    content = content.split(' ')
    ends = []
    for quotation in quotations:
        quotation_start = quotation.quotationOffset
        start_qm_pos = quotation_start - 1
        if '``' in content[start_qm_pos]:
            bq = False
        elif content[start_qm_pos] == '<blockquote>':
            bq = True
        else:
            ends.append(-1)
            continue

        end_qm_pos = start_qm_pos + 1
        for i in range(start_qm_pos + 1, len(content)):
            if content[i] == "''" or (bq and content == '</blockquote>'):
                end_qm_pos = i
                break
        if content[end_qm_pos] == "''":
            ends.append(end_qm_pos)
        else:
            ends.append(-1)
    return ends


articles.select('articleID', 'content', 'quotations', get_ends('content', 'quotations').alias('ends'), 'names', 'url').write.parquet(PATH_TO_OUT)
