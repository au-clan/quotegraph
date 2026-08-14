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
    tokens = content.split(" ")
    ends = []
    for quotation in quotations:
        start_qm_pos = quotation.quotationOffset - 1
        if start_qm_pos < 0 or start_qm_pos >= len(tokens):
            ends.append(-1)
            continue
        opener = tokens[start_qm_pos]
        low = opener.lower()
        if "``" in opener:
            blockquote = False
        elif low == "<blockquote>" or low.startswith("<blockquote") or low == "\\blockquote":
            blockquote = True
        else:
            ends.append(-1)
            continue
        end_qm_pos = -1
        for i in range(start_qm_pos + 1, len(tokens)):
            tok = tokens[i]
            tok_low = tok.lower()
            if blockquote and tok_low in {"</blockquote>", "\\endblockquote"}:
                end_qm_pos = i
                break
            if not blockquote and "''" in tok:
                end_qm_pos = i
                break
        ends.append(end_qm_pos)
    return ends


articles.select('articleID', 'content', 'quotations', get_ends('content', 'quotations').alias('ends'), 'names', 'url').write.parquet(PATH_TO_OUT)
