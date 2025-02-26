from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

def start_spark(config=None, appName='Quotegraph', n_threads=24):
    spark = (SparkSession.builder.master(f'local[{n_threads}]')
             .appName(appName)
             .config("spark.driver.memory", "60g")
             .config("spark.executor.memory", "32g")
             .config('spark.local.dir', '/dlabdata1/culjak/tmp')
             .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
             .config('spark.driver.maxResultSize', '16g'))
    if config is not None:
        for k, v in config:
            spark = spark.config(k, v)

    return spark.getOrCreate()