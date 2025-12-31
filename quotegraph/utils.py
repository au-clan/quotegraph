import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pickle
import orjson as json

from pyspark.sql import SparkSession
from tld import get_fld, get_tld


def start_spark(
    config=None, appName="quotegraph", n_threads=24, driver_memory="120g"
) -> pyspark.sql.SparkSession:
    spark = (
        pyspark.sql.SparkSession.builder.master(f"local[{n_threads}]")
        .appName(appName)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", "100g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.maxResultSize", "100g")
    )
    if config is not None:
        for k, v in config:
            spark = spark.config(k, v)

    return spark.getOrCreate()


@F.udf(T.StringType())
def extract_outlet_domain(url: str) -> str:
    domain = get_fld(url)
    return domain


@F.udf(T.StringType())
def date(quoteID: str) -> str:
    return quoteID[:10]


@F.udf(T.StringType())
def year(quoteID: str) -> str:
    return quoteID[:4]


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)



@F.udf(T.ArrayType(T.StringType()))
def get_alias_list(aliases: str | None) -> list:
    if aliases is None:
        return []

    aliases = json.loads(aliases)
    return [alias['value'] for alias in aliases if alias['language'] == 'en']


@F.udf(T.StringType())
def get_label(labels: str|None) -> str:
    if labels is None:
        return ""

    return json.loads(labels)['value']
