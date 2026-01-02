import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pickle
import ujson as json

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
        .config("spark.local.dir", "/dlabdata1/culjak/tmp")
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


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


@F.udf(T.ArrayType(T.StringType()))
def get_alias_list(aliases: str | None) -> list:
    if aliases is None:
        return []

    aliases = json.loads(aliases)
    return [alias['value'] for alias in aliases if alias['language'] == 'en']


@F.udf(T.StringType())
def get_label(labels: str | None) -> str:
    if labels is None:
        return ""

    return json.loads(labels)['value']

item_statement_struct = T.StructType([
    T.StructField('property', T.StringType(), False),
    T.StructField('qid', T.StringType(), False)
])

@F.udf(T.ArrayType(item_statement_struct))
def get_item_statements(line: str | None) -> list:
    if line is None:
        return []

    claims = json.loads(line)
    statements = []
    
    for prop, snaks in claims.items():
        for snak in snaks:
            snak = snak['mainsnak']
            if 'datavalue' not in snak:
                continue
            datavalue = snak['datavalue']
            
            valuetype = datavalue['type']
            if valuetype == 'wikibase-entityid' and datavalue['value']['entity-type'] == 'item':
                if 'id' not in datavalue['value']:
                    datavalue['value']['id'] = 'Q' + str(datavalue['value']['numeric-id'])
                statements.append((prop, datavalue['value']['id']))
    return statements



sitelink_struct = T.StructType([
    T.StructField("type", T.StringType(), False),
    T.StructField("title", T.StringType(), False)
])

@F.udf(T.ArrayType(sitelink_struct))
def get_titles(line: str | None) -> list:
    if line is None:
        return []

    sitelinks = json.loads(line)
    return [(i, j['title']) for i, j in sitelinks.items()]


@F.udf(T.ArrayType(T.StringType()))
def label_aliases(label: str | None, aliases: list) -> list:
    if label is None:
        return aliases
    if len(aliases) == 0:
        return [label]
    return [label] + aliases

qid_alias_struct_type = T.StructType([
    T.StructField("qid", T.StringType(), nullable=False),
    T.StructField("aliases", T.ArrayType(T.StringType()), nullable=False),
])

@F.udf(qid_alias_struct_type)
def get_qid_alias_mapping_from_tsv(line):
    line = line.strip().split('\t')
    qid = line[0]
    label = line[1]
    aliases = line[2].split('|') if len(line) > 2 and line[2] else []

    if label == "":
        values = aliases
    else:
        values = [label] + ([a for a in aliases if a] if aliases else [])

    return {"qid": qid, "aliases": values}


@F.udf(T.ArrayType(T.StringType()))
def ascii_aliases(aliases: list) -> list:
    return [alias.lower() for alias in aliases] + [''.join(c if ord(c) < 128 else '?' for c in alias) for alias in aliases]

def has_sitelink(sitelink_name):
    @F.udf(T.BooleanType())
    def has_sitelink_udf(sitelinks):
        for sitelink in sitelinks:
            if sitelink['type'] == sitelink_name:
                return True
        return False

    return has_sitelink_udf

def is_instance_of(qid):
    @F.udf(T.BooleanType())
    def is_instance_of_udf(statements):
        for statement in statements:
            if statement['property'] == 'P31' and statement['qid'] == qid:
                return True
        return False
    return is_instance_of_udf

@F.udf(T.ArrayType(T.StringType()))
def phase_a_aliases(aliases: list) -> list:
    aliases = []
    for alias in aliases:
        alias = alias.lower()
        alias = alias.encode('utf-8')
        alias = alias.decode('Latin-1')
        aliases.append(alias)
    return aliases


@F.udf(T.ArrayType(T.StringType()))
def phase_b_aliases(aliases: list) -> list:
    aliases = []
    for alias in aliases:
        alias = alias.lower()
        alias = ''.join(c if ord(c) < 256 else '?' for c in alias)
        aliases.append(alias)
    return aliases

@F.udf(T.ArrayType(T.StringType()))
def phase_c_aliases(aliases: list) -> list:
    aliases = []
    for alias in aliases:
        alias = alias.lower()
        alias = ''.join(c if ord(c) < 128 else '?' for c in alias)
        aliases.append(alias)
    return aliases

@F.udf(T.ArrayType(T.StringType()))
def phase_d_aliases(aliases: list) -> list:
    aliases = []
    for alias in aliases:
        alias = ''.join(c if ord(c) < 128 else '?' for c in alias)
        aliases.append(alias)
    return aliases


@F.udf(T.ArrayType(T.StringType()))
def phase_e_aliases(aliases: list) -> list:
    return aliases

PHASE_ALIASES_MAPPING = {
    'A': phase_a_aliases,
    'B': phase_b_aliases,
    'C': phase_c_aliases,
    'D': phase_d_aliases,
    'E': phase_e_aliases,
}

@F.udf(T.ArrayType(T.StringType()))
def fit_aliases_to_phase(phase, aliases: list) -> list:
    return PHASE_ALIASES_MAPPING[phase](aliases)



