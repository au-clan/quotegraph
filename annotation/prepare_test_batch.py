import os
import json
import ast
import re
import requests

from tqdm import tqdm
from quotegraph.utils import start_spark, get_ends_dict, save_pickle
from nltk.tokenize.stanford import StanfordTokenizer


def fetch_wikidata(qid):
    import time
    for attempt in range(3):
        resp = requests.get(
            'https://www.wikidata.org/w/api.php',
            params={'action': 'wbgetentities', 'ids': qid, 'props': 'labels|descriptions', 'languages': 'en', 'format': 'json'},
            headers={'User-Agent': 'quotegraph/1.0'}
        )
        if resp.text:
            break
        time.sleep(2 ** attempt)
    data = resp.json()['entities'][qid]
    label = data.get('labels', {}).get('en', {}).get('value', '')
    description = data.get('descriptions', {}).get('en', {}).get('value', '')
    return label, description

def detokenize_and_match_offsets(article):
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    content = article['content']
    quotations = article['quotations'] or []
    names = article['names'] or []

    tokens = content.split(' ')
    detokenizer = TreebankWordDetokenizer()
    detokenized = detokenizer.detokenize(tokens)
    # Remove any whitespace before punctuation that the detokenizer missed
    detokenized = re.sub(r'\s+([.,!?;:])', r'\1', detokenized)

    # Map PTB tokens that get rewritten during detokenization
    TOKEN_SURFACE = {'``': '"', "''": '"'}

    def build_char_starts(tok_list):
        char_starts = []
        pos = 0
        for token in tok_list:
            surface = TOKEN_SURFACE.get(token, token)
            idx = detokenized.find(surface, pos)
            if idx != -1:
                char_starts.append(idx)
                pos = idx + len(surface)
            else:
                char_starts.append(-1)
        return char_starts

    # Quotations use split(' ') token indices
    token_char_starts = build_char_starts(tokens)

    # Names use split() token indices (strips and collapses whitespace)
    name_tokens = content.split()
    name_token_char_starts = build_char_starts(name_tokens)

    def token_char_end(tok_idx, tok_list, char_starts):
        if tok_idx < 0 or tok_idx >= len(tok_list):
            return -1
        start = char_starts[tok_idx]
        if start == -1:
            return -1
        surface = TOKEN_SURFACE.get(tok_list[tok_idx], tok_list[tok_idx])
        return start + len(surface)

    # Process quotations: add charStart/charEnd and contextCharStart/contextCharEnd
    ends = get_ends_dict(content, quotations)
    result_quotations = []
    for quotation, end_tok_idx in zip(quotations, ends):
        q = dict(quotation)
        start_tok = quotation['quotationOffset']
        # opening quote mark is one token before quotationOffset
        open_qm = start_tok - 1
        if end_tok_idx != -1:
            q['charStart'] = token_char_starts[open_qm] if open_qm >= 0 else token_char_starts[start_tok]
            q['charEnd'] = token_char_end(end_tok_idx, tokens, token_char_starts)
        else:
            q['charStart'] = -1
            q['charEnd'] = -1

        ctx_start = quotation.get('contextStart')
        ctx_end = quotation.get('contextEnd')
        q['contextCharStart'] = token_char_starts[ctx_start] if ctx_start is not None and 0 <= ctx_start < len(token_char_starts) else -1
        q['contextCharEnd'] = token_char_end(ctx_end, tokens, token_char_starts) if ctx_end is not None else -1
        result_quotations.append(q)

    # Process names: each entry is a dict {name: offsets_str}
    # offsets_str is a string repr of a nested list [[start, end], ...]
    result_names = []

    for name in names:
        spans = ast.literal_eval(name['offsets'])
        char_spans = []
        for span in spans:
            start_tok, end_tok = span[0], span[1]
            char_spans.append([
                name_token_char_starts[start_tok] if 0 <= start_tok < len(name_token_char_starts) else -1,
                token_char_end(end_tok - 1, name_tokens, name_token_char_starts)
            ])
        name['char_offsets'] = char_spans
        result_names.append(name)

    article['detokenized_content'] = detokenized
    article['quotations'] = result_quotations
    article['names'] = result_names
    return article




def print_highlighted(article):
    # ANSI colors
    RESET   = '\033[0m'
    YELLOW  = '\033[43m\033[30m'  # quote background
    GREEN   = '\033[42m\033[30m'  # name background
    OVERLAP = '\033[45m\033[30m'  # name inside quote

    text = article['detokenized_content']
    n = len(text)

    in_quote = [False] * n
    in_name  = [False] * n

    for q in article['quotations']:
        s, e = q.get('charStart', -1), q.get('charEnd', -1)
        if s != -1 and e != -1:
            for i in range(s, min(e, n)):
                in_quote[i] = True

    for name in article['names']:
        for s, e in name['char_offsets']:
            if s != -1 and e != -1:
                for i in range(s, min(e, n)):
                    in_name[i] = True

    # Render with color transitions
    out = []
    prev_color = None
    for i, ch in enumerate(text):
        if in_name[i] and in_quote[i]:
            color = OVERLAP
        elif in_name[i]:
            color = GREEN
        elif in_quote[i]:
            color = YELLOW
        else:
            color = RESET
        if color != prev_color:
            out.append(color)
            prev_color = color
        out.append(ch)
    out.append(RESET)
    rendered = ''.join(out)

    print(f"\n{'='*60}")
    print(f"Article: {article.get('articleID')}  |  {article.get('date')}")
    print(f"Title  : {article.get('title')}")
    print(f"{'='*60}")
    print(rendered)
    print(f"\n--- Quotations ({len(article['quotations'])}) ---")
    for q in article['quotations']:
        s, e = q.get('charStart', -1), q.get('charEnd', -1)
        snippet = text[s:e] if s != -1 and e != -1 else '<no offset>'
        print(f"  [{q.get('quoteID')}] speaker={q.get('globalTopSpeaker')} | {snippet[:80]!r}")
    print(f"\n--- Names ({len(article['names'])}) ---")
    for name in article['names']:
        spans = name['char_offsets']
        snippets = [text[s:e] for s, e in spans if s != -1 and e != -1]
        print(f"  {name['name']!r}: {spans} → {snippets}")


if __name__ == "__main__":
    spark = start_spark()
    DATA_DIR = os.path.join(os.path.expanduser("~"), "data")
    quotegraph_articles = spark.read.parquet(f"{DATA_DIR}/quotegraph_articles.parquet")
    
    quotegraph_articles.printSchema()
    
    quotegraph = spark.read.parquet(f'{DATA_DIR}/quotegraph.parquet')
    
    # quotegraph_quotes = set(quotegraph.select('quoteID').toPandas()['quoteID'])
    quotegraph_pandas = quotegraph.select('quoteID', 'speaker', 'target').toPandas()
    quotegraph_tuples = zip(quotegraph_pandas['quoteID'], quotegraph_pandas['speaker'], quotegraph_pandas['target'])
    quotegraph_dict = {q: (s, t) for q, s, t in tqdm(quotegraph_tuples)} 
    
    test_batch = quotegraph_articles.sample(0.05).limit(10).toPandas()
    test_batch = json.loads(test_batch.to_json(orient='records'))
    
    for article in test_batch:
        article['quotations'] = [quotation for quotation in article['quotations'] if 
        quotation['quoteID'] in quotegraph_dict]

        for quotation in article['quotations']:
            edge = quotegraph_dict[quotation['quoteID']]
            speaker_label, speaker_description = fetch_wikidata(edge[0])
            target_label, target_description = fetch_wikidata(edge[1])
            quotation['edge'] = {
                'speaker': {
                    'qid': edge[0],
                    'wikidata_label': speaker_label,
                    'wikidata_description': speaker_description,
                },
                'target': {
                    'qid': edge[1],
                    'wikidata_label': target_label,
                    'wikidata_description': target_description
                }
            }
        
    save_json(test_batch, f'{DATA_DIR}/test_batch.json')
    


    test_article = next((a for a in test_batch if a.get('names')), test_batch[0])
    test_article
    
    result = detokenize_and_match_offsets(test_article)
    print_highlighted(result)