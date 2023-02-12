# Quotegraph
Quotegraph is a large heterogeneous social network respresented as a directed graph extracted from quotations in [Quotebank](https://zenodo.org/record/4277311). Edges point from the speaker of a quotation to a person mentioned in that quotation. Person names in Quotebank are linked to Wikidata entities, which allows for studying roles of different person attributes in the interactions emerging in Quotebank. Quotegraph boasts 528 thousand nodes and 8.6 million edges, which makes it suitable for a large-scale analysis of speaker ineractions in news articles.

Quotegraph was built as a part of a MS thesis project and was used to investigate the patterns in the use of personal reference expressions in public discourse. The thesis is in [Quotegraph_report.pdf](https://github.com/epfl-dlab/quotegraph/blob/main/Quotegraph_report.pdf). In the document, you can find the details on Quotegraph building (chapter 3), and Quotegraph's structural properties (chapter 4).

Quotegraph is currently not publicly available. You can find the dataset at `/dlabdata1/culjak/quotegraph.parquet` on dlab's internal nodes. Below is the schema of the dataset:
```
 |-- quoteID: string 
 |-- speaker: string
 |-- target: string
 |-- quotation: string
```
- `quoteID` - Primary key of the quotation (format: "YYYY-MM-DD-{increasing int:06d}"). Can be used for joining with Quotebank data.
- `speaker` - Most likely Wikidata QID of the most likely speaker of the quotation. Can be used for joining with Wikidata to extract the speaker's attributes.
- `target` - Most likely Wikidata QID of a mention appearing in the quotation
- `quotation` - Text of the quotation

## Caveats
### Speaker-to-mention sentiment analysis
As a part of the thesis, we explored research questions focusing on speaker-to-mention sentiment. However, without a sophisticated large-scale targeted sentiment analysis tool, we cannot be sure whether the overall sentiment of the quotation in which a person is mentioned truly is the sentiment of the quotation's speaker towards the mentioned person. We can merely measure the correlation of positive/negative words and mentions of certain persons in the quotations. 

### Quotebank issues
Since it is derived from Quotebank, Quotegraph inherits all Quotebank's shortcomings, including the imperfect speaker attribution [1] and disambiguation [2]. Due to encoding issues, those shortcomings are further emphasized in the earlier phases of Quotebank (A-C). The data from the phases A-C should therefore be analyzed with great caution.

## Open research questions
1. When a prominent speakers mention someone, does that person get more quoted or mentioned in other quotes? Can we identify leaders and their followers by investigating the patterns of person mentions?
2. Who are the hubs and the authorities in Quotegraph and what are their properties?
3. What are the properties of non-reciprocated interactions? 
4. What are the properties of the communities detected in Quotegraph?

## References
[1] Timoté Vaucher, Andreas Spitz, Michele Catasta, and Robert West. “Quotebank: A Corpus of Quotations from a Decade of News”. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining. 2021.

[2] Marko Čuljak, Andreas Spitz, Robert West, Akhil Arora. “Strong Heuristics for Named Entity Linking”. In Proceedings 2022 Conference of the North American Chapter of the Association for Compuational Linguistics: Student Research Workshop.
