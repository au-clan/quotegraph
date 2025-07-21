# Quotegraph
Quotegraph is a large social network represented as a directed graph extracted from quotations in [Quotebank](https://zenodo.org/record/4277311). Edges point from the speaker of a quotation to a person mentioned in that quotation. The names of the actors are linked to Wikidata using [quotebank-toolkit](https://github.com/epfl-dlab/quotebank-toolkit). Quotegraph boasts 528 thousand nodes and 8.6 million edges, which makes it suitable for a large-scale analysis of speaker interactions in news articles. The data is available at https://zenodo.org/records/16275215.

Preprint is coming soon...

## Caveats
### Quotebank issues
Since it is derived from Quotebank, Quotegraph inherits all Quotebank's shortcomings, including the imperfect speaker attribution [1] and disambiguation [2]. Due to encoding issues, those shortcomings are further emphasized in the earlier phases of Quotebank (A-C). The data from the phases A-C should therefore be analyzed with great caution. Still, those shortcomings are alleviated owing to the scale and redundancy, i.e., each quotation appearing across multiple sources as errors of the speaker attribution and disambiguation modules get mitigated after aggregating their predictions over instances of the same quotation.

### Size
Since Quotegraph is a fairly large network, it necessitates the use of efficient network analysis tools such as [networkit](https://networkit.github.io/), [graph-tool](https://graph-tool.skewed.de/), or [snap](http://snap.stanford.edu/snappy/index.html).

## References
[1] Timoté Vaucher, Andreas Spitz, Michele Catasta, and Robert West. “Quotebank: A Corpus of Quotations from a Decade of News”. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining. 2021.

[2] Marko Čuljak, Andreas Spitz, Robert West, Akhil Arora. “Strong Heuristics for Named Entity Linking”. In Proceedings 2022 Conference of the North American Chapter of the Association for Compuational Linguistics: Student Research Workshop.
