# Datasets

Every graph is used **structurally only** — edges, plus node labels where they exist. Published node features are
ignored by design, so a gain cannot be credited to the attributes instead of to the rewiring under test. The registry
lives in `virgo/config.py`; the builders in `virgo/data/`.

## Role in the study

| group | datasets | role |
|---|---|---|
| discovery panel | cora, enzymes, ogbn_arxiv, ogbl_ddi, roman_empire, tolokers, questions | stage-1 rules fitted here |
| held-out (Module 3) | citeseer_linqs, proteins, pubmed, actor, minesweeper, amazon_photo, lastfm_asia, amazon_ratings, squirrel_filtered | pre-registered test of the stage-1 rules |
| held-out (Modules 5, 7) | reed98, amherst41, johnshopkins55, cornell5 | the two-stage framework run end to end |
| earlier batch | chameleon_filtered, texas, twitch_pt | measured before stage 1 existed; no pre-registered call |

Two registered graphs carry no scoreboard row: `citeseer` (the Identity2Vec author's graph, no aligned labels) and
`politics` (ships no labels). Both are link-prediction only.

The Facebook100 label is **gender, missing for ~10% of users** (LINKX codes it −1). Those nodes are left out of the
`.labels` file rather than written as a third class; the graph keeps every node.

## Sources

| dataset(s) | domain | reference |
|---|---|---|
| `cora`, `citeseer`, `citeseer_linqs`, `pubmed` | paper citation | Sen et al., *Collective Classification in Network Data*, AI Magazine 29(3), 2008. Cora also McCallum et al., *Information Retrieval* 3(2):127–163, 2000; PubMed also Namata et al., MLG workshop, 2012 |
| `enzymes`, `proteins` | protein structure | Borgwardt et al., *Bioinformatics* 21(suppl_1):i47–i56, 2005. PROTEINS also Dobson & Doig, *J. Mol. Biol.* 330(4):771–783, 2003 |
| `politics` (rt-pol) | Twitter retweet | Rossi & Ahmed, *The Network Data Repository*, AAAI, 2015 |
| `ogbn_arxiv` | paper citation (MAG) | Hu et al., *Open Graph Benchmark*, NeurIPS, 2020; graph from Wang et al., *Microsoft Academic Graph*, QSS 1(1), 2020 |
| `ogbl_ddi` | drug–drug interaction | Hu et al., *Open Graph Benchmark*, NeurIPS, 2020; interactions from Wishart et al., *DrugBank 5.0*, NAR 46(D1), 2018 |
| `roman_empire`, `amazon_ratings`, `minesweeper`, `tolokers`, `questions`, `squirrel_filtered` | Wikipedia text · co-purchase · synthetic grid · crowdsourcing · Q&A · Wikipedia web | Platonov et al., *A Critical Look at the Evaluation of GNNs under Heterophily*, ICLR, 2023. `squirrel_filtered` is their de-duplicated rebuild of the graph of Rozemberczki, Allen & Sarkar, *J. Complex Networks* 9(2), 2021 |
| `actor` | film co-occurrence | Pei et al., *Geom-GCN*, ICLR, 2020; induced from Tang et al., KDD, 2009 |
| `amazon_photo` | co-purchase | Shchur et al., *Pitfalls of GNN Evaluation*, NeurIPS R2L workshop, 2018; from McAuley et al., SIGIR, 2015 |
| `lastfm_asia` | music social network | Rozemberczki & Sarkar, *Characteristic Functions on Graphs*, CIKM, 2020 |
| `reed98`, `amherst41`, `johnshopkins55`, `cornell5` | Facebook100 college social | Lim et al., *Large Scale Learning on Non-Homophilous Graphs*, NeurIPS, 2021; networks from Traud, Mucha & Porter, *Physica A* 391(16), 2012 |

**How they were obtained.** `cora`, `citeseer`, `enzymes`, `proteins` and `politics` come from the Identity2Vec
author's `input.zip` (Network Repository files); their labels are rebuilt from LINQS (`cora`, `citeseer_linqs`) or
Network Repository (`enzymes`, `proteins`). `pubmed`, `actor`, `amazon_photo` and the four LINKX graphs are built
through `torch_geometric`; `lastfm_asia` from SNAP directly (PyG's host is dead); the six Platonov graphs from the
authors' `.npz` release; the two OGB graphs through `ogb`.
