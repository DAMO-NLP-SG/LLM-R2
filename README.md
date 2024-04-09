
<div align="center">

**LLM-R2: A Large Language Model Enhanced Rule-based Rewrite System for Boosting Query Efficiency**

------

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Installation">Installation</a> •
  <a href="#Datasets">Datasets</a> •
  <a href="#Training and inference">Training and inference</a> •
</p>

</div>


## Overview 
**LLM-R2**  Query rewrite, which aims to generate more efficient queries by altering a SQL query’s structure without changing the query result, has been an important research problem. In order to maintain equivalence between the rewritten query and the original one during rewriting, traditional query rewrite methods always rewrite the queries following certain rewrite rules. However, some problems still remain. Firstly, existing methods of finding the optimal choice or sequence of rewrite rules are still limited and the process always costs a lot of resources. Methods involving discovering new rewrite rules typically require complicated proofs of structural logic or extensive user interactions. Secondly, current query rewrite methods usually rely highly on DBMS cost estimators which are often not accurate. In this paper, we address these problems by proposing a novel method of query rewrite named LLM-R2, adopting a large language model (LLM) to propose possible rewrite rules for a database rewrite system. To further improve the inference ability of LLM in recommending rewrite rules, we train a contrastive model by curriculum to learn query representations and select effective query demonstrations for the LLM. Experimental results have shown that our method can significantly improve the query executing efficiency and outperform the baseline methods. In addition, our method enjoys high robustness across different datasets.



</div>

## Installation

**PostgreSQL requirement: PostgreSQL 14.4**

**Java requirement: JDK>=11**

**Python requirement: Python>=3.8**

Before running the project, install the Python dependencies by: ```pip install -r requirement.txt```


## Datasets

We used three datasets in this work and you can download the datasets from the following links:

**TPC-H:** https://github.com/gregrahn/tpch-kit

**IMDB:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2QYZBT

**DSB:** https://github.com/microsoft/dsb


## Training and inference

To train your own demonstration selector, use the trainer ```bash src/run_CLTrain.sh```

If you already have a trained demonstration selector in ```src/simcse_models/```, simply run ```python src/LLM_R2.py``` at inference time
