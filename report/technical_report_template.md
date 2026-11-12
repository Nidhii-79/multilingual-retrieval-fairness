# Multilingual Retrieval Fairness: Technical Report

## 1. Problem and research question
A multilingual public-services assistant should retrieve useful evidence across English, Hindi and Malayalam. This mini-project measures retrieval differences across these languages and tests whether a hybrid retriever changes those differences.

## 2. Dataset and preprocessing
- Dataset: Facebook Belebele, test split
- Configurations: `eng_Latn`, `hin_Deva`, `mal_Mlym`

## 3. Methods
Compare BM25, `intfloat/multilingual-e5-small`, and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Evaluate English-English, Hindi-Hindi, Malayalam-Malayalam, Hindi-English and Malayalam-English retrieval. Metrics: Recall@1/5/10 and MRR. Hybrid mitigation: BM25 + E5 using RRF with k=60.


