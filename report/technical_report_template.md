# Multilingual Retrieval Fairness: Technical Report

> Replace every bracketed field with values from the completed experiment. Do not invent scores or claim a mitigation worked unless the results support it.

## 1. Problem and research question
A multilingual public-services assistant should retrieve useful evidence across English, Hindi and Malayalam. This mini-project measures retrieval differences across these languages and tests whether a hybrid retriever changes those differences.

## 2. Dataset and preprocessing
- Dataset: Facebook Belebele, test split
- Configurations: `eng_Latn`, `hin_Deva`, `mal_Mlym`
- Rows per language: [actual count]
- Unique passages per language: [actual count]
- Alignment validation: [describe identifier and passage-group checks]
- Important scope note: Belebele is a multilingual reading-comprehension benchmark, not a public-services corpus.

## 3. Methods
Compare BM25, `intfloat/multilingual-e5-small`, and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Evaluate English-English, Hindi-Hindi, Malayalam-Malayalam, Hindi-English and Malayalam-English retrieval. Metrics: Recall@1/5/10 and MRR. Hybrid mitigation: BM25 + E5 using RRF with k=60.

## 4. Results
Insert actual baseline and hybrid tables from `results/`. Report Recall@5 gaps as English mono Recall@5 minus Hindi/Malayalam mono Recall@5. Include confidence intervals and measured runtime/hardware.

## 5. Human evidence review
- Number reviewed per language: [actual count]
- Reviewer count: [actual count]
- Evidence-sufficiency rate: [actual result]
- Answer accuracy based only on retrieved passage: [actual result]
- Limitations: [reviewer subjectivity, sample size, etc.]

## 6. Discussion, limitations and conclusion
Discuss which language/system combinations differ, whether hybrid retrieval changes the gap, and whether observed differences are practically meaningful. State that benchmark results do not establish performance on real government-service queries. Include hardware and model limitations.
