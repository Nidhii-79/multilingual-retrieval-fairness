# Model and Data Card

## Dataset
- **Name:** Belebele (`facebook/belebele`)
- **Configurations:** `eng_Latn`, `hin_Deva`, `mal_Mlym`
- **Split:** test
- **Task:** multilingual reading-comprehension questions paired with passages and four answer options
- **Use in this project:** each question's associated passage is treated as the designated relevant retrieval document
- **Known limitation:** Belebele is not a public-services dataset. Results are evidence about this benchmark, not proof of equal performance on real government-service queries.
- **Preprocessing:** missing passage/question values are converted to empty strings; duplicate passages are indexed once; question-to-passage indices are retained; parallel repeated-passage grouping is checked.

## Retrieval systems

### BM25
Lexical sparse retrieval using Unicode-aware word tokenization. It is a non-neural baseline and may be sensitive to vocabulary/script mismatch.

### multilingual-e5-small
Model ID: `intfloat/multilingual-e5-small`. Sentence-transformers interface; query and passage prefixes are used as expected by E5. Embeddings are normalized and compared using dot product. Model card and licence should be checked at the model repository before redistribution.

### multilingual MiniLM
Model ID: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Compact multilingual sentence embedding baseline. Embeddings are normalized and compared using dot product. Model card and licence should be checked at the model repository before redistribution.

### Hybrid RRF
Combines BM25 and E5 rankings using Reciprocal Rank Fusion with `k=60`. It does not fine-tune either model.

## Intended use
A small academic comparison of retrieval behaviour across three benchmark language configurations. Not intended for deployment, public-service decision-making, or use as a safety-critical system.

## Evaluation
Recall@1/5/10, MRR, English-minus-language monolingual Recall@5 gaps, bootstrap confidence intervals, measured runtime and a separate small human evidence review.

## Risks and limitations
- Benchmark domain and question style may not represent real user queries.
- The designated passage is treated as the single relevant document by benchmark convention.
- Cross-language evaluation depends on verified parallel row and passage-group alignment.
- Model performance and latency depend on hardware, library versions and model revisions.
- Human review is small and subjective; reviewer count and protocol must be disclosed.
- The review answer key must be kept separate from the reviewer-facing sheet.

## Reporting checklist
Record the exact row counts, unique-passage counts, model IDs, package versions, hardware, run time, reviewer count and actual results. Do not fill result fields with estimates.
