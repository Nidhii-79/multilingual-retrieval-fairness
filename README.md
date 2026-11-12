# Multilingual Retrieval Fairness (English, Hindi, Malayalam)

A compact, notebook-centred mini-project comparing BM25 and two multilingual embedding retrievers, then testing BM25 + E5 with Reciprocal Rank Fusion (RRF).

## Research question
How does retrieval quality differ across English, Hindi and Malayalam, and can a lightweight hybrid retriever reduce the gap?

## Project structure
```text
notebooks/Interactive_Multilingual_Retrieval_Demo_PySpark_Reordered.ipynb  # end-to-end experiment
src/data.py                                      # dataset loading and passage indexing
src/metrics.py                                  # Recall@k, MRR, bootstrap CI
src/retrieval.py                                # BM25 and dense retrieval helpers
src/evaluation.py                               # comparable evaluation rows
src/hybrid.py                                   # RRF fusion
src/answer_quality.py                           # blinded human review workflow
tests/                                           # small unit tests
report/technical_report_template.md             # report scaffold
results/                                         # generated outputs
```

## Environment setup (once per machine)
Python 3.10+ recommended. From the repository root:
```bash
python -m venv .venv
# Windows CMD: .venv\Scripts\activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
If the virtual environment and packages already exist, do not recreate or reinstall them for each commit.

## Run the experiment
Open `notebooks/multilingual_retrieval_fairness.ipynb` in Jupyter or upload it to Colab. Run all cells from top to bottom. Set `MAX_ITEMS = 60` for a smoke test and `MAX_ITEMS = None` for the full test split. The first run downloads Belebele and the compact embedding models; later runs generally reuse local caches. GPU is helpful but not mandatory.

## Methods
- Dataset: `facebook/belebele`, configurations `eng_Latn`, `hin_Deva`, `mal_Mlym`, test split.
- Baselines: BM25, `intfloat/multilingual-e5-small`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Conditions: English→English, Hindi→Hindi, Malayalam→Malayalam, Hindi→English, Malayalam→English.
- Mitigation: BM25 + E5 via RRF (k=60).
- Metrics: Recall@1/5/10, MRR, English-minus-language Recall@5 gap, bootstrap interval, measured runtime.
- Human review: blinded sample assessing whether retrieved text provides sufficient evidence and whether a reviewer can select the correct answer using only that text.

## Outputs
The notebook writes CSV tables, a Recall@5 plot, environment metadata, a blinded review template, a private answer key and a results summary to `results/`. 
## Important scope and interpretation
Belebele is a multilingual reading-comprehension benchmark, not a public-services corpus. Its designated passage is used as the relevant document. Retrieval recall is not the same as answer accuracy. Human review is a separate, small-sample assessment. 
## Team division
- Nidhi S: data validation, baseline retrieval, metrics, language-gap analysis and figures. See `MEMBER_A_PUSH_CHECKLIST.md`.
- Sanjana V Shastry: RRF hybrid, human review workflow, bootstrap analysis, model/data documentation and report. See `MEMBER_B_PUSH_CHECKLIST.md`.

## Tests
```bash
pytest -q
```
