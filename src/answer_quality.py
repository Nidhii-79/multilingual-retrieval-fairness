"""Create blinded human-review sheets and summarize completed reviews."""
import pandas as pd

def make_review_sheet(frames, top_passages, n_per_language=25, seed=42):
    """top_passages maps language -> list of retrieved passage strings, one per query."""
    public_rows, private_rows=[],[]
    rng=__import__('numpy').random.default_rng(seed)
    for lang,df in frames.items():
        count=min(n_per_language,len(df)); chosen=sorted(rng.choice(len(df),size=count,replace=False).tolist())
        for qid in chosen:
            r=df.iloc[qid]
            public_rows.append({"language":lang,"query_id":qid,"question":r.question,
                "choice_1":r.mc_answer1,"choice_2":r.mc_answer2,"choice_3":r.mc_answer3,"choice_4":r.mc_answer4,
                "retrieved_passage":top_passages[lang][qid],"evidence_sufficient_0_or_1":"",
                "reviewer_answer_number_1_to_4":"","reviewer_id":"","notes":""})
            private_rows.append({"language":lang,"query_id":qid,"gold_answer_number":str(r.correct_answer_num)})
    return pd.DataFrame(public_rows),pd.DataFrame(private_rows)

def summarize_review(completed_csv, private_key_csv):
    review=pd.read_csv(completed_csv); key=pd.read_csv(private_key_csv)
    merged=review.merge(key,on=["language","query_id"],how="left")
    for c in ("evidence_sufficient_0_or_1","reviewer_answer_number_1_to_4","gold_answer_number"):
        merged[c]=pd.to_numeric(merged[c],errors="coerce")
    merged["answer_correct"]=(merged["reviewer_answer_number_1_to_4"]==merged["gold_answer_number"]).astype(float)
    return merged.groupby("language").agg(n_reviewed=("query_id","count"),
        evidence_sufficient_rate=("evidence_sufficient_0_or_1","mean"),
        answer_accuracy_from_retrieved_passage=("answer_correct","mean")).reset_index()
