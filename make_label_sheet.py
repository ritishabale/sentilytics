"""Creates data/label_me.csv - 80 random unique comments for YOU to label by hand (positive / neutral / negative).
This is the 'ground truth' used by evaluate.py for the model comparison."""
import pandas as pd
from forms_import import parse_form

raw = pd.read_csv("data/student_feedback.csv", encoding="utf-8-sig")
long_df, _, _ = parse_form(raw)
u = long_df.drop_duplicates("comment")
sample = pd.concat([g.sample(min(len(g), 27), random_state=1) for _, g in u.groupby("question")]).head(80)
sample[["question", "comment"]].assign(label="").to_csv("data/label_me.csv", index=False)
print("Wrote data/label_me.csv with", len(sample), "rows. Fill the 'label' column with positive / neutral / negative.")
