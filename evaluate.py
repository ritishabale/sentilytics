"""Model comparison: VADER vs TF-IDF+LogReg vs RoBERTa on comments YOU labelled by hand.
1) python make_label_sheet.py   2) fill the 'label' column in data/label_me.csv   3) python evaluate.py"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from ai.sentiment import vader_predict, _load as load_roberta

path = sys.argv[1] if len(sys.argv) > 1 else "data/label_me.csv"
df = pd.read_csv(path, encoding="utf-8-sig")
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["positive", "neutral", "negative"])]
if len(df) < 20:
    sys.exit(f"Only {len(df)} labelled rows in {path}. Fill the 'label' column (positive/neutral/negative) for at least 40-80 comments first.")
print(f"{len(df)} labelled comments: {df['label'].value_counts().to_dict()}")
strat = df["label"] if df["label"].value_counts().min() >= 2 else None
Xtr, Xte, ytr, yte = train_test_split(df["comment"], df["label"], test_size=0.3, random_state=42, stratify=strat)
LABELS = ["negative", "neutral", "positive"]
preds = {"VADER": [p[0] for p in vader_predict(Xte.tolist())]}
lr = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
preds["TF-IDF + Logistic Regression"] = lr.fit(Xtr, ytr).predict(Xte)
rob = load_roberta()
if rob is not None:
    preds["RoBERTa (pretrained)"] = [r["label"].lower() for r in rob(Xte.tolist(), batch_size=16, truncation=True, max_length=512)]
else:
    print("RoBERTa not available (FAST mode or no internet) - skipping it.")

rows = []
fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4))
axes = [axes] if len(preds) == 1 else axes
for ax, (name, p) in zip(axes, preds.items()):
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="macro", zero_division=0)
    rows.append({"Model": name, "Accuracy": accuracy_score(yte, p), "Precision": pr, "Recall": rc, "F1 (macro)": f1})
    ConfusionMatrixDisplay(confusion_matrix(yte, p, labels=LABELS), display_labels=LABELS).plot(ax=ax, colorbar=False)
    ax.set_title(name, fontsize=10)
plt.tight_layout()
plt.savefig("results/confusion_matrices.png", dpi=150)
res = pd.DataFrame(rows).round(3)
res.to_csv("results/model_comparison.csv", index=False)
print(res.to_string(index=False))
print(f"\nTest set size: {len(yte)}. Saved results/model_comparison.csv and results/confusion_matrices.png")
