# Sentilytics - AI-Based Student Feedback & Sentiment Analytics Platform

Reads student feedback (Google Forms CSV), analyzes every written comment with AI models and shows an admin dashboard.

| Task | Model |
|---|---|
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` |
| Aspect (topic) detection | `facebook/bart-large-mnli` (zero-shot) |
| Summaries | `sshleifer/distilbart-cnn-12-6` |
| Keywords | TF-IDF |
| Baselines for comparison | VADER, TF-IDF + Logistic Regression |

## How to run (Windows + VS Code)
1. Install **Python 3.10 or 3.11** from python.org (tick "Add Python to PATH"). Unzip this folder, open it in VS Code (File > Open Folder).
2. Open a terminal in VS Code (Ctrl + `) and run:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
   Mac/Linux: `source venv/bin/activate` instead of the activate line.
3. Browser opens at http://localhost:8501.
4. Sidebar > **Admin login** > password `admin123` > click **Load project dataset (500 responses)**.
5. Open **Dashboard** and **Insights**.

**First run needs internet** (downloads ~2.5 GB of models once) and the dataset analysis takes several minutes on CPU. Results are cached in `.analysis_cache.pkl`, so the next load is instant. Do this the day before your presentation.

Too slow / low RAM? Run with lighter settings (quality is lower - don't use this for the final demo):
- skip only the slowest model: `set SENTILYTICS_ZEROSHOT=0` then `streamlit run app.py`
- skip all big models: `set SENTILYTICS_FAST=1` (Mac/Linux: `export ...`)

The sidebar always shows which AI engine is active.

## Model comparison slide
```
python make_label_sheet.py        # creates data/label_me.csv with 80 comments
```
Open `data/label_me.csv` in Excel, fill the `label` column yourself with positive / neutral / negative, save, then:
```
python evaluate.py
```
Creates `results/model_comparison.csv` and `results/confusion_matrices.png` for your PPT.

## Files
app.py (UI) | db.py (SQLite) | forms_import.py (reads your Google Form CSV) | ai/ (models) | evaluate.py | data/student_feedback.csv (your responses)

## Demo flow (5 minutes)
1. Problem statement and architecture (slide).
2. **Submit Feedback** page: type 2-3 live comments (one positive, one negative, one sarcastic) and show the AI's sentiment/emotion/aspect.
3. **Dashboard**: KPIs, alerts, sentiment by question and year, aspect heatmap, word clouds.
4. **Insights**: AI summary, most repeated complaints, scorecard by year, rating-vs-comment mismatch table.
5. Model comparison slide, then limitations and future scope.

## Limitations (say these yourself - it earns marks)
English models handle Hinglish and sarcasm poorly; the form has no course/teacher column so analysis is by year of study; many students wrote similar comments; star ratings and comment text do not always agree.
