import os
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

import db
from ai.pipeline import analyze_dataframe, analyze_one, engine_name
from ai.summarizer import summarize, top_keywords, top_recurring
from forms_import import CLR_OPTIONS, SAT, SAT_OPTIONS, CLR, anon_id, parse_form

st.set_page_config(page_title="Sentilytics", page_icon="📊", layout="wide")
db.init_db()

ADMIN_PASSWORD = os.getenv("SENTILYTICS_ADMIN_PASSWORD", "admin123")
COLORS = {"positive": "#2ca02c", "neutral": "#9aa0a6", "negative": "#d62728"}
YEARS = ["First Year", "Second Year", "Third Year", "Fourth Year"]
SCORES = {"teaching": "Teaching quality", "clarity": "Clarity of explanation", "materials": "Study materials",
          "facilities": "Facilities & infrastructure", "rating": "Overall experience"}
DATA_FILE = "data/student_feedback.csv"


@st.cache_resource(show_spinner="Loading AI models (first run downloads them - please wait)...")
def warm_up():
    analyze_one("warm up")
    return engine_name()


def import_and_save(raw):
    long_df, found, text_cols = parse_form(raw)
    st.caption(f"Detected {len(raw)} respondents → {len(long_df)} written comments "
               f"({', '.join(text_cols)}).")
    bar = st.progress(0.0, text="Analyzing comments with AI models...")
    out = analyze_dataframe(long_df, lambda p: bar.progress(p, text=f"Analyzing comments... {p:.0%}"))
    db.insert_df(out)
    bar.empty()
    return out


# ---------------- sidebar ----------------
st.sidebar.title("📊 Sentilytics")
st.sidebar.caption(f"AI engine: **{warm_up()}**")
st.session_state.setdefault("admin", False)
if not st.session_state.admin:
    with st.sidebar.expander("🔐 Admin login"):
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if pw == ADMIN_PASSWORD:
                st.session_state.admin = True
                st.rerun()
            else:
                st.error("Wrong password")
else:
    st.sidebar.success("Logged in as admin")
    if st.sidebar.button("Logout"):
        st.session_state.admin = False
        st.rerun()

pages = ["Submit Feedback", "Bulk Upload"] + (["Dashboard", "Insights"] if st.session_state.admin else [])
page = st.sidebar.radio("Menu", pages)

if st.session_state.admin:
    st.sidebar.divider()
    if st.sidebar.button("Load project dataset (500 responses)"):
        import_and_save(pd.read_csv(DATA_FILE, encoding="utf-8-sig"))
        st.sidebar.success("Dataset loaded")
        st.rerun()
    if st.sidebar.button("Clear all data"):
        db.clear()
        st.rerun()

# ---------------- Submit Feedback ----------------
if page == "Submit Feedback":
    st.title("Submit Feedback")
    st.caption("Anonymous - no name or email is stored.")
    with st.form("fb"):
        year = st.selectbox("Year of study", YEARS)
        c1, c2 = st.columns(2)
        teaching = c1.select_slider("Quality of teaching", SAT_OPTIONS, value="neutral")
        facilities = c1.select_slider("Facilities & infrastructure", SAT_OPTIONS, value="neutral")
        materials = c2.select_slider("Study materials", SAT_OPTIONS, value="neutral")
        clarity = c2.select_slider("How clearly do teachers explain?", CLR_OPTIONS, value="average")
        rating = st.slider("Overall college experience (1-5)", 1, 5, 3)
        likes = st.text_area("What do you like most about your college?")
        problems = st.text_area("What is the biggest problem you face as a student?")
        improve = st.text_area("What should be improved in your college?")
        ok = st.form_submit_button("Submit")
    if ok:
        answers = {"Likes": likes, "Problems": problems, "Improvements": improve}
        answers = {k: v.strip() for k, v in answers.items() if v.strip()}
        if not answers:
            st.warning("Please write at least one comment.")
        else:
            sid = anon_id(pd.Timestamp.now().isoformat())
            rows = [dict(student_id=sid, year=year, question=q, rating=rating, teaching=SAT[teaching], facilities=SAT[facilities],
                         materials=SAT[materials], clarity=CLR[clarity], comment=t, **analyze_one(t)) for q, t in answers.items()]
            db.insert_df(pd.DataFrame(rows))
            st.success("Thank you! Feedback saved. Here is what the AI understood:")
            st.dataframe(pd.DataFrame(rows)[["question", "comment", "sentiment", "confidence", "emotion", "aspect"]], width="stretch")

# ---------------- Bulk Upload ----------------
elif page == "Bulk Upload":
    st.title("Bulk Upload (Google Forms CSV)")
    st.write("Upload the responses CSV exported from your Google Form (a .zip containing the CSV also works).")
    f = st.file_uploader("Upload CSV", type=["csv", "zip"])
    if f:
        raw = pd.read_csv(f, encoding="utf-8-sig", compression="zip" if f.name.endswith(".zip") else "infer")
        st.write(f"**{len(raw)} rows, {len(raw.columns)} columns**")
        st.dataframe(raw.head(), width="stretch")
        if st.button("Analyze & save"):
            try:
                out = import_and_save(raw)
                st.success(f"Saved {len(out)} analyzed comments. Open the Dashboard.")
                st.dataframe(out[["year", "question", "comment", "sentiment", "emotion", "aspect"]].head(50), width="stretch")
            except ValueError as e:
                st.error(str(e))

# ---------------- Dashboard / Insights ----------------
else:
    df = db.fetch_all()
    if df.empty:
        st.title(page)
        st.info("No data yet. Click **Load project dataset** in the sidebar, or use Bulk Upload.")
        st.stop()
    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    sel_y = st.sidebar.multiselect("Year of study", sorted(df.year.dropna().unique()))
    sel_q = st.sidebar.multiselect("Question", sorted(df.question.unique()))
    if sel_y:
        df = df[df.year.isin(sel_y)]
    if sel_q:
        df = df[df.question.isin(sel_q)]
    if df.empty:
        st.warning("No rows for this filter.")
        st.stop()
    people = df.drop_duplicates("student_id")  # one row per respondent for rating-based charts

    if page == "Dashboard":
        st.title("Dashboard")
        k = st.columns(5)
        k[0].metric("Respondents", people.student_id.nunique())
        k[1].metric("Comments analyzed", len(df))
        k[2].metric("Avg overall rating", f"{people.rating.mean():.2f} / 5")
        k[3].metric("% Positive comments", f"{(df.sentiment == 'positive').mean():.0%}")
        k[4].metric("% Negative comments", f"{(df.sentiment == 'negative').mean():.0%}")

        # attention alerts
        neg = df[df.sentiment == "negative"]
        if len(neg):
            share = (neg.aspect.value_counts(normalize=True) * 100).round(0)
            st.warning(f"🚨 Biggest source of negative feedback: **{share.index[0]}** ({share.iloc[0]:.0f}% of negative comments)")
        avgs = {v: people[c].mean() for c, v in SCORES.items() if c != "rating" and people[c].notna().any()}
        if avgs:
            low = min(avgs, key=avgs.get)
            st.warning(f"🚨 Lowest-rated area: **{low}** (average {avgs[low]:.2f} / 5)")
        by_year = df.groupby("year").sentiment.apply(lambda s: (s == "negative").mean())
        if len(by_year) > 1 and by_year.max() - by_year.mean() >= 0.05:
            st.warning(f"🚨 **{by_year.idxmax()}** students write the most negative comments ({by_year.max():.0%} vs {df.sentiment.eq('negative').mean():.0%} overall)")

        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(df, names="sentiment", title="Overall comment sentiment", color="sentiment", color_discrete_map=COLORS, hole=0.4), width="stretch")
        c2.plotly_chart(px.histogram(df, x="question", color="sentiment", barmode="stack", color_discrete_map=COLORS, title="Sentiment by question"), width="stretch")

        c3, c4 = st.columns(2)
        sc = people.groupby("year")[list(SCORES)].mean().rename(columns=SCORES).reset_index().melt("year", var_name="Area", value_name="Avg score (1-5)")
        c3.plotly_chart(px.bar(sc, x="Area", y="Avg score (1-5)", color="year", barmode="group", title="Average satisfaction by area and year"), width="stretch")
        c4.plotly_chart(px.histogram(df, x="year", color="sentiment", barmode="stack", color_discrete_map=COLORS,
                                     category_orders={"year": YEARS}, title="Comment sentiment by year of study"), width="stretch")

        c5, c6 = st.columns(2)
        ct = pd.crosstab(df.aspect, df.sentiment)
        c5.plotly_chart(px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn_r", title="Aspect × sentiment heatmap"), width="stretch")
        emo = df.emotion.value_counts().reset_index()
        emo.columns = ["emotion", "count"]
        c6.plotly_chart(px.bar(emo, x="emotion", y="count", color="emotion", title="Emotion distribution"), width="stretch")

        if df.created_at.dt.date.nunique() >= 3:
            trend = df.assign(week=df.created_at.dt.to_period("W").dt.start_time).groupby(["week", "sentiment"]).size().reset_index(name="count")
            st.plotly_chart(px.line(trend, x="week", y="count", color="sentiment", color_discrete_map=COLORS, markers=True, title="Weekly sentiment trend"), width="stretch")
        else:
            st.caption("Trend chart appears automatically when responses span 3+ different days.")

        st.subheader("Word clouds")
        w1, w2 = st.columns(2)
        try:
            from wordcloud import WordCloud
            for col, label, cmap, mask in [(w1, "Negative comments", "Reds", df.sentiment == "negative"), (w2, "Positive comments", "Greens", df.sentiment == "positive")]:
                text = " ".join(df[mask].comment)
                col.write(f"**{label}**")
                if text.strip():
                    col.image(WordCloud(width=700, height=350, background_color="white", colormap=cmap).generate(text).to_array())
        except Exception as e:
            st.caption(f"Word cloud unavailable: {e}")
        with st.expander("Raw data"):
            st.dataframe(df.sort_values("created_at", ascending=False), width="stretch")

    else:  # Insights
        st.title("Insights")
        neg = df[df.sentiment == "negative"]
        pos = df[df.sentiment == "positive"]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔴 Top complaints")
            st.write("**AI summary:**", summarize(neg.comment.tolist()))
            st.write("**Keywords:**", ", ".join(top_keywords(neg.comment.tolist())) or "n/a")
            st.write("**Most repeated complaints:**")
            st.dataframe(pd.DataFrame(top_recurring(neg.comment.tolist()), columns=["Complaint", "Mentions"]), width="stretch", hide_index=True)
            st.write("**Negative comments by aspect:**")
            st.dataframe(neg.aspect.value_counts().rename("count"), width="stretch")
        with c2:
            st.subheader("🟢 Top praises")
            st.write("**AI summary:**", summarize(pos.comment.tolist()))
            st.write("**Keywords:**", ", ".join(top_keywords(pos.comment.tolist())) or "n/a")
            st.write("**Most repeated praises:**")
            st.dataframe(pd.DataFrame(top_recurring(pos.comment.tolist()), columns=["Praise", "Mentions"]), width="stretch", hide_index=True)
            st.write("**Positive comments by aspect:**")
            st.dataframe(pos.aspect.value_counts().rename("count"), width="stretch")

        st.subheader("Scorecard by year of study")
        card = people.groupby("year")[list(SCORES)].mean().rename(columns=SCORES).round(2)
        card["Respondents"] = people.groupby("year").size()
        st.dataframe(card, width="stretch")

        st.subheader("Rating vs. comment mismatch")
        mm = df[((df.rating >= 4) & (df.sentiment == "negative")) | ((df.rating <= 2) & (df.sentiment == "positive"))]
        st.caption(f"{len(mm)} of {len(df)} comments disagree with the student's own overall rating (4-5 stars but negative text, or 1-2 stars but positive text). "
                   "Useful for spotting sarcasm, careless ratings or data-quality problems.")
        st.dataframe(mm[["year", "question", "rating", "comment", "sentiment"]].head(200), width="stretch")

        report = (f"# Sentilytics Report\n\nRespondents: {people.student_id.nunique()}  \nComments analyzed: {len(df)}  \n"
                  f"Average overall rating: {people.rating.mean():.2f}/5\n\n## Complaints\n{summarize(neg.comment.tolist())}\n\n"
                  f"Keywords: {', '.join(top_keywords(neg.comment.tolist()))}\n\n## Praises\n{summarize(pos.comment.tolist())}\n\n"
                  f"Keywords: {', '.join(top_keywords(pos.comment.tolist()))}\n\n## Scorecard by year\n{card.to_string()}\n")
        st.download_button("⬇ Download report", report, "sentilytics_report.md")
