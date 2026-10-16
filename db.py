import sqlite3
import pandas as pd

DB_PATH = "sentilytics_v2.db"
COLUMNS = ["student_id", "year", "question", "rating", "teaching", "facilities", "materials", "clarity",
           "comment", "sentiment", "confidence", "emotion", "aspect", "created_at"]


def conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, year TEXT, question TEXT, rating REAL,
            teaching REAL, facilities REAL, materials REAL, clarity REAL, comment TEXT, sentiment TEXT,
            confidence REAL, emotion TEXT, aspect TEXT, created_at TEXT)""")


def insert_df(df):
    df = df.copy()
    for col in COLUMNS:
        if col not in df:
            df[col] = pd.Timestamp.now() if col == "created_at" else None
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    with conn() as c:
        df[COLUMNS].to_sql("feedback", c, if_exists="append", index=False)


def fetch_all():
    with conn() as c:
        df = pd.read_sql("SELECT * FROM feedback", c)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def clear():
    with conn() as c:
        c.execute("DELETE FROM feedback")
