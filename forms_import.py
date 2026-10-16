"""Converts the Google Forms response CSV into one row per written comment (long format)."""
import hashlib
import pandas as pd

SAT = {"very dissatisfied": 1, "dissatisfied": 2, "neutral": 3, "satisfied": 4, "very satisfied": 5}
CLR = {"very poorly": 1, "poorly": 2, "average": 3, "clearly": 4, "very clearly": 5}
SAT_OPTIONS = list(SAT)
CLR_OPTIONS = list(CLR)
TEXT_QUESTIONS = {"Likes": ["like most"], "Problems": ["biggest problem", "difficulty"], "Improvements": ["should be improved", "improved in"]}


def anon_id(s):
    return "S-" + hashlib.sha256(str(s).encode()).hexdigest()[:8]


def _find(cols, *keys, exclude=()):
    for c in cols:
        lc = c.lower()
        if all(k in lc for k in keys) and not any(x in lc for x in exclude):
            return c
    return None


def parse_form(raw):
    raw = raw.copy()
    raw.columns = [" ".join(str(c).split()) for c in raw.columns]
    cols = list(raw.columns)
    found = {
        "time": _find(cols, "timestamp"), "year": _find(cols, "year of study"),
        "teaching": _find(cols, "quality of teaching"), "facilities": _find(cols, "facilities"),
        "materials": _find(cols, "study materials"), "clarity": _find(cols, "explain concepts"),
        "rating": _find(cols, "overall college experience"),
    }
    text_cols = {}
    for label, keys in TEXT_QUESTIONS.items():
        for k in keys:
            c = _find(cols, k)
            if c:
                text_cols[label] = c
                break
    if not text_cols:
        raise ValueError(f"Could not find the comment columns. Columns in file: {cols}")

    base = pd.DataFrame(index=raw.index)
    ts = raw[found["time"]].astype(str).str.replace(r"\s*GMT.*$", "", regex=True) if found["time"] else None
    base["created_at"] = pd.to_datetime(ts, errors="coerce", format="mixed").fillna(pd.Timestamp.now()) if ts is not None else pd.Timestamp.now()
    base["student_id"] = [anon_id(f"{i}|{t}") for i, t in zip(raw.index, base["created_at"])]
    base["year"] = raw[found["year"]].astype(str).str.strip() if found["year"] else "Unknown"
    for key, mapping in [("teaching", SAT), ("facilities", SAT), ("materials", SAT), ("clarity", CLR)]:
        base[key] = raw[found[key]].astype(str).str.strip().str.lower().map(mapping) if found[key] else None
    base["rating"] = pd.to_numeric(raw[found["rating"]].astype(str).str.extract(r"(\d)")[0], errors="coerce") if found["rating"] else None

    parts = []
    for label, col in text_cols.items():
        d = base.copy()
        d["question"] = label
        d["comment"] = raw[col].fillna("").astype(str).str.strip()
        parts.append(d[d["comment"].str.len() > 0])
    return pd.concat(parts, ignore_index=True), found, text_cols
