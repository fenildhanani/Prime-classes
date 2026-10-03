"""Shared helpers: text cleaning + data loading."""
import re
import pandas as pd

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|co\.uk|info|biz)\S*)", re.I)
PHONE_RE = re.compile(r"\b\d{5,}\b")          # long digit strings (phone / short codes)
MONEY_RE = re.compile(r"[$£€₹]\s?\d+[\d,\.]*|\b\d+[\d,\.]*\s?(usd|gbp|rs|inr)\b", re.I)


def clean_text(text: str) -> str:
    """Lowercase, replace URLs / phone numbers / money with tokens, strip punctuation."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = PHONE_RE.sub(" numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)       # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path: str) -> pd.DataFrame:
    """Load a spam dataset and return a DataFrame with columns: label (0=ham, 1=spam), text.

    Supports:
      * Kaggle 'spam.csv'            (columns v1, v2; latin-1 encoding)
      * UCI 'SMSSpamCollection'      (tab separated, no header: label<TAB>text)
      * Any CSV with columns label/text, or category/message, etc.
    """
    if path.endswith((".tsv", ".txt")) or "SMSSpamCollection" in path:
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="latin-1")
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        cols = {c.lower().strip(): c for c in df.columns}
        label_col = next(cols[c] for c in ("v1", "label", "category", "class", "type") if c in cols)
        text_col = next(cols[c] for c in ("v2", "text", "message", "sms", "email") if c in cols)
        df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})

    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(
        {"spam": 1, "ham": 0, "1": 1, "0": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df
