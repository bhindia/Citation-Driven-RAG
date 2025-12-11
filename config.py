from pathlib import Path

DATA_DIR = Path("data")
SCI_DIR = DATA_DIR / "scifact"
SCIFACT_TGZ_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_TGZ = SCI_DIR / "data.tar.gz"

BM25_TOP_DOCS = 200
SENT_RERANK_K = 200
TOP_EVIDENCE = 3
MIN_SENT_WORDS = 8
MIN_FOCUS = 0.35
CONFIDENCE_THRESH = 0.52

STOP = set([
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by",
    "with","without","as","is","are","was","were","be","been","being","this","that","these",
    "those","it","its","from","into","over","under","than","such","so","not","no","yes"
])
