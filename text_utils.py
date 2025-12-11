import re
from typing import List
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
from sentence_transformers import SentenceTransformer, util
from config import STOP, MIN_FOCUS

_word_re = re.compile(r"[A-Za-z][A-Za-z-]+")

def words(text: str) -> List[str]:
    return [w.lower() for w in wordpunct_tokenize(text) if re.search(r"[A-Za-z]", w)]

def content_words(text: str) -> List[str]:
    return [w for w in words(text) if w not in STOP and len(w) > 2]

def sentence_required_overlap(sent: str, query: str) -> bool:
    qset = set(content_words(query))
    if not qset:
        return True
    toks = set(re.findall(_word_re, sent.lower()))
    return bool(toks & qset)

def focus_score(q: str, s: str, embedder: SentenceTransformer) -> float:
    qw = set(content_words(q))
    sw = set(content_words(s))
    lex = len(qw & sw) / max(1.0, (len(qw)*len(sw))**0.5) if qw and sw else 0.0
    qv = embedder.encode([q], convert_to_tensor=True, normalize_embeddings=True)
    sv = embedder.encode([s], convert_to_tensor=True, normalize_embeddings=True)
    sem = float(util.cos_sim(qv, sv)[0][0].item())
    sem = (sem + 1.0)/2.0
    return 0.55 * sem + 0.45 * lex

def chunk_abstracts_text(text: str, max_tokens: int = 400) -> List[str]:
    if not text:
        return []
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    chunks = []
    cur, cur_len = [], 0
    for s in sents:
        toks = s.split()
        if cur_len + len(toks) > max_tokens and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += len(toks)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def dedupe_passages(passages: List[dict]) -> List[dict]:
    seen = set()
    uniq = []
    for p in passages:
        txt = p.get("text", "")
        if isinstance(txt, dict): txt = str(txt)
        elif isinstance(txt, (list, tuple)): txt = " ".join(map(str, txt))
        txt = txt.strip()
        if not txt: continue
        if txt not in seen:
            seen.add(txt)
            uniq.append({"text": txt, "src": p.get("src"), "pmid": p.get("pmid")})
    return uniq

