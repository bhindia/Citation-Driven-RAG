import numpy as np
from typing import List, Optional
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from config import BM25_TOP_DOCS, SENT_RERANK_K, MIN_SENT_WORDS
import logging

log = logging.getLogger("cd_rag_optionc.retrieval")

def retrieve_and_rerank(query: str, corpus_texts: List[str], corpus_srcs: List[str], bm25: BM25Okapi,
                        embedder: SentenceTransformer, cross_encoder: Optional[object] = None,
                        corpus_top_docs: int = BM25_TOP_DOCS, sentence_rerank_k: int = SENT_RERANK_K):
    qtok = query.split()
    bm_scores = bm25.get_scores(qtok)
    top_doc_idxs = np.argsort(-bm_scores)[:corpus_top_docs]

    candidate_sentences = []
    seen_sent = set()
    for doc_idx in top_doc_idxs:
        doc_text = corpus_texts[int(doc_idx)]
        src = corpus_srcs[int(doc_idx)]
        for s in sent_tokenize(doc_text):
            s = s.strip()
            if not s or len(s.split()) < MIN_SENT_WORDS: continue
            key = s.lower()
            if key in seen_sent: continue
            seen_sent.add(key)
            candidate_sentences.append((s, int(doc_idx), src))

    if not candidate_sentences: return [], {"bm25_top_docs": len(top_doc_idxs), "candidate_sentences":0}

    # rerank
    if cross_encoder:
        pairs = [(query, s) for s, _, _ in candidate_sentences]
        if len(pairs) > sentence_rerank_k:
            q_emb = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            s_embs = embedder.encode([s for s, _, _ in candidate_sentences], convert_to_tensor=True, normalize_embeddings=True)
            sims = util.cos_sim(q_emb, s_embs)[0].detach().cpu().numpy()
            order = np.argsort(-sims)[:sentence_rerank_k]
            candidate_sentences = [candidate_sentences[i] for i in order]
            pairs = [(query, s) for s, _, _ in candidate_sentences]
        try:
            scores = cross_encoder.predict(pairs, show_progress_bar=False)
        except Exception:
            q_emb = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            s_embs = embedder.encode([s for s, _, _ in candidate_sentences], convert_to_tensor=True, normalize_embeddings=True)
            scores = util.cos_sim(q_emb, s_embs)[0].detach().cpu().numpy()
    else:
        q_emb = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        s_embs = embedder.encode([s for s, _, _ in candidate_sentences], convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(q_emb, s_embs)[0].detach().cpu().numpy()

    scored = []
    for (s, idx, src), sc in zip(candidate_sentences, scores):
        scored.append((s, idx, float(sc), src))
    scored_sorted = sorted(scored, key=lambda x: -x[2])
    diagnostics = {"bm25_top_docs": len(top_doc_idxs), "candidate_sentences": len(candidate_sentences)}
    return scored_sorted, diagnostics
