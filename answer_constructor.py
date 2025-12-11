from typing import List, Tuple
from text_utils import sentence_required_overlap, focus_score
from config import TOP_EVIDENCE, MIN_FOCUS, MIN_SENT_WORDS, CONFIDENCE_THRESH
from sentence_transformers import SentenceTransformer

def construct_answer_from_reranked(query: str, reranked: List[Tuple[str,int,float,str]],
                                    embedder: SentenceTransformer, corpus_texts: List[str],
                                    corpus_srcs: List[str], corpus_pmids: List[str], diagnostics: dict,
                                    n_top: int = TOP_EVIDENCE):
    selected_spans = []
    cite_labels = []
    used_docs = set()

    for s, doc_idx, score, src in reranked:
        if doc_idx in used_docs: continue
        if len(s.split()) < MIN_SENT_WORDS: continue
        if score < CONFIDENCE_THRESH: continue
        if not sentence_required_overlap(s, query) and focus_score(query, s, embedder) < MIN_FOCUS:
            continue
        fs = focus_score(query, s, embedder)
        if fs < MIN_FOCUS: continue
        selected_spans.append(s)
        pmid = corpus_pmids[doc_idx]
        cite_label = f"[{src}{' PMID:'+str(pmid) if pmid else ''}]"
        cite_labels.append(cite_label)
        used_docs.add(int(doc_idx))
        if len(selected_spans) >= n_top: break

    if not selected_spans:
        return "I don't know. No strong evidence found.", [], []

    full_answer = " || ".join([f"{sp} {lbl}" for sp, lbl in zip(selected_spans, cite_labels)])
    return full_answer, selected_spans, cite_labels

