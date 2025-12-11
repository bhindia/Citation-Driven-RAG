from pathlib import Path
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import logging
from text_utils import chunk_abstracts_text, dedupe_passages
from utils import read_jsonl

log = logging.getLogger("cd_rag_optionc.corpus")

def pubmedqa_extract_contexts(ds_split) -> List[dict]:
    out = []
    for r in ds_split:
        pmid = r.get("pubid") or r.get("id") or r.get("pmid") or r.get("pubmed_id")
        ctx = r.get("context")
        if isinstance(ctx, str) and ctx.strip():
            out.append({"text": ctx.strip(), "pmid": str(pmid) if pmid else None, "src": "PubMedQA"})
        elif isinstance(ctx, dict):
            contexts = ctx.get("contexts") or []
            for c in contexts:
                if isinstance(c, str) and c.strip():
                    out.append({"text": c.strip(), "pmid": str(pmid) if pmid else None, "src": "PubMedQA"})
        elif isinstance(ctx, list):
            for c in ctx:
                if isinstance(c, str) and c.strip():
                    out.append({"text": c.strip(), "pmid": str(pmid) if pmid else None, "src": "PubMedQA"})
                elif isinstance(c, dict):
                    sub = c.get("contexts") or []
                    for s in sub:
                        if isinstance(s, str) and s.strip():
                            out.append({"text": s.strip(), "pmid": str(pmid) if pmid else None, "src": "PubMedQA"})
    return out

def build_corpus_and_indices(scifact_jsonl_path: Path, use_pubmedqa: bool = True) -> Tuple[List[str], List[str], List[str], BM25Okapi]:
    scifact_records = read_jsonl(scifact_jsonl_path)
    scifact_passages = []
    for rec in scifact_records:
        abst = rec.get("abstract")
        if isinstance(abst, list):
            abst_text = " ".join([s for s in abst if isinstance(s, str)])
        else:
            abst_text = abst or ""
        if not abst_text.strip(): continue
        for c in chunk_abstracts_text(abst_text, max_tokens=400):
            scifact_passages.append({"text": c, "pmid": None, "src": "SciFact"})
    log.info(f"SciFact passages: {len(scifact_passages)}")

    pmqa_contexts = []
    if use_pubmedqa:
        try:
            log.info("Loading PubMedQA...")
            pmqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
            pmqa_train = pmqa["train"]
            pmqa_contexts = pubmedqa_extract_contexts(pmqa_train)
            log.info(f"PubMedQA contexts: {len(pmqa_contexts)}")
        except Exception as e:
            log.warning(f"Could not load PubMedQA: {e}")
            pmqa_contexts = []

    merged = scifact_passages + pmqa_contexts
    deduped = dedupe_passages(merged)
    corpus_texts = [d["text"] for d in deduped]
    corpus_srcs = [d.get("src") for d in deduped]
    corpus_pmids = [d.get("pmid") for d in deduped]
    bm25 = BM25Okapi([c.split() for c in corpus_texts])
    return corpus_texts, corpus_srcs, corpus_pmids, bm25
