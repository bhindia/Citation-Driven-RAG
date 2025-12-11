import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from config import DATA_DIR, SCI_DIR, SCIFACT_TGZ_URL, SCIFACT_TGZ
from utils import download_url, safe_extract_tar_gz
from corpus_builder import build_corpus_and_indices
from retrieval import retrieve_and_rerank
from answer_constructor import construct_answer_from_reranked
from metrics import FourMetrics

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cd_rag_optionc.main")

try:
    CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    HAS_CE = True
except Exception:
    CROSS_ENCODER = None
    HAS_CE = False

def main():
    # Download and extract
    download_url(SCIFACT_TGZ_URL, SCIFACT_TGZ)
    safe_extract_tar_gz(SCIFACT_TGZ, SCI_DIR)

    corpus_jsonl = next((p for p in SCI_DIR.rglob("corpus.jsonl")), None)
    if corpus_jsonl is None:
        raise FileNotFoundError("Couldn't find SciFact corpus.jsonl under data/scifact")

    corpus_texts, corpus_srcs, corpus_pmids, bm25 = build_corpus_and_indices(corpus_jsonl, use_pubmedqa=True)
    log.info(f"Corpus size: {len(corpus_texts)}")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    demo_queries = [
        "Does low-dose aspirin prevent myocardial infarction in patients with diabetes?",
        "Does omega-3 supplementation reduce the risk of cardiovascular events in adults?",
        "Does bariatric surgery improve glycemic control in obese patients with type 2 diabetes?",
        "Does influenza vaccination reduce hospitalizations in elderly populations?",
        "Does early administration of corticosteroids improve outcomes in severe COVID-19 patients?",
        "Does high-fiber diet reduce the incidence of colorectal cancer?",
        "Does physical exercise prevent cognitive decline in older adults?",
        "Does ACE inhibitor therapy reduce mortality in patients with heart failure?",
        "Does mindfulness meditation decrease symptoms in patients with anxiety disorders?",
        "Does vitamin B12 supplementation prevent neuropathy in patients with type 2 diabetes?",
        "Does antibiotic prophylaxis reduce post-surgical infections in orthopedic procedures?",
        "qwerty asdf nonsense query that should trigger abstention"
    ]

    metrics = FourMetrics()
    for q in demo_queries:
        print("\n=== QUERY ===\n", q)
        reranked, diagnostics = retrieve_and_rerank(q, corpus_texts, corpus_srcs, bm25, embedder, CROSS_ENCODER)
        ans, selected_spans, cite_labels = construct_answer_from_reranked(
            q, reranked, embedder, corpus_texts, corpus_srcs, corpus_pmids, diagnostics
        )
        print("A:", ans)
        metrics.update(ans, selected_spans)

    print("\nFinal Metrics:", metrics.report())

if __name__ == "__main__":
    main()
