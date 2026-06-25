from typing import List, Any, Tuple
from .retrieve_search_documents import retrieve_search_documents

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)

try:
    from .load_documents import config as cf
except ImportError:
    from load_documents import config as cf


client = cf.client
model = cf.model
reranker = cf.reranker
COLLECTION = cf.COLLECTION

CANDIDATE_MULTIPLIER: int = int(getattr(cf, "config", {}).get("search", {}).get("candidate_multiplier", 8) if hasattr(cf, "config") else 8)
MIN_CANDIDATES: int = int(getattr(cf, "config", {}).get("search", {}).get("min_candidates", 80) if hasattr(cf, "config") else 80)
RERANK_POOL_SIZE: int = int(getattr(cf, "config", {}).get("search", {}).get("rerank_pool_size", 24) if hasattr(cf, "config") else 24)
MAX_CHUNKS_PER_DOCUMENT: int = int(getattr(cf, "config", {}).get("search", {}).get("max_chunks_per_document", 3) if hasattr(cf, "config") else 3)
AGGREGATION_TOP_K: int = int(getattr(cf, "config", {}).get("search", {}).get("aggregation_top_k", 3) if hasattr(cf, "config") else 3)
AGGREGATION_MAX_WEIGHT: float = float(getattr(cf, "config", {}).get("search", {}).get("aggregation_max_weight", 0.9) if hasattr(cf, "config") else 0.9)
AGGREGATION_MEAN_WEIGHT: float = float(getattr(cf, "config", {}).get("search", {}).get("aggregation_mean_weight", 0.1) if hasattr(cf, "config") else 0.1)
HYBRID_DENSE_WEIGHT: float = float(getattr(cf, "config", {}).get("search", {}).get("hybrid_dense_weight", 0.7) if hasattr(cf, "config") else 0.7)

# On conserve également SEARCH_LIMIT pour la fonction retrieve_documents
SEARCH_LIMIT: int = int(getattr(cf, "config", {}).get("search", {}).get("limit", 3) if hasattr(cf, "config") else 3)
FULL_DOCUMENT_CHUNK_THRESHOLD: int = int(getattr(cf, "config", {}).get("search", {}).get("full_document_chunk_threshold", 12) if hasattr(cf, "config") else 12)
SCROLL_LIMIT_PER_DOC: int = int(getattr(cf, "config", {}).get("search", {}).get("scroll_limit_per_doc", 10000) if hasattr(cf, "config") else 10000)
WINDOW_RADIUS: int = int(getattr(cf, "config", {}).get("search", {}).get("window_radius", 4) if hasattr(cf, "config") else 4)


def _load_document_chunks(document_id: str) -> list[Any]:
    all_points = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]),
            limit=SCROLL_LIMIT_PER_DOC, offset=offset, with_payload=True, with_vectors=False,
        )
        if not points: break
        all_points.extend(points)
        if offset is None: break
    return sorted(all_points, key=lambda p: int((p.payload or {}).get("chunk_index", 0)))


def _build_partial_header(filename: str, best_chunk_index: int, start_idx: int, end_idx: int, total_chunks: int) -> str:
    return (
        "[PARTIAL DOCUMENT ONLY]\n"
        f"filename={filename}\n"
        f"best_chunk_index={best_chunk_index}\n"
        f"returned_chunk_range={start_idx}-{end_idx}\n"
        f"total_chunks={total_chunks}\n"
        "note=Only a local window around the best matching chunk is returned, not the full document.\n\n"
    )


def _load_best_view_for_document(
    document_id: str,
    best_chunk_index: int,
    return_full_document: bool = True,
    is_single_doc: bool = False,
) -> tuple[str, str]:
    ordered = _load_document_chunks(document_id)

    if not ordered:
        return document_id, ""

    filename = str((ordered[0].payload or {}).get("filename", document_id))
    total_chunks = len(ordered)

    if not return_full_document:
        for p in ordered:
            payload = p.payload or {}
            if int(payload.get("chunk_index", 0)) == best_chunk_index:
                return filename, str(payload.get("text", ""))
        return filename, ""

    if total_chunks <= FULL_DOCUMENT_CHUNK_THRESHOLD:
        full_text = "\n".join(str((p.payload or {}).get("text", "")) for p in ordered if (p.payload or {}).get("text")).strip()
        return filename, full_text

    eff_radius = WINDOW_RADIUS * 3 if is_single_doc else WINDOW_RADIUS
    start_idx = best_chunk_index - eff_radius
    end_idx = best_chunk_index + eff_radius

    if start_idx < 0:
        end_idx += (0 - start_idx)
        start_idx = 0
    if end_idx >= total_chunks:
        start_idx -= (end_idx - (total_chunks - 1))
        end_idx = total_chunks - 1
    if start_idx < 0:
        start_idx = 0

    selected = [p for p in ordered if start_idx <= int((p.payload or {}).get("chunk_index", 0)) <= end_idx]
    partial_text = "\n".join(str((p.payload or {}).get("text", "")) for p in selected if (p.payload or {}).get("text")).strip()
    header = _build_partial_header(filename, best_chunk_index, start_idx, end_idx, total_chunks)

    return filename, header + partial_text


# =====================================================================
# PIPELINE 2 : RECHERCHE LLM (Avec reconstruction des documents)
# =====================================================================
def retrieve_documents(
    search_terms: str,
    limit: int = 10,
    return_full_document: bool = True,
    tags: list = None,
) -> List[Tuple[str, str, str, Any, float, list]]:
    if limit is None:
        limit = SEARCH_LIMIT

    try:
        search_results = retrieve_search_documents(search_terms, tags=tags, limit=limit)

        if not search_results:
            return []

        is_single_doc = (limit == 1)
        final_results: List[Tuple[str, str, str, Any, float, list]] = []

        for (filename, best_chunk_text, doc_summary, chunk0_id, score, tags_list, document_id, best_chunk_index) in search_results:
            _, text = _load_best_view_for_document(
                document_id=document_id,
                best_chunk_index=best_chunk_index,
                return_full_document=return_full_document,
                is_single_doc=is_single_doc,
            )

            final_results.append((
                filename,
                tags_list,
                doc_summary,
                text,
                score,
            ))

        return final_results

    except Exception as e:
        import traceback
        print(f"\n[!!! ERROR !!!] Crash in retrieve_documents: {e}")
        traceback.print_exc()
        return []


if __name__ == "__main__":
    query = input("Query: ").strip()
    results = retrieve_search_documents(query, limit=8)
    for i, (filename, text, summary, chunk0_id, score, tags) in enumerate(results, start=1):
        print(f"[{i}] {filename} | chunk_0_id={chunk0_id} | score={score:.4f} | tags={tags}")
