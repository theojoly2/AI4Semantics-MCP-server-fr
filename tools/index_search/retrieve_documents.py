from typing import List, Any, Dict
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
        if not points:
            break
        all_points.extend(points)
        if offset is None:
            break
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

    # Always return only the best matching chunk (no surrounding context window)
    for p in ordered:
        payload = p.payload or {}
        if int(payload.get("chunk_index", 0)) == best_chunk_index:
            return filename, str(payload.get("text", ""))
    return filename, ""


def _payload_to_chunk_dict(payload: Dict[str, Any], score: float = 0.0) -> Dict[str, Any]:
    """
    Normalise un payload Qdrant en dictionnaire de chunk exploitable
    par l'UI et les outils (resolve_links, compare_concepts).
    """
    return {
        "text": str(payload.get("text", "")),
        "filename": str(payload.get("filename", "")),
        "document_name": str(payload.get("document_name", "")),
        "document_id": str(payload.get("document_id", "")),
        "chunk_index": int(payload.get("chunk_index", 0)),
        "chunk_count": int(payload.get("chunk_count", 1)),
        "source_path": str(payload.get("source_path", "")),
        "source_extension": str(payload.get("source_extension", "")),
        "doctype": str(payload.get("doctype", "generic")),
        "tags": payload.get("tags", []) or [],
        "term": str(payload.get("term", "")),
        "definition": str(payload.get("definition", "")),
        "synonyms": payload.get("synonyms", []) or [],
        "article": str(payload.get("article", "")),
        "cited_articles": payload.get("cited_articles", []) or [],
        "concept_uri": str(payload.get("concept_uri", "")),
        "broader": payload.get("broader", []) or [],
        "narrower": payload.get("narrower", []) or [],
        "related": payload.get("related", []) or [],
        "section_title": str(payload.get("section_title", "")),
        "doc_summary": str(payload.get("doc_summary", "")),
        "score": float(score),
    }


def retrieve_documents(
    search_terms: str,
    limit: int = 10,
    return_full_document: bool = True,
    tags: list = None,
    document_filter: str = None,
) -> List[Dict[str, Any]]:
    """
    Recherche des documents et retourne une liste de chunks enrichis
    avec leurs métadonnées (term, article, concept_uri, related, etc.).
    """
    if limit is None:
        limit = SEARCH_LIMIT

    try:
        search_results = retrieve_search_documents(
            search_terms=search_terms,
            tags=tags,
            limit=limit,
            document_filter=document_filter,
        )

        if not search_results:
            return []

        is_single_doc = (limit == 1)
        final_results: List[Dict[str, Any]] = []

        for (
            filename,
            best_chunk_text,
            doc_summary,
            chunk0_id,
            score,
            tags_list,
            document_id,
            best_chunk_index,
        ) in search_results:
            _, reconstructed_text = _load_best_view_for_document(
                document_id=document_id,
                best_chunk_index=best_chunk_index,
                return_full_document=return_full_document,
                is_single_doc=is_single_doc,
            )

            final_results.append({
                "text": reconstructed_text,
                "filename": filename,
                "document_name": filename.replace(".", "_").rsplit("_", 1)[0] if "." in filename else filename,
                "document_id": document_id,
                "tags": tags_list,
                "doc_summary": doc_summary,
                "score": float(score),
                "best_chunk_index": int(best_chunk_index),
                # Métadonnées détaillées du meilleur chunk
                "chunk": _payload_to_chunk_dict({
                    "text": best_chunk_text,
                    "filename": filename,
                    "document_name": filename.replace(".", "_").rsplit("_", 1)[0] if "." in filename else filename,
                    "document_id": document_id,
                    "chunk_index": best_chunk_index,
                }, score=score),
            })

        return final_results

    except Exception as e:
        import traceback
        print(f"\n[!!! ERROR !!!] Crash in retrieve_documents: {e}")
        traceback.print_exc()
        return []


if __name__ == "__main__":
    query = input("Query: ").strip()
    results = retrieve_documents(query, limit=8)
    for i, r in enumerate(results, start=1):
        print(f"[{i}] {r['filename']} | score={r['score']:.4f} | tags={r['tags']}")
        print(r["text"][:500])
        print("-" * 50)
