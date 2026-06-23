from typing import List, Any, Tuple, Dict
from collections import defaultdict

from qdrant_client.models import (
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

try:
    from .load_documents import config as cf
except ImportError:
    from load_documents import config as cf


client = cf.client
model = cf.model
reranker = cf.reranker
COLLECTION = cf.COLLECTION

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

SEARCH_LIMIT: int = int(
    getattr(cf, "config", {}).get("search", {}).get("limit", 3)
    if hasattr(cf, "config")
    else 3
)

CANDIDATE_MULTIPLIER: int = int(
    getattr(cf, "config", {}).get("search", {}).get("candidate_multiplier", 8)
    if hasattr(cf, "config")
    else 8
)

MIN_CANDIDATES: int = int(
    getattr(cf, "config", {}).get("search", {}).get("min_candidates", 80)
    if hasattr(cf, "config")
    else 80
)

RERANK_POOL_SIZE: int = int(
    getattr(cf, "config", {}).get("search", {}).get("rerank_pool_size", 24)
    if hasattr(cf, "config")
    else 24
)

MAX_CHUNKS_PER_DOCUMENT: int = int(
    getattr(cf, "config", {}).get("search", {}).get("max_chunks_per_document", 3)
    if hasattr(cf, "config")
    else 3
)

SCROLL_LIMIT_PER_DOC: int = int(
    getattr(cf, "config", {}).get("search", {}).get("scroll_limit_per_doc", 10000)
    if hasattr(cf, "config")
    else 10000
)

WINDOW_RADIUS: int = int(
    getattr(cf, "config", {}).get("search", {}).get("window_radius", 4)
    if hasattr(cf, "config")
    else 4
)

FULL_DOCUMENT_CHUNK_THRESHOLD: int = int(
    getattr(cf, "config", {}).get("search", {}).get("full_document_chunk_threshold", 12)
    if hasattr(cf, "config")
    else 12
)

AGGREGATION_TOP_K: int = int(
    getattr(cf, "config", {}).get("search", {}).get("aggregation_top_k", 3)
    if hasattr(cf, "config")
    else 3
)

AGGREGATION_MAX_WEIGHT: float = float(
    getattr(cf, "config", {}).get("search", {}).get("aggregation_max_weight", 0.9)
    if hasattr(cf, "config")
    else 0.9
)

AGGREGATION_MEAN_WEIGHT: float = float(
    getattr(cf, "config", {}).get("search", {}).get("aggregation_mean_weight", 0.1)
    if hasattr(cf, "config")
    else 0.1
)

HYBRID_DENSE_WEIGHT: float = float(
    getattr(cf, "config", {}).get("search", {}).get("hybrid_dense_weight", 0.7)
    if hasattr(cf, "config")
    else 0.7
)


def detect_model_capabilities(model) -> dict[str, Any]:
    capabilities = {
        "has_dense": False,
        "has_sparse": False,
        "dense_dim": None,
    }

    test_text = "test"

    try:
        result = model.encode(
            [test_text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        if isinstance(result, dict):
            dense_vecs = result.get("dense_vecs")
            lexical_weights = result.get("lexical_weights")

            if dense_vecs is not None and len(dense_vecs) > 0:
                capabilities["has_dense"] = True
                capabilities["dense_dim"] = len(dense_vecs[0])

            if lexical_weights is not None:
                capabilities["has_sparse"] = True

            return capabilities

    except Exception:
        pass

    try:
        dense = model.encode([test_text])
        first_dense = dense[0] if hasattr(dense, "__len__") else dense
        capabilities["has_dense"] = True
        capabilities["dense_dim"] = len(first_dense)
        return capabilities
    except Exception as e:
        raise ValueError(f"Could not detect model capabilities: {e}")


def _lexical_to_sparse_vector(lexical_weights: dict[Any, Any]) -> SparseVector:
    return SparseVector(
        indices=[int(k) for k in lexical_weights.keys()],
        values=[float(v) for v in lexical_weights.values()],
    )


def encode_query(query_text: str, capabilities: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}

    if capabilities["has_dense"] and capabilities["has_sparse"]:
        result = model.encode(
            [query_text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = result.get("dense_vecs", [])
        lexical_weights = result.get("lexical_weights", [])

        if len(dense_vecs) > 0:
            dense = dense_vecs[0]
            outputs["dense"] = dense.tolist() if hasattr(dense, "tolist") else list(dense)

        if len(lexical_weights) > 0:
            outputs["sparse"] = _lexical_to_sparse_vector(lexical_weights[0])

        return outputs

    if capabilities["has_dense"]:
        dense = model.encode([query_text])
        first_dense = dense[0] if hasattr(dense, "__len__") else dense
        outputs["dense"] = (
            first_dense.tolist()
            if hasattr(first_dense, "tolist")
            else list(first_dense)
        )
        return outputs

    if capabilities["has_sparse"]:
        result = model.encode(
            [query_text],
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        lexical_weights = result.get("lexical_weights", [])
        if len(lexical_weights) > 0:
            outputs["sparse"] = _lexical_to_sparse_vector(lexical_weights[0])
        return outputs

    raise ValueError("No supported vector output available from model")


def _run_chunk_search(search_terms: str, raw_limit: int, tags: list = None):
    capabilities = cf.MODEL_CAPABILITIES
    query_vectors = encode_query(search_terms, capabilities)

    query_filter = None
    if tags:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="tags",
                    match=MatchAny(any=tags)
                )
            ]
        )

    dense_res = None
    sparse_res = None

    if "dense" in query_vectors:
        dense_res = client.query_points(
            collection_name=COLLECTION,
            query=query_vectors["dense"],
            query_filter=query_filter,
            using=DENSE_VECTOR_NAME,
            limit=raw_limit,
            with_payload=True,
        )

    if "sparse" in query_vectors:
        sparse_res = client.query_points(
            collection_name=COLLECTION,
            query=query_vectors["sparse"],
            query_filter=query_filter,
            using=SPARSE_VECTOR_NAME,
            limit=raw_limit,
            with_payload=True,
        )

    if dense_res is None and sparse_res is None:
        raise ValueError("No query vectors generated")

    if sparse_res is None:
        return dense_res
    if dense_res is None:
        return sparse_res

    merged: dict[Any, dict[str, Any]] = {}

    for p in dense_res.points:
        merged[p.id] = {
            "point": p,
            "dense": float(p.score),
            "sparse": 0.0,
        }

    for p in sparse_res.points:
        entry = merged.get(p.id)
        if entry is None:
            entry = {
                "point": p,
                "dense": 0.0,
                "sparse": 0.0,
            }
            merged[p.id] = entry
        entry["sparse"] = float(p.score)

    def minmax(scores: dict[Any, float]) -> dict[Any, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        vmin, vmax = min(vals), max(vals)
        if vmax <= vmin:
            return {k: 0.0 for k in scores.keys()}
        rng = vmax - vmin
        return {k: (v - vmin) / rng for k, v in scores.items()}

    dense_scores = {pid: v["dense"] for pid, v in merged.items()}
    sparse_scores = {pid: v["sparse"] for pid, v in merged.items()}

    dense_norm = minmax(dense_scores)
    sparse_norm = minmax(sparse_scores)

    alpha = HYBRID_DENSE_WEIGHT
    for pid, v in merged.items():
        d = dense_norm.get(pid, 0.0)
        s = sparse_norm.get(pid, 0.0)
        hybrid = alpha * d + (1.0 - alpha) * s
        v["hybrid"] = hybrid
        try:
            v["point"].score = hybrid
        except Exception:
            pass

    sorted_points = [
        merged[pid]["point"]
        for pid in sorted(
            merged.keys(),
            key=lambda pid: merged[pid]["hybrid"],
            reverse=True,
        )
    ][:raw_limit]

    class QueryResult:
        def __init__(self, points):
            self.points = points

    return QueryResult(sorted_points)


def _limit_chunks_per_document(
    points: list[Any],
    max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
) -> list[Any]:
    counts: dict[str, int] = defaultdict(int)
    limited: list[Any] = []

    for point in points:
        payload = point.payload or {}
        document_id = str(payload.get("document_id") or payload.get("filename") or point.id)
        if counts[document_id] >= max_chunks_per_document:
            continue
        limited.append(point)
        counts[document_id] += 1

    return limited


def _group_best_chunk_per_document(points: list[Any]) -> list[dict[str, Any]]:
    grouped: Dict[str, dict[str, Any]] = {}

    for rank, point in enumerate(points):
        payload = point.payload or {}
        score = float(getattr(point, "score", 0.0) or 0.0)

        document_id = str(payload.get("document_id") or payload.get("filename") or f"doc_{rank}")
        filename = str(payload.get("filename") or document_id)
        chunk_index = int(payload.get("chunk_index", 0))
        chunk_count = int(payload.get("chunk_count", 1))
        chunk_text = str(payload.get("text", ""))

        current = grouped.get(document_id)
        if current is None:
            grouped[document_id] = {
                "document_id": document_id,
                "filename": filename,
                "scores": [score],
                "best_score": score,
                "best_chunk_index": chunk_index,
                "best_chunk_count": chunk_count,
                "best_chunk_text": chunk_text,
            }
            continue

        current["scores"].append(score)

        if score > current["best_score"]:
            current["best_score"] = score
            current["best_chunk_index"] = chunk_index
            current["best_chunk_count"] = chunk_count
            current["best_chunk_text"] = chunk_text

    aggregated_results: list[dict[str, Any]] = []

    for doc in grouped.values():
        doc = dict(doc)
        doc["aggregated_score"] = _compute_document_aggregated_score(
            scores=doc["scores"],
            top_k=AGGREGATION_TOP_K,
            max_weight=AGGREGATION_MAX_WEIGHT,
            mean_weight=AGGREGATION_MEAN_WEIGHT,
        )
        aggregated_results.append(doc)

    return sorted(
        aggregated_results,
        key=lambda x: (x["aggregated_score"], x["best_score"]),
        reverse=True,
    )


def _load_document_chunks(document_id: str) -> list[Any]:
    all_points = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            limit=SCROLL_LIMIT_PER_DOC,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        all_points.extend(points)

        if offset is None:
            break

    return sorted(
        all_points,
        key=lambda p: int((p.payload or {}).get("chunk_index", 0)),
    )


def _build_partial_header(
    filename: str,
    best_chunk_index: int,
    start_idx: int,
    end_idx: int,
    total_chunks: int,
) -> str:
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
        full_text = "\n".join(
            str((p.payload or {}).get("text", ""))
            for p in ordered
            if (p.payload or {}).get("text")
        ).strip()
        return filename, full_text

    # Rayon effectif : triplé si on ne demande qu'un seul document
    eff_radius = WINDOW_RADIUS * 3 if is_single_doc else WINDOW_RADIUS

    start_idx = best_chunk_index - eff_radius
    end_idx = best_chunk_index + eff_radius

    # Décalage vers le bas si on manque de chunks au-dessus
    if start_idx < 0:
        missing_start = 0 - start_idx
        end_idx += missing_start
        start_idx = 0

    # Décalage vers le haut si on manque de chunks en-dessous
    if end_idx >= total_chunks:
        missing_end = end_idx - (total_chunks - 1)
        start_idx -= missing_end
        end_idx = total_chunks - 1

    # Sécurisation finale pour éviter les index négatifs (si le document est très petit)
    if start_idx < 0:
        start_idx = 0

    selected = [
        p for p in ordered
        if start_idx <= int((p.payload or {}).get("chunk_index", 0)) <= end_idx
    ]

    partial_text = "\n".join(
        str((p.payload or {}).get("text", ""))
        for p in selected
        if (p.payload or {}).get("text")
    ).strip()

    header = _build_partial_header(
        filename=filename,
        best_chunk_index=best_chunk_index,
        start_idx=start_idx,
        end_idx=end_idx,
        total_chunks=total_chunks,
    )

    return filename, header + partial_text


def _rerank_documents(
    query: str,
    docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not docs:
        return []

    pairs = [
        [query, doc.get("rerank_text", "")]
        for doc in docs
    ]

    scores = reranker.compute_score(pairs, normalize=True)

    if not isinstance(scores, list):
        scores = [scores]

    reranked = []
    for doc, rerank_score in zip(docs, scores):
        item = dict(doc)
        item["rerank_score"] = float(rerank_score)
        reranked.append(item)

    reranked.sort(
        key=lambda x: (x["rerank_score"], x["aggregated_score"], x["best_score"]),
        reverse=True,
    )
    return reranked


def _compute_document_aggregated_score(
    scores: list[float],
    top_k: int = AGGREGATION_TOP_K,
    max_weight: float = AGGREGATION_MAX_WEIGHT,
    mean_weight: float = AGGREGATION_MEAN_WEIGHT,
) -> float:
    if not scores:
        return 0.0

    sorted_scores = sorted(scores, reverse=True)
    best_score = sorted_scores[0]
    top_scores = sorted_scores[:max(1, top_k)]
    mean_top_score = sum(top_scores) / len(top_scores)

    weight_sum = max_weight + mean_weight
    if weight_sum <= 0:
        return best_score

    max_weight = max_weight / weight_sum
    mean_weight = mean_weight / weight_sum

    return (max_weight * best_score) + (mean_weight * mean_top_score)


def retrieve_documents(
    search_terms: str,
    limit: int = 10,
    return_full_document: bool = True,
    tags: list = None,
) -> List[Tuple[str, str, float, list]]:
    if limit is None:
        limit = SEARCH_LIMIT

    try:
        raw_limit = max(
            MIN_CANDIDATES,
            RERANK_POOL_SIZE * 4,
            limit * CANDIDATE_MULTIPLIER,
        )

        results = _run_chunk_search(search_terms, raw_limit, tags=tags)
        points = getattr(results, "points", [])

        if not points:
            return []

        doc_tags_map = {}
        for p in points:
            payload = p.payload or {}
            doc_id = str(payload.get("document_id") or payload.get("filename") or p.id)
            if doc_id not in doc_tags_map:
                t = payload.get("tags", [])
                doc_tags_map[doc_id] = [t] if isinstance(t, str) else list(t)

        points = _limit_chunks_per_document(
            points,
            max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
        )

        grouped_docs = _group_best_chunk_per_document(points)
        stable_candidate_docs = grouped_docs[:RERANK_POOL_SIZE]

        # On active la compensation maximale si le LLM ne veut qu'un seul résultat
        is_single_doc = (limit == 1)

        candidate_docs: list[dict[str, Any]] = []
        for doc in stable_candidate_docs:
            filename, text = _load_best_view_for_document(
                document_id=doc["document_id"],
                best_chunk_index=doc["best_chunk_index"],
                return_full_document=return_full_document,
                is_single_doc=is_single_doc,
            )

            candidate_docs.append({
                **doc,
                "filename": filename,
                "text": text,
                "rerank_text": text,
            })

        reranked_docs = _rerank_documents(search_terms, candidate_docs)
        final_docs = reranked_docs[:limit]

        final_results: List[Tuple[str, str, float]] = []
        for doc in final_docs:
            doc_id = doc["document_id"]
            doc_tags = doc_tags_map.get(doc_id, [])
            final_results.append(
                (doc["filename"], doc["text"], doc["rerank_score"], doc_tags)
            )

        return final_results

    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []


if __name__ == "__main__":
    query = input("Query: ").strip()
    results = retrieve_documents(query, limit=8, return_full_document=True)

    print("=" * 80)
    print(f"Results for: {query}")
    print("=" * 80)

    for i, (filename, text, score) in enumerate(results, start=1):
        preview = text[:800].replace("\n", " ")
        print(f"\n[{i}] {filename} | score={score:.4f}")
        print(preview)
        if len(text) > 800:
            print("...")
