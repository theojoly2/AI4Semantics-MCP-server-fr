from typing import List, Tuple, Literal, Optional, Any

from qdrant_client.models import (
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)
try:
    from .load_documents import config as cf
except ImportError:
    from load_documents import config as cf

client = cf.client
model = cf.model
COLLECTION = cf.COLLECTION

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


SEARCH_LIMIT: int = int(
    getattr(cf, "config", {}).get("search", {}).get("limit", 3)
    if hasattr(cf, "config")
    else 3
)


Vocabulary = Literal[
    "CAV",
    "CBV",
    "CCCEV",
    "CLV",
    "CPEV",
    "CPOV",
    "CPSV",
    "CPV",
    "OSLO_Observaties_en_Metingen",
    "OSLO_Organisatie",
    "OSLO_Persoon",
    "OSLO_Adres",
    "OSLO_Gebouw",
]


VOCABULARIES: List[Vocabulary] = [
    "CAV",
    "CBV",
    "CCCEV",
    "CLV",
    "CPEV",
    "CPOV",
    "CPSV",
    "CPV",
    "OSLO_Observaties_en_Metingen",
    "OSLO_Organisatie",
    "OSLO_Persoon",
    "OSLO_Adres",
    "OSLO_Gebouw",
]


def _extract_vocabulary_from_filename(filename: str) -> Optional[str]:
    """
    Heuristic: vocabulary is the prefix before the first underscore in the file name.
    E.g. 'CPV_AP_SHACL_96.json' -> 'CPV'
    """
    if not filename:
        return None
    prefix = filename.split("_", 1)[0]
    return prefix if prefix in VOCABULARIES else None


def detect_model_capabilities(model) -> dict[str, Any]:
    """
    Detect if the loaded model supports dense vectors, sparse vectors, or both.
    """
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
    """
    Encode a query into dense and/or sparse vectors depending on model capabilities.
    """
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


def retrieve_documents(
    search_terms: str,
    vocabularies: Optional[List[Vocabulary]] = None,
    limit: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """
    Retrieve relevant documents from Qdrant using:
    - dense search if model is dense-only
    - sparse search if model is sparse-only
    - hybrid search (RRF fusion) if model supports both
    """
    if limit is None:
        limit = SEARCH_LIMIT

    try:
        capabilities = cf.MODEL_CAPABILITIES
        query_vectors = encode_query(search_terms, capabilities)

        if "dense" in query_vectors and "sparse" in query_vectors:
            results = client.query_points(
                collection_name=COLLECTION,
                prefetch=[
                    Prefetch(
                        query=query_vectors["dense"],
                        using=DENSE_VECTOR_NAME,
                        limit=max(limit * 3, 10),
                    ),
                    Prefetch(
                        query=query_vectors["sparse"],
                        using=SPARSE_VECTOR_NAME,
                        limit=max(limit * 3, 10),
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )

        elif "dense" in query_vectors:
            results = client.query_points(
                collection_name=COLLECTION,
                query=query_vectors["dense"],
                limit=limit,
                with_payload=True,
            )

        elif "sparse" in query_vectors:
            results = client.query_points(
                collection_name=COLLECTION,
                query=query_vectors["sparse"],
                using=SPARSE_VECTOR_NAME,
                limit=limit,
                with_payload=True,
            )

        else:
            raise ValueError("No query vectors generated")

        raw_results: List[Tuple[str, str, float]] = [
            (r.payload["filename"], r.payload["text"], r.score)
            for r in results.points
        ]

        if not vocabularies:
            return raw_results

        vocab_set = set(vocabularies)
        filtered_results: List[Tuple[str, str, float]] = []

        for filename, text, score in raw_results:
            vocab = _extract_vocabulary_from_filename(filename)
            if vocab is not None and vocab in vocab_set:
                filtered_results.append((filename, text, score))

        if not filtered_results:
            print(
                f"[retrieve_documents] No results after vocabulary filtering "
                f"for {vocabularies}; falling back to unfiltered results."
            )
            return raw_results

        return filtered_results

    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []
