from typing import List, Optional, Any, Tuple
from qdrant_client.models import (
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
    Filter,
    FieldCondition,
    MatchAny,
)

try:
    from .load_documents import config as cf
except ImportError:
    from load_documents import config as cf

from .config_loader import VOCABULARIES

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
    vocabularies: Optional[list[str]] = None,
    limit: int = 10,
):
    if vocabularies is None:
        vocabularies = VOCABULARIES
    """
    Retrieve relevant documents from Qdrant using:
    - dense search if model is dense-only
    - sparse search if model is sparse-only
    - hybrid search (RRF fusion) if model supports both
    
    Filters by vocabulary using Qdrant's native filtering BEFORE vector search.
    If vocabularies are specified and no results match, returns an empty list.
    """
    if limit is None:
        limit = SEARCH_LIMIT

    try:
        capabilities = cf.MODEL_CAPABILITIES
        query_vectors = encode_query(search_terms, capabilities)

        # ✅ Crée un filtre Qdrant si vocabularies est spécifié
        query_filter = None
        if vocabularies:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="vocabulary",
                        match=MatchAny(any=list(vocabularies))
                    )
                ]
            )

        # Recherche hybride (dense + sparse)
        if "dense" in query_vectors and "sparse" in query_vectors:
            results = client.query_points(
                collection_name=COLLECTION,
                prefetch=[
                    Prefetch(
                        query=query_vectors["dense"],
                        using=DENSE_VECTOR_NAME,
                        limit=max(limit * 3, 10),
                        filter=query_filter,  # ✅ Filtre AVANT la recherche
                    ),
                    Prefetch(
                        query=query_vectors["sparse"],
                        using=SPARSE_VECTOR_NAME,
                        limit=max(limit * 3, 10),
                        filter=query_filter,  # ✅ Filtre AVANT la recherche
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )

        # Recherche dense uniquement
        elif "dense" in query_vectors:
            results = client.query_points(
                collection_name=COLLECTION,
                query=query_vectors["dense"],
                limit=limit,
                query_filter=query_filter,  # ✅ Filtre AVANT la recherche
                with_payload=True,
            )

        # Recherche sparse uniquement
        elif "sparse" in query_vectors:
            results = client.query_points(
                collection_name=COLLECTION,
                query=query_vectors["sparse"],
                using=SPARSE_VECTOR_NAME,
                limit=limit,
                query_filter=query_filter,  # ✅ Filtre AVANT la recherche
                with_payload=True,
            )

        else:
            raise ValueError("No query vectors generated")

        # ✅ Retourne directement les résultats (déjà filtrés par Qdrant)
        return [
            (r.payload["filename"], r.payload["text"], r.score)
            for r in results.points
        ]

    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []
