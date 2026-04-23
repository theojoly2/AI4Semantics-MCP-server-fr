from typing import List, Tuple, Literal, Optional
from tools.index_search.load_documents import config as cf

client = cf.client
model = cf.model
COLLECTION = cf.COLLECTION

# Default search limit from YAML config (falls back to 3)
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


async def retrieve_documents(
    search_terms: str,
    vocabularies: Optional[List[Vocabulary]] = None,
    limit: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """
    Provides context regarding the query by retrieving
    relevant documents from a local Qdrant index, optionally
    constrained to a list of vocabularies.

    Args:
        search_terms (str):
            Keywords summarizing the user's input or
            a short summary of the user's input.

        vocabularies (list[Vocabulary] | None):
            Optional whitelist of vocabularies to filter on.
            If provided, only documents whose inferred vocabulary
            (from the filename prefix) is in this list are returned.

        limit (int | None):
            Maximum number of documents to retrieve from Qdrant
            before vocabulary filtering is applied. If None,
            it defaults to the value from `config.yaml` (search.limit)
            or 3 if not configured.

    Returns:
        List[Tuple[str, str, float]]:
            A list of tuples `(filename, text, score)` for each
            retrieved document.
    """
    if limit is None:
        limit = SEARCH_LIMIT

    try:
        embedding = model.encode(search_terms)
        results = client.query_points(
            collection_name=COLLECTION,
            query=embedding.tolist(),
            limit=limit,
        )

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

        return filtered_results

    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

