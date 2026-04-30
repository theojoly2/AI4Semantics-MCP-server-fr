from os import getenv
from typing import Any

from pathlib import Path
from yaml import safe_load
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = safe_load(f)

def _load_qdrant_client() -> QdrantClient:
    qdrant_cfg = config["qdrant"]

    api_key_env_name = qdrant_cfg.get("api_key")
    api_key_value = getenv(api_key_env_name) if api_key_env_name else None

    return QdrantClient(
        host=qdrant_cfg["host"],
        port=qdrant_cfg["port"],
        timeout=qdrant_cfg.get("timeout", 120),
        api_key=api_key_value,
        https=False,
    )


def _load_model():
    model_name = config["model"]["name"]

    if "bge-m3" in model_name.lower():
        from FlagEmbedding import BGEM3FlagModel

        model = BGEM3FlagModel(
            model_name,
            use_fp16=True,
        )
        print(f"✓ Loaded hybrid model: {model_name}")
        return model

    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        print(f"✓ Loaded dense model: {model_name}")
        return model


def _detect_model_capabilities(model) -> dict[str, Any]:
    """
    Detect whether model supports dense, sparse, or both.
    Runs once at startup.
    """
    capabilities = {
        "has_dense": False,
        "has_sparse": False,
        "dense_dim": None,
    }

    test_text = "test"

    # Try BGE-M3 / FlagEmbedding style first
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

    # Fallback: SentenceTransformer-like dense model
    try:
        dense = model.encode([test_text])
        first_dense = dense[0] if hasattr(dense, "__len__") else dense
        capabilities["has_dense"] = True
        capabilities["dense_dim"] = len(first_dense)
        return capabilities

    except Exception as e:
        raise ValueError(f"Could not detect model capabilities for configured model: {e}")


client = _load_qdrant_client()
model = _load_model()
MODEL_CAPABILITIES = _detect_model_capabilities(model)


COLLECTION = config["collection"]["name"]
BATCH_SIZE = int(config["indexing"].get("batch_size", 5))
DOCUMENTS_PATH = config["indexing"].get("documents_path", "documents")
SEARCH_LIMIT = int(config.get("search", {}).get("limit", 3))


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


print(
    "✓ Model capabilities:",
    {
        "has_dense": MODEL_CAPABILITIES["has_dense"],
        "has_sparse": MODEL_CAPABILITIES["has_sparse"],
        "dense_dim": MODEL_CAPABILITIES["dense_dim"],
    },
)
