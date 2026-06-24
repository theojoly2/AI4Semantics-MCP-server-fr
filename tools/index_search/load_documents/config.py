from pathlib import Path
from os import getenv, environ

PROJECT_ROOT = Path(__file__).resolve().parent
environ["HF_HOME"] = str(PROJECT_ROOT / ".cache_hf")

from typing import Any, List, Tuple, Union

from yaml import safe_load
from qdrant_client import QdrantClient
from dotenv import load_dotenv

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType
from huggingface_hub import snapshot_download
from filelock import FileLock

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = safe_load(f)


# =========================================================
# RERANKER
# =========================================================
class HFTransformerReranker:
    """
    GPU  → FP16
    CPU  → ONNX INT8 (Q8 auto export + quantization)
    """

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device == "cuda":
            self.model = self._load_gpu()
        else:
            self.model = self._load_cpu_q8()

    # =====================================================
    # GPU FP16
    # =====================================================
    def _load_gpu(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        )
        return model.to(self.device)

    # =====================================================
    # CPU ONNX + AUTO Q8
    # =====================================================
    def _load_cpu_q8(self):
        base_dir = PROJECT_ROOT / "models/onnx"
        fp32_dir = base_dir / "bge-reranker-v2-m3"
        int8_dir = base_dir / "bge-reranker-v2-m3-int8"

        lock = FileLock(str(int8_dir) + ".lock")

        with lock:

            # -----------------------------
            # 1. si INT8 existe → direct
            # -----------------------------
            if int8_dir.exists() and any(int8_dir.iterdir()):
                print(f"✓ Loading ONNX INT8 reranker: {int8_dir}")
                return ORTModelForSequenceClassification.from_pretrained(
                    str(int8_dir)
                )

            # -----------------------------
            # 2. sinon FP32 ONNX existe ?
            # -----------------------------
            if not fp32_dir.exists() or not any(fp32_dir.iterdir()):
                print("⚠ Export ONNX FP32...")

                snapshot_download(repo_id=self.model_name)

                main_export(
                    model_name_or_path=self.model_name,
                    output=str(fp32_dir),
                    task="text-classification",
                )

                print(f"✓ FP32 ONNX exported → {fp32_dir}")

            # -----------------------------
            # 3. quantization INT8 (Q8)
            # -----------------------------
            print("⚡ Quantizing ONNX → INT8 (Q8)...")

            fp32_model = fp32_dir / "model.onnx"
            int8_model = int8_dir

            int8_model.mkdir(parents=True, exist_ok=True)

            quantize_dynamic(
                model_input=str(fp32_model),
                model_output=str(int8_model / "model.onnx"),
                weight_type=QuantType.QInt8,
            )

            # copier config/tokenizer
            for f in fp32_dir.glob("*"):
                if f.suffix != ".onnx":
                    (int8_model / f.name).write_bytes(f.read_bytes())

            print(f"✓ INT8 ONNX ready → {int8_model}")

        print(f"✓ Loading ONNX INT8 reranker: {int8_dir}")

        return ORTModelForSequenceClassification.from_pretrained(
            str(int8_dir)
        )

    # =====================================================
    # SCORE
    # =====================================================
    @torch.inference_mode()
    def compute_score(self, sentence_pairs, normalize=False, max_length=512):
        if isinstance(sentence_pairs, tuple):
            pairs = [sentence_pairs]
        else:
            pairs = list(sentence_pairs)

        queries = [q for q, _ in pairs]
        docs = [d for _, d in pairs]

        inputs = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        logits = self.model(**inputs).logits.view(-1)

        scores = logits.float()

        if normalize:
            scores = torch.sigmoid(scores)

        scores = scores.cpu().tolist()

        if isinstance(sentence_pairs, tuple):
            return scores[0]

        return scores


# =========================================================
# QDRANT
# =========================================================
def _load_qdrant_client() -> QdrantClient:
    qdrant_cfg = config["qdrant"]

    api_key_env_name = qdrant_cfg.get("api_key")
    api_key_value = getenv(api_key_env_name) if api_key_env_name else None

    return QdrantClient(
        host=qdrant_cfg["host"],
        port=qdrant_cfg["port"],
        timeout=qdrant_cfg.get("timeout", 120),
        api_key=api_key_value,
        https=qdrant_cfg.get("https", False),
        check_compatibility=qdrant_cfg.get("check_compatibility", False),
    )


# =========================================================
# EMBEDDINGS
# =========================================================
def _load_embedding_model():
    model_embedding = config["model"]["embedding"]

    if "bge-m3" in model_embedding.lower():
        from FlagEmbedding import BGEM3FlagModel

        model = BGEM3FlagModel(model_embedding, use_fp16=True)
        print(f"✓ Loaded hybrid embedding model: {model_embedding}")
        return model

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_embedding)
    print(f"✓ Loaded dense embedding model: {model_embedding}")
    return model


# =========================================================
# RERANKER LOADER
# =========================================================
def _load_reranker_model():
    model_reranker = config["model"]["reranker"]

    reranker = HFTransformerReranker(model_name=model_reranker)

    print(f"✓ Loaded reranker: {model_reranker}")
    return reranker


# =========================================================
# CAPABILITIES SAFE (BGE-M3 FIX)
# =========================================================
def _detect_model_capabilities(model) -> dict[str, Any]:
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

        # ---------------- bge-m3 dict output ----------------
        if isinstance(result, dict):
            dense = result.get("dense_vecs")

            if dense is not None and len(dense) > 0:
                vec = dense[0]
                capabilities["has_dense"] = True
                capabilities["dense_dim"] = len(vec)

            if result.get("lexical_weights") is not None:
                capabilities["has_sparse"] = True

            return capabilities

        # ---------------- numpy / list / tensor ----------------
        if result is not None:

            # numpy array safe check
            try:
                import numpy as np
                if isinstance(result, np.ndarray):
                    if result.size > 0:
                        vec = result[0]
                        capabilities["has_dense"] = True
                        capabilities["dense_dim"] = len(vec)
                        return capabilities
            except Exception:
                pass

            # list / tensor fallback
            try:
                if hasattr(result, "__len__") and len(result) > 0:
                    vec = result[0]

                    # si déjà vector 1D
                    if isinstance(vec, (float, int)):
                        vec = result

                    capabilities["has_dense"] = True
                    capabilities["dense_dim"] = len(vec)
            except Exception:
                pass

        return capabilities

    except Exception as e:
        raise ValueError(f"Embedding capability detection failed: {e}") from e


# =========================================================
# INIT
# =========================================================
client = _load_qdrant_client()

EMBEDDING_MODEL_NAME = config["model"]["embedding"]
RERANKER_MODEL_NAME = config["model"]["reranker"]

EMBEDDING_MODEL = _load_embedding_model()
RERANKER_MODEL = _load_reranker_model()

MODEL_CAPABILITIES = _detect_model_capabilities(EMBEDDING_MODEL)

model = EMBEDDING_MODEL
reranker = RERANKER_MODEL

COLLECTION = config["collection"]["name"]

BATCH_SIZE = int(config["indexing"].get("batch_size", 5))
BATCHSIZE = BATCH_SIZE

DOCUMENTS_PATH = config["indexing"].get("documents_path", "documents")

SEARCH_LIMIT = int(config.get("search", {}).get("limit", 3))
CANDIDATE_MULTIPLIER = int(config.get("search", {}).get("candidate_multiplier", 10))
MIN_CANDIDATES = int(config.get("search", {}).get("min_candidates", 30))
SCROLL_LIMIT_PER_DOC = int(config.get("search", {}).get("scroll_limit_per_doc", 10000))
WINDOW_RADIUS = int(config.get("search", {}).get("window_radius", 4))

FULL_DOCUMENT_CHUNK_THRESHOLD = int(
    config.get("search", {}).get("full_document_chunk_threshold", 12)
)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


print(
    "✓ Embedding model capabilities:",
    {
        "has_dense": MODEL_CAPABILITIES["has_dense"],
        "has_sparse": MODEL_CAPABILITIES["has_sparse"],
        "dense_dim": MODEL_CAPABILITIES["dense_dim"],
    },
)

print(f"✓ Reranker ready: {RERANKER_MODEL_NAME}")
