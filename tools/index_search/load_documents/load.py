from pathlib import Path
from hashlib import sha256
import re
import codecs
import json
import csv
import os
from io import StringIO
from typing import Any
import pymupdf4llm
import base64

from dotenv import load_dotenv
from openai import OpenAI

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseVector,
    PayloadSchemaType,
)

try:
    from .load_documents import config as cf
except ImportError:
    import config as cf


# ─── Chargement du .env à la racine du projet ───────────────────────────────
_ROOT_ENV = Path(__file__).resolve().parent
while _ROOT_ENV.parent != _ROOT_ENV:
    if (_ROOT_ENV / ".env").exists():
        break
    _ROOT_ENV = _ROOT_ENV.parent
load_dotenv(_ROOT_ENV / ".env")

# ─── Client LLM (API OpenAI-compatible) ──────────────────────────────────
_LLM_API_KEY = os.getenv("LLM_API_KEY", "")
_URL_API = os.getenv("URL_LLM_API", "")
_llm_client = OpenAI(
    base_url=_URL_API,
    api_key=_LLM_API_KEY,
)
_LLM_MODEL = os.getenv("LLM_MODEL", "")
_SUMMARY_MAX_INPUT_CHARS = 500_000

client = cf.client
model = cf.model
COLLECTION = cf.COLLECTION
BATCH_SIZE = cf.BATCH_SIZE


XML_DECL_RE = re.compile(br'<\?xml[^>]+encoding\s*=\s*["\']([^"\']+)', re.IGNORECASE)
TEXT_BOMS = (
    codecs.BOM_UTF8,
    codecs.BOM_UTF16_LE,
    codecs.BOM_UTF16_BE,
    codecs.BOM_UTF32_LE,
    codecs.BOM_UTF32_BE,
)
TEXT_WHITELIST = set(b"\t\n\r")
PRINTABLE_ASCII = set(range(32, 127))

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

CHUNK_SIZE = int(
    getattr(cf, "config", {}).get("chunking", {}).get("chunk_size", 4000)
    if hasattr(cf, "config")
    else 4000
)

CHUNK_OVERLAP = int(
    getattr(cf, "config", {}).get("chunking", {}).get("chunk_overlap", 400)
    if hasattr(cf, "config")
    else 400
)

_SUMMARY_SYSTEM_PROMPT = (
    "Tu es un assistant expert en résumé de documents techniques et réglementaires. "
    "Produis un résumé factuel, neutre et très concis du document fourni. "
    "Le résumé doit :\n"
    "- Présenter l'objet principal du document en une phrase.\n"
    "- Optionnellement lister 1 à 2 points clés.\n"
    "- Ne rien inventer ni extrapoler au-delà du texte fourni.\n"
    "- Être rédigé en français.\n"
    "- Avoir une longueur totale comprise entre 200 et 500 caractères MAXIMUM."
)


# ─── Génération du résumé ────────────────────────────────────────────────────

def generate_doc_summary(filename: str, full_text: str) -> str:
    """
    Appelle un LLM (via API OpenAI-compatible) pour produire un résumé du document.
    Tronque à _SUMMARY_MAX_INPUT_CHARS si nécessaire.
    Retourne une chaîne vide en cas d'échec (non bloquant pour l'indexation).
    """
    if not full_text.strip():
        return ""

    truncated = False
    text_input = full_text
    if len(full_text) > _SUMMARY_MAX_INPUT_CHARS:
        text_input = full_text[:_SUMMARY_MAX_INPUT_CHARS]
        truncated = True
        print(
            f"[~] '{filename}' truncated for summary: "
            f"{len(full_text)} → {_SUMMARY_MAX_INPUT_CHARS} chars"
        )

    user_message = (
        f"Voici le contenu du document '{filename}'"
        + (" (tronqué, suite non disponible)" if truncated else "")
        + f" :\n\n{text_input}\n\nRésume ce document selon les instructions."
    )

    try:
        response = _llm_client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        summary = response.choices[0].message.content or ""
        return summary.strip()
    except Exception as e:
        print(f"[!] Summary generation failed for '{filename}': {e}")
        return ""


# ─── Détection des capacités du modèle ──────────────────────────────────────

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

            if dense_vecs is not None:
                first_dense = dense_vecs[0] if len(dense_vecs) > 0 else None
                if first_dense is not None:
                    capabilities["has_dense"] = True
                    capabilities["dense_dim"] = len(first_dense)

            if lexical_weights is not None:
                capabilities["has_sparse"] = True

            print(
                f"Model capabilities detected: "
                f"dense={capabilities['has_dense']}, "
                f"sparse={capabilities['has_sparse']}, "
                f"dense_dim={capabilities['dense_dim']}"
            )
            return capabilities

    except Exception:
        pass

    try:
        dense = model.encode([test_text])
        first_dense = dense[0] if hasattr(dense, "__len__") else dense
        capabilities["has_dense"] = True
        capabilities["dense_dim"] = len(first_dense)
        print(
            f"Model capabilities detected: dense=True, sparse=False, "
            f"dense_dim={capabilities['dense_dim']}"
        )
        return capabilities
    except Exception as e:
        raise ValueError(f"Could not detect model capabilities: {e}")


# ─── Détection d'encodage ────────────────────────────────────────────────────

def detect_bom_encoding(raw: bytes) -> str | None:
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw.startswith(codecs.BOM_UTF32_LE):
        return "utf-32-le"
    if raw.startswith(codecs.BOM_UTF32_BE):
        return "utf-32-be"
    if raw.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    if raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    return None


def detect_xml_decl_encoding(raw: bytes) -> str | None:
    head = raw[:2048]
    match = XML_DECL_RE.search(head)
    if match:
        try:
            return match.group(1).decode("ascii").lower()
        except Exception:
            return None
    return None


def is_probably_binary(filepath: Path, chunk_size: int = 4096) -> bool:
    raw = filepath.read_bytes()
    if not raw:
        return False

    sample = raw[:chunk_size]

    if any(sample.startswith(bom) for bom in TEXT_BOMS):
        return False

    if b"\x00" in sample:
        return True

    odd = 0
    for b in sample:
        if b in TEXT_WHITELIST or b in PRINTABLE_ASCII:
            continue
        if b >= 128:
            continue
        odd += 1

    return (odd / len(sample)) > 0.30


def guess_text_encodings(raw: bytes) -> list[str]:
    candidates = []

    bom_encoding = detect_bom_encoding(raw)
    if bom_encoding:
        candidates.append(bom_encoding)

    xml_decl_encoding = detect_xml_decl_encoding(raw)
    if xml_decl_encoding:
        candidates.append(xml_decl_encoding)

    candidates.extend(
        [
            "utf-8",
            "utf-8-sig",
            "utf-16",
            "utf-16-le",
            "utf-16-be",
            "utf-32",
            "utf-32-le",
            "utf-32-be",
            "cp1252",
            "latin-1",
        ]
    )

    deduped = []
    seen = set()
    for enc in candidates:
        if enc and not enc in seen:
            deduped.append(enc)
            seen.add(enc)

    return deduped


# ─── Lecture des fichiers ────────────────────────────────────────────────────

def read_text_document(filepath: Path) -> tuple[str, str]:
    raw = filepath.read_bytes()

    if is_probably_binary(filepath):
        raise ValueError(f"{filepath.name} appears to be a binary/non-text file")

    encodings_to_try = guess_text_encodings(raw)

    for encoding in encodings_to_try:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
        except LookupError:
            continue

    raise ValueError(
        f"Unable to decode file {filepath.name}. "
        f"Tried encodings: {', '.join(encodings_to_try)}"
    )


def read_pdf_document(filepath: Path) -> tuple[str, str]:
    print(f"[~] Converting PDF to Markdown via pymupdf4llm: {filepath.name}")
    md_text = pymupdf4llm.to_markdown(str(filepath))
    return md_text, "pdf-to-markdown"


# ─── Optimisation du contenu ─────────────────────────────────────────────────

def optimize_json_preserving_standards(data: Any) -> str:
    def transform(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, str):
                    out[k] = v.strip()
                else:
                    out[k] = transform(v)
            return out
        if isinstance(obj, list):
            return [transform(item) for item in obj]
        if isinstance(obj, str):
            return obj.strip()
        return obj

    optimized = transform(data)
    return json.dumps(optimized, separators=(",", ":"), ensure_ascii=False)


def optimize_xml_preserving_standards(xml_content: str) -> str:
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_content)

        def strip_whitespace(elem):
            if elem.text is not None:
                elem.text = elem.text.strip() or None
            if elem.tail is not None:
                elem.tail = elem.tail.strip() or None
            for child in list(elem):
                strip_whitespace(child)

        strip_whitespace(root)
        return ET.tostring(root, encoding="unicode", method="xml")
    except Exception:
        compact = re.sub(r">\s+<", "><", xml_content)
        return compact.strip()


def optimize_rdflike_text(content: str) -> str:
    lines = []
    previous_blank = False

    for line in content.splitlines():
        stripped = line.strip()

        if not stripped:
            if not previous_blank:
                lines.append("")
                previous_blank = True
            continue

        previous_blank = False

        if stripped.startswith("@"):
            lines.append(stripped)
            continue

        compressed = re.sub(r"\s+", " ", stripped)
        lines.append(compressed)

    return "\n".join(lines)


def optimize_yaml_content(text: str) -> str:
    try:
        import yaml

        data = yaml.safe_load(text)
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        lines = []
        for line in text.splitlines():
            if not line.strip():
                continue
            lines.append(line.rstrip())
        return "\n".join(lines)


def optimize_csv_content(text: str) -> str:
    try:
        reader = csv.reader(StringIO(text))
        output = StringIO()
        writer = csv.writer(output, lineterminator="\n")

        for row in reader:
            if not row:
                continue
            cleaned = [cell.strip() for cell in row]
            if not any(cleaned):
                continue
            writer.writerow(cleaned)

        return output.getvalue().strip()
    except Exception:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lines.append(",".join(part.strip() for part in stripped.split(",")))
        return "\n".join(lines)


def optimize_html_content(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r">\s+<", "><", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def optimize_markdown_content(text: str) -> str:
    lines = []
    previous_blank = False

    for line in text.splitlines():
        stripped = line.rstrip()

        if not stripped.strip():
            if not previous_blank:
                lines.append("")
                previous_blank = True
            continue

        lines.append(stripped)
        previous_blank = False

    return "\n".join(lines).strip()


def optimize_generic_text(text: str) -> str:
    lines = []
    previous_blank = False

    for line in text.splitlines():
        stripped = line.rstrip()

        if not stripped.strip():
            if not previous_blank:
                lines.append("")
                previous_blank = True
            continue

        lines.append(re.sub(r"\s+", " ", stripped))
        previous_blank = False

    return "\n".join(lines).strip()


def optimize_document_content(filepath: Path, text: str) -> tuple[str, bool]:
    ext = filepath.suffix.lower()

    try:
        if ext == ".json":
            data = json.loads(text)
            optimized = optimize_json_preserving_standards(data)
            return optimized, optimized != text

        if ext in (".xml", ".xmi", ".xsd"):
            optimized = optimize_xml_preserving_standards(text)
            return optimized, optimized != text

        if ext in (".ttl", ".rdf", ".owl", ".n3", ".nt", ".trig"):
            optimized = optimize_rdflike_text(text)
            return optimized, optimized != text

        if ext in (".yaml", ".yml"):
            optimized = optimize_yaml_content(text)
            return optimized, optimized != text

        if ext == ".csv":
            optimized = optimize_csv_content(text)
            return optimized, optimized != text

        if ext in (".html", ".htm", ".xhtml"):
            optimized = optimize_html_content(text)
            return optimized, optimized != text

        if ext in (".md", ".markdown", ".rst", ".adoc", ".txt", ".pdf"):
            optimized = optimize_markdown_content(text)
            return optimized, optimized != text

        optimized = optimize_generic_text(text)
        if len(optimized) < len(text) * 0.95:
            return optimized, True

        return text, False

    except Exception as e:
        print(f"[!] Optimization failed for {filepath.name}: {e}")
        return text, False


# ─── Utilitaires d'indexation ────────────────────────────────────────────────

def generate_document_id(filepath: Path) -> str:
    base = str(filepath.resolve())
    return sha256(base.encode("utf-8")).hexdigest()[:24]


def generate_chunk_stable_id(document_id: str, chunk_index: int) -> int:
    base = f"{document_id}:{chunk_index}"
    hash_bytes = sha256(base.encode("utf-8")).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


def split_text_uniformly(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 10)

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            window_start = max(start, end - 300)
            newline_pos = text.rfind("\n", window_start, end)
            space_pos = text.rfind(" ", window_start, end)
            best_cut = max(newline_pos, space_pos)
            if best_cut > start + (chunk_size // 2):
                end = best_cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(start + step, end - overlap)

    return chunks


# ─── Gestion de la collection Qdrant ────────────────────────────────────────

def setup_collection(capabilities: dict[str, Any]) -> bool:
    collections = client.get_collections().collections
    collection_exists = any(c.name == COLLECTION for c in collections)

    if collection_exists:
        print(f"[!] Collection '{COLLECTION}' already exists.")
        choice = input("Delete and recreate? (y/n): ").strip().lower()
        if choice == "y":
            client.delete_collection(collection_name=COLLECTION)
            print(f"[✓] Deleted collection '{COLLECTION}'")
        else:
            print(f"[~] Using existing collection '{COLLECTION}'")
            return False

    if capabilities["has_dense"] and capabilities["has_sparse"]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=capabilities["dense_dim"],
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )
        print(
            f"[✓] Created hybrid collection '{COLLECTION}' "
            f"(dense:{capabilities['dense_dim']}, sparse:yes)"
        )

    elif capabilities["has_dense"]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=capabilities["dense_dim"],
                distance=Distance.COSINE,
            ),
        )
        print(
            f"[✓] Created dense-only collection '{COLLECTION}' "
            f"(dense:{capabilities['dense_dim']})"
        )

    elif capabilities["has_sparse"]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={},
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )
        print(f"[✓] Created sparse-only collection '{COLLECTION}'")

    else:
        raise ValueError("Model has neither dense nor sparse capability")

    print("[~] Creating payload index for 'tags'...")
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="tags",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("[✓] Payload index for 'tags' created.")

    return True


def get_existing_ids():
    try:
        existing_ids = set()
        next_offset = None

        while True:
            result = client.scroll(
                collection_name=COLLECTION,
                limit=10000,
                offset=next_offset,
                with_payload=False,
                with_vectors=False,
            )
            points, next_offset = result
            existing_ids.update(point.id for point in points)

            if next_offset is None:
                break

        print(f"[~] Found {len(existing_ids)} existing chunks")
        return existing_ids
    except Exception as e:
        print(f"[!] Failed to retrieve existing IDs: {e}")
        return set()


# ─── Encodage et construction des points ────────────────────────────────────

def _lexical_to_sparse_vector(lexical_weights: dict[Any, Any]) -> SparseVector:
    indices = [int(k) for k in lexical_weights.keys()]
    values = [float(v) for v in lexical_weights.values()]
    return SparseVector(indices=indices, values=values)


def encode_batch(texts: list[str], capabilities: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = []

    if capabilities["has_dense"] and capabilities["has_sparse"]:
        result = model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = result.get("dense_vecs", [])
        lexical_weights = result.get("lexical_weights", [])

        for i in range(len(texts)):
            item = {}
            if i < len(dense_vecs):
                dense = dense_vecs[i]
                item["dense"] = dense.tolist() if hasattr(dense, "tolist") else list(dense)
            if i < len(lexical_weights):
                item["sparse"] = _lexical_to_sparse_vector(lexical_weights[i])
            outputs.append(item)

        return outputs

    if capabilities["has_dense"]:
        dense_vecs = model.encode(texts)
        for dense in dense_vecs:
            outputs.append(
                {"dense": dense.tolist() if hasattr(dense, "tolist") else list(dense)}
            )
        return outputs

    if capabilities["has_sparse"]:
        result = model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        lexical_weights = result.get("lexical_weights", [])
        for lw in lexical_weights:
            outputs.append({"sparse": _lexical_to_sparse_vector(lw)})
        return outputs

    raise ValueError("No supported vector output available from model")


def build_point(
    docid: int,
    text: str,
    filename: str,
    encodingused: str,
    vectors: dict[str, Any],
    capabilities: dict[str, Any],
    payloadextra: dict[str, Any] | None = None,
) -> PointStruct:
    payload = {
        "text": text,
        "filename": filename,
        "encoding": encodingused,
    }
    if payloadextra:
        payload.update(payloadextra)

    if capabilities["has_dense"] and capabilities["has_sparse"]:
        return PointStruct(
            id=docid,
            vector={
                DENSE_VECTOR_NAME: vectors["dense"],
                SPARSE_VECTOR_NAME: vectors["sparse"],
            },
            payload=payload,
        )

    if capabilities["has_dense"]:
        return PointStruct(
            id=docid,
            vector=vectors["dense"],
            payload=payload,
        )

    if capabilities["has_sparse"]:
        return PointStruct(
            id=docid,
            vector={SPARSE_VECTOR_NAME: vectors["sparse"]},
            payload=payload,
        )

    raise ValueError("Unable to build point: no vectors available")


def flush_batch(
    batch_docs: list[dict[str, Any]],
    points: list[PointStruct],
    capabilities: dict[str, Any],
) -> int:
    texts_to_embed = [
        f"Source : {', '.join(item['payloadextra'].get('tags', []))}\nFichier : {item['filename']}\n\n{item['text']}" 
        for item in batch_docs
    ]
    
    encoded_vectors = encode_batch(texts_to_embed, capabilities)

    for item, vectors in zip(batch_docs, encoded_vectors):
        point = build_point(
            docid=item["docid"],
            text=item["text"],  # Le payload UI conserve le texte brut (sans l'ajout)
            filename=item["filename"],
            encodingused=item["encoding"],
            vectors=vectors,
            capabilities=capabilities,
            payloadextra=item.get("payloadextra"),
        )
        points.append(point)
        print(
            f"[~] Loaded {item['filename']} "
            f"(chunk {item['payloadextra'].get('chunk_index', 0) + 1}/"
            f"{item['payloadextra'].get('chunk_count', 1)}) "
            f"({item['encoding']})"
        )

    client.upsert(collection_name=COLLECTION, points=points)
    uploaded = len(points)
    print(f"[✓] Uploaded batch of {uploaded}")

    points.clear()
    batch_docs.clear()

    return uploaded


# ─── Indexation principale ───────────────────────────────────────────────────

def index_documents():
    print("Detecting model capabilities...")
    capabilities = cf.MODEL_CAPABILITIES

    is_fresh = setup_collection(capabilities)
    existing_ids = set() if is_fresh else get_existing_ids()

    docs_path = Path(__file__).parent / "documents"

    total_indexed = 0
    total_skipped = 0
    total_nontext = 0
    total_optimized = 0
    total_chunks = 0

    batch_docs = []
    points = []

    if not docs_path.exists():
        print(f"[!] Directory not found: {docs_path}")
        return

    print(f"[~] Scanning documents recursively in {docs_path}")

    def pushdoc(
        filename: str,
        content: str,
        encodingused: str,
        payloadextra: dict[str, Any] | None = None,
    ):
        nonlocal total_skipped

        payloadextra = payloadextra or {}
        document_id = str(payloadextra.get("document_id"))
        chunk_index = int(payloadextra.get("chunk_index", 0))
        docid = generate_chunk_stable_id(document_id, chunk_index)

        if not is_fresh and docid in existing_ids:
            print(f"[~] Skipped duplicate: {filename} chunk={chunk_index}")
            total_skipped += 1
            return

        batch_docs.append(
            {
                "filename": filename,
                "text": content,
                "encoding": encodingused,
                "docid": docid,
                "payloadextra": payloadextra,
            }
        )

    for filepath in docs_path.rglob("*"):
        if not filepath.is_file():
            continue

        relative_path = filepath.relative_to(docs_path)
        tags = list(relative_path.parent.parts)

        ext = filepath.suffix.lower()

        try:
            file_base64 = None
            try:
                with open(filepath, "rb") as f:
                    file_base64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                print(f"[!] Warning: Could not base64 encode {filepath.name}: {e}")

            if ext == ".pdf":
                text, encodingused = read_pdf_document(filepath)
            else:
                text, encodingused = read_text_document(filepath)

            optimized_text, was_optimized = optimize_document_content(filepath, text)

            if was_optimized:
                total_optimized += 1
                origin_len = len(text)
                optimized_len = len(optimized_text)
                reduction = ((1 - (optimized_len / origin_len)) * 100) if origin_len else 0
                print(
                    f"[~] Optimized {filepath.name}: "
                    f"{origin_len} → {optimized_len} chars "
                    f"({reduction:.1f}% reduction)"
                )

            document_id = generate_document_id(filepath)
            chunks = split_text_uniformly(optimized_text)

            if not chunks:
                continue

            total_chunks += len(chunks)
            print(f"[~] Chunked {filepath.name} into {len(chunks)} chunks (Tags: {tags})")

            # ── Génération du résumé (une fois par document, si non déjà indexé) ──
            chunk0_id = generate_chunk_stable_id(document_id, 0)
            if not is_fresh and chunk0_id in existing_ids:
                print(f"[~] Document already indexed, skipping summary: {filepath.name}")
                doc_summary = None
            else:
                print(f"[~] Generating summary for '{filepath.name}'...")
                doc_summary = generate_doc_summary(filepath.name, optimized_text)
                if doc_summary:
                    print(f"[✓] Summary generated ({len(doc_summary)} chars)")
                else:
                    print(f"[!] No summary generated for '{filepath.name}'")
            # ─────────────────────────────────────────────────────────────────────

            for chunk_index, chunk_text in enumerate(chunks):
                payload_extra = {
                    "doctype": "uniform_chunk",
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "is_child_chunk": True,
                    "source_path": str(filepath.resolve()),
                    "source_extension": filepath.suffix.lower(),
                    "document_name": filepath.stem,
                    "tags": tags,
                }

                # Chunk 0 : résumé + base64
                if chunk_index == 0:
                    if doc_summary:
                        payload_extra["doc_summary"] = doc_summary
                    if file_base64:
                        payload_extra["file_base64"] = file_base64

                pushdoc(
                    filename=filepath.name,
                    content=chunk_text,
                    encodingused=encodingused,
                    payloadextra=payload_extra,
                )

                if len(batch_docs) >= BATCH_SIZE:
                    try:
                        total_indexed += flush_batch(batch_docs, points, capabilities)
                    except Exception as e:
                        print(f"[!] Batch upload failed: {e}")
                        batch_docs.clear()
                        points.clear()

        except ValueError as e:
            print(f"[!] Skipped non-text file {filepath.name}: {e}")
            total_nontext += 1
        except Exception as e:
            print(f"[!] Failed to read {filepath.name}: {e}")

    if batch_docs:
        try:
            total_indexed += flush_batch(batch_docs, points, capabilities)
        except Exception as e:
            print(f"[!] Final batch upload failed: {e}")

    print("=" * 50)
    print("Indexing complete!")
    print(f"[✓] Chunks indexed: {total_indexed}")
    print(f"[✓] Documents optimized: {total_optimized}")
    print(f"[✓] Total chunks created: {total_chunks}")
    if total_skipped > 0:
        print(f"[~] Chunks skipped (duplicates): {total_skipped}")
    if total_nontext > 0:
        print(f"[!] Files skipped (non-text): {total_nontext}")
    print("=" * 50)


if __name__ == "__main__":
    index_documents()
