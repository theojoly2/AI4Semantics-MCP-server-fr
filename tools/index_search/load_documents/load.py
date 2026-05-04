from pathlib import Path
from hashlib import sha256
import re
import codecs
import json
import csv
from io import StringIO
from typing import Any, Optional
import xml.etree.ElementTree as ET

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseVector,
)

try:
    from .load_documents import config as cf
except ImportError:
    import config as cf

try:
    import xmlschema
except ImportError:
    xmlschema = None


def _extract_vocabulary_from_filename(filename: str) -> Optional[str]:
    """
    Extract vocabulary prefix from filename.
    E.g. 'CPV_AP_SHACL_96.json' -> 'CPV'
    """
    if not filename:
        return None
    
    # Liste des vocabulaires valides
    VOCABULARIES = [
        "CAV", "CBV", "CCCEV", "CLV", "CPEV", "CPOV", "CPSV", "CPV",
        "SDG-AC", "SDG-ACM", "SDG-AERP", "SDG-AL", "SDG-AP", "SDG-AV",
        "SDG-BS", "SDG-BUDG", "SDG-CATAL", "SDG-CDBRRC", "SDG-CMC",
        "SDG-CMM", "SDG-COLL", "SDG-DAE", "SDG-DECP", "SDG-DELIB",
        "SDG-DISAI", "SDG-DOCP", "SDG-DONNA", "SDG-DYNA", "SDG-EA",
        "SDG-ENSPAY", "SDG-EQUIP", "SDG-FONTEA", "SDG-FRI", "SDG-HR",
        "SDG-IDLL", "SDG-IDT", "SDG-IDYN", "SDG-IR", "SDG-IRA",
        "SDG-ISPN", "SDG-ISTAT", "SDG-IT", "SDG-LC", "SDG-LDP",
        "SDG-LSTAT", "SDG-MCOL", "SDG-OA", "SDG-OINS", "SDG-ONP",
        "SDG-PASSN", "SDG-PEI", "SDG-PMCOLL", "SDG-PNN", "SDG-PROG",
        "SDG-PTER", "SDG-REA", "SDG-SC", "SDG-SCMS", "SDG-SDIR",
        "SDG-SECT", "SDG-SEE", "SDG-SEPE", "SDG-SESE", "SDG-SETE",
        "SDG-SFDPA", "SDG-SMN", "SDG-SP", "SDG-ST", "SDG-SUBV",
        "SDG-UP", "SDG-VFERP", "SDG-VFERPS", "SDG-ZFE",
    ]
    
    prefix = filename.split("_", 1)[0]
    return prefix if prefix in VOCABULARIES else None


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

XML_SPLIT_THRESHOLD = 180_000
XML_MAX_CHUNK_CHARS = 120_000
XML_GLOBAL_COMPONENT_TAGS = (
    "complexType",
    "simpleType",
    "element",
    "group",
    "attributeGroup",
    "attribute",
)
XSD_REL_ATTRS = ("base", "ref", "type", "substitutionGroup")


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

    candidates.extend([
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
    ])

    deduped = []
    seen = set()
    for enc in candidates:
        if enc and enc not in seen:
            deduped.append(enc)
            seen.add(enc)

    return deduped


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
        root = ET.fromstring(xml_content)

        def strip_whitespace(elem: ET.Element) -> None:
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


def localname(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def strip_prefix(value: str) -> str:
    return value.split(":")[-1].strip() if value else value


def collect_documentation_text(elem: ET.Element) -> list[str]:
    docs = []
    for sub in elem.iter():
        if localname(sub.tag) == "documentation":
            text = " ".join("".join(sub.itertext()).split())
            if text:
                docs.append(text)
    return docs[:10]


def collect_appinfo_text(elem: ET.Element) -> list[str]:
    appinfos = []
    for sub in elem.iter():
        if localname(sub.tag) == "appinfo":
            text = " ".join("".join(sub.itertext()).split())
            if text:
                appinfos.append(text)
            else:
                raw = ET.tostring(sub, encoding="unicode", method="xml")
                appinfos.append(raw[:2000])
    return appinfos[:10]


def extract_xsd_dependencies_from_element(elem: ET.Element) -> dict[str, list[str]]:
    deps = {
        "base": [],
        "ref": [],
        "type": [],
        "substitutionGroup": [],
        "groupref": [],
        "attributeGroupref": [],
    }

    for sub in elem.iter():
        tag = localname(sub.tag)

        for attr_key, attr_value in sub.attrib.items():
            attr_name = localname(attr_key)
            if attr_name in XSD_REL_ATTRS and attr_value:
                deps[attr_name].append(strip_prefix(attr_value))

        if tag == "group" and "ref" in sub.attrib:
            deps["groupref"].append(strip_prefix(sub.attrib["ref"]))

        if tag == "attributeGroup" and "ref" in sub.attrib:
            deps["attributeGroupref"].append(strip_prefix(sub.attrib["ref"]))

    cleaned = {}
    for key, values in deps.items():
        seen = set()
        uniq = []
        for v in values:
            if v and v not in seen:
                seen.add(v)
                uniq.append(v)
        cleaned[key] = uniq

    return cleaned


def summarize_dependencies(depmap: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    flat = []
    labeled = []

    for rel_type in ["base", "type", "ref", "substitutionGroup", "groupref", "attributeGroupref"]:
        for item in depmap.get(rel_type, []):
            flat.append(item)
            labeled.append(f"{rel_type}:{item}")

    flat_unique = []
    seen = set()
    for item in flat:
        if item not in seen:
            seen.add(item)
            flat_unique.append(item)

    labeled_unique = []
    seen = set()
    for item in labeled:
        if item not in seen:
            seen.add(item)
            labeled_unique.append(item)

    return flat_unique, labeled_unique


def extract_schema_link_directives(root: ET.Element) -> dict[str, list[str]]:
    links = {
        "includes": [],
        "imports": [],
        "redefines": [],
        "overrides": [],
    }

    for child in list(root):
        tag = localname(child.tag)
        schema_loc = child.attrib.get("schemaLocation")

        if not schema_loc:
            continue

        if tag == "include":
            links["includes"].append(schema_loc)
        elif tag == "import":
            links["imports"].append(schema_loc)
        elif tag == "redefine":
            links["redefines"].append(schema_loc)
        elif tag == "override":
            links["overrides"].append(schema_loc)

    for key, values in links.items():
        seen = set()
        uniq = []
        for v in values:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        links[key] = uniq

    return links


def resolve_local_schema_links(
    filepath: Path, schema_links: dict[str, list[str]]
) -> dict[str, list[str]]:
    resolved = {
        "includes": [],
        "imports": [],
        "redefines": [],
        "overrides": [],
    }

    basedir = filepath.parent

    for key, items in schema_links.items():
        for item in items:
            candidate = (basedir / item).resolve()
            if candidate.exists():
                resolved[key].append(str(candidate))
            else:
                resolved[key].append(item)

    return resolved


def build_component_registry_etree(
    root: ET.Element, filepath: Path
) -> list[dict[str, Any]]:
    registry = []

    for child in list(root):
        kind = localname(child.tag)

        if kind not in XML_GLOBAL_COMPONENT_TAGS:
            continue

        name = (
            child.attrib.get("name")
            or child.attrib.get("ref")
            or child.attrib.get("id")
        )

        if not name:
            name = f"anonymous_{len(registry)+1}"

        xml_text = ET.tostring(child, encoding="unicode", method="xml")
        xml_text = optimize_xml_preserving_standards(xml_text)

        dep_map = extract_xsd_dependencies_from_element(child)
        depends_on, depends_on_labeled = summarize_dependencies(dep_map)

        registry.append(
            {
                "componentid": f"{kind}:{strip_prefix(name)}",
                "componentname": strip_prefix(name),
                "componentkind": kind,
                "targetnamespace": root.attrib.get("targetNamespace", ""),
                "schemafile": filepath.name,
                "xml": xml_text,
                "documentation": collect_documentation_text(child),
                "appinfo": collect_appinfo_text(child),
                "depmap": dep_map,
                "dependson": depends_on,
                "dependsonlabeled": depends_on_labeled,
            }
        )

    return registry


def build_component_registry_xmlschema(
    filepath: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if xmlschema is None:
        return [], {}

    try:
        schema = xmlschema.XMLSchema(str(filepath))
    except Exception:
        return [], {}

    registry = []

    try:
        root_elem = ET.fromstring(filepath.read_text(encoding="utf-8", errors="ignore"))
        schema_links = resolve_local_schema_links(
            filepath, extract_schema_link_directives(root_elem)
        )
    except Exception:
        schema_links = {"includes": [], "imports": [], "redefines": [], "overrides": []}

    for component in schema.iter_components():
        cls_name = component.__class__.__name__
        name = getattr(component, "name", None)

        if not name:
            continue

        target_namespace = getattr(component, "target_namespace", None) or getattr(
            schema, "target_namespace", ""
        )

        component_id = f"{cls_name}:{strip_prefix(name)}"

        xml_text = ""
        elem = getattr(component, "elem", None)
        if elem is not None:
            try:
                xml_text = ET.tostring(elem, encoding="unicode", method="xml")
                xml_text = optimize_xml_preserving_standards(xml_text)
            except Exception:
                xml_text = ""

        documentation = []
        appinfo = []
        dep_map = {
            "base": [],
            "ref": [],
            "type": [],
            "substitutionGroup": [],
            "groupref": [],
            "attributeGroupref": [],
        }

        if elem is not None:
            documentation = collect_documentation_text(elem)
            appinfo = collect_appinfo_text(elem)
            dep_map = extract_xsd_dependencies_from_element(elem)

        depends_on, depends_on_labeled = summarize_dependencies(dep_map)

        registry.append(
            {
                "componentid": component_id,
                "componentname": strip_prefix(name),
                "componentkind": cls_name,
                "targetnamespace": target_namespace,
                "schemafile": filepath.name,
                "xml": xml_text,
                "documentation": documentation,
                "appinfo": appinfo,
                "depmap": dep_map,
                "dependson": depends_on,
                "dependsonlabeled": depends_on_labeled,
            }
        )

    return registry, schema_links


def build_reverse_references(registry: list[dict[str, Any]]) -> dict[str, list[str]]:
    reverse = {}

    for item in registry:
        source_id = item["componentid"]
        source_name = item["componentname"]

        reverse.setdefault(source_name, [])

        for dep in item["dependson"]:
            reverse.setdefault(dep, []).append(source_id)

    for key, refs in reverse.items():
        seen = set()
        uniq = []
        for ref in refs:
            if ref not in seen:
                seen.add(ref)
                uniq.append(ref)
        reverse[key] = uniq

    return reverse


def build_parent_xml_summary(
    filepath: Path,
    root_tag: str,
    registry: list[dict[str, Any]],
    reverse_refs: dict[str, list[str]],
    schema_links: dict[str, list[str]],
) -> str:
    counts = {}
    all_ids = []

    for item in registry:
        counts[item["componentkind"]] = counts.get(item["componentkind"], 0) + 1
        all_ids.append(item["componentid"])

    counts_text = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "none"
    ids_text = (
        ", ".join(all_ids[:40]) + ", ..." if len(all_ids) > 40 else ", ".join(all_ids)
    )

    most_referenced = sorted(
        [(name, len(refs)) for name, refs in reverse_refs.items() if refs],
        key=lambda x: x[1],
        reverse=True,
    )
    most_referenced_text = (
        ", ".join(f"{name}({count})" for name, count in most_referenced[:20]) or "none"
    )

    return f"""--- XML DOCUMENT SUMMARY ---
SOURCE DOCUMENT: {filepath.name}
ROOT ELEMENT: {root_tag}
TOTAL COMPONENTS: {len(registry)}
COMPONENT BREAKDOWN: {counts_text}
KNOWN COMPONENTS: {ids_text}
INCLUDES: {", ".join(schema_links.get("includes", [])) or "none"}
IMPORTS: {", ".join(schema_links.get("imports", [])) or "none"}
REDEFINES: {", ".join(schema_links.get("redefines", [])) or "none"}
OVERRIDES: {", ".join(schema_links.get("overrides", [])) or "none"}
MOST REFERENCED COMPONENTS: {most_referenced_text}

NOTE: Parent summary of a larger XML/XSD standard, split into semantic chunks while preserving raw XML and schema relations.
--- XML DOCUMENT SUMMARY ---"""


def build_xml_chunk_text(
    item: dict[str, Any],
    source_file: str,
    previous_chunk: str | None = None,
    next_chunk: str | None = None,
    referenced_by: list[str] = [],
    schema_links: dict[str, list[str]] = {},
) -> str:
    docs_text = " ".join(item["documentation"][:5]) if item["documentation"] else "none"
    appinfo_text = " ".join(item["appinfo"][:3]) if item["appinfo"] else "none"
    deps_text = (
        ", ".join(item["dependsonlabeled"][:25])
        if item["dependsonlabeled"]
        else "none"
    )
    refs_text = ", ".join(referenced_by[:25]) if referenced_by else "none"

    header = f"""--- XML CHUNK CONTEXT ---
SOURCE DOCUMENT: {source_file}
COMPONENT ID: {item["componentid"]}
COMPONENT KIND: {item["componentkind"]}
COMPONENT NAME: {item["componentname"]}
TARGET NAMESPACE: {item["targetnamespace"] or "none"}
SCHEMA FILE: {item["schemafile"]}
PREVIOUS CHUNK: {previous_chunk or "none"}
NEXT CHUNK: {next_chunk or "none"}
DEPENDS ON (LABELED): {deps_text}
REFERENCED BY: {refs_text}
INCLUDES: {", ".join(schema_links.get("includes", [])) or "none"}
IMPORTS: {", ".join(schema_links.get("imports", [])) or "none"}
REDEFINES: {", ".join(schema_links.get("redefines", [])) or "none"}
OVERRIDES: {", ".join(schema_links.get("overrides", [])) or "none"}
DOCUMENTATION: {docs_text}
APPINFO: {appinfo_text}
--- XML CHUNK CONTEXT ---

"""

    body = item["xml"] or "<!-- NO RAW XML COMPONENT AVAILABLE -->"

    footer = f"""

--- XML CHUNK RELATIONS ---
SOURCE DOCUMENT: {source_file}
COMPONENT ID: {item["componentid"]}
DEPENDS ON (LABELED): {deps_text}
REFERENCED BY: {refs_text}

NOTE: This chunk preserves raw XML plus extracted schema relations and annotations.
--- XML CHUNK RELATIONS ---"""

    final_text = header + body + footer

    if len(final_text) > XML_MAX_CHUNK_CHARS:
        abstract = f"""--- XML COMPONENT ABSTRACT ---
COMPONENT ID: {item["componentid"]}
COMPONENT KIND: {item["componentkind"]}
COMPONENT NAME: {item["componentname"]}
TARGET NAMESPACE: {item["targetnamespace"] or "none"}
DOCUMENTATION: {docs_text}
APPINFO: {appinfo_text}
DEPENDENCIES: {deps_text}
--- XML COMPONENT ABSTRACT ---

"""
        final_text = header + abstract + body[:90000] + footer

    return final_text


def detect_xml_type(root: ET.Element) -> str:
    """Détermine si c'est un XSD, du XMI/UML, ou un document XML d'instance."""
    root_tag = localname(root.tag)

    if root_tag in ("XMI", "Model", "uml:Model"):
        return "xmi"

    for attr_name in root.attrib.keys():
        if "xmi" in attr_name.lower():
            return "xmi"

    has_uml_elements = False
    for child in list(root)[:10]:
        tag_lower = child.tag.lower()
        if "uml" in child.tag or "packagedelement" in tag_lower or "xmi" in tag_lower:
            has_uml_elements = True
            break

    if has_uml_elements:
        return "xmi"

    if root_tag == "schema":
        return "xsd"

    has_xsd_components = any(
        localname(child.tag) in XML_GLOBAL_COMPONENT_TAGS for child in root
    )
    if has_xsd_components:
        return "xsd"

    return "xmlinstance"


def extract_uml_element_info(elem: ET.Element) -> dict[str, Any]:
    """Extrait les infos importantes d'un élément UML."""
    tag = localname(elem.tag)

    xmi_type = (
        elem.attrib.get("{http://schema.omg.org/spec/XMI/2.1}type")
        or elem.attrib.get("{http://www.omg.org/spec/XMI/20131001}type")
        or elem.attrib.get("{http://www.omg.org/XMI}type")
        or elem.attrib.get("type")
        or tag
    )

    xmi_id = (
        elem.attrib.get("{http://schema.omg.org/spec/XMI/2.1}id")
        or elem.attrib.get("{http://www.omg.org/spec/XMI/20131001}id")
        or elem.attrib.get("{http://www.omg.org/XMI}id")
        or elem.attrib.get("id")
    )

    name = elem.attrib.get("name") or xmi_id or f"anonymous_{tag}"

    return {
        "tag": tag,
        "xmitype": strip_prefix(xmi_type),
        "xmiid": xmi_id,
        "name": name,
    }


def collect_uml_dependencies(elem: ET.Element) -> dict[str, list[str]]:
    """Collecte les dépendances UML: types, associations, généralisations, etc."""
    deps = {
        "type": [],
        "association": [],
        "generalization": [],
        "realization": [],
        "dependency": [],
        "aggregation": [],
        "idref": [],
    }

    for sub in elem.iter():
        type_ref = sub.attrib.get("type")
        if type_ref:
            deps["type"].append(strip_prefix(type_ref))

        assoc = sub.attrib.get("association")
        if assoc:
            deps["association"].append(strip_prefix(assoc))

        general = sub.attrib.get("general")
        if general:
            deps["generalization"].append(strip_prefix(general))

        supplier = sub.attrib.get("supplier")
        if supplier:
            deps["realization"].append(strip_prefix(supplier))

        client = sub.attrib.get("client")
        if client:
            deps["dependency"].append(strip_prefix(client))

        idref = sub.attrib.get("idref") or sub.attrib.get("href")
        if idref:
            deps["idref"].append(strip_prefix(idref))

    cleaned = {}
    for key, values in deps.items():
        seen = set()
        uniq = []
        for v in values:
            if v and v not in seen:
                seen.add(v)
                uniq.append(v)
        cleaned[key] = uniq

    return cleaned


def split_xmi_by_packagedelements(
    filepath: Path, root: ET.Element
) -> list[dict[str, Any]]:
    """Découpe un fichier XMI par packagedElement (classes, packages, associations, etc.)."""
    results = []
    packagedelements = []

    def collect_packagedelements(parent: ET.Element, depth: int = 0):
        """Collecte récursivement les packagedElement importants."""
        for child in parent:
            tag = localname(child.tag)

            if tag == "packagedElement" or "packagedElement" in child.tag:
                info = extract_uml_element_info(child)
                xml_text = ET.tostring(child, encoding="unicode", method="xml")
                xml_text = optimize_xml_preserving_standards(xml_text)

                if len(xml_text) < 3000:
                    continue

                deps = collect_uml_dependencies(child)
                deps_flat = []
                deps_labeled = []
                for rel_type, items in deps.items():
                    for item in items:
                        deps_flat.append(item)
                        deps_labeled.append(f"{rel_type}:{item}")

                deps_flat = list(dict.fromkeys(deps_flat))
                deps_labeled = list(dict.fromkeys(deps_labeled))

                packagedelements.append(
                    {
                        "tag": info["tag"],
                        "xmitype": info["xmitype"],
                        "xmiid": info["xmiid"],
                        "name": info["name"],
                        "xml": xml_text,
                        "depth": depth,
                        "dependson": deps_flat,
                        "dependsonlabeled": deps_labeled,
                        "index": len(packagedelements),
                    }
                )

            if tag in ("packagedElement", "Model", "Package") and depth < 3:
                collect_packagedelements(child, depth + 1)

    collect_packagedelements(root)

    if len(packagedelements) < 2:
        return []

    reverse_refs = {}
    for elem in packagedelements:
        elem_id = elem["xmiid"] or elem["name"]
        for dep in elem["dependson"]:
            reverse_refs.setdefault(dep, []).append(elem_id)

    root_info = extract_uml_element_info(root)

    element_types = {}
    for elem in packagedelements:
        element_types[elem["xmitype"]] = element_types.get(elem["xmitype"], 0) + 1

    types_text = ", ".join(f"{k}:{v}" for k, v in sorted(element_types.items())) or "none"
    ids_text = ", ".join(
        [e["xmiid"] or e["name"] for e in packagedelements[:40]]
    )
    if len(packagedelements) > 40:
        ids_text += ", ..."

    most_referenced = sorted(
        [(name, len(refs)) for name, refs in reverse_refs.items() if refs],
        key=lambda x: x[1],
        reverse=True,
    )
    most_referenced_text = (
        ", ".join(f"{name}({count})" for name, count in most_referenced[:20]) or "none"
    )

    parent_summary = f"""--- XMI DOCUMENT SUMMARY ---
SOURCE DOCUMENT: {filepath.name}
ROOT ELEMENT: {root_info["tag"]}
ROOT XMI TYPE: {root_info["xmitype"]}
ROOT NAME: {root_info["name"]}
TOTAL PACKAGED ELEMENTS: {len(packagedelements)}
ELEMENT TYPE BREAKDOWN: {types_text}
KNOWN ELEMENTS: {ids_text}
MOST REFERENCED ELEMENTS: {most_referenced_text}

NOTE: This is a parent summary of a UML/XMI model split by packagedElement while preserving raw XML and UML relations.
--- XMI DOCUMENT SUMMARY ---"""

    results.append(
        {
            "filename": f"{filepath.stem}_PARENT_SUMMARY.xmichunk",
            "text": parent_summary,
            "payloadextra": {
                "doctype": "xmi:parentsummary",
                "componentid": None,
                "componentkind": root_info["xmitype"],
                "componentname": root_info["name"],
                "targetnamespace": root.attrib.get("targetNamespace", ""),
                "dependson": [],
                "dependsonlabeled": [],
                "referencedby": [],
                "schemalinks": {},
            },
        }
    )

    element_ids = [e["xmiid"] or e["name"] for e in packagedelements]

    for i, elem in enumerate(packagedelements):
        previous_elem = element_ids[i - 1] if i > 0 else None
        next_elem = element_ids[i + 1] if i < len(element_ids) - 1 else None

        elem_id = elem["xmiid"] or elem["name"]

        referenced_by = reverse_refs.get(elem["xmiid"], []) + reverse_refs.get(
            elem["name"], []
        )
        referenced_by = list(dict.fromkeys(referenced_by))

        deps_text = (
            ", ".join(elem["dependsonlabeled"][:25])
            if elem["dependsonlabeled"]
            else "none"
        )
        refs_text = ", ".join(referenced_by[:25]) if referenced_by else "none"

        header = f"""--- XMI ELEMENT CONTEXT ---
SOURCE DOCUMENT: {filepath.name}
XMI ID: {elem["xmiid"] or "none"}
XMI TYPE: {elem["xmitype"]}
ELEMENT NAME: {elem["name"]}
ELEMENT INDEX: {i+1}/{len(packagedelements)}
PREVIOUS ELEMENT: {previous_elem or "none"}
NEXT ELEMENT: {next_elem or "none"}
DEPENDS ON (LABELED): {deps_text}
REFERENCED BY: {refs_text}
--- XMI ELEMENT CONTEXT ---

"""

        footer = f"""

--- XMI ELEMENT RELATIONS ---
SOURCE DOCUMENT: {filepath.name}
XMI ID: {elem["xmiid"] or "none"}
XMI TYPE: {elem["xmitype"]}
DEPENDS ON (LABELED): {deps_text}
REFERENCED BY: {refs_text}

NOTE: This chunk preserves raw XMI/UML element plus extracted model relations.
--- XMI ELEMENT RELATIONS ---"""

        final_text = header + elem["xml"] + footer

        if len(final_text) > XML_MAX_CHUNK_CHARS:
            abstract = f"""--- XMI ELEMENT ABSTRACT ---
XMI ID: {elem["xmiid"] or "none"}
XMI TYPE: {elem["xmitype"]}
ELEMENT NAME: {elem["name"]}
DEPENDENCIES: {deps_text}
--- XMI ELEMENT ABSTRACT ---

"""
            final_text = header + abstract + elem["xml"][:90000] + footer

        safe_type = re.sub(r"[^A-Za-z0-9._-]", "", elem["xmitype"])
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "", elem["name"])

        results.append(
            {
                "filename": f"{filepath.stem}_{safe_type}_{safe_name}.xmichunk",
                "text": final_text,
                "payloadextra": {
                    "doctype": "xmi:packagedelement",
                    "componentid": elem["xmiid"],
                    "componentkind": elem["xmitype"],
                    "componentname": elem["name"],
                    "targetnamespace": None,
                    "dependson": elem["dependson"],
                    "dependsonlabeled": elem["dependsonlabeled"],
                    "referencedby": referenced_by,
                    "schemalinks": {},
                },
            }
        )

    return results


def split_large_xml_semantically(
    filepath: Path, xml_content: str
) -> list[dict[str, Any]]:
    """Version améliorée qui détecte XSD vs XMI vs XML instance."""
    optimized_full = optimize_xml_preserving_standards(xml_content)

    if len(xml_content) < XML_SPLIT_THRESHOLD:
        return [
            {
                "filename": filepath.name,
                "text": optimized_full,
                "payloadextra": {
                    "doctype": "xml:full",
                    "componentid": None,
                    "componentkind": None,
                    "componentname": None,
                    "targetnamespace": None,
                    "dependson": [],
                    "dependsonlabeled": [],
                    "referencedby": [],
                    "schemalinks": {},
                },
            }
        ]

    try:
        root = ET.fromstring(xml_content)
    except Exception:
        return [
            {
                "filename": filepath.name,
                "text": optimized_full,
                "payloadextra": {
                    "doctype": "xml:full",
                    "componentid": None,
                    "componentkind": None,
                    "componentname": None,
                    "targetnamespace": None,
                    "dependson": [],
                    "dependsonlabeled": [],
                    "referencedby": [],
                    "schemalinks": {},
                },
            }
        ]

    xml_type = detect_xml_type(root)

    if xml_type == "xmi":
        xmi_chunks = split_xmi_by_packagedelements(filepath, root)
        if not xmi_chunks:
            return [
                {
                    "filename": filepath.name,
                    "text": optimized_full,
                    "payloadextra": {
                        "doctype": "xmi:full",
                        "componentid": None,
                        "componentkind": "XMI",
                        "componentname": filepath.stem,
                        "targetnamespace": None,
                        "dependson": [],
                        "dependsonlabeled": [],
                        "referencedby": [],
                        "schemalinks": {},
                    },
                }
            ]
        return xmi_chunks

    elif xml_type == "xsd":
        registry, schema_links = build_component_registry_xmlschema(filepath)

        if not registry:
            registry = build_component_registry_etree(root, filepath)
            schema_links = resolve_local_schema_links(
                filepath, extract_schema_link_directives(root)
            )

        if not registry:
            return [
                {
                    "filename": filepath.name,
                    "text": optimized_full,
                    "payloadextra": {
                        "doctype": "xml:full",
                        "componentid": None,
                        "componentkind": None,
                        "componentname": None,
                        "targetnamespace": root.attrib.get("targetNamespace", ""),
                        "dependson": [],
                        "dependsonlabeled": [],
                        "referencedby": [],
                        "schemalinks": schema_links,
                    },
                }
            ]

        reverse_refs = build_reverse_references(registry)
        chunk_ids = [item["componentid"] for item in registry]

        results = []

        parent_summary = build_parent_xml_summary(
            filepath=filepath,
            root_tag=localname(root.tag),
            registry=registry,
            reverse_refs=reverse_refs,
            schema_links=schema_links,
        )

        results.append(
            {
                "filename": f"{filepath.stem}_PARENT_SUMMARY.xmlchunk",
                "text": parent_summary,
                "payloadextra": {
                    "doctype": "xml:parentsummary",
                    "componentid": None,
                    "componentkind": None,
                    "componentname": None,
                    "targetnamespace": root.attrib.get("targetNamespace", ""),
                    "dependson": [],
                    "dependsonlabeled": [],
                    "referencedby": [],
                    "schemalinks": schema_links,
                },
            }
        )

        for i, item in enumerate(registry):
            previous_chunk = chunk_ids[i - 1] if i > 0 else None
            next_chunk = chunk_ids[i + 1] if i < len(chunk_ids) - 1 else None

            referenced_by = reverse_refs.get(item["componentname"], []) + reverse_refs.get(
                item["componentid"], []
            )
            seen = set()
            referenced_by = [x for x in referenced_by if not (x in seen or seen.add(x))]

            chunk_text = build_xml_chunk_text(
                item=item,
                source_file=filepath.name,
                previous_chunk=previous_chunk,
                next_chunk=next_chunk,
                referenced_by=referenced_by,
                schema_links=schema_links,
            )

            safe_name = re.sub(r"[^A-Za-z0-9._-]", "", item["componentname"])
            safe_kind = re.sub(r"[^A-Za-z0-9._-]", "", item["componentkind"])

            results.append(
                {
                    "filename": f"{filepath.stem}_{safe_kind}_{safe_name}.xmlchunk",
                    "text": chunk_text,
                    "payloadextra": {
                        "doctype": "xml:componentchunk",
                        "componentid": item["componentid"],
                        "componentkind": item["componentkind"],
                        "componentname": item["componentname"],
                        "targetnamespace": item["targetnamespace"],
                        "dependson": item["dependson"],
                        "dependsonlabeled": item["dependsonlabeled"],
                        "referencedby": referenced_by,
                        "schemalinks": schema_links,
                    },
                }
            )

        return results

    else:
        return [
            {
                "filename": filepath.name,
                "text": optimized_full,
                "payloadextra": {
                    "doctype": "xml:instance:full",
                    "componentid": None,
                    "componentkind": None,
                    "componentname": None,
                    "targetnamespace": root.attrib.get("targetNamespace", ""),
                    "dependson": [],
                    "dependsonlabeled": [],
                    "referencedby": [],
                    "schemalinks": {},
                },
            }
        ]


def optimize_document_content(filepath: Path, text: str) -> tuple[str, bool]:
    ext = filepath.suffix.lower()

    try:
        if ext == ".json":
            data = json.loads(text)
            optimized = optimize_json_preserving_standards(data)
            return optimized, optimized != text

        if ext == ".xml":
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

        if ext in (".md", ".markdown", ".rst", ".adoc", ".txt"):
            optimized = optimize_markdown_content(text)
            return optimized, optimized != text

        optimized = optimize_generic_text(text)
        if len(optimized) < len(text) * 0.95:
            return optimized, True

        return text, False

    except Exception as e:
        print(f"[!] Optimization failed for {filepath.name}: {e}")
        return text, False


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

    return True


def get_existing_ids():
    try:
        result = client.scroll(
            collection_name=COLLECTION,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {point.id for point in result[0]}
        print(f"[~] Found {len(existing_ids)} existing documents")
        return existing_ids
    except Exception as e:
        print(f"[!] Failed to retrieve existing IDs: {e}")
        return set()


def generate_stable_id(text: str) -> int:
    hash_bytes = sha256(text.encode("utf-8")).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


def generate_component_stable_id(filename: str, component_id: str | None, text: str) -> int:
    base = f"{filename}:{component_id}" if component_id else f"{filename}:{text[:1000]}"
    hash_bytes = sha256(base.encode("utf-8")).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


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


def buildpoint(
    docid: int,
    text: str,
    filename: str,
    encodingused: str,
    vectors: dict[str, Any],
    capabilities: dict[str, Any],
    payloadextra: dict[str, Any] | None = None,
) -> PointStruct:
    # ✅ Extrait le vocabulaire du filename
    vocabulary = _extract_vocabulary_from_filename(filename)
    
    payload = {
        "text": text,
        "filename": filename,
        "encoding": encodingused,
        "vocabulary": vocabulary,  # ✅ Stocke explicitement le vocabulaire
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
    texts = [item["text"] for item in batch_docs]
    encoded_vectors = encode_batch(texts, capabilities)

    for item, vectors in zip(batch_docs, encoded_vectors):
        point = buildpoint(
            docid=item["docid"],
            text=item["text"],
            filename=item["filename"],
            encodingused=item["encoding"],
            vectors=vectors,
            capabilities=capabilities,
            payloadextra=item.get("payloadextra"),
        )
        points.append(point)
        print(f"[~] Loaded {item['filename']} ({item['encoding']})")

    client.upsert(collection_name=COLLECTION, points=points)
    uploaded = len(points)
    print(f"[✓] Uploaded batch of {uploaded}")

    points.clear()
    batch_docs.clear()

    return uploaded


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
    total_xml_chunks = 0

    batch_docs = []
    points = []

    if not docs_path.exists():
        print(f"[!] Directory not found: {docs_path}")
        return

    print(f"[~] Scanning documents in {docs_path}")

    def pushdoc(
        filename: str,
        content: str,
        encodingused: str,
        payloadextra: dict[str, Any] | None = None,
    ):
        nonlocal total_skipped

        component_id = payloadextra.get("componentid") if payloadextra else None
        docid = generate_component_stable_id(filename, component_id, content)

        if not is_fresh and docid in existing_ids:
            print(f"[~] Skipped duplicate: {filename}")
            total_skipped += 1
            return

        batch_docs.append(
            {
                "filename": filename,
                "text": content,
                "encoding": encodingused,
                "docid": docid,
                "payloadextra": payloadextra or {},
            }
        )

    for filepath in docs_path.iterdir():
        if not filepath.is_file():
            continue

        try:
            text, encodingused = read_text_document(filepath)

            if filepath.suffix.lower() == ".xml" and len(text) > XML_SPLIT_THRESHOLD:
                chunks = split_large_xml_semantically(filepath, text)
                print(f"[~] Split XML {filepath.name} into {len(chunks)} chunks")
                total_xml_chunks += max(0, len(chunks) - 1)

                for chunk in chunks:
                    optimized_chunk, was_optimized = optimize_document_content(
                        Path(chunk["filename"]), chunk["text"]
                    )
                    if was_optimized:
                        total_optimized += 1

                    pushdoc(
                        filename=chunk["filename"],
                        content=optimized_chunk,
                        encodingused=encodingused,
                        payloadextra=chunk["payloadextra"],
                    )

                if len(batch_docs) >= BATCH_SIZE:
                    try:
                        total_indexed += flush_batch(batch_docs, points, capabilities)
                    except Exception as e:
                        print(f"[!] Batch upload failed: {e}")
                        batch_docs.clear()
                        points.clear()

                continue

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

            pushdoc(
                filename=filepath.name,
                content=optimized_text,
                encodingused=encodingused,
                payloadextra={
                    "doctype": "regulardocument",
                    "componentid": None,
                    "componentkind": None,
                    "componentname": None,
                    "targetnamespace": None,
                    "dependson": [],
                    "dependsonlabeled": [],
                    "referencedby": [],
                    "schemalinks": {},
                },
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
    print(f"[✓] Documents indexed: {total_indexed}")
    print(f"[✓] Documents optimized: {total_optimized}")
    print(f"[✓] XML/XMI semantic chunks created: {total_xml_chunks}")
    if total_skipped > 0:
        print(f"[~] Documents skipped (duplicates): {total_skipped}")
    if total_nontext > 0:
        print(f"[!] Files skipped (non-text): {total_nontext}")
    print("=" * 50)


if __name__ == "__main__":
    index_documents()
