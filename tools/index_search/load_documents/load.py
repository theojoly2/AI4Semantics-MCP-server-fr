from pathlib import Path
from hashlib import sha256
import re
import codecs
import json
import csv
from io import StringIO
from typing import Any
import xml.etree.ElementTree as ET

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseVector,
)

import config as cf

try:
    import xmlschema
except ImportError:
    xmlschema = None


client = cf.client
model = cf.model
COLLECTION = cf.COLLECTION
BATCH_SIZE = cf.BATCH_SIZE

XML_DECL_RE = re.compile(
    br'<\?xml[^>]*encoding=["\']([A-Za-z0-9._\-]+)["\']',
    re.IGNORECASE
)

TEXT_BOMS = (
    codecs.BOM_UTF8,
    codecs.BOM_UTF16_LE,
    codecs.BOM_UTF16_BE,
    codecs.BOM_UTF32_LE,
    codecs.BOM_UTF32_BE,
)

TEXT_WHITELIST = set(b"\n\r\t\f\b")
PRINTABLE_ASCII = set(range(32, 127))

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

XML_SPLIT_THRESHOLD = 180_000
XML_MAX_CHUNK_CHARS = 120_000

XML_GLOBAL_COMPONENT_TAGS = {
    "complexType",
    "simpleType",
    "element",
    "group",
    "attributeGroup",
    "attribute",
}

XSD_REL_ATTRS = {"base", "ref", "type", "substitutionGroup"}


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
                f"✓ Model capabilities detected: dense={capabilities['has_dense']}, "
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
            f"✓ Model capabilities detected: dense=True, sparse=False, "
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
    candidates: list[str] = []

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
        f"Unable to decode file '{filepath.name}'. Tried encodings: {', '.join(encodings_to_try)}"
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


def optimize_rdf_like_text(content: str) -> str:
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

        if stripped.startswith("#"):
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
        lines.append(re.sub(r"[ \t]+", " ", stripped))
        previous_blank = False

    return "\n".join(lines).strip()


def local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def strip_prefix(value: str) -> str:
    return value.split(":")[-1].strip() if value else value


def collect_documentation_text(elem: ET.Element) -> list[str]:
    docs = []
    for sub in elem.iter():
        if local_name(sub.tag) == "documentation":
            text = " ".join("".join(sub.itertext()).split())
            if text:
                docs.append(text)
    return docs[:10]


def collect_appinfo_text(elem: ET.Element) -> list[str]:
    appinfos = []
    for sub in elem.iter():
        if local_name(sub.tag) == "appinfo":
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
        "group_ref": [],
        "attributeGroup_ref": [],
    }

    for sub in elem.iter():
        tag = local_name(sub.tag)

        for attr_key, attr_value in sub.attrib.items():
            attr_name = local_name(attr_key)
            if attr_name in XSD_REL_ATTRS and attr_value:
                deps[attr_name].append(strip_prefix(attr_value))

        if tag == "group" and "ref" in sub.attrib:
            deps["group_ref"].append(strip_prefix(sub.attrib["ref"]))

        if tag == "attributeGroup" and "ref" in sub.attrib:
            deps["attributeGroup_ref"].append(strip_prefix(sub.attrib["ref"]))

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


def summarize_dependencies(dep_map: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    flat = []
    labeled = []

    for rel_type in ["base", "type", "ref", "substitutionGroup", "group_ref", "attributeGroup_ref"]:
        for item in dep_map.get(rel_type, []):
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
        tag = local_name(child.tag)
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


def resolve_local_schema_links(filepath: Path, schema_links: dict[str, list[str]]) -> dict[str, list[str]]:
    resolved = {
        "includes": [],
        "imports": [],
        "redefines": [],
        "overrides": [],
    }

    base_dir = filepath.parent
    for key, items in schema_links.items():
        for item in items:
            candidate = (base_dir / item).resolve()
            if candidate.exists():
                resolved[key].append(str(candidate))
            else:
                resolved[key].append(item)

    return resolved


def build_component_registry_etree(root: ET.Element, filepath: Path) -> list[dict[str, Any]]:
    registry = []

    for child in list(root):
        kind = local_name(child.tag)
        if kind not in XML_GLOBAL_COMPONENT_TAGS:
            continue

        name = child.attrib.get("name") or child.attrib.get("ref") or child.attrib.get("id")
        if not name:
            name = f"anonymous_{len(registry) + 1}"

        xml_text = ET.tostring(child, encoding="unicode", method="xml")
        xml_text = optimize_xml_preserving_standards(xml_text)

        dep_map = extract_xsd_dependencies_from_element(child)
        depends_on, depends_on_labeled = summarize_dependencies(dep_map)

        registry.append({
            "component_id": f"{kind}:{strip_prefix(name)}",
            "component_name": strip_prefix(name),
            "component_kind": kind,
            "target_namespace": root.attrib.get("targetNamespace", ""),
            "schema_file": filepath.name,
            "xml": xml_text,
            "documentation": collect_documentation_text(child),
            "appinfo": collect_appinfo_text(child),
            "dep_map": dep_map,
            "depends_on": depends_on,
            "depends_on_labeled": depends_on_labeled,
        })

    return registry


def build_component_registry_xmlschema(filepath: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if xmlschema is None:
        return [], {}

    try:
        schema = xmlschema.XMLSchema(str(filepath))
    except Exception:
        return [], {}

    registry = []

    try:
        root_elem = ET.fromstring(filepath.read_text(encoding="utf-8", errors="ignore"))
        schema_links = resolve_local_schema_links(filepath, extract_schema_link_directives(root_elem))
    except Exception:
        schema_links = {"includes": [], "imports": [], "redefines": [], "overrides": []}

    for component in schema.iter_components():
        cls_name = component.__class__.__name__
        name = getattr(component, "name", None)

        if not name:
            continue

        target_namespace = getattr(component, "target_namespace", "") or getattr(schema, "target_namespace", "")
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
            "group_ref": [],
            "attributeGroup_ref": [],
        }

        if elem is not None:
            documentation = collect_documentation_text(elem)
            appinfo = collect_appinfo_text(elem)
            dep_map = extract_xsd_dependencies_from_element(elem)

        depends_on, depends_on_labeled = summarize_dependencies(dep_map)

        registry.append({
            "component_id": component_id,
            "component_name": strip_prefix(name),
            "component_kind": cls_name,
            "target_namespace": target_namespace,
            "schema_file": filepath.name,
            "xml": xml_text,
            "documentation": documentation,
            "appinfo": appinfo,
            "dep_map": dep_map,
            "depends_on": depends_on,
            "depends_on_labeled": depends_on_labeled,
        })

    return registry, schema_links


def build_reverse_references(registry: list[dict[str, Any]]) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}

    for item in registry:
        source_id = item["component_id"]
        source_name = item["component_name"]
        reverse.setdefault(source_name, [])

        for dep in item["depends_on"]:
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
    counts: dict[str, int] = {}
    all_ids = []

    for item in registry:
        counts[item["component_kind"]] = counts.get(item["component_kind"], 0) + 1
        all_ids.append(item["component_id"])

    counts_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
    ids_text = ", ".join(all_ids[:40]) + (", ..." if len(all_ids) > 40 else "")

    most_referenced = sorted(
        ((name, len(refs)) for name, refs in reverse_refs.items() if refs),
        key=lambda x: x[1],
        reverse=True,
    )
    most_referenced_text = ", ".join(f"{name}({count})" for name, count in most_referenced[:20]) or "none"

    return (
        "[XML_DOCUMENT_SUMMARY]\n"
        f"SOURCE_DOCUMENT: {filepath.name}\n"
        f"ROOT_ELEMENT: {root_tag}\n"
        f"TOTAL_COMPONENTS: {len(registry)}\n"
        f"COMPONENT_BREAKDOWN: {counts_text}\n"
        f"KNOWN_COMPONENTS: {ids_text}\n"
        f"INCLUDES: {', '.join(schema_links.get('includes', [])) or 'none'}\n"
        f"IMPORTS: {', '.join(schema_links.get('imports', [])) or 'none'}\n"
        f"REDEFINES: {', '.join(schema_links.get('redefines', [])) or 'none'}\n"
        f"OVERRIDES: {', '.join(schema_links.get('overrides', [])) or 'none'}\n"
        f"MOST_REFERENCED_COMPONENTS: {most_referenced_text}\n"
        "NOTE: Parent summary of a larger XML/XSD standard, split into semantic chunks while preserving raw XML and schema relations.\n"
        "[/XML_DOCUMENT_SUMMARY]"
    )


def build_xml_chunk_text(
    item: dict[str, Any],
    source_file: str,
    previous_chunk: str | None,
    next_chunk: str | None,
    referenced_by: list[str],
    schema_links: dict[str, list[str]],
) -> str:
    docs_text = " | ".join(item["documentation"][:5]) if item["documentation"] else "none"
    appinfo_text = " | ".join(item["appinfo"][:3]) if item["appinfo"] else "none"
    deps_text = ", ".join(item["depends_on_labeled"][:25]) if item["depends_on_labeled"] else "none"
    refs_text = ", ".join(referenced_by[:25]) if referenced_by else "none"

    header = (
        "[XML_CHUNK_CONTEXT]\n"
        f"SOURCE_DOCUMENT: {source_file}\n"
        f"COMPONENT_ID: {item['component_id']}\n"
        f"COMPONENT_KIND: {item['component_kind']}\n"
        f"COMPONENT_NAME: {item['component_name']}\n"
        f"TARGET_NAMESPACE: {item['target_namespace'] or 'none'}\n"
        f"SCHEMA_FILE: {item['schema_file']}\n"
        f"PREVIOUS_CHUNK: {previous_chunk or 'none'}\n"
        f"NEXT_CHUNK: {next_chunk or 'none'}\n"
        f"DEPENDS_ON_LABELED: {deps_text}\n"
        f"REFERENCED_BY: {refs_text}\n"
        f"INCLUDES: {', '.join(schema_links.get('includes', [])) or 'none'}\n"
        f"IMPORTS: {', '.join(schema_links.get('imports', [])) or 'none'}\n"
        f"REDEFINES: {', '.join(schema_links.get('redefines', [])) or 'none'}\n"
        f"OVERRIDES: {', '.join(schema_links.get('overrides', [])) or 'none'}\n"
        f"DOCUMENTATION: {docs_text}\n"
        f"APPINFO: {appinfo_text}\n"
        "[/XML_CHUNK_CONTEXT]\n\n"
    )

    body = item["xml"] or "[NO_RAW_XML_COMPONENT_AVAILABLE]"

    footer = (
        "\n\n[XML_CHUNK_RELATIONS]\n"
        f"SOURCE_DOCUMENT: {source_file}\n"
        f"COMPONENT_ID: {item['component_id']}\n"
        f"DEPENDS_ON_LABELED: {deps_text}\n"
        f"REFERENCED_BY: {refs_text}\n"
        "NOTE: This chunk preserves raw XML plus extracted schema relations and annotations.\n"
        "[/XML_CHUNK_RELATIONS]"
    )

    final_text = header + body + footer

    if len(final_text) > XML_MAX_CHUNK_CHARS:
        abstract = (
            "[XML_COMPONENT_ABSTRACT]\n"
            f"COMPONENT_ID: {item['component_id']}\n"
            f"COMPONENT_KIND: {item['component_kind']}\n"
            f"COMPONENT_NAME: {item['component_name']}\n"
            f"TARGET_NAMESPACE: {item['target_namespace'] or 'none'}\n"
            f"DOCUMENTATION: {docs_text}\n"
            f"APPINFO: {appinfo_text}\n"
            f"DEPENDENCIES: {deps_text}\n"
            "[/XML_COMPONENT_ABSTRACT]\n\n"
        )
        final_text = header + abstract + body[:90000] + footer

    return final_text


def detect_xml_type(root: ET.Element) -> str:
    """Détermine si c'est un XSD, du XMI/UML, ou un document XML d'instance"""
    root_tag = local_name(root.tag)
    
    # XMI/UML
    if root_tag in ("XMI", "Model", "uml:Model"):
        return "xmi"
    
    # Détecte XMI par attributs
    for attr_name in root.attrib.keys():
        if 'xmi' in attr_name.lower():
            return "xmi"
    
    # Détecte XMI par enfants
    has_uml_elements = False
    for child in list(root)[:10]:  # Check premiers enfants
        tag_lower = child.tag.lower()
        if 'uml:' in child.tag or 'packagedelement' in tag_lower or 'xmi' in tag_lower:
            has_uml_elements = True
            break
    
    if has_uml_elements:
        return "xmi"
    
    # XSD schema
    if root_tag == "schema":
        return "xsd"
    
    # Cherche des composants XSD dans les enfants
    has_xsd_components = any(
        local_name(child.tag) in XML_GLOBAL_COMPONENT_TAGS
        for child in root
    )
    
    if has_xsd_components:
        return "xsd"
    
    return "xml_instance"


def extract_uml_element_info(elem: ET.Element) -> dict[str, Any]:
    """Extrait les infos importantes d'un élément UML"""
    tag = local_name(elem.tag)
    
    # Cherche les attributs XMI standards
    xmi_type = (
        elem.attrib.get("{http://schema.omg.org/spec/XMI/2.1}type") or
        elem.attrib.get("{http://www.omg.org/spec/XMI/20131001}type") or
        elem.attrib.get("{http://www.omg.org/XMI}type") or
        elem.attrib.get("type") or
        tag
    )
    
    xmi_id = (
        elem.attrib.get("{http://schema.omg.org/spec/XMI/2.1}id") or
        elem.attrib.get("{http://www.omg.org/spec/XMI/20131001}id") or
        elem.attrib.get("{http://www.omg.org/XMI}id") or
        elem.attrib.get("id")
    )
    
    name = elem.attrib.get("name") or xmi_id or f"anonymous_{tag}"
    
    return {
        "tag": tag,
        "xmi_type": strip_prefix(xmi_type),
        "xmi_id": xmi_id,
        "name": name,
    }


def collect_uml_dependencies(elem: ET.Element) -> dict[str, list[str]]:
    """Collecte les dépendances UML (types, associations, généralisations, etc.)"""
    deps = {
        "type": [],
        "association": [],
        "generalization": [],
        "realization": [],
        "dependency": [],
        "aggregation": [],
        "idref": [],
    }
    
    # Parcourt tous les sous-éléments
    for sub in elem.iter():
        # Attribut 'type' référence un autre élément
        type_ref = sub.attrib.get("type")
        if type_ref:
            deps["type"].append(strip_prefix(type_ref))
        
        # Association
        assoc = sub.attrib.get("association")
        if assoc:
            deps["association"].append(strip_prefix(assoc))
        
        # Generalization/specialization
        general = sub.attrib.get("general")
        if general:
            deps["generalization"].append(strip_prefix(general))
        
        # Realization
        supplier = sub.attrib.get("supplier")
        if supplier:
            deps["realization"].append(strip_prefix(supplier))
        
        # Dependency
        client = sub.attrib.get("client")
        if client:
            deps["dependency"].append(strip_prefix(client))
        
        # Idref (références XMI génériques)
        idref = sub.attrib.get("idref") or sub.attrib.get("href")
        if idref:
            deps["idref"].append(strip_prefix(idref))
    
    # Déduplique
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


def split_xmi_by_packaged_elements(filepath: Path, root: ET.Element) -> list[dict[str, Any]]:
    """
    Découpe un fichier XMI par packagedElement (classes, packages, associations, etc.)
    """
    results: list[dict[str, Any]] = []
    
    # Collecte tous les packagedElement de niveau supérieur
    packaged_elements = []
    
    def collect_packaged_elements(parent: ET.Element, depth: int = 0):
        """Collecte récursivement les packagedElement importants"""
        for child in parent:
            tag = local_name(child.tag)
            
            # packagedElement est le conteneur standard UML
            if tag == "packagedElement" or 'packagedElement' in child.tag:
                info = extract_uml_element_info(child)
                xml_text = ET.tostring(child, encoding="unicode", method="xml")
                xml_text = optimize_xml_preserving_standards(xml_text)
                
                # Skip si trop petit
                if len(xml_text) < 3000:
                    continue
                
                deps = collect_uml_dependencies(child)
                deps_flat = []
                deps_labeled = []
                for rel_type, items in deps.items():
                    for item in items:
                        deps_flat.append(item)
                        deps_labeled.append(f"{rel_type}:{item}")
                
                # Déduplique
                deps_flat = list(dict.fromkeys(deps_flat))
                deps_labeled = list(dict.fromkeys(deps_labeled))
                
                packaged_elements.append({
                    "tag": info["tag"],
                    "xmi_type": info["xmi_type"],
                    "xmi_id": info["xmi_id"],
                    "name": info["name"],
                    "xml": xml_text,
                    "depth": depth,
                    "depends_on": deps_flat,
                    "depends_on_labeled": deps_labeled,
                    "index": len(packaged_elements),
                })
            
            # Descend récursivement si package ou model
            if tag in ("packagedElement", "Model", "Package") and depth < 3:
                collect_packaged_elements(child, depth + 1)
    
    collect_packaged_elements(root)
    
    # Si pas assez d'éléments, ne découpe pas
    if len(packaged_elements) < 2:
        return []
    
    # Crée un index inverse (qui référence quoi)
    reverse_refs: dict[str, list[str]] = {}
    for elem in packaged_elements:
        elem_id = elem["xmi_id"] or elem["name"]
        for dep in elem["depends_on"]:
            reverse_refs.setdefault(dep, []).append(elem_id)
    
    # Parent summary
    root_info = extract_uml_element_info(root)
    element_types = {}
    for elem in packaged_elements:
        element_types[elem["xmi_type"]] = element_types.get(elem["xmi_type"], 0) + 1
    
    types_text = ", ".join(f"{k}={v}" for k, v in sorted(element_types.items())) or "none"
    ids_text = ", ".join(e["xmi_id"] or e["name"] for e in packaged_elements[:40])
    if len(packaged_elements) > 40:
        ids_text += ", ..."
    
    most_referenced = sorted(
        ((name, len(refs)) for name, refs in reverse_refs.items() if refs),
        key=lambda x: x[1],
        reverse=True,
    )
    most_referenced_text = ", ".join(f"{name}({count})" for name, count in most_referenced[:20]) or "none"
    
    parent_summary = (
        "[XMI_DOCUMENT_SUMMARY]\n"
        f"SOURCE_DOCUMENT: {filepath.name}\n"
        f"ROOT_ELEMENT: {root_info['tag']}\n"
        f"ROOT_XMI_TYPE: {root_info['xmi_type']}\n"
        f"ROOT_NAME: {root_info['name']}\n"
        f"TOTAL_PACKAGED_ELEMENTS: {len(packaged_elements)}\n"
        f"ELEMENT_TYPE_BREAKDOWN: {types_text}\n"
        f"KNOWN_ELEMENTS: {ids_text}\n"
        f"MOST_REFERENCED_ELEMENTS: {most_referenced_text}\n"
        "NOTE: This is a parent summary of a UML/XMI model split by packagedElement while preserving raw XML and UML relations.\n"
        "[/XMI_DOCUMENT_SUMMARY]"
    )
    
    results.append({
        "filename": f"{filepath.stem}__PARENT_SUMMARY.xmichunk",
        "text": parent_summary,
        "payload_extra": {
            "doc_type": "xmi_parent_summary",
            "component_id": None,
            "component_kind": root_info["xmi_type"],
            "component_name": root_info["name"],
            "target_namespace": root.attrib.get("targetNamespace"),
            "depends_on": [],
            "depends_on_labeled": [],
            "referenced_by": [],
            "schema_links": {},
        }
    })
    
    # Chunks pour chaque élément
    element_ids = [e["xmi_id"] or e["name"] for e in packaged_elements]
    
    for i, elem in enumerate(packaged_elements):
        previous_elem = element_ids[i - 1] if i > 0 else None
        next_elem = element_ids[i + 1] if i < len(element_ids) - 1 else None
        
        elem_id = elem["xmi_id"] or elem["name"]
        referenced_by = reverse_refs.get(elem_id, []) + reverse_refs.get(elem["name"], [])
        referenced_by = list(dict.fromkeys(referenced_by))  # Déduplique
        
        deps_text = ", ".join(elem["depends_on_labeled"][:25]) if elem["depends_on_labeled"] else "none"
        refs_text = ", ".join(referenced_by[:25]) if referenced_by else "none"
        
        header = (
            "[XMI_ELEMENT_CONTEXT]\n"
            f"SOURCE_DOCUMENT: {filepath.name}\n"
            f"XMI_ID: {elem['xmi_id'] or 'none'}\n"
            f"XMI_TYPE: {elem['xmi_type']}\n"
            f"ELEMENT_NAME: {elem['name']}\n"
            f"ELEMENT_INDEX: {i + 1}/{len(packaged_elements)}\n"
            f"PREVIOUS_ELEMENT: {previous_elem or 'none'}\n"
            f"NEXT_ELEMENT: {next_elem or 'none'}\n"
            f"DEPENDS_ON_LABELED: {deps_text}\n"
            f"REFERENCED_BY: {refs_text}\n"
            "[/XMI_ELEMENT_CONTEXT]\n\n"
        )
        
        footer = (
            "\n\n[XMI_ELEMENT_RELATIONS]\n"
            f"SOURCE_DOCUMENT: {filepath.name}\n"
            f"XMI_ID: {elem['xmi_id'] or 'none'}\n"
            f"XMI_TYPE: {elem['xmi_type']}\n"
            f"DEPENDS_ON_LABELED: {deps_text}\n"
            f"REFERENCED_BY: {refs_text}\n"
            "NOTE: This chunk preserves raw XMI/UML element plus extracted model relations.\n"
            "[/XMI_ELEMENT_RELATIONS]"
        )
        
        final_text = header + elem["xml"] + footer
        
        if len(final_text) > XML_MAX_CHUNK_CHARS:
            abstract = (
                "[XMI_ELEMENT_ABSTRACT]\n"
                f"XMI_ID: {elem['xmi_id'] or 'none'}\n"
                f"XMI_TYPE: {elem['xmi_type']}\n"
                f"ELEMENT_NAME: {elem['name']}\n"
                f"DEPENDENCIES: {deps_text}\n"
                "[/XMI_ELEMENT_ABSTRACT]\n\n"
            )
            final_text = header + abstract + elem["xml"][:90000] + footer
        
        safe_type = re.sub(r"[^A-Za-z0-9_.-]+", "_", elem["xmi_type"])
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", elem["name"])
        
        results.append({
            "filename": f"{filepath.stem}__{safe_type}__{safe_name}.xmichunk",
            "text": final_text,
            "payload_extra": {
                "doc_type": "xmi_packaged_element",
                "component_id": elem["xmi_id"],
                "component_kind": elem["xmi_type"],
                "component_name": elem["name"],
                "target_namespace": None,
                "depends_on": elem["depends_on"],
                "depends_on_labeled": elem["depends_on_labeled"],
                "referenced_by": referenced_by,
                "schema_links": {},
            }
        })
    
    return results


def split_large_xml_semantically(filepath: Path, xml_content: str) -> list[dict[str, Any]]:
    """Version améliorée qui détecte XSD vs XMI vs XML instance"""
    optimized_full = optimize_xml_preserving_standards(xml_content)

    if len(xml_content) < XML_SPLIT_THRESHOLD:
        return [{
            "filename": filepath.name,
            "text": optimized_full,
            "payload_extra": {
                "doc_type": "xml_full",
                "component_id": None,
                "component_kind": None,
                "component_name": None,
                "target_namespace": None,
                "depends_on": [],
                "depends_on_labeled": [],
                "referenced_by": [],
                "schema_links": {},
            }
        }]

    try:
        root = ET.fromstring(xml_content)
    except Exception:
        return [{
            "filename": filepath.name,
            "text": optimized_full,
            "payload_extra": {
                "doc_type": "xml_full",
                "component_id": None,
                "component_kind": None,
                "component_name": None,
                "target_namespace": None,
                "depends_on": [],
                "depends_on_labeled": [],
                "referenced_by": [],
                "schema_links": {},
            }
        }]

    # Détecte le type de XML
    xml_type = detect_xml_type(root)
    
    if xml_type == "xmi":
        # C'est du XMI/UML → découpe par packagedElement
        xmi_chunks = split_xmi_by_packaged_elements(filepath, root)
        
        if not xmi_chunks:
            # Pas assez d'éléments → retourne le fichier entier
            return [{
                "filename": filepath.name,
                "text": optimized_full,
                "payload_extra": {
                    "doc_type": "xmi_full",
                    "component_id": None,
                    "component_kind": "XMI",
                    "component_name": filepath.stem,
                    "target_namespace": None,
                    "depends_on": [],
                    "depends_on_labeled": [],
                    "referenced_by": [],
                    "schema_links": {},
                }
            }]
        
        return xmi_chunks
    
    elif xml_type == "xsd":
        # C'est un schéma XSD → découpe par composants
        registry, schema_links = build_component_registry_xmlschema(filepath)

        if not registry:
            registry = build_component_registry_etree(root, filepath)
            schema_links = resolve_local_schema_links(filepath, extract_schema_link_directives(root))

        if not registry:
            # Pas de composants trouvés → retourne le fichier entier
            return [{
                "filename": filepath.name,
                "text": optimized_full,
                "payload_extra": {
                    "doc_type": "xml_full",
                    "component_id": None,
                    "component_kind": None,
                    "component_name": None,
                    "target_namespace": root.attrib.get("targetNamespace"),
                    "depends_on": [],
                    "depends_on_labeled": [],
                    "referenced_by": [],
                    "schema_links": schema_links,
                }
            }]

        # Découpe par composants XSD
        reverse_refs = build_reverse_references(registry)
        chunk_ids = [item["component_id"] for item in registry]

        results: list[dict[str, Any]] = []

        parent_summary = build_parent_xml_summary(
            filepath=filepath,
            root_tag=local_name(root.tag),
            registry=registry,
            reverse_refs=reverse_refs,
            schema_links=schema_links,
        )

        results.append({
            "filename": f"{filepath.stem}__PARENT_SUMMARY.xmlchunk",
            "text": parent_summary,
            "payload_extra": {
                "doc_type": "xml_parent_summary",
                "component_id": None,
                "component_kind": None,
                "component_name": None,
                "target_namespace": root.attrib.get("targetNamespace"),
                "depends_on": [],
                "depends_on_labeled": [],
                "referenced_by": [],
                "schema_links": schema_links,
            }
        })

        for i, item in enumerate(registry):
            previous_chunk = chunk_ids[i - 1] if i > 0 else None
            next_chunk = chunk_ids[i + 1] if i < len(chunk_ids) - 1 else None
            referenced_by = reverse_refs.get(item["component_name"], []) + reverse_refs.get(item["component_id"], [])
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

            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", item["component_name"])
            safe_kind = re.sub(r"[^A-Za-z0-9_.-]+", "_", item["component_kind"])

            results.append({
                "filename": f"{filepath.stem}__{safe_kind}__{safe_name}.xmlchunk",
                "text": chunk_text,
                "payload_extra": {
                    "doc_type": "xml_component_chunk",
                    "component_id": item["component_id"],
                    "component_kind": item["component_kind"],
                    "component_name": item["component_name"],
                    "target_namespace": item["target_namespace"],
                    "depends_on": item["depends_on"],
                    "depends_on_labeled": item["depends_on_labeled"],
                    "referenced_by": referenced_by,
                    "schema_links": schema_links,
                }
            })

        return results
    
    else:
        # C'est un document XML d'instance → retourne entier pour l'instant
        return [{
            "filename": filepath.name,
            "text": optimized_full,
            "payload_extra": {
                "doc_type": "xml_instance_full",
                "component_id": None,
                "component_kind": None,
                "component_name": None,
                "target_namespace": root.attrib.get("targetNamespace"),
                "depends_on": [],
                "depends_on_labeled": [],
                "referenced_by": [],
                "schema_links": {},
            }
        }]


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

        if ext in {".ttl", ".rdf", ".owl", ".n3", ".nt", ".trig"}:
            optimized = optimize_rdf_like_text(text)
            return optimized, optimized != text

        if ext in {".yaml", ".yml"}:
            optimized = optimize_yaml_content(text)
            return optimized, optimized != text

        if ext == ".csv":
            optimized = optimize_csv_content(text)
            return optimized, optimized != text

        if ext in {".html", ".htm", ".xhtml"}:
            optimized = optimize_html_content(text)
            return optimized, optimized != text

        if ext in {".md", ".markdown", ".rst", ".adoc", ".txt"}:
            optimized = optimize_markdown_content(text)
            return optimized, optimized != text

        optimized = optimize_generic_text(text)
        if len(optimized) < len(text) * 0.95:
            return optimized, True
        return text, False

    except Exception as e:
        print(f"  ⚠ Optimization failed for {filepath.name}: {e}")
        return text, False


def setup_collection(capabilities: dict[str, Any]) -> bool:
    collections = client.get_collections().collections
    collection_exists = any(c.name == COLLECTION for c in collections)

    if collection_exists:
        print(f"⚠ Collection '{COLLECTION}' already exists.")
        choice = input("  Delete and recreate? (y/n): ").strip().lower()

        if choice == "y":
            client.delete_collection(collection_name=COLLECTION)
            print(f"✓ Deleted collection '{COLLECTION}'")
        else:
            print(f"✓ Using existing collection '{COLLECTION}'")
            return False

    if capabilities["has_dense"] and capabilities["has_sparse"]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=capabilities["dense_dim"],
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams()
            },
        )
        print(
            f"✓ Created hybrid collection '{COLLECTION}' "
            f"(dense={capabilities['dense_dim']}, sparse=yes)"
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
            f"✓ Created dense-only collection '{COLLECTION}' "
            f"(dense={capabilities['dense_dim']})"
        )

    elif capabilities["has_sparse"]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={},
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams()
            },
        )
        print(f"✓ Created sparse-only collection '{COLLECTION}'")

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
        print(f"  → Found {len(existing_ids)} existing documents")
        return existing_ids
    except Exception as e:
        print(f"  ✗ Failed to retrieve existing IDs: {e}")
        return set()


def generate_stable_id(text: str) -> int:
    hash_bytes = sha256(text.encode("utf-8")).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


def generate_component_stable_id(filename: str, component_id: str | None, text: str) -> int:
    base = f"{filename}::{component_id}" if component_id else f"{filename}::{text[:1000]}"
    hash_bytes = sha256(base.encode("utf-8")).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


def _lexical_to_sparse_vector(lexical_weights: dict[Any, Any]) -> SparseVector:
    indices = [int(k) for k in lexical_weights.keys()]
    values = [float(v) for v in lexical_weights.values()]
    return SparseVector(indices=indices, values=values)


def encode_batch(texts: list[str], capabilities: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

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
            item: dict[str, Any] = {}
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
            outputs.append({"dense": dense.tolist() if hasattr(dense, "tolist") else list(dense)})
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
    doc_id: int,
    text: str,
    filename: str,
    encoding_used: str,
    vectors: dict[str, Any],
    capabilities: dict[str, Any],
    payload_extra: dict[str, Any] | None = None,
) -> PointStruct:
    payload = {
        "text": text,
        "filename": filename,
        "encoding": encoding_used,
    }

    if payload_extra:
        payload.update(payload_extra)

    if capabilities["has_dense"] and capabilities["has_sparse"]:
        return PointStruct(
            id=doc_id,
            vector={
                DENSE_VECTOR_NAME: vectors["dense"],
                SPARSE_VECTOR_NAME: vectors["sparse"],
            },
            payload=payload,
        )

    if capabilities["has_dense"]:
        return PointStruct(
            id=doc_id,
            vector=vectors["dense"],
            payload=payload,
        )

    if capabilities["has_sparse"]:
        return PointStruct(
            id=doc_id,
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
        point = build_point(
            doc_id=item["doc_id"],
            text=item["text"],
            filename=item["filename"],
            encoding_used=item["encoding"],
            vectors=vectors,
            capabilities=capabilities,
            payload_extra=item.get("payload_extra"),
        )
        points.append(point)
        print(f"✓ Loaded: {item['filename']} [{item['encoding']}]")

    client.upsert(collection_name=COLLECTION, points=points)
    uploaded = len(points)
    print(f"  → Uploaded batch of {uploaded}")
    points.clear()
    batch_docs.clear()
    return uploaded


def index_documents():
    print("\n🔍 Detecting model capabilities...")
    capabilities = cf.MODEL_CAPABILITIES

    is_fresh = setup_collection(capabilities)
    existing_ids = set() if is_fresh else get_existing_ids()

    docs_path = Path(__file__).parent / "documents"
    total_indexed = 0
    total_skipped = 0
    total_non_text = 0
    total_optimized = 0
    total_xml_chunks = 0

    batch_docs: list[dict[str, Any]] = []
    points: list[PointStruct] = []

    if not docs_path.exists():
        print(f"✗ Directory not found: {docs_path}")
        return

    print(f"\n📂 Scanning documents in: {docs_path}")

    def push_doc(filename: str, content: str, encoding_used: str, payload_extra: dict[str, Any] | None = None):
        nonlocal total_skipped

        component_id = payload_extra.get("component_id") if payload_extra else None
        doc_id = generate_component_stable_id(filename, component_id, content)

        if not is_fresh and doc_id in existing_ids:
            print(f"⊘ Skipped (duplicate): {filename}")
            total_skipped += 1
            return

        batch_docs.append({
            "filename": filename,
            "text": content,
            "encoding": encoding_used,
            "doc_id": doc_id,
            "payload_extra": payload_extra or {},
        })

    for filepath in docs_path.iterdir():
        if not filepath.is_file():
            continue

        try:
            text, encoding_used = read_text_document(filepath)

            if filepath.suffix.lower() == ".xml" and len(text) >= XML_SPLIT_THRESHOLD:
                chunks = split_large_xml_semantically(filepath, text)
                print(f"  ✂ Split XML {filepath.name} into {len(chunks)} chunks")
                total_xml_chunks += max(0, len(chunks) - 1)

                # Ajoute tous les chunks au batch
                for chunk in chunks:
                    optimized_chunk, was_optimized = optimize_document_content(Path(chunk["filename"]), chunk["text"])
                    if was_optimized:
                        total_optimized += 1

                    push_doc(
                        filename=chunk["filename"],
                        content=optimized_chunk,
                        encoding_used=encoding_used,
                        payload_extra=chunk["payload_extra"],
                    )

                # CORRECTION : Flush immédiatement après avoir ajouté tous les chunks de ce fichier
                # pour garantir qu'ils sont tous uploadés ensemble
                if batch_docs:
                    try:
                        total_indexed += flush_batch(batch_docs, points, capabilities)
                    except Exception as e:
                        print(f"  ✗ Batch upload failed: {e}")
                        batch_docs.clear()
                        points.clear()
                
                continue

            optimized_text, was_optimized = optimize_document_content(filepath, text)

            if was_optimized:
                total_optimized += 1
                original_len = len(text)
                optimized_len = len(optimized_text)
                reduction = (1 - optimized_len / original_len) * 100 if original_len else 0
                print(
                    f"  ⚡ Optimized {filepath.name}: "
                    f"{original_len} → {optimized_len} chars ({reduction:.1f}% reduction)"
                )

            push_doc(
                filename=filepath.name,
                content=optimized_text,
                encoding_used=encoding_used,
                payload_extra={
                    "doc_type": "regular_document",
                    "component_id": None,
                    "component_kind": None,
                    "component_name": None,
                    "target_namespace": None,
                    "depends_on": [],
                    "depends_on_labeled": [],
                    "referenced_by": [],
                    "schema_links": {},
                },
            )

            # Pour les fichiers non-XML, on flush seulement si le batch est plein
            if len(batch_docs) >= BATCH_SIZE:
                try:
                    total_indexed += flush_batch(batch_docs, points, capabilities)
                except Exception as e:
                    print(f"  ✗ Batch upload failed: {e}")
                    batch_docs.clear()
                    points.clear()

        except ValueError as e:
            print(f"⊘ Skipped non-text file {filepath.name}: {e}")
            total_non_text += 1

        except Exception as e:
            print(f"✗ Failed to read {filepath.name}: {e}")

    # Flush final pour les documents restants
    if batch_docs:
        try:
            total_indexed += flush_batch(batch_docs, points, capabilities)
        except Exception as e:
            print(f"  ✗ Final batch upload failed: {e}")

    print(f"\n{'=' * 50}")
    print("✓ Indexing complete")
    print(f"  • Documents indexed: {total_indexed}")
    print(f"  • Documents optimized: {total_optimized}")
    print(f"  • XML/XMI semantic chunks created: {total_xml_chunks}")
    if total_skipped > 0:
        print(f"  • Documents skipped (duplicates): {total_skipped}")
    if total_non_text > 0:
        print(f"  • Files skipped (non-text): {total_non_text}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    index_documents()
