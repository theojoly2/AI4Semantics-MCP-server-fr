from __future__ import annotations

from os import (
    mkdir,
    getcwd,
)
from os.path import (
    exists,
)
from typing import (
    Any,
    Literal,
)
from pathlib import (
    Path,
)
from json import (
    dump,
    load,
)
from resources.semantic_model.utils import MODELS_PATH
from rdflib import Graph, Namespace, RDFS, OWL, RDF, SKOS, URIRef, Literal, XSD
import uuid
from json import load, dump
from os.path import exists
import json

if not exists(MODELS_PATH):
    mkdir(MODELS_PATH)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ttl_to_uml_json.py
Convert an OWL/RDF ontology in Turtle (TTL) to an Enterprise Architect-like UML JSON.

- Classes (owl:Class)  -> elements of type "uml:Class"
- ObjectProperties     -> connectors of type "Association" (domain -> range)
- DatatypeProperties   -> attributes on the domain class
- rdfs:subClassOf      -> connectors of type "Generalization" (child -> parent)
- owl:Ontology         -> root Package (name from rdfs:label @en if present)

IDs are deterministic per URI (uuid5), prefixed EA-like: EAPK_ for packages, EAID_ for classes.
"""
# ---------- Helpers ----------

def ea_id(prefix: str, uri: str) -> str:
    """
    EA-like deterministic ID from a URI: EAID/EAPK + UUID5 (underscored groups, uppercased).
    """
    u = uuid.uuid5(uuid.NAMESPACE_URL, str(uri))
    s = str(u).replace('-', '_').upper()
    return f"{prefix}_{s}"

def local_name(uri: URIRef) -> str:
    """
    Derive a human-friendly name from a URI: last segment after '/' or '#'.
    """
    s = str(uri)
    if '#' in s:
        s = s.split('#')[-1]
    else:
        s = s.rstrip('/').split('/')[-1]
    return s

def get_label(g: Graph, s: URIRef, lang: str = "en") -> str | None:
    vals = list(g.objects(s, RDFS.label))
    if not vals:
        return None
    # prefer requested language
    for v in vals:
        if isinstance(v, Literal) and v.language == lang:
            return str(v)
    # fallback to first literal/string
    return str(vals[0])

def get_comment(g: Graph, s: URIRef, lang: str = "en") -> str | None:
    vals = list(g.objects(s, RDFS.comment))
    if not vals:
        return None
    for v in vals:
        if isinstance(v, Literal) and v.language == lang:
            return str(v)
    return str(vals[0])

def primitive_from_range(r: URIRef) -> str | None:
    """
    Map RDF/XSD types to UML primitive names; None means treat as classifier name.
    Extend as needed.
    """
    s = str(r)
    if s in (str(RDFS.Literal), str(RDF.langString)):
        return "String"
    # Common XSD mappings
    xsd_map = {
        str(XSD.string): "String",
        str(XSD.boolean): "Boolean",
        str(XSD.integer): "Integer",
        str(XSD.int): "Integer",
        str(XSD.long): "Integer",
        str(XSD.float): "Real",
        str(XSD.double): "Real",
        str(XSD.decimal): "Real",
        str(XSD.date): "Date",
        str(XSD.dateTime): "DateTime",
        str(XSD.time): "Time",
        str(XSD.gYear): "String",
        str(XSD.gYearMonth): "String",
        str(XSD.anyURI): "URI",
    }
    return xsd_map.get(s)


# Reverse map: friendly/short type names (as callers supply them) → canonical XSD/RDF URIs.
# Also handles full URIs that are already correct so resolve_type_uri is always safe to call.
_FRIENDLY_TO_XSD_URI: dict[str, str] = {
    # Plain labels used in the UML model
    "string":      str(XSD.string),
    "text":        str(RDF.langString),   # rdf:langString -> rendered as "Text" in UML
    "literal":     str(RDFS.Literal),
    "boolean":     str(XSD.boolean),
    "integer":     str(XSD.integer),
    "int":         str(XSD.int),
    "long":        str(XSD.long),
    "float":       str(XSD.float),
    "real":        str(XSD.double),
    "double":      str(XSD.double),
    "decimal":     str(XSD.decimal),
    "date":        str(XSD.date),
    "datetime":    str(XSD.dateTime),
    "time":        str(XSD.time),
    "gyear":       str(XSD.gYear),
    "gyearmonth":  str(XSD.gYearMonth),
    "uri":         str(XSD.anyURI),
    "anyuri":      str(XSD.anyURI),
    # Custom compound types used by SEMIC / Core Vocabularies
    "genericdate": str(XSD.date),         # m8g:GenericDate -> nearest XSD equivalent
    # Short prefixed forms
    "xsd:string":   str(XSD.string),
    "xsd:boolean":  str(XSD.boolean),
    "xsd:integer":  str(XSD.integer),
    "xsd:int":      str(XSD.int),
    "xsd:long":     str(XSD.long),
    "xsd:float":    str(XSD.float),
    "xsd:double":   str(XSD.double),
    "xsd:decimal":  str(XSD.decimal),
    "xsd:date":     str(XSD.date),
    "xsd:dateTime": str(XSD.dateTime),
    "xsd:time":     str(XSD.time),
    "xsd:anyURI":   str(XSD.anyURI),
    "rdf:langString": str(RDF.langString),
    "rdfs:Literal":   str(RDFS.Literal),
}


def resolve_type_uri(attr_type: str) -> tuple[str, bool]:
    """
    Given a caller-supplied type string (friendly name, prefixed form, or full URI),
    return (canonical_uri, is_primitive).

    is_primitive=True  → declare as owl:DatatypeProperty, add rdfs:range with the URI.
    is_primitive=False → declare as owl:ObjectProperty, add rdfs:range with the URI.
    """
    if not attr_type:
        return ("", False)

    # Already a full URI? Check directly against the primitive map.
    if "://" in attr_type or attr_type.startswith("urn:"):
        prim = primitive_from_range(URIRef(attr_type))
        return (attr_type, prim is not None)

    # Friendly / prefixed name: look up in reverse map (case-insensitive key).
    canonical = _FRIENDLY_TO_XSD_URI.get(attr_type) or _FRIENDLY_TO_XSD_URI.get(attr_type.lower())
    if canonical:
        return (canonical, True)   # all entries in the map are primitives

    # Unknown name — treat as a class URI fragment; caller is responsible for
    # passing a valid URI in this case.
    return (attr_type, False)

def role_name_from_label(label: str | None) -> str | None:
    """
    EA diagrams often show role names like '+isAbout'. We'll prefix with '+' if label exists.
    """
    if not label:
        return None
    return f"+{label}"

# ---------- Extraction ----------

def extract_ontology_package(g: Graph, custom_name: str | None = None) -> dict:
    # Identify owl:Ontology subject(s)
    ontos = list(g.subjects(RDF.type, OWL.Ontology))
    if ontos:
        onto = ontos[0]
        name = custom_name or get_label(g, onto, "en") or local_name(onto)
        pkg_uri = str(onto)
    else:
        # Fallback package when no owl:Ontology triple is present
        name = custom_name or "Generated"
        pkg_uri = f"urn:pkg:{name}"
    pkg_id = ea_id("EAPK", pkg_uri)
    return {
        "name": name,
        "ID": pkg_id,
        "type": "uml:Package",
        "package": None,  # root has no container, or set to another if desired
        "tags": []  # you can add {"name":"uri","value": pkg_uri} if you need
    }

def build_model(g: Graph, package_name: str | None = None) -> dict:
    model = {"elements": [], "connectors": []}

    # Root package
    root_pkg = extract_ontology_package(g, custom_name=package_name)
    model["elements"].append(root_pkg)
    root_pkg_id = root_pkg["ID"]

    # Index classes for quick lookup
    class_elems: dict[str, dict] = {}

    # 1) Classes
    for cls in g.subjects(RDF.type, OWL.Class):
        uri = str(cls)
        name = get_label(g, cls, "en") or local_name(cls)
        cls_id = ea_id("EAID", uri)
        tags = [{"name": "uri", "value": uri}]
        comment = get_comment(g, cls, "en")
        if comment:
            tags.append({"name": "definition-en", "value": comment})
        label = get_label(g, cls, "en")
        if label:
            tags.append({"name": "label-en", "value": label})
        # Optional: usage scope tag aligned to many SEMIC models
        tags.append({"name": "class-usage-scope", "value": "main"})

        elem = {
            "name": name,
            "ID": cls_id,
            "type": "uml:Class",
            "package": root_pkg_id,
            "tags": tags,
            "attributes": []
        }
        class_elems[uri] = elem
        model["elements"].append(elem)

    # Helper to ensure a class element exists for non-primitive ranges.
    # IMPORTANT: never call this for XSD/RDF primitive URIs — check
    # primitive_from_range first so we never create phantom class elements
    # for things like xsd:string.
    def ensure_class(uri_ref: URIRef) -> dict:
        u = str(uri_ref)
        # Safety guard: if somehow a primitive URI reaches here, refuse to
        # create a class element for it and return a sentinel empty dict.
        if primitive_from_range(uri_ref) is not None:
            raise ValueError(
                f"ensure_class called with primitive URI {u!r}. "
                "Use primitive_from_range instead."
            )
        if u not in class_elems:
            name = get_label(g, uri_ref, "en") or local_name(uri_ref)
            cls_id = ea_id("EAID", u)
            elem = {
                "name": name,
                "ID": cls_id,
                "type": "uml:Class",
                "package": root_pkg_id,
                "tags": [{"name": "uri", "value": u}],
                "attributes": []
            }
            class_elems[u] = elem
            model["elements"].append(elem)
        return class_elems[u]

    # 2) Datatype properties -> attributes on their domain class.
    #    These MUST NOT produce connectors.
    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        prop_uri = str(prop)
        prop_label = get_label(g, prop, "en")
        prop_comment = get_comment(g, prop, "en")
        # domains
        for domain in g.objects(prop, RDFS.domain):
            domain_uri = str(domain)
            if domain_uri not in class_elems:
                ensure_class(domain)
            domain_elem = class_elems[domain_uri]
            # range
            ranges = list(g.objects(prop, RDFS.range)) or [RDFS.Literal]
            for rng in ranges:
                primitive = primitive_from_range(rng)
                if primitive:
                    attr_type = primitive
                else:
                    # Non-primitive custom datatype: surface as the class name
                    # but still keep it as an attribute (not a connector).
                    rng_elem = ensure_class(rng)
                    attr_type = rng_elem["name"]

                attr_name = prop_label or local_name(prop)
                # Use "tags_attribute" to match the XMI JSON convention.
                attr_tags = [{"name": "uri", "value": prop_uri}]
                if prop_label:
                    attr_tags.append({"name": "label-en", "value": prop_label})
                if prop_comment:
                    attr_tags.append({"name": "definition-en", "value": prop_comment})

                domain_elem["attributes"].append({
                    "name": attr_name,
                    "type": attr_type,
                    "lower_bounds": "",
                    "upper_bounds": "",
                    "tags_attribute": attr_tags,
                })

    # 3) Object properties -> associations
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_uri = str(prop)
        prop_label = get_label(g, prop, "en")
        prop_comment = get_comment(g, prop, "en")

        domains = list(g.objects(prop, RDFS.domain))
        ranges = list(g.objects(prop, RDFS.range))
        # If any side is missing, we cannot create a robust connector
        if not domains or not ranges:
            continue

        for domain in domains:
            domain_elem = ensure_class(domain)
            for rng in ranges:
                range_elem = ensure_class(rng)
                connector = {
                    "source_name": domain_elem["name"],
                    "target_name": range_elem["name"],
                    "relationship": "Association",
                    "lb": None,             # left multiplicity bound (optional)
                    "lt": None,             # left role (optional)
                    "rb": None,             # right multiplicity bound (optional)
                    "rt": role_name_from_label(prop_label),  # right role name
                    "tags": [],
                    "tags_source": [],
                    "tags_target": []
                }
                # Put property metadata under tags_target (as in your example)
                tgt_tags = [{"name": "uri", "value": prop_uri}]
                if prop_comment:
                    tgt_tags.append({"name": "definition-en", "value": prop_comment})
                if prop_label:
                    tgt_tags.append({"name": "label-en", "value": prop_label})
                connector["tags_target"] = tgt_tags
                model["connectors"].append(connector)

    # 4) rdfs:subClassOf -> generalizations
    for child, parent in g.subject_objects(RDFS.subClassOf):
        # skip blank-node restrictions here; focus on named superclass
        if isinstance(parent, URIRef):
            child_elem = ensure_class(child)
            parent_elem = ensure_class(parent)
            gen = {
                "source_name": child_elem["name"],
                "target_name": parent_elem["name"],
                "relationship": "Generalization",
                "lb": None, "lt": None, "rb": None, "rt": None,
                "tags": [], "tags_source": [], "tags_target": []
            }
            model["connectors"].append(gen)

    return model

def _iri_base(iri: str) -> str:
    if not iri or "://" not in iri:
        return None
    # Split on last slash or hash (support both styles)
    for sep in ("/", "#"):
        if sep in iri:
            head, tail = iri.rsplit(sep, 1)
            return head if head else None
    return None

def _label_matches_en(lbl_node: dict[str, Any], expected: str) -> bool:
    return (
        isinstance(lbl_node, dict)
        and lbl_node.get("@language") == "en"
        and lbl_node.get("@value") == expected
    )

def _find_xmi_class_iri_by_name(model: dict[str, Any], class_name: str) -> str:
    xmi = model.get("xmi", {})
    for el in xmi.get("elements", []):
        if el.get("type") == "uml:Class" and el.get("name") == class_name:
            for t in el.get("tags", []):
                if t.get("name") == "uri":
                    return t.get("value")
    return None

def find_class_by_label(g: Graph, class_label: str):
    for s in g.subjects():
        label = g.value(s, RDFS.label)
        type = g.value(s, RDF.type)
        if label and str(label).lower() == class_label.lower() and str(type) == str(OWL.Class):
            uri = s
            return uri
    return None

def _find_file(
    user: str,
    name: str,
) -> Path:
    return MODELS_PATH/user/f"{name}.json"


def upload_model(
    model: dict[str, Any],
    user: str = "",
    name: str = "",
) -> dict[str, Any]:
    """
    Save a model in JSON format.
    """
    folder_path = MODELS_PATH/user
    if not exists(folder_path):
        mkdir(folder_path)

    with open(_find_file(user, name), "w") as f:
        dump(model, f)

    with open(_find_file(user, name), "r") as f:
        model = load(f)

    return model


def _serialize_graph(g: Graph, model: dict[str, Any]) -> dict[str, Any]:
    """
    Serialize an RDF graph back into both the 'ttl' (JSON-LD) and 'ttl_raw' (Turtle)
    fields of a model dict, and rebuild the 'xmi' field.

    This is a shared helper used by add_class, add_attribute, and add_connector
    to avoid duplicating serialization logic.
    """
    json_ld_str: str = g.serialize(format="json-ld", indent=4)
    model["ttl"] = json.loads(json_ld_str) if json_ld_str else {}

    ttl_raw = g.serialize(format="turtle")
    model["ttl_raw"] = (
        ttl_raw.decode("utf-8") if isinstance(ttl_raw, (bytes, bytearray)) else str(ttl_raw)
    )

    model["xmi"] = build_model(g)
    return model


def _save_and_reload(fp: Path, model: dict[str, Any]) -> dict[str, Any]:
    """Write model to disk and return the freshly loaded copy."""
    with open(fp, "w") as f:
        dump(model, f)
    with open(fp, "r") as f:
        return load(f)


def add_class(
    title: str,
    definition: str,
    usage_note: str,
    user: str = "",
    name: str = "",
    package: str = "",
    ID: str = "",
) -> dict[str, Any]:
    """
    Creates a dictionary representing a UML class with associated metadata,
    and adds it to the `user`'s model `name`.

    :param package_id: The ID of the package to which the class belongs.
    :param title: The name of the class.
    :param definition: A textual definition or description of the class.
    :param usage_note: Additional usage notes for the class.
    :return: A dictionary with the following keys:
        - "name": The name of the class (title).
        - "ID": A unique identifier for the class.
        - "type": The UML element type, set to "uml:Class".
        - "package": The ID of the package to which the class belongs.
        - "tags": A list of dictionaries containing metadata tags:
            - "definition": A tag for the class's definition.
            - "label": A tag for the class's label or title.
            - "usage_note": A tag for the class's usage notes.
    """
    fp = _find_file(user, name)
    if not exists(fp):
        return {}

    with open(fp, "r") as f:
        model: dict[str, Any] = load(f)

    # --- OWL/TTL mode ---
    if "ttl_raw" in model:
        g = Graph()
        g.parse(data=model["ttl_raw"], format="turtle")

        # Prevent duplicates by label
        if find_class_by_label(g, title):
            return {"error": "Class already exists"}

        new_class_uri = URIRef(ID)
        g.add((new_class_uri, RDF.type, OWL.Class))
        g.add((new_class_uri, RDFS.label, Literal(title, lang="en")))
        g.add((new_class_uri, RDFS.comment, Literal(definition, lang="en")))
        g.add((new_class_uri, SKOS.scopeNote, Literal(usage_note, lang="en")))

        model = _serialize_graph(g, model)
        model = _save_and_reload(fp, model)

        # Return the XMI element that corresponds to the newly added class so the
        # caller gets a predictable, structured object rather than a random JSON-LD node.
        for el in model.get("xmi", {}).get("elements", []):
            if el.get("type") == "uml:Class" and el.get("name") == title:
                return el
        return {}

    # --- XMI JSON mode ---
    element: dict[str, Any] = {
        "name": title,
        "ID": ID,
        "type": "uml:Class",
        "package": package,
        "tags": [
            {"name": "label", "value": title},
            {"name": "definition", "value": definition},
            {"name": "usage_note", "value": usage_note},
        ],
        # Ensure the attributes list is always present so add_attribute works
        # without having to handle its absence separately.
        "attributes": [],
    }

    model.setdefault("elements", []).append(element)
    model = _save_and_reload(fp, model)

    elements: list[dict[str, Any]] = model.get("elements", [])
    return elements[-1] if elements else {}


def add_attribute(
    class_name: str,
    attr_label: str,
    attr_definition: str,
    attr_uri: str,
    attr_usage_note: str = "",
    attr_type: str = "",
    user: str = "",
    name: str = "",
) -> dict[str, Any]:
    """
    Adds an attribute to an existing UML class in the model file belonging to `user` and model `name`.

    JSON shape produced:
      {
        "name": "<attr_label>",
        "type": "<attr_type>",
        "lower_bounds": "",
        "upper_bounds": "",
        "tags_attribute": [
          { "name": "uri",          "value": "<attr_uri>" },
          { "name": "label-en",     "value": "<attr_label>" },
          { "name": "definition-en","value": "<attr_definition>" },
          { "name": "usageNote-en", "value": "<attr_usage_note>" }
        ]
      }

    Preconditions:
      - The model file path is resolved via `_find_file(user, name)` and must exist.
      - A UML class element with `type == "uml:Class"` and `name == class_name` must exist.

    Parameters:
      :param class_name: The exact UML class name to which the attribute will be added.
      :param attr_label: The attribute label.
      :param attr_definition: Human-readable definition for the attribute.
      :param attr_uri: URI identifying the attribute property.
      :param attr_usage_note: Optional usage note; defaults to empty string.
      :param attr_type: Optional type name or XSD/class URI for the attribute.
      :param user: Logical user/tenant identifier.
      :param name: Logical model name (file identifier).

    Returns:
      - The stored attribute dictionary, or an error dict.
    """
    fp = _find_file(user, name)
    if not exists(fp):
        return {"error": "Model file not found"}

    with open(fp, "r") as f:
        model: dict[str, Any] = load(f)

    # --- OWL/TTL mode ---
    if "ttl_raw" in model:
        g = Graph()
        g.parse(data=model["ttl_raw"], format="turtle")

        class_uri = find_class_by_label(g, class_name)
        if not class_uri:
            return {"error": "Class does not exist"}

        prop_uri = URIRef(attr_uri)
        g.add((prop_uri, RDFS.label, Literal(attr_label, lang="en")))
        g.add((prop_uri, RDFS.comment, Literal(attr_definition, lang="en")))
        g.add((prop_uri, SKOS.scopeNote, Literal(attr_usage_note, lang="en")))
        g.add((prop_uri, RDFS.domain, class_uri))

        # Resolve the caller-supplied type (friendly name, short prefix, or full URI)
        # to a canonical URI and learn whether it is a primitive/datatype.
        # This is the critical step: using primitive_from_range(URIRef(attr_type))
        # directly would ALWAYS return None for friendly names like "Text", "String",
        # "GenericDate", or short forms like "xsd:string", causing every attribute to
        # be mis-declared as an ObjectProperty and generate a phantom class + connector.
        canonical_range_uri, is_primitive = resolve_type_uri(attr_type)

        if is_primitive:
            g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            if canonical_range_uri:
                g.add((prop_uri, RDFS.range, URIRef(canonical_range_uri)))
        else:
            # Genuine object property: range is a named class URI.
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            if canonical_range_uri:
                g.add((prop_uri, RDFS.range, URIRef(canonical_range_uri)))

        model = _serialize_graph(g, model)
        model = _save_and_reload(fp, model)

        # Return the matching attribute from the rebuilt XMI.
        # Use "tags_attribute" (not "tags") to match build_model's output convention.
        for el in model.get("xmi", {}).get("elements", []):
            if el.get("type") == "uml:Class" and el.get("name") == class_name:
                attrs = el.get("attributes", [])
                for attr in attrs:
                    for tag in attr.get("tags_attribute", []):
                        if tag.get("name") == "uri" and tag.get("value") == attr_uri:
                            return attr
                return attrs[-1] if attrs else {}
        return {}

    # --- XMI mode ---
    elements: list[dict[str, Any]] = model.get("elements", [])
    target_class: dict[str, Any] = next(
        (el for el in elements if el.get("type") == "uml:Class" and el.get("name") == class_name),
        None,
    )
    if not target_class:
        return {"error": "Class not found"}

    # FIX: guarantee the attributes list exists before appending.
    if not isinstance(target_class.get("attributes"), list):
        target_class["attributes"] = []

    attribute: dict[str, Any] = {
        "name": attr_label,
        "type": attr_type,
        "lower_bounds": "",
        "upper_bounds": "",
        "tags_attribute": [
            {"name": "uri",           "value": attr_uri},
            {"name": "label-en",      "value": attr_label},
            {"name": "definition-en", "value": attr_definition},
            {"name": "usageNote-en",  "value": attr_usage_note},
        ],
    }
    target_class["attributes"].append(attribute)

    # FIX: persist the mutated model (which already holds the updated target_class
    # in-memory) and reload to return the stored copy.
    model = _save_and_reload(fp, model)

    elements = model.get("elements", [])
    target_class = next(
        (el for el in elements if el.get("type") == "uml:Class" and el.get("name") == class_name),
        None,
    )
    if not target_class:
        return {}
    attrs = target_class.get("attributes", [])
    return attrs[-1] if attrs else {}


def add_connector(
    source_name: str,
    target_name: str,
    rel_label: str,
    rel_definition: str,
    rel_uri: str,
    relationship: str,  # e.g., "Association", "Aggregation", "Composition", "Generalization", "Realization"
    rb: str = None,
    rt: str = None,
    rel_usage_note: str = "",
    user: str = "",
    name: str = "",
) -> dict[str, Any]:
    """
    Adds a connector between two existing elements identified by their names and annotates
    the target end with semantic tags (URI, label, definition, usage note).

    JSON shape produced:
      {
        "source_name": "<source_name>",
        "target_name": "<target_name>",
        "relationship": "<Association|Generalization|...>",
        "lb": null,
        "lt": null,
        "rb": "<rb or null>",
        "rt": "<rt or null>",
        "tags": [],
        "tags_source": [],
        "tags_target": [
          { "name": "uri",          "value": "<rel_uri>" },
          { "name": "label-en",     "value": "<rel_label>" },
          { "name": "definition-en","value": "<rel_definition>" },
          { "name": "usageNote-en", "value": "<rel_usage_note>" }
        ]
      }

    Parameters:
      :param source_name: Name of the source element (must exist in `elements`).
      :param target_name: Name of the target element (must exist in `elements`).
      :param rel_label: Human-readable role/relationship label.
      :param rel_definition: Definition for the relationship end.
      :param rel_uri: URI identifying the relationship / object property.
      :param relationship: UML relationship type (e.g., "Association", "Generalization").
      :param rb: Optional right-end multiplicity (e.g., "0..*", "1").
      :param rt: Optional right-end role name (e.g., "+domicile").
      :param rel_usage_note: Optional usage note for the relationship end.
      :param user: Logical user identifier.
      :param name: Logical model name (file identifier).

    Returns:
      - The stored connector dictionary, or an error dict.
    """
    fp = _find_file(user, name)
    if not exists(fp):
        return {"error": "Model file not found"}

    with open(fp, "r") as f:
        model: dict[str, Any] = load(f)

    # --- OWL/TTL mode ---
    if "ttl_raw" in model:
        g = Graph()
        g.parse(data=model["ttl_raw"], format="turtle")

        source_uri = find_class_by_label(g, source_name)
        if not source_uri:
            return {"error": f"Source class '{source_name}' does not exist"}

        # FIX: also resolve the target class so we can set rdfs:range correctly.
        target_uri = find_class_by_label(g, target_name)
        if not target_uri:
            return {"error": f"Target class '{target_name}' does not exist"}

        prop_uri = URIRef(rel_uri)
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDFS.label, Literal(rel_label, lang="en")))
        g.add((prop_uri, RDFS.comment, Literal(rel_definition, lang="en")))
        g.add((prop_uri, SKOS.scopeNote, Literal(rel_usage_note, lang="en")))
        g.add((prop_uri, RDFS.domain, source_uri))
        # FIX: set the range to the target class URI (was entirely missing before).
        g.add((prop_uri, RDFS.range, target_uri))

        model = _serialize_graph(g, model)
        model = _save_and_reload(fp, model)

        # Return the matching connector from the rebuilt XMI.
        for conn in model.get("xmi", {}).get("connectors", []):
            for tag in conn.get("tags_target", []):
                if tag.get("name") == "uri" and tag.get("value") == rel_uri:
                    return conn
        # Fallback: return last connector.
        connectors = model.get("xmi", {}).get("connectors", [])
        return connectors[-1] if connectors else {}

    # --- XMI mode ---
    elements: list[dict[str, Any]] = model.get("elements", [])

    # FIX: restrict the existence check to actual classes/elements, not packages,
    # to avoid false positives when a package shares a name with a class.
    src_exists = any(
        el.get("name") == source_name and el.get("type") != "uml:Package"
        for el in elements
    )
    tgt_exists = any(
        el.get("name") == target_name and el.get("type") != "uml:Package"
        for el in elements
    )
    if not src_exists:
        return {"error": f"Source element '{source_name}' not found"}
    if not tgt_exists:
        return {"error": f"Target element '{target_name}' not found"}

    connector: dict[str, Any] = {
        "source_name": source_name,
        "target_name": target_name,
        "relationship": relationship,
        "lb": None,
        "lt": None,
        "rb": rb,
        "rt": rt,
        "tags": [],
        "tags_source": [],
        "tags_target": [
            {"name": "uri",           "value": rel_uri},
            {"name": "label-en",      "value": rel_label},
            {"name": "definition-en", "value": rel_definition},
            {"name": "usageNote-en",  "value": rel_usage_note},
        ],
    }

    # FIX: use setdefault so the key is always present whether or not the
    # original model file was created with it.
    model.setdefault("connectors", []).append(connector)

    model = _save_and_reload(fp, model)

    connectors: list[dict[str, Any]] = model.get("connectors", [])
    return connectors[-1] if connectors else {}
