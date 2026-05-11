import os
import re
import json
import time
import requests
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase


try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    types = None
    GEMINI_AVAILABLE = False

from typing import Any
from typing_extensions import TypedDict

class QueryPlan(TypedDict):
    use_kg: bool
    query: str
    params: dict[str, Any]
    rationale: str

# -----------------------
# Load .env / Streamlit secrets
# -----------------------

def get_config_value(name: str, default: str = "", aliases: list[str] | None = None) -> str:
    """
    Reads config from Streamlit secrets first, then environment variables.
    This works both locally and after deployment to Streamlit Cloud.
    """
    keys = [name] + (aliases or [])

    for key in keys:
        try:
            if key in st.secrets and str(st.secrets[key]).strip():
                return str(st.secrets[key]).strip()
        except Exception:
            # st.secrets may not be available outside Streamlit runtime
            pass

    return default


# Neo4j Aura expects an encrypted URI like:
# neo4j+s://xxxxxxxx.databases.neo4j.io
# Keep localhost as fallback so the same file can still run during local testing.
DEFAULT_NEO4J_URI = get_config_value("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = get_config_value("NEO4J_USERNAME", "neo4j", aliases=["NEO4J_USER"])
DEFAULT_NEO4J_PASSWORD = get_config_value("NEO4J_PASSWORD", "")
DEFAULT_NEO4J_DB = get_config_value("NEO4J_DATABASE", "neo4j", aliases=["NEO4J_DB"])

DEFAULT_GRAPHDB_URL = get_config_value("GRAPHDB_URL", "http://localhost:7200")
DEFAULT_GRAPHDB_REPO = get_config_value("GRAPHDB_REPOSITORY", "test")

DEFAULT_ARANGO_URL = get_config_value("ARANGO_URL", "http://localhost:8529")
DEFAULT_ARANGO_DB = get_config_value("ARANGO_DB", "_system")
DEFAULT_ARANGO_USER = get_config_value("ARANGO_USER", "root")
DEFAULT_ARANGO_PASSWORD = get_config_value("ARANGO_PASSWORD", "")

DEFAULT_GEMINI_API_KEY = get_config_value("GEMINI_API_KEY", "")
DEFAULT_MODEL = get_config_value("GEMINI_MODEL", "gemini-3-flash-preview")

st.set_page_config(page_title="LLM + Neo4j Aura", layout="wide")
st.title("Unified Plant Engineering Assistant")


# ============================================================
# Hidden settings
# ============================================================
# All settings are read from .streamlit/secrets.toml locally
# or from Streamlit Cloud > App > Settings > Secrets after deployment.
# Nothing is shown in the sidebar, so the supervisor only sees the app workflow.
neo4j_uri = DEFAULT_NEO4J_URI
neo4j_user = DEFAULT_NEO4J_USER
neo4j_password = DEFAULT_NEO4J_PASSWORD
neo4j_db = DEFAULT_NEO4J_DB

gemini_api_key = DEFAULT_GEMINI_API_KEY
model_name = DEFAULT_MODEL

shared_system_prompt = get_config_value(
    "SYSTEM_PROMPT",
    (
        "You are a careful thesis research assistant. "
        "Use the provided query results faithfully. "
        "Do not invent labels, relationships, classes, predicates, collections, or properties. "
        "If results exist, answer from results only."
    )
)

try:
    shared_temp = float(get_config_value("LLM_TEMPERATURE", "1.0"))
except ValueError:
    shared_temp = 1.0

try:
    shared_max_tokens = int(get_config_value("LLM_MAX_OUTPUT_TOKENS", "2048"))
except ValueError:
    shared_max_tokens = 2048


# ============================================================
# Neo4j helpers
# ============================================================
@st.cache_resource(show_spinner=False)
def get_neo4j_driver(uri, user, password):
    uri = (uri or "").strip()
    user = (user or "").strip()
    password = (password or "").strip()

    if not uri:
        raise ValueError("Neo4j URI is missing. For Aura use: neo4j+s://xxxxx.databases.neo4j.io")
    if not user:
        raise ValueError("Neo4j username is missing.")
    if not password:
        raise ValueError("Neo4j password is missing.")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def run_cypher(driver, query, params=None, database="neo4j"):
    params = params or {}
    with driver.session(database=database) as session:
        result = session.run(query, params)
        return [r.data() for r in result]


# ============================================================
# Gemini helpers
# ============================================================
@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str):
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini SDK not installed. Run: pip install -U google-genai")
    if api_key and api_key.strip():
        return genai.Client(api_key=api_key.strip())
    return genai.Client()


def gemini_response_text(response):
    text = getattr(response, "text", None)
    if text:
        return text

    texts = []
    try:
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    texts.append(part_text)
    except Exception:
        pass

    return "\n".join(texts).strip() if texts else str(response)


def chat_with_gemini(api_key, model, prompt, system=None, context=None, temperature=1.0, max_tokens=None, response_mime_type=None, response_schema=None):
    ctx_block = ""
    if context is not None:
        if isinstance(context, (list, dict)):
            context_text = json.dumps(context, ensure_ascii=False, indent=2)
        else:
            context_text = str(context)
        ctx_block = f"\n\n[CONTEXT_JSON]\n{context_text}\n[/CONTEXT_JSON]\n"

    full_prompt = f"{prompt}{ctx_block}"

    client = get_gemini_client(api_key)

    config_kwargs = {}
    if system:
        config_kwargs["system_instruction"] = system
    if temperature is not None:
        config_kwargs["temperature"] = float(temperature)
    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = int(max_tokens)
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    if response_schema is not None:
        config_kwargs["response_schema"] = response_schema

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=config
    )
    return gemini_response_text(response)

def gemini_debug_info(response):
    out = {
        "text": getattr(response, "text", None),
        "finish_reasons": [],
        "usage_metadata": getattr(response, "usage_metadata", None),
    }

    try:
        for c in getattr(response, "candidates", []) or []:
            out["finish_reasons"].append(getattr(c, "finish_reason", None))
    except Exception:
        pass

    return out

# ============================================================
# GraphDB helpers
# ============================================================
def graphdb_repo_url(base_url: str, repo: str) -> str:
    return f"{base_url.rstrip('/')}/repositories/{repo}"


def run_sparql(graphdb_base_url, repository, query):
    url = graphdb_repo_url(graphdb_base_url, repository)
    headers = {
        "Accept": "application/sparql-results+json, application/json, text/turtle, application/ld+json, text/plain"
    }
    resp = requests.post(
        url,
        data={"query": query},
        headers=headers,
        timeout=120
    )
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()

    if "application/sparql-results+json" in content_type or "application/json" in content_type:
        data = resp.json()

        if "boolean" in data:
            return {"type": "ask", "rows": [{"boolean": data["boolean"]}], "raw": data}

        vars_ = data.get("head", {}).get("vars", [])
        bindings = data.get("results", {}).get("bindings", [])
        rows = []
        for b in bindings:
            row = {}
            for var in vars_:
                row[var] = b.get(var, {}).get("value")
            rows.append(row)

        return {"type": "select", "rows": rows, "raw": data}

    return {"type": "text", "text": resp.text, "raw": resp.text}


# ============================================================
# ArangoDB helpers
# ============================================================
def arango_auth_tuple(user, password):
    return (user, password)


def run_aql_query(arango_base_url, database, user, password, query, bind_vars=None):
    bind_vars = bind_vars or {}
    url = f"{arango_base_url.rstrip('/')}/_db/{database}/_api/cursor"
    payload = {
        "query": query,
        "bindVars": bind_vars
    }
    resp = requests.post(
        url,
        json=payload,
        auth=arango_auth_tuple(user, password),
        timeout=120
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("result", []), data


def list_arango_collections(arango_base_url, database, user, password):
    url = f"{arango_base_url.rstrip('/')}/_db/{database}/_api/collection"
    resp = requests.get(
        url,
        auth=arango_auth_tuple(user, password),
        timeout=60
    )
    resp.raise_for_status()
    return resp.json().get("result", [])

# ============================================================
# Schema normalization
# ============================================================
def unique_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        marker = json.dumps(x, sort_keys=True, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
        if marker not in seen:
            seen.add(marker)
            out.append(x)
    return out

def stringify_value(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def split_namespace(uri: str):
    if "#" in uri:
        ns, local = uri.rsplit("#", 1)
        return ns + "#", local
    if "/" in uri:
        ns, local = uri.rsplit("/", 1)
        return ns + "/", local
    return uri, uri

def infer_prefixes_from_uris(uris):
    standard = {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
        "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
        "http://www.w3.org/2001/XMLSchema#": "xsd",
        "http://www.w3.org/2002/07/owl#": "owl",
    }

    ns_to_prefix = dict(standard)
    used = set(standard.values())
    counter = 1

    for uri in uris:
        if not isinstance(uri, str) or not uri.startswith("http"):
            continue
        ns, _ = split_namespace(uri)
        if ns in ns_to_prefix:
            continue

        if "example.org" in ns and "ex" not in used:
            prefix = "ex"
        else:
            while f"ns{counter}" in used:
                counter += 1
            prefix = f"ns{counter}"

        ns_to_prefix[ns] = prefix
        used.add(prefix)

    return {prefix: ns for ns, prefix in ns_to_prefix.items()}

def compact_uri(uri: str, prefixes: dict):
    if not isinstance(uri, str):
        return uri
    for prefix, ns in prefixes.items():
        if uri.startswith(ns):
            return f"{prefix}:{uri[len(ns):]}"
    return uri

def compact_list(uris, prefixes):
    return [compact_uri(u, prefixes) for u in uris]

def sanitize_arango_doc(doc: dict):
    if not isinstance(doc, dict):
        return doc
    return {k: v for k, v in doc.items() if k not in ["_id", "_key", "_rev"]}
# ============================================================
# Schema helpers
# ============================================================
    # ----------------------------
    # Neo4j
    # ----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_neo4j_schema(_driver, database="neo4j"):
    schema = {}

    labels_res = run_cypher(
        _driver,
        "CALL db.labels() YIELD label RETURN collect(label) AS labels",
        database=database
    )
    schema["labels"] = labels_res[0]["labels"] if labels_res else []

    rels_res = run_cypher(
        _driver,
        "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS rels",
        database=database
    )
    schema["relationship_types"] = rels_res[0]["rels"] if rels_res else []

    props_res = run_cypher(
        _driver,
        "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS props",
        database=database
    )
    schema["property_keys"] = props_res[0]["props"] if props_res else []

    label_details = {}
    for label in schema["labels"]:
        q = f"""
        MATCH (n:`{label}`)
        WITH n LIMIT 25
        UNWIND keys(n) AS k
        RETURN k AS property, collect(DISTINCT n[k])[0..5] AS example_values
        ORDER BY property
        """
        try:
            rows = run_cypher(_driver, q, database=database)
            label_details[label] = rows
        except Exception:
            label_details[label] = []
    schema["label_properties"] = label_details

    rel_patterns_query = """
    MATCH (a)-[r]->(b)
    RETURN
      labels(a) AS from_labels,
      type(r) AS rel_type,
      labels(b) AS to_labels,
      count(*) AS count
    ORDER BY count DESC
    LIMIT 200
    """
    try:
        schema["relationship_patterns"] = run_cypher(_driver, rel_patterns_query, database=database)
    except Exception:
        schema["relationship_patterns"] = []

    example_nodes = {}
    for label in schema["labels"]:
        q = f"""
        MATCH (n:`{label}`)
        RETURN n
        LIMIT 3
        """
        try:
            example_nodes[label] = run_cypher(_driver, q, database=database)
        except Exception:
            example_nodes[label] = []
    schema["example_nodes"] = example_nodes

    schema["entity_types"] = schema["labels"]
    schema["relation_types"] = schema["relationship_types"]
    schema["type_properties"] = schema["label_properties"]
    schema["example_entities"] = schema["example_nodes"]
    schema["backend_specific"] = {
        "labels": schema["labels"],
        "relationship_types": schema["relationship_types"],
        "property_keys_raw": schema["property_keys"],
    }

    return schema

    # ----------------------------
    # graphDB
    # ----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_graphdb_schema(graphdb_base_url, repository):
    schema = {}

    class_query = """
    SELECT DISTINCT ?class
    WHERE {
      ?s a ?class .
    }
    LIMIT 100
    """

    all_pred_query = """
    SELECT DISTINCT ?p
    WHERE {
      ?s ?p ?o .
      FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
    }
    LIMIT 300
    """

    literal_pred_query = """
    SELECT DISTINCT ?p
    WHERE {
      ?s ?p ?o .
      FILTER(isLiteral(?o))
      FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
    }
    LIMIT 300
    """

    object_pred_query = """
    SELECT DISTINCT ?p
    WHERE {
      ?s ?p ?o .
      FILTER(isIRI(?o) || isBlank(?o))
      FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
    }
    LIMIT 300
    """

    relationship_pattern_query = """
    SELECT ?sType ?p ?oType (COUNT(*) AS ?count)
    WHERE {
      ?s a ?sType .
      ?s ?p ?o .
      FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
      OPTIONAL { ?o a ?oType . }
    }
    GROUP BY ?sType ?p ?oType
    ORDER BY DESC(?count)
    LIMIT 200
    """

    sample_query = """
    SELECT ?s ?p ?o
    WHERE {
      ?s ?p ?o .
    }
    LIMIT 30
    """

    try:
        class_res = run_sparql(graphdb_base_url, repository, class_query)
        classes = [r["class"] for r in class_res.get("rows", [])]
    except Exception:
        classes = []

    try:
        all_pred_res = run_sparql(graphdb_base_url, repository, all_pred_query)
        all_predicates = [r["p"] for r in all_pred_res.get("rows", [])]
    except Exception:
        all_predicates = []

    try:
        literal_pred_res = run_sparql(graphdb_base_url, repository, literal_pred_query)
        literal_predicates = [r["p"] for r in literal_pred_res.get("rows", [])]
    except Exception:
        literal_predicates = []

    try:
        object_pred_res = run_sparql(graphdb_base_url, repository, object_pred_query)
        object_predicates = [r["p"] for r in object_pred_res.get("rows", [])]
    except Exception:
        object_predicates = []

    try:
        pattern_res = run_sparql(graphdb_base_url, repository, relationship_pattern_query)
        relationship_patterns_raw = pattern_res.get("rows", [])
    except Exception:
        relationship_patterns_raw = []

    try:
        sample_res = run_sparql(graphdb_base_url, repository, sample_query)
        sample_triples = sample_res.get("rows", [])
    except Exception:
        sample_triples = []

    uri_pool = []
    uri_pool.extend(classes)
    uri_pool.extend(all_predicates)
    for row in sample_triples:
        for k in ["s", "p", "o"]:
            v = row.get(k)
            if isinstance(v, str) and v.startswith("http"):
                uri_pool.append(v)

    prefixes = infer_prefixes_from_uris(uri_pool)

    type_properties = {}
    example_entities = {}

    for cls in classes:
        prop_query = f"""
        SELECT ?s ?p ?o
        WHERE {{
          ?s a <{cls}> .
          ?s ?p ?o .
          FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
        }}
        LIMIT 200
        """
        subject_rows = []
        try:
            prop_res = run_sparql(graphdb_base_url, repository, prop_query)
            subject_rows = prop_res.get("rows", [])
        except Exception:
            subject_rows = []

        prop_map = {}
        entity_map = {}

        for row in subject_rows:
            s = row.get("s")
            p = row.get("p")
            o = row.get("o")

            if p:
                prop_map.setdefault(p, [])
                if o is not None and len(prop_map[p]) < 5:
                    prop_map[p].append(stringify_value(o))

            if s:
                entity_map.setdefault(s, {"id": compact_uri(s, prefixes), "properties": {}})
                if p and o is not None and len(entity_map[s]["properties"].get(compact_uri(p, prefixes), [])) < 5:
                    key = compact_uri(p, prefixes)
                    entity_map[s]["properties"].setdefault(key, [])
                    entity_map[s]["properties"][key].append(stringify_value(o))

        type_properties[compact_uri(cls, prefixes)] = [
            {
                "property": compact_uri(p, prefixes),
                "example_values": unique_preserve(vals)[:5]
            }
            for p, vals in sorted(prop_map.items())
        ]

        example_entities[compact_uri(cls, prefixes)] = list(entity_map.values())[:3]

    relationship_patterns = []
    for row in relationship_patterns_raw:
        relationship_patterns.append({
            "from_types": [compact_uri(row["sType"], prefixes)] if row.get("sType") else [],
            "rel_type": compact_uri(row["p"], prefixes) if row.get("p") else None,
            "to_types": [compact_uri(row["oType"], prefixes)] if row.get("oType") else [],
            "count": int(row["count"]) if row.get("count") else 0
        })

    schema["entity_types"] = compact_list(classes, prefixes)
    schema["relation_types"] = compact_list(object_predicates, prefixes)
    schema["property_keys"] = compact_list(literal_predicates, prefixes)
    schema["type_properties"] = type_properties
    schema["relationship_patterns"] = relationship_patterns
    schema["example_entities"] = example_entities

    schema["backend_specific"] = {
        "classes_raw": classes,
        "all_predicates_raw": all_predicates,
        "literal_predicates_raw": literal_predicates,
        "object_predicates_raw": object_predicates,
        "sample_triples": sample_triples,
        "prefixes": prefixes
    }

    # Keep old keys too if you still use them elsewhere
    schema["classes"] = classes
    schema["predicates"] = all_predicates
    schema["sample_triples"] = sample_triples
    schema["prefixes"] = prefixes

    return schema

    # ----------------------------
    # ArangoDB
    # ----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_arangodb_schema(arango_base_url, database, user, password):
    schema = {}

    try:
        collections = list_arango_collections(arango_base_url, database, user, password)
    except Exception:
        collections = []

    non_system = [c for c in collections if not c.get("isSystem", False)]
    doc_collections = [c for c in non_system if c.get("type") == 2]
    edge_collections = [c for c in non_system if c.get("type") == 3]

    entity_types = []
    relation_types = []
    property_keys_set = set()
    type_properties = {}
    example_entities = {}
    relationship_patterns_map = {}

    # ----------------------------
    # Document collections -> entity types
    # ----------------------------
    for c in doc_collections:
        name = c["name"]
        entity_types.append(name)

        try:
            rows, _ = run_aql_query(
                arango_base_url, database, user, password,
                f"FOR d IN `{name}` LIMIT 25 RETURN UNSET(d, ['_id', '_key', '_rev'])"
            )
        except Exception:
            rows = []

        rows = [sanitize_arango_doc(r) for r in rows if isinstance(r, dict)]

        prop_map = {}
        for doc in rows:
            for k, v in doc.items():
                property_keys_set.add(k)
                prop_map.setdefault(k, [])
                if len(prop_map[k]) < 5:
                    prop_map[k].append(stringify_value(v))

        type_properties[name] = [
            {
                "property": k,
                "example_values": unique_preserve(vals)[:5]
            }
            for k, vals in sorted(prop_map.items())
        ]

        example_entities[name] = rows[:3]

    # ----------------------------
    # Edge collections -> relation types + patterns
    # ----------------------------
    for c in edge_collections:
        edge_col = c["name"]

        try:
            edge_rows, _ = run_aql_query(
                arango_base_url, database, user, password,
                f"""
                FOR e IN `{edge_col}`
                  LIMIT 300
                  LET fromCol = PARSE_IDENTIFIER(e._from).collection
                  LET toCol = PARSE_IDENTIFIER(e._to).collection
                  RETURN MERGE(
                    UNSET(e, ['_id', '_key', '_rev']),
                    {{
                      _from_collection: fromCol,
                      _to_collection: toCol
                    }}
                  )
                """
            )
        except Exception:
            edge_rows = []

        for e in edge_rows:
            if not isinstance(e, dict):
                continue

            rel = e.get("relation") or e.get("type") or edge_col
            from_col = e.get("_from_collection")
            to_col = e.get("_to_collection")

            relation_types.append(rel)

            key = (from_col, rel, to_col)
            if key not in relationship_patterns_map:
                relationship_patterns_map[key] = 0
            relationship_patterns_map[key] += 1

    relationship_patterns = []
    for (from_col, rel, to_col), count in relationship_patterns_map.items():
        relationship_patterns.append({
            "from_types": [from_col] if from_col else [],
            "rel_type": rel,
            "to_types": [to_col] if to_col else [],
            "count": count
        })

    schema["entity_types"] = sorted(unique_preserve(entity_types))
    schema["relation_types"] = sorted(unique_preserve(relation_types))
    schema["property_keys"] = sorted(property_keys_set)
    schema["type_properties"] = type_properties
    schema["relationship_patterns"] = sorted(
        relationship_patterns,
        key=lambda x: x.get("count", 0),
        reverse=True
    )[:200]
    schema["example_entities"] = example_entities

    schema["backend_specific"] = {
        "collections": [
            {
                "name": c.get("name"),
                "type": c.get("type"),
                "system": c.get("isSystem", False)
            }
            for c in collections
        ],
        "document_collections": [c["name"] for c in doc_collections],
        "edge_collections": [c["name"] for c in edge_collections]
    }

    # Keep old keys too
    schema["collections"] = schema["backend_specific"]["collections"]
    schema["collection_samples"] = example_entities
    schema["collection_keys"] = {
        k: [item["property"] for item in v] for k, v in type_properties.items()
    }

    return schema

# ============================================================
# Guardrails
# ============================================================
def strip_comments(text: str) -> str:
    return re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.MULTILINE | re.DOTALL).strip().lower()


def is_readonly_cypher(query: str) -> bool:
    q = strip_comments(query)
    blocked = [
        "create ", "merge ", "set ", "delete ", "detach delete", "drop ",
        "remove ", "load csv", "call apoc.", "apoc.", "dbms.", "admin ",
        "grant ", "revoke "
    ]
    return not any(b in q for b in blocked)


def is_readonly_sparql(query: str) -> bool:
    q = strip_comments(query)
    blocked = [
        "insert", "delete", "load", "clear", "create", "drop", "move", "copy", "add"
    ]
    return not any(b in q for b in blocked)


def is_readonly_aql(query: str) -> bool:
    q = strip_comments(query)
    blocked = [
        "insert ", "update ", "replace ", "remove ", "upsert ", "truncate ",
        "drop ", "create "
    ]
    return not any(b in q for b in blocked)


# ============================================================
# LLM planners
# ============================================================
def extract_json(raw: str):
    raw = raw.strip()

    # remove markdown fences if model adds them
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    decoder = json.JSONDecoder()

    # try parsing from every "{" until one works
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(raw[i:])
                return obj
            except Exception:
                continue

    raise ValueError(f"No valid JSON object found in model output:\n{raw}")


def compact_neo4j_schema(schema: dict) -> dict:
    """
    Keep only the schema parts needed for query generation.
    This prevents Gemini from getting overloaded by huge example_nodes.
    """
    return {
        "labels": schema.get("labels", []),
        "relationship_types": schema.get("relationship_types", []),
        "property_keys": schema.get("property_keys", []),
        "label_properties": schema.get("label_properties", {}),
        "relationship_patterns": schema.get("relationship_patterns", []),
    }

def fallback_neo4j_plan(user_question: str):
    q = user_question.lower()

    if "flow velocity" in q or "velocity" in q:
        return {
            "use_kg": True,
            "query": """
MATCH (s:Stream {id: "stream-feed"})
MATCH (p:Pipe {id: "p1-ext-flash"})
RETURN
  s.volumetric_flow_rate AS volumetric_flow_rate_m3_per_h,
  p.cross_sectional_area AS cross_sectional_area_m2,
  (s.volumetric_flow_rate / 3600.0) / p.cross_sectional_area AS flow_velocity_m_per_s
LIMIT 50
""".strip(),
            "params": {},
            "rationale": "Fallback query for calculating flow velocity from volumetric flow rate and cross-sectional area."
        }

    return None
def plan_neo4j_query_with_llm(api_key, model, user_question, schema):
    planner_system = """
You are a Neo4j Cypher query planner.

Your job is to convert the user question into a Neo4j read-only Cypher query.

CRITICAL OUTPUT RULES:
- Return ONLY one valid JSON object.
- Do not return markdown.
- Do not use code fences.
- Do not write explanations outside JSON.
- Do not return raw Cypher outside the JSON.
- The first character of your response must be {.
- The last character of your response must be }.
- The JSON must be parseable by Python json.loads().
- Escape all newline characters inside the query string if needed.
- Do not include trailing commas.

Required JSON structure:
{
  "use_kg": true,
  "query": "MATCH ... RETURN ...",
  "params": {},
  "rationale": "short reason"
}

JSON field rules:
- use_kg must be true if a Cypher query can answer the question.
- use_kg must be false only if the question cannot be answered from the graph schema.
- query must be a complete Cypher query string when use_kg is true.
- query must be an empty string only when use_kg is false.
- params must always be an object.
- rationale must be short.

Cypher rules:
- Generate exactly ONE read-only Cypher query.
- Allowed clauses: MATCH, OPTIONAL MATCH, WITH, RETURN, WHERE, ORDER BY, LIMIT.
- Forbidden clauses: CREATE, MERGE, SET, DELETE, REMOVE, DROP, LOAD CSV, CALL, APOC.
- Do not use semicolons.
- Do not use comments.
- Use only labels, properties, and relationships from the provided schema.
- Do not invent labels.
- Do not invent properties.
- Do not invent relationships.
- If no relationship is needed, use separate MATCH statements.
- Use stable ids when available.
- Use LIMIT 50 by default unless the query returns one aggregated/calculated row.
- Put LIMIT at the end of the query after RETURN/ORDER BY.

Engineering calculation rule:
For flow velocity:
velocity_m_per_s = (volumetric_flow_rate / 3600.0) / cross_sectional_area

Use:
- Stream.volumetric_flow_rate
- Pipe.cross_sectional_area

Only calculate velocity if both values are available in the schema/query path.

If the question asks for flow velocity, generate a query like:
MATCH (s:Stream)-[:FLOWS]->(c:Conceptual)
OPTIONAL MATCH (c)-[:HAS_COMPONENT]->(p:Pipe)
WHERE s.volumetric_flow_rate IS NOT NULL
  AND p.cross_sectional_area IS NOT NULL
RETURN
  s.id AS stream_id,
  s.type AS stream_type,
  p.id AS pipe_id,
  s.volumetric_flow_rate AS volumetric_flow_rate_m3_per_h,
  p.cross_sectional_area AS cross_sectional_area_m2,
  (s.volumetric_flow_rate / 3600.0) / p.cross_sectional_area AS velocity_m_per_s
LIMIT 50

If the question cannot be answered using the schema, return:
{
  "use_kg": false,
  "query": "",
  "params": {},
  "rationale": "The question cannot be answered from the provided graph schema."
}
""".strip()

    compact_schema = compact_neo4j_schema(schema)

    planner_prompt = f"""
GRAPH SCHEMA:
{json.dumps(compact_schema, ensure_ascii=False, indent=2)}

USER QUESTION:
{user_question}

Return only the JSON object.
"""

    raw = chat_with_gemini(
        api_key=api_key,
        model=model,
        prompt=planner_prompt,
        system=planner_system,
        temperature=0.0,
        max_tokens=2048,
        response_mime_type="application/json"
    )

    fallback = None
    try:
        plan = extract_json(raw)
    except Exception:
        fallback = fallback_neo4j_plan(user_question)
        if fallback:
            return fallback, raw

        return {
            "use_kg": False,
            "query": "",
            "params": {},
            "rationale": "Planner returned invalid or truncated JSON."
        }, raw

    plan.setdefault("use_kg", False)
    plan.setdefault("query", "")
    plan.setdefault("params", {})
    plan.setdefault("rationale", "")
    return plan, raw


def plan_graphdb_query_with_llm(api_key, model, user_question, schema):
    planner_system = """
You are a GraphDB / SPARQL expert.

Generate ONLY read-only SPARQL.
Prefer SELECT queries.
Do not invent classes or predicates.
Use only names that appear in the schema.
Use LIMIT 50 by default unless the user asks for all rows.

Return exactly one JSON object and nothing else.

Return ONLY valid JSON in this format:
{
  "use_kg": true,
  "query": "SELECT ... WHERE { ... } LIMIT 10",
  "params": {},
  "rationale": "brief explanation"
}
""".strip()

    planner_prompt = f"""
GRAPH SCHEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

USER QUESTION:
{user_question}

Return ONLY JSON.
"""
    raw = chat_with_gemini(
        api_key=api_key,
        model=model,
        prompt=planner_prompt,
        system=planner_system,
        temperature=1.0,
        max_tokens=700,
        response_mime_type="application/json"
    )
    plan = extract_json(raw)
    plan.setdefault("use_kg", False)
    plan.setdefault("query", "")
    plan.setdefault("params", {})
    plan.setdefault("rationale", "")
    return plan, raw


def plan_arango_query_with_llm(api_key, model, user_question, schema):
    planner_system = """
You are an ArangoDB AQL expert.

Generate ONLY read-only AQL.
Do not invent collections or field names.
Use only names that appear in the schema.
Use LIMIT 50 by default unless the user asks for all rows.

Return ONLY valid JSON in this format:
{
  "use_kg": true,
  "query": "FOR d IN collection FILTER ... RETURN d LIMIT 10",
  "params": {},
  "rationale": "brief explanation"
}
""".strip()

    planner_prompt = f"""
GRAPH SCHEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

USER QUESTION:
{user_question}

Return ONLY JSON.
"""
    raw = chat_with_gemini(
        api_key=api_key,
        model=model,
        prompt=planner_prompt,
        system=planner_system,
        temperature=1.0,
        max_tokens=700,
        response_mime_type="application/json"
    )
    plan = extract_json(raw)
    plan.setdefault("use_kg", False)
    plan.setdefault("query", "")
    plan.setdefault("params", {})
    plan.setdefault("rationale", "")
    return plan, raw


# ============================================================
# Executors
# ============================================================
def execute_neo4j_plan(driver, plan, database):
    q = (plan.get("query") or "").strip()
    params = plan.get("params") or {}

    if not q:
        return None, "No query generated."

    if not is_readonly_cypher(q):
        return None, "Blocked: generated Cypher is not read-only."

    try:
        rows = run_cypher(driver, q, params=params, database=database)
        return rows, None
    except Exception as e:
        return None, f"Cypher execution error: {e}"


def execute_graphdb_plan(graphdb_base_url, repository, plan):
    q = (plan.get("query") or "").strip()

    if not q:
        return None, "No query generated."

    if not is_readonly_sparql(q):
        return None, "Blocked: generated SPARQL is not read-only."

    try:
        res = run_sparql(graphdb_base_url, repository, q)
        if res["type"] in ["select", "ask"]:
            return res["rows"], None
        return [{"text_result": res.get("text", "")}], None
    except Exception as e:
        return None, f"SPARQL execution error: {e}"


def execute_arango_plan(arango_base_url, database, user, password, plan):
    q = (plan.get("query") or "").strip()
    params = plan.get("params") or {}

    if not q:
        return None, "No query generated."

    if not is_readonly_aql(q):
        return None, "Blocked: generated AQL is not read-only."

    try:
        rows, _ = run_aql_query(arango_base_url, database, user, password, q, bind_vars=params)
        return rows, None
    except Exception as e:
        return None, f"AQL execution error: {e}"


# ============================================================
# Final answer helper
# ============================================================
def answer_from_results(
    db_name,
    user_question,
    schema,
    plan,
    query_used,
    results,
    execution_error,
    api_key,
    model,
    system_prompt,
    temperature,
    max_tokens
):
    answer_prompt = f"""
You are answering a thesis-research question about {db_name}.

Your job:
Explain the query result in plain English so an engineer or IT reader can understand what the result means.

Rules:
- Start directly with the actual answer.
- Do not start with generic phrases like:
  "Based on the provided query results"
  "Here are the details"
  "The query returned"
- Use only the query results as factual source.
- Do not invent anything not present in the schema or results.
- If results are empty, say exactly: The query returned no rows.
- If there is an execution error, explain the error clearly.
- If the result contains numbers, explain what each number means.
- If the result contains IDs, explain what object each ID refers to.
- If the result contains null, None, empty, or placeholder values, say that this row is incomplete.
- If multiple rows are returned, use bullet points.
- Finish the answer completely.
- Do not stop mid-sentence.
- Keep the answer short but complete.

User question:
{user_question}

Generated query:
{query_used}

Query results:
{json.dumps(results, ensure_ascii=False, indent=2)}

Execution error:
{execution_error}
"""

    context_payload = {
        "database": db_name,
        "schema": schema,
        "plan": plan,
        "query": query_used,
        "results": results,
        "execution_error": execution_error
    }

    return chat_with_gemini(
        api_key=api_key,
        model=model,
        prompt=answer_prompt,
        system=system_prompt if system_prompt.strip() else None,
        context=context_payload,
        temperature=float(temperature),
        max_tokens=int(max_tokens) if max_tokens > 0 else None
    )


# ============================================================
# UI tabs
# ============================================================
tab1, tab2 = st.tabs([
    "1) LLM Assistant - Neo4j",
    "2) Test Connections"
])


# ============================================================
# TAB 1 - LLM Assistant - Neo4j
# ============================================================
with tab1:
    st.subheader("LLM Assistant - Neo4j")
    st.caption(f"Model: {model_name}")

    user_question = st.text_area(
        "How may I assist you?",
        height=120,
        value=""
    )

    col1, col2 = st.columns(2)
    with col1:
        show_schema = st.checkbox("Show Neo4j schema", value=False, key="neo4j_show_schema")
    with col2:
        show_debug = st.checkbox("Show debug", value=True, key="neo4j_show_debug")

    if st.button("Send"):
        try:
            driver = get_neo4j_driver(neo4j_uri, neo4j_user, neo4j_password)
            schema = get_neo4j_schema(driver, database=neo4j_db)

            if show_schema:
                st.markdown("### Schema")
                st.json(schema)

            plan, planner_raw = plan_neo4j_query_with_llm(
                api_key=gemini_api_key,
                model=model_name,
                user_question=user_question,
                schema=schema
            )

            results, execution_error = execute_neo4j_plan(driver, plan, neo4j_db)
            query_used = (plan.get("query") or "").strip()

            st.markdown("### LLM Response")

            if not query_used:
                st.error("No query generated. The planner failed before Neo4j execution.")
            elif execution_error:
                st.error(execution_error)
            else:
                resp = answer_from_results(
                    db_name="Neo4j",
                    user_question=user_question,
                    schema=schema,
                    plan=plan,
                    query_used=query_used,
                    results=results,
                    execution_error=execution_error,
                    api_key=gemini_api_key,
                    model=model_name,
                    system_prompt=shared_system_prompt,
                    temperature=shared_temp,
                    max_tokens=shared_max_tokens
                )
                st.write(resp)

            if show_debug:
                st.markdown("### Debug")
                st.markdown("**Planner raw output**")
                st.code(planner_raw)

                st.markdown("**Parsed plan**")
                st.json(plan)

                st.markdown("**Generated query**")
                st.code(query_used)

                if execution_error:
                    st.error(execution_error)

                if results is not None:
                    if results:
                        st.dataframe(pd.DataFrame(results).head(50), width="stretch")
                    else:
                        st.info("Query returned no rows.")

        except Exception as e:
            st.error(f"Neo4j LLM error: {e}")



# ============================================================
# TAB 2 - Connection tests
# ============================================================
with tab2:
    st.subheader("Connection checks")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Neo4j Aura**")
        if st.button("Test Neo4j"):
            try:
                driver = get_neo4j_driver(neo4j_uri, neo4j_user, neo4j_password)
                pong = run_cypher(driver, "RETURN 1 AS ok", database=neo4j_db)
                st.success(f"Neo4j OK → {pong}")
            except Exception as e:
                st.error(f"Neo4j failed: {e}")

    with col2:
        st.markdown("**Gemini API**")
        if st.button("Test Gemini"):
            try:
                t0 = time.time()
                reply = chat_with_gemini(
                    gemini_api_key,
                    model_name,
                    "Reply with pong.",
                    temperature=1.0,
                    max_tokens=8
                )
                dt = time.time() - t0
                st.success(f"Gemini OK in {dt:.2f}s → {reply.strip()}")
            except Exception as e:
                st.error(f"Gemini failed: {e}")
