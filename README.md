# **AI4Semantics-MCP-Server**

MCP server for the SEMIC Data Modelling Chatbot. It exposes a set of tools and resources that allow an AI agent to validate, analyse, and improve semantic data models against the [SEMIC Style Guide](https://semiceu.github.io/style-guide/1.0.0/index.html).

---

## Resources

Resources are static files and directories the agent can read and reference.

| Resource | Path | Description |
|---|---|---|
| SEMIC Style Guide (Excel) | `resources/semantic_conventions/style_guide/SEMIC_Style_Guide.xlsx` | Machine-readable conventions and rules of the SEMIC style guide |
| SEMIC Style Guide (Text) | `resources/semantic_conventions/style_guide/style_guide.txt` | Human-readable description of the SEMIC style guide |
| User Models | `resources/semantic_model/models/<user>/<session>/` | Uploaded data models — a folder is created per user and session |

---

## Tools

Tools are callable functions the agent can invoke to perform analysis and modifications on data models.

### 1. `get_resources`
Retrieves available resources and model files for a given user and session.

---

### 2. `index_search`
Searches the Azure AI Search index for relevant concepts from standard vocabularies (e.g. FOAF, Schema.org, DCAT). Used to find candidate standard concepts for reuse recommendations.

---

### 3. `metadata_checker`
Validates metadata completeness and terminology consistency of a semantic model against the SEMIC style guide general conventions.

Checks the following rules:
- **GC-R3** — Metadata completeness: every class, attribute and association must have a URI, label, definition, and usage note
- **GC-R4** — Consistent terminology style across the vocabulary
- **GC-R5** — Consistent definition elaboration across the vocabulary
- **GC-R7** — Deontic modality indicators must not be used as semantic or normative values

Supports two modes:
- **Full model check** — inspects all classes, attributes and associations in the model
- **Targeted check** — inspects only a specified subset of classes/associations using an LLM

Supported model formats: UML XMI JSON and JSON-LD (Turtle).

---

### 4. `planning_orchestrator`
Generates a step-by-step plan for answering a user's question using the available tools. Acts as a planner agent that produces a structured plan for a separate executor agent.

---

### 5. `semantic_model`
Reads and modifies the user's uploaded data model. Used to apply corrections or improvements suggested by other tools.

---

### 6. `reuse_check`
Checks whether classes in a data model properly reuse standard concepts from well-known vocabularies, according to the SEMIC style guide reuse conventions.

For each class, searches the index for relevant standard concepts and evaluates reuse quality:
- **Reuse as-is** — adopt original URI, label, and definition with no changes
- **Terminological adaptation** — create a subclass, indicate adaptation, do not change semantics
- **Semantic adaptation** — create a subclass with new label/definition, document the reuse chain
- Mandatory properties from the original should be included; optional ones only if relevant
- Reuse must be made explicit with notes, hyperlinks, or dereferenceable URIs

Supports full model and targeted (class-level) checks. If no relevant standard is found, suggests domain vocabularies to consider.

---

### 7. `style_guide_checks`
Produces a structured Markdown assessment report summarising the results of a full style guide evaluation, combining SHACL validation results, metadata quality checks, and reuse of standards checks into a single human-readable report.

---

### 8. `validator_check`
Validates a data model against the SEMIC style guide using the [ITB SHACL validator](https://www.itb.ec.europa.eu/shacl/semicstyleguide/api/validate) and generates LLM-based explanations for each violation.

- Extracts the Turtle serialization of the model
- Posts it to the ITB validation API
- For each SHACL violation, calls an LLM to produce a human-readable explanation and actionable resolution suggestion

Supported validation types: `owl`, `shacl`, `uml`.

---

## File Structure
```
ai4semantics_mcp_server/
│
├── server.py                          # MCP server entry point
│
├── resources/
│   ├── semantic_conventions/
│   │   └── style_guide/
│   │       ├── SEMIC_Style_Guide.xlsx # Machine-readable SEMIC style guide rules and conventions
│   │       └── style_guide.txt        # Human-readable description of the SEMIC style guide
│   └── semantic_model/
│       └── models/
│           └── <user>/<session>/      # Uploaded data models, organised by user and session
│
├── tools/
│   ├── get_resources/                 # Lists available resources and models
│   ├── index_search/                  # Azure AI Search integration for standard vocabulary lookup
│   ├── model_metadata_checks/         # Metadata completeness and terminology checks (GC-R3/4/5/7)
│   ├── planning_orchestrator/         # Plans and orchestrates multi-step tool execution
│   ├── semantic_model/                # Reads and modifies the user's data model
│   ├── semantic_reuse_of_existing_concepts_checks/  # Reuse of standard concepts checks
│   ├── style_guide_checks/            # Aggregates all checks into a Markdown assessment report
│   └── style_guide_validator/         # SHACL validation via ITB API with LLM explanations
│
├── config.py                          # Loads and exposes configuration to the application
├── config.yaml                        # Application configuration (paths, settings)
├── docker-compose.yml                 # Runs Qdrant vector store locally
├── .env                               # API keys and environment variables (not committed)
└── .env.sample                        # Sample environment variable file
```

---


## Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd ai4semantics_mcp_server
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.sample` to `.env` and fill in your values:
```bash
cp .env.sample .env
```

### 5. Set up the vector index (Qdrant)

The index search tool uses a Qdrant vector database to search for relevant concepts from standard vocabularies. We have pre-indexed a set of core vocabularies including **Core Vocabularies**, **Schema.org**, and **LOV** — these are not committed to the repository due to size. You can either request access to the pre-built index or build your own using your own documents.

**a) Place your documents** in the documents folder:
```bash
mkdir -p tools/index_search/load_documents/documents
```
Add your files to the `documents/` folder. These can be any vocabulary or ontology documents you want the agent to search over.

**b) Start the Qdrant database:**
```bash
docker-compose up -d
```
This starts Qdrant on `localhost:6333` with persistent storage in `./data/qdrant`.

**c) Run the indexing script:**
```bash
python tools/index_search/load_documents/load.py
```
This loads all documents, creates embeddings, and stores them in Qdrant.

**d) (Optional) Test the index:**
```bash
python tools/index_search/load_documents/retrieve.py
```
Use `retrieve_documents(query)` to quickly test the index and verify the top results are relevant.

### 6. Start the server
```bash
python -m server
```
