# Architecture — Neo4j Insurance GraphRAG

A reference implementation demonstrating multiple retrieval strategies over a Neo4j knowledge
graph for insurance underwriting.

---

## Overview

This system integrates four retrieval strategies against a single Neo4j knowledge graph:

- **GraphRAG** — two-phase retrieval: semantic vector search over document chunks, then structured
  graph traversal to assemble applicant, rule, and policy context into a structured dict that is
  passed directly to an LLM.
- **Text2Cypher** — natural language to Cypher generation; GPT-4o reads the question and the graph
  schema and writes the query directly. Best for entity lookups and structural queries where vector
  similarity adds no value.
- **Auto Routing** — a GPT-4o classifier selects the retrieval strategy at query time based on
  question type. The routing decision is transparent: `selected_strategy` and `router_reason` appear
  in every response.
- **Hybrid Retrieval** — GraphRAG and Text2Cypher run in parallel; GPT-4o synthesizes a single
  answer from both contexts, delivering semantic understanding and structural precision simultaneously.

All four strategies operate against the same graph model and return the same API response shape.

---

## Business Problem

Insurance underwriting decisions require:

- **Policy interpretation** — which rules govern the applicant's conditions for the product they are
  applying for
- **Risk evaluation** — structured facts about the applicant: age, conditions, lab results, controlled
  status
- **Explainable decisions** — every conclusion must be traceable to a specific source in the
  underwriting manual or the rule graph
- **Regulatory traceability** — a compliance reviewer must be able to reproduce exactly what evidence
  the system considered

Traditional vector-only RAG is insufficient for these requirements. A similarity search returns the
most relevant text chunks, but it cannot:

- Identify *which specific applicant* has *which specific condition*
- Determine *which rules* apply to *that condition* for *that policy*
- Produce citations that trace to a specific graph path rather than an approximate embedding
  neighbourhood

A graph traversal assembles that reasoning chain explicitly. Combined with semantic retrieval and
LLM-generated Cypher, the system can answer the full range of underwriting questions — from
conceptual ("how does controlled diabetes affect underwriting?") to structural ("which rules apply to
John Smith?").

---

## Solution Architecture

```text
User Question
      │
      ▼
Retrieval Strategy
(GraphRAG / Text2Cypher / Auto / Hybrid)
      │
      ▼
Neo4j Knowledge Graph
(Vector Index + Graph Traversal / Direct Cypher Execution)
      │
      ▼
GPT-4o Reasoning
(Structured context → decision, reasoning, citations)
      │
      ▼
API Response
(decision, reasoning, citations, generated_cypher, raw_query_results,
 selected_strategy, router_reason)
```

**Major components:**

- **FastAPI application** (`app/main.py`) — routes requests by mode, manages the driver lifecycle,
  handles automatic re-indexing on embedding provider switch
- **GraphRAG pipeline** (`app/graphrag_pipeline.py`, `app/graph_retriever.py`) — two-phase retrieval:
  HNSW vector search then structured graph traversal
- **Text2Cypher service** (`app/text2cypher_service.py`) — schema-grounded LLM Cypher generation
  and execution against the shared Neo4j driver
- **Retrieval router** (`app/retrieval_router.py`) — GPT-4o zero-shot classifier; routes to
  `openai_graph`, `text2cypher`, or `hybrid`
- **Embedding providers** (`app/embed.py`) — `MockEmbeddingProvider` and `OpenAIEmbeddingProvider`;
  automatic re-indexing on provider switch
- **LLM providers** (`app/mock_llm.py`, `app/openai_llm.py`) — identical interface; `MockLLM` for
  local validation without API cost, `OpenAILLM` for production
- **User interface layer** (`static/`) — single-page application that calls `POST /ask` and renders
  all pipeline stages: Phase 1 chunks, Phase 2 graph context, decision badge, citations, generated
  Cypher, raw query results, and router decisions

---

## Knowledge Graph Model

### Node labels

| Label | Purpose | Key properties |
| --- | --- | --- |
| `Applicant` | Person applying for coverage | `name`, `age` |
| `Policy` | Insurance product | `name`, `type`, `class_name` |
| `RiskFactor` | Medical or lifestyle condition | `name`, `category`, `controlled` |
| `LabResult` | Raw lab measurement | `test_name`, `value`, `unit` |
| `UnderwritingRule` | Decision rule from the underwriting manual | `title`, `text`, `decision` |
| `DocumentChunk` | Source text passage with vector embedding | `source`, `text`, `embedding` |

### Relationship types

```text
(Applicant)         -[:APPLIES_FOR]→    (Policy)
(Applicant)         -[:HAS_CONDITION]→  (RiskFactor)
(Applicant)         -[:HAS_LAB_RESULT]→ (LabResult)
(Policy)            -[:HAS_RULE]→       (UnderwritingRule)
(RiskFactor)        -[:EVALUATED_BY]→   (UnderwritingRule)
(UnderwritingRule)  -[:SUPPORTED_BY]→   (DocumentChunk)
```

### Why a graph instead of a relational model

An underwriting decision requires traversing a chain:

> Applicant → what conditions do they have → which rules govern those conditions → what policy are they applying for → what does the rule say to do

In SQL, this is 4+ JOINs across normalised tables. In Cypher:

```cypher
MATCH (a:Applicant)-[:HAS_CONDITION]->(rf)-[:EVALUATED_BY]->(r:UnderwritingRule)
MATCH (p:Policy)-[:HAS_RULE]->(r)
RETURN a.name, rf.name, r.decision, p.name
```

Adding a new relationship type (e.g., connecting a lab result directly to a rule) requires a new
relationship — no schema migration, no `ALTER TABLE`.

### Why embeddings live on DocumentChunk, not UnderwritingRule

UnderwritingRule text is short and precise:

> "Controlled Type 2 Diabetes with A1C below 7.0 may be referred for underwriting review."

Embedding models perform best on paragraph-length text with surrounding semantic context. The
`DocumentChunk` holds the manual passage from which the rule was extracted — richer vocabulary and
better embedding quality.

**Design principle:** embed what is verbose and semantically rich; traverse to what is precise and
structured.

---

## Retrieval Strategies

| Capability | GraphRAG | Text2Cypher | Hybrid |
| --- | --- | --- | --- |
| Semantic retrieval | ✓ | Limited | ✓ |
| Entity lookup | Moderate | Excellent | Excellent |
| Explainability | High | High | Very High |
| Aggregations | Limited | Excellent | Excellent |
| Cost | Medium | Medium | Highest |
| Complexity | Medium | Medium | High |

**GraphRAG** is the right choice when the question requires semantic understanding — reasoning about
what a rule means in context, or retrieving documentation by concept proximity. The vector index finds
relevant document chunks; graph traversal assembles the structured context around them.

**Text2Cypher** is the right choice when the question targets a specific entity — names, IDs,
enumerations, counts. There is no useful semantic proximity for "which rules apply to John Smith" —
the correct answer requires exact graph traversal, not approximate embedding matching.

**Hybrid** is the right choice when the question simultaneously requires both: semantic context *and*
structured entity data. Running both retrievers and merging the results gives GPT-4o more complete
context than either retriever alone.

**Auto routing** removes the burden of mode selection from the caller. GPT-4o classifies the question
and selects the strategy at request time. The decision is transparent: `selected_strategy` and
`router_reason` are returned with every response.

---

## GraphRAG Architecture

GraphRAG retrieval is two-phase: HNSW vector similarity search locates the most relevant document
chunks, then graph traversal expands outward from those chunks to assemble the structured context —
rules, risk factors, applicants, policies.

```text
User question
    │
    ▼  provider.embed(question)  →  float[1536]
HNSW Vector Index
(db.index.vector.queryNodes)
    │
    ▼  Top-k DocumentChunk nodes
Graph Traversal (UNWIND + OPTIONAL MATCH)
    │
    ▼  {rules, risk_factors, policies, applicants}
GPT-4o  llm.generate_answer(question, context)
    │
    ▼
{decision, reasoning, supporting_rules, risk_factors, citations}
```

### Phase 1 — Vector search

```cypher
CALL db.index.vector.queryNodes($index, $top_k, $vector)
YIELD node, score
RETURN node.id AS id, node.source AS source, node.text AS text, score
```

Returns the `DocumentChunk` nodes whose embeddings are closest to the query embedding. The Neo4j
vector index uses HNSW (Hierarchical Navigable Small World), which organises vectors into a
multi-layer approximate nearest-neighbour graph — O(log n) search complexity rather than O(n) linear
scan. The same algorithm underlies Pinecone, Weaviate, and pgvector.

In Learning Mode, similarity scores cluster near 0.5 because SHA-256 mock embeddings are not
semantic. In production with `text-embedding-3-small`, relevant chunks score near 1.0 and irrelevant
ones near 0.0. The retrieval logic is identical; only the embedding quality changes.

### Phase 2 — Graph traversal

```cypher
UNWIND $chunk_ids AS chunk_id
MATCH (d:DocumentChunk {id: chunk_id})<-[:SUPPORTED_BY]-(r:UnderwritingRule)
OPTIONAL MATCH (rf:RiskFactor)-[:EVALUATED_BY]->(r)
OPTIONAL MATCH (p:Policy)-[:HAS_RULE]->(r)
OPTIONAL MATCH (a_cond:Applicant)-[:HAS_CONDITION]->(rf)
OPTIONAL MATCH (a_pol:Applicant)-[:APPLIES_FOR]->(p)
RETURN
    chunk_id,
    r.id, r.title, r.text, r.decision,
    collect(DISTINCT rf) AS risk_factors,
    collect(DISTINCT p)  AS policies,
    collect(DISTINCT a_cond) + collect(DISTINCT a_pol) AS applicants
```

Key design points:

- `UNWIND` batches all chunk IDs into one round-trip, not one query per chunk
- `OPTIONAL MATCH` means nodes with no connections still return — no silent filtering
- Deduplication happens in Python after the query, not in Cypher — easier to test and debug

The assembled context dict — not raw text — is passed to GPT-4o. The LLM receives structured
entities and relationships, which is the basis for traceable citations.

---

## Text2Cypher Architecture

Text2Cypher replaces the two-phase GraphRAG pipeline with a single LLM-driven Cypher generation
step. No vector index is queried; GPT-4o reads the question and the graph schema and writes the
Cypher directly.

```text
User question
    │
    ▼  System prompt: question + schema description
GPT-4o
    │
    ▼  Generated Cypher (string)
Neo4j driver.run(cypher)
    │
    ▼  Raw records (list of dicts)
    │
    ▼  GPT-4o: summarise records into plain-language answer
{reasoning, answer}
```

### Schema grounding

The system prompt passed to GPT-4o includes:

- All node labels and their key properties
- All relationship types and their directionality
- The exact property names to reference in `WHERE`, `RETURN`, and `MATCH` clauses
- 2–3 example question/Cypher pairs (few-shot) to anchor the output format

Without schema grounding, the LLM generates labels and property names that do not exist in the
graph. With it, GPT-4o consistently produces valid, runnable Cypher for the question types the
schema supports.

### Query execution

The generated Cypher string is extracted from the LLM response and executed against Neo4j via the
shared driver. Raw records are returned directly and surfaced to the caller in `raw_query_results`.
The `generated_cypher` field in the API response contains the exact query that was executed — fully
auditable.

### Production guardrails

| Concern | Mitigation |
| --- | --- |
| Hallucinated labels or properties | Schema grounding + few-shot examples reduce generation errors |
| Write operations in generated query | Application-level validation rejects write keywords (`CREATE`, `MERGE`, `SET`, `DELETE`, `DETACH`, `REMOVE`, `DROP`, `FOREACH`) before query execution |
| Unbounded result sets | Append `LIMIT 25` when the generated query does not include a LIMIT clause; additionally cap returned records to 50 rows in application code before synthesis |
| Syntax errors | Wrap execution in try/except; return a graceful error rather than a 500 |
| Silent wrong answers | Log generated Cypher alongside results; expose both in the API response for auditability |
| Schema drift | Any change to node labels or property names requires updating the schema prompt |

### Cypher observability

`generated_cypher` is returned in the API response so every query is inspectable. In production, log
it alongside the question, the record count, and the latency. This makes it possible to audit the
system's behaviour and detect prompt-quality degradation over time.

---

## Auto Router Architecture

Auto Mode adds a classification step before retrieval. The router reads the question and selects the
strategy best suited to answer it — the caller does not need to know which mode to use.

```text
POST /ask  (mode=auto)
    │
    ▼  RetrievalRouter.classify(question)
       System prompt: question + strategy descriptions
       GPT-4o (zero-shot classification)
       Returns: {selected_strategy, router_reason}
    │
    ├── "openai_graph" ──► GraphRAGPipeline.run()
    │                        (vector search + graph traversal)
    │
    ├── "text2cypher" ───► Text2CypherService.run()
    │                        (LLM → Cypher → Neo4j records)
    │
    └── "hybrid" ────────► both in parallel → GPT-4o synthesis
```

### Router design

The router is a stateless classification function — it does not perform retrieval and has no memory
of previous calls. It adds one GPT-4o call to the request latency.

The classification prompt describes each strategy:

- `openai_graph` — best for semantic or contextual questions
- `text2cypher` — best for entity lookups and structural queries
- `hybrid` — best for questions that require both semantic understanding and structured facts

GPT-4o returns a JSON object with `selected_strategy` and `router_reason`. Both are surfaced in the
API response under the same field names.

### Router observability

`selected_strategy` and `router_reason` appear in the API response and the browser UI, making the
routing decision fully transparent. If the router selects the wrong strategy, the reason field shows
why, and the classification prompt can be adjusted accordingly.

In production, log `selected_strategy` per request. Over a corpus of real queries, the distribution
of strategies reveals whether the router is calibrated correctly — a workload dominated by structural
queries should route heavily to `text2cypher`, not `openai_graph`.

---

## Hybrid Retrieval Architecture

Hybrid mode runs both the GraphRAG pipeline and the Text2Cypher pipeline for the same question and
merges their output before the final LLM synthesis call.

```text
Question
    │
    ├── GraphRAGPipeline.run() ─────────────────────────────────────┐
    │       vector search + graph traversal                          │
    │       → matched_chunks, graph_context, citations               │
    │                                                                 ▼
    └── Text2CypherService.run() ─────────────────────────► Merge contexts
            LLM → Cypher → Neo4j records                            │
            → generated_cypher, raw_query_results                   ▼
                                                           GPT-4o synthesis
                                                            (single answer
                                                             from both contexts)
```

The response includes all fields from both retrievers simultaneously. The UI renders both GraphRAG
result sections (Phase 1 / Phase 2 / Decision / Citations) and Text2Cypher result sections (Generated
Cypher / Raw Query Results).

### Why hybrid can outperform pure GraphRAG

GraphRAG Phase 1 finds semantically similar chunks. For questions that target a specific named entity,
the semantic signal may be weak — the nearest chunks describe the concept in general, not the specific
applicant. Text2Cypher fetches the exact entity records directly. Hybrid gives GPT-4o both the
conceptual context and the specific structured facts.

### Why hybrid can outperform pure Text2Cypher

Text2Cypher returns raw records without semantic context. If the question requires reasoning about
*what a rule means* rather than *which records match*, the raw records alone may not be sufficient.
GraphRAG provides the manual text and rule context that turns raw records into an interpretable
reasoning chain.

---

## Embedding Providers

Both providers implement the same interface:

```python
provider.embed(text: str) -> list[float]   # returns float[1536]
provider.model_name                         # "mock" | "text-embedding-3-small"
```

`GraphRAGPipeline.for_mode(driver, mode)` wires the correct provider for each mode —
`MockEmbeddingProvider` for Learning Mode, `OpenAIEmbeddingProvider` for OpenAI Mode. The pipeline
never checks environment variables at query time; the mode parameter is explicit.

**Automatic re-indexing on provider switch:**
The active model name is stored on every `DocumentChunk` node at seed time (`embedding_model`
property). Before each query, `_auto_reseed_if_needed()` reads the stored name and compares it to the
active provider's `model_name`. If they differ, `reindex_embeddings()` re-embeds all chunks using the
correct provider before the query runs — no manual intervention required.

`GraphRetriever._check_embedding_compatibility()` performs a final check and returns a warning string
if the models still differ after the auto-reindex attempt (e.g., because the reindex itself failed
due to an API error). The `compatibility_warning` field in the API response carries this warning to
the client; the UI displays it as an amber banner.

---

## LLM Providers

Both providers implement the same interface:

```python
llm.generate_answer(question: str, context: dict) -> dict
# returns {decision, reasoning, supporting_rules, risk_factors, citations}
```

`GraphRAGPipeline.for_mode(driver, mode)` returns the pipeline wired with the correct LLM: `MockLLM`
for Learning Mode, `OpenAILLM` for OpenAI Mode. The pipeline, API response shape, and citation format
are identical in both modes.

`OpenAILLM` uses `response_format={"type": "json_object"}` (JSON mode) so the response is always
valid JSON. The system prompt instructs the model to base its answer only on the retrieved context
and to use `REQUIRE_ADDITIONAL_REVIEW` when context is insufficient.

### MockLLM

MockLLM provides deterministic local validation of the retrieval pipeline without requiring
external API calls. Its purpose is to verify:

- retrieval quality
- graph traversal
- context assembly
- citation generation

without incurring model cost.

For production-style reasoning, `OpenAILLM` uses GPT-4o and structured JSON responses while
preserving the same API contract.

---

## User Interface Layer

The UI (`static/index.html`, `styles.css`, `app.js`) is a thin visualisation layer over the API.
It calls `POST /ask` and renders all pipeline stages — Phase 1 chunks, Phase 2 graph context,
decision badge, citations, Generated Cypher card, Raw Query Results card, and the router reason
callout for Auto Mode.

Design constraints:

- No framework dependencies (no React, no build tools, no npm) — zero setup friction
- No CDN dependencies — fully functional offline once the server is running
- No WebSockets — a single fetch per question is sufficient for this access pattern
- `StaticFiles` mount at `/static`, `FileResponse` at `/` — two lines in FastAPI
- `aiofiles` is the only additional dependency; required by FastAPI's `StaticFiles`

The API (`/ask`, `/health`) is the primary surface. The UI is optional scaffolding on top.

---

## API Layer

### Lifespan — one driver, multiple pipelines, one lock

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    driver = get_driver()                              # opened once at startup
    app.state.driver = driver
    app.state.reseed_lock = asyncio.Lock()             # prevents concurrent re-indexes
    app.state.pipelines = {
        "demo":   GraphRAGPipeline.for_mode(driver, "demo"),
        "openai": GraphRAGPipeline.for_mode(driver, "openai"),  # only if OPENAI_API_KEY set
    }
    if OPENAI_API_KEY:
        app.state.t2c_service      = Text2CypherService(driver)
        app.state.auto_router      = RetrievalRouter()
        app.state.auto_synthesizer = HybridSynthesizer()
    yield
    driver.close()                                     # closed cleanly at shutdown
```

The Neo4j driver maintains an internal connection pool. Opening it once and sharing across requests
is correct; creating a new driver per request would exhaust connections under load. When
`OPENAI_API_KEY` is set, `Text2CypherService`, `RetrievalRouter`, and `HybridSynthesizer` are also
initialized at startup and stored in `app.state`.

The `asyncio.Lock` serialises embedding re-index operations: if two requests arrive simultaneously
after a mode switch, only the first acquires the lock and re-indexes; the second re-checks inside the
lock and skips the re-index because it is already complete.

### Error handling

| Exception | HTTP code | Meaning for caller |
| --- | --- | --- |
| Blank question | 400 | Fix the request |
| `ServiceUnavailable` / `AuthError` | 503 | Backend down — retry later |
| Unexpected `Exception` | 500 | Server error — not caller's fault |

### Response shape

```json
{
  "question": "...",
  "mode": "auto",
  "decision": "REFER_FOR_REVIEW",
  "reasoning": ["...", "..."],
  "supporting_rules": [{"id": "...", "title": "...", "decision": "..."}],
  "risk_factors": [{"name": "...", "category": "..."}],
  "citations": [
    {"type": "DocumentChunk",    "source": "...", "relevance_score": 0.506},
    {"type": "UnderwritingRule", "title": "...",  "decision": "..."}
  ],
  "retrieval_summary": {
    "matched_chunks": 3,
    "rules": 3,
    "risk_factors": 3,
    "policies": 1,
    "applicants": 1
  },
  "embedding_provider": "text-embedding-3-small",
  "llm_provider": "OpenAILLM",
  "compatibility_warning": null,
  "reindexed": false,
  "matched_chunks": [{"id": "chunk_001", "source": "Underwriting Manual v3.2", "text": "...", "score": 0.93}],
  "graph_context": {"rules": [], "risk_factors": [], "policies": [], "applicants": []},
  "generated_cypher": "MATCH (a:Applicant {name: 'John Smith'})-[:HAS_CONDITION]->(rf) RETURN rf.name",
  "raw_query_results": [{"rf.name": "Type 2 Diabetes"}, {"rf.name": "Controlled A1C"}],
  "selected_strategy": "hybrid",
  "router_reason": "The question requires both specific entity data and semantic rule interpretation.",
  "retrieval_strategy": null
}
```

Field notes:

- `retrieval_summary` — zero values signal retrieval failure without inspecting the full response
- `reindexed: true` — set on the first request after a provider switch; signals embeddings were re-indexed
- `compatibility_warning` — non-null only when auto-reindex was attempted but failed
- `matched_chunks` / `graph_context` — populated for GraphRAG and Hybrid responses; empty list / empty object for Text2Cypher-only responses
- `generated_cypher` / `raw_query_results` — present whenever Text2Cypher retrieval participates in the response (Text2Cypher Mode, Auto Mode with `text2cypher` strategy, or Auto Mode with `hybrid` strategy); null or empty otherwise
- `selected_strategy` / `router_reason` — present for `auto` mode only; null for all other modes
- `retrieval_strategy` — `null` for Learning Mode and OpenAI Mode; set for Text2Cypher Mode (`"Text2Cypher"`) and all Auto Mode paths (`"openai_graph"`, `"text2cypher"`, or `"Hybrid"` depending on the router decision)

---

## Mode Comparison

| Component | Learning Mode | OpenAI Mode | Text2Cypher Mode | Auto Mode |
| --- | --- | --- | --- | --- |
| Embeddings | SHA-256 hash (not semantic) | `text-embedding-3-small` | None | `text-embedding-3-small` (if routed to graph) |
| LLM | Deterministic Python logic | `gpt-4o` | `gpt-4o` | `gpt-4o` (router + answer) |
| Retrieval | Vector search + graph | Vector search + graph | NL → Cypher → graph records | Router-selected: graph / cypher / hybrid |
| Graph traversal | Yes | Yes | No (direct Cypher execution) | Depends on selected strategy |
| API key needed | No | Yes | Yes | Yes |
| Cost | Zero | OpenAI API charges apply | OpenAI API charges apply | OpenAI API charges apply |

The graph schema, traversal query, context assembly, and API contract are unchanged across modes.
Transitioning to production is a substitution and hardening exercise — the graph model and retrieval
architecture are already production-equivalent.

---

## Production Considerations

### Authentication

The API is currently unauthenticated. For production deployment, add authentication middleware before
network exposure — OAuth 2.0 / JWT validation at the FastAPI layer, or a reverse proxy (API Gateway,
Azure APIM, NGINX) handling token verification upstream.

### Observability

Log the following fields per request for operational visibility:

- `mode`, `selected_strategy` — routing distribution across strategies
- `embedding_provider`, `llm_provider` — model tracking across deployments
- `retrieval_summary.matched_chunks`, `rules` — retrieval quality signals
- `generated_cypher` — Text2Cypher auditability
- Request latency broken down by phase (embed / search / traverse / LLM)

LangSmith or equivalent LLM tracing integrations can capture prompt/response pairs, token counts, and
latency per LLM call — useful for detecting prompt degradation over time.

### Evaluation and golden datasets

Retrieval quality is not detectable from latency alone. Establish a golden evaluation dataset of
questions with known correct answers and run it periodically to detect:

- Router miscalibration — wrong strategy selected for a given question type
- Embedding drift — a model upgrade changes similarity rankings
- Cypher generation degradation — generated queries produce wrong or empty results

### Router evaluation

Monitor `selected_strategy` distribution over production traffic. A workload dominated by structural
queries should route heavily to `text2cypher`. If `openai_graph` dominates, the classification prompt
may need adjustment. Log `router_reason` alongside `selected_strategy` to identify systematic
misclassifications.

### Prompt versioning

The schema grounding prompt for Text2Cypher and the strategy description prompt for the router are
load-bearing. Version them alongside application code — a prompt change is a code change. Any update
to node labels, property names, or relationship types requires a corresponding update to the schema
prompt.

### Cost controls

Auto Mode and Hybrid Mode issue multiple GPT-4o calls per request (router classification, retrieval,
and synthesis for Hybrid). In production:

- Cache router decisions for identical or near-identical questions
- Gate Hybrid Mode behind explicit user intent rather than automatic routing for cost-sensitive
  workloads
- Monitor token consumption per request; set per-session spending limits at the application layer

### Neo4j Aura deployment

For production, replace the local Docker Compose instance with Neo4j Aura (managed service). Update
`NEO4J_URI` in `.env` to the Aura connection string. The application code requires no changes — the
driver factory reads from environment variables at startup.

### Graph schema evolution

Any change to the graph schema — new node labels, relationship types, or properties — requires
updating:

- `app/seed.py` — data population
- `app/graph_retriever.py` — GraphRAG traversal query
- `app/text2cypher_service.py` — schema grounding prompt
- `CYPHER_QUERIES.md` — validation queries

---

## Known Limitations

1. **Learning Mode embeddings are not semantic.** SHA-256 hash embeddings produce similarity scores
   that cluster near 0.5 regardless of question relevance. The pipeline retrieves chunks, but not
   necessarily the most relevant ones. This is intentional — it allows the full retrieval pipeline to
   be exercised and verified without an OpenAI API key.

2. **MockLLM is deterministic by design.** It validates retrieval quality, graph traversal, context
   assembly, and citation generation without requiring external API calls. It is not intended to
   replicate the reasoning breadth, contextual understanding, or adaptability of GPT-4o. Learning
   Mode therefore validates the retrieval architecture and response contract, while OpenAI Mode
   provides production-style reasoning over the same graph context.

3. **Single-applicant seed data.** The seed data models one applicant (John Smith, age 48) with four
   underwriting rules. Multi-applicant retrieval (e.g., "which of our applicants need additional
   review?") is valid in the graph model but not exercised by the current seed data.

4. **No authentication.** The API is unauthenticated. See Production Considerations above.
