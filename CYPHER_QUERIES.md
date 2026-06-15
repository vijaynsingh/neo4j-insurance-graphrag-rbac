# Cypher Validation Queries

Run these in Neo4j Browser at http://localhost:7474 after seeding the database.
They are ordered from simplest to most useful for understanding the graph model.

---

## Query 1 — Count All Nodes

Verify the seed ran correctly. You should see 6 node labels.

```cypher
MATCH (n)
RETURN labels(n) AS label, count(n) AS count
ORDER BY count DESC
```

**Expected output:**

Because each applicant carries two labels (`:Applicant` plus its tier), `labels(n)`
groups applicants by their full label-set — so applicants appear as three separate
rows of 2, not one row of 6:

| label | count |
|---|---|
| `["UnderwritingRule"]` | 12 |
| `["DocumentChunk"]` | 12 |
| `["RiskFactor"]` | 11 |
| `["LabResult"]` | 7 |
| `["Policy"]` | 3 |
| `["Applicant","Standard"]` | 2 |
| `["Applicant","Restricted"]` | 2 |
| `["Applicant","Confidential"]` | 2 |

That is 8 rows totalling 6 applicants + 3 policies + 11 risk factors + 7 lab results +
12 rules + 12 chunks. The tier split in the applicant rows is itself a quick visual
confirmation that the RBAC tier labels are applied. To count applicants as a single
group regardless of tier, query `MATCH (a:Applicant) RETURN count(a)` instead.

---

## Query 2 — Full Applicant Underwriting Context

This is the most important query. It answers:
**"What is the complete underwriting picture for John Smith?"**

Starting from one Applicant node, it traverses the entire graph in a single query.

```cypher
MATCH (a:Applicant {name: "John Smith"})
OPTIONAL MATCH (a)-[:APPLIES_FOR]->(p:Policy)
OPTIONAL MATCH (a)-[:HAS_CONDITION]->(rf:RiskFactor)
OPTIONAL MATCH (a)-[:HAS_LAB_RESULT]->(lab:LabResult)
RETURN
  a.name        AS applicant,
  a.age         AS age,
  p.name        AS policy,
  p.class_name  AS class,
  collect(DISTINCT rf.name)       AS risk_factors,
  collect(DISTINCT lab.test_name + ": " + toString(lab.value) + lab.unit) AS lab_results
```

**Why this matters for GraphRAG:**
This is the context the LLM will receive. One query replaces what would be multiple
JOINs or separate vector lookups. The graph makes it trivial.

---

## Query 3 — Policy Rules

Answers: **"What rules govern the Preferred Term Life policy?"**

```cypher
MATCH (p:Policy)-[:HAS_RULE]->(r:UnderwritingRule)
RETURN
  p.name       AS policy,
  r.title      AS rule_title,
  r.text       AS rule_text,
  r.decision   AS decision
ORDER BY r.id
```

**Expected:** 13 rows total — `policy_001` and `policy_002` each link to 5 rules and
`policy_003` to 3, across a range of decision outcomes. (To see the rules for a single
policy, add a filter such as `WHERE p.name = "Preferred Term Life"`.)

---

## Query 4 — Risk Factors and Their Governing Rules

Answers: **"Which underwriting rules apply to each of John's risk factors?"**

```cypher
MATCH (a:Applicant {name: "John Smith"})-[:HAS_CONDITION]->(rf:RiskFactor)-[:EVALUATED_BY]->(r:UnderwritingRule)
RETURN
  a.name      AS applicant,
  rf.name     AS risk_factor,
  rf.category AS category,
  r.title     AS governing_rule,
  r.decision  AS decision
ORDER BY rf.name
```

**Why this is powerful:**
This query walks three hops: Applicant → RiskFactor → UnderwritingRule.
In a relational DB you'd need two JOINs. In a vector store, you couldn't express
"which rule governs this specific risk factor for this specific applicant" at all.

---

## Query 5 — Source Document Chunks for Each Rule

Answers: **"Where in the underwriting manual does each rule come from?"**

```cypher
MATCH (r:UnderwritingRule)-[:SUPPORTED_BY]->(c:DocumentChunk)
RETURN
  r.title    AS rule,
  r.decision AS decision,
  c.source   AS manual_section,
  c.text     AS source_text
ORDER BY r.id
```

**Why DocumentChunk nodes exist:**
In the GraphRAG pipeline, these chunks carry vector embeddings.
GraphRetriever finds the most relevant chunk by semantic similarity,
then the graph traversal walks back to the rule and applicant context.
The chunk is the *semantic entry point*; the graph is the *structured context*.

---

## Query 6 — End-to-End: Applicant → Rule → Source Text

This is the full chain the GraphRAG retriever will walk.
One query returns everything an LLM needs to reason about this applicant.

```cypher
MATCH (a:Applicant {name: "John Smith"})
MATCH (a)-[:APPLIES_FOR]->(p:Policy)-[:HAS_RULE]->(r:UnderwritingRule)-[:SUPPORTED_BY]->(c:DocumentChunk)
MATCH (a)-[:HAS_CONDITION]->(rf:RiskFactor)-[:EVALUATED_BY]->(r)
OPTIONAL MATCH (a)-[:HAS_LAB_RESULT]->(lab:LabResult)
RETURN
  a.name                   AS applicant,
  a.age                    AS age,
  rf.name                  AS risk_factor,
  r.title                  AS rule,
  r.decision               AS decision,
  lab.test_name + ": " + toString(lab.value) + lab.unit AS lab_result,
  c.source                 AS source,
  c.text                   AS supporting_text
ORDER BY r.id
```

This mirrors the structured context assembled by the GraphRAG traversal.
It returns structured entities and relationships — not just raw text chunks — which is
what makes graph retrieval richer than pure vector search.

---

## Why This Graph Model Is Useful for GraphRAG

**The core problem with flat vector search for underwriting:**

Imagine embedding all underwriting rules as text chunks and doing similarity search.
A query like "Is John Smith eligible for Preferred class?" might retrieve the right
rule chunks — but the LLM has no way to know:
- That John specifically has Type 2 Diabetes (linked via HAS_CONDITION)
- That his A1C is 6.8% (linked via HAS_LAB_RESULT)
- That he is 48 years old (stored on the Applicant node)
- That rule_004 applies to him because he is over 45 AND has diabetes

**What the graph adds:**
The graph encodes *who this rule applies to and why*. When GraphRetriever
finds `chunk_002` (the A1C rule text) as semantically similar to the question,
the graph traversal immediately walks:

```
chunk_002 ← SUPPORTED_BY ← rule_002 ← EVALUATED_BY ← rf_001 (Type 2 Diabetes)
                                     ← HAS_CONDITION ← applicant_001 (John Smith, age 48)
                                     ← HAS_LAB_RESULT ← lab_001 (A1C: 6.8%)
```

The LLM receives not just the chunk text but the entire structured context —
applicant profile, lab result, risk factors, and the rule decision — all from
a single vector hit + graph traversal.

**GraphRAG summary:**
> "The DocumentChunk nodes are semantic entry points into the graph. Vector search
> finds the relevant chunk; the graph provides the structured reasoning context
> around it. Together they enable multi-hop, explainable answers that neither
> approach can produce alone."

---

## Vector Index and Embedding Validation

### Query 7 — Verify Vector Index Exists

```cypher
SHOW VECTOR INDEXES
YIELD name, state, labelsOrTypes, properties
```

**Expected:**

| name | state | labelsOrTypes | properties |
|---|---|---|---|
| document_chunk_embeddings | ONLINE | [DocumentChunk] | [embedding] |

If `state` is `POPULATING`, wait a few seconds and re-run — Neo4j is still indexing.

---

### Query 8 — Verify Embeddings Are Stored on DocumentChunk Nodes

```cypher
MATCH (n:DocumentChunk)
RETURN
  n.id                   AS id,
  n.source               AS source,
  size(n.embedding)      AS embedding_dimensions,
  n.embedding[0]         AS first_value,
  n.embedding[1535]      AS last_value
ORDER BY n.id
```

**Expected:** 12 rows, each with `embedding_dimensions = 1536`.

`first_value` and `last_value` should be small floats (e.g., `-0.024`, `0.019`).
These are normalized — the sum of squares of all 1536 values equals 1.0.

---

### Query 9 — Verify Normalization (Unit Length Check)

The embedding must be a unit vector for cosine similarity to work correctly.
This query computes the magnitude — it should be exactly 1.0 for every chunk.

```cypher
MATCH (n:DocumentChunk)
WITH n, reduce(acc = 0.0, v IN n.embedding | acc + v * v) AS sumSquares
RETURN n.id AS id, sqrt(sumSquares) AS magnitude
ORDER BY n.id
```

**Expected:** `magnitude ≈ 1.0` for all rows (may show as `1.0000001` due to float precision).

---

### Query 10 — Similarity Search Using a Stored Embedding as Query

Neo4j Browser can't easily accept raw float arrays as parameters. This workaround uses
an existing node's own embedding as the query vector — which should return itself with
score = 1.0, confirming the index is working.

```cypher
MATCH (seed:DocumentChunk {id: 'chunk_002'})
CALL db.index.vector.queryNodes('document_chunk_embeddings', 4, seed.embedding)
YIELD node, score
RETURN
  node.id     AS id,
  node.source AS source,
  score
ORDER BY score DESC
```

**Expected:**
- `chunk_002` returns first with score = `1.0` (exact match with itself)
- The other chunks return with scores < 1.0

**NOTE:** With mock embeddings (Learning Mode), the relative scores of the other chunks have no
semantic meaning — they reflect hash proximity. When using OpenAI Mode with
text-embedding-3-small, top results reflect semantic similarity.

---

### Run the Full Embedding Validation from Python

The Python script does all of the above plus runs 3 semantic queries:

```bash
python -m app.vector_index
```

---

## GraphRAG Traversal Validation

These queries show each hop of the two-phase retrieval separately, then combined.
Run them in Neo4j Browser to validate what `GraphRetriever` does under the hood.

---

### Query A — Vector Hit → Rule (Phase 2, Hop 1)

From each DocumentChunk, walk backwards through SUPPORTED_BY to find the UnderwritingRule
it came from. This is the first graph hop after the vector search lands on a chunk.

```cypher
MATCH (d:DocumentChunk)<-[:SUPPORTED_BY]-(r:UnderwritingRule)
RETURN
  d.id     AS chunk_id,
  d.source AS manual_section,
  r.title  AS rule_title,
  r.decision AS decision
ORDER BY d.id
```

**Expected:** 12 rows — each of the 12 document chunks maps to the underwriting rule it supports.

**Why this hop matters:** The vector search finds chunks by semantic similarity to the
question. But chunks alone are just text. Walking to the UnderwritingRule gives us the
structured decision logic (`REFER_FOR_REVIEW`, `REQUIRE_ADDITIONAL_REVIEW`, etc.) that
the LLM can reason over precisely.

---

### Query B — Rule → Risk Factor (Phase 2, Hop 2)

From each rule, find the risk factors it evaluates. This tells us *which clinical
conditions are governed by which rules*.

```cypher
MATCH (rf:RiskFactor)-[:EVALUATED_BY]->(r:UnderwritingRule)
RETURN
  rf.name     AS risk_factor,
  rf.category AS category,
  r.title     AS governing_rule,
  r.decision  AS decision
ORDER BY rf.name
```

**Expected:** 14 rows (some risk factors are evaluated by more than one rule — e.g.
Type 2 Diabetes is evaluated by both the controlled-diabetes and the age-and-diabetes rules).

**Why this hop matters:** Pure vector search on "diabetes underwriting rule" retrieves
text chunks. The graph hop tells us specifically which applicant conditions are governed
by each rule — a fact that is encoded in the relationship structure, not in any text chunk.

---

### Query C — Full GraphRAG Traversal (All Hops Combined)

This is the complete multi-hop path the GraphRetriever assembles: Applicant → Policy →
Rule → DocumentChunk. Every node in the underwriting decision chain in one query.

```cypher
MATCH (a:Applicant)-[:APPLIES_FOR]->(p:Policy)-[:HAS_RULE]->(r:UnderwritingRule)-[:SUPPORTED_BY]->(d:DocumentChunk)
RETURN
  a.name     AS applicant,
  a.age      AS age,
  p.name     AS policy,
  r.title    AS rule,
  r.decision AS decision,
  d.source   AS source_text_from
ORDER BY r.id
```

**Expected:** 26 rows. The query has no applicant filter, so it spans all 6 applicants
across their policies and rules: `(2 applicants × 5 rules)` for policy_001 +
`(2 × 5)` for policy_002 + `(2 × 3)` for policy_003 = 26. To scope to one applicant,
add `WHERE a.name = "John Smith"`.

**Why this matters for GraphRAG:**
This single Cypher query replaces what would require:
- In SQL: 4 JOINs across 5 tables
- In pure vector RAG: multiple separate lookups with no guaranteed connection between results

The graph encodes the *reasoning chain* as structure. An LLM receiving this output
can explain: "Rule X applies because this applicant has condition Y, which is governed
by that rule under this policy."

---

### Query D — Full Context Assembly (GraphRetriever._traverse_from_chunks)

This is the exact traversal `GraphRetriever._traverse_from_chunks()` runs internally.
Replace `$chunk_ids` with actual IDs to test manually.

```cypher
UNWIND ['chunk_001', 'chunk_002'] AS chunk_id
MATCH (d:DocumentChunk {id: chunk_id})<-[:SUPPORTED_BY]-(r:UnderwritingRule)
OPTIONAL MATCH (rf:RiskFactor)-[:EVALUATED_BY]->(r)
OPTIONAL MATCH (p:Policy)-[:HAS_RULE]->(r)
OPTIONAL MATCH (a_cond:Applicant)-[:HAS_CONDITION]->(rf)
OPTIONAL MATCH (a_pol:Applicant)-[:APPLIES_FOR]->(p)
RETURN
    chunk_id                                                                 AS source_chunk,
    r.title                                                                  AS rule,
    r.decision                                                               AS decision,
    [x IN collect(DISTINCT rf) WHERE x IS NOT NULL | x.name]                AS risk_factors,
    [x IN collect(DISTINCT p)  WHERE x IS NOT NULL | x.name]                AS policies,
    [x IN (collect(DISTINCT a_cond) + collect(DISTINCT a_pol)) WHERE x IS NOT NULL | x.name] AS applicants
```

This is the internal traversal that assembles structured context from matched chunks for GraphRAG retrieval and context assembly.

---

## Text2Cypher Validation Examples

These queries represent the type of Cypher that `Text2CypherService` generates from natural language
questions in Text2Cypher Mode and Auto Mode. Run them in Neo4j Browser to confirm that the graph
model supports structured entity lookups directly, without vector search.

---

### Which underwriting rules apply to John Smith?

```cypher
MATCH (a:Applicant {name: "John Smith"})-[:HAS_CONDITION]->(rf:RiskFactor)-[:EVALUATED_BY]->(r:UnderwritingRule)
RETURN DISTINCT a.name AS applicant, rf.name AS risk_factor, r.title AS rule, r.decision AS decision
ORDER BY rf.name
LIMIT 25
```

**Expected:** Rows linking John Smith's RiskFactor nodes to their governing UnderwritingRule nodes,
with the structured decision outcome for each.

---

### What risk factors does John Smith have?

```cypher
MATCH (a:Applicant {name: "John Smith"})-[:HAS_CONDITION]->(rf:RiskFactor)
RETURN rf.name AS risk_factor, rf.category AS category, rf.controlled AS controlled
ORDER BY rf.name
LIMIT 25
```

**Expected:** 3 rows — the risk factors stored on John Smith's node via `HAS_CONDITION` relationships,
with category and controlled status. These are the structured facts Text2Cypher retrieves directly
without requiring semantic similarity.

---

### What policy is John Smith applying for?

```cypher
MATCH (a:Applicant {name: "John Smith"})-[:APPLIES_FOR]->(p:Policy)
RETURN p.name AS policy, p.type AS type, p.class_name AS class
LIMIT 25
```

**Expected:** 1 row — the Policy node linked to John Smith via `APPLIES_FOR`, showing product name,
type, and underwriting class. This is a structured lookup that Text2Cypher handles precisely where
vector search would return semantically similar — but not necessarily correct — results.

---

## RBAC Validation Queries

These queries verify the role-based access control layer. The first group runs as the
Neo4j **admin** user (to inspect the roles themselves); the second group connects as
each **role user** to prove the access scoping. See [RBAC.md](RBAC.md) for the full design.

### Query R1 — Confirm Enterprise Edition

RBAC requires Neo4j Enterprise. This should return `enterprise`.

```cypher
CALL dbms.components() YIELD edition RETURN edition;
```

### Query R2 — List roles and users

```cypher
SHOW ROLES;
SHOW USERS;
```

**Expected roles include:** `underwriter`, `senior_underwriter`, `underwriting_manager`
(alongside the built-in roles). **Expected users:** `uw_standard`, `uw_senior`,
`uw_manager` (alongside `neo4j`).

### Query R3 — Inspect the privileges granted to a role

```cypher
SHOW ROLE underwriter PRIVILEGES AS COMMANDS;
```

**Expected:** the full-read grants (`GRANT ACCESS` on the database, `GRANT MATCH {*}`
on `NODES *`, and `GRANT TRAVERSE` + `GRANT READ {*}` on `RELATIONSHIPS *`), plus
`DENY TRAVERSE` and `DENY READ {*}` on the `Restricted` and `Confidential` labels.

### Query R4 — Tiers on applicant nodes

```cypher
MATCH (a:Applicant)
RETURN a.name AS applicant, a.sensitivity AS tier, labels(a) AS labels
ORDER BY a.sensitivity, a.name;
```

**Expected:** 6 rows; each applicant has two labels (e.g. `["Applicant","Standard"]`)
and a matching `sensitivity` property.

---

### Connecting as each role to prove scoping

Run these from a shell — each connects as a different Neo4j role user. The **same
query** returns a different number of applicants per role.

```bash
# Underwriter — 2 applicants (Standard only)
docker exec -it neo4j-insurance-graphrag-rbac cypher-shell -u uw_standard -p demo1234 \
  "MATCH (a:Applicant) RETURN a.name, a.sensitivity ORDER BY a.name;"

# Senior Underwriter — 4 applicants (+ Restricted)
docker exec -it neo4j-insurance-graphrag-rbac cypher-shell -u uw_senior -p demo1234 \
  "MATCH (a:Applicant) RETURN a.name, a.sensitivity ORDER BY a.name;"

# Underwriting Manager — all 6 applicants
docker exec -it neo4j-insurance-graphrag-rbac cypher-shell -u uw_manager -p demo1234 \
  "MATCH (a:Applicant) RETURN a.name, a.sensitivity ORDER BY a.name;"
```

### Query R5 — Enforcement survives traversal

The key property: scoping holds even when applicants are reached *through* other nodes.
Run this as `uw_standard` — it starts from `Policy` and walks into `Applicant`, yet still
returns only the two Standard-tier applicants.

```cypher
MATCH (p:Policy)<-[:APPLIES_FOR]-(a:Applicant)
RETURN p.name AS policy, a.name AS applicant
ORDER BY a.name;
```

**Expected (as `uw_standard`):** only John Smith and Maria Garcia, even though
confidential applicants also hold policies. This is why the GraphRAG pipeline — which
traverses into applicants from matched chunks — is scoped automatically, with no
pipeline changes.

### Query R6 — Defense-in-depth: same generated query, different results

This mirrors what Text2Cypher executes. Run the identical query as different role users;
the row count differs because the engine filters, not the query.

```cypher
MATCH (a:Applicant)-[:APPLIES_FOR]->(p:Policy)
RETURN a.name AS applicant_name, p.type AS policy_type
LIMIT 25;
```

**Expected:** 2 rows as `uw_standard`, 4 as `uw_senior`, 6 as `uw_manager` — same
Cypher, role-scoped results.
