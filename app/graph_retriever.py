from app.graph import get_driver, run_query
from app.embed import get_embedding_provider, EMBEDDING_DIMENSION

INDEX_NAME = "document_chunk_embeddings"


class GraphRetriever:
    """
    Two-phase retriever:
      Phase 1 — vector search: embed the question and find the most similar DocumentChunk nodes.
      Phase 2 — graph traversal: walk outward from each matched chunk to collect structured
                business context (rules, risk factors, policies, applicants).

    The embedding_provider parameter controls which model is used for query-time embedding.
    It must match the provider used when seed.py stored embeddings on DocumentChunk nodes.
    If they differ, retrieval quality degrades silently — GraphRetriever logs a warning.
    """

    def __init__(self, driver, embedding_provider=None):
        self.driver = driver
        self.embedding_provider = embedding_provider or get_embedding_provider()

    # ------------------------------------------------------------------
    # Embedding/model compatibility check
    # ------------------------------------------------------------------

    def _check_embedding_compatibility(self) -> str | None:
        """
        Reads the embedding_model stored on DocumentChunk nodes during seeding and
        compares it to the current provider's model name. Returns a warning string
        if they differ, None otherwise. Skips silently (returns None) if no metadata
        is stored (pre-Step 8 seed data).
        """
        rows = run_query(
            self.driver,
            "MATCH (d:DocumentChunk) WHERE d.embedding_model IS NOT NULL "
            "RETURN d.embedding_model AS model LIMIT 1",
        )
        if not rows:
            return None
        stored_model = rows[0]["model"]
        current_model = self.embedding_provider.model_name
        if stored_model != current_model:
            return (
                f"Embedding mismatch: the index was built with '{stored_model}' "
                f"but this request uses '{current_model}'. "
                "Similarity scores are meaningless — "
                "re-run `python3 -m app.seed` with the matching provider."
            )
        return None

    # ------------------------------------------------------------------
    # Phase 1: vector search
    # ------------------------------------------------------------------

    def _vector_search(self, question: str, top_k: int) -> list[dict]:
        vector = self.embedding_provider.embed(question)
        return run_query(
            self.driver,
            """
            CALL db.index.vector.queryNodes($index, $top_k, $vector)
            YIELD node, score
            RETURN node.id AS id, node.source AS source, node.text AS text, score
            """,
            {"index": INDEX_NAME, "top_k": top_k, "vector": vector},
        )

    # ------------------------------------------------------------------
    # Phase 2: graph traversal
    # ------------------------------------------------------------------

    def _traverse_from_chunks(self, chunk_ids: list[str]) -> list[dict]:
        """
        Single round-trip to Neo4j. UNWIND lets us traverse all chunk_ids at once
        rather than making one query per chunk — important when top_k is large.

        Traversal path from each chunk:
            DocumentChunk <-[SUPPORTED_BY]- UnderwritingRule
                            <-[EVALUATED_BY]- RiskFactor
                            <-[HAS_RULE]- Policy
            RiskFactor     <-[HAS_CONDITION]- Applicant (via condition)
            Policy         <-[APPLIES_FOR]- Applicant   (via policy)

        Returns one row per (chunk_id, rule) pair with aggregated risk factors,
        policies, and applicants collected per rule.
        """
        return run_query(
            self.driver,
            """
            UNWIND $chunk_ids AS chunk_id
            MATCH (d:DocumentChunk {id: chunk_id})<-[:SUPPORTED_BY]-(r:UnderwritingRule)
            OPTIONAL MATCH (rf:RiskFactor)-[:EVALUATED_BY]->(r)
            OPTIONAL MATCH (p:Policy)-[:HAS_RULE]->(r)
            OPTIONAL MATCH (a_cond:Applicant)-[:HAS_CONDITION]->(rf)
            OPTIONAL MATCH (a_pol:Applicant)-[:APPLIES_FOR]->(p)
            RETURN
                chunk_id   AS source_chunk,
                r.id       AS rule_id,
                r.title    AS rule_title,
                r.text     AS rule_text,
                r.decision AS decision,
                [x IN collect(DISTINCT rf) WHERE x IS NOT NULL
                    | {name: x.name, category: x.category}]                          AS risk_factors,
                [x IN collect(DISTINCT p) WHERE x IS NOT NULL
                    | {name: x.name, type: x.type}]                                   AS policies,
                [x IN (collect(DISTINCT a_cond) + collect(DISTINCT a_pol)) WHERE x IS NOT NULL
                    | {name: x.name, age: x.age}]                                     AS applicants
            """,
            {"chunk_ids": chunk_ids},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_context(self, question: str, top_k: int = 3, check_compatibility: bool = True) -> dict:
        """
        Returns a structured context dict ready to be serialised into an LLM prompt.

        Keys:
          matched_chunks       — the vector search hits (id, source, text, score)
          rules                — UnderwritingRule nodes reachable from those chunks
          risk_factors         — RiskFactor nodes linked to those rules via EVALUATED_BY
          policies             — Policy nodes that own those rules via HAS_RULE
          applicants           — Applicant nodes connected through HAS_CONDITION or APPLIES_FOR
          compatibility_warning — str if embedding provider mismatches index, else None
        """
        warning = self._check_embedding_compatibility() if check_compatibility else None

        chunks = self._vector_search(question, top_k)
        if not chunks:
            return {
                "matched_chunks": [],
                "rules": [],
                "risk_factors": [],
                "policies": [],
                "applicants": [],
                "compatibility_warning": warning,
            }

        chunk_ids = [c["id"] for c in chunks]
        traversal_rows = self._traverse_from_chunks(chunk_ids)

        # Deduplicate by id/name across all traversal rows.
        rules: dict[str, dict] = {}
        risk_factors: dict[str, dict] = {}
        policies: dict[str, dict] = {}
        applicants: dict[str, dict] = {}

        for row in traversal_rows:
            rule_id = row.get("rule_id")
            if rule_id and rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "title": row["rule_title"],
                    "text": row["rule_text"],
                    "decision": row["decision"],
                }
            for rf in row.get("risk_factors") or []:
                if rf.get("name"):
                    risk_factors[rf["name"]] = rf
            for p in row.get("policies") or []:
                if p.get("name"):
                    policies[p["name"]] = p
            for a in row.get("applicants") or []:
                if a.get("name"):
                    applicants[a["name"]] = a

        return {
            "matched_chunks": chunks,
            "rules": list(rules.values()),
            "risk_factors": list(risk_factors.values()),
            "policies": list(policies.values()),
            "applicants": list(applicants.values()),
            "compatibility_warning": warning,
        }


# ------------------------------------------------------------------
# Display helper
# ------------------------------------------------------------------

def print_context(context: dict, question: str) -> None:
    line = "─" * 60

    print(f"\n{line}")
    print(f"QUESTION: {question}")
    print(line)

    chunks = context["matched_chunks"]
    print(f"\n[Phase 1 — Vector Search]  {len(chunks)} chunk(s) matched\n")
    for c in chunks:
        print(f"  [{c['score']:.4f}]  {c['source']}")
        print(f"           {c['text'][:110]}...")
        print()

    rules = context["rules"]
    print(f"[Phase 2 — Graph Traversal]  found from {len(chunks)} chunk(s):\n")

    print(f"  Underwriting Rules ({len(rules)})")
    for r in rules:
        print(f"    • {r['title']}")
        print(f"      Text:     {r['text']}")
        print(f"      Decision: {r['decision']}")

    risk_factors = context["risk_factors"]
    print(f"\n  Risk Factors ({len(risk_factors)})")
    for rf in risk_factors:
        print(f"    • {rf['name']}  [{rf['category']}]")

    policies = context["policies"]
    print(f"\n  Policies ({len(policies)})")
    for p in policies:
        print(f"    • {p['name']}  (type: {p.get('type', 'n/a')})")

    apps = context["applicants"]
    print(f"\n  Applicants ({len(apps)})")
    for a in apps:
        print(f"    • {a['name']}, age {a['age']}")

    print(f"\n{line}")
    print("NOTE: With mock embeddings chunk ranking is not semantic.")
    print("In Step 6, mock_embed() is replaced by a real embedder.")
    print("The context structure above is what the LLM will receive.")
    print(line)


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

def main() -> None:
    driver = get_driver()
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j.")
    except Exception as e:
        print(f"ERROR: Cannot connect — {e}")
        print("Is Docker running?  docker compose up -d")
        return

    retriever = GraphRetriever(driver)

    questions = [
        "Should a diabetic applicant with A1C below 7.0 qualify for preferred term life?",
        "How does tobacco use affect underwriting classification?",
        "What additional review is required for applicants over 45 with diabetes?",
    ]

    for question in questions:
        context = retriever.retrieve_context(question, top_k=2)
        print_context(context, question)

    driver.close()


if __name__ == "__main__":
    main()
