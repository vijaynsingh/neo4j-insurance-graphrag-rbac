import json
from pathlib import Path
from neo4j import GraphDatabase
from app.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
from app.embed import EMBEDDING_DIMENSION, get_embedding_provider

DATA_FILE = Path(__file__).parent.parent / "data" / "underwriting_sample.json"


def load_data():
    with open(DATA_FILE) as f:
        return json.load(f)


def clear_sample_data(session, data):
    all_ids = (
        [a["id"] for a in data["applicants"]]
        + [p["id"] for p in data["policies"]]
        + [r["id"] for r in data["risk_factors"]]
        + [l["id"] for l in data["lab_results"]]
        + [u["id"] for u in data["underwriting_rules"]]
        + [c["id"] for c in data["document_chunks"]]
    )
    session.run("MATCH (n) WHERE n.id IN $ids DETACH DELETE n", ids=all_ids)
    print(f"  Targeted {len(all_ids)} node IDs for deletion.")


def create_constraints(session):
    constraints = [
        "CREATE CONSTRAINT applicant_id IF NOT EXISTS FOR (n:Applicant) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (n:Policy) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT risk_factor_id IF NOT EXISTS FOR (n:RiskFactor) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT lab_result_id IF NOT EXISTS FOR (n:LabResult) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (n:UnderwritingRule) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:DocumentChunk) REQUIRE n.id IS UNIQUE",
    ]
    for cql in constraints:
        session.run(cql)
    print(f"  {len(constraints)} constraints created (or already existed).")


_SENSITIVITY_LABELS = {
    "standard": "Standard",
    "restricted": "Restricted",
    "confidential": "Confidential",
}


def create_nodes(session, data):
    for node in data["applicants"]:
        props = {k: v for k, v in node.items() if k != "id"}
        session.run("MERGE (n:Applicant {id: $id}) SET n += $props", id=node["id"], props=props)
        tier = node.get("sensitivity")
        if tier and tier in _SENSITIVITY_LABELS:
            # Cypher cannot accept a label name as a parameter, so we use an
            # allowlist to safely interpolate the validated label string.
            label = _SENSITIVITY_LABELS[tier]
            session.run(f"MATCH (n:Applicant {{id: $id}}) SET n:{label}", id=node["id"])
    print(f"  {len(data['applicants'])} Applicant(s)")

    for node in data["policies"]:
        props = {k: v for k, v in node.items() if k != "id"}
        session.run("MERGE (n:Policy {id: $id}) SET n += $props", id=node["id"], props=props)
    print(f"  {len(data['policies'])} Policy(s)")

    for node in data["risk_factors"]:
        props = {k: v for k, v in node.items() if k != "id"}
        session.run("MERGE (n:RiskFactor {id: $id}) SET n += $props", id=node["id"], props=props)
    print(f"  {len(data['risk_factors'])} RiskFactor(s)")

    for node in data["lab_results"]:
        props = {k: v for k, v in node.items() if k != "id"}
        session.run("MERGE (n:LabResult {id: $id}) SET n += $props", id=node["id"], props=props)
    print(f"  {len(data['lab_results'])} LabResult(s)")

    for node in data["underwriting_rules"]:
        props = {k: v for k, v in node.items() if k != "id"}
        session.run("MERGE (n:UnderwritingRule {id: $id}) SET n += $props", id=node["id"], props=props)
    print(f"  {len(data['underwriting_rules'])} UnderwritingRule(s)")

    for node in data["document_chunks"]:
        props = {k: v for k, v in node.items() if k != "id"}
        session.run("MERGE (n:DocumentChunk {id: $id}) SET n += $props", id=node["id"], props=props)
    print(f"  {len(data['document_chunks'])} DocumentChunk(s)")


def attach_embeddings(session, data) -> None:
    """
    Embeds each DocumentChunk.text and stores it via db.create.setNodeVectorProperty.

    db.create.setNodeVectorProperty stores Float32[] — what Neo4j's HNSW vector index
    expects. A plain SET n.embedding = $list stores Float64[] and may cause type
    mismatches on some Neo4j versions.

    Also stores embedding_provider and embedding_model as node properties so
    GraphRetriever can detect provider/model mismatches at query time.

    IMPORTANT: The embedding provider used here must match the provider used when
    querying. If you switch providers (e.g. mock → OpenAI), re-run this script
    before running the pipeline.
    """
    provider = get_embedding_provider()
    print(f"  Embedding provider: {provider.model_name}")

    for chunk in data["document_chunks"]:
        embedding = provider.embed(chunk["text"])
        session.run(
            """
            MATCH (n:DocumentChunk {id: $id})
            CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
            SET n.embedding_model    = $model,
                n.embedding_provider = $provider_type
            """,
            id=chunk["id"],
            embedding=embedding,
            model=provider.model_name,
            provider_type=type(provider).__name__,
        )
    print(
        f"  {len(data['document_chunks'])} DocumentChunk(s) — "
        f"{EMBEDDING_DIMENSION}-dim embeddings stored  [model: {provider.model_name}]"
    )


def reindex_embeddings(driver, provider) -> int:
    """
    Re-embeds every DocumentChunk node using the given provider and updates
    embedding_model / embedding_provider metadata on each node.

    Does NOT touch graph structure (nodes, relationships, constraints) — only
    the embedding vectors and their metadata properties.  Much faster than a
    full reseed.  Called automatically by the API when a mode switch is detected.

    Returns the number of chunks re-indexed.
    """
    data = load_data()
    chunks = data["document_chunks"]
    with driver.session(database=NEO4J_DATABASE) as session:
        for chunk in chunks:
            embedding = provider.embed(chunk["text"])
            session.run(
                """
                MATCH (n:DocumentChunk {id: $id})
                CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
                SET n.embedding_model    = $model,
                    n.embedding_provider = $provider_type
                """,
                id=chunk["id"],
                embedding=embedding,
                model=provider.model_name,
                provider_type=type(provider).__name__,
            )
    print(
        f"[auto-reindex] {len(chunks)} chunk(s) re-embedded "
        f"[model: {provider.model_name}]"
    )
    return len(chunks)


def create_relationships(session, data):
    rels = data["relationships"]

    for r in rels["applies_for"]:
        session.run(
            "MATCH (a:Applicant {id: $a}), (p:Policy {id: $p}) MERGE (a)-[:APPLIES_FOR]->(p)",
            a=r["applicant_id"], p=r["policy_id"],
        )
    print(f"  {len(rels['applies_for'])} APPLIES_FOR")

    for r in rels["has_condition"]:
        session.run(
            "MATCH (a:Applicant {id: $a}), (rf:RiskFactor {id: $rf}) MERGE (a)-[:HAS_CONDITION]->(rf)",
            a=r["applicant_id"], rf=r["risk_factor_id"],
        )
    print(f"  {len(rels['has_condition'])} HAS_CONDITION")

    for r in rels["has_lab_result"]:
        session.run(
            "MATCH (a:Applicant {id: $a}), (l:LabResult {id: $l}) MERGE (a)-[:HAS_LAB_RESULT]->(l)",
            a=r["applicant_id"], l=r["lab_result_id"],
        )
    print(f"  {len(rels['has_lab_result'])} HAS_LAB_RESULT")

    for r in rels["has_rule"]:
        session.run(
            "MATCH (p:Policy {id: $p}), (u:UnderwritingRule {id: $u}) MERGE (p)-[:HAS_RULE]->(u)",
            p=r["policy_id"], u=r["rule_id"],
        )
    print(f"  {len(rels['has_rule'])} HAS_RULE")

    for r in rels["supported_by"]:
        session.run(
            "MATCH (u:UnderwritingRule {id: $u}), (c:DocumentChunk {id: $c}) MERGE (u)-[:SUPPORTED_BY]->(c)",
            u=r["rule_id"], c=r["chunk_id"],
        )
    print(f"  {len(rels['supported_by'])} SUPPORTED_BY")

    for r in rels["evaluated_by"]:
        session.run(
            "MATCH (rf:RiskFactor {id: $rf}), (u:UnderwritingRule {id: $u}) MERGE (rf)-[:EVALUATED_BY]->(u)",
            rf=r["risk_factor_id"], u=r["rule_id"],
        )
    print(f"  {len(rels['evaluated_by'])} EVALUATED_BY")


def main():
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()
        print("Connection verified.\n")
    except Exception as e:
        print(f"ERROR: Could not connect to Neo4j — {e}")
        print("Is Docker running? Try: docker compose up -d")
        return

    data = load_data()
    print(f"Loaded data from {DATA_FILE}\n")

    with driver.session(database=NEO4J_DATABASE) as session:
        print("Step 1: Clearing existing sample data...")
        clear_sample_data(session, data)

        print("\nStep 2: Creating constraints...")
        create_constraints(session)

        print("\nStep 3: Creating nodes...")
        create_nodes(session, data)

        print("\nStep 4: Creating relationships...")
        create_relationships(session, data)

        print("\nStep 5: Attaching embeddings to DocumentChunk nodes...")
        attach_embeddings(session, data)

    driver.close()

    print("\nSeed complete.")
    print("─" * 50)
    print("Open Neo4j Browser: http://localhost:7474")
    print("Validate with:  MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC")
    print("Next:           python -m app.vector_index  (creates index + runs similarity search)")
    print("Full walkthrough: see CYPHER_QUERIES.md")


if __name__ == "__main__":
    main()
