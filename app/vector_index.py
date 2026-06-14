from app.graph import get_driver, run_query
from app.embed import mock_embed, EMBEDDING_DIMENSION
from app.config import NEO4J_DATABASE

INDEX_NAME = "document_chunk_embeddings"


def create_vector_index(driver) -> None:
    """
    Creates the HNSW vector index on DocumentChunk.embedding.
    IF NOT EXISTS makes this safe to call repeatedly.
    The index is built over existing nodes immediately on creation.
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(
            f"""
            CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
            FOR (n:DocumentChunk)
            ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {EMBEDDING_DIMENSION},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
        )
    print(f"  Index '{INDEX_NAME}' created (or already exists).")


def verify_index(driver) -> bool:
    """Returns True if the index exists and is ONLINE."""
    all_indexes = run_query(
        driver,
        "SHOW VECTOR INDEXES YIELD name, state, labelsOrTypes, properties",
    )
    match = [idx for idx in all_indexes if idx["name"] == INDEX_NAME]
    if not match:
        print(f"  ERROR: index '{INDEX_NAME}' not found.")
        print("  Have you run: python -m app.seed ?")
        return False

    idx = match[0]
    print(f"  Name:       {idx['name']}")
    print(f"  State:      {idx['state']}")
    print(f"  Node label: {idx['labelsOrTypes']}")
    print(f"  Property:   {idx['properties']}")
    return idx["state"] == "ONLINE"


def similarity_search(driver, query_text: str, top_k: int = 3) -> list[dict]:
    """
    Embeds query_text and returns the top_k DocumentChunk nodes by cosine similarity.
    With mock embeddings the scores reflect hash proximity, not semantic meaning.
    """
    query_vector = mock_embed(query_text)
    return run_query(
        driver,
        """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        RETURN node.id AS id, node.source AS source, node.text AS text, score
        """,
        {
            "index_name": INDEX_NAME,
            "top_k": top_k,
            "query_vector": query_vector,
        },
    )


def main() -> None:
    driver = get_driver()
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j.\n")
    except Exception as e:
        print(f"ERROR: Cannot connect — {e}")
        print("Is Docker running? Try: docker compose up -d")
        return

    print("Step 1: Creating vector index...")
    create_vector_index(driver)

    print("\nStep 2: Verifying index is ONLINE...")
    online = verify_index(driver)
    if not online:
        driver.close()
        return

    print("\nStep 3: Sample similarity search")
    queries = [
        "diabetes and A1C control for life insurance",
        "tobacco use and preferred classification",
        "age over 45 additional medical review",
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        results = similarity_search(driver, query, top_k=2)
        for r in results:
            print(f"    [{r['score']:.4f}] {r['source']}")
            print(f"             {r['text'][:80]}...")

    driver.close()

    print("\n" + "─" * 55)
    print("Vector index is working.")
    print("NOTE: scores are hash-based, not semantic — this is expected.")
    print("Swap mock_embed() for a real embedder in Step 6 for meaningful results.")


if __name__ == "__main__":
    main()
