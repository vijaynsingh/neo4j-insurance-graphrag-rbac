from app.config import USE_OPENAI_LLM, OPENAI_API_KEY
from app.graph import get_driver
from app.graph_retriever import GraphRetriever
from app.mock_llm import MockLLM


# ------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------

def _get_llm():
    """
    Returns OpenAILLM when USE_OPENAI_LLM=true and OPENAI_API_KEY is set.
    Falls back to MockLLM otherwise — deterministic, zero cost, zero API dependency.
    """
    if USE_OPENAI_LLM and OPENAI_API_KEY:
        from app.openai_llm import OpenAILLM
        return OpenAILLM()
    return MockLLM()


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class GraphRAGPipeline:
    """
    End-to-end GraphRAG pipeline:

        Question
          ↓  GraphRetriever.retrieve_context()
              Phase 1: vector search  (mock or OpenAI embeddings)
              Phase 2: graph traversal
        Structured graph context
          ↓  LLM.generate_answer()   (MockLLM or OpenAILLM)
        Final answer with decision, reasoning, and citations
    """

    def __init__(self, driver, embedding_provider=None, llm=None):
        self.retriever = GraphRetriever(driver, embedding_provider=embedding_provider)
        self.llm = llm if llm is not None else _get_llm()

    @classmethod
    def for_mode(cls, driver, mode: str) -> "GraphRAGPipeline":
        """
        Factory that wires the correct embedding provider and LLM for the given mode.

        mode='demo'   → MockEmbeddingProvider + MockLLM   (no API calls, zero cost)
        mode='openai' → OpenAIEmbeddingProvider + OpenAILLM (requires OPENAI_API_KEY)
        """
        if mode == "openai":
            from app.embed import OpenAIEmbeddingProvider
            from app.openai_llm import OpenAILLM
            return cls(driver, embedding_provider=OpenAIEmbeddingProvider(), llm=OpenAILLM())
        from app.embed import MockEmbeddingProvider
        return cls(driver, embedding_provider=MockEmbeddingProvider(), llm=MockLLM())

    def run(self, question: str, top_k: int = 3, check_compatibility: bool = True) -> dict:
        context = self.retriever.retrieve_context(
            question, top_k=top_k, check_compatibility=check_compatibility
        )
        answer = self.llm.generate_answer(question, context)
        return {
            "question": question,
            "context": context,
            "answer": answer,
        }

    def run_with_driver(
        self,
        question: str,
        scoped_driver,
        top_k: int = 3,
        check_compatibility: bool = True,
    ) -> dict:
        """
        Run the full pipeline using `scoped_driver` for all Neo4j queries.

        Creates a temporary GraphRetriever authenticated as the RBAC user so
        Neo4j's DENY rules suppress denied-tier applicants in both Phase 1
        (vector search) and Phase 2 (graph traversal).  The embedding provider
        and LLM are shared with the base pipeline — no extra API clients are
        allocated per request.
        """
        scoped_retriever = GraphRetriever(
            scoped_driver,
            embedding_provider=self.retriever.embedding_provider,
        )
        context = scoped_retriever.retrieve_context(
            question, top_k=top_k, check_compatibility=check_compatibility
        )
        answer = self.llm.generate_answer(question, context)
        return {
            "question": question,
            "context": context,
            "answer": answer,
        }


# ------------------------------------------------------------------
# Display helper
# ------------------------------------------------------------------

def _mode_line(pipeline: GraphRAGPipeline) -> str:
    embed_model = pipeline.retriever.embedding_provider.model_name
    llm_name = type(pipeline.llm).__name__
    return f"Embedding: {embed_model}  |  LLM: {llm_name}"


def _print_result(result: dict, pipeline: GraphRAGPipeline) -> None:
    line = "─" * 60
    answer = result["answer"]
    context = result["context"]

    print(f"\n{line}")
    print("QUESTION:")
    print(f"  {result['question']}")

    print(f"\n{line}")
    print("GRAPH CONTEXT:")

    chunks = context["matched_chunks"]
    print(f"  Matched chunks ({len(chunks)}):")
    for c in chunks:
        print(f"    [{c['score']:.4f}]  {c['source']}")

    rules = context["rules"]
    print(f"  Underwriting rules ({len(rules)}):")
    for r in rules:
        print(f"    •  {r['title']}  →  {r['decision']}")

    risk_factors = context["risk_factors"]
    print(f"  Risk factors ({len(risk_factors)}):")
    for rf in risk_factors:
        print(f"    •  {rf['name']}  [{rf['category']}]")

    applicants = context["applicants"]
    print(f"  Applicants ({len(applicants)}):")
    for a in applicants:
        print(f"    •  {a['name']}, age {a['age']}")

    print(f"\n{line}")
    print("FINAL DECISION:")
    print(f"  {answer['decision']}")

    print(f"\n{line}")
    print("REASONING:")
    for i, reason in enumerate(answer["reasoning"], 1):
        print(f"  {i}. {reason}")

    print(f"\n{line}")
    print("CITATIONS:")
    for cit in answer["citations"]:
        if isinstance(cit, dict):
            if cit.get("type") == "DocumentChunk":
                print(f"  [DocumentChunk]    {cit['source']}  (score: {cit.get('relevance_score', 'n/a')})")
            elif cit.get("type") == "UnderwritingRule":
                print(f"  [UnderwritingRule] {cit['title']}  →  {cit['decision']}")
            else:
                print(f"  {cit}")
        else:
            print(f"  {cit}")

    print(f"\n{line}")
    print(f"MODE: {_mode_line(pipeline)}")
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

    pipeline = GraphRAGPipeline(driver)
    print(f"Mode: {_mode_line(pipeline)}\n")

    question = (
        "Should a diabetic applicant with A1C below 7.0 qualify for preferred term life?"
    )
    result = pipeline.run(question)
    _print_result(result, pipeline)

    driver.close()


if __name__ == "__main__":
    main()
