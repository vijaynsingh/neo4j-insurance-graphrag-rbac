"""
Retrieval router and hybrid synthesizer for Auto Mode.

RetrievalRouter   — one GPT-4o call to classify the question into a retrieval strategy.
HybridSynthesizer — one GPT-4o call to merge GraphRAG and Text2Cypher results.

Both classes are stateless beyond the OpenAI client; one instance each is held in
app.state for the lifetime of the server.
"""

import json

from app.config import OPENAI_API_KEY, OPENAI_LLM_MODEL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_STRATEGIES = {"openai_graph", "text2cypher", "hybrid"}
_FALLBACK_STRATEGY = "openai_graph"
_FALLBACK_REASON = "Fallback to openai_graph because router output was invalid."

_ROUTER_SYSTEM_PROMPT = """\
You are a retrieval strategy router for an insurance underwriting knowledge graph system.

Given a user question, classify it into exactly one of these three retrieval strategies.

--- openai_graph ---
Use when the question:
- asks for underwriting guidance or policy interpretation
- asks what the underwriting manual says
- is semantic and does not require a specific named-entity lookup
Examples:
  "Explain how controlled diabetes affects preferred underwriting."
  "What does the underwriting manual say about tobacco use?"

--- text2cypher ---
Use when the question:
- is a structured lookup (list, count, filter)
- asks what entities exist in the graph
- names a specific entity but asks only for stored facts, not recommendations
Examples:
  "What risk factors does John Smith have?"
  "Which rules are connected to Type 2 Diabetes?"
  "What policy is John Smith applying for?"

--- hybrid ---
Use when the question:
- names a specific entity AND asks for a recommendation, qualification, or explanation
- requires both entity-level facts AND policy/manual interpretation
Examples:
  "Should John Smith qualify for preferred term life based on his diabetes?"
  "Does John Smith require additional underwriting review?"
  "Explain whether John Smith's A1C affects his preferred policy eligibility."
  "Based on John Smith's profile, what underwriting rules apply and what is the recommendation?"

Return valid JSON only. No prose outside the JSON object.

Required JSON shape:
{
  "selected_strategy": "openai_graph" | "text2cypher" | "hybrid",
  "router_reason": "<one sentence explaining why this strategy was chosen>"
}
"""

_HYBRID_SYNTHESIS_SYSTEM_PROMPT = """\
You are an insurance underwriting assistant performing a final synthesis.

You have two independent analyses of the same question:
1. GraphRAG analysis — from vector search over the underwriting manual and graph traversal.
2. Text2Cypher analysis — from a direct Cypher query against the knowledge graph.

Synthesize both into a single, coherent final underwriting recommendation.

Guidelines:
- Use Text2Cypher results for entity-specific facts (who, what, which).
- Use GraphRAG analysis for policy rules and manual guidance.
- Base your answer ONLY on the provided context. Do not invent facts.
- If context is insufficient for a confident decision, use REQUIRE_ADDITIONAL_REVIEW.

Return valid JSON only. No prose outside the JSON object.

Required JSON shape:
{
  "decision": "APPROVE | REFER_FOR_REVIEW | REQUIRE_ADDITIONAL_REVIEW | DECLINE",
  "reasoning": ["step 1 ...", "step 2 ..."]
}
"""


# ---------------------------------------------------------------------------
# RetrievalRouter
# ---------------------------------------------------------------------------

class RetrievalRouter:
    """
    Classifies a natural-language question into one retrieval strategy.

    Returns {"selected_strategy": ..., "router_reason": ...}.
    Falls back to openai_graph on any failure or invalid response.
    """

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_LLM_MODEL

    def classify(self, question: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                    {"role": "user",   "content": question},
                ],
                temperature=0,
            )
            parsed = json.loads(response.choices[0].message.content)
            strategy = parsed.get("selected_strategy", "")
            reason = str(parsed.get("router_reason", ""))
            if strategy not in _VALID_STRATEGIES:
                return {"selected_strategy": _FALLBACK_STRATEGY, "router_reason": _FALLBACK_REASON}
            return {"selected_strategy": strategy, "router_reason": reason}
        except Exception:
            return {"selected_strategy": _FALLBACK_STRATEGY, "router_reason": _FALLBACK_REASON}


# ---------------------------------------------------------------------------
# HybridSynthesizer
# ---------------------------------------------------------------------------

class HybridSynthesizer:
    """
    Merges a GraphRAG result and a Text2Cypher result into one final answer.

    If t2c_result is None (Text2Cypher failed), synthesizes from GraphRAG alone.
    Falls back to the raw GraphRAG answer if the GPT-4o synthesis call itself fails.
    """

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_LLM_MODEL

    def synthesize(self, question: str, graphrag_result: dict, t2c_result: dict | None) -> dict:
        user_prompt = (
            f"Question: {question}\n\n"
            f"=== GraphRAG Analysis ===\n{_fmt_graphrag(graphrag_result)}\n\n"
            f"=== Text2Cypher Analysis ===\n{_fmt_t2c(t2c_result)}\n\n"
            "Synthesize both analyses into a final underwriting recommendation."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _HYBRID_SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
            )
            parsed = json.loads(response.choices[0].message.content)
            decision = parsed.get("decision", "ANSWERED")
            reasoning = parsed.get("reasoning", [])
            if not isinstance(reasoning, list):
                reasoning = [str(reasoning)]
            return {"decision": decision, "reasoning": reasoning}
        except Exception:
            # Degrade gracefully to the raw GraphRAG answer
            answer = graphrag_result.get("answer", {})
            return {
                "decision": answer.get("decision", "ANSWERED"),
                "reasoning": answer.get("reasoning", []),
            }


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------

def _fmt_graphrag(result: dict) -> str:
    answer = result.get("answer", {})
    context = result.get("context", {})
    lines = [
        f"Decision: {answer.get('decision', 'n/a')}",
        f"Reasoning: {'; '.join(answer.get('reasoning', []))}",
    ]
    rules = context.get("rules", [])
    if rules:
        lines.append(f"Rules: {'; '.join(r.get('title', '') for r in rules)}")
    chunks = context.get("matched_chunks", [])
    if chunks:
        lines.append(f"Document sources: {'; '.join(c.get('source', '') for c in chunks)}")
    return "\n".join(lines)


def _fmt_t2c(result: dict | None) -> str:
    if result is None:
        return "Text2Cypher retrieval was not available for this request."
    return (
        f"Cypher query: {result.get('generated_cypher', 'n/a')}\n"
        f"Answer: {result.get('answer', 'n/a')}\n"
        f"Query results: {json.dumps(result.get('raw_query_results', []), default=str)}"
    )
