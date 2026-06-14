"""
Text2Cypher service: natural language → Cypher → Neo4j → GPT-4o answer.

Flow:
  1. generate_cypher()  — GPT-4o translates the question into a read-only Cypher query
  2. validate_cypher()  — safety check: reject writes, strip markdown, inject LIMIT if absent
  3. execute_cypher()   — run the query against Neo4j, cap rows at 50
  4. generate_answer()  — GPT-4o synthesises a plain-English answer from question + Cypher + rows
  5. run()              — orchestrates 1-4 and returns the standard result dict
"""

import json
import re

from neo4j.exceptions import CypherSyntaxError, Neo4jError

from app.config import OPENAI_API_KEY, OPENAI_LLM_MODEL
from app.graph import run_query

# ---------------------------------------------------------------------------
# Schema context injected into the Cypher-generation prompt
# ---------------------------------------------------------------------------

_SCHEMA = """
Graph schema:

Node labels and properties:
  Applicant(id, name, age)
  Policy(id, name, type, class_name)
  RiskFactor(id, name, category, controlled)
  LabResult(id, test_name, value, unit)
  UnderwritingRule(id, title, text, decision)
  DocumentChunk(id, source, text)

Relationship types:
  (Applicant)-[:APPLIES_FOR]->(Policy)
  (Applicant)-[:HAS_CONDITION]->(RiskFactor)
  (Applicant)-[:HAS_LAB_RESULT]->(LabResult)
  (Policy)-[:HAS_RULE]->(UnderwritingRule)
  (RiskFactor)-[:EVALUATED_BY]->(UnderwritingRule)
  (UnderwritingRule)-[:SUPPORTED_BY]->(DocumentChunk)
"""

_CYPHER_SYSTEM_PROMPT = f"""\
You are a Cypher query generator for a Neo4j insurance underwriting knowledge graph.

{_SCHEMA}

Rules:
- Return ONLY a valid Cypher MATCH query. Nothing else.
- No markdown code fences. No backticks. No explanation. No comments.
- Queries must be read-only: never use CREATE, MERGE, SET, DELETE, DETACH, REMOVE, DROP, \
LOAD CSV, FOREACH, or CALL procedures that mutate data.
- Use only the node labels and relationship types defined in the schema above.
- Use meaningful aliases in RETURN so results are self-describing.
- Always include LIMIT 25 unless a smaller limit is more appropriate for the question.
- If the question cannot be answered from this schema, return exactly: MATCH (n) RETURN null LIMIT 0
"""

# ---------------------------------------------------------------------------
# Keywords that make a Cypher query unsafe to execute
# ---------------------------------------------------------------------------

_WRITE_TOKENS = {
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE",
    "DROP", "FOREACH",
}

# Dangerous CALL sub-patterns (checked as substrings of normalised text)
_DANGEROUS_CALL_PATTERNS = [
    "call dbms",
    "call apoc.periodic",
    "call db.create",
    "call db.index",
    "call gds",
    "in transactions",
    "load csv",
]

_MAX_RECORDS = 50


class Text2CypherService:
    """
    Translates a natural-language question into a Cypher query, executes it
    against Neo4j, and uses GPT-4o to synthesise a plain-English answer.

    Requires OPENAI_API_KEY.  The Neo4j driver is shared from app lifespan state.
    """

    def __init__(self, driver):
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_LLM_MODEL
        self.driver = driver

    # ------------------------------------------------------------------
    # Step 1 — Generate Cypher from natural language
    # ------------------------------------------------------------------

    def generate_cypher(self, question: str) -> str:
        """Ask GPT-4o to produce a Cypher query for the given question."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _CYPHER_SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Step 2 — Validate and sanitise the generated Cypher
    # ------------------------------------------------------------------

    def validate_cypher(self, cypher: str) -> str:
        """
        Safety-check the generated Cypher.

        - Strip markdown fences if the model accidentally returned them.
        - Reject any write or dangerous operations.
        - Append LIMIT 25 when no LIMIT clause is present.

        Returns the safe, ready-to-execute Cypher string.
        Raises ValueError with a human-readable reason if the query is unsafe.
        """
        cypher = _strip_markdown_fences(cypher).strip()

        if not cypher:
            raise ValueError("Generated Cypher is empty.")

        normalised = cypher.upper()

        # Check for dangerous CALL sub-patterns first (case-insensitive substrings)
        lower = cypher.lower()
        for pattern in _DANGEROUS_CALL_PATTERNS:
            if pattern in lower:
                raise ValueError(
                    f"Generated Cypher contains a disallowed operation: '{pattern}'. "
                    "Only read-only queries are permitted."
                )

        # Tokenise and check for write keywords
        tokens = set(re.findall(r"[A-Z]+", normalised))
        bad = tokens & _WRITE_TOKENS
        if bad:
            raise ValueError(
                f"Generated Cypher contains disallowed write keyword(s): {sorted(bad)}. "
                "Only read-only queries are permitted."
            )

        # Must begin with a read clause
        first_token = normalised.split()[0] if normalised.split() else ""
        allowed_starts = {"MATCH", "OPTIONAL", "WITH", "UNWIND", "CALL", "RETURN"}
        if first_token not in allowed_starts:
            raise ValueError(
                f"Generated Cypher starts with '{first_token}', which is not an allowed "
                "read-only opening clause."
            )

        # Inject LIMIT if absent
        if "LIMIT" not in normalised:
            cypher = cypher.rstrip("; \n") + "\nLIMIT 25"

        return cypher

    # ------------------------------------------------------------------
    # Step 3 — Execute the validated Cypher against Neo4j
    # ------------------------------------------------------------------

    def execute_cypher(self, cypher: str) -> list[dict]:
        """
        Run the Cypher query and return rows as plain dicts.
        Caps results at _MAX_RECORDS (50) before returning.
        Wraps driver exceptions into ValueError so callers get clean messages.
        """
        try:
            rows = run_query(self.driver, cypher)
        except CypherSyntaxError as exc:
            raise ValueError(f"Cypher syntax error in generated query: {exc.message}") from exc
        except Neo4jError as exc:
            raise ValueError(f"Neo4j error executing generated query: {exc.message}") from exc

        return rows[:_MAX_RECORDS]

    # ------------------------------------------------------------------
    # Step 4 — Synthesise a plain-English answer from question + results
    # ------------------------------------------------------------------

    def generate_answer(
        self, question: str, cypher: str, records: list[dict]
    ) -> dict:
        """
        Call GPT-4o with the original question, the executed Cypher, and the
        Neo4j result rows.  Returns a structured dict with answer and reasoning.
        """
        records_text = json.dumps(records, indent=2, default=str) if records else "[]"

        user_prompt = (
            f"Question: {question}\n\n"
            f"Cypher query executed:\n{cypher}\n\n"
            f"Query results (JSON):\n{records_text}\n\n"
            "Based solely on the query results above, provide:\n"
            "1. A concise direct answer to the question.\n"
            "2. A brief explanation of what the data shows.\n\n"
            "Return valid JSON only, with this exact shape:\n"
            '{\n'
            '  "answer": "<one or two sentence direct answer>",\n'
            '  "reasoning": ["<step 1>", "<step 2>"]\n'
            '}'
        )

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an insurance underwriting assistant. "
                        "Answer questions based only on the provided Neo4j query results. "
                        "Do not invent facts not present in the results. "
                        "Return valid JSON only."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)

        answer = parsed.get("answer", "No answer generated.")
        reasoning = parsed.get("reasoning", [answer])
        if not isinstance(reasoning, list):
            reasoning = [str(reasoning)]

        return {
            "answer": answer,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self, question: str) -> dict:
        """
        Full Text2Cypher pipeline: generate → validate → execute → answer.

        Returns a dict with all fields needed by main.py to build AskResponse.
        On any failure the ValueError propagates to the /ask handler, which
        wraps it in an HTTP 500.
        """
        raw_cypher = self.generate_cypher(question)
        print(f"[text2cypher] generated cypher: {raw_cypher!r}")

        safe_cypher = self.validate_cypher(raw_cypher)
        print(f"[text2cypher] validated cypher: {safe_cypher!r}")

        records = self.execute_cypher(safe_cypher)
        print(f"[text2cypher] rows returned: {len(records)}")

        synthesis = self.generate_answer(question, safe_cypher, records)

        return {
            "answer":            synthesis["answer"],
            "reasoning":         synthesis["reasoning"],
            "generated_cypher":  safe_cypher,
            "raw_query_results": records,
            "mode":              "text2cypher",
            "llm_provider":      "OpenAILLM",
            "retrieval_strategy": "Text2Cypher",
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```cypher ... ``` or ``` ... ``` wrappers if present."""
    text = text.strip()
    # Remove opening fence (```cypher or ```)
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    # Remove closing fence
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
