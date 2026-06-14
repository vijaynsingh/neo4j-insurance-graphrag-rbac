import json

from app.config import OPENAI_API_KEY, OPENAI_LLM_MODEL

_SYSTEM_PROMPT = """\
You are an insurance underwriting assistant.

You receive structured context retrieved from an insurance knowledge graph: underwriting rules,
risk factors, policies, and applicant data. Analyse the context and return a JSON underwriting
recommendation.

Guidelines:
- Base your answer ONLY on the provided context. Do not invent facts not present in the context.
- If the context is insufficient to make a confident decision, use REQUIRE_ADDITIONAL_REVIEW.
- reasoning: ordered plain-English steps that explain the decision.
- supporting_rules: only rules from the context that directly drove the decision.
- risk_factors: only factors from the context that influenced the decision.
- citations: include every DocumentChunk source and UnderwritingRule title that supports the answer.

Return valid JSON only. No prose outside the JSON object.

Required JSON shape — every field is required, every nested object must match exactly:
{
  "decision": "APPROVE | REFER_FOR_REVIEW | REQUIRE_ADDITIONAL_REVIEW | DECLINE",
  "reasoning": ["step 1 ...", "step 2 ..."],
  "supporting_rules": [
    {"id": "<rule id from context>", "title": "<rule title>", "decision": "<rule decision enum>"}
  ],
  "risk_factors": [
    {"name": "<factor name from context>", "category": "<factor category from context>"}
  ],
  "citations": [
    {"type": "DocumentChunk",    "source": "<manual section title>", "relevance_score": 0.0},
    {"type": "UnderwritingRule", "title": "<rule title>",            "decision": "<rule decision enum>"}
  ]
}
"""


def _format_context(context: dict) -> str:
    lines: list[str] = []

    lines.append("=== Matched Document Chunks ===")
    for c in context.get("matched_chunks", []):
        lines.append(f"  Source : {c['source']} (similarity score: {c['score']:.4f})")
        lines.append(f"  Text   : {c['text']}")

    lines.append("\n=== Underwriting Rules ===")
    for r in context.get("rules", []):
        lines.append(f"  ID       : {r['id']}")
        lines.append(f"  Title    : {r['title']}")
        lines.append(f"  Rule text: {r['text']}")
        lines.append(f"  Decision : {r['decision']}")

    lines.append("\n=== Risk Factors ===")
    for rf in context.get("risk_factors", []):
        lines.append(f"  Name    : {rf['name']}")
        lines.append(f"  Category: {rf['category']}")

    lines.append("\n=== Policies ===")
    for p in context.get("policies", []):
        lines.append(f"  {p['name']}  (type: {p.get('type', 'n/a')})")

    lines.append("\n=== Applicants ===")
    for a in context.get("applicants", []):
        lines.append(f"  {a['name']}, age {a['age']}")

    return "\n".join(lines)


def _normalize(parsed: dict, context: dict) -> dict:
    """
    Converts GPT output into the exact dict shapes that AskResponse expects,
    matching MockLLM's output contract regardless of how GPT formatted each field.
    Uses context as a lookup table so string references resolve to full objects.
    """
    rules_by_title = {r["title"]: r for r in context.get("rules", [])}
    rfs_by_name    = {rf["name"]: rf for rf in context.get("risk_factors", [])}
    chunks_by_src  = {c["source"]: c for c in context.get("matched_chunks", [])}

    # supporting_rules — each item must be {"id", "title", "decision"}
    supporting_rules = []
    for item in parsed.get("supporting_rules", []):
        if isinstance(item, dict):
            supporting_rules.append(item)
        else:
            rule = rules_by_title.get(str(item), {})
            supporting_rules.append({
                "id":       rule.get("id", ""),
                "title":    rule.get("title", str(item)),
                "decision": rule.get("decision", ""),
            })

    # risk_factors — each item must be {"name", "category"}
    risk_factors = []
    for item in parsed.get("risk_factors", []):
        if isinstance(item, dict):
            risk_factors.append(item)
        else:
            rf = rfs_by_name.get(str(item), {})
            risk_factors.append({
                "name":     rf.get("name", str(item)),
                "category": rf.get("category", ""),
            })

    # citations — each item must be {"type", ...}
    citations = []
    for item in parsed.get("citations", []):
        if isinstance(item, dict):
            citations.append(item)
        else:
            s = str(item)
            chunk = chunks_by_src.get(s)
            if chunk:
                citations.append({
                    "type":            "DocumentChunk",
                    "source":          chunk["source"],
                    "relevance_score": round(chunk.get("score", 0.0), 4),
                })
            else:
                rule = rules_by_title.get(s)
                if rule:
                    citations.append({
                        "type":     "UnderwritingRule",
                        "title":    rule["title"],
                        "decision": rule["decision"],
                    })
                else:
                    citations.append({
                        "type":            "DocumentChunk",
                        "source":          s,
                        "relevance_score": 0.0,
                    })

    return {
        "decision":        parsed.get("decision", "REQUIRE_ADDITIONAL_REVIEW"),
        "reasoning":       parsed.get("reasoning", []),
        "supporting_rules": supporting_rules,
        "risk_factors":    risk_factors,
        "citations":       citations,
    }


class OpenAILLM:
    """
    Calls the OpenAI chat completions API with JSON mode enabled.

    Model is set by OPENAI_LLM_MODEL (default: gpt-4o).
    The model receives structured graph context — not raw text — so it
    reasons over explicit entities rather than inferring them from passages.
    """

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_LLM_MODEL

    def generate_answer(self, question: str, context: dict) -> dict:
        user_prompt = (
            f"Question: {question}\n\n"
            f"Graph context:\n{_format_context(context)}\n\n"
            "Return a JSON underwriting recommendation following the required shape exactly."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        return _normalize(parsed, context)
