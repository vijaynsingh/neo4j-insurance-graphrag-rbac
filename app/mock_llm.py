class MockLLM:
    """
    Deterministic mock LLM that applies hard-coded underwriting business logic
    to graph context assembled by GraphRetriever.

    No external API calls. Validates the full pipeline end-to-end before
    a real LLM is introduced. The business logic is intentionally explicit
    so the reasoning trace is fully auditable — a design property worth keeping
    even with a real LLM (citations + supporting_rules).
    """

    def generate_answer(self, question: str, context: dict) -> dict:
        risk_factor_names = {rf["name"].lower() for rf in context.get("risk_factors", [])}
        rules = context.get("rules", [])
        chunks = context.get("matched_chunks", [])

        has_diabetes = any("diabetes" in n for n in risk_factor_names)
        has_controlled_a1c = any("a1c" in n for n in risk_factor_names)
        has_no_tobacco = any("tobacco" in n for n in risk_factor_names)

        # Core decision logic
        if has_diabetes and has_controlled_a1c:
            decision = "REFER_FOR_REVIEW"
            reasoning = [
                "Type 2 Diabetes is present in the applicant's risk profile — a chronic "
                "condition that triggers mandatory underwriting review.",
                "A1C is controlled (below 7.0 threshold) — the condition is actively managed, "
                "which favourably adjusts the severity assessment.",
                "Preferred class requires underwriting review for any chronic condition, even "
                "when controlled. Controlled status reduces but does not eliminate the referral "
                "requirement.",
            ]
            if has_no_tobacco:
                reasoning.append(
                    "No tobacco use is recorded, which provides a favourable lifestyle "
                    "adjustment to the overall risk profile."
                )
        elif has_diabetes:
            decision = "REQUIRE_ADDITIONAL_REVIEW"
            reasoning = [
                "Type 2 Diabetes is present in the applicant's risk profile.",
                "A1C control status is not confirmed — additional lab evidence is required "
                "before Preferred class can be assessed.",
                "Without confirmed A1C control, the chronic condition remains unqualified "
                "and triggers a full additional review.",
            ]
        else:
            decision = "APPROVE"
            reasoning = [
                "No disqualifying chronic conditions found in the retrieved risk profile.",
                "No tobacco use recorded — applicant qualifies for non-tobacco rate class.",
                "Applicant appears eligible for Preferred class pending full underwriting review.",
            ]

        supporting_rules = [
            {"id": r["id"], "title": r["title"], "decision": r["decision"]}
            for r in rules
        ]

        citations = []
        for chunk in chunks:
            citations.append({
                "type": "DocumentChunk",
                "source": chunk.get("source", ""),
                "relevance_score": round(chunk.get("score", 0.0), 4),
            })
        for rule in rules:
            citations.append({
                "type": "UnderwritingRule",
                "title": rule.get("title", ""),
                "decision": rule.get("decision", ""),
            })

        return {
            "decision": decision,
            "reasoning": reasoning,
            "supporting_rules": supporting_rules,
            "risk_factors": list(context.get("risk_factors", [])),
            "citations": citations,
        }
