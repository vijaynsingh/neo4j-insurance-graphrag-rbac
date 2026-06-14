import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from neo4j.exceptions import AuthError, ServiceUnavailable
from pydantic import BaseModel

from app.config import OPENAI_API_KEY
from app.graph import get_driver, run_query
from app.graphrag_pipeline import GraphRAGPipeline

STATIC_DIR = Path(__file__).parent.parent / "static"

VALID_MODES = {"demo", "openai", "text2cypher", "auto"}


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    mode: str = "demo"


class RetrievalSummary(BaseModel):
    matched_chunks: int
    rules: int
    risk_factors: int
    policies: int
    applicants: int


class AskResponse(BaseModel):
    # Core answer — original contract, unchanged
    question: str
    decision: str
    reasoning: list[str]
    supporting_rules: list[dict]
    risk_factors: list[dict]
    citations: list[dict]
    retrieval_summary: RetrievalSummary
    # Extended retrieval detail — added in Step 9 for UI visualization
    matched_chunks: list[dict] = []
    graph_context: dict = {}
    # Step 10 — mode and provider metadata
    mode: str = "demo"
    embedding_provider: str = "mock"
    llm_provider: str = "MockLLM"
    compatibility_warning: str | None = None
    # True when embeddings were automatically re-indexed for this request's mode
    reindexed: bool = False
    # Text2Cypher-specific fields — None / empty for Learning Mode and OpenAI Mode
    generated_cypher:    str | None  = None
    raw_query_results:   list[dict]  = []
    retrieval_strategy:  str | None  = None
    # Auto Mode fields — None for all other modes
    selected_strategy:   str | None  = None
    router_reason:       str | None  = None


# ------------------------------------------------------------------
# Auto-reseed helper
# ------------------------------------------------------------------

async def _auto_reseed_if_needed(
    pipeline: GraphRAGPipeline,
    driver,
    lock: asyncio.Lock,
) -> bool:
    """
    Compares the embedding model stored on DocumentChunk nodes with the model
    the active pipeline's provider would produce.  If they differ, re-embeds
    all DocumentChunk nodes using the pipeline's provider — no graph structure
    changes, only the embedding vectors and metadata.

    Returns True  if reseeding was performed successfully.
    Returns False if no reseed was needed, or if reseeding failed.

    The asyncio.Lock prevents concurrent requests from triggering a double-reseed.
    Each reseed operation runs in a thread pool so the async event loop is not blocked.
    """
    def _stored_model() -> str | None:
        rows = run_query(
            driver,
            "MATCH (d:DocumentChunk) WHERE d.embedding_model IS NOT NULL "
            "RETURN d.embedding_model AS model LIMIT 1",
        )
        return rows[0]["model"] if rows else None

    stored = await asyncio.to_thread(_stored_model)
    target = pipeline.retriever.embedding_provider.model_name

    if stored is None or stored == target:
        return False  # already consistent, nothing to do

    async with lock:
        # Re-check inside the lock: a concurrent request may have reseeded already
        stored = await asyncio.to_thread(_stored_model)
        if stored == target:
            return False  # done by a concurrent request, not by us

        try:
            from app.seed import reindex_embeddings
            await asyncio.to_thread(
                reindex_embeddings, driver, pipeline.retriever.embedding_provider
            )
            return True
        except Exception as exc:
            print(f"[auto-reseed] failed: {exc}")
            return False


# ------------------------------------------------------------------
# Lifespan: open the Neo4j driver once and reuse it across requests
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    driver = get_driver()
    app.state.driver = driver
    app.state.reseed_lock = asyncio.Lock()

    pipelines: dict[str, GraphRAGPipeline] = {
        "demo": GraphRAGPipeline.for_mode(driver, "demo"),
    }
    if OPENAI_API_KEY:
        pipelines["openai"] = GraphRAGPipeline.for_mode(driver, "openai")

    app.state.pipelines = pipelines

    if OPENAI_API_KEY:
        from app.text2cypher_service import Text2CypherService
        app.state.t2c_service = Text2CypherService(driver)
    else:
        app.state.t2c_service = None

    if OPENAI_API_KEY:
        from app.retrieval_router import RetrievalRouter, HybridSynthesizer
        app.state.auto_router     = RetrievalRouter()
        app.state.auto_synthesizer = HybridSynthesizer()
    else:
        app.state.auto_router      = None
        app.state.auto_synthesizer = None

    yield
    driver.close()


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="Neo4j Insurance GraphRAG",
    description="Insurance underwriting question-answering via graph-augmented retrieval.",
    version="0.10.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "neo4j-insurance-graphrag"}


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be blank")

    mode = body.mode
    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of: {sorted(VALID_MODES)}",
        )

    # ── Text2Cypher mode ────────────────────────────────────────────────────
    if mode == "text2cypher":
        t2c = request.app.state.t2c_service
        if t2c is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "mode 'text2cypher' is not available — "
                    "set OPENAI_API_KEY in .env and restart the server to enable it"
                ),
            )
        try:
            t2c_result = await asyncio.to_thread(t2c.run, question)
        except (ServiceUnavailable, AuthError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Neo4j unavailable — is Docker running? ({exc})",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return AskResponse(
            question=question,
            decision="ANSWERED",
            reasoning=t2c_result["reasoning"],
            supporting_rules=[],
            risk_factors=[],
            citations=[],
            retrieval_summary=RetrievalSummary(
                matched_chunks=0,
                rules=0,
                risk_factors=0,
                policies=0,
                applicants=0,
            ),
            matched_chunks=[],
            graph_context={},
            mode="text2cypher",
            embedding_provider="none",
            llm_provider="OpenAILLM",
            compatibility_warning=None,
            reindexed=False,
            generated_cypher=t2c_result["generated_cypher"],
            raw_query_results=t2c_result["raw_query_results"],
            retrieval_strategy="Text2Cypher",
        )

    # ── Auto Mode (router → openai_graph | text2cypher | hybrid) ───────────
    if mode == "auto":
        router     = request.app.state.auto_router
        synthesizer = request.app.state.auto_synthesizer
        t2c        = request.app.state.t2c_service
        pipelines_map: dict = request.app.state.pipelines
        openai_pipeline = pipelines_map.get("openai")
        driver     = request.app.state.driver
        lock: asyncio.Lock = request.app.state.reseed_lock

        if router is None or openai_pipeline is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "mode 'auto' is not available — "
                    "set OPENAI_API_KEY in .env and restart the server to enable it"
                ),
            )

        # Step 1 — classify the question
        try:
            route = await asyncio.to_thread(router.classify, question)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Router failed: {exc}")

        strategy     = route["selected_strategy"]
        router_reason = route["router_reason"]
        print(f"[auto] strategy={strategy!r}  reason={router_reason!r}")

        # Step 2a — openai_graph path
        if strategy == "openai_graph":
            reseeded = await _auto_reseed_if_needed(openai_pipeline, driver, lock)
            try:
                result = await asyncio.to_thread(
                    lambda: openai_pipeline.run(question, check_compatibility=not reseeded)
                )
            except (ServiceUnavailable, AuthError) as exc:
                raise HTTPException(status_code=503, detail=f"Neo4j unavailable — {exc}")
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
            ctx = result["context"]
            ans = result["answer"]
            return AskResponse(
                question=question,
                decision=ans["decision"],
                reasoning=ans["reasoning"],
                supporting_rules=ans["supporting_rules"],
                risk_factors=ans["risk_factors"],
                citations=ans["citations"],
                retrieval_summary=RetrievalSummary(
                    matched_chunks=len(ctx["matched_chunks"]),
                    rules=len(ctx["rules"]),
                    risk_factors=len(ctx["risk_factors"]),
                    policies=len(ctx["policies"]),
                    applicants=len(ctx["applicants"]),
                ),
                matched_chunks=ctx["matched_chunks"],
                graph_context={
                    "rules": ctx["rules"], "risk_factors": ctx["risk_factors"],
                    "policies": ctx["policies"], "applicants": ctx["applicants"],
                },
                mode="auto",
                embedding_provider=openai_pipeline.retriever.embedding_provider.model_name,
                llm_provider=type(openai_pipeline.llm).__name__,
                compatibility_warning=ctx.get("compatibility_warning"),
                reindexed=reseeded,
                retrieval_strategy="openai_graph",
                selected_strategy="openai_graph",
                router_reason=router_reason,
            )

        # Step 2b — text2cypher path
        if strategy == "text2cypher":
            if t2c is None:
                raise HTTPException(
                    status_code=400,
                    detail="text2cypher service not available — OPENAI_API_KEY required",
                )
            try:
                t2c_result = await asyncio.to_thread(t2c.run, question)
            except (ServiceUnavailable, AuthError) as exc:
                raise HTTPException(status_code=503, detail=f"Neo4j unavailable — {exc}")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
            return AskResponse(
                question=question,
                decision="ANSWERED",
                reasoning=t2c_result["reasoning"],
                supporting_rules=[],
                risk_factors=[],
                citations=[],
                retrieval_summary=RetrievalSummary(
                    matched_chunks=0, rules=0, risk_factors=0, policies=0, applicants=0,
                ),
                matched_chunks=[],
                graph_context={},
                mode="auto",
                embedding_provider="none",
                llm_provider="OpenAILLM",
                compatibility_warning=None,
                reindexed=False,
                generated_cypher=t2c_result["generated_cypher"],
                raw_query_results=t2c_result["raw_query_results"],
                retrieval_strategy="text2cypher",
                selected_strategy="text2cypher",
                router_reason=router_reason,
            )

        # Step 2c — hybrid path (run both in parallel, synthesize)
        reseeded = await _auto_reseed_if_needed(openai_pipeline, driver, lock)

        graphrag_coro = asyncio.to_thread(
            lambda: openai_pipeline.run(question, check_compatibility=not reseeded)
        )
        if t2c is not None:
            t2c_coro = asyncio.to_thread(t2c.run, question)
            graphrag_r, t2c_r = await asyncio.gather(
                graphrag_coro, t2c_coro, return_exceptions=True
            )
        else:
            graphrag_r = await graphrag_coro
            t2c_r = RuntimeError("Text2Cypher service not available.")

        # GraphRAG failure is fatal for hybrid
        if isinstance(graphrag_r, BaseException):
            raise HTTPException(
                status_code=500,
                detail=f"GraphRAG pipeline failed in hybrid mode: {graphrag_r}",
            )

        # Text2Cypher failure is tolerated
        t2c_result = None
        t2c_note = ""
        if isinstance(t2c_r, BaseException):
            t2c_note = f" Text2Cypher failed: {t2c_r}."
            print(f"[auto/hybrid] Text2Cypher failed, continuing with GraphRAG only: {t2c_r}")
        else:
            t2c_result = t2c_r

        # Synthesize
        try:
            synthesis = await asyncio.to_thread(
                synthesizer.synthesize, question, graphrag_r, t2c_result
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Hybrid synthesis failed: {exc}")

        ctx  = graphrag_r["context"]
        ans  = graphrag_r["answer"]
        return AskResponse(
            question=question,
            decision=synthesis.get("decision", "ANSWERED"),
            reasoning=synthesis["reasoning"],
            supporting_rules=ans["supporting_rules"],
            risk_factors=ans["risk_factors"],
            citations=ans["citations"],
            retrieval_summary=RetrievalSummary(
                matched_chunks=len(ctx["matched_chunks"]),
                rules=len(ctx["rules"]),
                risk_factors=len(ctx["risk_factors"]),
                policies=len(ctx["policies"]),
                applicants=len(ctx["applicants"]),
            ),
            matched_chunks=ctx["matched_chunks"],
            graph_context={
                "rules": ctx["rules"], "risk_factors": ctx["risk_factors"],
                "policies": ctx["policies"], "applicants": ctx["applicants"],
            },
            mode="auto",
            embedding_provider=openai_pipeline.retriever.embedding_provider.model_name,
            llm_provider="OpenAILLM",
            compatibility_warning=ctx.get("compatibility_warning"),
            reindexed=reseeded,
            generated_cypher=t2c_result["generated_cypher"] if t2c_result else None,
            raw_query_results=t2c_result["raw_query_results"] if t2c_result else [],
            retrieval_strategy="Hybrid",
            selected_strategy="hybrid",
            router_reason=router_reason + t2c_note,
        )

    # ── GraphRAG modes (demo / openai) ──────────────────────────────────────
    pipelines: dict[str, GraphRAGPipeline] = request.app.state.pipelines
    if mode not in pipelines:
        raise HTTPException(
            status_code=400,
            detail=(
                f"mode '{mode}' is not available — "
                "set OPENAI_API_KEY in .env and restart the server to enable it"
            ),
        )

    pipeline = pipelines[mode]
    driver = request.app.state.driver
    lock: asyncio.Lock = request.app.state.reseed_lock

    # If the stored embeddings don't match the requested mode's provider,
    # re-embed all DocumentChunk nodes before running the query.
    # The first request after a mode switch is slightly slower; subsequent ones are fast.
    reseeded = await _auto_reseed_if_needed(pipeline, driver, lock)

    try:
        # Skip the in-query compatibility check when we just reseeded (saves a DB round-trip)
        result = pipeline.run(question, check_compatibility=not reseeded)
    except (ServiceUnavailable, AuthError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j unavailable — is Docker running? ({exc})",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    context = result["context"]
    answer = result["answer"]

    return AskResponse(
        # Original fields
        question=question,
        decision=answer["decision"],
        reasoning=answer["reasoning"],
        supporting_rules=answer["supporting_rules"],
        risk_factors=answer["risk_factors"],
        citations=answer["citations"],
        retrieval_summary=RetrievalSummary(
            matched_chunks=len(context["matched_chunks"]),
            rules=len(context["rules"]),
            risk_factors=len(context["risk_factors"]),
            policies=len(context["policies"]),
            applicants=len(context["applicants"]),
        ),
        # Extended retrieval detail for UI
        matched_chunks=context["matched_chunks"],
        graph_context={
            "rules":        context["rules"],
            "risk_factors": context["risk_factors"],
            "policies":     context["policies"],
            "applicants":   context["applicants"],
        },
        # Step 10 — mode and provider metadata
        mode=mode,
        embedding_provider=pipeline.retriever.embedding_provider.model_name,
        llm_provider=type(pipeline.llm).__name__,
        compatibility_warning=context.get("compatibility_warning"),
        reindexed=reseeded,
    )
