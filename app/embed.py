import hashlib
import math

# Matches OpenAI text-embedding-3-small.
# Changing this constant requires recreating the vector index AND re-seeding embeddings.
EMBEDDING_DIMENSION = 1536


# ------------------------------------------------------------------
# Standalone mock function — kept for backward compatibility
# ------------------------------------------------------------------

def mock_embed(text: str) -> list[float]:
    """
    Deterministic mock embedding — same text always returns the same vector.
    Not semantically meaningful. Used to validate vector index infrastructure
    without any API dependency.

    Algorithm: iteratively SHA-256 hash the input to generate enough bytes,
    map each byte to [-1, 1], then L2-normalize to unit length.
    """
    seed = text.encode()
    raw: list[float] = []
    while len(raw) < EMBEDDING_DIMENSION:
        seed = hashlib.sha256(seed).digest()
        for byte in seed:
            raw.append((byte / 127.5) - 1.0)
    values = raw[:EMBEDDING_DIMENSION]
    magnitude = math.sqrt(sum(v * v for v in values))
    return [v / magnitude for v in values]


# ------------------------------------------------------------------
# Embedding provider abstraction
# ------------------------------------------------------------------

class MockEmbeddingProvider:
    """Wraps mock_embed() in the provider interface. Zero cost, zero API calls."""

    model_name = "mock"

    def embed(self, text: str) -> list[float]:
        return mock_embed(text)


class OpenAIEmbeddingProvider:
    """
    Calls OpenAI's embeddings API.

    IMPORTANT: The model used here must match the model used when seed.py
    stored embeddings on DocumentChunk nodes. If you switch models, re-run
    `python3 -m app.seed` to rebuild stored embeddings before querying.
    """

    def __init__(self):
        from openai import OpenAI
        from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = OPENAI_EMBEDDING_MODEL

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return response.data[0].embedding


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def get_embedding_provider() -> MockEmbeddingProvider | OpenAIEmbeddingProvider:
    """
    Returns the active embedding provider based on environment configuration.

    Rules:
      USE_OPENAI_EMBEDDINGS=true AND OPENAI_API_KEY set → OpenAIEmbeddingProvider
      Otherwise → MockEmbeddingProvider (default, safe fallback)

    The provider used at seed time must match the provider used at query time.
    If you switch providers, re-run `python3 -m app.seed` before querying.
    """
    from app.config import USE_OPENAI_EMBEDDINGS, OPENAI_API_KEY
    if USE_OPENAI_EMBEDDINGS and OPENAI_API_KEY:
        return OpenAIEmbeddingProvider()
    return MockEmbeddingProvider()
