from dotenv import load_dotenv
import os

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# OpenAI — all optional; set in .env to enable real embeddings / LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
USE_OPENAI_LLM = os.getenv("USE_OPENAI_LLM", "false").lower() == "true"
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")

# RBAC — maps API role names → Neo4j usernames created by rbac_setup.py.
# The default role (underwriting_manager) gives full access, so existing
# callers that omit the role field see unchanged behaviour.
RBAC_ROLE_MAP: dict[str, str] = {
    "underwriter":          "uw_standard",
    "senior_underwriter":   "uw_senior",
    "underwriting_manager": "uw_manager",
}
RBAC_USER_PASSWORD: str = os.getenv("RBAC_USER_PASSWORD", "demo1234")
RBAC_DEFAULT_ROLE:  str = "underwriting_manager"
