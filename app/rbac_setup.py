"""
app/rbac_setup.py
-----------------
Creates Neo4j 5 Enterprise RBAC roles, privilege grants/denies, and demo users
for the three sensitivity tiers (Standard / Restricted / Confidential).

Role hierarchy
  underwriter           → reads :Standard applicants only
  senior_underwriter    → reads :Standard + :Restricted applicants
  underwriting_manager  → reads all three tiers (no restrictions)

All three roles have unrestricted read on every non-Applicant node
(Policy, RiskFactor, LabResult, UnderwritingRule, DocumentChunk) and
on every relationship type.

Enforcement mechanism:
  Each role receives GRANT MATCH {*} ON GRAPH neo4j NODES * (full graph read).
  DENY TRAVERSE + DENY READ on the tier labels (:Restricted, :Confidential)
  then make the denied nodes invisible.  Neo4j evaluates DENY before GRANT
  across all roles simultaneously, so a :Confidential node is not traversable
  even when another role (including PUBLIC) grants NODES *.

Requires: Neo4j 5 Enterprise (RBAC commands are not available on Community).
Run with:  python -m app.rbac_setup
           (safe to re-run — roles/users are created with IF NOT EXISTS)
"""

import sys
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

from app.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

GRAPH = "neo4j"  # the data database that holds the underwriting graph

ROLES = ("underwriter", "senior_underwriter", "underwriting_manager")

# (neo4j_username, password, role_to_grant)
USERS = [
    ("uw_standard", "demo1234", "underwriter"),
    ("uw_senior",   "demo1234", "senior_underwriter"),
    ("uw_manager",  "demo1234", "underwriting_manager"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run(session, cypher: str, label: str) -> None:
    """Execute one system-db statement and print the outcome."""
    try:
        session.run(cypher)
        print(f"  OK    {label}")
    except ClientError as exc:
        # Surface the Neo4j error but keep going.  Most are benign on re-runs
        # (e.g. "already exists") but some may indicate a real issue.
        print(f"  WARN  {label}")
        print(f"        {exc.message}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Roles
# ──────────────────────────────────────────────────────────────────────────────

def setup_roles(session) -> None:
    print("\n── Step 1: Roles ───────────────────────────────────────────────")
    for role in ROLES:
        _run(session,
             f"CREATE ROLE {role} IF NOT EXISTS",
             f"CREATE ROLE {role}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Grants (all three roles — full graph read)
# ──────────────────────────────────────────────────────────────────────────────

def setup_grants(session) -> None:
    """
    Each role starts with unrestricted read on the entire neo4j graph.
    DENY statements in the next step selectively suppress tier labels.
    """
    print("\n── Step 2: Grants (all roles — full read) ──────────────────────")

    for role in ROLES:
        # Allow users in this role to connect to the data database
        _run(session,
             f"GRANT ACCESS ON DATABASE {GRAPH} TO {role}",
             f"GRANT ACCESS on database '{GRAPH}' → {role}")

        # MATCH = TRAVERSE + READ {*} combined; grants both visibility and property access
        _run(session,
             f"GRANT MATCH {{*}} ON GRAPH {GRAPH} NODES * TO {role}",
             f"GRANT MATCH {{*}} NODES * → {role}")

        # Relationships: TRAVERSE lets the query engine follow edges;
        # READ {*} exposes relationship properties
        _run(session,
             f"GRANT TRAVERSE ON GRAPH {GRAPH} RELATIONSHIPS * TO {role}",
             f"GRANT TRAVERSE RELATIONSHIPS * → {role}")

        _run(session,
             f"GRANT READ {{*}} ON GRAPH {GRAPH} RELATIONSHIPS * TO {role}",
             f"GRANT READ {{*}} RELATIONSHIPS * → {role}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Denies (tier-label suppression per role)
# ──────────────────────────────────────────────────────────────────────────────

def setup_denies(session) -> None:
    """
    DENY TRAVERSE makes a node completely invisible to MATCH patterns — the
    traversal engine skips it as if it does not exist.  DENY READ {*} removes
    property access as a belt-and-suspenders measure.  Both DENY statements
    target the tier label, not the shared :Applicant label, so Policy /
    RiskFactor / etc. nodes (which do not carry tier labels) are unaffected.

    A node with labels :Applicant:Confidential is blocked by DENY on
    :Confidential regardless of the GRANT on NODES * — DENY always wins.
    """
    print("\n── Step 3: Denies (tier suppression) ───────────────────────────")

    # underwriter: Standard only → deny Restricted and Confidential
    for tier_label in ("Restricted", "Confidential"):
        _run(session,
             f"DENY TRAVERSE ON GRAPH {GRAPH} NODES {tier_label} TO underwriter",
             f"DENY TRAVERSE :{tier_label} → underwriter")
        _run(session,
             f"DENY READ {{*}} ON GRAPH {GRAPH} NODES {tier_label} TO underwriter",
             f"DENY READ {{*}} :{tier_label} → underwriter")

    # senior_underwriter: Standard + Restricted → deny Confidential only
    _run(session,
         f"DENY TRAVERSE ON GRAPH {GRAPH} NODES Confidential TO senior_underwriter",
         "DENY TRAVERSE :Confidential → senior_underwriter")
    _run(session,
         f"DENY READ {{*}} ON GRAPH {GRAPH} NODES Confidential TO senior_underwriter",
         "DENY READ {*} :Confidential → senior_underwriter")

    # underwriting_manager: no denies — all three tiers visible


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Users
# ──────────────────────────────────────────────────────────────────────────────

def setup_users(session) -> None:
    print("\n── Step 4: Users ───────────────────────────────────────────────")
    for username, password, role in USERS:
        _run(session,
             f"CREATE USER {username} IF NOT EXISTS "
             f"SET PASSWORD '{password}' "
             f"SET PASSWORD CHANGE NOT REQUIRED",
             f"CREATE USER {username}  (password: demo, role: {role})")
        _run(session,
             f"GRANT ROLE {role} TO {username}",
             f"GRANT ROLE {role} → {username}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Connecting to {NEO4J_URI} as '{NEO4J_USERNAME}'...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()
        print("Connected.")
    except Exception as exc:
        print(f"ERROR: Cannot connect to Neo4j — {exc}")
        print("Is Docker running?  docker compose up -d")
        sys.exit(1)

    # RBAC commands run against the 'system' database, never the data graph.
    with driver.session(database="system") as session:
        setup_roles(session)
        setup_grants(session)
        setup_denies(session)
        setup_users(session)

    driver.close()

    print("\n" + "─" * 60)
    print("RBAC setup complete.\n")
    print("Expected applicant visibility per user:")
    print("  uw_standard  →  John Smith, Maria Garcia                (Standard only)")
    print("  uw_senior    →  + Patricia Williams, Robert Chen         (+ Restricted)")
    print("  uw_manager   →  + James Hartford, Victoria Ashworth      (+ Confidential)")
    print("\nVerify with (see instructions below):")
    print("  docker exec -it neo4j-insurance-graphrag-rbac \\")
    print("    cypher-shell -u uw_standard -p demo -d neo4j \\")
    print('    "MATCH (a:Applicant) RETURN a.name, a.sensitivity ORDER BY a.name;"')
    print("\n(demo passwords are 'demo1234' — Neo4j requires ≥ 8 characters)")


if __name__ == "__main__":
    main()
