# Security Configuration

## Neo4j credentials

Kinegraph does not ship a Neo4j password. Generate a unique value and keep it
only in the untracked `.env` file or a production secret manager:

```bash
openssl rand -base64 32
docker compose --env-file .env -f infra/docker-compose.yml up -d
```

`infra/docker-compose.yml` fails closed when `NEO4J_PASSWORD` is absent. The
application also requires at least 16 characters.

The historical development password was committed to version control. Removing
it from the current tree does not revoke it or erase Git history. Rotate the
password in every Neo4j deployment that ever used it, update the corresponding
secret store, restart dependent services, and invalidate any derived snapshots
or credentials. Rewriting public Git history is a separate repository-owner
decision and does not replace rotation.

A Google API key-shaped value was committed in `docs/FIX_API_KEY.md` and in
derived copies under `data/documents/` and `data/chunks/`. The working tree
now uses a placeholder. Removing it does not revoke the credential or erase
Git history. If that value was ever issued, revoke it in the Google Cloud or
AI Studio console, rotate any configuration that used it, and treat historical
commits as exposed. Rewriting history remains a repository-owner decision.

## Generated Cypher

The optional PropertyGraph Text-to-Cypher retriever is disabled. Requesting
`use_cypher=True` raises an error before database or model clients are initialized.
It must remain disabled until its execution path enforces validated, bounded
queries and database read-only authority.

The separate `Neo4jService.graph_search` path treats LLM-generated Cypher as
untrusted input. Only a single read-only `MATCH` or `OPTIONAL MATCH` statement
with `RETURN` is accepted. Comments, multiple statements, procedures,
administrative clauses, write clauses, and LLM-provided pagination are rejected.
That path executes validated queries in Neo4j read access mode and applies its
own parameterized result limit.

Production should additionally use a Neo4j account whose database privileges
are read-only for retrieval workloads. Application validation is defense in
depth, not a replacement for database authorization.

## CORS

Allowed browser origins are explicit and comma-separated:

```dotenv
CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
CORS_ALLOW_CREDENTIALS=true
```

Wildcard origins cannot be combined with credentials. The development default
allows only the local UI origins and disables credentialed cross-origin calls.
