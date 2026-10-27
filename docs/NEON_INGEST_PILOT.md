# Neon-backed ingestion pilot deployment

This deployment profile is a single-workspace, single-host pilot. It is not host-loss resilient: successful source files are deleted, and Chroma/Neo4j volumes have no backup in this release.

## Required inputs

1. A Neon project and branch with a pooled Postgres URL, a direct migration URL, and a **private** Object Storage bucket on that same branch. Supply its S3 endpoint, region, and branch-scoped S3 credentials. The S3 client uses SigV4 and path-style addressing. Do not commit actual values.
2. A public DNS hostname pointing to the Compose host, reachable on TCP 443 for HTTPS certificate issuance. Supply an existing OIDC issuer, audience, HTTPS JWKS URL, client ID/secret, cookie secret, group-claim name, and allowed group. Verify the group appears in both browser session and API access tokens.
3. A host sized for the API, parser, Redis, Chroma, Neo4j, two ingestion workers and one control worker. Confirm free disk capacity for local indexes and temporary 100 MB uploads; the runbook does not substitute for host sizing or backup.
4. A private `.env` with the names in `.env.example` populated. Keep the environment file outside version control; never paste credentials into issues, logs, or `tasks/`.

## Release path

Run the direct migration against a disposable Neon branch first. Verify the schema and rollback strategy before applying it to the production branch. Use `infra/docker-compose.prod.yml` by itself; **do not combine it with the development Compose file**. The production profile publishes only 443 and contains no source-code bind mounts or Uvicorn reload. Its Caddy edge routes API traffic through OAuth2 Proxy, and FastAPI checks the allowed group again for API calls.

The API streams at most 100 MiB into a temporary file, verifies PDF/DOCX/TXT form, reserves active-source capacity, writes the private object, and returns `202` with `doc_id` and a status URL. Postgres holds status and outbox intent even if Redis is unavailable. Celery Beat dispatches and reconciles jobs. Two ingest worker processes cap active processing; the control queue remains separate. Successful originals are deleted; failed originals expire after 24 hours. Direct browser-to-Neon multipart is **not enabled**.

Entity extraction requires a valid response containing both entity and relationship arrays. Explicit empty arrays are allowed; provider failures and malformed responses enter the bounded task retry path. If retries are exhausted, durable status is `failed` with `error_code=extraction_failed`, and the source remains subject to failed-source retention. Chroma or lexical writes may already have occurred before extraction fails, so the terminal failure does not imply rollback or absence of partial indexed data.

## Verification still required before public release

- Run migration and API smoke on a disposable Neon branch; verify the bucket is private and that Put/Get/Delete work with the actual branch endpoint. Confirm source cleanup and quota reconciliation.
- Check OIDC cookie and bearer paths with an allowed-group user, denied-group user, expired token, wrong issuer, and wrong audience. Check Caddy strips incoming identity headers. Confirm cross-origin clients use an explicitly allowed CORS origin.
- Test a PDF at 100 MiB and one byte over, a 500-page PDF and 501-page PDF, representative DOCX/TXT files, malformed archives, and failed-source expiry. DOCX does not yet have a trustworthy page count: keep the universal 500-page claim open until a conversion/page-verification path exists.
- Kill a worker before/after Chroma and Neo4j writes; restart Redis/API/worker; check one truthful terminal status and no duplicate chunks/edges. Test two concurrent jobs and storage near the 4 GiB admission high-water mark.
- Inject provider and malformed extraction responses after the vector write; verify retry, exhausted `extraction_failed` status, source retention, partial-index behavior and successful recovery against the real services.
- Verify HTTP/TLS, Caddy upload limit, OAuth2 Proxy flags, and Compose health on a real host. The local development machine had no Docker daemon during implementation.

The current code has deterministic tests and a production profile, but the live checks above are not yet evidence of completion. Signed webhooks, separate extraction queue/chord, direct multipart, eval jobs, and community rebuild remain later increments.
