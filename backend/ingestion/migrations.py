"""Versioned PostgreSQL migrations. Run only with the direct Neon connection."""

import os

from sqlalchemy import create_engine, text


MIGRATIONS = (
    (
        1,
        (
            """CREATE TABLE ingest_documents (
                doc_id uuid PRIMARY KEY,
                owner_sub text NOT NULL,
                idempotency_key text,
                content_sha256 text NOT NULL,
                source_key text,
                file_name text NOT NULL,
                kind text NOT NULL CHECK (kind IN ('pdf', 'docx', 'text')),
                metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                size_bytes bigint NOT NULL CHECK (size_bytes > 0),
                state text NOT NULL CHECK (state IN ('uploading', 'accepted', 'processing', 'done', 'failed')),
                stage text NOT NULL,
                attempt integer NOT NULL DEFAULT 0,
                error_code text,
                validation jsonb,
                lease_until timestamptz,
                expires_at timestamptz,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                completed_at timestamptz
            )""",
            "CREATE UNIQUE INDEX ingest_documents_idempotency ON ingest_documents(owner_sub, idempotency_key) WHERE idempotency_key IS NOT NULL",
            "CREATE INDEX ingest_documents_reconcile ON ingest_documents(state, lease_until, expires_at)",
            """CREATE TABLE ingest_stages (
                doc_id uuid NOT NULL REFERENCES ingest_documents(doc_id) ON DELETE CASCADE,
                stage text NOT NULL,
                state text NOT NULL,
                attempt integer NOT NULL DEFAULT 0,
                error_code text,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (doc_id, stage)
            )""",
            """CREATE TABLE ingest_outbox (
                event_id uuid PRIMARY KEY,
                doc_id uuid NOT NULL REFERENCES ingest_documents(doc_id) ON DELETE CASCADE,
                kind text NOT NULL,
                state text NOT NULL DEFAULT 'pending',
                attempts integer NOT NULL DEFAULT 0,
                next_attempt_at timestamptz NOT NULL DEFAULT now(),
                created_at timestamptz NOT NULL DEFAULT now(),
                UNIQUE (doc_id, kind)
            )""",
            "CREATE INDEX ingest_outbox_pending ON ingest_outbox(state, next_attempt_at)",
        ),
    ),
)


def migrate(url: str | None = None) -> None:
    """Apply each migration atomically; never use a pooled connection here."""
    # Keep the migration entry point independent from application settings so
    # it does not require provider keys or graph credentials just to migrate.
    direct_url = url or os.environ.get("INGEST_MIGRATION_DATABASE_URL")
    if not direct_url or "-pooler" in direct_url.split("@")[-1].split("/")[0]:
        raise ValueError("A direct INGEST_MIGRATION_DATABASE_URL is required")
    engine = create_engine(direct_url, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            connection.execute(text("SELECT pg_advisory_xact_lock(482320)"))
            connection.execute(text("CREATE TABLE IF NOT EXISTS ingest_schema_migrations (version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"))
        for version, statements in MIGRATIONS:
            with engine.begin() as connection:
                connection.execute(text("SELECT pg_advisory_xact_lock(482320)"))
                if connection.execute(text("SELECT 1 FROM ingest_schema_migrations WHERE version=:version"), {"version": version}).scalar():
                    continue
                for statement in statements:
                    connection.execute(text(statement))
                connection.execute(text("INSERT INTO ingest_schema_migrations(version) VALUES (:version)"), {"version": version})
    finally:
        engine.dispose()


if __name__ == "__main__":
    migrate()
