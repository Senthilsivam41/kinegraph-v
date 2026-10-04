"""Postgres is the source of truth; Celery result state is never authoritative."""

from functools import lru_cache
from uuid import uuid4

from sqlalchemy import create_engine, text

from backend.core.config import settings


class AdmissionConflict(ValueError):
    pass


class AdmissionCapacity(ValueError):
    pass


@lru_cache(maxsize=1)
def _engine():
    if not settings.INGEST_DATABASE_URL:
        raise RuntimeError("INGEST_DATABASE_URL is required for durable ingestion")
    return create_engine(settings.INGEST_DATABASE_URL, pool_pre_ping=True, pool_size=3, max_overflow=2)


def reserve_document(*, owner_sub: str, idempotency_key: str | None, digest: str, file_name: str, kind: str, size_bytes: int, metadata: dict | None = None) -> tuple[str, bool]:
    """Reserve quota and admission atomically, including concurrent retries."""
    if not owner_sub or size_bytes <= 0:
        raise ValueError("Owner and non-empty document are required")
    doc_id = str(uuid4())
    with _engine().begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(482319)"))
        if idempotency_key:
            previous = connection.execute(text("""SELECT doc_id, content_sha256 FROM ingest_documents
                WHERE owner_sub=:owner AND idempotency_key=:key"""), {"owner": owner_sub, "key": idempotency_key}).mappings().first()
            if previous:
                if previous["content_sha256"] != digest:
                    raise AdmissionConflict("Idempotency key already used for different content")
                return str(previous["doc_id"]), False
        used = connection.execute(text("SELECT COALESCE(SUM(size_bytes), 0) FROM ingest_documents WHERE source_key IS NOT NULL" )).scalar_one()
        if used + size_bytes > settings.INGEST_ACTIVE_OBJECT_LIMIT_BYTES:
            raise AdmissionCapacity("Active upload storage limit reached")
        recent = connection.execute(text("SELECT count(*) FROM ingest_documents WHERE created_at > now() - interval '24 hours'" )).scalar_one()
        if recent >= 100:
            raise AdmissionCapacity("Daily admission limit reached")
        from json import dumps

        connection.execute(text("""INSERT INTO ingest_documents
            (doc_id, owner_sub, idempotency_key, content_sha256, source_key, file_name, kind, size_bytes, metadata, state, stage)
            VALUES (CAST(:id AS uuid), :owner, :key, :digest, :source, :name, :kind, :size, CAST(:metadata AS jsonb), 'uploading', 'uploading')"""),
            {"id": doc_id, "owner": owner_sub, "key": idempotency_key, "digest": digest,
             "source": f"uploads/{doc_id}", "name": file_name, "kind": kind, "size": size_bytes,
             "metadata": dumps(metadata or {})})
    return doc_id, True


def mark_accepted(doc_id: str) -> None:
    """Commit user-visible admission and dispatch intent in one transaction."""
    with _engine().begin() as connection:
        changed = connection.execute(text("""UPDATE ingest_documents SET state='accepted', stage='queued', updated_at=now()
            WHERE doc_id=CAST(:id AS uuid) AND state='uploading'"""), {"id": doc_id})
        if changed.rowcount != 1:
            raise ValueError("Document is not awaiting upload completion")
        connection.execute(text("""INSERT INTO ingest_outbox(event_id, doc_id, kind)
            VALUES (CAST(:event AS uuid), CAST(:id AS uuid), 'process') ON CONFLICT (doc_id, kind) DO NOTHING"""),
            {"event": str(uuid4()), "id": doc_id})


def get_document(doc_id: str, owner_sub: str | None = None) -> dict | None:
    with _engine().connect() as connection:
        row = connection.execute(text("""SELECT doc_id, owner_sub, file_name, kind, size_bytes, state, stage,
            error_code, validation, source_key, created_at, updated_at, completed_at
            FROM ingest_documents WHERE doc_id=CAST(:id AS uuid) AND (:owner IS NULL OR owner_sub=:owner)"""),
            {"id": doc_id, "owner": owner_sub}).mappings().first()
        if not row:
            return None
        result = dict(row)
        result["doc_id"] = str(result["doc_id"])
        result["stages"] = [dict(item) for item in connection.execute(text("""SELECT stage, state, attempt, error_code, updated_at
            FROM ingest_stages WHERE doc_id=CAST(:id AS uuid) ORDER BY updated_at"""), {"id": doc_id}).mappings()]
        return result


def claim_document(doc_id: str) -> dict | None:
    with _engine().begin() as connection:
        row = connection.execute(text("""UPDATE ingest_documents SET state='processing', stage='parsing',
            attempt=attempt+1, lease_until=now()+interval '2 hours', updated_at=now()
            WHERE doc_id=CAST(:id AS uuid) AND state IN ('accepted', 'processing')
              AND (lease_until IS NULL OR lease_until < now())
            RETURNING doc_id, file_name, kind, source_key, content_sha256, metadata"""), {"id": doc_id}).mappings().first()
        return dict(row) if row else None


def set_stage(doc_id: str, stage: str, state: str = "running", error_code: str | None = None) -> None:
    if stage not in {"parsing", "chunking", "embedding", "extracting", "graph", "finalizing"}:
        raise ValueError("Unknown ingestion stage")
    with _engine().begin() as connection:
        connection.execute(text("""INSERT INTO ingest_stages(doc_id, stage, state, attempt, error_code)
            VALUES (CAST(:id AS uuid), :stage, :state, 1, :error)
            ON CONFLICT (doc_id, stage) DO UPDATE SET state=excluded.state,
            attempt=ingest_stages.attempt+1, error_code=excluded.error_code, updated_at=now()"""),
            {"id": doc_id, "stage": stage, "state": state, "error": error_code})
        connection.execute(text("""UPDATE ingest_documents SET stage=:stage,
            lease_until=now()+interval '2 hours', updated_at=now() WHERE doc_id=CAST(:id AS uuid) AND state='processing'"""),
            {"id": doc_id, "stage": stage})


def finish_document(doc_id: str, *, validation: dict, success: bool, error_code: str | None = None) -> None:
    from json import dumps

    with _engine().begin() as connection:
        connection.execute(text("""UPDATE ingest_documents SET state=:state, stage='finalizing',
            validation=CAST(:validation AS jsonb), error_code=:error,
            lease_until=NULL, completed_at=now(), updated_at=now(),
            expires_at=CASE WHEN :success THEN NULL ELSE now()+interval '24 hours' END
            WHERE doc_id=CAST(:id AS uuid) AND state='processing'"""),
            {"id": doc_id, "state": "done" if success else "failed", "validation": dumps(validation),
             "error": error_code, "success": success})


def fail_document(doc_id: str, error_code: str) -> None:
    with _engine().begin() as connection:
        connection.execute(text("""UPDATE ingest_documents SET state='failed', error_code=:error,
            lease_until=NULL, expires_at=now()+interval '24 hours', updated_at=now()
            WHERE doc_id=CAST(:id AS uuid) AND state IN ('uploading', 'accepted', 'processing')"""),
            {"id": doc_id, "error": error_code})


def clear_source(doc_id: str) -> None:
    with _engine().begin() as connection:
        connection.execute(text("UPDATE ingest_documents SET source_key=NULL, updated_at=now() WHERE doc_id=CAST(:id AS uuid)"), {"id": doc_id})


def pending_dispatches(limit: int = 20) -> list[dict]:
    with _engine().connect() as connection:
        return [dict(row) for row in connection.execute(text("""SELECT event_id, doc_id FROM ingest_outbox
            WHERE kind='process' AND state='pending' AND next_attempt_at<=now()
            ORDER BY created_at LIMIT :limit"""), {"limit": limit}).mappings()]


def record_dispatch(event_id: str, success: bool) -> None:
    with _engine().begin() as connection:
        connection.execute(text("""UPDATE ingest_outbox SET attempts=attempts+1,
            state=CASE WHEN :success THEN 'dispatched' ELSE 'pending' END,
            next_attempt_at=now()+interval '30 seconds'
            WHERE event_id=CAST(:id AS uuid)"""), {"id": event_id, "success": success})


def expired_sources() -> list[dict]:
    with _engine().connect() as connection:
        return [dict(row) for row in connection.execute(text("""SELECT doc_id, source_key, state FROM ingest_documents
            WHERE source_key IS NOT NULL AND (
                (state='failed' AND expires_at < now()) OR
                (state='uploading' AND created_at < now()-interval '1 hour') OR
                state='done')""")).mappings()]


def stale_processing() -> list[str]:
    with _engine().begin() as connection:
        rows = connection.execute(text("""UPDATE ingest_documents SET state='accepted', stage='queued', lease_until=NULL, updated_at=now()
            WHERE state='processing' AND lease_until < now() RETURNING doc_id""")).scalars().all()
        for doc_id in rows:
            connection.execute(text("""UPDATE ingest_outbox SET state='pending', next_attempt_at=now()
                WHERE doc_id=CAST(:id AS uuid) AND kind='process'"""), {"id": str(doc_id)})
        return [str(doc_id) for doc_id in rows]


def requeue_unstarted() -> list[str]:
    """Recover a published event whose broker message never started a worker."""
    with _engine().begin() as connection:
        rows = connection.execute(text("""SELECT d.doc_id FROM ingest_documents d
            JOIN ingest_outbox o ON o.doc_id=d.doc_id AND o.kind='process'
            WHERE d.state='accepted' AND o.state='dispatched'
              AND d.updated_at < now()-interval '10 minutes'""")).scalars().all()
        for doc_id in rows:
            connection.execute(text("""UPDATE ingest_outbox SET state='pending', next_attempt_at=now()
                WHERE doc_id=CAST(:id AS uuid) AND kind='process'"""), {"id": str(doc_id)})
            connection.execute(text("""UPDATE ingest_documents SET updated_at=now()
                WHERE doc_id=CAST(:id AS uuid)"""), {"id": str(doc_id)})
        return [str(doc_id) for doc_id in rows]


def release_retry(doc_id: str) -> None:
    with _engine().begin() as connection:
        connection.execute(text("""UPDATE ingest_documents SET state='accepted', stage='queued', lease_until=NULL,
            updated_at=now() WHERE doc_id=CAST(:id AS uuid) AND state='processing'"""), {"id": doc_id})
