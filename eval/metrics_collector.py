"""
Custom Metrics Collector — KineticGraph-Vectra
Lightweight observability layer that works WITHOUT external services.
Stores metrics in SQLite (dev) or PostgreSQL (prod).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional psycopg2 for PostgreSQL
# ---------------------------------------------------------------------------
try:
    import psycopg2
    import psycopg2.extras
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

import sqlite3


# ---------------------------------------------------------------------------
# Schema DDL (SQLite-compatible; works in PG too with minor tweaks)
# ---------------------------------------------------------------------------
_DDL = """
CREATE TABLE IF NOT EXISTS query_events (
    id          TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    mode        TEXT NOT NULL,
    latency_ms  REAL NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_events (
    id          TEXT PRIMARY KEY,
    query_id    TEXT NOT NULL,
    agent       TEXT NOT NULL,
    result_json TEXT NOT NULL,
    latency_ms  REAL NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generation_events (
    id           TEXT PRIMARY KEY,
    query_id     TEXT NOT NULL,
    answer       TEXT NOT NULL,
    confidence   REAL DEFAULT 0.0,
    tokens_used  INTEGER DEFAULT 0,
    created_at   TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Collects and queries RAG pipeline metrics.

    - SQLite backend for local development (default)
    - PostgreSQL backend for production (set DATABASE_URL env var)

    Usage::

        mc = MetricsCollector()
        qid = mc.record_query("my query", "hybrid", latency_ms=120.4, chunk_count=8)
        mc.record_retrieval(qid, "vector_agent", results=[...], latency_ms=45.2)
        mc.record_generation(qid, answer="...", confidence=0.87, tokens_used=512)
        stats = mc.get_dashboard_stats(time_window_hours=24)
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        sqlite_path: str = "eval/metrics.db",
    ) -> None:
        self.db_url = db_url or os.getenv("DATABASE_URL", "")
        self.sqlite_path = sqlite_path
        self._use_postgres = bool(self.db_url and _PG_AVAILABLE)
        self._conn: Any = None
        self._init_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> Any:
        if self._use_postgres:
            if self._conn is None or self._conn.closed:
                self._conn = psycopg2.connect(self.db_url)
            return self._conn
        # SQLite — new connection per call (thread-safe)
        return sqlite3.connect(self.sqlite_path)

    def _init_db(self) -> None:
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            for stmt in _DDL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
            conn.commit()
            if not self._use_postgres:
                conn.close()
            logger.info(
                "MetricsCollector DB ready (%s)",
                "postgres" if self._use_postgres else f"sqlite:{self.sqlite_path}",
            )
        except Exception as exc:
            logger.error("MetricsCollector DB init error: %s", exc)

    def _execute(self, sql: str, params: tuple = ()) -> Optional[List[tuple]]:
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(sql, params)
            if sql.strip().upper().startswith("SELECT"):
                rows = cur.fetchall()
                if not self._use_postgres:
                    conn.close()
                return rows
            conn.commit()
            if not self._use_postgres:
                conn.close()
            return None
        except Exception as exc:
            logger.error("MetricsCollector query error: %s | SQL: %s", exc, sql[:80])
            return None

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def record_query(
        self,
        query: str,
        mode: str,
        latency_ms: float,
        chunk_count: int = 0,
        query_id: Optional[str] = None,
    ) -> str:
        """
        Record a completed query.

        Returns:
            query_id (str) — use for subsequent record_retrieval / record_generation calls.
        """
        qid = query_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            "INSERT INTO query_events (id, query, mode, latency_ms, chunk_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)" if not self._use_postgres else
            "INSERT INTO query_events (id, query, mode, latency_ms, chunk_count, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (qid, query, mode, latency_ms, chunk_count, now),
        )
        return qid

    def record_retrieval(
        self,
        query_id: str,
        agent: str,
        results: List[Any],
        latency_ms: float,
    ) -> None:
        """
        Record retrieval results from a specific agent (vector_agent / graph_agent / fusion_node).

        Args:
            query_id: ID returned by record_query.
            agent: Name of the retrieval agent.
            results: Raw result list (serialised to JSON for storage).
            latency_ms: Time taken by this agent.
        """
        rid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        # Store compact JSON snapshot (content + score only)
        compact = [
            {"content": r.get("content", "")[:200], "score": r.get("score", 0)}
            for r in (results or [])
        ]
        self._execute(
            "INSERT INTO retrieval_events (id, query_id, agent, result_json, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)" if not self._use_postgres else
            "INSERT INTO retrieval_events (id, query_id, agent, result_json, latency_ms, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (rid, query_id, agent, json.dumps(compact), latency_ms, now),
        )

    def record_generation(
        self,
        query_id: str,
        answer: str,
        confidence: float,
        tokens_used: int,
    ) -> None:
        """
        Record LLM generation result.

        Args:
            query_id: ID returned by record_query.
            answer: Generated answer string.
            confidence: LLM self-reported confidence (0–1).
            tokens_used: Total tokens consumed.
        """
        gid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            "INSERT INTO generation_events (id, query_id, answer, confidence, tokens_used, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)" if not self._use_postgres else
            "INSERT INTO generation_events (id, query_id, answer, confidence, tokens_used, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (gid, query_id, answer[:2000], confidence, tokens_used, now),
        )

    # ------------------------------------------------------------------
    # Read / analytics
    # ------------------------------------------------------------------

    def get_dashboard_stats(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Return aggregated metrics for the Grafana/Streamlit dashboard.

        Args:
            time_window_hours: Look-back window in hours.

        Returns:
            Dict with keys: total_queries, mode_distribution, latency_stats,
            retrieval_agent_latency, confidence_stats, token_stats.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        ).isoformat()

        ph = "%s" if self._use_postgres else "?"  # placeholder style

        # Total queries
        rows = self._execute(
            f"SELECT COUNT(*), AVG(latency_ms), MIN(latency_ms), MAX(latency_ms) "
            f"FROM query_events WHERE created_at >= {ph}",
            (cutoff,),
        )
        total, avg_lat, min_lat, max_lat = (rows[0] if rows else (0, 0, 0, 0))

        # Mode distribution
        mode_rows = self._execute(
            f"SELECT mode, COUNT(*) FROM query_events WHERE created_at >= {ph} GROUP BY mode",
            (cutoff,),
        ) or []
        mode_dist = {r[0]: r[1] for r in mode_rows}

        # Retrieval latency per agent
        agent_rows = self._execute(
            f"SELECT agent, COUNT(*), AVG(latency_ms) FROM retrieval_events "
            f"WHERE created_at >= {ph} GROUP BY agent",
            (cutoff,),
        ) or []
        agent_latency = {r[0]: {"count": r[1], "avg_ms": round(r[2] or 0, 2)} for r in agent_rows}

        # Confidence and token stats
        gen_rows = self._execute(
            f"SELECT AVG(confidence), MIN(confidence), MAX(confidence), "
            f"AVG(tokens_used), SUM(tokens_used) "
            f"FROM generation_events WHERE created_at >= {ph}",
            (cutoff,),
        )
        avg_conf, min_conf, max_conf, avg_tok, total_tok = (
            gen_rows[0] if gen_rows else (0, 0, 0, 0, 0)
        )

        # Context hit rate — heuristic: count retrievals with >= 1 result
        hit_rows = self._execute(
            f"SELECT COUNT(*) FROM retrieval_events "
            f"WHERE created_at >= {ph} AND result_json != '[]'",
            (cutoff,),
        )
        hit_count = (hit_rows[0][0] if hit_rows else 0)
        total_ret = sum(v["count"] for v in agent_latency.values()) or 1

        return {
            "time_window_hours": time_window_hours,
            "total_queries": total or 0,
            "latency_stats": {
                "avg_ms": round(avg_lat or 0, 2),
                "min_ms": round(min_lat or 0, 2),
                "max_ms": round(max_lat or 0, 2),
            },
            "mode_distribution": mode_dist,
            "retrieval_agent_latency": agent_latency,
            "context_hit_rate": round(hit_count / total_ret, 4),
            "confidence_stats": {
                "avg": round(avg_conf or 0, 4),
                "min": round(min_conf or 0, 4),
                "max": round(max_conf or 0, 4),
            },
            "token_stats": {
                "avg_per_query": round(avg_tok or 0, 1),
                "total_in_window": int(total_tok or 0),
            },
        }

    def get_slow_queries(
        self, threshold_ms: float = 2000, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return queries slower than *threshold_ms*."""
        ph = "%s" if self._use_postgres else "?"
        rows = self._execute(
            f"SELECT id, query, mode, latency_ms, chunk_count, created_at "
            f"FROM query_events WHERE latency_ms >= {ph} "
            f"ORDER BY latency_ms DESC LIMIT {ph}",
            (threshold_ms, limit),
        ) or []
        keys = ["query_id", "query", "mode", "latency_ms", "chunk_count", "created_at"]
        return [dict(zip(keys, r)) for r in rows]

    def get_low_confidence_queries(
        self, threshold: float = 0.5, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return generation events with confidence below *threshold*."""
        ph = "%s" if self._use_postgres else "?"
        rows = self._execute(
            f"SELECT g.query_id, q.query, q.mode, g.confidence, g.tokens_used, g.created_at "
            f"FROM generation_events g JOIN query_events q ON g.query_id = q.id "
            f"WHERE g.confidence < {ph} ORDER BY g.confidence ASC LIMIT {ph}",
            (threshold, limit),
        ) or []
        keys = ["query_id", "query", "mode", "confidence", "tokens_used", "created_at"]
        return [dict(zip(keys, r)) for r in rows]

    def get_mode_performance(self) -> Dict[str, Dict[str, float]]:
        """Compare latency across hybrid / vector / graph modes."""
        rows = self._execute(
            "SELECT mode, COUNT(*), AVG(latency_ms), AVG(chunk_count) "
            "FROM query_events GROUP BY mode"
        ) or []
        return {
            r[0]: {
                "count": r[1],
                "avg_latency_ms": round(r[2] or 0, 2),
                "avg_chunks": round(r[3] or 0, 2),
            }
            for r in rows
        }

    def close(self) -> None:
        """Close PostgreSQL connection if open."""
        if self._use_postgres and self._conn and not self._conn.closed:
            self._conn.close()
