---
name: fastapi
description: Describe what this skill does and when to use it. Include keywords that help agents identify relevant tasks.
---

# skill.md: FastAPI Project Guidelines

This file provides context to AI agents about the best practices and specifics of this FastAPI project.

## Capabilities

*   Can build high-performance, asynchronous APIs using FastAPI, Pydantic, and SQLAlchemy 2.0.
*   Handles request validation and data serialization automatically using Pydantic models.
*   Generates interactive API documentation via OpenAPI/Swagger UI and ReDoc.

## Limitations

*   Cannot use blocking I/O operations in `async def` path functions. Must use `run_in_threadpool()` or a task queue for CPU-intensive/blocking tasks.
*   Project currently does not have a dedicated task queue (e.g., Celery) configured. All background tasks must use FastAPI's built-in `BackgroundTasks` feature.
*   Only supports PostgreSQL as a database backend.

## Best Practices

*   **Async First:** Prioritize asynchronous code (`async def`) for all path operations and database calls.
*   **Type Hints:** Use standard Python type hints everywhere for clarity and validation.
*   **Dependencies:** Leverage FastAPI's dependency injection (`Depends`) for authentication, database sessions, and shared logic.
*   **Structure:** Follow a modular structure (e.g., `routers`, `schemas`, `services`, `database`).
*   **Error Handling:** Use `HTTPException` for standard errors, and define custom exception handlers for global errors.
