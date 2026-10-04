"""Fail-closed authorization for the single-workspace ingestion API."""

from functools import lru_cache

from fastapi import HTTPException, Request

from backend.core.config import settings


@lru_cache(maxsize=1)
def _jwks_client():
    import jwt

    if not settings.OIDC_JWKS_URL or not settings.OIDC_JWKS_URL.startswith("https://"):
        raise RuntimeError("An HTTPS OIDC_JWKS_URL is required")
    return jwt.PyJWKClient(settings.OIDC_JWKS_URL)


def require_workspace(request: Request) -> str:
    """Return the single workspace key after group authorization."""
    allowed = settings.OIDC_ALLOWED_GROUP
    if not allowed or not settings.OIDC_ISSUER_URL or not settings.OIDC_AUDIENCE:
        raise HTTPException(status_code=503, detail="OIDC authorization is not configured")
    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if separator and scheme.lower() == "bearer":
        import jwt

        token = token.strip()
        try:
            claims = jwt.decode(
                token,
                _jwks_client().get_signing_key_from_jwt(token).key,
                algorithms=["RS256", "ES256"],
                issuer=settings.OIDC_ISSUER_URL,
                audience=settings.OIDC_AUDIENCE,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=401, detail="Invalid OIDC bearer token") from exc
        groups = claims.get(settings.OIDC_GROUP_CLAIM, [])
    elif authorization:
        raise HTTPException(status_code=401, detail="Unsupported authorization scheme")
    elif settings.TRUST_PROXY_AUTH_HEADERS:
        if not request.headers.get("x-forwarded-user"):
            raise HTTPException(status_code=401, detail="Sign-in required")
        groups = [group.strip() for group in request.headers.get("x-forwarded-groups", "").split(",")]
    else:
        raise HTTPException(status_code=401, detail="OIDC bearer token required")
    if isinstance(groups, str):
        groups = [groups]
    if not isinstance(groups, list) or allowed not in groups:
        raise HTTPException(status_code=403, detail="Workspace group required")
    return "workspace"
