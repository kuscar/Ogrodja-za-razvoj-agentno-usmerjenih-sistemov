"""
FastAPI

  * GET  /             (UI — serves static/index.html)
  * POST /auth/signup  (create account; email auto-confirmed via admin API)
  * POST /auth/signin  (password sign-in; returns JWT session)
  * POST /onboarding   (rate-limited; size & MIME-bound upload)
  * POST /cv/base      (rate-limited)
  * POST /cv/targeted  (rate-limited)
  * GET  /cv/download/{filename}  (authenticated PDF download)
  * GET  /livez        (liveness)
  * GET  /readyz       (readiness — pings downstreams)
  * GET  /metrics      (Prometheus scrape target)
"""

import asyncio
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx
import jwt
from jwt.algorithms import ECAlgorithm
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.evaluation import configure_langsmith
from src.graph.graph import get_graph
from src.graph.state import AgentState
from src.observability import (
    RequestLogMiddleware,
    configure_logging,
    get_logger,
    user_id_var,
)
from src.observability.metrics import metrics_response
from src.security import sanitize_user_text
from src.security.rate_limiter import (
    limit_for,
    limiter,
    rate_limit_handler,
)


configure_logging()
configure_langsmith()
log = get_logger("api")

app = FastAPI(title="CV Builder", version="0.2.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestLogMiddleware)

auth_scheme = HTTPBearer()


_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
_SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "90"))
ALLOWED_UPLOAD_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}


@lru_cache(maxsize=1)
def _supabase_jwks() -> dict[str, object]:
    """Fetch Supabase JWKS once at first use and cache for the process lifetime."""
    try:
        import urllib.request as _ur
        with _ur.urlopen(f"{_SUPABASE_URL}/auth/v1/.well-known/jwks.json", timeout=5) as r:
            data = json.loads(r.read())
        return {k["kid"]: ECAlgorithm.from_jwk(json.dumps(k)) for k in data.get("keys", []) if k.get("kty") == "EC"}
    except Exception as exc:
        log.warning("jwks.fetch_failed", reason=str(exc))
        return {}


def current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> str:
    token = creds.credentials
    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "HS256")
        if alg == "ES256":
            kid = header.get("kid", "")
            keys = _supabase_jwks()
            pub_key = keys.get(kid)
            if pub_key is None:
                raise jwt.InvalidTokenError("unknown kid")
            decoded = jwt.decode(token, pub_key, algorithms=["ES256"], audience="authenticated")
        else:
            decoded = jwt.decode(
                token,
                os.environ["SUPABASE_JWT_SECRET"],
                algorithms=["HS256"],
                audience="authenticated",
            )
        user_id = decoded["sub"]
    except jwt.PyJWTError as exc:
        log.warning("auth.invalid_token", reason=str(exc))
        raise HTTPException(401, "invalid token") from None

    request.state.user_id = user_id
    user_id_var.set(user_id)
    return user_id


async def _stream_to_disk(upload: UploadFile, dest: Path) -> int:
    """Read upload in chunks, abort if it exceeds MAX_UPLOAD_BYTES."""
    written = 0
    chunk_size = 64 * 1024
    with dest.open("wb") as fh:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "upload exceeds size limit")
            fh.write(chunk)
    return written


async def _invoke_graph(state: AgentState) -> dict:
    try:
        return await asyncio.wait_for(
            get_graph().ainvoke(state), timeout=REQUEST_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        log.error("graph.timeout", timeout_s=REQUEST_TIMEOUT_SECONDS)
        raise HTTPException(504, "request timed out") from None
    except Exception as exc:
        err = str(exc)
        log.error("graph.error", error=err[:300])
        if "Recursion limit" in err:
            detail = "Agent pipeline stalled — complete Step 1 (upload your CV) before generating a CV."
        else:
            detail = f"Pipeline error: {err[:250]}"
        raise HTTPException(500, detail) from None


@app.get("/")
def root():
    html = Path(__file__).parent.parent.parent / "static" / "index.html"
    if html.exists():
        return HTMLResponse(html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>CV Builder</h1><p>See <a href='/docs'>/docs</a>.</p>")


async def _supabase_password_signin(email: str, password: str) -> JSONResponse:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_SUPABASE_URL}/auth/v1/token?grant_type=password",
            headers={"apikey": _SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"email": email, "password": password},
        )
    body = r.json()
    if r.status_code != 200:
        msg = body.get("error_description") or body.get("msg") or body.get("message") or "Sign in failed"
        log.error("supabase.signin_failed", status=r.status_code, supabase_body=body)
        raise HTTPException(400, msg)
    return JSONResponse({
        "access_token": body["access_token"],
        "refresh_token": body.get("refresh_token", ""),
        "expires_in": body.get("expires_in", 3600),
        "email": body.get("user", {}).get("email", email),
    })


@app.post("/auth/signup")
@limiter.limit("10/hour;2/minute")
async def auth_signup(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
) -> JSONResponse:
    """Create account (email auto-confirmed via admin API) and return a session."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_SUPABASE_URL}/auth/v1/admin/users",
            headers={
                "apikey": _SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {_SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            },
            json={"email": email, "password": password, "email_confirm": True},
        )
    body = r.json()
    if r.status_code not in (200, 201):
        msg = body.get("msg") or body.get("message") or body.get("error_description") or "Signup failed"
        log.error("supabase.signup_failed", status=r.status_code, supabase_body=body)
        raise HTTPException(400, msg)
    return await _supabase_password_signin(email, password)


@app.post("/auth/signin")
@limiter.limit("20/hour;5/minute")
async def auth_signin(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
) -> JSONResponse:
    """Sign in with email + password and return a session."""
    return await _supabase_password_signin(email, password)


@app.get("/cv/download/{filename}")
def download_artifact(filename: str, user_id: str = Depends(current_user)):
    """Serve a compiled PDF from the artifacts volume (user-scoped)."""
    safe = Path(filename).name
    path = Path("/var/cv-artifacts") / user_id / safe
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise HTTPException(404, "file not found")
    # The file on disk already carries the friendly name; serve it as-is.
    return FileResponse(str(path), filename=safe, media_type="application/pdf")


@app.post("/onboarding")
@limiter.limit(limit_for("/onboarding"))
async def onboarding(
    request: Request,
    cv: UploadFile = File(...),
    enrichment: str | None = Form(None),
    user_id: str = Depends(current_user),
):
    if cv.content_type not in ALLOWED_UPLOAD_MIMES:
        raise HTTPException(415, "unsupported file type")

    upload_dir = Path("/var/cv-uploads") / user_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    raw_path = upload_dir / f"{uuid.uuid4()}-{cv.filename}"
    size = await _stream_to_disk(cv, raw_path)
    log.info("onboarding.upload", bytes=size, mime=cv.content_type)

    state: AgentState = {
        "user_id": user_id,
        "phase": "onboarding",
        "raw_cv_path": str(raw_path),
        "enrichment_text": sanitize_user_text(enrichment or ""),
        "messages": [],
        "errors": [],
    }
    return await _invoke_graph(state)


@app.post("/cv/base")
@limiter.limit(limit_for("/cv/base"))
async def base_cv(request: Request, user_id: str = Depends(current_user)):
    state: AgentState = {
        "user_id": user_id,
        "phase": "base_cv",
        "messages": [],
        "errors": [],
    }
    return await _invoke_graph(state)


@app.post("/cv/targeted")
@limiter.limit(limit_for("/cv/targeted"))
async def targeted_cv(
    request: Request,
    job_description: str = Form(...),
    user_id: str = Depends(current_user),
):
    state: AgentState = {
        "user_id": user_id,
        "phase": "targeted",
        "job_description": sanitize_user_text(job_description),
        "messages": [],
        "errors": [],
    }
    return await _invoke_graph(state)


@app.get("/livez")
def livez():
    """Process is up. Used by liveness probe."""
    return {"ok": True}


@app.get("/readyz")
async def readyz():
    """Downstreams reachable. Used by readiness probe / load balancer."""
    checks: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=2.0) as client:
        for name, url_env in [
            ("supabase_mcp", "SUPABASE_MCP_URL"),
            ("prompt_guard", "PROMPT_GUARD_URL"),
        ]:
            url = os.environ.get(url_env)
            if not url:
                checks[name] = "unset"
                continue
            try:
                r = await client.get(f"{url}/healthz")
                checks[name] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
            except Exception:
                checks[name] = "unreachable"
    overall = all(v == "ok" for v in checks.values())
    return Response(
        content=str({"ok": overall, "checks": checks}),
        status_code=200 if overall else 503,
        media_type="application/json",
    )


@app.get("/metrics")
def prometheus():
    """Scrape target. In prod, lock this behind a network policy or auth proxy."""
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)
