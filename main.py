# api.py — Full-featured Telegram search API (ICMR-style) for Vercel
import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Dict, Optional

import duckdb
import gradio as gr
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────
# URL to your pre-built DuckDB file on Hugging Face
DB_URL = os.environ.get(
    "TG_DB_URL",
    "https://huggingface.co/yourusername/telegram_db/resolve/main/telegram.duckdb"
)

SEARCH_FIELDS = [
    "user_id", "username", "first_name", "last_name",
    "phone", "email", "status", "linked_id",
    "linked_name", "linked_handle"
]
NUMBER_FIELDS = ["user_id", "phone"]

PARALLELISM = int(os.environ.get("TG_PARALLEL", "1"))      # Vercel: keep low
THREADS_PER_CONN = int(os.environ.get("TG_THREADS", "1"))
DUPLICATE_CAP = 2

# ── DuckDB Connection Pool (thread‑local) ──────────────────────
_conns: List[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")

def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # Vercel: use /tmp for extensions
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # Attach remote DB read‑only
    con.execute(f"ATTACH '{DB_URL}' AS tg (READ_ONLY)")
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con

def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid

def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]

# ── Dedup & helpers ──────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    uid = (row.get("user_id") or "").strip()
    ph = (row.get("phone") or "").strip()
    if uid or ph:
        return (uid, ph)
    return (row.get("first_name") or "").strip(), (row.get("last_name") or "").strip()

def _cap_duplicates(rows: List[dict]) -> List[dict]:
    seen: Dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            out.append(r)
    return out

# ── Search Logic ──────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")

    if mode == "exact":
        # Exact match on any field – uses index for user_id and phone
        sql = f"SELECT * FROM tg.tg WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    elif mode == "contains":
        # Contains search – only efficient for phone (due to index) but works for all
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = f"SELECT * FROM tg.tg WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}

def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    if not q:
        return {"query": q, "count": 0, "results": []}
    is_num = q.isdigit() and len(q) >= 5

    if is_num:
        all_rows = []
        searched = []
        # Try user_id first (indexed)
        if True:  # always available
            r = _run_field_search("user_id", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("user_id")
        # If none, try phone exact
        if not all_rows:
            r = _run_field_search("phone", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("phone")
        # If still none, try contains on phone
        if not all_rows:
            r = _run_field_search("phone", q, "contains", limit)
            all_rows.extend(r["results"])
            searched.append("phone(contains)")
        all_rows = _cap_duplicates(all_rows)[:limit]
        return {
            "query": q,
            "searched_fields": searched,
            "count": len(all_rows),
            "results": all_rows
        }
    else:
        # Text search across multiple fields (no index – scan)
        text_fields = ["username", "first_name", "last_name", "email", "linked_name", "linked_handle"]
        conditions = []
        for f in text_fields:
            v = q.replace("'", "''")
            conditions.append(f"{f} ILIKE '%{v}%'")
        sql = f"SELECT * FROM tg.tg WHERE {' OR '.join(conditions)} LIMIT {limit}"
        con = _get_conn()
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
        results = [dict(zip(cols, r)) for r in rows]
        return {"query": q, "searched_fields": text_fields, "count": len(results), "results": results}

# ── FastAPI App ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up a connection (optional, to avoid first‑request latency)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(pool, _get_conn)
    yield

fastapi_app = FastAPI(
    title="Telegram Search API",
    description="Full‑featured search over 1.65B Telegram records",
    lifespan=lifespan
)

class BatchRequest(BaseModel):
    queries: List[Dict[str, Any]]
    limit: int = 10

@fastapi_app.get("/")
def root():
    return {
        "app": "Telegram Search API",
        "records": "~1.65B",
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@NischayYadav615"
    }

@fastapi_app.get("/health")
def health():
    try:
        con = _get_conn()
        count = con.execute("SELECT COUNT(*) FROM tg.tg").fetchone()[0]
        return {"status": "ok", "records": count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@fastapi_app.get("/search")
async def search(
    q: Optional[str] = Query(None),
    mobile: Optional[str] = Query(None),
    field: Optional[str] = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True)
):
    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile")
    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(pool, _run_field_search, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)
    result = {"success": data.get("count", 0) > 0, "query": q_val, "total": data.get("count", 0), **data}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")

@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            pool,
            _run_field_search,
            item.get("field", "user_id"),
            item.get("value", ""),
            item.get("mode", "exact"),
            int(item.get("limit", req.limit))
        )
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return Response(
        content=json.dumps({"searches": len(req.queries), "results": results}, indent=2, ensure_ascii=False),
        media_type="application/json"
    )

@fastapi_app.get("/user/{user_id}")
async def get_user_by_id(user_id: str):
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _run_field_search, "user_id", user_id, "exact", 1)
    if data.get("count", 0) == 0:
        raise HTTPException(404, f"User {user_id} not found")
    return Response(
        content=json.dumps(data["results"][0], indent=2, ensure_ascii=False),
        media_type="application/json"
    )

# ── Gradio UI ────────────────────────────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val:
            lines.append(f"**{field}:** {val}")
    return "\n\n".join(lines) if lines else "No data"

def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Please enter a search query (user_id, phone, name, etc.)"
    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    count = data.get("count", 0)
    results = data.get("results", [])
    searched = ", ".join(data.get("searched_fields", []))
    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n❌ No results found."
    header = f"🔍 **Query:** `{q}`  |  **Found:** {count}  |  **Searched:** {searched}\n\n---\n\n"
    parts = []
    for i, row in enumerate(results, 1):
        parts.append(f"### Result {i}\n{format_result(row)}")
    return header + "\n\n---\n\n".join(parts)

def build_ui():
    with gr.Blocks(title="Telegram Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Telegram User Search")
        gr.Markdown("Search **1.65 billion** Telegram records — user_id, phone, name, username & more")
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="user_id, phone number, name, or username...",
                    lines=1
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Max Results"
                )
        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")
        search_btn.click(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        query_input.submit(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        gr.Markdown("---")
        with gr.Accordion("📡 API Info", open=False):
            gr.Markdown("""
**Endpoints:**
- `GET /search?q=<query>` — Auto‑detect search
- `GET /search?field=user_id&q=123` — Field‑specific search
- `GET /user/{user_id}` — Fast lookup by user_id
- `POST /search/parallel` — Batch search (max 50)
- `GET /health` — Health check
- `GET /docs` — Swagger UI

**Source:** [AnyJobHub/telegram](https://huggingface.co/datasets/AnyJobHub/telegram)
            """)
        gr.Markdown("---\n<div style='text-align:center;color:#888;'>👨‍💻 Deployed on Vercel with DuckDB + remote index</div>")
    return demo

demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
