# api.py — ICMR-style Telegram search (partitioned indexes)
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

# ── Config ──────────────────────────────────────────────────────
HF_INDEX_BASE = os.environ.get(
    "TG_HF_INDEX_BASE",
    "https://huggingface.co/datasets/yourusername/telegram_index/resolve/main",
).rstrip("/")

PARALLELISM = int(os.environ.get("TG_PARALLEL", "1"))   # Vercel: keep low
THREADS_PER_CONN = int(os.environ.get("TG_THREADS", "1"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "user_id", "username", "first_name", "last_name",
    "phone", "email", "status", "linked_id",
    "linked_name", "linked_handle"
]
NUMBER_FIELDS = ["user_id", "phone"]

# Remote index files (6 partitions each)
REMOTE_INDEXES = {
    "user_id": [f"{HF_INDEX_BASE}/idx_user_id.{i}.parquet" for i in range(6)],
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(6)],
}

def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES

# ── DuckDB Connection Pool ────────────────────────────────────
_conns: List[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")

def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # Create views for each index
    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        lst = ", ".join(f"'{u}'" for u in urls)
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet([{lst}])")
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

# ── Dedup & helpers ────────────────────────────────────────────
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

# ── Search Logic ────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")

    if mode == "exact":
        if field == "user_id" and _idx_ready("user_id"):
            view = "people_user_id"
        elif field == "phone" and _idx_ready("phone"):
            view = "people_phone"
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        sql = f"SELECT * FROM {view} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    elif mode == "contains":
        # For contains, we can only use the phone index (or fallback to scan)
        if field == "phone" and _idx_ready("phone"):
            view = "people_phone"
            v2 = v.replace("%", r"\%").replace("_", r"\_")
            sql = f"SELECT * FROM {view} WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
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
        # Try user_id first (fastest)
        if _idx_ready("user_id"):
            r = _run_field_search("user_id", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("user_id")
        # If not found, try phone
        if not all_rows and _idx_ready("phone"):
            r = _run_field_search("phone", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("phone")
        # If still nothing, try contains on phone
        if not all_rows and _idx_ready("phone"):
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
        # Text search: scan all text fields (slow, but we rarely use it)
        # We'll limit aggressively.
        text_fields = ["username", "first_name", "last_name", "email", "linked_name", "linked_handle"]
        conditions = []
        for f in text_fields:
            v = q.replace("'", "''")
            conditions.append(f"{f} ILIKE '%{v}%'")
        if not conditions:
            return {"query": q, "count": 0, "results": []}
        sql = f"SELECT * FROM people_user_id WHERE {' OR '.join(conditions)} LIMIT {limit}"
        con = _get_conn()
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
        results = [dict(zip(cols, r)) for r in rows]
        return {"query": q, "searched_fields": text_fields, "count": len(results), "results": results}

# ── FastAPI + Lifespan ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up one connection (only if needed)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(pool, _get_conn)
    yield

fastapi_app = FastAPI(
    title="Telegram Search (ICMR-style)",
    description="Partitioned indexes for user_id and phone",
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
        "indexes": {"user_id": _idx_ready("user_id"), "phone": _idx_ready("phone")},
        "columns": SEARCH_FIELDS,
        "docs": "/docs"
    }

@fastapi_app.get("/health")
def health():
    return {"status": "ok", "indexes": {"user_id": _idx_ready("user_id"), "phone": _idx_ready("phone")}}

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

# ── Gradio UI (optional) ────────────────────────────────────────
# Keep it minimal to avoid extra load, but include for consistency
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val:
            lines.append(f"**{field}:** {val}")
    return "\n\n".join(lines) if lines else "No data"

def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Enter a query (user_id, phone, name, etc.)"
    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    count = data.get("count", 0)
    results = data.get("results", [])
    searched = ", ".join(data.get("searched_fields", []))
    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n❌ No results."
    header = f"🔍 **Query:** `{q}`  |  **Found:** {count}  |  **Searched:** {searched}\n\n---\n\n"
    parts = []
    for i, row in enumerate(results, 1):
        parts.append(f"### Result {i}\n{format_result(row)}")
    return header + "\n\n---\n\n".join(parts)

def build_ui():
    with gr.Blocks(title="Telegram Search") as demo:
        gr.Markdown("# 🔍 Telegram Search (ICMR-style)")
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Search", placeholder="user_id, phone, name...", lines=1)
            with gr.Column(scale=1):
                limit_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Max Results")
        search_btn = gr.Button("🔍 Search", variant="primary")
        output = gr.Markdown(label="Results")
        search_btn.click(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        query_input.submit(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        gr.Markdown("---\n**Source:** Partitioned indexes on Hugging Face")
    return demo

demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
