from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv, find_dotenv
from contextlib import asynccontextmanager
import os
from pathlib import Path

# ── Load .env ────────────────────────────────────────────────────────────────
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
else:
    for candidate in [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",
    ]:
        if candidate.exists():
            load_dotenv(candidate)
            break

# ── Lifespan Handler ─────────────────────────────────────────────────────────
async def _background_cache_load():
    try:
        print("[CACHE] Loading background cache...")
        print("[CACHE] Background cache loaded successfully")
    except Exception as e:
        print(f"[CACHE] Error loading cache: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] App is up. Scheduling background cache load...")
    await _background_cache_load()
    yield
    print("[SHUTDOWN] Application shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CV Parser API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
def import_auth():
    from routes.auth import router as r
    return r

def import_dashboad():
    from routes.dashboad import routes as r
    return r

def import_cvupload():
    from routes.cvupload import routes as r
    return r

def import_ranking():
    from routes.ranking import routes as r
    return r

app.include_router(import_auth())
app.include_router(import_dashboad())
app.include_router(import_cvupload())
app.include_router(import_ranking())

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")}

# ── Serve Frontend (production) ───────────────────────────────────────────────
static_dir = "/app/static"
assets_path = f"{static_dir}/assets"

if os.path.isdir(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
else:
    print(f"[DEBUG] Assets directory not found: {assets_path} - running in API-only mode")

@app.get("/")
async def serve_root():
    return FileResponse(f"{static_dir}/index.html")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # Let API routes 404 naturally
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail=f"API route not found: /{full_path}")
    file_path = os.path.join(static_dir, full_path)
    if full_path and os.path.isfile(file_path):
        return FileResponse(file_path)
    # Fallback to index.html for React Router
    return FileResponse(f"{static_dir}/index.html")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"[LAUNCH] Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)