import os
import io
import uuid
import logging
from typing import Optional, List
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
try:
    from pptx import Presentation
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False
try:
    import supabase as sb
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
supa = None
if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    try:
        supa = sb.create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase connected")
    except Exception as exc:
        logger.warning("Supabase init failed: %s", exc)
app = FastAPI(title="CORTEX AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
async def startup():
    logger.info("CORTEX AI started")
    logger.info("Anthropic: %s", "set" if ANTHROPIC_KEY else "MISSING")
    logger.info("OpenAI: %s", "set" if OPENAI_KEY else "MISSING")
    logger.info("Supabase: %s", "connected" if supa else "not configured")
class Message(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
    provider: str
    model: str
    messages: List[Message]
    system: Optional[str] = None
    use_knowledge: bool = False
def chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return [c for c in chunks if len(c.strip()) > 40]
async def get_embedding(text: str) -> List[float]:
    if not OPENAI_KEY:
        raise HTTPException(400, "OPENAI_API_KEY not set on server.")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": "Bearer " + OPENAI_KEY},
            json={"model": "text-embedding-3-small", "input": text[:8000]},
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
async def search_knowledge(query: str, top_k: int = 5) -> List[dict]:
    if not supa:
        return []
    try:
        embedding = await get_embedding(query)
        result = supa.rpc(
            "match_cortex_chunks",
            {"query_embedding": embedding, "match_count": top_k},
        ).execute()
        return result.data or []
    except Exception as exc:
        logger.error("search_knowledge error: %s", exc)
        return []
def extract_text(filename: str, content: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    text = ""
    if ext == "pdf":
        if not HAS_PDF:
            raise HTTPException(400, "pdfplumber not installed.")
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    elif ext in ("docx", "doc"):
        if not HAS_DOCX:
            raise HTTPException(400, "python-docx not installed.")
        doc = DocxDocument(io.BytesIO(content))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext in ("pptx", "ppt"):
        if not HAS_PPTX:
            raise HTTPException(400, "python-pptx not installed.")
        prs = Presentation(io.BytesIO(content))
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text += shape.text + "\n"
    elif ext == "txt":
        text = content.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(400, "Unsupported format: " + ext)
    return text.strip()
@app.get("/")
async def root():
    if os.path.isfile("static/index.html"):
        return FileResponse("static/index.html")
    return {"status": "CORTEX AI running"}
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "supabase": bool(supa),
        "anthropic": bool(ANTHROPIC_KEY),
        "openai": bool(OPENAI_KEY),
    }
@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = [m.dict() for m in req.messages]
    system_parts = []
    if req.system:
        system_parts.append(req.system)
    used_kb = False
    if req.use_knowledge and supa:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        chunks = await search_knowledge(last_user)
        if chunks:
            used_kb = True
            context = "\n\n---\n\n".join(
                "[" + c.get("doc_name", "Doc") + "]\n" + c.get("content", "")
                for c in chunks
            )
            system_parts.append(
                "Use the following knowledge base excerpts to answer. "
                "If the answer is not in the excerpts, say so.\n\n"
                "KNOWLEDGE BASE:\n" + context
            )
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    if req.provider == "claude":
        if not ANTHROPIC_KEY:
            raise HTTPException(400, "ANTHROPIC_API_KEY not set on server.")
        body = {
            "model": req.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system_prompt:
            body["system"] = system_prompt
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            data = r.json()
            if not r.is_success:
                msg = data.get("error", {}).get("message", "Anthropic API error")
                raise HTTPException(r.status_code, msg)
            return {"reply": data["content"][0]["text"], "used_knowledge": used_kb}
    elif req.provider == "openai":
        if not OPENAI_KEY:
            raise HTTPException(400, "OPENAI_API_KEY not set on server.")
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": "Bearer " + OPENAI_KEY,
                    "Content-Type": "application/json",
                },
                json={"model": req.model, "messages": messages, "max_tokens": 4096},
            )
            data = r.json()
            if not r.is_success:
                msg = data.get("error", {}).get("message", "OpenAI API error")
                raise HTTPException(r.status_code, msg)
            return {"reply": data["choices"][0]["message"]["content"], "used_knowledge": used_kb}
    else:
        raise HTTPException(400, "Invalid provider. Use claude or openai.")
@app.post("/api/knowledge/upload")
async def upload_document(file: UploadFile = File(...)):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large. Max 20MB.")
    text = extract_text(file.filename, content)
    if not text:
        raise HTTPException(400, "Could not extract text from file.")
    doc_id = str(uuid.uuid4())
    supa.table("cortex_documents").insert({
        "id": doc_id,
        "name": file.filename,
        "char_count": len(text),
        "chunk_count": 0,
    }).execute()
    chunks = chunk_text(text)
    embedded = 0
    for i, chunk in enumerate(chunks):
        try:
            embedding = await get_embedding(chunk)
            supa.table("cortex_chunks").insert({
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "doc_name": file.filename,
                "content": chunk,
                "chunk_index": i,
                "embedding": embedding,
            }).execute()
            embedded += 1
        except Exception as exc:
            logger.error("chunk %d error: %s", i, exc)
    supa.table("cortex_documents").update({"chunk_count": embedded}).eq("id", doc_id).execute()
    return {"doc_id": doc_id, "name": file.filename, "chunks": embedded, "char_count": len(text)}
@app.get("/api/knowledge/documents")
async def list_documents():
    if not supa:
        return []
    result = supa.table("cortex_documents").select("*").order("created_at", desc=True).execute()
    return result.data or []
@app.delete("/api/knowledge/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    supa.table("cortex_chunks").delete().eq("doc_id", doc_id).execute()
    supa.table("cortex_documents").delete().eq("id", doc_id).execute()
    return {"deleted": doc_id}
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
