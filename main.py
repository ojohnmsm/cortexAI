import os
import io
import json
import uuid
import httpx
import asyncio
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import supabase as sb
# ── optional deps (graceful import) ──────────────────────────────────────────
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
# ── env ───────────────────────────────────────────────────────────────────────
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
# ── supabase client ───────────────────────────────────────────────────────────
supa: Optional[sb.Client] = None
if SUPABASE_URL and SUPABASE_KEY:
 supa = sb.create_client(SUPABASE_URL, SUPABASE_KEY)
# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CORTEX AI")
app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],
 allow_methods=["*"],
 allow_headers=["*"],
)
# ── models ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
 role: str
 content: str
class ChatRequest(BaseModel):
 provider: str # "claude" | "openai"
 model: str
 messages: List[Message]
 system: Optional[str] = None
 use_knowledge: bool = False
 persona_id: Optional[str] = None
class DeleteDocRequest(BaseModel):
 doc_id: str
# ── helpers ───────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
 """Split text into overlapping chunks."""
 words = text.split()
 chunks, i = [], 0
 while i < len(words):
 chunk = " ".join(words[i:i + size])
 chunks.append(chunk)
 i += size - overlap
 return [c for c in chunks if len(c.strip()) > 40]
async def get_embedding(text: str) -> List[float]:
 """Call OpenAI embeddings API."""
 if not OPENAI_KEY:
 raise HTTPException(400, "OPENAI_API_KEY não configurada no servidor.")
 async with httpx.AsyncClient(timeout=30) as client:
 r = await client.post(
 "https://api.openai.com/v1/embeddings",
 headers={"Authorization": f"Bearer {OPENAI_KEY}"},
 json={"model": "text-embedding-3-small", "input": text[:8000]},
 )
 r.raise_for_status()
 return r.json()["data"][0]["embedding"]
async def search_knowledge(query: str, top_k: int = 5) -> List[dict]:
 """Vector search in Supabase."""
 if not supa:
 return []
 try:
 embedding = await get_embedding(query)
 result = supa.rpc(
 "match_cortex_chunks",
 {"query_embedding": embedding, "match_count": top_k},
 ).execute()
 return result.data or []
 except Exception as e:
 print(f"[search_knowledge] error: {e}")
 return []
def extract_text(filename: str, content: bytes) -> str:
 """Extract plain text from PDF, DOCX, PPTX or TXT."""
 ext = filename.rsplit(".", 1)[-1].lower()
 text = ""
 if ext == "pdf":
 if not HAS_PDF:
 raise HTTPException(400, "pdfplumber não instalado no servidor.")
 with pdfplumber.open(io.BytesIO(content)) as pdf:
 for page in pdf.pages:
 t = page.extract_text()
 if t:
 text += t + "\n"
 elif ext in ("docx", "doc"):
 if not HAS_DOCX:
 raise HTTPException(400, "python-docx não instalado no servidor.")
 doc = DocxDocument(io.BytesIO(content))
 text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
 elif ext in ("pptx", "ppt"):
 if not HAS_PPTX:
 raise HTTPException(400, "python-pptx não instalado no servidor.")
 prs = Presentation(io.BytesIO(content))
 for slide in prs.slides:
 for shape in slide.shapes:
 if hasattr(shape, "text") and shape.text.strip():
 text += shape.text + "\n"
 elif ext == "txt":
 text = content.decode("utf-8", errors="ignore")
 else:
 raise HTTPException(400, f"Formato '{ext}' não suportado. Use PDF, DOCX, PPTX ou TXT. return text.strip()
# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
 return FileResponse("static/index.html")
@app.get("/health")
async def health():
 return {"status": "ok", "supabase": bool(supa)}
# ── CHAT ─────────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
 messages = [m.dict() for m in req.messages]
 system_parts = []
 if req.system:
 system_parts.append(req.system)
 # RAG injection
 if req.use_knowledge and supa:
 last_user = next(
 (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
 )
 chunks = await search_knowledge(last_user)
 if chunks:
 context = "\n\n---\n\n".join(
 f"[{c.get('doc_name','Documento')}]\n{c.get('content','')}"
 for c in chunks
 )
 system_parts.append(
 f"Use os seguintes trechos da base de conhecimento para embasar sua resposta. f"Se a resposta não estiver nos trechos, diga que não encontrou na base.\n\n"
 f"BASE DE CONHECIMENTO:\n{context}"
 )
 system_prompt = "\n\n".join(system_parts) if system_parts else None
 # ── Claude ──
 if req.provider == "claude":
 if not ANTHROPIC_KEY:
 raise HTTPException(400, "ANTHROPIC_API_KEY não configurada no servidor.")
 body = {"model": req.model, "max_tokens": 4096, "messages": messages}
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
 raise HTTPException(r.status_code, data.get("error", {}).get("message", "Erro return {"reply": data["content"][0]["text"], "used_knowledge": req.use_knowledge}
 # ── OpenAI ──
 elif req.provider == "openai":
 if not OPENAI_KEY:
 raise HTTPException(400, "OPENAI_API_KEY não configurada no servidor.")
 if system_prompt:
 messages = [{"role": "system", "content": system_prompt}] + messages
 async with httpx.AsyncClient(timeout=60) as client:
 r = await client.post(
 "https://api.openai.com/v1/chat/completions",
 headers={
 "Authorization": f"Bearer {OPENAI_KEY}",
 "Content-Type": "application/json",
 },
 json={"model": req.model, "messages": messages, "max_tokens": 4096},
 )
 data = r.json()
 if not r.is_success:
 raise HTTPException(r.status_code, data.get("error", {}).get("message", "Erro
 return {"reply": data["choices"][0]["message"]["content"], "used_knowledge": req. else:
 raise HTTPException(400, "Provider inválido. Use 'claude' ou 'openai'.")
# ── KNOWLEDGE BASE ────────────────────────────────────────────────────────────
@app.post("/api/knowledge/upload")
async def upload_document(file: UploadFile = File(...)):
 if not supa:
 raise HTTPException(503, "Supabase não configurado no servidor.")
 content = await file.read()
 if len(content) > 20 * 1024 * 1024: # 20MB limit
 raise HTTPException(400, "Arquivo muito grande. Limite: 20MB.")
 # Extract text
 text = extract_text(file.filename, content)
 if not text:
 raise HTTPException(400, "Não foi possível extrair texto do arquivo.")
 doc_id = str(uuid.uuid4())
 # Save doc metadata
 supa.table("cortex_documents").insert({
 "id": doc_id,
 "name": file.filename,
 "char_count": len(text),
 "chunk_count": 0,
 }).execute()
 # Chunk + embed
 chunks = chunk_text(text)
 embedded = 0
 errors = 0
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
 except Exception as e:
 print(f"[upload] chunk {i} error: {e}")
 errors += 1
 # Update chunk count
 supa.table("cortex_documents").update({"chunk_count": embedded}).eq("id", doc_id).execute return {
 "doc_id": doc_id,
 "name": file.filename,
 "chunks": embedded,
 "errors": errors,
 "char_count": len(text),
 }
@app.get("/api/knowledge/documents")
async def list_documents():
 if not supa:
 return []
 result = supa.table("cortex_documents").select("*").order("created_at", desc=True).execut return result.data or []
@app.delete("/api/knowledge/documents/{doc_id}")
async def delete_document(doc_id: str):
 if not supa:
 raise HTTPException(503, "Supabase não configurado.")
 supa.table("cortex_chunks").delete().eq("doc_id", doc_id).execute()
 supa.table("cortex_documents").delete().eq("id", doc_id).execute()
 return {"deleted": doc_id}
# ── static files (after API routes) ──────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")