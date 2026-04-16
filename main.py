import os
import io
import uuid
import logging
from typing import Optional, List

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
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


# ---------------------------------------------------------------
# AUTH HELPERS
# ---------------------------------------------------------------

async def get_current_user(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing authorization header.")
    token = authorization.split(" ", 1)[1]
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    try:
        user_resp = supa.auth.get_user(token)
        user = user_resp.user
        if not user:
            raise HTTPException(401, "Invalid token.")
    except Exception as exc:
        raise HTTPException(401, "Invalid or expired token: " + str(exc))

    profile_resp = supa.table("user_profiles").select("*").eq("id", user.id).single().execute()
    if not profile_resp.data:
        raise HTTPException(403, "User profile not found.")
    profile = profile_resp.data
    if not profile.get("active", True):
        raise HTTPException(403, "Account disabled.")
    return profile


def require_role(allowed_roles: List[str]):
    async def checker(profile: dict = Depends(get_current_user)):
        if profile["role"] not in allowed_roles:
            raise HTTPException(403, "Your role does not have access to this feature.")
        return profile
    return checker


async def check_token_limit(profile: dict, estimated_tokens: int = 500):
    used = profile.get("tokens_used", 0)
    limit = profile.get("tokens_limit", 100000)
    if limit > 0 and used + estimated_tokens > limit:
        raise HTTPException(429, "Token limit reached. Contact your administrator.")


def update_token_usage(user_id: str, provider: str, model: str, input_tokens: int, output_tokens: int):
    total = input_tokens + output_tokens
    try:
        current = supa.table("user_profiles").select("tokens_used").eq("id", user_id).single().execute().data
        current_used = (current or {}).get("tokens_used", 0)
        supa.table("user_profiles").update({"tokens_used": current_used + total}).eq("id", user_id).execute()
        supa.table("usage_log").insert({
            "user_id": user_id,
            "provider": provider,
            "model": model,
            "tokens_input": input_tokens,
            "tokens_output": output_tokens,
            "tokens_total": total,
        }).execute()
    except Exception as exc:
        logger.error("update_token_usage error: %s", exc)


def get_default_personality_id() -> Optional[str]:
    if not supa:
        return None
    try:
        resp = (
            supa.table("ai_personalities")
            .select("id")
            .eq("scope", "global")
            .eq("is_default", True)
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0]["id"] if rows else None
    except Exception as exc:
        logger.error("get_default_personality_id error: %s", exc)
        return None


def get_personality_for_user(personality_id: Optional[str], user_id: str) -> Optional[dict]:
    if not supa or not personality_id:
        return None
    try:
        resp = supa.table("ai_personalities").select("*").eq("id", personality_id).eq("is_active", True).limit(1).execute()
        rows = resp.data or []
        if not rows:
            return None
        item = rows[0]
        if item.get("scope") == "global":
            return item
        if item.get("scope") == "personal" and item.get("created_by") == user_id:
            return item
        return None
    except Exception as exc:
        logger.error("get_personality_for_user error: %s", exc)
        return None


def ensure_conversation_owner(conversation_id: str, user_id: str) -> dict:
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    resp = (
        supa.table("ai_conversations")
        .select("*")
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        raise HTTPException(404, "Conversation not found.")
    return rows[0]


# ---------------------------------------------------------------
# PYDANTIC MODELS
# ---------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    provider: str
    model: str
    messages: List[Message]
    system: Optional[str] = None
    personality_id: Optional[str] = None
    use_knowledge: bool = False


class UpdateProfileRequest(BaseModel):
    user_id: str
    role: Optional[str] = None
    tokens_limit: Optional[int] = None
    active: Optional[bool] = None


class PersonalityCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    system_prompt: str
    scope: str = "personal"
    is_default: bool = False


class PersonalityUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = "Novo chat"
    provider: str
    model: str
    personality_id: Optional[str] = None
    use_knowledge: bool = False


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    personality_id: Optional[str] = None
    use_knowledge: Optional[bool] = None


class ConversationMessageItem(BaseModel):
    role: str
    content: str
    time: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    usedKB: Optional[bool] = False
    tokens: Optional[int] = 0


class ConversationMessagesReplaceRequest(BaseModel):
    messages: List[ConversationMessageItem]


# ---------------------------------------------------------------
# KNOWLEDGE HELPERS
# ---------------------------------------------------------------

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


# ---------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------

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


# --- Auth routes ---

@app.get("/api/auth/me")
async def get_me(profile: dict = Depends(get_current_user)):
    return {
        "id": profile["id"],
        "email": profile["email"],
        "role": profile["role"],
        "tokens_used": profile["tokens_used"],
        "tokens_limit": profile["tokens_limit"],
        "active": profile["active"],
    }


# --- Admin routes ---

@app.get("/api/admin/users")
async def list_users(profile: dict = Depends(require_role(["admin"]))):
    result = supa.table("user_profiles").select("*").order("created_at", desc=True).execute()
    return result.data or []


@app.post("/api/admin/users/update")
async def update_user(req: UpdateProfileRequest, profile: dict = Depends(require_role(["admin"]))):
    update = {}
    if req.role is not None:
        update["role"] = req.role
    if req.tokens_limit is not None:
        update["tokens_limit"] = req.tokens_limit
    if req.active is not None:
        update["active"] = req.active
    if not update:
        raise HTTPException(400, "Nothing to update.")
    supa.table("user_profiles").update(update).eq("id", req.user_id).execute()
    return {"updated": req.user_id}


@app.post("/api/admin/users/reset-tokens")
async def reset_tokens(req: UpdateProfileRequest, profile: dict = Depends(require_role(["admin"]))):
    supa.table("user_profiles").update({"tokens_used": 0}).eq("id", req.user_id).execute()
    return {"reset": req.user_id}


# --- Personalities ---

@app.get("/api/personalities")
async def list_personalities(profile: dict = Depends(get_current_user)):
    global_resp = (
        supa.table("ai_personalities")
        .select("id,name,description,scope,is_default,created_by,is_active")
        .eq("scope", "global")
        .eq("is_active", True)
        .order("name")
        .execute()
    )
    personal_resp = (
        supa.table("ai_personalities")
        .select("id,name,description,scope,is_default,created_by,is_active")
        .eq("scope", "personal")
        .eq("created_by", profile["id"])
        .eq("is_active", True)
        .order("name")
        .execute()
    )
    items = (global_resp.data or []) + (personal_resp.data or [])
    return {"items": items, "default_id": get_default_personality_id()}


@app.post("/api/personalities")
async def create_personality(req: PersonalityCreateRequest, profile: dict = Depends(get_current_user)):
    scope = (req.scope or "personal").strip().lower()
    if scope not in ("global", "personal"):
        raise HTTPException(400, "Invalid scope.")
    if scope == "global" and profile["role"] != "admin":
        raise HTTPException(403, "Only admins can create global personalities.")
    if not req.name.strip() or not req.system_prompt.strip():
        raise HTTPException(400, "Name and system_prompt are required.")

    if scope == "global" and req.is_default:
        supa.table("ai_personalities").update({"is_default": False}).eq("scope", "global").execute()

    payload = {
        "id": str(uuid.uuid4()),
        "name": req.name.strip(),
        "slug": req.name.strip().lower().replace(" ", "-") + "-" + str(uuid.uuid4())[:8],
        "description": (req.description or "").strip() or None,
        "system_prompt": req.system_prompt.strip(),
        "scope": scope,
        "created_by": None if scope == "global" else profile["id"],
        "is_active": True,
        "is_default": bool(req.is_default) if scope == "global" else False,
    }
    supa.table("ai_personalities").insert(payload).execute()
    return payload


@app.put("/api/personalities/{personality_id}")
async def update_personality(personality_id: str, req: PersonalityUpdateRequest, profile: dict = Depends(get_current_user)):
    current = get_personality_for_user(personality_id, profile["id"])
    if not current and profile["role"] == "admin":
        resp = supa.table("ai_personalities").select("*").eq("id", personality_id).limit(1).execute()
        rows = resp.data or []
        current = rows[0] if rows else None
    if not current:
        raise HTTPException(404, "Personality not found.")
    if current.get("scope") == "global" and profile["role"] != "admin":
        raise HTTPException(403, "Only admins can edit global personalities.")

    update = {}
    if req.name is not None:
        update["name"] = req.name.strip()
    if req.description is not None:
        update["description"] = req.description.strip() or None
    if req.system_prompt is not None:
        update["system_prompt"] = req.system_prompt.strip()
    if req.is_active is not None:
        update["is_active"] = req.is_active
    if req.is_default is not None and current.get("scope") == "global":
        if profile["role"] != "admin":
            raise HTTPException(403, "Only admins can set the default global personality.")
        if req.is_default:
            supa.table("ai_personalities").update({"is_default": False}).eq("scope", "global").execute()
        update["is_default"] = req.is_default
    if not update:
        raise HTTPException(400, "Nothing to update.")
    supa.table("ai_personalities").update(update).eq("id", personality_id).execute()
    return {"updated": personality_id}


@app.delete("/api/personalities/{personality_id}")
async def delete_personality(personality_id: str, profile: dict = Depends(get_current_user)):
    current = get_personality_for_user(personality_id, profile["id"])
    if not current and profile["role"] == "admin":
        resp = supa.table("ai_personalities").select("*").eq("id", personality_id).limit(1).execute()
        rows = resp.data or []
        current = rows[0] if rows else None
    if not current:
        raise HTTPException(404, "Personality not found.")
    if current.get("scope") == "global" and profile["role"] != "admin":
        raise HTTPException(403, "Only admins can delete global personalities.")
    if current.get("scope") == "global" and current.get("is_default"):
        raise HTTPException(400, "Cannot delete the default global personality before choosing another one.")
    supa.table("ai_personalities").delete().eq("id", personality_id).execute()
    return {"deleted": personality_id}


# --- Conversations ---

@app.get("/api/conversations")
async def list_conversations(profile: dict = Depends(get_current_user)):
    conv_resp = (
        supa.table("ai_conversations")
        .select("*")
        .eq("user_id", profile["id"])
        .order("updated_at", desc=True)
        .execute()
    )
    conversations = conv_resp.data or []
    if not conversations:
        return []

    msg_resp = (
        supa.table("ai_messages")
        .select("*")
        .eq("user_id", profile["id"])
        .order("message_order")
        .execute()
    )
    all_messages = msg_resp.data or []
    grouped = {}
    for row in all_messages:
        grouped.setdefault(row["conversation_id"], []).append({
            "role": row["role"],
            "content": row["content"],
            "time": row.get("time_ms"),
            "provider": row.get("provider"),
            "model": row.get("model"),
            "usedKB": row.get("used_knowledge", False),
            "tokens": row.get("tokens_total", 0),
        })

    items = []
    for conv in conversations:
        items.append({
            "id": conv["id"],
            "title": conv.get("title") or "Novo chat",
            "provider": conv.get("provider") or "openai",
            "model": conv.get("model") or "gpt-4o",
            "personality_id": conv.get("personality_id"),
            "use_knowledge": conv.get("use_knowledge", False),
            "created_at": conv.get("created_at"),
            "updated_at": conv.get("updated_at"),
            "messages": grouped.get(conv["id"], []),
        })
    return items


@app.post("/api/conversations")
async def create_conversation(req: ConversationCreateRequest, profile: dict = Depends(get_current_user)):
    personality_id = req.personality_id
    if personality_id:
        p = get_personality_for_user(personality_id, profile["id"])
        if not p:
            raise HTTPException(400, "Invalid personality_id.")
    payload = {
        "id": str(uuid.uuid4()),
        "user_id": profile["id"],
        "title": (req.title or "Novo chat").strip() or "Novo chat",
        "provider": req.provider,
        "model": req.model,
        "personality_id": personality_id,
        "use_knowledge": req.use_knowledge,
    }
    supa.table("ai_conversations").insert(payload).execute()
    payload["messages"] = []
    return payload


@app.put("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, req: ConversationUpdateRequest, profile: dict = Depends(get_current_user)):
    ensure_conversation_owner(conversation_id, profile["id"])
    update = {}
    if req.title is not None:
        update["title"] = req.title.strip() or "Novo chat"
    if req.provider is not None:
        update["provider"] = req.provider
    if req.model is not None:
        update["model"] = req.model
    if req.use_knowledge is not None:
        update["use_knowledge"] = req.use_knowledge
    if req.personality_id is not None:
        if req.personality_id:
            p = get_personality_for_user(req.personality_id, profile["id"])
            if not p:
                raise HTTPException(400, "Invalid personality_id.")
            update["personality_id"] = req.personality_id
        else:
            update["personality_id"] = None
    if not update:
        raise HTTPException(400, "Nothing to update.")
    supa.table("ai_conversations").update(update).eq("id", conversation_id).eq("user_id", profile["id"]).execute()
    return {"updated": conversation_id}


@app.put("/api/conversations/{conversation_id}/messages")
async def replace_conversation_messages(conversation_id: str, req: ConversationMessagesReplaceRequest, profile: dict = Depends(get_current_user)):
    ensure_conversation_owner(conversation_id, profile["id"])
    supa.table("ai_messages").delete().eq("conversation_id", conversation_id).eq("user_id", profile["id"]).execute()

    rows = []
    for idx, msg in enumerate(req.messages):
        rows.append({
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "user_id": profile["id"],
            "message_order": idx,
            "role": msg.role,
            "content": msg.content,
            "time_ms": msg.time,
            "provider": msg.provider,
            "model": msg.model,
            "used_knowledge": bool(msg.usedKB),
            "tokens_total": int(msg.tokens or 0),
        })
    if rows:
        supa.table("ai_messages").insert(rows).execute()
    return {"saved": len(rows)}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, profile: dict = Depends(get_current_user)):
    ensure_conversation_owner(conversation_id, profile["id"])
    supa.table("ai_messages").delete().eq("conversation_id", conversation_id).eq("user_id", profile["id"]).execute()
    supa.table("ai_conversations").delete().eq("id", conversation_id).eq("user_id", profile["id"]).execute()
    return {"deleted": conversation_id}


# --- Chat ---

@app.post("/api/chat")
async def chat(req: ChatRequest, profile: dict = Depends(get_current_user)):
    if req.provider == "claude" and profile["role"] not in ("claude_user", "admin"):
        raise HTTPException(403, "Claude access requires the claude_user role. Contact your administrator.")

    await check_token_limit(profile)

    messages = [m.dict() for m in req.messages]
    system_parts = []

    if req.system:
        system_parts.append(req.system)

    personality = get_personality_for_user(req.personality_id, profile["id"]) if req.personality_id else None
    if not personality:
        default_id = get_default_personality_id()
        if default_id:
            personality = get_personality_for_user(default_id, profile["id"]) or get_personality_for_user(default_id, "")
            if not personality:
                try:
                    resp = supa.table("ai_personalities").select("*").eq("id", default_id).limit(1).execute()
                    rows = resp.data or []
                    personality = rows[0] if rows else None
                except Exception:
                    personality = None
    if personality and personality.get("system_prompt"):
        system_parts.append(personality["system_prompt"])

    used_kb = False
    if req.use_knowledge and supa:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        chunks = await search_knowledge(last_user)
        if chunks:
            used_kb = True
            context = "\n\n---\n\n".join(
                "[" + c.get("doc_name", "Doc") + "]\n" + c.get("content", "")
                for c in chunks
            )
            system_parts.append(
                "Use the following knowledge base excerpts to answer. "
                "Always cite the document name when using information from it. "
                "If the answer is not in the excerpts, say so clearly.\n\n"
                "KNOWLEDGE BASE:\n" + context
            )

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    input_tokens = 0
    output_tokens = 0

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
            reply = data["content"][0]["text"]
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

    elif req.provider == "openai":
        if not OPENAI_KEY:
            raise HTTPException(400, "OPENAI_API_KEY not set on server.")
        openai_messages = list(messages)
        if system_prompt:
            openai_messages = [{"role": "system", "content": system_prompt}] + openai_messages
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": "Bearer " + OPENAI_KEY,
                    "Content-Type": "application/json",
                },
                json={"model": req.model, "messages": openai_messages, "max_tokens": 4096},
            )
            data = r.json()
            if not r.is_success:
                msg = data.get("error", {}).get("message", "OpenAI API error")
                raise HTTPException(r.status_code, msg)
            reply = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
    else:
        raise HTTPException(400, "Invalid provider.")

    if supa and (input_tokens + output_tokens) > 0:
        update_token_usage(profile["id"], req.provider, req.model, input_tokens, output_tokens)

    return {
        "reply": reply,
        "used_knowledge": used_kb,
        "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
    }


# --- Knowledge base ---

@app.post("/api/knowledge/upload")
async def upload_document(
    file: UploadFile = File(...),
    profile: dict = Depends(require_role(["admin", "claude_user"]))
):
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
        "uploaded_by": profile["id"],
        "uploaded_by_email": profile.get("email", ""),
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
async def list_documents(profile: dict = Depends(get_current_user)):
    if not supa:
        return []
    result = supa.table("cortex_documents").select("*").order("created_at", desc=True).execute()
    return result.data or []


@app.delete("/api/knowledge/documents/{doc_id}")
async def delete_document(doc_id: str, profile: dict = Depends(get_current_user)):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    doc_resp = supa.table("cortex_documents").select("uploaded_by").eq("id", doc_id).single().execute()
    if not doc_resp.data:
        raise HTTPException(404, "Document not found.")
    is_owner = doc_resp.data.get("uploaded_by") == profile["id"]
    is_admin = profile["role"] == "admin"
    if not is_owner and not is_admin:
        raise HTTPException(403, "You can only delete documents you uploaded.")
    supa.table("cortex_chunks").delete().eq("doc_id", doc_id).execute()
    supa.table("cortex_documents").delete().eq("id", doc_id).execute()
    return {"deleted": doc_id}


if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------
# AUTH ENDPOINTS (login / register via Supabase Auth)
# ---------------------------------------------------------------

class AuthRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/login")
async def login(req: AuthRequest):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    try:
        resp = supa.auth.sign_in_with_password({"email": req.email, "password": req.password})
        session = resp.session
        user_id = resp.user.id
    except Exception:
        raise HTTPException(401, "Invalid email or password.")

    profile = supa.table("user_profiles").select("*").eq("id", user_id).single().execute().data
    if not profile:
        raise HTTPException(403, "User profile not found.")
    if not profile.get("active", True):
        raise HTTPException(403, "Account disabled.")

    return {
        "access_token": session.access_token,
        "user": {
            "id": user_id,
            "email": req.email,
            "role": profile["role"],
            "tokens_used": profile["tokens_used"],
            "tokens_limit": profile["tokens_limit"],
            "active": profile["active"],
        }
    }


@app.post("/api/auth/register")
async def register(req: AuthRequest):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    if not req.email.lower().endswith("@vale.com"):
        raise HTTPException(400, "Only @vale.com emails are allowed.")
    try:
        resp = supa.auth.sign_up({"email": req.email, "password": req.password})
        user = resp.user
        session = resp.session
    except Exception as exc:
        raise HTTPException(400, str(exc))
    if not user:
        raise HTTPException(400, "Could not create user.")
    profile = supa.table("user_profiles").select("*").eq("id", user.id).single().execute().data
    return {
        "access_token": session.access_token if session else None,
        "user": {
            "id": user.id,
            "email": req.email,
            "role": profile["role"] if profile else "user",
            "tokens_used": (profile or {}).get("tokens_used", 0),
            "tokens_limit": (profile or {}).get("tokens_limit", 100000),
            "active": (profile or {}).get("active", True),
        }
    }
