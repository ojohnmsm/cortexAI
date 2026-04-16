import os
import io
import uuid
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

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
    """Validate Supabase JWT and return user profile."""
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
    """Dependency factory: checks user has one of the allowed roles."""
    async def checker(profile: dict = Depends(get_current_user)):
        if profile["role"] not in allowed_roles:
            raise HTTPException(403, "Your role does not have access to this feature.")
        return profile
    return checker


async def check_token_limit(profile: dict, estimated_tokens: int = 500):
    """Raise 429 if user is over their token limit."""
    used = profile.get("tokens_used", 0)
    limit = profile.get("tokens_limit", 100000)
    if limit > 0 and used + estimated_tokens > limit:
        raise HTTPException(429, "Token limit reached. Contact your administrator.")


def get_allowed_openai_models_for_admin() -> List[str]:
    return [
        "gpt-5.4-thinking",
        "gpt-5.4-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
    ]


def apply_chat_model_policy(req: ChatRequest, profile: dict) -> tuple[str, str]:
    provider = req.provider
    model = req.model

    if provider == "claude":
        if profile["role"] not in ("claude_user", "admin"):
            raise HTTPException(403, "Claude access requires the claude_user role. Contact your administrator.")
        return provider, model

    if provider != "openai":
        raise HTTPException(400, "Unsupported provider.")

    if profile["role"] == "admin":
        if model not in get_allowed_openai_models_for_admin():
            raise HTTPException(400, "OpenAI model not allowed for admin selection.")
        return "openai", model

    return "openai", ("gpt-4o" if req.think else "gpt-5.4-nano")


def update_token_usage(user_id: str, provider: str, model: str, input_tokens: int, output_tokens: int):
    """Update token counters in background (best-effort)."""
    total = input_tokens + output_tokens
    try:
        supa.table("user_profiles").update({
            "tokens_used": supa.table("user_profiles")
                .select("tokens_used").eq("id", user_id).single().execute().data["tokens_used"] + total
        }).eq("id", user_id).execute()
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
    think: bool = False


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
    title: Optional[str] = None
    provider: str = "openai"
    model: str = "gpt-5.4-nano"
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
    usedKB: Optional[bool] = None
    tokens: Optional[int] = None


class ConversationMessagesSyncRequest(BaseModel):
    messages: List[ConversationMessageItem]


class GlobalRulesUpdateRequest(BaseModel):
    system_prompt: str
    is_active: bool = True


def slugify(text: str) -> str:
    value = ''.join(ch.lower() if ch.isalnum() else '-' for ch in text.strip())
    while '--' in value:
        value = value.replace('--', '-')
    return value.strip('-') or 'personalidade'


def normalize_personality_scope(scope: str) -> str:
    scope = (scope or 'personal').strip().lower()
    if scope not in ('global', 'personal'):
        raise HTTPException(400, "Invalid scope. Use 'global' or 'personal'.")
    return scope


def personality_visible_to_user(personality: dict, profile: dict) -> bool:
    if not personality or not personality.get('is_active', True):
        return False
    if personality.get('scope') == 'global':
        return True
    return personality.get('owner_user_id') == profile['id']


def personality_can_manage(personality: dict, profile: dict) -> bool:
    if profile['role'] == 'admin':
        return True
    return personality.get('scope') == 'personal' and personality.get('owner_user_id') == profile['id']


def clear_existing_default_global():
    if not supa:
        return
    supa.table('ai_personalities').update({'is_default': False}).eq('scope', 'global').eq('is_default', True).execute()


def serialize_personality(personality: dict, profile: dict) -> dict:
    return {
        'id': personality['id'],
        'name': personality['name'],
        'slug': personality.get('slug'),
        'description': personality.get('description'),
        'scope': personality.get('scope', 'personal'),
        'system_prompt': personality.get('system_prompt'),
        'is_active': personality.get('is_active', True),
        'is_default': personality.get('is_default', False),
        'owner_user_id': personality.get('owner_user_id'),
        'created_at': personality.get('created_at'),
        'updated_at': personality.get('updated_at'),
        'can_edit': personality_can_manage(personality, profile),
        'can_delete': personality_can_manage(personality, profile),
    }


def get_default_global_personality() -> Optional[dict]:
    if not supa:
        return None
    try:
        resp = supa.table('ai_personalities').select('*').eq('scope', 'global').eq('is_active', True).eq('is_default', True).limit(1).execute()
        data = resp.data or []
        return data[0] if data else None
    except Exception as exc:
        logger.error('get_default_global_personality error: %s', exc)
        return None


def get_personality_by_id(personality_id: str) -> Optional[dict]:
    if not supa:
        return None
    resp = supa.table('ai_personalities').select('*').eq('id', personality_id).limit(1).execute()
    data = resp.data or []
    return data[0] if data else None


def get_visible_personalities(profile: dict) -> dict:
    if not supa:
        return {'items': [], 'default_id': None}
    globals_resp = supa.table('ai_personalities').select('*').eq('scope', 'global').eq('is_active', True).order('name').execute()
    personals_resp = supa.table('ai_personalities').select('*').eq('scope', 'personal').eq('owner_user_id', profile['id']).eq('is_active', True).order('name').execute()
    items = (globals_resp.data or []) + (personals_resp.data or [])
    default_item = next((p for p in items if p.get('scope') == 'global' and p.get('is_default')), None)
    return {
        'items': [serialize_personality(p, profile) for p in items],
        'default_id': default_item['id'] if default_item else None,
    }


def resolve_system_prompt(req: ChatRequest, profile: dict) -> Optional[str]:
    system_parts = []

    global_rules = get_active_global_rules_text()
    if global_rules:
        system_parts.append(global_rules)

    if req.personality_id:
        personality = get_personality_by_id(req.personality_id)
        if not personality or not personality_visible_to_user(personality, profile):
            raise HTTPException(403, 'Selected personality is not available for this user.')
        if personality.get('system_prompt'):
            system_parts.append(personality['system_prompt'])
    elif req.system:
        system_parts.append(req.system)
    else:
        default_personality = get_default_global_personality()
        if default_personality and default_personality.get('system_prompt'):
            system_parts.append(default_personality['system_prompt'])

    return "\n\n".join(system_parts) if system_parts else None



# ---------------------------------------------------------------
# GLOBAL RULES & CHAT HISTORY HELPERS
# ---------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def get_global_rules_record() -> Optional[dict]:
    if not supa:
        return None
    try:
        resp = supa.table("chat_global_rules").select("*").order("updated_at", desc=True).limit(1).execute()
        data = resp.data or []
        return data[0] if data else None
    except Exception as exc:
        logger.error("get_global_rules_record error: %s", exc)
        return None


def get_active_global_rules_text() -> Optional[str]:
    row = get_global_rules_record()
    if row and row.get("is_active", True):
        return (row.get("system_prompt") or "").strip() or None
    return None


def serialize_global_rules(row: Optional[dict]) -> dict:
    row = row or {}
    return {
        "id": row.get("id"),
        "system_prompt": row.get("system_prompt") or "",
        "is_active": row.get("is_active", True),
        "updated_by": row.get("updated_by"),
        "updated_at": row.get("updated_at"),
    }


def get_user_conversation_or_404(conversation_id: str, profile: dict) -> dict:
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    resp = supa.table("ai_conversations").select("*").eq("id", conversation_id).eq("user_id", profile["id"]).limit(1).execute()
    data = resp.data or []
    if not data:
        raise HTTPException(404, "Conversation not found.")
    return data[0]


def list_conversation_messages(conversation_id: str) -> List[dict]:
    if not supa:
        return []
    resp = supa.table("ai_messages").select("*").eq("conversation_id", conversation_id).order("position").execute()
    items = resp.data or []
    return [
        {
            "role": item.get("role"),
            "content": item.get("content") or "",
            "time": item.get("message_time"),
            "provider": item.get("provider"),
            "usedKB": item.get("used_kb", False),
            "tokens": item.get("tokens_total"),
        }
        for item in items
    ]


def serialize_conversation(row: dict, include_messages: bool = False) -> dict:
    data = {
        "id": row.get("id"),
        "title": row.get("title") or "Novo chat",
        "provider": row.get("provider") or "openai",
        "model": row.get("model") or "gpt-4o",
        "personality_id": row.get("personality_id"),
        "use_knowledge": row.get("use_knowledge", False),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }
    if include_messages:
        data["messages"] = list_conversation_messages(row["id"])
    return data


def replace_conversation_messages(conversation_id: str, messages: List[ConversationMessageItem], provider: Optional[str] = None) -> int:
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    supa.table("ai_messages").delete().eq("conversation_id", conversation_id).execute()
    if not messages:
        return 0
    payload = []
    for i, msg in enumerate(messages):
        payload.append({
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "position": i,
            "role": msg.role,
            "content": msg.content,
            "message_time": msg.time,
            "provider": msg.provider or provider,
            "used_kb": bool(msg.usedKB),
            "tokens_total": msg.tokens,
        })
    supa.table("ai_messages").insert(payload).execute()
    return len(payload)


def touch_conversation(conversation_id: str, update: Optional[dict] = None):
    if not supa:
        return
    update = dict(update or {})
    update["updated_at"] = utc_now_iso()
    supa.table("ai_conversations").update(update).eq("id", conversation_id).execute()


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


# --- Chat ---

@app.post("/api/chat")
async def chat(req: ChatRequest, profile: dict = Depends(get_current_user)):
    provider, model = apply_chat_model_policy(req, profile)

    # Check token limit
    await check_token_limit(profile)

    messages = [m.dict() for m in req.messages]
    system_parts = []

    resolved_system = resolve_system_prompt(req, profile)
    if resolved_system:
        system_parts.append(resolved_system)

    used_kb = False
    chunks = []
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
                "Always cite the document name when using information from it. "
                "If the answer is not in the excerpts, say so clearly.\n\n"
                "KNOWLEDGE BASE:\n" + context
            )

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    input_tokens = 0
    output_tokens = 0

    if provider == "claude":
        if not ANTHROPIC_KEY:
            raise HTTPException(400, "ANTHROPIC_API_KEY not set on server.")
        body = {
            "model": model,
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

    elif provider == "openai":
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
                json={"model": model, "messages": messages, "max_tokens": 4096},
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

    # Update token usage (best-effort)
    if supa and (input_tokens + output_tokens) > 0:
        update_token_usage(profile["id"], provider, model, input_tokens, output_tokens)

    return {
        "reply": reply,
        "used_knowledge": used_kb,
        "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
    }



# --- Personalities ---

@app.get("/api/personalities")
async def list_personalities(profile: dict = Depends(get_current_user)):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    return get_visible_personalities(profile)


@app.post("/api/personalities")
async def create_personality(req: PersonalityCreateRequest, profile: dict = Depends(get_current_user)):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")

    scope = normalize_personality_scope(req.scope)
    if scope == 'global' and profile['role'] != 'admin':
        raise HTTPException(403, 'Only admins can create global personalities.')

    name = (req.name or '').strip()
    prompt = (req.system_prompt or '').strip()
    if not name or not prompt:
        raise HTTPException(400, 'Name and system prompt are required.')

    slug = slugify(name)
    payload = {
        'name': name,
        'slug': slug if scope == 'global' else f"{slug}-{profile['id'][:8]}",
        'description': (req.description or '').strip() or None,
        'system_prompt': prompt,
        'scope': scope,
        'owner_user_id': None if scope == 'global' else profile['id'],
        'is_active': True,
        'is_default': bool(req.is_default) if scope == 'global' else False,
    }

    if payload['is_default']:
        clear_existing_default_global()

    try:
        resp = supa.table('ai_personalities').insert(payload).execute()
        item = (resp.data or [None])[0]
    except Exception as exc:
        raise HTTPException(400, f'Could not create personality: {exc}')

    return serialize_personality(item, profile)


@app.put("/api/personalities/{personality_id}")
async def update_personality(personality_id: str, req: PersonalityUpdateRequest, profile: dict = Depends(get_current_user)):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")

    current = get_personality_by_id(personality_id)
    if not current:
        raise HTTPException(404, 'Personality not found.')
    if not personality_can_manage(current, profile):
        raise HTTPException(403, 'You do not have permission to edit this personality.')

    update = {}
    if req.name is not None:
        name = req.name.strip()
        if not name:
            raise HTTPException(400, 'Name cannot be empty.')
        update['name'] = name
        if current.get('scope') == 'global':
            update['slug'] = slugify(name)
    if req.description is not None:
        update['description'] = req.description.strip() or None
    if req.system_prompt is not None:
        prompt = req.system_prompt.strip()
        if not prompt:
            raise HTTPException(400, 'System prompt cannot be empty.')
        update['system_prompt'] = prompt
    if req.is_active is not None:
        update['is_active'] = req.is_active
    if req.is_default is not None:
        if current.get('scope') != 'global':
            raise HTTPException(400, 'Only global personalities can be default.')
        if profile['role'] != 'admin':
            raise HTTPException(403, 'Only admins can change the default personality.')
        update['is_default'] = req.is_default
        if req.is_default:
            clear_existing_default_global()

    if not update:
        raise HTTPException(400, 'Nothing to update.')

    resp = supa.table('ai_personalities').update(update).eq('id', personality_id).execute()
    item = (resp.data or [None])[0] or get_personality_by_id(personality_id)
    return serialize_personality(item, profile)


@app.delete("/api/personalities/{personality_id}")
async def delete_personality(personality_id: str, profile: dict = Depends(get_current_user)):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")

    current = get_personality_by_id(personality_id)
    if not current:
        raise HTTPException(404, 'Personality not found.')
    if not personality_can_manage(current, profile):
        raise HTTPException(403, 'You do not have permission to delete this personality.')

    if current.get('scope') == 'global' and current.get('is_default'):
        raise HTTPException(400, 'Cannot delete the default global personality. Set another default first.')

    supa.table('ai_personalities').delete().eq('id', personality_id).execute()
    return {'deleted': personality_id}



# --- Chat history ---

@app.get("/api/conversations")
async def list_conversations(profile: dict = Depends(get_current_user)):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    resp = supa.table("ai_conversations").select("*").eq("user_id", profile["id"]).order("updated_at", desc=True).execute()
    items = resp.data or []
    return {"items": [serialize_conversation(item, include_messages=True) for item in items]}


@app.post("/api/conversations")
async def create_conversation(req: ConversationCreateRequest, profile: dict = Depends(get_current_user)):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    now = utc_now_iso()
    payload = {
        "id": str(uuid.uuid4()),
        "user_id": profile["id"],
        "title": (req.title or "Novo chat").strip() or "Novo chat",
        "provider": req.provider,
        "model": req.model,
        "personality_id": req.personality_id,
        "use_knowledge": bool(req.use_knowledge),
        "created_at": now,
        "updated_at": now,
    }
    resp = supa.table("ai_conversations").insert(payload).execute()
    row = (resp.data or [payload])[0]
    return serialize_conversation(row, include_messages=True)


@app.put("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, req: ConversationUpdateRequest, profile: dict = Depends(get_current_user)):
    current = get_user_conversation_or_404(conversation_id, profile)
    update = {}
    if req.title is not None:
        update["title"] = req.title.strip() or "Novo chat"
    if req.provider is not None:
        update["provider"] = req.provider
    if req.model is not None:
        update["model"] = req.model
    if req.personality_id is not None:
        update["personality_id"] = req.personality_id
    if req.use_knowledge is not None:
        update["use_knowledge"] = req.use_knowledge
    if not update:
        return serialize_conversation(current, include_messages=True)
    touch_conversation(conversation_id, update)
    updated = get_user_conversation_or_404(conversation_id, profile)
    return serialize_conversation(updated, include_messages=True)


@app.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, profile: dict = Depends(get_current_user)):
    get_user_conversation_or_404(conversation_id, profile)
    return {"items": list_conversation_messages(conversation_id)}


@app.put("/api/conversations/{conversation_id}/messages")
async def sync_conversation_messages(conversation_id: str, req: ConversationMessagesSyncRequest, profile: dict = Depends(get_current_user)):
    current = get_user_conversation_or_404(conversation_id, profile)
    count = replace_conversation_messages(conversation_id, req.messages, provider=current.get("provider"))
    touch_conversation(conversation_id)
    return {"conversation_id": conversation_id, "count": count}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, profile: dict = Depends(get_current_user)):
    get_user_conversation_or_404(conversation_id, profile)
    supa.table("ai_messages").delete().eq("conversation_id", conversation_id).execute()
    supa.table("ai_conversations").delete().eq("id", conversation_id).execute()
    return {"deleted": conversation_id}


# --- Admin: global rules ---

@app.get("/api/admin/global-rules")
async def get_admin_global_rules(profile: dict = Depends(require_role(["admin"]))):
    return serialize_global_rules(get_global_rules_record())


@app.put("/api/admin/global-rules")
async def update_admin_global_rules(req: GlobalRulesUpdateRequest, profile: dict = Depends(require_role(["admin"]))):
    if not supa:
        raise HTTPException(503, "Supabase not configured.")
    prompt = (req.system_prompt or "").strip()
    now = utc_now_iso()
    current = get_global_rules_record()
    payload = {
        "system_prompt": prompt,
        "is_active": bool(req.is_active),
        "updated_by": profile["id"],
        "updated_at": now,
    }
    if current and current.get("id"):
        resp = supa.table("chat_global_rules").update(payload).eq("id", current["id"]).execute()
        row = (resp.data or [None])[0] or get_global_rules_record()
    else:
        payload.update({"id": str(uuid.uuid4()), "created_at": now})
        resp = supa.table("chat_global_rules").insert(payload).execute()
        row = (resp.data or [payload])[0]
    return serialize_global_rules(row)


# --- Knowledge base (admin only for upload/delete, all authenticated for list) ---

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
    # Fetch doc to check ownership
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
    except Exception as exc:
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
        raise HTTPException(403, "Registration is restricted to @vale.com email addresses.")
    try:
        resp = supa.auth.sign_up({"email": req.email, "password": req.password})
        if not resp.user:
            raise HTTPException(400, "Registration failed.")
        session = resp.session
        user_id = resp.user.id
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, "Registration error: " + str(exc))

    # Profile created by trigger; fetch it
    import time
    time.sleep(0.5)
    profile = supa.table("user_profiles").select("*").eq("id", user_id).single().execute().data
    role = profile["role"] if profile else "user"

    if not session:
        return {"message": "Account created. Check your email to confirm before logging in."}

    return {
        "access_token": session.access_token,
        "user": {
            "id": user_id,
            "email": req.email,
            "role": role,
            "tokens_used": 0,
            "tokens_limit": 100000,
            "active": True,
        }
    }
