# Auditoria de Segurança — CORTEX AI

Data: 2026-04-17
Escopo: repositório inteiro (`main.py`, `static/index.html`, `render.yml`, `requirements.txt`).

## Resumo
Sistema monolítico FastAPI + frontend estático embutido. Principais riscos identificados: **vazamento cross-user na base de conhecimento (RAG)**, **superfície web permissiva (CORS + token em localStorage)**, **ausência de controles defensivos essenciais (rate limit, validação de upload robusta, pipeline de segurança)**.

## Top achados
1. RAG sem filtro de autorização por usuário/tenant.
2. Listagem global de documentos para qualquer usuário autenticado.
3. CORS totalmente aberto com bearer token em frontend SPA.
4. Token persistido em `localStorage`.
5. Falta de rate limiting em login/chat/upload.
6. Prompt injection não mitigado (documentos/anexos entram direto no system context).
7. Endpoint `/health` expõe estado interno de integrações.
8. Mensagens de erro potencialmente verbosas com detalhes de exceção.
9. Upload baseado em extensão/MIME declarativo (sem assinatura/magic bytes).
10. Ausência de SAST/secret scan/dependency scan automatizados no CI.
