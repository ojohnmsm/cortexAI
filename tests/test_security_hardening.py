from fastapi.testclient import TestClient

import main


class _FakeUserObj:
    def __init__(self, user_id="u1"):
        self.id = user_id


class _FakeAuthResponse:
    def __init__(self, user_id="u1"):
        self.user = _FakeUserObj(user_id)
        self.session = type("Session", (), {"access_token": "tok"})()


class _FakeAuth:
    def get_user(self, token):
        return type("UserResp", (), {"user": _FakeUserObj("u1")})()

    def sign_in_with_password(self, payload):
        return _FakeAuthResponse("u1")

    def sign_up(self, payload):
        return _FakeAuthResponse("u1")


class _FakeQuery:
    def __init__(self, data=None):
        self._data = data if data is not None else []

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def single(self):
        return self

    def execute(self):
        return type("Resp", (), {"data": self._data})()

    def order(self, *args, **kwargs):
        return self


class _FakeSupa:
    def __init__(self):
        self.auth = _FakeAuth()

    def table(self, name):
        if name == "user_profiles":
            return _FakeQuery(
                {
                    "id": "u1",
                    "email": "user@vale.com",
                    "role": "admin",
                    "tokens_used": 0,
                    "tokens_limit": 1000,
                    "active": True,
                }
            )
        return _FakeQuery([])


def _client():
    main.supa = _FakeSupa()
    main.RATE_LIMIT_BUCKETS.clear()
    return TestClient(main.app)


def test_health_is_minimal():
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_invalid_token_error_not_verbose():
    class _FailAuth(_FakeAuth):
        def get_user(self, token):
            raise RuntimeError("internal fail details should not leak")

    class _FailSupa(_FakeSupa):
        def __init__(self):
            self.auth = _FailAuth()

    main.supa = _FailSupa()
    client = TestClient(main.app)
    resp = client.get("/api/auth/me", headers={"Authorization": "Bearer abc"})
    assert resp.status_code == 401
    assert "internal fail details" not in resp.text


def test_login_rate_limit():
    client = _client()
    main.RATE_LIMIT_WINDOWS["login"] = (60, 2)
    payload = {"email": "user@vale.com", "password": "Secret123!"}
    r1 = client.post("/api/auth/login", json=payload)
    r2 = client.post("/api/auth/login", json=payload)
    r3 = client.post("/api/auth/login", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


def test_kb_upload_policy_default_allows_authenticated_user(monkeypatch):
    monkeypatch.delenv("KB_UPLOAD_ALLOWED_ROLES", raising=False)
    assert main.can_upload_knowledge({"role": "user"}) is True


def test_kb_upload_policy_can_be_restricted(monkeypatch):
    monkeypatch.setenv("KB_UPLOAD_ALLOWED_ROLES", "admin,knowledge_editor")
    assert main.can_upload_knowledge({"role": "user"}) is False
    assert main.can_upload_knowledge({"role": "admin"}) is True


def test_register_existing_user_returns_friendly_conflict():
    class _DupAuth(_FakeAuth):
        def sign_up(self, payload):
            raise RuntimeError("User already registered")

    class _DupSupa(_FakeSupa):
        def __init__(self):
            self.auth = _DupAuth()

    main.supa = _DupSupa()
    client = TestClient(main.app)
    resp = client.post("/api/auth/register", json={"email": "user@vale.com", "password": "Secret123!"})
    assert resp.status_code == 409
    assert resp.json().get("detail") == "User already registered."
