import main


def test_validate_csv_signature_accepts_utf8():
    payload = b"col1,col2\n1,2\n"
    main.validate_file_signature("dados.csv", "text/csv", payload)


def test_validate_xls_signature_rejects_invalid_payload():
    try:
        main.validate_file_signature("legacy.xls", "application/vnd.ms-excel", b"not-an-xls")
        assert False, "Expected HTTPException for invalid XLS payload"
    except main.HTTPException as exc:
        assert exc.detail == "Invalid XLS file signature."


def test_extract_text_csv_returns_rows():
    payload = "a,b\n1,2\n".encode("utf-8")
    out = main.extract_text("teste.csv", payload)
    assert "a, b" in out
    assert "1, 2" in out


def test_extract_text_csv_autodetects_semicolon_delimiter():
    payload = "a;b\n1;2\n".encode("utf-8")
    out = main.extract_text("teste.csv", payload)
    assert "a, b" in out
    assert "1, 2" in out


def test_extract_text_xlsx_uses_openpyxl_when_available(monkeypatch):
    seen = {"data_only": None}

    class _FakeWS:
        title = "Dados"

        def iter_rows(self, values_only=True):
            yield ("Coluna A", "Coluna B")
            yield ("1", "2")

    class _FakeWB:
        worksheets = [_FakeWS()]

    def _fake_load_workbook(filename=None, read_only=True, data_only=True):
        seen["data_only"] = data_only
        return _FakeWB()

    monkeypatch.setattr(main, "HAS_OPENPYXL", True)
    monkeypatch.setattr(main, "load_workbook", _fake_load_workbook)
    out = main.extract_text("planilha.xlsx", b"fake-xlsx-content")
    assert seen["data_only"] is False
    assert "[Sheet: Dados]" in out
    assert "Coluna A | Coluna B" in out


def test_extract_text_xls_uses_xlrd_when_available(monkeypatch):
    class _FakeSheet:
        name = "Legacy"
        nrows = 2
        ncols = 2

        def cell_value(self, r, c):
            values = {
                (0, 0): "c1",
                (0, 1): "c2",
                (1, 0): "v1",
                (1, 1): "v2",
            }
            return values[(r, c)]

    class _FakeBook:
        def sheets(self):
            return [_FakeSheet()]

    class _FakeXLRD:
        @staticmethod
        def open_workbook(file_contents=None):
            return _FakeBook()

    monkeypatch.setattr(main, "HAS_XLRD", True)
    monkeypatch.setattr(main, "xlrd", _FakeXLRD())
    payload = main.MAGIC_SIGNATURES["xls_ole"] + b"rest"
    out = main.extract_text("arquivo.xls", payload)
    assert "[Sheet: Legacy]" in out
    assert "c1 | c2" in out
