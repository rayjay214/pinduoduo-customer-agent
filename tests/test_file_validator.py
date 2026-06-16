import builtins
import sys
import types

from utils.file_validator import ExcelValidator


def test_excel_validator_binary_read_error_hides_internal_exception(monkeypatch, tmp_path):
    path = tmp_path / "demo.xlsx"
    path.write_bytes(b"PK\x03\x04" + b"0" * 32)

    real_open = builtins.open

    def fake_open(file, mode="r", *args, **kwargs):
        if str(file) == str(path) and mode == "rb":
            raise OSError(f"Permission denied: {path}; token=secret-token")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    result = ExcelValidator().validate_basic(str(path))

    assert result.is_valid is False
    assert result.error_type == "READ_ERROR"
    assert "secret-token" not in result.error_message
    assert str(path) not in result.error_message


def test_excel_validator_readable_binary_error_hides_internal_exception(monkeypatch, tmp_path):
    path = tmp_path / "demo.xls"
    path.write_bytes(b"excel-data")

    real_open = builtins.open

    def fake_open(file, mode="r", *args, **kwargs):
        if str(file) == str(path) and mode == "rb":
            raise OSError(f"Permission denied: {path}; token=secret-token")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    result = ExcelValidator().validate_readable(str(path))

    assert result.is_valid is False
    assert result.error_type == "READ_ERROR"
    assert "secret-token" not in result.error_message
    assert str(path) not in result.error_message


def test_excel_validator_openpyxl_error_hides_internal_exception(monkeypatch, tmp_path):
    path = tmp_path / "demo.xls"
    path.write_bytes(b"excel-data")

    fake_openpyxl = types.SimpleNamespace(
        load_workbook=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError(f"cannot parse {path}; api_key=secret-token")
        )
    )
    monkeypatch.setitem(sys.modules, "openpyxl", fake_openpyxl)

    result = ExcelValidator().validate_readable(str(path))

    assert result.is_valid is False
    assert result.error_type == "READ_ERROR"
    assert "secret-token" not in result.error_message
    assert str(path) not in result.error_message
