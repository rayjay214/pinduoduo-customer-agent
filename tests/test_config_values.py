from utils.config_values import as_bool, as_float, as_int


def test_as_bool_parses_common_true_values():
    for value in (True, 1, "1", "true", "YES", "on", "开启", "启用"):
        assert as_bool(value, default=False) is True


def test_as_bool_parses_common_false_values():
    for value in (False, 0, "0", "false", "NO", "off", "关闭", "禁用"):
        assert as_bool(value, default=True) is False


def test_as_bool_falls_back_for_unknown_values():
    assert as_bool("maybe", default=True) is True
    assert as_bool(None, default=False) is False


def test_as_int_falls_back_for_invalid_values():
    assert as_int("12", default=7) == 12
    assert as_int("bad", default=7) == 7
    assert as_int(None, default=7) == 7


def test_as_float_falls_back_for_invalid_values():
    assert as_float("1.5", default=2.5) == 1.5
    assert as_float("bad", default=2.5) == 2.5
    assert as_float(None, default=2.5) == 2.5


def test_numeric_helpers_reject_non_finite_values():
    assert as_bool(float("nan"), default=False) is False
    assert as_bool(float("inf"), default=False) is False
    assert as_int(float("inf"), default=7) == 7
    assert as_float("nan", default=2.5) == 2.5
    assert as_float("inf", default=2.5) == 2.5


def test_numeric_helpers_reject_overflow_inputs():
    huge_int = 10 ** 10000

    assert as_bool(huge_int, default=False) is False
    assert as_float(huge_int, default=2.5) == 2.5
